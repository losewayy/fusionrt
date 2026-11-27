#include "fusionrt/models.h"
#include "cuda_executor.h"
#include "../core/moe_route_trace.h"
#include "../core/spec_cost.h"

#ifdef FUSIONRT_ENABLE_CUDA
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>

namespace fusionrt {

std::vector<float> FusionModel::prefill_cached_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt,
                                                    bool use_fused, int* prefilled_tokens, LayerTrace* trace,
                                                    OpTrace* op_trace, const std::atomic_bool* cancel) {
    sessions_->require_loaded();
    if (cancel && cancel->load(std::memory_order_relaxed)) throw CudaPrefillCancelled{};
    if (prompt.empty()) throw std::invalid_argument("Prompt is empty");
    if (prompt.size() > static_cast<size_t>(cfg_.dense.max_seq_len))
        throw prompt_context_error(prompt.size(), cfg_.dense.max_seq_len);
    validate_token_ids(prompt, 0, cfg_.vocab_size);
    return executor_->prefill_cached(ctx, prompt, use_fused, prefilled_tokens, trace, op_trace, cancel);
}

static void lookup_draft(const std::vector<int32_t>& history, int limit, std::vector<int32_t>& input) {
    const int count = static_cast<int>(history.size());
    for (int n = std::min(3, count); n >= 2; --n) {
        for (int end = count - 1; end >= n; --end) {
            if (std::equal(history.end() - n, history.end(), history.begin() + end - n)) {
                input.insert(input.end(), history.begin() + end, history.begin() + end + std::min(limit, count - end));
                return;
            }
        }
    }
}

static int32_t verify_sample(const Sampler::Distribution& target, const Sampler::Distribution& draft,
                             int32_t proposal, std::mt19937_64& rng) {
    const auto probability = [](const Sampler::Distribution& distribution, int32_t id) {
        const auto it = std::lower_bound(distribution.begin(), distribution.end(), id,
                                         [](const auto& entry, int32_t token) { return entry.second < token; });
        return it == distribution.end() || it->second != id ? 0.0 : it->first;
    };
    const auto candidate = std::find_if(target.begin(), target.end(),
                                        [proposal](const auto& entry) { return entry.second == proposal; });
    if (candidate != target.end() &&
        std::generate_canonical<double, 53>(rng) * probability(draft, proposal) < candidate->first)
        return proposal;

    // A point-mass lookup proposal uses the same residual rule as a draft model.
    Sampler::Distribution residual;
    residual.reserve(target.size());
    for (const auto& [weight, id] : target) {
        const double difference = weight - probability(draft, id);
        if (difference > 0) residual.emplace_back(difference, id);
    }
    return Sampler::draw(residual, rng);
}

std::vector<int32_t> FusionModel::generate_cuda(CudaContext& ctx, const std::vector<int32_t>& prompt_ids,
                                                int max_new_tokens, bool use_fused, bool use_graph,
                                                GenerationStats* stats, bool keep_state, int start_pos,
                                                int32_t stop_eos_id, const GenerationOptions& options) {
    sessions_->require_loaded();
    options.sampling.validate();
    const auto cancelled = [&] { return options.cancel && options.cancel->load(std::memory_order_relaxed); };
    if (max_new_tokens < 0 || start_pos < 0) throw std::invalid_argument("Invalid generation length or start position");
    auto* draft = options.draft_model;
    const auto* json = options.json_constraint;
    if (json && json->vocab_size() != static_cast<size_t>(cfg_.vocab_size))
        throw std::invalid_argument("JSON constraint vocabulary does not match model");
    const bool native_mtp = options.spec_mtp;
    const bool auto_mtp_depth = native_mtp && (options.auto_draft_tokens || options.draft_tokens == 0);
    if (native_mtp && (!std::isfinite(options.spec_p_min) || options.spec_p_min < 0.f || options.spec_p_min > 1.f))
        throw std::invalid_argument("MTP draft probability cutoff must be in [0, 1]");
    const int modes = (options.spec_ngram > 0) + (draft != nullptr) + native_mtp;
    if (options.spec_ngram < 0 || options.draft_tokens < 0 || (options.draft_tokens == 0 && !native_mtp) || modes > 1) {
        throw std::invalid_argument("Choose one speculative decoding mode. A zero draft limit selects native MTP auto depth");
    }
    if (draft && (draft == this || !draft->sessions_->initialized() || draft->cfg_.vocab_size != cfg_.vocab_size || keep_state)) {
        throw std::invalid_argument("Draft generation needs a separate CUDA model with matching vocabulary and a fresh prompt");
    }
    if (native_mtp && (!has_native_mtp() || keep_state))
        throw std::invalid_argument("Native MTP needs a loaded NextN block and a fresh prompt");
    if (!keep_state) start_pos = 0;
    std::vector<int32_t> tokens = prompt_ids;
    if (tokens.empty()) tokens.push_back(std::min(1, cfg_.vocab_size - 1));
    validate_token_ids(tokens, start_pos, cfg_.vocab_size);
    const auto embeds = options.inputs_embeds;
    CudaExecutor::validate_input_embeddings(embeds, prompt_ids.size(), static_cast<int>(weights_->embeddings().shape[1]));
    if (!embeds.empty() && (keep_state || modes > 0))
        throw std::invalid_argument("Image input requires a fresh prompt and standard decoding");
    const auto total_tokens = static_cast<size_t>(start_pos) + tokens.size();
    if (total_tokens > static_cast<size_t>(cfg_.dense.max_seq_len)) {
        throw prompt_context_error(total_tokens, cfg_.dense.max_seq_len);
    }
    const int prompt_len = static_cast<int>(tokens.size());
    max_new_tokens = std::min(max_new_tokens, cfg_.dense.max_seq_len - start_pos - prompt_len);
    validate_cuda_kv_capacity(start_pos + prompt_len + max_new_tokens);
    if (json && max_new_tokens == 0) throw std::invalid_argument("JSON generation needs room for output tokens");
    const int batch_limit = modes > 0 && max_new_tokens > 1
        ? std::min(prefill_batch_limit(ctx, true), native_mtp ? MtpDepthController::max_depth + 1 : cfg_.dense.max_seq_len) : 1;
    const bool speculative = batch_limit >= 2;
    if (speculative && options.op_trace) {
        throw std::invalid_argument("Operation profiling is unavailable during speculative decoding");
    }
    const bool reuse = options.reuse_prefix && !keep_state && embeds.empty();
    struct RestoreHiddenCapture {
        CudaExecutor& executor;
        ~RestoreHiddenCapture() { executor.set_mtp_hidden_capture(false); }
    } restore_capture{*executor_};
    executor_->set_mtp_hidden_capture(native_mtp && speculative);
    if (keep_state) sessions_->clear_cuda_history();
    else if (!reuse) reset_kv_and_state(ctx);
    sessions_->promote_cuda_expert_slabs(ctx, start_pos + prompt_len + max_new_tokens, weights_->borrowable_experts());
    auto* route_trace = ctx.moe_route_trace;
    if (route_trace) route_trace->begin(&weights_->moe_layers());
    const bool greedy = options.sampling.greedy() && !json;
    const bool stochastic = options.sampling.temperature > 0;
    if (use_graph) {
        const char* fallback = nullptr;
        if (options.op_trace) fallback = "operation tracing is enabled";
        else if (route_trace) fallback = "MoE route tracing is enabled";
        else if (json) fallback = "JSON grammar is enabled";
        else if (stochastic && !native_mtp) fallback = "temperature is greater than zero";
        else if (options.sampling.repeat_penalty != 1.0f && !native_mtp) fallback = "repeat_penalty is not 1";
        else if (!executor_->can_use_cuda_graph() &&
                 (!options.warn_on_graph_fallback ||
                  !executor_->has_graph_prefix(use_fused)))
            fallback = "no eligible MoE GPU prefix";
        if (fallback && options.warn_on_graph_fallback)
            std::clog << "--graph requested, using eager decoding because " << fallback << '\n';
        use_graph = fallback == nullptr;
    }
    std::mt19937_64 rng;
    if (!greedy) rng.seed(options.sampling.seed);
    std::mt19937_64 draft_rng(options.sampling.seed ^ UINT64_C(0x9e3779b97f4a7c15));
    auto grammar = json ? json->initial_state() : JsonGrammarState{};
    OpTrace prefill_profile;
    if (options.op_trace) prefill_profile.mode = "prefill";

    auto sample_cpu = [&](float* row, int remaining) {
        if (json) json->mask({row, static_cast<size_t>(cfg_.vocab_size)}, grammar, remaining);
        return Sampler::sample(row, cfg_.vocab_size, tokens, options.sampling, rng);
    };

    auto tp0 = std::chrono::high_resolution_clock::now();
    int prefilled = prompt_len;
    OpTrace* prefill_trace = options.op_trace ? &prefill_profile : nullptr;
    std::vector<float> logits;
    int draft_pos = 0;
    try {
        logits = reuse ? prefill_cached_cuda(ctx, tokens, use_fused, &prefilled, options.layer_trace, prefill_trace, options.cancel)
                       : executor_->prefill_cuda_impl(ctx, tokens.data(), prompt_len, start_pos, use_fused,
                                             options.layer_trace, prefill_trace, false, embeds, options.cancel);
        // A failing callback or CUDA operation must not leave a reusable recurrent state.
        sessions_->forget_tokens();
        if (speculative && draft && prompt_len < draft->cfg_.dense.max_seq_len) {
            draft->reset_kv_and_state(ctx);
            // Leave the first generated token pending, as on the target.
            draft->executor_->execute_prefill_chunks(ctx, tokens.data(), prompt_len, 0, use_fused, nullptr, nullptr,
                                                     false, false, {}, options.cancel);
            draft_pos = prompt_len;
        }
        if (cancelled()) throw CudaPrefillCancelled{};
    } catch (const CudaPrefillCancelled&) {
        sessions_->clear_cuda_history();
        if (stats) {
            *stats = {};
            stats->prompt_tokens = prompt_len;
            stats->cancelled = true;
        }
        return tokens;
    }
    int32_t next_tok = -1;
    if (max_new_tokens > 0) {
        next_tok = greedy ? executor_->sample_gpu(ctx) : sample_cpu(logits.data(), max_new_tokens);
    }
    auto tp1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    int pos = start_pos + prompt_len;
    int mtp_pos = native_mtp ? sessions_->mtp_kv_valid_len() : 0;
    int gen_count = 0;
    int spec_proposed = 0, spec_accepted = 0;
    GenerationStats mtp_stats;
    auto request = executor_->begin_decode(use_fused, options.op_trace, max_new_tokens > 1);
    if (options.op_trace && max_new_tokens > 1) options.op_trace->mode = "decode";
    bool used_cuda_graph = false;
    auto decode = [&](bool first_step) {
        used_cuda_graph = executor_->decode(ctx, *request, pos++, use_graph, greedy, first_step);
    };
    auto emit = [&](int32_t token) {
        if (cancelled()) return false;
        tokens.push_back(token);
        ++gen_count;
        if (json && !json->accept(grammar, token)) throw std::logic_error("Sampled a token outside the JSON grammar");
        if (stop_eos_id >= 0 && token == stop_eos_id) return false;
        if (options.on_token && !options.on_token(token)) return false;
        return gen_count < max_new_tokens && !(json && (grammar.sealed() || json->is_end(token)));
    };
    auto td0 = std::chrono::high_resolution_clock::now();
    if (!speculative && greedy && use_graph && executor_->can_use_cuda_graph() && !options.on_token && !options.cancel && stop_eos_id < 0 && max_new_tokens > 1 &&
        static_cast<int64_t>(pos) + max_new_tokens <= executor_->token_history_capacity()) {
        // Graph replay feeds the next greedy step and records IDs on the device.
        executor_->clear_invalid_token(ctx);
        int first_pos = pos;
        for (int step = 1; step < max_new_tokens; ++step) {
            decode(step == 1);
            if (!used_cuda_graph) executor_->store_token_history(ctx, pos);
        }
        tokens.push_back(next_tok);
        tokens.resize(static_cast<size_t>(prompt_len) + max_new_tokens);
        executor_->download_token_history(ctx, 1 + first_pos,
            {tokens.data() + prompt_len + 1, static_cast<size_t>(max_new_tokens - 1)}, true);
        if (std::find(tokens.begin() + prompt_len, tokens.end(), -1) != tokens.end())
            throw std::runtime_error("Sampling found no selectable logits");
        gen_count = max_new_tokens;
    } else if (!speculative) {
        for (int step = 0; step < max_new_tokens && !cancelled(); ++step) {
            if (!emit(next_tok)) break;
            if (!greedy) {
                executor_->set_token(ctx, &next_tok);
            }
            decode(step == 0);
            if (greedy) {
                next_tok = executor_->download_argmax(ctx);
            } else {
                HostTensor host_logits = ctx.download(executor_->logits());
                next_tok = sample_cpu(host_logits.data<float>(), max_new_tokens - gen_count);
            }
        }
    } else {
        auto& spec_arena = ctx.scratch_arena();
        struct RestoreArena {
            CudaArena& arena;
            size_t mark;
            ~RestoreArena() { arena.reset(mark); }
        } restore{spec_arena, spec_arena.mark()};
        auto verify = executor_->begin_verify(ctx, native_mtp, use_graph, batch_limit);
        const bool profile_verify = native_mtp && std::getenv("FUSIONRT_MTP_PROFILE");
        const bool time_verify = profile_verify || auto_mtp_depth;
        MtpDepthController depth_controller;
        std::array<int, 9> verify_calls{};
        std::array<double, 9> verify_ms{};
        auto* draft_sessions = draft_pos > 0 ? draft->sessions_.get() : nullptr;
        const size_t draft_ssm = draft_sessions ? draft_sessions->recurrent_bytes() : 0;
        const size_t target_ssm = sessions_->recurrent_bytes();
        CudaArena snapshots;
        void* target_snapshot = nullptr;
        void* draft_snapshot = nullptr;
        if (target_ssm || draft_ssm) {
            snapshots.init(target_ssm + draft_ssm);
            if (target_ssm) target_snapshot = snapshots.alloc(target_ssm);
            if (draft_ssm) draft_snapshot = snapshots.alloc(draft_ssm);
        }
        auto draft_request = (draft ? draft->executor_ : executor_)->begin_decode(use_fused);
        const bool draft_greedy = options.sampling.greedy();
        const bool draft_graph = use_graph && draft && draft->executor_->can_use_cuda_graph();
        auto draft_step = [&](const int32_t* token, bool need_argmax, bool first_step = true) {
            return draft->executor_->decode(ctx, *draft_request, draft_pos++, draft_graph, need_argmax, first_step, token);
        };
        std::vector<int32_t> input, row_ids, draft_history;
        std::vector<float> rows;
        std::vector<Sampler::Distribution> draft_probs;
        auto mtp = native_mtp ? executor_->begin_mtp(ctx, options.spec_p_min > 0.f) :
            CudaExecutor::Request<CudaExecutor::MtpDraft>{nullptr, nullptr};
        const auto fill_verified_mtp = [&](int end) {
            while (mtp_pos < end && !cancelled()) {
                executor_->mtp_step(ctx, *mtp, mtp_pos, tokens[mtp_pos], mtp_pos - 1,
                                    false, false, use_fused, *verify);
                ++mtp_pos;
            }
        };
        const int draft_cap = options.draft_tokens > 0 ? options.draft_tokens : MtpDepthController::max_depth;
        int ngram_depth = 1, ngram_misses = 0, ngram_pause = 0;
        bool pending_on_device = greedy, reset_decode_position = true;
        bool running = max_new_tokens > 0 && emit(next_tok);
        while (running && !cancelled()) {
            const int remaining = max_new_tokens - gen_count;
            const int start = pos;
            const bool ngram_probe = options.spec_ngram > 0 && ngram_pause == 0;
            if (ngram_pause > 0) --ngram_pause;
            int limit = std::min({remaining - 1, batch_limit - 1, (draft || native_mtp) ? draft_cap :
                                  ngram_probe ? ngram_depth : 0});
            if (auto_mtp_depth) limit = depth_controller.choose_depth(limit);
            input.assign(1, tokens.back());
            draft_probs.clear();
            if (draft_sessions && limit > 0 && start < draft->cfg_.dense.max_seq_len) {
                limit = std::min(limit, draft->cfg_.dense.max_seq_len - start);
                while (draft_pos < start && !cancelled()) draft_step(&tokens[draft_pos], false);
                if (cancelled()) break;
                draft_sessions->save_recurrent(ctx, draft_snapshot);
                if (draft_greedy) {
                    for (int i = 0; i < limit; ++i) {
                        if (cancelled()) break;
                        if (!draft_step(i == 0 ? input.data() : nullptr, true, i == 0))
                            draft->executor_->store_token_history(ctx, 1 + start + i);
                    }
                    if (cancelled()) break;
                    input.resize(1 + limit);
                    draft->executor_->download_token_history(ctx, 1 + start,
                        {input.data() + 1, static_cast<size_t>(limit)});
                } else {
                    draft_history = tokens;
                    for (int i = 0; i < limit && !cancelled(); ++i) {
                        draft_step(&input.back(), false);
                        auto host = ctx.download(draft->executor_->logits());
                        auto q = Sampler::distribution(host.data<float>(), cfg_.vocab_size, draft_history, options.sampling);
                        const auto proposal = stochastic ? Sampler::draw(q, draft_rng) : q.front().second;
                        if (stochastic) {
                            std::sort(q.begin(), q.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
                            draft_probs.push_back(std::move(q));
                        }
                        draft_history.push_back(proposal);
                        input.push_back(proposal);
                    }
                }
            } else if (native_mtp && limit > 0) {
                executor_->ensure_kv(ctx, mtp_pos, start + limit);
                const bool initial_mtp_fill = auto_mtp_depth && mtp_pos < prompt_len;
                fill_verified_mtp(start);
                if (initial_mtp_fill) ctx.synchronize();
                for (int i = 0; i < limit && !cancelled(); ++i) {
                    const auto draft_start = auto_mtp_depth ? std::chrono::steady_clock::now() :
                                                             std::chrono::steady_clock::time_point{};
                    if (executor_->mtp_step(ctx, *mtp, start + i, input.back(), start - 1,
                                            i != 0, true, use_fused, *verify))
                        ++mtp_stats.mtp_draft_graph_steps;
                    if (i == 0) {
                        // The pending token was seeded with the verified trunk row.
                        mtp_pos = start + 1;
                        ++mtp_stats.mtp_reused_kv_rows;
                    } else {
                        // Later rows used draft hidden states. Refresh them from verified seeds before reuse.
                        ++mtp_stats.mtp_rewound_kv_rows;
                    }
                    const auto [proposal, confidence] = executor_->read_mtp(ctx, *mtp);
                    if (auto_mtp_depth) depth_controller.observe_draft(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - draft_start).count());
                    if (confidence < options.spec_p_min) {
                        ++mtp_stats.mtp_cutoffs;
                        break;
                    }
                    input.push_back(proposal);
                    ++mtp_stats.mtp_proposed_by_depth[i];
                }
            } else if (ngram_probe && limit > 0) {
                lookup_draft(tokens, limit, input);
            }
            if (cancelled()) break;
            const int proposed = static_cast<int>(input.size()) - 1;
            if (proposed == 0 && !native_mtp) {
                if (ngram_probe && limit > 0 && ++ngram_misses >= 3) {
                    ngram_pause = 8;
                    ngram_misses = 0;
                }
                if (!pending_on_device) {
                    executor_->set_token(ctx, input.data());
                }
                decode(reset_decode_position);
                pending_on_device = greedy;
                reset_decode_position = false;
                if (greedy) next_tok = executor_->download_argmax(ctx);
                else {
                    auto host = ctx.download(executor_->logits());
                    next_tok = sample_cpu(host.data<float>(), remaining);
                }
                running = emit(next_tok);
                continue;
            }

            spec_proposed += proposed;
            if (proposed > 0) sessions_->save_recurrent(ctx, target_snapshot);
            const auto verify_start = time_verify ? std::chrono::steady_clock::now() :
                                                   std::chrono::steady_clock::time_point{};
            if (executor_->verify_cuda(ctx, input, start, use_fused, rows, row_ids, greedy,
                            reuse && !native_mtp && target_snapshot,
                            native_mtp ? verify.get() : nullptr)) {
                used_cuda_graph = true;
                ++mtp_stats.mtp_verify_graph_steps;
            }
            if (time_verify) {
                const double ms = std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - verify_start).count();
                if (auto_mtp_depth) depth_controller.observe_verify(proposed + 1, ms);
                if (profile_verify) {
                    ++verify_calls[proposed + 1];
                    verify_ms[proposed + 1] += ms;
                }
            }
            pending_on_device = false;
            reset_decode_position = true;
            int accepted_count = 0;
            for (int i = 0; i <= proposed; ++i) {
                const bool bonus = i == proposed;
                if (greedy) next_tok = row_ids[i];
                else {
                    auto* row = rows.data() + static_cast<size_t>(i) * cfg_.vocab_size;
                    if (!stochastic || bonus) next_tok = sample_cpu(row, max_new_tokens - gen_count);
                    else {
                        if (json) json->mask({row, static_cast<size_t>(cfg_.vocab_size)}, grammar, max_new_tokens - gen_count);
                        const auto p = Sampler::distribution(row, cfg_.vocab_size, tokens, options.sampling);
                        const auto proposal = input[i + 1];
                        next_tok = draft ? verify_sample(p, draft_probs[i], proposal, rng) :
                                           verify_sample(p, {{1.0, proposal}}, proposal, rng);
                    }
                }
                const bool accepted = !bonus && next_tok == input[i + 1];
                spec_accepted += accepted;
                accepted_count += accepted;
                if (native_mtp && !bonus) {
                    ++mtp_stats.mtp_checked_by_depth[i];
                    mtp_stats.mtp_accepted_by_depth[i] += accepted;
                    if (auto_mtp_depth) depth_controller.observe_acceptance(i + 1, accepted);
                }
                running = emit(next_tok);
                if (!accepted || !running) break;
            }
            if (ngram_probe) {
                if (accepted_count == proposed) {
                    ngram_misses = 0;
                    if (proposed == ngram_depth && ngram_depth < options.spec_ngram) ++ngram_depth;
                } else {
                    ngram_depth = std::max(1, accepted_count);
                    ngram_pause = accepted_count == 0 ? 8 : 0;
                    ngram_misses = 0;
                }
            }
            // The last emitted token is pending. Only earlier tokens belong to the live KV/SSM prefix.
            const int kept = start_pos + static_cast<int>(tokens.size()) - 1;
            pos = start + proposed + 1;
            if (kept < pos) {
                const auto rewind_start = auto_mtp_depth ? std::chrono::steady_clock::now() :
                                                          std::chrono::steady_clock::time_point{};
                sessions_->truncate(kept);
                if (target_snapshot) {
                    sessions_->restore_recurrent(ctx, target_snapshot);
                    if (!executor_->restore_verified_recurrence(ctx, *verify, kept - start, use_fused)) {
                        executor_->execute_prefill_chunks(ctx, input.data(), kept - start, start,
                                                          use_fused, nullptr, nullptr, reuse);
                    }
                }
                if (auto_mtp_depth) {
                    if (target_snapshot) ctx.synchronize();
                    depth_controller.observe_rewind(pos - kept, std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - rewind_start).count());
                }
                mtp_pos = std::min(mtp_pos, kept);
                pos = kept;
            }
            if (draft_sessions && draft_pos > kept) {
                draft_sessions->truncate(kept);
                if (draft_snapshot) {
                    draft_sessions->restore_recurrent(ctx, draft_snapshot);
                    draft_pos = start;
                    while (draft_pos < kept) draft_step(&tokens[draft_pos], false);
                } else draft_pos = kept;
            }
        }
        // Materialize NextN rows seeded by verified trunk states before sharing this prefix.
        if (native_mtp && reuse && mtp_pos < pos && !cancelled()) {
            executor_->ensure_kv(ctx, mtp_pos, pos);
            fill_verified_mtp(pos);
        }
        ctx.synchronize();
        if (profile_verify) for (size_t m = 1; m < verify_calls.size(); ++m) {
            if (verify_calls[m]) std::clog << "MTP verify M=" << m << " calls=" << verify_calls[m]
                << " mean_ms=" << verify_ms[m] / verify_calls[m] << '\n';
        }
    }
    if (cancelled()) sessions_->clear_cuda_history();
    else {
        if (native_mtp) sessions_->set_mtp_kv_valid_len(std::min(mtp_pos, pos));
        // The most recently sampled token has not been forwarded into KV/SSM state.
        if (reuse) {
            if (pos > prompt_len) {
                // After speculative rollback, the last verify logit can belong to a discarded row.
                executor_->save_prefix_checkpoint(ctx, pos, native_mtp && speculative ? nullptr : &executor_->logits());
            }
            sessions_->retain_tokens({tokens.data(), static_cast<size_t>(pos)}, use_fused);
        }
    }
    if (options.op_trace && options.op_trace->ops.empty()) *options.op_trace = std::move(prefill_profile);
    auto td1 = std::chrono::high_resolution_clock::now();
    double decode_ms = std::chrono::duration<double, std::milli>(td1 - td0).count();
    if (route_trace) route_trace->finish(start_pos + prompt_len, pos);

    if (stats) {
        stats->prompt_tokens = static_cast<int>(prompt_ids.size());
        stats->generated_tokens = gen_count;
        stats->prefilled_tokens = prefilled;
        stats->spec_proposed = spec_proposed;
        stats->spec_accepted = spec_accepted;
        stats->mtp_proposed_by_depth = mtp_stats.mtp_proposed_by_depth;
        stats->mtp_accepted_by_depth = mtp_stats.mtp_accepted_by_depth;
        stats->mtp_checked_by_depth = mtp_stats.mtp_checked_by_depth;
        stats->mtp_cutoffs = mtp_stats.mtp_cutoffs;
        stats->mtp_draft_graph_steps = mtp_stats.mtp_draft_graph_steps;
        stats->mtp_verify_graph_steps = mtp_stats.mtp_verify_graph_steps;
        stats->mtp_reused_kv_rows = mtp_stats.mtp_reused_kv_rows;
        stats->mtp_rewound_kv_rows = mtp_stats.mtp_rewound_kv_rows;
        stats->used_cuda_graph = used_cuda_graph;
        stats->cancelled = cancelled();
        stats->json_complete = json && grammar.complete();
        stats->speculative_active = speculative;
        stats->prefill_ms = prefill_ms;
        stats->decode_ms = decode_ms;
        stats->prefill_tok_per_s = (prefill_ms > 0.0) ? (static_cast<double>(prefilled) * 1000.0 / prefill_ms) : 0.0;
        stats->decode_tok_per_s = (decode_ms > 0.0 && gen_count > 1) ? (static_cast<double>(gen_count - 1) * 1000.0 / decode_ms) : 0.0;
    }
    return tokens;
}

std::vector<GenerationStats> FusionModel::bench_model_cuda(CudaContext& ctx, int pp, int tg, int reps, bool use_graph,
                                                           OpTrace* trace, int decode_prefix) {
    if (decode_prefix < 0 || tg < 0 || decode_prefix > cfg_.dense.max_seq_len - tg) {
        throw std::invalid_argument("Decode benchmark exceeds model context");
    }
    using Clock = std::chrono::steady_clock;
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    std::mt19937 rng(42);
    std::uniform_int_distribution<int32_t> token_id(0, cfg_.vocab_size - 1);
    std::vector<int32_t> prompt(pp), steps(tg), prefix(decode_prefix);
    std::generate(prompt.begin(), prompt.end(), [&] { return token_id(rng); });
    std::generate(steps.begin(), steps.end(), [&] { return token_id(rng); });
    std::generate(prefix.begin(), prefix.end(), [&] { return token_id(rng); });
    std::vector<GenerationStats> results(reps);

    auto bench = executor_->begin_bench(ctx, pp);
    auto* host_args = executor_->bench_args(*bench);
    use_graph = use_graph && executor_->can_use_cuda_graph();
    const auto capture = [&](int position, bool prefill) {
        if (!use_graph) return;
        try {
            executor_->capture_bench(ctx, *bench, position, prefill);
        } catch (const std::runtime_error& error) {
            executor_->clear_bench_graph(*bench);
            use_graph = false;
            std::clog << "CUDA Graph benchmark capture failed: " << error.what() << '\n';
        }
    };
    reset_kv_and_state(ctx);

    if (pp > 0) {
        executor_->ensure_kv(ctx, 0, pp);
        std::copy(prompt.begin(), prompt.end(), host_args);
        capture(pp, true);
        for (int rep = -1; rep < reps; ++rep) {
            sessions_->zero_cuda_gdn(ctx);
            ctx.synchronize();
            auto start = Clock::now();
            if (use_graph) executor_->launch_bench(ctx, *bench);
            else executor_->bench_prefill(ctx, *bench, pp);
            ctx.synchronize();
            if (rep >= 0) {
                auto& s = results[rep];
                s.prompt_tokens = pp;
                s.prefill_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                s.prefill_tok_per_s = 1000.0 * pp / s.prefill_ms;
            }
        }
        arena.reset(mark);
        executor_->clear_bench_graph(*bench);
    }

    if (tg > 0) {
        executor_->ensure_kv(ctx, 0, decode_prefix + tg);
        executor_->prepare_bench_decode(ctx, *bench);
        capture(decode_prefix, false);
        CudaSessionManager::HostRecurrentState recurrent_prefix;
        if (!prefix.empty()) {
            sessions_->zero_cuda_gdn(ctx);
            executor_->execute_prefill_chunks(ctx, prefix.data(), decode_prefix, 0, true, nullptr, nullptr, false);
            sessions_->download_recurrent(ctx, recurrent_prefix);
            ctx.synchronize();
        }
        for (int rep = -1; rep < reps; ++rep) {
            if (prefix.empty()) {
                sessions_->zero_cuda_gdn(ctx);
                ctx.synchronize();
            }
            // Decode only overwrites KV after the prefix. GDN state must be restored in place.
            sessions_->upload_recurrent(ctx, recurrent_prefix);
            auto start = Clock::now();
            for (int pos = 0; pos < tg; ++pos) {
                host_args[0] = steps[pos];
                host_args[1] = decode_prefix + pos;
                host_args[2] = decode_prefix + pos + 1;
                if (use_graph) executor_->launch_bench(ctx, *bench);
                else executor_->bench_decode(ctx, *bench, decode_prefix + pos, false);
                ctx.synchronize();
            }
            if (rep >= 0) {
                auto& s = results[rep];
                s.generated_tokens = tg;
                s.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
                s.decode_tok_per_s = 1000.0 * tg / s.decode_ms;
            }
        }
        executor_->clear_bench_graph(*bench);
    }
    arena.reset(mark);
    if (trace) {
        reset_kv_and_state(ctx);
        if (tg > 0) {
            trace->mode = "decode";
            executor_->ensure_kv(ctx, 0, decode_prefix + tg);
            if (!prefix.empty()) executor_->execute_prefill_chunks(ctx, prefix.data(), decode_prefix, 0, true, nullptr, nullptr, false);
            auto trace_request = executor_->begin_decode(true, trace, true);
            for (int pos = 0; pos < tg; ++pos)
                executor_->decode(ctx, *trace_request, decode_prefix + pos, false, false, true, &steps[pos], false);
        } else {
            trace->mode = "prefill";
            (void)executor_->prefill_cuda_impl(ctx, prompt.data(), pp, 0, true, nullptr, trace);
        }
    }
    for (auto& result : results) result.used_cuda_graph = use_graph;
    return results;
}


}  // namespace fusionrt
#endif
