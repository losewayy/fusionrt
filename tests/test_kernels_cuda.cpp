#include "test_support.h"

namespace fusionrt::test {

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_fused_rmsnorm_linear(CudaContext& ctx) {
    std::cout << "[2/7] Testing CUDA Fused Pre-Norm + Linear (F32, Q8_0, Q4_0)...\n";
    std::mt19937_64 rng(202);
    int M = 3, K = 512, N = 256;
    HostTensor hx = rand_f32({M, K}, rng, 0.5f);
    HostTensor hnw = rand_f32({K}, rng, 0.5f);
    HostTensor hw = rand_f32({N, K}, rng, 0.2f);

    // FP64 reference
    std::vector<double> ref(static_cast<size_t>(M * N), 0.0);
    for (int m = 0; m < M; ++m) {
        double ss = 0.0;
        for (int k = 0; k < K; ++k) {
            double v = hx.data<float>()[m * K + k];
            ss += v * v;
        }
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (hx.data<float>()[m * K + k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            }
            ref[static_cast<size_t>(m * N + n)] = acc;
        }
    }

    Tensor dx = ctx.upload(hx, false);
    Tensor dnw = ctx.upload(hnw, false);
    Tensor dw = ctx.upload(hw, false);
    Tensor dy = ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
    cuda_ops::fused_rmsnorm_linear_f32(dx.ptr<float>(), dnw.ptr<float>(), dw.ptr<float>(), dy.ptr<float>(), M, N, K, 1e-6f, ctx.stream());
    HostTensor hy = ctx.download(dy);
    expect_close(std::vector<float>(hy.data<float>(), hy.data<float>() + M * N), ref, 1e-4, 1e-4, "CUDA fused_rmsnorm_linear_f32");

    // Q8_0 and Q4_0 fused vs unfused
    HostTensor hq8 = HostTensor::create({N, K}, DType::Q8_0);
    HostTensor hq4 = HostTensor::create({N, K}, DType::Q4_0);
    for (int n = 0; n < N; ++n) {
        quantize_row_q8_0(hw.data<float>() + n * K, reinterpret_cast<BlockQ8_0*>(hq8.storage.data()) + n * (K / 32), K);
        quantize_row_q4_0(hw.data<float>() + n * K, reinterpret_cast<BlockQ4_0*>(hq4.storage.data()) + n * (K / 32), K);
    }
    Tensor dq8 = ctx.upload(hq8, false);
    Tensor dq4 = ctx.upload(hq4, false);
    Tensor dy_q8 = ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
    Tensor dy_q4 = ctx.scratch_arena().alloc_tensor({M, N}, DType::F32);
    cuda_ops::fused_rmsnorm_linear_q8_0(dx.ptr<float>(), dnw.ptr<float>(), dq8.ptr(), dy_q8.ptr<float>(), M, N, K, 1e-6f, ctx.stream());
    cuda_ops::fused_rmsnorm_linear_q4_0(dx.ptr<float>(), dnw.ptr<float>(), dq4.ptr(), dy_q4.ptr<float>(), M, N, K, 1e-6f, ctx.stream());
    HostTensor hy8 = ctx.download(dy_q8);
    HostTensor hy4 = ctx.download(dy_q4);
    expect_close(std::vector<float>(hy8.data<float>(), hy8.data<float>() + M * N), ref, 0.08, 0.02, "CUDA fused_rmsnorm_linear_q8_0");
    expect_close(std::vector<float>(hy4.data<float>(), hy4.data<float>() + M * N), ref, 0.65, 0.12, "CUDA fused_rmsnorm_linear_q4_0");

    // Verify M >= 16 WMMA Tensor Core prefill path (Q8_0, Q4_0, Q4_1, F16) against FP64 reference
    {
        int MP = 32;
        HostTensor hxp = rand_f32({MP, K}, rng, 0.4f);
        std::vector<double> ref_p(static_cast<size_t>(MP * N), 0.0);
        for (int m = 0; m < MP; ++m) {
            double ss = 0.0;
            for (int k = 0; k < K; ++k) {
                double v = hxp.data<float>()[m * K + k];
                ss += v * v;
            }
            double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += (hxp.data<float>()[m * K + k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
                }
                ref_p[static_cast<size_t>(m * N + n)] = acc;
            }
        }
        Tensor dxp = ctx.upload(hxp, false);
        Tensor dyp_q8 = ctx.scratch_arena().alloc_tensor({MP, N}, DType::F32);
        Tensor dyp_q4 = ctx.scratch_arena().alloc_tensor({MP, N}, DType::F32);
        cuda_ops::fused_rmsnorm_linear_q8_0(dxp.ptr<float>(), dnw.ptr<float>(), dq8.ptr(), dyp_q8.ptr<float>(), MP, N, K, 1e-6f, ctx.stream());
        cuda_ops::fused_rmsnorm_linear_q4_0(dxp.ptr<float>(), dnw.ptr<float>(), dq4.ptr(), dyp_q4.ptr<float>(), MP, N, K, 1e-6f, ctx.stream());
        HostTensor hyp8 = ctx.download(dyp_q8);
        HostTensor hyp4 = ctx.download(dyp_q4);
        expect_close(std::vector<float>(hyp8.data<float>(), hyp8.data<float>() + MP * N), ref_p, 0.15, 0.02, "CUDA WMMA prefill M=32 Q8_0");
        expect_close(std::vector<float>(hyp4.data<float>(), hyp4.data<float>() + MP * N), ref_p, 0.85, 0.14, "CUDA WMMA prefill M=32 Q4_0");
    }
    ctx.scratch_arena().reset();
}

void test_cuda_step1_fast_paths(CudaContext& ctx) {
    std::cout << "[2b/7] Testing CUDA Q4_A/Q8_A dp4a, cuBLAS FP16 prefill, and 3-stage GDN prefill...\n";
    std::mt19937_64 rng(2026);
    int N = 256, K = 512;
    HostTensor hx1 = rand_f32({1, K}, rng, 0.5f);
    HostTensor hnw = rand_f32({K}, rng, 0.5f);
    HostTensor hw = rand_f32({N, K}, rng, 0.25f);

    // FP64 reference M=1 normalized
    std::vector<double> ref1(static_cast<size_t>(N), 0.0);
    {
        double ss = 0.0;
        for (int k = 0; k < K; ++k) ss += hx1.data<float>()[k] * hx1.data<float>()[k];
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (hx1.data<float>()[k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            }
            ref1[static_cast<size_t>(n)] = acc;
        }
    }

    HostTensor hq8a = HostTensor::create({N, K}, DType::Q8_A);
    HostTensor hq4a = HostTensor::create({N, K}, DType::Q4_A);
    {
        std::vector<BlockQ8_0> tmp8(static_cast<size_t>(K / 32));
        std::vector<BlockQ4_0> tmp4(static_cast<size_t>(K / 32));
        for (int n = 0; n < N; ++n) {
            quantize_row_q8_0(hw.data<float>() + n * K, tmp8.data(), K);
            repack_q8_0_to_q8_a(tmp8.data(), reinterpret_cast<BlockQ8_A*>(hq8a.storage.data()) + n * (K / 256), K);
            quantize_row_q4_0(hw.data<float>() + n * K, tmp4.data(), K);
            repack_q4_0_to_q4_a(tmp4.data(), reinterpret_cast<BlockQ4_A*>(hq4a.storage.data()) + n * (K / 256), K);
        }
    }
    Tensor dx1 = ctx.upload(hx1, false);
    Tensor dnw = ctx.upload(hnw, false);
    Tensor dq8a = ctx.upload(hq8a, false);
    Tensor dq4a = ctx.upload(hq4a, false);
    Tensor dy8a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
    Tensor dy4a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
    cuda_ops::fused_rmsnorm_linear_q8_a(dx1.ptr<float>(), dnw.ptr<float>(), dq8a.ptr(), dy8a.ptr<float>(), 1, N, K, 1e-6f, ctx.stream());
    cuda_ops::fused_rmsnorm_linear_q4_a(dx1.ptr<float>(), dnw.ptr<float>(), dq4a.ptr(), dy4a.ptr<float>(), 1, N, K, 1e-6f, ctx.stream());
    HostTensor hy8a = ctx.download(dy8a);
    HostTensor hy4a = ctx.download(dy4a);
    expect_close(std::vector<float>(hy8a.data<float>(), hy8a.data<float>() + N), ref1, 0.08, 0.02, "CUDA Q8_A dp4a M=1");
    expect_close(std::vector<float>(hy4a.data<float>(), hy4a.data<float>() + N), ref1, 0.65, 0.12, "CUDA Q4_A dp4a M=1");

    // cuBLAS prefill M=32 parity on Q4_A and Q8_A
    {
        int MP = 32;
        HostTensor hxp = rand_f32({MP, K}, rng, 0.4f);
        std::vector<double> refp(static_cast<size_t>(MP * N), 0.0);
        for (int m = 0; m < MP; ++m) {
            double ss = 0.0;
            for (int k = 0; k < K; ++k) ss += hxp.data<float>()[m * K + k] * hxp.data<float>()[m * K + k];
            double inv = 1.0 / std::sqrt(ss / K + 1e-6);
            for (int n = 0; n < N; ++n) {
                double acc = 0.0;
                for (int k = 0; k < K; ++k) {
                    acc += (hxp.data<float>()[m * K + k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
                }
                refp[static_cast<size_t>(m * N + n)] = acc;
            }
        }
        Tensor dxp = ctx.upload(hxp, false);
        Tensor dx16 = ctx.scratch_arena().alloc_tensor({MP, K}, DType::F16);
        Tensor dw16 = ctx.scratch_arena().alloc_tensor({N, K}, DType::F16);
        Tensor dycb = ctx.scratch_arena().alloc_tensor({MP, N}, DType::F32);
        cuda_ops::rmsnorm_or_copy_to_f16(dxp.ptr<float>(), dnw.ptr<float>(), dx16.ptr<uint16_t>(), MP, K, 1e-6f, ctx.stream());
        cuda_ops::dequantize_to_f16(dq8a.ptr(), DType::Q8_A, dw16.ptr<uint16_t>(), N, K, ctx.stream());
        cuda_ops::cublas_linear_f16_f32(ctx.cublas(), dx16.ptr<uint16_t>(), dw16.ptr<uint16_t>(), dycb.ptr<float>(), MP, N, K);
        HostTensor hycb = ctx.download(dycb);
        expect_close(std::vector<float>(hycb.data<float>(), hycb.data<float>() + MP * N), refp, 0.15, 0.02, "CUDA cuBLAS prefill M=32 Q8_A");
    }

    // GDN prefill r1 (M=12, nk=nv=4, dk=dv=64) 3-stage vs M x single-step
    {
        int M = 12, nk = 4, nv = 4, dk = 64, dv = 64;
        int conv_dim = 2 * nk * dk + nv * dv;
        int v_dim = nv * dv;
        HostTensor hqkv = rand_f32({M, conv_dim}, rng, 0.25f);
        HostTensor hz = rand_f32({M, v_dim}, rng, 0.25f);
        HostTensor ha = rand_f32({M, nv}, rng, 0.15f);
        HostTensor hb = rand_f32({M, nv}, rng, 0.15f);
        HostTensor hcw = rand_f32({conv_dim, 4}, rng, 0.2f);
        HostTensor hcb = rand_f32({conv_dim}, rng, 0.05f);
        HostTensor hal = HostTensor::create({nv}, DType::F32);
        for (int i = 0; i < nv; ++i) hal.data<float>()[i] = -1.2f;
        HostTensor hdt = rand_f32({nv}, rng, 0.05f);
        HostTensor hnw_g = rand_f32({dv}, rng, 0.5f);

        Tensor d_qkv = ctx.upload(hqkv, false), d_z = ctx.upload(hz, false);
        Tensor d_a = ctx.upload(ha, false), d_b = ctx.upload(hb, false);
        Tensor d_cw = ctx.upload(hcw, false), d_cb = ctx.upload(hcb, false);
        Tensor d_al = ctx.upload(hal, false), d_dt = ctx.upload(hdt, false), d_nwg = ctx.upload(hnw_g, false);

        Tensor d_cs1 = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
        Tensor d_ss1 = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
        Tensor d_cs2 = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
        Tensor d_ss2 = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
        FUSIONRT_CUDA_CHECK(cudaMemset(d_cs1.ptr(), 0, d_cs1.nbytes()));
        FUSIONRT_CUDA_CHECK(cudaMemset(d_ss1.ptr(), 0, d_ss1.nbytes()));
        FUSIONRT_CUDA_CHECK(cudaMemset(d_cs2.ptr(), 0, d_cs2.nbytes()));
        FUSIONRT_CUDA_CHECK(cudaMemset(d_ss2.ptr(), 0, d_ss2.nbytes()));

        Tensor out_step = ctx.scratch_arena().alloc_tensor({M, v_dim}, DType::F32);
        Tensor out_pf = ctx.scratch_arena().alloc_tensor({M, v_dim}, DType::F32);
        Tensor gdn_ws = ctx.scratch_arena().alloc_tensor({static_cast<int64_t>(
            cuda_ops::gated_deltanet_prefill_scratch_floats(M, nk, nv, dk, dv))}, DType::F32);
        float qs = 1.0f / std::sqrt(static_cast<float>(dk));
        for (int m = 0; m < M; ++m) {
            cuda_ops::fused_gated_deltanet_step_f32(
                d_qkv.ptr<float>() + m * conv_dim, d_z.ptr<float>() + m * v_dim,
                d_a.ptr<float>() + m * nv, d_b.ptr<float>() + m * nv,
                d_cw.ptr<float>(), d_cb.ptr<float>(), d_al.ptr<float>(), d_dt.ptr<float>(),
                d_nwg.ptr<float>(), d_cs1.ptr<float>(), d_ss1.ptr<float>(),
                out_step.ptr<float>() + m * v_dim, nk, nv, dk, dv, 1e-6f, qs, ctx.stream());
        }
        cuda_ops::fused_gated_deltanet_prefill_f32(
            d_qkv.ptr<float>(), conv_dim, d_z.ptr<float>(), v_dim,
            d_a.ptr<float>(), nv, d_b.ptr<float>(), nv,
            d_cw.ptr<float>(), d_cb.ptr<float>(), d_al.ptr<float>(), d_dt.ptr<float>(),
            d_nwg.ptr<float>(), d_cs2.ptr<float>(), d_ss2.ptr<float>(),
            out_pf.ptr<float>(), M, nk, nv, dk, dv, 1e-6f, qs, ctx.stream(), gdn_ws.ptr<float>(), gdn_ws.numel());
        HostTensor h1 = ctx.download(out_step);
        HostTensor h2 = ctx.download(out_pf);
        std::vector<double> ref_v(h1.data<float>(), h1.data<float>() + M * v_dim);
        expect_close(std::vector<float>(h2.data<float>(), h2.data<float>() + M * v_dim),
                     ref_v, 1e-4, 1e-4, "CUDA GDN 3-stage prefill M=12 vs step-by-step");
    }

    // Step 2: Fused Add+RMSNorm+Linear Q4_A/Q8_A + MiniGraph optimization stats & IR parity
    {
        HostTensor hres_b = rand_f32({1, K}, rng, 0.3f);
        Tensor db = ctx.upload(hres_b, false);
        Tensor d_res_out = ctx.scratch_arena().alloc_tensor({1, K}, DType::F32);
        Tensor dy_add_q4a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
        Tensor dy_add_q8a = ctx.scratch_arena().alloc_tensor({1, N}, DType::F32);
        cuda_ops::fused_add_rmsnorm_linear_q4_a(dx1.ptr<float>(), db.ptr<float>(), dnw.ptr<float>(),
                                                dq4a.ptr(), d_res_out.ptr<float>(), dy_add_q4a.ptr<float>(),
                                                N, K, 1e-6f, ctx.stream());
        cuda_ops::fused_add_rmsnorm_linear_q8_a(dx1.ptr<float>(), db.ptr<float>(), dnw.ptr<float>(),
                                                dq8a.ptr(), d_res_out.ptr<float>(), dy_add_q8a.ptr<float>(),
                                                N, K, 1e-6f, ctx.stream());

        std::vector<double> ref_sum(static_cast<size_t>(K));
        double ss = 0.0;
        for (int k = 0; k < K; ++k) {
            ref_sum[k] = static_cast<double>(hx1.data<float>()[k]) + static_cast<double>(hres_b.data<float>()[k]);
            ss += ref_sum[k] * ref_sum[k];
        }
        double inv = 1.0 / std::sqrt(ss / K + 1e-6);
        std::vector<double> ref_add_y(static_cast<size_t>(N), 0.0);
        for (int n = 0; n < N; ++n) {
            double acc = 0.0;
            for (int k = 0; k < K; ++k) {
                acc += (ref_sum[k] * inv * hnw.data<float>()[k]) * hw.data<float>()[n * K + k];
            }
            ref_add_y[static_cast<size_t>(n)] = acc;
        }
        HostTensor hres_chk = ctx.download(d_res_out);
        HostTensor hy_add8 = ctx.download(dy_add_q8a);
        HostTensor hy_add4 = ctx.download(dy_add_q4a);
        expect_close(std::vector<float>(hres_chk.data<float>(), hres_chk.data<float>() + K), ref_sum, 1e-5, 1e-5, "CUDA fused_add_rmsnorm res_out");
        expect_close(std::vector<float>(hy_add8.data<float>(), hy_add8.data<float>() + N), ref_add_y, 0.08, 0.02, "CUDA fused_add_rmsnorm_linear_q8_a");
        expect_close(std::vector<float>(hy_add4.data<float>(), hy_add4.data<float>() + N), ref_add_y, 0.75, 0.14, "CUDA fused_add_rmsnorm_linear_q4_a");
    }
    ctx.scratch_arena().reset();
}

void test_cuda_gated_deltanet_fused_vs_unfused_and_fp64(CudaContext& ctx) {
    std::cout << "[3/7] Testing CUDA Fused Register-Resident GatedDeltaNet vs FP64 ref...\n";
    std::mt19937_64 rng(303);
    int nk = 4, nv = 8, dk = 64, dv = 64;
    int q_dim = nk * dk, v_dim = nv * dv, conv_dim = 2 * q_dim + v_dim;

    HostTensor h_cw = rand_f32({conv_dim, 4}, rng, 0.2f);
    HostTensor h_cb = rand_f32({conv_dim}, rng, 0.05f);
    HostTensor h_alog = HostTensor::create({nv}, DType::F32);
    for (int i = 0; i < nv; ++i) h_alog.data<float>()[i] = -1.5f;
    HostTensor h_dtb = rand_f32({nv}, rng, 0.05f);
    HostTensor h_nw = rand_f32({dv}, rng, 0.5f);

    Tensor d_cw = ctx.upload(h_cw, false);
    Tensor d_cb = ctx.upload(h_cb, false);
    Tensor d_alog = ctx.upload(h_alog, false);
    Tensor d_dtb = ctx.upload(h_dtb, false);
    Tensor d_nw = ctx.upload(h_nw, false);

    Tensor d_cs_f = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
    Tensor d_ss_f = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
    Tensor d_cs_u = ctx.scratch_arena().alloc_tensor({conv_dim, 3}, DType::F32);
    Tensor d_ss_u = ctx.scratch_arena().alloc_tensor({nv, dk, dv}, DType::F32);
    FUSIONRT_CUDA_CHECK(cudaMemset(d_cs_f.ptr(), 0, d_cs_f.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(d_ss_f.ptr(), 0, d_ss_f.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(d_cs_u.ptr(), 0, d_cs_u.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(d_ss_u.ptr(), 0, d_ss_u.nbytes()));

    // Host FP64 reference state
    std::vector<double> r_cs(static_cast<size_t>(conv_dim * 3), 0.0);
    std::vector<double> r_ss(static_cast<size_t>(nv * dk * dv), 0.0);
    std::vector<double> r_out(static_cast<size_t>(v_dim), 0.0);

    float q_scale = 1.0f / std::sqrt(static_cast<float>(dk));
    int steps = 6;
    Tensor d_out_f = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
    Tensor d_out_u = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);
    Tensor sqkv = ctx.scratch_arena().alloc_tensor({conv_dim}, DType::F32);
    Tensor sy = ctx.scratch_arena().alloc_tensor({v_dim}, DType::F32);

    for (int step = 0; step < steps; ++step) {
        HostTensor h_qkv = rand_f32({conv_dim}, rng, 0.3f);
        HostTensor h_z = rand_f32({v_dim}, rng, 0.3f);
        HostTensor h_a = rand_f32({nv}, rng, 0.2f);
        HostTensor h_b = rand_f32({nv}, rng, 0.2f);

        Tensor dqkv = ctx.upload(h_qkv, false);
        Tensor dz = ctx.upload(h_z, false);
        Tensor da = ctx.upload(h_a, false);
        Tensor db = ctx.upload(h_b, false);

        cuda_ops::fused_gated_deltanet_step_f32(
            dqkv.ptr<float>(), dz.ptr<float>(), da.ptr<float>(), db.ptr<float>(),
            d_cw.ptr<float>(), d_cb.ptr<float>(), d_alog.ptr<float>(), d_dtb.ptr<float>(),
            d_nw.ptr<float>(), d_cs_f.ptr<float>(), d_ss_f.ptr<float>(), d_out_f.ptr<float>(),
            nk, nv, dk, dv, 1e-6f, q_scale, ctx.stream(), nullptr, true);

        cuda_ops::unfused_gated_deltanet_step_f32(
            dqkv.ptr<float>(), dz.ptr<float>(), da.ptr<float>(), db.ptr<float>(),
            d_cw.ptr<float>(), d_cb.ptr<float>(), d_alog.ptr<float>(), d_dtb.ptr<float>(),
            d_nw.ptr<float>(), d_cs_u.ptr<float>(), d_ss_u.ptr<float>(), d_out_u.ptr<float>(),
            sqkv.ptr<float>(), sy.ptr<float>(), nk, nv, dk, dv, 1e-6f, q_scale, ctx.stream(), true);

        // FP64 step
        std::vector<double> qkv_silu(static_cast<size_t>(conv_dim), 0.0);
        for (int ch = 0; ch < conv_dim; ++ch) {
            double c0 = r_cs[ch * 3 + 0], c1 = r_cs[ch * 3 + 1], c2 = r_cs[ch * 3 + 2];
            double cur = h_qkv.data<float>()[ch];
            double acc = c0 * h_cw.data<float>()[ch * 4 + 0] + c1 * h_cw.data<float>()[ch * 4 + 1]
                       + c2 * h_cw.data<float>()[ch * 4 + 2] + cur * h_cw.data<float>()[ch * 4 + 3]
                       + h_cb.data<float>()[ch];
            r_cs[ch * 3 + 0] = c1; r_cs[ch * 3 + 1] = c2; r_cs[ch * 3 + 2] = cur;
            qkv_silu[ch] = acc / (1.0 + std::exp(-acc));
        }
        for (int kh = 0; kh < nk; ++kh) {
            double qss = 0.0, kss = 0.0;
            for (int i = 0; i < dk; ++i) {
                qss += qkv_silu[kh * dk + i] * qkv_silu[kh * dk + i];
                kss += qkv_silu[q_dim + kh * dk + i] * qkv_silu[q_dim + kh * dk + i];
            }
            double qinv = 1.0 / std::sqrt(qss + 1e-6) * q_scale;
            double kinv = 1.0 / std::sqrt(kss + 1e-6);
            for (int i = 0; i < dk; ++i) {
                qkv_silu[kh * dk + i] *= qinv;
                qkv_silu[q_dim + kh * dk + i] *= kinv;
            }
        }
        for (int vh = 0; vh < nv; ++vh) {
            int kh = vh % nk;
            double x_sp = h_a.data<float>()[vh] + h_dtb.data<float>()[vh];
            double sp = (x_sp > 20.0) ? x_sp : std::log1p(std::exp(x_sp));
            double g = std::exp(-std::exp(h_alog.data<float>()[vh]) * sp);
            double beta = 1.0 / (1.0 + std::exp(-h_b.data<float>()[vh]));
            std::vector<double> y(static_cast<size_t>(dv), 0.0);
            for (int j = 0; j < dv; ++j) {
                double kv_mem = 0.0;
                for (int i = 0; i < dk; ++i) {
                    size_t idx = (static_cast<size_t>(vh) * dk + i) * dv + j;
                    r_ss[idx] *= g;
                    kv_mem += r_ss[idx] * qkv_silu[q_dim + kh * dk + i];
                }
                double v_j = qkv_silu[2 * q_dim + vh * dv + j];
                double delta = (v_j - kv_mem) * beta;
                for (int i = 0; i < dk; ++i) {
                    size_t idx = (static_cast<size_t>(vh) * dk + i) * dv + j;
                    r_ss[idx] += qkv_silu[q_dim + kh * dk + i] * delta;
                    y[j] += r_ss[idx] * qkv_silu[kh * dk + i];
                }
            }
            double yss = 0.0;
            for (int j = 0; j < dv; ++j) yss += y[j] * y[j];
            double yinv = 1.0 / std::sqrt(yss / dv + 1e-6);
            for (int j = 0; j < dv; ++j) {
                double zv = h_z.data<float>()[vh * dv + j];
                double sz = zv / (1.0 + std::exp(-zv));
                r_out[vh * dv + j] = (y[j] * yinv * h_nw.data<float>()[j]) * sz;
            }
        }
    }

    HostTensor hf = ctx.download(d_out_f);
    HostTensor hu = ctx.download(d_out_u);
    expect_close(std::vector<float>(hf.data<float>(), hf.data<float>() + v_dim), r_out, 1e-4, 1e-4, "CUDA fused GatedDeltaNet 6-step vs FP64");
    expect_close(std::vector<float>(hu.data<float>(), hu.data<float>() + v_dim), r_out, 1e-4, 1e-4, "CUDA unfused GatedDeltaNet 6-step vs FP64");
    ctx.scratch_arena().reset();
}

void test_cuda_wave1_decode(CudaContext& ctx) {
    std::cout << "Testing Q8_1 decode kernels and CUDA Graph replay...\n";
    std::mt19937_64 rng(451);
    auto& arena = ctx.scratch_arena();
    size_t mark = arena.mark();
    constexpr int M = 2, N = 19, K = 1280;
    constexpr int blocks = K / 256;
    const float eps = 1e-6f;
    HostTensor x = rand_f32({M, K}, rng);
    HostTensor res = rand_f32({M, K}, rng);
    HostTensor norm = rand_f32({K}, rng, 1.0f);
    HostTensor gate_up = rand_f32({M, 2 * K}, rng);
    std::fill_n(x.data<float>(), 32, 0.0f);
    std::fill_n(res.data<float>(), 32, 0.0f);
    std::fill_n(gate_up.data<float>(), 32, 0.0f);
    Tensor dx = ctx.upload(x, false);
    Tensor dr = ctx.upload(res, false);
    Tensor dn = ctx.upload(norm, false);
    Tensor dgu = ctx.upload(gate_up, false);
    Tensor res_out = arena.alloc_tensor({M, K}, DType::F32);
    Tensor qx = arena.alloc_tensor({M, K}, DType::Q8_1);
    Tensor qsw = arena.alloc_tensor({M, K}, DType::Q8_1);
    cuda_ops::quantize_x_q8_1(dx.ptr<float>(), dr.ptr<float>(), dn.ptr<float>(), res_out.ptr<float>(),
                             qx.ptr(), M, K, eps, ctx.stream());
    cuda_ops::swiglu_interleaved_q8_1(dgu.ptr<float>(), qsw.ptr(), M, K, ctx.stream());

    std::vector<double> activation(M * K), swiglu(M * K), residual(M * K);
    for (int m = 0; m < M; ++m) {
        double squares = 0.0;
        for (int k = 0; k < K; ++k) {
            int j = m * K + k;
            residual[j] = x.data<float>()[j] + res.data<float>()[j];
            squares += residual[j] * residual[j];
            float gate = gate_up.data<float>()[2 * m * K + k];
            float up = gate_up.data<float>()[2 * m * K + K + k];
            swiglu[j] = gate / (1.0 + std::exp(-gate)) * up;
        }
        double inv = 1.0 / std::sqrt(squares / K + eps);
        for (int k = 0; k < K; ++k) activation[m * K + k] = residual[m * K + k] * inv * norm.data<float>()[k];
    }
    auto hr = ctx.download(res_out);
    expect_close({hr.data<float>(), hr.data<float>() + M * K}, residual, 1e-7, 1e-7, "Q8_1 residual");
    auto check_q8 = [&](const Tensor& t, const std::vector<double>& ref, const char* label) {
        auto host = ctx.download(t);
        std::vector<float> reconstructed(M * K);
        for (int b = 0; b < M * K / 32; ++b) {
            const auto& q = host.data<BlockQ8_1>()[b];
            const auto* bytes = reinterpret_cast<const int8_t*>(q.qs);
            int lo = 0, hi = 0;
            for (int i = 0; i < 32; ++i) {
                reconstructed[32 * b + i] = q.d * bytes[i];
                (i < 16 ? lo : hi) += bytes[i];
            }
            if (q.s_lo != q.d * lo || q.s_hi != q.d * hi || q.s_all != q.d * (lo + hi)) {
                throw std::runtime_error("Incorrect Q8_1 activation sums");
            }
        }
        expect_close(reconstructed, ref, 0.025, 0.012, label);
        return reconstructed;
    };
    auto quantized = check_q8(qx, activation, "residual + RMSNorm + Q8_1");
    check_q8(qsw, swiglu, "SwiGLU + Q8_1");

    {
        constexpr int wide = 4096;
        const auto wide_x = ctx.upload(rand_f32({2, wide}, rng), false);
        const auto wide_res = ctx.upload(rand_f32({2, wide}, rng), false);
        const auto wide_norm = ctx.upload(rand_f32({wide}, rng), false);
        const auto output = arena.alloc_tensor({1, wide}, DType::Q8_1);
        const auto reference = arena.alloc_tensor({2, wide}, DType::Q8_1);
        const auto out_res = arena.alloc_tensor({1, wide}, DType::F32);
        const auto ref_res = arena.alloc_tensor({2, wide}, DType::F32);
        for (const float* add : std::array<const float*, 2>{nullptr, wide_res.ptr<float>()}) {
            cuda_ops::quantize_x_q8_1(wide_x.ptr<float>(), add, wide_norm.ptr<float>(),
                                     ref_res.ptr<float>(), reference.ptr(), 2, wide, eps, ctx.stream());
            cuda_ops::quantize_x_q8_1(wide_x.ptr<float>(), add, wide_norm.ptr<float>(),
                                     out_res.ptr<float>(), output.ptr(), 1, wide, eps, ctx.stream());
            if (std::memcmp(ctx.download(output).storage.data(), ctx.download(reference).storage.data(), output.nbytes()) ||
                std::memcmp(ctx.download(out_res).storage.data(), ctx.download(ref_res).storage.data(), out_res.nbytes())) {
                throw std::runtime_error("Single-row Q8_1 RMSNorm changed quantized or residual values");
            }
        }
    }

    auto check_gemv = [&]<typename W>(DType dtype, auto gemv, auto dequantize) {
        HostTensor w = HostTensor::create({N, K}, dtype);
        for (uint8_t& byte : w.storage) byte = static_cast<uint8_t>(rng());
        for (int b = 0; b < N * blocks; ++b) {
            W& block = w.data<W>()[b];
            if constexpr (std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ8_A>) {
                for (auto& d : block.d) d = fp32_to_fp16(0.002f + 0.0001f * (b % 7));
            } else {
                block.d = fp32_to_fp16(0.00002f + 0.000001f * (b % 7));
                if constexpr (!std::is_same_v<W, BlockQ6_K>) block.dmin = fp32_to_fp16(0.0001f);
            }
        }
        Tensor dw = ctx.upload(w, false);
        Tensor dy = arena.alloc_tensor({N}, DType::F32);
        gemv(qx.ptr(), dw.ptr(), dy.ptr<float>(), N, K, ctx.stream(), {});
        auto hy = ctx.download(dy);
        std::vector<float> wf(K);
        std::vector<double> ref(N), ref_quantized(N);
        for (int n = 0; n < N; ++n) {
            dequantize(w.data<W>() + n * blocks, wf.data(), K);
            for (int k = 0; k < K; ++k) {
                ref[n] += wf[k] * activation[k];
                ref_quantized[n] += static_cast<double>(wf[k]) * quantized[k];
            }
        }
        std::vector<float> got(hy.data<float>(), hy.data<float>() + N);
        expect_close(got, ref_quantized, 2e-5, 1e-5, dtype_name(dtype));
        expect_close(got, ref, 0.05, 0.022, std::string(dtype_name(dtype)) + " vs FP32 activations");
        if constexpr (std::is_same_v<W, BlockQ5_K>) {
            Tensor prefill = arena.alloc_tensor({M, N}, DType::F32);
            cuda_ops::fused_gemv_q5_k(dx.ptr<float>(), dw.ptr(), prefill.ptr<float>(), M, N, K, ctx.stream());
            auto prefill_out = ctx.download(prefill);
            Tensor dw16 = arena.alloc_tensor({N, K}, DType::F16);
            cuda_ops::dequantize_to_f16(dw.ptr(), dtype, dw16.ptr<uint16_t>(), N, K, ctx.stream());
            auto hw16 = ctx.download(dw16);
            std::vector<double> prefill_ref(M * N, 0.0);
            for (int n = 0; n < N; ++n) {
                dequantize(w.data<W>() + n * blocks, wf.data(), K);
                for (int k = 0; k < K; ++k) {
                    uint16_t half = hw16.data<uint16_t>()[n * K + k];
                    if (half != fp32_to_fp16(wf[k]) && std::abs(fp16_to_fp32(half) - wf[k]) > 2e-5f) {
                        throw std::runtime_error("Incorrect Q5_K to F16 conversion");
                    }
                    for (int m = 0; m < M; ++m) {
                        prefill_ref[m * N + n] += static_cast<double>(wf[k]) * x.data<float>()[m * K + k];
                    }
                }
            }
            expect_close({prefill_out.data<float>(), prefill_out.data<float>() + M * N},
                         prefill_ref, 1e-5, 1e-5, "Q5_K short prefill");
        }
    };
    check_gemv.operator()<BlockQ4_A>(DType::Q4_A, cuda_ops::gemv_q4_a_q8_1, dequantize_row_q4_a);
    check_gemv.operator()<BlockQ8_A>(DType::Q8_A, cuda_ops::gemv_q8_a_q8_1, dequantize_row_q8_a);
    check_gemv.operator()<BlockQ6_K>(DType::Q6_K, cuda_ops::gemv_q6_k_q8_1, dequantize_row_q6_k);
    check_gemv.operator()<BlockQ4_K>(DType::Q4_K, cuda_ops::gemv_q4_k_q8_1, dequantize_row_q4_k);
    check_gemv.operator()<BlockQ5_K>(DType::Q5_K, cuda_ops::gemv_q5_k_q8_1, dequantize_row_q5_k);
    arena.reset(mark);

    constexpr int nk = 4, nv = 16, dk = 128, dv = 128;
    constexpr int vdim = nv * dv, conv = 2 * nk * dk + vdim;
    auto upload = [&](const std::vector<int64_t>& shape, float scale) {
        return ctx.upload(rand_f32(shape, rng, scale), false);
    };
    Tensor cw = upload({conv, 4}, 0.2f), cb = upload({conv}, 0.05f);
    Tensor alog = upload({nv}, 0.4f), dt = upload({nv}, 0.1f), nw = upload({dv}, 0.5f);
    auto init_conv = rand_f32({conv, 3}, rng, 0.2f);
    auto init_ssm = rand_f32({nv, dk, dv}, rng, 0.1f);
    Tensor conv_state[] = {ctx.upload(init_conv, false), ctx.upload(init_conv, false)};
    Tensor ssm[] = {ctx.upload(init_ssm, false), ctx.upload(init_ssm, false)};
    Tensor out[] = {arena.alloc_tensor({vdim}, DType::F32), arena.alloc_tensor({vdim}, DType::F32)};
    Tensor ws = arena.alloc_tensor({conv + vdim + 2 * nv}, DType::F32);
    for (int step = 0; step < 3; ++step) {
        Tensor qkv = upload({conv}, 0.3f), z = upload({vdim}, 0.3f);
        Tensor a = upload({nv}, 0.2f), b = upload({nv}, 0.2f);
        for (int i = 0; i < 2; ++i) {
            cuda_ops::fused_gated_deltanet_step_f32(
                qkv.ptr<float>(), z.ptr<float>(), a.ptr<float>(), b.ptr<float>(), cw.ptr<float>(), cb.ptr<float>(),
                alog.ptr<float>(), dt.ptr<float>(), nw.ptr<float>(), conv_state[i].ptr<float>(), ssm[i].ptr<float>(),
                out[i].ptr<float>(), nk, nv, dk, dv, eps, 1.0f / std::sqrt(float(dk)), ctx.stream(),
                i == 0 ? nullptr : ws.ptr<float>());
        }
    }
    auto gdn_ref = ctx.download(out[0]), gdn_split = ctx.download(out[1]);
    for (int i = 0; i < vdim; ++i) {
        if (!(std::abs(gdn_split.data<float>()[i] - gdn_ref.data<float>()[i]) < 1e-4f)) {
            throw std::runtime_error("64-CTA GDN decode exceeded max_abs");
        }
    }
    expect_close({gdn_split.data<float>(), gdn_split.data<float>() + vdim},
                 {gdn_ref.data<float>(), gdn_ref.data<float>() + vdim}, 9.99e-5, 1e-4, "64-CTA GDN decode");
    arena.reset(mark);

    HybridModelConfig cfg;
    cfg.arch = "hybrid";
    cfg.vocab_size = 256;
    cfg.num_layers = 4;
    cfg.dense.hidden_size = cfg.gdn.hidden_size = 256;
    cfg.dense.num_attention_heads = 4;
    cfg.dense.num_key_value_heads = 2;
    cfg.dense.head_dim = 64;
    cfg.dense.intermediate_size = cfg.gdn.intermediate_size = 512;
    cfg.dense.max_seq_len = 256;
    cfg.dense.weight_dtype = cfg.gdn.weight_dtype = DType::Q4_A;
    cfg.gdn.num_k_heads = 4;
    cfg.gdn.num_v_heads = 16;
    cfg.gdn.head_k_dim = cfg.gdn.head_v_dim = 128;
    FusionModel model(cfg);
    size_t weights_mark = ctx.weight_arena().mark();
    model.init_synthetic(ctx, 452);
    int swiglu_fusions = 0;
    for (const auto& node : model.fused_ir_graph().nodes) {
        if (node.op == OpKind::SWIGLU_Q8_LINEAR) ++swiglu_fusions;
    }
    if (swiglu_fusions != cfg.num_layers) throw std::runtime_error("Missing Q8_1 SwiGLU fusion");
    const std::vector<int32_t> prompt{1, 10, 20};
    auto eager = model.generate_cuda(ctx, prompt, 16, true, false);
    auto graph = model.generate_cuda(ctx, prompt, 16, true, true);
    auto replay = model.generate_cuda(ctx, prompt, 16, true, true);
    if (graph != eager || replay != eager || eager.size() != prompt.size() + 16) {
        throw std::runtime_error("Q8_1 CUDA Graph token-ID replay mismatch");
    }
    std::cout << "  [PASS/CHECK] Q8_1 hybrid eager and CUDA Graph token IDs match exactly\n";
    ctx.weight_arena().reset(weights_mark);
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_sliding_window(CudaContext& ctx) {
    constexpr int seq = 192, window = 32, nq = 4, nkv = 2, splits = 8;
    std::mt19937_64 rng(343);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    for (const int hd : {64, 256}) {
        const int q_dim = nq * hd, kv_dim = nkv * hd;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const auto q = rand_f32({seq, q_dim}, rng, 0.5f);
        const auto k = rand_f32({seq, kv_dim}, rng, 0.5f);
        const auto v = rand_f32({seq, kv_dim}, rng, 0.25f);
        const auto dq = ctx.upload(q, false), dk = ctx.upload(k, false), dv = ctx.upload(v, false);
        const auto out = arena.alloc_tensor({seq, q_dim}, DType::F32);
        const auto decoded = arena.alloc_tensor({q_dim}, DType::F32);
        const auto partial = arena.alloc_tensor({nq, splits, hd}, DType::F32);
        const auto stats = arena.alloc_tensor({nq, splits, 2}, DType::F32);

        std::vector<double> reference(seq * q_dim);
        for (int pos = 0; pos < seq; ++pos) {
            const int first = std::max(0, pos + 1 - window);
            for (int h = 0; h < nq; ++h) {
                const int kh = h / (nq / nkv);
                const auto* query = q.data<float>() + pos * q_dim + h * hd;
                auto* dst = reference.data() + pos * q_dim + h * hd;
                float probability[window];
                float sum = 0.0f;
                for (int t = first; t <= pos; ++t) {
                    const auto* key = k.data<float>() + t * kv_dim + kh * hd;
                    float dot = 0.0f;
                    for (int i = 0; i < hd; ++i) dot += query[i] * key[i];
                    const float p = std::exp(dot * scale);
                    probability[t - first] = p;
                    sum += p;
                }
                for (int t = first; t <= pos; ++t) {
                    const auto* value = v.data<float>() + t * kv_dim + kh * hd;
                    for (int i = 0; i < hd; ++i) dst[i] += (probability[t - first] / sum) * value[i];
                }
            }
        }
        for (const auto dtype : {DType::F32, DType::F16}) {
            const auto cache = make_cuda_kv_view(arena, dtype, seq, nkv, hd);
            CudaKvPool pool;
            CudaKvPool::Table pages;
            pool.ensure(ctx, std::span(&cache, 1), pages, 0, seq);
            cuda_ops::batched_qk_norm_rope_kv_append_f32(dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(),
                nullptr, nullptr, nullptr, nullptr, seq, nq, nkv, hd, 0, 0, seq, 10000.0f, 1e-6f, true,
                ctx.stream(), q_dim, kv_dim, kv_dim, nullptr, dtype, cache);
            auto read = [&](const Tensor& t) {
                const auto host = ctx.download(t);
                return std::vector<float>(host.data<float>(), host.data<float>() + host.desc.numel());
            };

            cuda_ops::gqa_prefill_online_f32(dq.ptr<float>(), q_dim, cache, out.ptr<float>(), seq, nq, nkv,
                hd, 0, scale, ctx.stream(), window);
            expect_close(read(out), reference, 2.5e-4, 2e-3, "SWA paged online prefill");
            cuda_ops::gqa_prefill_causal_f32(dq.ptr<float>(), nullptr, nullptr, out.ptr<float>(), seq, nq, nkv,
                hd, 0, seq, scale, ctx.stream(), q_dim, dtype, cache, window);
            expect_close(read(out), reference, 8e-5, 1e-3, "SWA paged tiled prefill");
            for (const int length : {3, kKvPageTokens + 3, seq}) {
                cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>() + (length - 1) * q_dim, nullptr, nullptr,
                    decoded.ptr<float>(), partial.ptr<float>(), stats.ptr<float>(), nq, nkv, hd, nullptr,
                    length, seq, scale, splits, ctx.stream(), true, dtype, nullptr, nullptr, cache, window);
                auto first = reference.begin() + (length - 1) * q_dim;
                expect_close(read(decoded), {first, first + q_dim}, 8e-5, 1e-3, "SWA paged Split-KV decode");
            }
        }
        arena.reset(mark);
    }
    std::cout << "  [PASS] SWA window 32, 192 tokens, online/tiled prefill and Split-KV decode vs FP32\n";
}

void test_cuda_gemma_epilogues(CudaContext& ctx) {
    constexpr int rows = 3, width = 64;
    constexpr float eps = 1e-6f, cap = 0.5f;
    std::mt19937_64 rng(419);
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    const auto gate_up = rand_f32({rows, 2 * width}, rng, 2.0f);
    const auto w = rand_f32({width}, rng, 1.0f);
    const auto dgu = ctx.upload(gate_up, false), dw = ctx.upload(w, false);
    const auto out = arena.alloc_tensor({rows, width}, DType::F32);
    const auto q8 = arena.alloc_tensor({rows, width}, DType::Q8_1);
    std::vector<double> act(rows * width), post(rows * width);
    for (int r = 0; r < rows; ++r) {
        const auto* gate = gate_up.data<float>() + r * 2 * width;
        float sq = 0.0f;
        for (int i = 0; i < width; ++i) {
            const float x = gate[i];
            const float y = 0.5f * x * (1.0f + std::tanh(std::sqrt(2.0f / 3.14159265359f) *
                (x + 0.044715f * x * x * x))) * gate[width + i];
            act[r * width + i] = y;
            sq += y * y;
        }
        const float inv = 1.0f / std::sqrt(sq / width + eps);
        for (int i = 0; i < width; ++i) post[r * width + i] = act[r * width + i] * inv * w.data<float>()[i];
    }
    auto read = [&]() {
        const auto h = ctx.download(out);
        return std::vector<float>(h.data<float>(), h.data<float>() + rows * width);
    };
    cuda_ops::geglu_interleaved_f32(dgu.ptr<float>(), out.ptr<float>(), rows, width, ctx.stream());
    expect_close(read(), act, 2e-6, 1e-6, "GeGLU F32");
    cuda_ops::geglu_interleaved_q8_1(dgu.ptr<float>(), q8.ptr(), rows, width, ctx.stream());
    const auto packed = ctx.download(q8);
    for (int i = 0; i < rows * width; i += 32) {
        const auto& block = packed.data<BlockQ8_1>()[i / 32];
        const auto* bytes = reinterpret_cast<const int8_t*>(block.qs);
        float maximum = 0.0f;
        int lo = 0, hi = 0;
        for (int j = 0; j < 32; ++j) {
            maximum = std::max(maximum, static_cast<float>(std::abs(act[i + j])));
            (j < 16 ? lo : hi) += bytes[j];
            if (std::abs(block.d * bytes[j] - act[i + j]) > block.d * 0.5f + 3e-6f) {
                throw std::runtime_error("GeGLU Q8_1 quantization parity");
            }
        }
        if (std::abs(block.d - maximum / 127.0f) > 1e-7f ||
            std::abs(block.s_lo - block.d * lo) > 1e-6f ||
            std::abs(block.s_hi - block.d * hi) > 1e-6f ||
            std::abs(block.s_all - block.d * (lo + hi)) > 1e-6f) {
            throw std::runtime_error("GeGLU Q8_1 block metadata parity");
        }
    }
    const auto normalized = arena.alloc_tensor({rows, width}, DType::F32);
    cuda_ops::rmsnorm_f32(out.ptr<float>(), dw.ptr<float>(), normalized.ptr<float>(), rows, width, eps, ctx.stream());
    auto host = ctx.download(normalized);
    expect_close({host.data<float>(), host.data<float>() + rows * width}, post, 2e-6, 1e-6, "Gemma post RMSNorm");
    cuda_ops::softcap_f32(normalized.ptr<float>(), rows * width, cap, ctx.stream());
    for (auto& x : post) x = cap * std::tanh(x / cap);
    host = ctx.download(normalized);
    expect_close({host.data<float>(), host.data<float>() + rows * width}, post, 2e-6, 1e-6, "Gemma logit softcap");
    arena.reset(mark);
    std::cout << "  [PASS] GeGLU F32/Q8_1, post RMSNorm and logit softcap vs FP32\n";
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
void test_cuda_wave2_prefill(CudaContext& ctx) {
    std::cout << "Testing INT8 MMQ, chunkwise GatedDeltaNet, and tiled GQA prefill...\n";
    std::mt19937_64 rng(452);
    auto& arena = ctx.scratch_arena();
    auto upload_random = [&](std::initializer_list<int64_t> shape, float scale = 0.25f) {
        return ctx.upload(rand_f32(shape, rng, scale), false);
    };
    auto read = [&](const Tensor& t) {
        auto h = ctx.download(t);
        return std::vector<float>(h.data<float>(), h.data<float>() + h.desc.numel());
    };
    size_t mark = arena.mark();
    constexpr int M = 128, N = 73, K = 768;
    HostTensor x = rand_f32({M, K}, rng);
    auto dx = ctx.upload(x, false);
    auto qx = arena.alloc_tensor({M, K}, DType::Q8_1);
    cuda_ops::quantize_x_q8_1(dx.ptr<float>(), nullptr, nullptr, nullptr, qx.ptr(), M, K, 0.0f, ctx.stream());
    auto hq = ctx.download(qx);
    std::vector<float> quantized(M * K);
    for (int block = 0; block < M * K / 32; ++block) {
        const auto& q = hq.data<BlockQ8_1>()[block];
        const auto* qs = reinterpret_cast<const int8_t*>(q.qs);
        for (int k = 0; k < 32; ++k) quantized[block * 32 + k] = q.d * qs[k];
    }

    auto check_mmq = [&]<typename W>(DType dtype, auto mmq, auto dequantize) {
        constexpr int blocks = K / 256;
        HostTensor w = HostTensor::create({N, K}, dtype);
        for (auto& byte : w.storage) byte = static_cast<uint8_t>(rng());
        for (int i = 0; i < N * blocks; ++i) {
            auto& block = w.data<W>()[i];
            if constexpr (std::is_same_v<W, BlockQ4_A> || std::is_same_v<W, BlockQ8_A>) {
                for (int sub = 0; sub < 8; ++sub) block.d[sub] = fp32_to_fp16(0.002f + 0.0001f * sub);
            } else {
                block.d = fp32_to_fp16(0.00002f + 0.000001f * (i % 7));
                if constexpr (!std::is_same_v<W, BlockQ6_K>) block.dmin = fp32_to_fp16(0.0001f);
            }
        }
        auto dw = ctx.upload(w, false);
        std::vector<float> weights(N * K);
        for (int n = 0; n < N; ++n) dequantize(w.data<W>() + n * blocks, weights.data() + n * K, K);
        for (int rows : {4, 17, 64, M}) {
            auto dy = arena.alloc_tensor({rows, N}, DType::F32);
            mmq(qx.ptr(), dw.ptr(), dy.ptr<float>(), rows, N, K, ctx.stream(), ctx.cost_hardware());
            std::vector<double> ref(rows * N), ref_quantized(rows * N);
            for (int m = 0; m < rows; ++m) {
                for (int n = 0; n < N; ++n) {
                    for (int k = 0; k < K; ++k) {
                        double weight = weights[n * K + k];
                        ref[m * N + n] += weight * x.data<float>()[m * K + k];
                        ref_quantized[m * N + n] += weight * quantized[m * K + k];
                    }
                }
            }
            std::string label = std::string("MMQ ") + dtype_name(dtype) + " M=" + std::to_string(rows);
            auto got = read(dy);
            expect_close(got, ref_quantized, 2e-5, 1e-5, label + " with dequantized inputs");
            expect_close(got, ref, 0.0, 0.015, label + " with FP32 activations");
        }
    };
    check_mmq.operator()<BlockQ4_A>(DType::Q4_A, cuda_ops::mmq_q4_a_q8_1, dequantize_row_q4_a);
    check_mmq.operator()<BlockQ8_A>(DType::Q8_A, cuda_ops::mmq_q8_a_q8_1, dequantize_row_q8_a);
    check_mmq.operator()<BlockQ6_K>(DType::Q6_K, cuda_ops::mmq_q6_k_q8_1, dequantize_row_q6_k);
    check_mmq.operator()<BlockQ4_K>(DType::Q4_K, cuda_ops::mmq_q4_k_q8_1, dequantize_row_q4_k);
    check_mmq.operator()<BlockQ5_K>(DType::Q5_K, cuda_ops::mmq_q5_k_q8_1, dequantize_row_q5_k);
    arena.reset(mark);

    constexpr int nk = 2, nv = 4, dk = 128, dv = 128;
    constexpr int vdim = nv * dv, conv = 2 * nk * dk + vdim;
    const float qs = 1.0f / std::sqrt(static_cast<float>(dk));
    auto qkv = upload_random({M, conv}), z = upload_random({M, vdim});
    auto a = upload_random({M, nv}), b = upload_random({M, nv});
    auto cw = upload_random({conv, 4}), cb = upload_random({conv}, 0.05f);
    auto al = upload_random({nv}), dt = upload_random({nv}), norm = upload_random({dv});
    auto init_conv = rand_f32({conv, 3}, rng), init_state = rand_f32({nv, dk, dv}, rng, 0.01f);
    auto cs_ref = ctx.upload(init_conv, false), ss_ref = ctx.upload(init_state, false);
    auto cs = ctx.upload(init_conv, false), ss = ctx.upload(init_state, false);
    auto ref = arena.alloc_tensor({M, vdim}, DType::F32), out = arena.alloc_tensor({M, vdim}, DType::F32);
    auto scratch = arena.alloc_tensor({static_cast<int64_t>(
        cuda_ops::gated_deltanet_prefill_scratch_floats(M, nk, nv, dk, dv))}, DType::F32);
    for (int m = 0; m < M; ++m) {
        cuda_ops::fused_gated_deltanet_step_f32(
            qkv.ptr<float>() + m * conv, z.ptr<float>() + m * vdim, a.ptr<float>() + m * nv, b.ptr<float>() + m * nv,
            cw.ptr<float>(), cb.ptr<float>(), al.ptr<float>(), dt.ptr<float>(), norm.ptr<float>(),
            cs_ref.ptr<float>(), ss_ref.ptr<float>(), ref.ptr<float>() + m * vdim,
            nk, nv, dk, dv, 1e-6f, qs, ctx.stream(), nullptr, true);
    }
    cuda_ops::fused_gated_deltanet_prefill_f32(
        qkv.ptr<float>(), conv, z.ptr<float>(), vdim, a.ptr<float>(), nv, b.ptr<float>(), nv,
        cw.ptr<float>(), cb.ptr<float>(), al.ptr<float>(), dt.ptr<float>(), norm.ptr<float>(),
        cs.ptr<float>(), ss.ptr<float>(), out.ptr<float>(), M, nk, nv, dk, dv, 1e-6f, qs, ctx.stream(),
        scratch.ptr<float>(), scratch.numel(), true);
    auto expected = read(ref);
    expect_close(read(out), {expected.begin(), expected.end()}, 9.99e-5, 9.99e-5, "GDN prefill output M=128");
    expected = read(ss_ref);
    expect_close(read(ss), {expected.begin(), expected.end()}, 9.99e-5, 9.99e-5, "GDN recurrent state M=128");
    expected = read(cs_ref);
    expect_close(read(cs), {expected.begin(), expected.end()}, 0.0, 0.0, "GDN convolution state M=128");
    arena.reset(mark);

    constexpr int nq = 6, nkv = 2, start = 5;
    for (int hd : {128, 256}) {
        int qcols = nq * hd, stride = qcols + 64;
        float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        auto q = upload_random({M, stride}, 0.8f);
        auto k = upload_random({start + M, nkv, hd}, 0.8f);
        auto v = upload_random({start + M, nkv, hd});
        for (int rows : {17, M}) {
            auto seq = arena.alloc_tensor({rows, qcols}, DType::F32);
            auto tiled = arena.alloc_tensor({rows, qcols}, DType::F32);
            for (int m = 0; m < rows; ++m) {
                cuda_ops::gqa_splitk_decode_f32(q.ptr<float>() + m * stride, k.ptr<float>(), v.ptr<float>(),
                    seq.ptr<float>() + m * qcols, nullptr, nullptr, nq, nkv, hd, nullptr, start + m + 1,
                    start + M, scale, 1, ctx.stream());
            }
            cuda_ops::gqa_prefill_causal_f32(q.ptr<float>(), k.ptr<float>(), v.ptr<float>(), tiled.ptr<float>(),
                rows, nq, nkv, hd, start, start + M, scale, ctx.stream(), stride);
            expected = read(seq);
            expect_close(read(tiled), {expected.begin(), expected.end()}, 9.99e-6, 0.0,
                "GQA prefill head_dim=" + std::to_string(hd) + " M=" + std::to_string(rows));
            if (rows == M && hd == 128) {
                Tensor kh = arena.alloc_tensor({start + M, nkv, hd}, DType::F16);
                Tensor vh = arena.alloc_tensor({start + M, nkv, hd}, DType::F16);
                cuda_ops::dequantize_to_f16(k.ptr(), DType::F32, kh.ptr<uint16_t>(), start + M, nkv * hd, ctx.stream());
                cuda_ops::dequantize_to_f16(v.ptr(), DType::F32, vh.ptr<uint16_t>(), start + M, nkv * hd, ctx.stream());
                cuda_ops::gqa_prefill_causal_f32(q.ptr<float>(), kh.ptr(), vh.ptr(), tiled.ptr<float>(),
                    M, nq, nkv, hd, start, start + M, scale, ctx.stream(), stride, DType::F16);
                expect_close(read(tiled), {expected.begin(), expected.end()}, 4e-4, 1e-3, "GQA FP16 KV prefill");
                constexpr int splits = 4;
                auto partial = arena.alloc_tensor({nq, splits, hd}, DType::F32);
                auto stats = arena.alloc_tensor({nq, splits, 2}, DType::F32);
                auto step = arena.alloc_tensor({qcols}, DType::F32);
                cuda_ops::gqa_splitk_decode_f32(q.ptr<float>() + (M - 1) * stride, kh.ptr(), vh.ptr(),
                    step.ptr<float>(), partial.ptr<float>(), stats.ptr<float>(), nq, nkv, hd,
                    nullptr, start + M, start + M, scale, splits, ctx.stream(), true, DType::F16);
                expect_close(read(step), {expected.end() - qcols, expected.end()}, 4e-4, 1e-3,
                    "Grouped decode after FP16 KV prefill");
                const auto reference = arena.alloc_tensor({1, qcols}, DType::Q8_1);
                const auto packed = arena.alloc_tensor({1, qcols}, DType::Q8_1);
                cuda_ops::apply_sigmoid_gate_f32(step.ptr<float>(), q.ptr<float>(), qcols, ctx.stream());
                cuda_ops::quantize_x_q8_1(step.ptr<float>(), nullptr, nullptr, nullptr, reference.ptr(),
                                          1, qcols, 0, ctx.stream());
                cuda_ops::gqa_splitk_decode_f32(q.ptr<float>() + (M - 1) * stride, kh.ptr(), vh.ptr(),
                    nullptr, partial.ptr<float>(), stats.ptr<float>(), nq, nkv, hd,
                    nullptr, start + M, start + M, scale, splits, ctx.stream(), true, DType::F16,
                    packed.ptr<BlockQ8_1>(), q.ptr<float>());
                if (ctx.download(reference).storage != ctx.download(packed).storage) {
                    throw std::runtime_error("Fused attention and gate must produce the same Q8_1 blocks");
                }
            }
        }
        arena.reset(mark);
    }
    const auto old_kv_dtype = ctx.kv_dtype();
    for (int hd : {64, 128, 256}) {
        const int prefix = kKvPageTokens - 3, rows = 23, total = prefix + rows;
        const int stride = nq * hd + 64;
        const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
        const auto q = upload_random({total, stride}, 0.5f);
        const auto k = upload_random({total, nkv * hd}, 0.5f);
        const auto v = upload_random({total, nkv * hd}, 0.25f);
        const auto reference_out = arena.alloc_tensor({rows, nq * hd}, DType::F32);
        const auto online_out = arena.alloc_tensor({rows, nq * hd}, DType::F32);
        std::vector<double> f16;
        for (auto dtype : {DType::F16, DType::F32, DType::Q8_0}) {
            ctx.set_kv_dtype(dtype);
            const auto persistent_mark = ctx.weight_arena().mark();
            {
                const auto cache = make_cuda_kv_view(ctx.weight_arena(), dtype, 262144, nkv, hd);
                CudaKvPool pool;
                CudaKvPool::Table pages;
                auto append = [&](int from, int count) {
                    pool.ensure(ctx, std::span(&cache, 1), pages, from, from + count);
                    cuda_ops::batched_qk_norm_rope_kv_append_f32(
                        q.ptr<float>() + from * stride, k.ptr<float>() + from * nkv * hd, v.ptr<float>() + from * nkv * hd,
                        nullptr, nullptr, nullptr, nullptr, count, nq, nkv, hd, 0, from, total, 10000.0f, 1e-6f,
                        true, ctx.stream(), stride, nkv * hd, nkv * hd, nullptr, dtype, cache);
                };
                append(0, prefix);
                if (pool.pages() != 1) throw std::runtime_error("KV must allocate only the first page");
                if (dtype == DType::F16) {
                    cuda_ops::gqa_prefill_causal_f32(q.ptr<float>(), nullptr, nullptr, reference_out.ptr<float>(),
                        rows, nq, nkv, hd, 0, total, scale, ctx.stream(), stride, dtype, cache);
                    cuda_ops::gqa_prefill_online_f32(q.ptr<float>(), stride, cache, online_out.ptr<float>(), rows,
                        nq, nkv, hd, 0, scale, ctx.stream());
                    const auto first_page = read(reference_out);
                    expect_close(read(online_out), {first_page.begin(), first_page.end()}, 2e-4, 1e-3,
                                 "Online attention against the short FP16 path");
                }
                append(prefix, rows);
                const auto bytes = 4 * kKvPageTokens * nkv * hd * dtype_type_size(dtype) / dtype_block_size(dtype);
                if (pool.pages() != 2 || pool.bytes() != bytes) throw std::runtime_error("KV page payload size mismatch");
                const float* query = q.ptr<float>() + prefix * stride;
                cuda_ops::gqa_prefill_causal_f32(query, nullptr, nullptr, reference_out.ptr<float>(), rows,
                    nq, nkv, hd, prefix, total, scale, ctx.stream(), stride, dtype, cache);
                cuda_ops::gqa_prefill_online_f32(query, stride, cache, online_out.ptr<float>(), rows,
                    nq, nkv, hd, prefix, scale, ctx.stream());
                const auto reference = read(reference_out);
                const auto online = read(online_out);
                expect_close(online, {reference.begin(), reference.end()}, 2e-4, 1e-3, "Paged online attention");
                if (dtype == DType::F16) f16.assign(reference.begin(), reference.end());
                if (dtype == DType::Q8_0) expect_close(online, f16, 1.5e-3, 1e-2, "Q8 KV attention accuracy");
            }
            ctx.weight_arena().reset(persistent_mark);
        }
        arena.reset(mark);
    }
    ctx.set_kv_dtype(old_kv_dtype);
}

void test_cuda_qk_rope_and_gqa_64k(CudaContext& ctx) {
    std::cout << "[4/7] Testing CUDA Fused Q/K Norm + Partial RoPE + GQA 64K Split-K...\n";
    std::mt19937_64 rng(404);
    int nq = 8, nkv = 2, hd = 64, rot_dim = 32, pos = 65535, max_seq = 65536;
    HostTensor hq = rand_f32({nq, hd}, rng, 0.3f);
    HostTensor hk = rand_f32({nkv, hd}, rng, 0.3f);
    HostTensor hv = rand_f32({nkv, hd}, rng, 0.3f);
    HostTensor hqn = rand_f32({hd}, rng, 0.5f);
    HostTensor hkn = rand_f32({hd}, rng, 0.5f);

    Tensor dq = ctx.upload(hq, false);
    Tensor dk = ctx.upload(hk, false);
    Tensor dv = ctx.upload(hv, false);
    Tensor dqn = ctx.upload(hqn, false);
    Tensor dkn = ctx.upload(hkn, false);
    Tensor dkc = ctx.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
    Tensor dvc = ctx.scratch_arena().alloc_tensor({max_seq, nkv, hd}, DType::F32);
    FUSIONRT_CUDA_CHECK(cudaMemset(dkc.ptr(), 0, dkc.nbytes()));
    FUSIONRT_CUDA_CHECK(cudaMemset(dvc.ptr(), 0, dvc.nbytes()));

    // Fused Q/K Norm + Partial RoPE (rot_dim=32) + Append at pos=65535
    cuda_ops::fused_qk_norm_rope_kv_append_f32(
        dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(),
        dqn.ptr<float>(), dkn.ptr<float>(), dkc.ptr<float>(), dvc.ptr<float>(),
        nq, nkv, hd, rot_dim, nullptr, pos, max_seq, 1000000.0f, 1e-6f, true, ctx.stream());

    Tensor out_s1 = ctx.scratch_arena().alloc_tensor({nq, hd}, DType::F32);
    Tensor out_s16 = ctx.scratch_arena().alloc_tensor({nq, hd}, DType::F32);
    Tensor part_out = ctx.scratch_arena().alloc_tensor({nq, 16, hd}, DType::F32);
    Tensor part_ml = ctx.scratch_arena().alloc_tensor({nq, 16, 2}, DType::F32);
    float scale = 1.0f / std::sqrt(static_cast<float>(hd));
    cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>(), dkc.ptr<float>(), dvc.ptr<float>(),
                                    out_s1.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                                    nq, nkv, hd, nullptr, 65536, max_seq, scale, 1, ctx.stream());
    cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>(), dkc.ptr<float>(), dvc.ptr<float>(),
                                    out_s16.ptr<float>(), part_out.ptr<float>(), part_ml.ptr<float>(),
                                    nq, nkv, hd, nullptr, 65536, max_seq, scale, 16, ctx.stream());
    HostTensor h1 = ctx.download(out_s1);
    HostTensor h16 = ctx.download(out_s16);
    std::vector<double> ref(h1.data<float>(), h1.data<float>() + nq * hd);
    expect_close(std::vector<float>(h16.data<float>(), h16.data<float>() + nq * hd), ref, 1e-5, 1e-5, "CUDA GQA 64K Split-K=16 vs Split-K=1");
    const auto old_kv_dtype = ctx.kv_dtype();
    const auto mark = ctx.weight_arena().mark();
    ctx.set_kv_dtype(DType::Q8_0);
    {
        CudaKvPool pool;
        CudaKvPool::Table pages;
        const auto cache = make_cuda_kv_view(ctx.weight_arena(), ctx.kv_dtype(), max_seq, nkv, hd);
        pool.ensure(ctx, std::span(&cache, 1), pages, pos, max_seq);
        ctx.upload_into(hq, dq);
        ctx.upload_into(hk, dk);
        cuda_ops::fused_qk_norm_rope_kv_append_f32(dq.ptr<float>(), dk.ptr<float>(), dv.ptr<float>(),
            dqn.ptr<float>(), dkn.ptr<float>(), nullptr, nullptr, nq, nkv, hd, rot_dim, nullptr, pos, max_seq,
            1000000.0f, 1e-6f, true, ctx.stream(), nullptr, DType::Q8_0, cache);
        constexpr int splits = 128;
        auto parts = ctx.scratch_arena().alloc_tensor({nq, splits, hd}, DType::F32);
        auto ml = ctx.scratch_arena().alloc_tensor({nq, splits, 2}, DType::F32);
        cuda_ops::gqa_splitk_decode_f32(dq.ptr<float>(), nullptr, nullptr, out_s16.ptr<float>(),
            parts.ptr<float>(), ml.ptr<float>(), nq, nkv, hd, nullptr, max_seq, max_seq, scale,
            splits, ctx.stream(), true, DType::Q8_0, nullptr, nullptr, cache);
        const auto actual = ctx.download(out_s16);
        expect_close({actual.data<float>(), actual.data<float>() + nq * hd}, ref,
                     1e-7, 1e-2, "Paged Q8 attention with 128 splits at 64K");
    }
    ctx.set_kv_dtype(old_kv_dtype);
    ctx.weight_arena().reset(mark);
    ctx.scratch_arena().reset();
}
#endif

#ifdef FUSIONRT_ENABLE_CUDA
void test_review_cuda_kernels(CudaContext& ctx) {
    auto& arena = ctx.scratch_arena();
    const auto mark = arena.mark();
    auto buffer = arena.alloc_tensor({4}, DType::F32);
    auto captured_grid = [&](auto launch) {
        FUSIONRT_CUDA_CHECK(cudaStreamBeginCapture(ctx.stream(), cudaStreamCaptureModeThreadLocal));
        launch(buffer.ptr<float>());
        cudaGraph_t raw = nullptr;
        FUSIONRT_CUDA_CHECK(cudaStreamEndCapture(ctx.stream(), &raw));
        std::unique_ptr<std::remove_pointer_t<cudaGraph_t>, decltype(&cudaGraphDestroy)> graph(raw, cudaGraphDestroy);
        cudaGraphNode_t node = nullptr;
        size_t count = 1;
        FUSIONRT_CUDA_CHECK(cudaGraphGetNodes(graph.get(), &node, &count));
        if (count != 1) throw std::runtime_error("Expected a single captured CUDA kernel");
        cudaKernelNodeParams params{};
        FUSIONRT_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &params));
        return params.gridDim;
    };
    // Inspect the actual launch without allocating gigabytes of attention inputs.
    const auto gqa = captured_grid([&](float* p) {
        cuda_ops::gqa_prefill_causal_f32(p, p, p, p, 32768, 32, 2, 512, 0, 32768, 1.0f, ctx.stream());
    });
    if (gqa.x != 65536 || gqa.y != 2) throw std::runtime_error("HD512 prefill grid must fit CUDA's Y dimension");
    const auto ple = captured_grid([&](float* p) {
        cuda_ops::gemma_ple_gate_f32(p, p, p, 65536, 32769, 32769, 0, ctx.stream());
    });
    if (ple.x != (uint64_t{65536} * 32769 + 255) / 256) throw std::runtime_error("Large Gemma PLE grid overflow");

    bool invalid_router = false;
    try {
        cuda_ops::moe_topk_router_f32(nullptr, nullptr, nullptr, 8, 5, true, ctx.stream(), nullptr, true, 1.0f, 2, 1);
    } catch (const std::invalid_argument&) {
        invalid_router = true;
    }
    if (!invalid_router) throw std::runtime_error("Grouped router accepted more experts than the selected groups contain");

    auto upload = [&](std::initializer_list<float> values, std::vector<int64_t> dims) {
        auto host = HostTensor::create(dims, DType::F32);
        std::copy(values.begin(), values.end(), host.data<float>());
        return ctx.upload(host, false);
    };
    const auto gate = upload({1, 2}, {2}), up = upload({2, 3}, {2});
    cuda_ops::swiglu_f32(gate.ptr<float>(), up.ptr<float>(), up.ptr<float>(), 2, ctx.stream());
    cuda_ops::add_residual_f32(gate.ptr<float>(), up.ptr<float>(), up.ptr<float>(), 2, ctx.stream());
    const auto updated = ctx.download(up);
    expect_close({updated.data<float>(), updated.data<float>() + 2},
                 {1 + 2 / (1 + std::exp(-1.0)), 2 + 6 / (1 + std::exp(-2.0))},
                 1e-6, 1e-6, "Aliased up/output and residual/output");

    const auto q = upload({-1e20f, 0, 0}, {1,3});
    const auto kv = upload({1, 0, 0}, {1,3});
    const auto norm = upload({1e20f}, {1});
    const auto k_b = upload({1}, {1,1,1}), v_b = upload({1}, {1,1,1});
    const auto frequency = upload({1}, {1});
    const auto cache = make_cuda_kv_view(arena, DType::F32, 1, 1, 3, true);
    auto* page = static_cast<uint8_t*>(arena.alloc(cache.page_bytes()));
    FUSIONRT_CUDA_CHECK(cudaMemcpyAsync(cache.pages, &page, sizeof(page), cudaMemcpyHostToDevice, ctx.stream()));
    auto absorbed = arena.alloc_tensor({3}, DType::F32), latent = arena.alloc_tensor({1}, DType::F32);
    cuda_ops::mla_attention_f32(q.ptr<float>(), kv.ptr<float>(), norm.ptr<float>(), k_b, v_b, frequency.ptr<float>(),
        absorbed.ptr<float>(), latent.ptr<float>(), buffer.ptr<float>(), cache, 1, 1, 3, 0, nullptr, 1e-6f, 1.0f, 1.0f, ctx.stream());
    if (ctx.download(buffer).data<float>()[0] != 0.0f) throw std::runtime_error("All -Inf MLA scores must produce zero");
    arena.reset(mark);
    std::cout << "  [PASS] HD512 grid.x=65536, 64-bit PLE, MoE launch validation and all-masked MLA zero output\n";
}
#endif

}  // namespace fusionrt::test
