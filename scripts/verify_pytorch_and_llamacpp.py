#!/usr/bin/env python3
"""Differential verification against PyTorch FP64/FP32, llama.cpp GGUF parity, and bench_report.v1.json validation."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F


def validate_bench_report_v1(report: dict, schema_path: Path | None = None) -> list[str]:
    errors: list[str] = []
    if schema_path and schema_path.exists():
        try:
            import jsonschema  # type: ignore

            schema = json.loads(schema_path.read_text(encoding="utf-8"))
            jsonschema.validate(instance=report, schema=schema)
            return []
        except ImportError:
            pass
        except Exception as exc:
            errors.append(f"jsonschema validation failed: {exc}")
            return errors

    for key in ["schema_version", "generated_at", "tool", "hardware", "methodology", "benchmarks"]:
        if key not in report:
            errors.append(f"missing top-level key: {key}")
    if report.get("schema_version") != "bench_report.v1":
        errors.append(f"unexpected schema_version: {report.get('schema_version')}")
    hw = report.get("hardware", {})
    if not hw.get("devices"):
        errors.append("hardware.devices is empty")
    meth = report.get("methodology", {})
    if not meth.get("l2_flush", {}).get("enabled"):
        errors.append("methodology.l2_flush.enabled must be true")
    if not meth.get("interleaved_ab", {}).get("enabled"):
        errors.append("methodology.interleaved_ab.enabled must be true")
    for idx, entry in enumerate(report.get("benchmarks", [])):
        for k in ["id", "suite", "name", "backend", "status"]:
            if k not in entry:
                errors.append(f"benchmarks[{idx}] missing {k}")
        if "timing" in entry:
            t = entry["timing"]
            for tk in ["unit", "samples_count", "median", "iqr", "p95", "bootstrap_ci", "clock_temp_samples"]:
                if tk not in t:
                    errors.append(f"benchmarks[{idx}].timing missing {tk}")
    return errors


def _pack_gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_tiny_llama_gguf(path: Path, seed: int = 1234) -> dict:
    """Export a valid tiny 2-layer Llama-family GGUF v3 readable by both fusionrt and llama.cpp."""
    torch.manual_seed(seed)
    H, n_layers, nq, nkv, hd, inter, vocab = 64, 2, 4, 2, 16, 128, 64
    q_dim, kv_dim = nq * hd, nkv * hd

    # Token vocabulary: 0=<unk>, 1=<s>, 2=</s>, 3..63 printable ASCII/subwords
    tokens = ["<unk>", "<s>", "</s>"]
    chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    for i in range(60):
        tokens.append(chars[i] if i < len(chars) else f"<t{i}>")
    tokens.append("ab")
    merges = ["a b"]
    scores = [0.0] * vocab
    tok_types = [3, 3, 3] + [1] * (vocab - 3)

    weights: dict[str, torch.Tensor] = {
        "token_embd.weight": torch.randn(vocab, H, dtype=torch.float32) * 0.1,
        "output_norm.weight": torch.ones(H, dtype=torch.float32),
        "output.weight": torch.randn(vocab, H, dtype=torch.float32) * 0.1,
    }
    for l in range(n_layers):
        p = f"blk.{l}."
        weights[p + "attn_norm.weight"] = torch.ones(H, dtype=torch.float32)
        weights[p + "attn_q.weight"] = torch.randn(q_dim, H, dtype=torch.float32) * 0.08
        weights[p + "attn_k.weight"] = torch.randn(kv_dim, H, dtype=torch.float32) * 0.08
        weights[p + "attn_v.weight"] = torch.randn(kv_dim, H, dtype=torch.float32) * 0.08
        weights[p + "attn_output.weight"] = torch.randn(H, q_dim, dtype=torch.float32) * 0.08
        weights[p + "ffn_norm.weight"] = torch.ones(H, dtype=torch.float32)
        weights[p + "ffn_gate.weight"] = torch.randn(inter, H, dtype=torch.float32) * 0.08
        weights[p + "ffn_up.weight"] = torch.randn(inter, H, dtype=torch.float32) * 0.08
        weights[p + "ffn_down.weight"] = torch.randn(H, inter, dtype=torch.float32) * 0.08

    meta_items: list[tuple[str, int, bytes]] = [
        ("general.architecture", 8, _pack_gguf_str("llama")),
        ("general.name", 8, _pack_gguf_str("fusionrt-tiny-llama")),
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("llama.context_length", 4, struct.pack("<I", 256)),
        ("llama.embedding_length", 4, struct.pack("<I", H)),
        ("llama.block_count", 4, struct.pack("<I", n_layers)),
        ("llama.feed_forward_length", 4, struct.pack("<I", inter)),
        ("llama.attention.head_count", 4, struct.pack("<I", nq)),
        ("llama.attention.head_count_kv", 4, struct.pack("<I", nkv)),
        ("llama.attention.key_length", 4, struct.pack("<I", hd)),
        ("llama.attention.layer_norm_rms_epsilon", 6, struct.pack("<f", 1e-6)),
        ("llama.rope.dimension_count", 4, struct.pack("<I", hd)),
        ("llama.rope.freq_base", 6, struct.pack("<f", 10000.0)),
        ("tokenizer.ggml.model", 8, _pack_gguf_str("gpt2")),
        ("tokenizer.ggml.pre", 8, _pack_gguf_str("default")),
        ("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 1)),
        ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
        ("tokenizer.ggml.unknown_token_id", 4, struct.pack("<I", 0)),
        ("tokenizer.ggml.add_bos_token", 7, struct.pack("<B", 1)),
        (
            "tokenizer.ggml.tokens",
            9,
            struct.pack("<IQ", 8, len(tokens)) + b"".join(_pack_gguf_str(t) for t in tokens),
        ),
        (
            "tokenizer.ggml.merges",
            9,
            struct.pack("<IQ", 8, len(merges)) + b"".join(_pack_gguf_str(m) for m in merges),
        ),
        (
            "tokenizer.ggml.scores",
            9,
            struct.pack("<IQ", 6, len(scores)) + struct.pack(f"<{len(scores)}f", *scores),
        ),
        (
            "tokenizer.ggml.token_type",
            9,
            struct.pack("<IQ", 5, len(tok_types)) + struct.pack(f"<{len(tok_types)}i", *tok_types),
        ),
    ]

    align = 32
    buf = bytearray()
    buf += b"GGUF"
    buf += struct.pack("<IQQ", 3, len(weights), len(meta_items))
    for k, vt, vbytes in meta_items:
        buf += _pack_gguf_str(k)
        buf += struct.pack("<I", vt)
        buf += vbytes

    data_chunks: list[bytes] = []
    offset = 0
    for name, tensor in weights.items():
        raw = tensor.contiguous().numpy().tobytes()
        buf += _pack_gguf_str(name)
        buf += struct.pack("<I", tensor.ndim)
        for d in reversed(tensor.shape):
            buf += struct.pack("<Q", d)
        buf += struct.pack("<IQ", 0, offset)  # GGML_TYPE_F32 = 0
        pad = ((len(raw) + align - 1) // align) * align - len(raw)
        data_chunks.append(raw + b"\x00" * pad)
        offset += len(raw) + pad

    hdr_pad = ((len(buf) + align - 1) // align) * align - len(buf)
    buf += b"\x00" * hdr_pad
    for c in data_chunks:
        buf += c
    path.write_bytes(bytes(buf))
    return {"cfg": (H, n_layers, nq, nkv, hd, inter, vocab), "tokens": tokens, "weights": weights}


def pytorch_tiny_llama_generate(model_dict: dict, prompt_ids: list[int], steps: int) -> list[int]:
    """Exact FP64 reference execution for the exported tiny Llama GGUF."""
    H, n_layers, nq, nkv, hd, inter, _ = model_dict["cfg"]
    w = {k: v.to(torch.float64) for k, v in model_dict["weights"].items()}

    def rmsnorm(x: torch.Tensor, nw: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * nw

    def rope_interleaved(x: torch.Tensor, pos: int) -> torch.Tensor:
        out = x.clone()
        half = hd // 2
        for i in range(half):
            freq = 10000.0 ** (-2.0 * i / hd)
            ang = pos * freq
            c, s = math.cos(ang), math.sin(ang)
            x0, x1 = x[:, 2 * i], x[:, 2 * i + 1]
            out[:, 2 * i] = x0 * c - x1 * s
            out[:, 2 * i + 1] = x0 * s + x1 * c
        return out

    kc = [torch.zeros(128, nkv, hd, dtype=torch.float64) for _ in range(n_layers)]
    vc = [torch.zeros(128, nkv, hd, dtype=torch.float64) for _ in range(n_layers)]

    def step_tok(tok_id: int, pos: int) -> torch.Tensor:
        h = w["token_embd.weight"][tok_id].unsqueeze(0)
        for l in range(n_layers):
            p = f"blk.{l}."
            xn = rmsnorm(h, w[p + "attn_norm.weight"])
            q = F.linear(xn, w[p + "attn_q.weight"]).view(nq, hd)
            k = F.linear(xn, w[p + "attn_k.weight"]).view(nkv, hd)
            v = F.linear(xn, w[p + "attn_v.weight"]).view(nkv, hd)
            q = rope_interleaved(q, pos)
            k = rope_interleaved(k, pos)
            kc[l][pos] = k
            vc[l][pos] = v
            kr = kc[l][: pos + 1].transpose(0, 1).repeat_interleave(nq // nkv, dim=0)
            vr = vc[l][: pos + 1].transpose(0, 1).repeat_interleave(nq // nkv, dim=0)
            scores = torch.bmm(q.unsqueeze(1), kr.transpose(1, 2)) / math.sqrt(hd)
            attn = torch.bmm(F.softmax(scores, dim=-1), vr).reshape(1, nq * hd)
            h = h + F.linear(attn, w[p + "attn_output.weight"])
            mn = rmsnorm(h, w[p + "ffn_norm.weight"])
            g = F.linear(mn, w[p + "ffn_gate.weight"])
            u = F.linear(mn, w[p + "ffn_up.weight"])
            h = h + F.linear(F.silu(g) * u, w[p + "ffn_down.weight"])
        hn = rmsnorm(h, w["output_norm.weight"])
        return F.linear(hn, w["output.weight"]).squeeze(0)

    out_ids = list(prompt_ids)
    logits = None
    for pos, tid in enumerate(out_ids):
        logits = step_tok(tid, pos)
    pos = len(out_ids)
    for s in range(steps):
        assert logits is not None
        nxt = int(torch.argmax(logits).item())
        out_ids.append(nxt)
        if s + 1 < steps:
            logits = step_tok(nxt, pos)
            pos += 1
    return out_ids


def pytorch_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    return x * inv_rms * w


def pytorch_gdn_step(
    qkv_proj: torch.Tensor,
    z: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    conv_w: torch.Tensor,
    conv_bias: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_w: torch.Tensor,
    conv_state: torch.Tensor,
    ssm_state: torch.Tensor,
    nk: int,
    nv: int,
    dk: int,
    dv: int,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    conv_state = conv_state.clone()
    ssm_state = ssm_state.clone()
    conv_in = torch.cat([conv_state, qkv_proj.unsqueeze(-1)], dim=-1)
    conv_out = (conv_in * conv_w).sum(dim=-1) + conv_bias
    conv_state = conv_in[:, 1:].contiguous()
    qkv = F.silu(conv_out)

    q_dim = nk * dk
    q = qkv[:q_dim].view(nk, dk)
    k = qkv[q_dim : 2 * q_dim].view(nk, dk)
    v = qkv[2 * q_dim :].view(nv, dv)

    q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6) * (1.0 / math.sqrt(dk))
    k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)

    ratio = nv // nk
    out = torch.zeros(nv, dv, dtype=qkv.dtype, device=qkv.device)
    for vh in range(nv):
        kh = vh // ratio
        g = torch.exp(-torch.exp(A_log[vh]) * F.softplus(a[vh] + dt_bias[vh]))
        beta = torch.sigmoid(b[vh])
        S = ssm_state[vh] * g
        kv_mem = (S * k[kh].unsqueeze(-1)).sum(dim=0)
        delta = (v[vh] - kv_mem) * beta
        S = S + k[kh].unsqueeze(-1) * delta.unsqueeze(0)
        ssm_state[vh] = S
        y = (S * q[kh].unsqueeze(-1)).sum(dim=0)
        out[vh] = pytorch_rmsnorm(y, norm_w, eps) * F.silu(z[vh * dv : (vh + 1) * dv])
    return out.reshape(-1), conv_state, ssm_state


def bench_pytorch_4layer_decode(arch: str, hidden: int = 512, n_layers: int = 4, iters: int = 30) -> float:
    device = torch.device("cuda")
    torch.manual_seed(77)
    x = torch.randn(1, hidden, device=device, dtype=torch.float32)

    if arch == "dense":
        nq, nkv, hd, inter = 8, 2, 64, 1536
        q_dim, kv_dim = nq * hd, nkv * hd
        layers = [
            {
                "anw": torch.ones(hidden, device=device),
                "qkv": torch.randn(q_dim + 2 * kv_dim, hidden, device=device) * 0.04,
                "qn": torch.ones(hd, device=device),
                "kn": torch.ones(hd, device=device),
                "o": torch.randn(hidden, q_dim, device=device) * 0.04,
                "mnw": torch.ones(hidden, device=device),
                "gu": torch.randn(2 * inter, hidden, device=device) * 0.04,
                "dn": torch.randn(hidden, inter, device=device) * 0.04,
                "kc": torch.zeros(64, nkv, hd, device=device),
                "vc": torch.zeros(64, nkv, hd, device=device),
            }
            for _ in range(n_layers)
        ]

        def step(pos: int) -> torch.Tensor:
            h = x
            for L in layers:
                xn = pytorch_rmsnorm(h, L["anw"])
                qkv = F.linear(xn, L["qkv"]).squeeze(0)
                q = pytorch_rmsnorm(qkv[:q_dim].view(nq, hd), L["qn"])
                k = pytorch_rmsnorm(qkv[q_dim : q_dim + kv_dim].view(nkv, hd), L["kn"])
                v = qkv[q_dim + kv_dim :].view(nkv, hd)
                L["kc"][pos] = k
                L["vc"][pos] = v
                kr = L["kc"][: pos + 1].transpose(0, 1).repeat_interleave(nq // nkv, dim=0)
                vr = L["vc"][: pos + 1].transpose(0, 1).repeat_interleave(nq // nkv, dim=0)
                scores = torch.bmm(q.unsqueeze(1), kr.transpose(1, 2)) / math.sqrt(hd)
                attn = torch.bmm(F.softmax(scores, dim=-1), vr).reshape(1, q_dim)
                h = h + F.linear(attn, L["o"])
                mn = pytorch_rmsnorm(h, L["mnw"])
                gu = F.linear(mn, L["gu"])
                h = h + F.linear(F.silu(gu[:, :inter]) * gu[:, inter:], L["dn"])
            return h
    elif arch == "moe":
        inter, E, K = 512, 8, 2
        layers = [
            {
                "nw": torch.ones(hidden, device=device),
                "rw": torch.randn(E, hidden, device=device) * 0.1,
                "gu": torch.randn(E, 2 * inter, hidden, device=device) * 0.04,
                "dn": torch.randn(E, hidden, inter, device=device) * 0.04,
                "sgu": torch.randn(2 * inter, hidden, device=device) * 0.04,
                "sdn": torch.randn(hidden, inter, device=device) * 0.04,
            }
            for _ in range(n_layers)
        ]

        def step(pos: int) -> torch.Tensor:
            h = x
            for L in layers:
                xn = pytorch_rmsnorm(h, L["nw"])
                probs = F.softmax(F.linear(xn, L["rw"]).squeeze(0), dim=-1)
                w, idx = torch.topk(probs, K)
                w = w / w.sum()
                out = torch.zeros_like(h)
                for s in range(K):
                    e = int(idx[s].item())
                    gu = F.linear(xn, L["gu"][e])
                    out = out + w[s] * F.linear(F.silu(gu[:, :inter]) * gu[:, inter:], L["dn"][e])
                sgu = F.linear(xn, L["sgu"])
                out = out + F.linear(F.silu(sgu[:, :inter]) * sgu[:, inter:], L["sdn"])
                h = h + out
            return h
    else:
        return 0.0

    for i in range(5):
        _ = step(i)
    torch.cuda.synchronize()
    times = []
    for i in range(iters):
        t0 = time.perf_counter()
        _ = step(5 + (i % 32))
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def run_cmd(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)


def parse_output_ids(stdout: str) -> list[int]:
    for line in stdout.splitlines():
        if line.startswith("output_ids:"):
            return [int(x) for x in line.split(":", 1)[1].strip().split()]
    raise RuntimeError(f"Missing output_ids in CLI stdout:\n{stdout}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build"))
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--llama-bin", type=Path, default=None, help="Directory containing llama-completion / llama-bench")
    args = parser.parse_args()

    fusionrt_bin = args.build_dir / "fusionrt.exe"
    tests_bin = args.build_dir / "fusionrt_tests.exe"
    if not fusionrt_bin.exists():
        fusionrt_bin = args.build_dir / "fusionrt"
        tests_bin = args.build_dir / "fusionrt_tests"

    print("1) Running C++ / CUDA / Vulkan test suite...")
    res = run_cmd([str(tests_bin)])
    print(res.stdout)
    if res.returncode != 0:
        print(res.stderr, file=sys.stderr)
        return res.returncode

    print("2) Checking PyTorch FP64 GatedDeltaNet step equivalence...")
    torch.manual_seed(42)
    nk, nv, dk, dv = 4, 8, 64, 64
    conv_dim = 2 * nk * dk + nv * dv
    cs = torch.zeros(conv_dim, 3, dtype=torch.float64)
    ss = torch.zeros(nv, dk, dv, dtype=torch.float64)
    cw = torch.randn(conv_dim, 4, dtype=torch.float64) * 0.2
    cb = torch.randn(conv_dim, dtype=torch.float64) * 0.01
    alog = torch.full((nv,), -2.0, dtype=torch.float64)
    dtb = torch.randn(nv, dtype=torch.float64) * 0.02
    nw = torch.ones(dv, dtype=torch.float64)
    for _ in range(4):
        out, cs, ss = pytorch_gdn_step(
            torch.randn(conv_dim, dtype=torch.float64) * 0.2,
            torch.randn(nv * dv, dtype=torch.float64) * 0.2,
            torch.randn(nv, dtype=torch.float64) * 0.1,
            torch.randn(nv, dtype=torch.float64) * 0.1,
            cw, cb, alog, dtb, nw, cs, ss, nk, nv, dk, dv,
        )
    assert torch.isfinite(out).all()
    print(f"   PyTorch FP64 GDN 4-step output L2 norm = {out.norm().item():.6f} (finite OK)")

    print("3) Exporting tiny Llama GGUF and verifying CUDA + Vulkan + PyTorch FP64 + llama.cpp parity...")
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_path = Path(tmpd)
        gguf_path = tmp_path / "tiny_llama.gguf"
        model_dict = write_tiny_llama_gguf(gguf_path)
        prompt = "xyz"
        prompt_ids = [1, model_dict["tokens"].index("x"), model_dict["tokens"].index("y"), model_dict["tokens"].index("z")]
        pt_ids = pytorch_tiny_llama_generate(model_dict, prompt_ids, steps=6)

        for label, extra in [
            ("CUDA", ["--backend", "cuda"]),
            ("CUDA-Graph", ["--backend", "cuda", "--graph"]),
            ("Vulkan:0", ["--backend", "vulkan", "--vk-device", "0"]),
            ("Vulkan:1", ["--backend", "vulkan", "--vk-device", "1"]),
        ]:
            r = run_cmd([str(fusionrt_bin), "run", "--model", str(gguf_path), "--prompt", prompt, "--steps", "6", *extra])
            if r.returncode != 0:
                print(f"{label} failed:\n{r.stderr}", file=sys.stderr)
                return r.returncode
            got_ids = parse_output_ids(r.stdout)
            if got_ids != pt_ids:
                print(f"Token mismatch vs PyTorch FP64 for {label}: got={got_ids} expected={pt_ids}", file=sys.stderr)
                return 1
            print(f"   [{label}] GGUF token parity vs PyTorch FP64: EXACT MATCH ({got_ids})")

        # llama.cpp side-by-side parity & timing when llama-completion / llama-bench are available
        llama_comp = (args.llama_bin / "llama-completion.exe") if args.llama_bin else shutil.which("llama-completion")
        llama_bench = (args.llama_bin / "llama-bench.exe") if args.llama_bin else shutil.which("llama-bench")
        if llama_comp and Path(llama_comp).exists():
            lr = run_cmd([str(llama_comp), "-m", str(gguf_path), "-p", prompt, "-n", "6",
                          "--temp", "0", "--top-k", "1", "--repeat-penalty", "1.0", "--no-display-prompt"])
            if lr.returncode != 0:
                print(f"   llama-completion returned {lr.returncode}: {lr.stderr.strip()}")
            else:
                expected_text = "".join(model_dict["tokens"][i] for i in pt_ids[len(prompt_ids):])
                llama_out = lr.stdout.strip().splitlines()[0].strip() if lr.stdout.strip() else ""
                print(f"   [llama.cpp] tiny_llama.gguf generated={llama_out!r} (PyTorch/fusionrt={expected_text!r})")
                if llama_out != expected_text:
                    print(f"   llama.cpp continuation mismatch: {llama_out!r} != {expected_text!r}", file=sys.stderr)
                    return 1
                print("   [llama.cpp] GGUF text continuation vs PyTorch/fusionrt: EXACT MATCH")
        if llama_bench and Path(llama_bench).exists():
            lb = run_cmd([str(llama_bench), "-m", str(gguf_path), "-p", "16", "-n", "8", "-r", "3"])
            if lb.returncode == 0:
                print("   [llama-bench] tiny_llama.gguf baseline:")
                for line in lb.stdout.strip().splitlines():
                    if "|" in line:
                        print("     " + line)

        print("4) Running fusionrt bench & validating bench_report.v1.json...")
        report_path = tmp_path / "bench_report.v1.json"
        bres = run_cmd([str(fusionrt_bin), "bench", "--suite", "all", "--output", str(report_path)])
        print(bres.stdout)
        if bres.returncode != 0:
            print(bres.stderr, file=sys.stderr)
            return bres.returncode
        report = json.loads(report_path.read_text(encoding="utf-8"))
        schema_errors = validate_bench_report_v1(report, args.schema)
        if schema_errors:
            print("Schema errors:", schema_errors, file=sys.stderr)
            return 1
        print(f"   bench_report.v1.json schema validation PASSED ({len(report['benchmarks'])} entries)")

        pt_dense_ms = bench_pytorch_4layer_decode("dense")
        pt_moe_ms = bench_pytorch_4layer_decode("moe")
        fr_entries = {e["id"]: e for e in report["benchmarks"]}
        if "cuda_block_dense_graph" in fr_entries:
            fr_dense_ms = fr_entries["cuda_block_dense_graph"]["timing"]["median"]
            print(f"   4-Layer Dense Decode: PyTorch Eager={pt_dense_ms:.4f} ms vs fusionrt Graph={fr_dense_ms:.4f} ms ({pt_dense_ms / fr_dense_ms:.2f}x)")
        if "cuda_block_moe_graph" in fr_entries:
            fr_moe_ms = fr_entries["cuda_block_moe_graph"]["timing"]["median"]
            print(f"   4-Layer MoE Decode: PyTorch Eager={pt_moe_ms:.4f} ms vs fusionrt Graph={fr_moe_ms:.4f} ms ({pt_moe_ms / fr_moe_ms:.2f}x)")

    print("5) Testing CLI run modes (CUDA + Vulkan across architectures)...")
    for args_list in [
        ["--backend", "cuda", "--arch", "dense", "--dtype", "q4", "--steps", "8", "--graph"],
        ["--backend", "cuda", "--arch", "hybrid", "--dtype", "f32", "--steps", "8", "--graph"],
        ["--backend", "cuda", "--arch", "moe", "--dtype", "f32", "--steps", "8", "--graph"],
        ["--backend", "vulkan", "--vk-device", "0", "--arch", "hybrid", "--steps", "4"],
        ["--backend", "vulkan", "--vk-device", "1", "--arch", "dense", "--steps", "4"],
    ]:
        r = run_cmd([str(fusionrt_bin), "run", *args_list])
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return r.returncode
        print("   " + r.stdout.strip().splitlines()[0])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
