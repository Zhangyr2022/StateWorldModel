#!/usr/bin/env python3
"""Phase 0 go/no-go: the Triton Sol-Attn kernel on SM80 (run `bash fetch_sol_attn.sh` first).

On SM80 the sol_attn package has no specialized CuTe kernel and no
CuTe runtime, so `get_sol_attn_backend` falls back to triton_ref — the exact
kernel/policy pairing upstream validated on 4x A100 (tau=1.0, exact threshold,
full-prefix KV sink, dense prefix queries). This probe answers the DESIGN §7.1
questions in order:

1. does the dispatch reach the Triton kernel on this GPU,
2. kernel numerics at full route (tau=-1000) vs the bf16 SDPA reference,
   against the same limits `H3SparseAttention._run_gate` enforces in production,
3. sink semantics: prefix query rows must come back exact,
4. route density rho at tau=1.0 for the real post-Ulysses shapes —
   rho > ~60% kills the direction (Triton-on-SM80 runs at ~50-70% of cuDNN
   flash efficiency, so the routed savings must beat that deficit),
5. throughput vs dense SDPA at those shapes.

Run: python tests/test_sol_triton_sm80.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "h3_runtime"))
sys.path.insert(0, str(ROOT / "h3_runtime" / "third_party"))

from sol_attn import get_sol_attn_backend, sol_attn  # noqa: E402

from h3_runtime.sparse_attention import _dense, _estimate_density  # noqa: E402

SINK_TOKENS = 951  # 537 text + 414 audio prefix rows of a 768p request


def report(name: str, got: torch.Tensor, want: torch.Tensor) -> dict:
    diff = (got.float() - want.float()).abs()
    stats = {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "rel_l2": float((torch.linalg.vector_norm(got.float() - want.float())
                         / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)).item()),
    }
    print(f"  {name:28s} max_abs={stats['max_abs']:.4f}  "
          f"mean_abs={stats['mean_abs']:.5f}  rel_l2={stats['rel_l2']:.5f}")
    return stats


def bench(fn, iters: int = 20, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> int:
    assert torch.cuda.is_available(), "needs a CUDA GPU"
    device = torch.device("cuda")
    torch.manual_seed(0)

    backend = get_sol_attn_backend(device)
    print(f"[sol-triton] dispatch: SM{torch.cuda.get_device_capability(device)[0]}"
          f"{torch.cuda.get_device_capability(device)[1]} -> backend {backend!r}")
    if backend != "triton":
        print("[sol-triton] FAIL: expected the triton fallback on SM80")
        return 1

    failures = []

    # -- 2. kernel numerics at full route (the production gate's contract) ----
    print(f"[sol-triton] kernel numerics, tau=-1000 (all blocks routed), "
          f"S=8192 H=7 (sink off, as in H3SparseAttention._run_gate)")
    q = torch.randn(1, 8192, 7, 128, device=device).to(torch.bfloat16)
    k = torch.randn(1, 8192, 7, 128, device=device).to(torch.bfloat16)
    v = torch.randn(1, 8192, 7, 128, device=device).to(torch.bfloat16)
    got = sol_attn(q, k, v, tau=-1000.0, thresh_type="diag",
                   sink_start=None, sink_tokens=0)
    want = _dense(q[0], k[0], v[0]).unsqueeze(0)
    stats = report("full-route vs SDPA", got, want)
    limits = {"max_abs": 0.08, "mean_abs": 0.002, "rel_l2": 0.005}
    if any(stats[name] > limit for name, limit in limits.items()):
        failures.append(f"full-route numerics outside production gate limits {limits}")

    # -- 3. sink semantics: prefix query rows are recomputed densely by the
    #       integration (sparse_attention.py), so the kernel's own prefix-row
    #       drift under the sink is informational, not a gate.
    print(f"[sol-triton] sink semantics, tau=1.0 routed, sink=[0, {SINK_TOKENS})")
    q = torch.randn(1, 16384, 7, 128, device=device).to(torch.bfloat16)
    k = torch.randn(1, 16384, 7, 128, device=device).to(torch.bfloat16)
    v = torch.randn(1, 16384, 7, 128, device=device).to(torch.bfloat16)
    got = sol_attn(q, k, v, tau=1.0, thresh_type="diag",
                   sink_start=0, sink_tokens=SINK_TOKENS)
    want = _dense(q[0], k[0], v[0]).unsqueeze(0)
    report("prefix rows [0, 951)", got[:, :SINK_TOKENS], want[:, :SINK_TOKENS])
    report("video rows (approximate)", got[:, SINK_TOKENS:], want[:, SINK_TOKENS:])
    del q, k, v, got, want

    # -- 4 + 5. route density and throughput at the real shapes ---------------
    # (rows, local_heads, label): post-Ulysses packed rows. The transformer
    # patchifies the latent grid 2x2, so a 768p frame carries 42*24=1008 video
    # rows and a 480p frame 27*15=405; prefix is 951 rows (text+audio).
    shapes = [
        (32256, 7, "768p@5s, 8 ranks"),
        (92672, 7, "768p@15s, 8 ranks"),
        (38272, 14, "480p@15s, 4 ranks"),
    ]
    for rows, heads, label in shapes:
        print(f"[sol-triton] shape {label}: S={rows} H={heads}")
        q = torch.randn(1, rows, heads, 128, device=device).to(torch.bfloat16)
        k = torch.randn(1, rows, heads, 128, device=device).to(torch.bfloat16)
        v = torch.randn(1, rows, heads, 128, device=device).to(torch.bfloat16)

        density = _estimate_density(
            q, k, v, tau=1.0, thresh_type="diag",
            sink_start=0, sink_tokens=SINK_TOKENS, compile_bucket_size=0)
        print(f"  {'route density':28s} threshold={density['threshold_density']}  "
              f"effective={density['effective_density']}  (go/no-go: < 0.60)")
        if float(density["effective_density"]) > 0.60:
            failures.append(f"route density {density['effective_density']} > 0.60 at {label}")

        t_jit = time.perf_counter()
        _ = sol_attn(q, k, v, tau=1.0, thresh_type="diag",
                     sink_start=0, sink_tokens=SINK_TOKENS)
        torch.cuda.synchronize()
        print(f"  {'first call (JIT+autotune)':28s} {time.perf_counter() - t_jit:7.2f} s")

        t_dense = bench(lambda: _dense(q[0], k[0], v[0]))
        t_routed = bench(lambda: sol_attn(q, k, v, tau=1.0, thresh_type="diag",
                                          sink_start=0, sink_tokens=SINK_TOKENS))
        t_full = bench(lambda: sol_attn(q, k, v, tau=-1000.0, thresh_type="diag",
                                        sink_start=None, sink_tokens=0))
        print(f"  {'dense SDPA':28s} {t_dense * 1e3:9.2f} ms")
        print(f"  {'sol tau=1.0 (routed)':28s} {t_routed * 1e3:9.2f} ms   "
              f"ratio {t_routed / t_dense:5.2f}x")
        print(f"  {'sol tau=-1000 (100%)':28s} {t_full * 1e3:9.2f} ms   "
              f"ratio {t_full / t_dense:5.2f}x")
        del q, k, v
        torch.cuda.empty_cache()

    print()
    if failures:
        for failure in failures:
            print(f"[sol-triton] FAIL: {failure}")
        return 1
    print("[sol-triton] all gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
