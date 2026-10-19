#!/usr/bin/env python3
"""Phase 0 go/no-go: FastVideo's Triton VSA kernel on SM80.

Correctness against masked-dense SDPA (the sparse kernel renormalizes the
softmax over selected blocks only, so an additive -inf mask on dense SDPA is
the exact reference), then a throughput sweep at the post-Ulysses shapes of a
768p request on 8 ranks (B=1, 7 local heads, ~32.8k rows, head_dim 128).

Run: python tests/test_vsa_kernel_sm80.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import torch
import triton

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "h3_runtime"))

from h3_runtime.vsa.block_sparse_attn_triton import (  # noqa: E402
    triton_block_sparse_attn_forward,
)

BLOCK = 64


def random_topology(n_q_blocks: int, n_kv_blocks: int, topk: int, heads: int,
                    device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per (query block, head): own diagonal block + topk random others, sorted."""
    topk = min(topk, n_kv_blocks)
    rand = torch.rand(heads, n_q_blocks, n_kv_blocks, device=device).argsort(dim=-1)
    rows = []
    for h in range(heads):
        per_head = []
        for qb in range(n_q_blocks):
            chosen = set(rand[h, qb, :topk].tolist())
            chosen.add(min(qb, n_kv_blocks - 1))
            per_head.append(sorted(chosen))
        rows.append(per_head)
    max_blks = max(len(s) for per_head in rows for s in per_head)
    index = torch.full((heads, n_q_blocks, max_blks), -1, dtype=torch.int32)
    counts = torch.empty((heads, n_q_blocks), dtype=torch.int32)
    for h in range(heads):
        for qb, chosen in enumerate(rows[h]):
            counts[h, qb] = len(chosen)
            index[h, qb, : len(chosen)] = torch.tensor(chosen, dtype=torch.int32)
    return index.unsqueeze(0).to(device), counts.unsqueeze(0).to(device)


def reference_sdpa_masked(q, k, v, q2k_index, q2k_num) -> torch.Tensor:
    """Dense SDPA with -inf on unselected blocks: exact same math as the sparse kernel."""
    B, H, Tq, D = q.shape
    Tkv = k.shape[2]
    mask = torch.full((H, Tq, Tkv), float("-inf"), dtype=torch.float32, device=q.device)
    n_q_blocks = Tq // BLOCK
    index = q2k_index[0].long()
    counts = q2k_num[0].long()
    for h in range(H):
        for qb in range(n_q_blocks):
            rows = slice(qb * BLOCK, qb * BLOCK + BLOCK)
            for j in range(counts[h, qb].item()):
                blk = index[h, qb, j].item()
                if blk < 0:
                    continue
                cols = slice(blk * BLOCK, blk * BLOCK + BLOCK)
                mask[h, rows, cols] = 0.0
    scores = (q[0].float() @ k[0].float().transpose(-1, -2)) / math.sqrt(D) + mask
    probs = torch.softmax(scores, dim=-1)
    del scores, mask
    out = probs @ v[0].float()
    return out.unsqueeze(0).to(q.dtype)


def test_numerics(device) -> None:
    torch.manual_seed(0)
    B, H, T, D = 1, 2, 2048, 128
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    var_blocks = torch.full((T // BLOCK,), BLOCK, dtype=torch.int32, device=device)
    q2k_index, q2k_num = random_topology(T // BLOCK, T // BLOCK, topk=6, heads=H, device=device)

    out, _ = triton_block_sparse_attn_forward(q, k, v, q2k_index, q2k_num, var_blocks)
    ref = reference_sdpa_masked(q, k, v, q2k_index, q2k_num)
    error = (out.float() - ref.float()).abs().max().item()
    scale = ref.float().abs().max().item()
    print(f"numerics: max_abs={error:.3e} relative={error / scale:.2e} "
          f"(shape {tuple(out.shape)})")
    assert error / scale < 5e-2, (error, scale)


def benchmark(device) -> None:
    """Post-Ulysses shape of a 768p request on 8 ranks vs dense SDPA."""
    B, H, T, D = 1, 7, 32768, 128
    q = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    k = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    v = torch.randn(B, H, T, D, dtype=torch.bfloat16, device=device)
    var_blocks = torch.full((T // BLOCK,), BLOCK, dtype=torch.int32, device=device)

    def timeit(fn, repeat=20):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - started) / repeat * 1000

    dense_ms = timeit(lambda: torch.nn.functional.scaled_dot_product_attention(q, k, v))
    q_flops = 2 * B * H * T * D * T
    dense_tflops = q_flops / (dense_ms / 1e3) / 1e12
    print(f"dense SDPA   {dense_ms:8.2f} ms  ({dense_tflops:6.1f} TFLOPS-equivalent)")

    for frac in (0.05, 0.125, 0.25, 0.5):
        topk = max(2, int(T // BLOCK * frac))
        q2k_index, q2k_num = random_topology(T // BLOCK, T // BLOCK, topk=topk, heads=H, device=device)
        ms = timeit(lambda: triton_block_sparse_attn_forward(q, k, v, q2k_index, q2k_num, var_blocks))
        # The topology build above happens on-GPU too; time it separately for honesty.
        topology_ms = timeit(lambda: random_topology(T // BLOCK, T // BLOCK, topk=topk, heads=H, device=device),
                             repeat=5)
        print(f"VSA topk={topk:4d} ({frac * 100:3.0f}% of KV)  {ms:8.2f} ms  "
              f"speedup vs dense x{dense_ms / ms:5.2f}   (topology build {topology_ms:.1f} ms)")


def main() -> int:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    name = torch.cuda.get_device_name(device)
    major, minor = torch.cuda.get_device_capability(device)
    print(f"device: {name} (SM{major}{minor}) | triton {triton.__version__}")
    test_numerics(device)
    benchmark(device)
    print("PHASE 0 KERNEL CHECK COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
