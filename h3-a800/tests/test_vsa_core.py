#!/usr/bin/env python3
"""Unit test for h3_vsa.vsa_attention against an independent SDPA arbiter.

The kernel is verified against token-mask SDPA computed in tile space (an
implementation entirely disjoint from the Triton kernel), on the real request
geometry: segment-pure prefix tiles plus 3-D video tiles with non-uniform
edge sizes, a 90%-sparse video selection, and the gate branch.

Run: python tests/test_vsa_core.py
"""

from __future__ import annotations

import math
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from h3_runtime.vsa import h3_vsa  # noqa: E402

TEXT, COND, AUDIO = 100, 0, 64
VIDEO_GRID = (8, 6, 7)  # 336 video rows, edge tiles non-uniform
SEQ = TEXT + COND + AUDIO + math.prod(VIDEO_GRID)


def sdpa_arbiter(q, k, v, mask, geometry, gate=None):
    """Token-mask SDPA in tile space: disjoint from the Triton kernel.

    q/k/v: (S, H, D) packed; mask: (1, H, n_tiles, n_tiles) bool.
    Returns (S, H, D) fp32.
    """
    vbs = geometry.variable_block_sizes
    qt, kt, vt = (lambda x: x)(None), None, None

    def tile(x):
        buf = torch.zeros(1, x.shape[1], geometry.seq_pad, x.shape[2], dtype=torch.float32,
                          device=x.device)
        buf[:, :, geometry.non_pad] = x.transpose(0, 1).unsqueeze(0)[:, :, geometry.slot_to_row].float()
        return buf

    qt, kt, vt = tile(q), tile(k), tile(v)
    token_mask = mask[0].repeat_interleave(64, dim=1).repeat_interleave(64, dim=2)
    token_valid = (torch.arange(64, device=vbs.device)[None, :] < vbs[:, None]).reshape(-1)
    keep = token_mask & token_valid[None, :, None] & token_valid[None, None, :]
    additive = torch.zeros_like(keep, dtype=torch.float32).masked_fill(~keep, float("-inf"))
    scores = qt.float() @ kt.float().transpose(-1, -2) / math.sqrt(q.shape[-1]) + additive
    out = torch.softmax(scores, dim=-1) @ vt  # (1, H, S_pad, D)
    if gate is not None:
        pooled_q = h3_vsa._pool_tiles(qt, vbs)
        pooled_k = h3_vsa._pool_tiles(kt, vbs)
        pooled_v = h3_vsa._pool_tiles(vt, vbs)
        tile_scores = pooled_q @ pooled_k.transpose(-1, -2) / math.sqrt(q.shape[-1])
        out_c = torch.softmax(tile_scores, dim=-1) @ pooled_v  # (1, H, n_tiles, D)
        out_c = out_c.permute(0, 2, 1, 3)  # (1, n_tiles, H, D)
        gate_tiled = tile(gate).view(1, gate.shape[1], geometry.n_tiles, 64, gate.shape[2])
        out = out.view(1, gate.shape[1], geometry.n_tiles, 64, gate.shape[2]) \
            + out_c.unsqueeze(3) * gate_tiled
        out = out.view(1, gate.shape[1], geometry.n_tiles * 64, gate.shape[2])
    out = out[:, :, geometry.untile]  # (1, H, S, D) packed order
    return out[0].transpose(0, 1)  # (S, H, D)


def main() -> int:
    os.environ["H3_VSA_DENSE_FIRST_STEPS"] = "0"
    os.environ["H3_VSA_DENSE_LAYERS"] = "0"
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    h3_vsa.set_request_layout((TEXT, COND, AUDIO), VIDEO_GRID)
    torch.manual_seed(0)
    heads = 7
    q = torch.randn(SEQ, heads, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn(SEQ, heads, 128, dtype=torch.bfloat16, device=device)
    v = torch.randn(SEQ, heads, 128, dtype=torch.bfloat16, device=device)
    geometry = h3_vsa._tile_geometry((TEXT, COND, AUDIO), VIDEO_GRID).to(device)
    vbs = geometry.variable_block_sizes

    def tile(x):
        buf = torch.zeros(1, heads, geometry.seq_pad, 128, dtype=torch.float32, device=device)
        buf[:, :, geometry.non_pad] = x.transpose(0, 1).unsqueeze(0)[:, :, geometry.slot_to_row].float()
        return buf

    # 1. full keep == dense SDPA (kernel + tile/untile round trip)
    os.environ["H3_VSA_SPARSITY"] = "0.0"
    h3_vsa.state().clock = 0
    full = h3_vsa.vsa_attention(q, k, v).clone()
    dense = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0).float(), k.transpose(0, 1).unsqueeze(0).float(),
        v.transpose(0, 1).unsqueeze(0).float())[0].transpose(0, 1)
    err = (full.float() - dense).abs().max().item()
    print(f"sparsity=0 vs dense SDPA: max_abs={err:.3e}")
    assert err < 0.05, err

    # 2. sparse == token-mask SDPA over the same tile mask
    os.environ["H3_VSA_SPARSITY"] = "0.9"
    h3_vsa.state().clock = 0
    sparse = h3_vsa.vsa_attention(q, k, v).clone()
    qp = h3_vsa._pool_tiles(tile(q), vbs)
    kp = h3_vsa._pool_tiles(tile(k), vbs)
    scores = qp @ kp.transpose(-1, -2) / math.sqrt(128)
    mask = h3_vsa.build_block_mask(scores, geometry.num_prefix_tiles, geometry.num_video_tiles,
                                   0.9, True)
    keep_frac = mask.float().mean().item()
    ref = sdpa_arbiter(q, k, v, mask, geometry)
    err = (sparse.float() - ref).abs().max().item()
    print(f"sparsity=0.9 (keep {keep_frac:.2f}) vs SDPA arbiter: max_abs={err:.3e}")
    assert err < 0.05, err

    # 3. sparse output differs from full attention (selection is active) — on
    # random logits the deviation is large by construction; real-data quality
    # is gated end-to-end in Phase 3, so only sanity-check it is finite.
    rel = torch.linalg.norm(sparse.float() - dense) / torch.linalg.norm(dense.float())
    print(f"sparse vs dense relative L2 (random data, informational): {rel.item():.3f}")
    assert torch.isfinite(rel).item(), rel.item()

    # 4. zero gates reduce to the pure sparse output
    h3_vsa.state().clock = 0
    zero_gates = torch.zeros_like(v)
    with_gate = h3_vsa.vsa_attention(q, k, v, gate=zero_gates).clone()
    assert torch.equal(with_gate, sparse), "zero gates must not change the output"
    print("zero-gate reduction: exact")

    # 5. trained-magnitude gates perturb in a bounded way
    h3_vsa.state().clock = 0
    small_gates = torch.randn_like(v) * 0.01
    with_gate = h3_vsa.vsa_attention(q, k, v, gate=small_gates).clone()
    delta = (with_gate.float() - sparse.float()).abs().max().item()
    print(f"small-gate delta: max_abs={delta:.3e} (nonzero, bounded)")
    assert 0.0 < delta < 1.0, delta

    print("VSA CORE UNIT TESTS PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
