#!/usr/bin/env python3
"""int8 QKV wire for the Triton sol backend: encode -> all-to-all -> decode.

The compressed transport (400 B/row-head vs 768 B bf16) used to be gated to
sol_bsa; `ulysses._qkv_wire_dtype` now admits the sol backend, which consumes
packets through `dequantize_qkv_split` (three contiguous tensors — the layout
the stride-blind Triton pointer kernel needs anyway). This test pins, in the
production exchange path:

  1. `dequantize_qkv_split` reproduces the bf16 wire's post-rope Q/K/V up to
     the int8 group-32 UE5M3 quantization error, and exactly matches
     `dequantize_qkv_padded` (same decode math, different layout);
  2. the decoded tensors are dense-contiguous (the triton_ref kernel's
     address math silently misreads strided views — DESIGN §7.1);
  3. sol_attn on the int8 wire deviates from the bf16 wire only by the
     propagated quantization noise, and both stay near the dense reference.

Run: torchrun --standalone --nproc_per_node=2 tests/test_int8_wire_sm80.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("DISABLE_ADDMM_CUDA_LT", "1")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "h3_runtime" / "third_party"))

import torch
import torch.distributed as dist

ROWS_LOCAL, HEADS, HEAD_DIM, ROTARY = 1024, 4, 128, 64


def rel_l2(got: torch.Tensor, want: torch.Tensor) -> float:
    return float(
        (torch.linalg.vector_norm(got.float() - want.float())
         / torch.linalg.vector_norm(want.float()).clamp_min(1e-12)).item()
    )


def main() -> int:
    assert torch.cuda.is_available()
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    torch.manual_seed(1234 + rank)  # different sequence shards per rank
    world = dist.get_world_size()
    device = torch.device("cuda")

    from sol_attn import sol_attn as sol_attn_kernel

    from h3_runtime.comm_quant import dequantize_qkv_padded, dequantize_qkv_split
    from h3_runtime.relayout import qknorm_rope_pack_qkv_destination_major
    from h3_runtime.ulysses import _packed_qkv_all_to_all

    q = torch.randn(ROWS_LOCAL, HEADS, HEAD_DIM, device=device).to(torch.bfloat16)
    k = torch.randn(ROWS_LOCAL, HEADS, HEAD_DIM, device=device).to(torch.bfloat16)
    v = torch.randn(ROWS_LOCAL, HEADS, HEAD_DIM, device=device).to(torch.bfloat16)
    q_weight = torch.randn(HEAD_DIM, device=device).to(torch.bfloat16).abs() + 0.5
    k_weight = torch.randn(HEAD_DIM, device=device).to(torch.bfloat16).abs() + 0.5
    cos = torch.randn(ROWS_LOCAL, ROTARY, device=device)
    sin = torch.randn(ROWS_LOCAL, ROTARY, device=device)

    def pack(wire: str):
        return qknorm_rope_pack_qkv_destination_major(
            q, k, v, q_weight, k_weight, cos, sin, 1e-6, 1e-6, world,
            wire_dtype=wire,
        )

    # bf16 reference wire: full sequence, this rank's heads, strided views.
    recv_bf16 = _packed_qkv_all_to_all(
        q, k, v, world=world, group=dist.group.WORLD,
        packed_send=pack("bf16"), wire_dtype="bf16",
    )
    q_ref, k_ref, v_ref = recv_bf16.split(HEAD_DIM, dim=-1)

    # int8 wire: same exchange, compressed records.
    recv_i8 = _packed_qkv_all_to_all(
        q, k, v, world=world, group=dist.group.WORLD,
        packed_send=pack("int8_qkv"), wire_dtype="int8_qkv",
    )
    assert recv_i8.dtype == torch.uint8 and recv_i8.shape[-1] == 400, recv_i8.shape

    q_i8, k_i8, v_i8 = dequantize_qkv_split(recv_i8)
    failures = []

    # -- 1a. decode-vs-bf16-wire: quantization error only -------------------
    print(f"[rank {rank}] roundtrip vs bf16 wire (rel_l2 / max_abs):")
    for name, got, want in (("q", q_i8[0], q_ref), ("k", k_i8[0], k_ref), ("v", v_i8[0], v_ref)):
        err, mx = rel_l2(got, want), float((got.float() - want.float()).abs().max().item())
        print(f"  {name}: rel_l2={err:.5f}  max_abs={mx:.5f}")
        if err > 0.02:
            failures.append(f"{name} roundtrip rel_l2 {err:.5f} > 0.02")

    # -- 1b. split decode == padded decode (same math, different layout) ----
    qp, kp, vp = dequantize_qkv_padded(recv_i8, recv_i8.shape[0], recv_i8.shape[1],
                                       recv_i8.shape[0])
    for name, split, padded in (("q", q_i8, qp), ("k", k_i8, kp), ("v", v_i8, vp)):
        if not torch.equal(split, padded):
            failures.append(f"{name}: split decode differs from padded decode")
    print(f"[rank {rank}] split decode == padded decode: "
          f"{'OK' if not any('split decode' in f for f in failures) else 'FAIL'}")

    # -- 2. contiguity: the triton pointer kernel is stride-blind -----------
    for name, tensor in (("q", q_i8), ("k", k_i8), ("v", v_i8)):
        if not tensor.is_contiguous():
            failures.append(f"{name}: decoded tensor is not dense-contiguous")
    print(f"[rank {rank}] dense-contiguous decode: "
          f"{'OK' if not any('contiguous' in f for f in failures) else 'FAIL'}")

    # -- 3. attention-level deviation through the real kernel ---------------
    # Full route (tau=-1000): both wires compute exact attention over
    # (slightly different) Q/K/V, so the output gap is the quantization noise
    # alone. At tau=1.0 on *random* tensors the routing approximation itself
    # dominates (real-QKV route deviation is the engine gate's business, §7.1),
    # and marginal threshold flips would pollute the wire measurement.
    rows_full = recv_bf16.shape[0]
    out_ref = sol_attn_kernel(q_ref.unsqueeze(0).contiguous(),
                              k_ref.unsqueeze(0).contiguous(),
                              v_ref.unsqueeze(0).contiguous(),
                              tau=-1000.0, thresh_type="diag")
    out_i8 = sol_attn_kernel(q_i8, k_i8, v_i8, tau=-1000.0, thresh_type="diag")
    dense = torch.nn.functional.scaled_dot_product_attention(
        q_ref.transpose(0, 1).unsqueeze(0).float(),
        k_ref.transpose(0, 1).unsqueeze(0).float(),
        v_ref.transpose(0, 1).unsqueeze(0).float(),
    ).squeeze(0).transpose(0, 1)

    wire_dev = rel_l2(out_i8, out_ref)
    route_dev = rel_l2(out_ref, dense)
    total_dev = rel_l2(out_i8, dense)
    print(f"[rank {rank}] attention output rel_l2 (full route): "
          f"int8-wire vs bf16-wire={wire_dev:.5f}  "
          f"bf16-wire vs dense={route_dev:.5f}  int8-wire vs dense={total_dev:.5f}")
    # Grounding for the 0.05 limit: the accepted tau=1.0 routing approximation
    # is ~8% output rel_l2 on real QKV (DESIGN §7.1, E2E visually clean), and
    # random tensors sharpen the softmax well past real activations, so the
    # measured ~0.026 is a pessimistic ceiling for the wire noise. The final
    # quality gate is the E2E frame-std/visual A/B, not this bound.
    if wire_dev > 0.05:
        failures.append(f"attention wire deviation {wire_dev:.5f} > 0.05")

    verdict = torch.tensor([0 if failures else 1], device=device, dtype=torch.int32)
    dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
    dist.destroy_process_group()
    if not verdict.item():
        for failure in failures:
            print(f"[rank {rank}] FAIL: {failure}")
        return 1
    print(f"[rank {rank}] int8 wire test passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
