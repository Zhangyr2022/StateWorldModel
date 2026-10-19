#!/usr/bin/env python3
"""SM80 smoke test: the four Triton fusion kernels compile and match eager.

No checkpoint needed — random tensors at production-like shapes. Run on one
GPU: python tests/test_fusions_sm80.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from h3_runtime.fusions import (  # noqa: E402
    fused_qknorm_rope, fused_residual_gate_rmsnorm_modulate, fused_rmsnorm_modulate,
    fused_swiglu,
)

HIDDEN = 5376
ROWS = 512
HEADS, HEAD_DIM = 56, 128
ROTARY_DIM = 64


def _chunked_tables(rows: int, hidden: int, device):
    """Modulation tables as the model produces them: chunk(6) views of one wide projection."""
    wide = torch.randn(rows, 6 * hidden, dtype=torch.bfloat16, device=device) * 0.02
    return wide.chunk(6, dim=-1)


def test_rmsnorm_modulate(device):
    torch.manual_seed(0)
    x = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    shift_msa, scale_msa, _, _, _, _ = _chunked_tables(16, HIDDEN, device)
    index = torch.randint(0, 16, (ROWS,), dtype=torch.long, device=device)
    eps = 1e-6

    fused = fused_rmsnorm_modulate(x, weight, scale_msa, shift_msa, index, eps)

    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    normed = normed * weight.float()
    reference = (
        normed * (1.0 + scale_msa.float().index_select(0, index))
        + shift_msa.float().index_select(0, index)
    )
    error = (fused.float() - reference).abs().max().item()
    relative = error / reference.abs().max().item()
    print(f"rmsnorm_modulate       max_abs={error:.3e} relative={relative:.2e}")
    assert relative < 2e-2, relative


def test_residual_gate(device):
    torch.manual_seed(1)
    residual = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    branch = torch.randn(ROWS, HIDDEN, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device)
    gate_msa, shift_mlp, scale_mlp, _, _, _ = _chunked_tables(16, HIDDEN, device)
    index = torch.randint(0, 16, (ROWS,), dtype=torch.long, device=device)
    eps = 1e-6

    hidden, normed = fused_residual_gate_rmsnorm_modulate(
        residual, branch, gate_msa, weight, scale_mlp, shift_mlp, index, eps
    )

    hidden_ref = residual.float() + gate_msa.float().index_select(0, index) * branch.float()
    normed_ref = hidden_ref * torch.rsqrt(hidden_ref.pow(2).mean(-1, keepdim=True) + eps)
    normed_ref = normed_ref * weight.float()
    normed_ref = (
        normed_ref * (1.0 + scale_mlp.float().index_select(0, index))
        + shift_mlp.float().index_select(0, index)
    )
    error_hidden = (hidden.float() - hidden_ref).abs().max().item()
    error_normed = (normed.float() - normed_ref).abs().max().item()
    scale = max(hidden_ref.abs().max().item(), normed_ref.abs().max().item())
    print(f"residual_gate_rmsnorm  max_abs hidden={error_hidden:.3e} normed={error_normed:.3e} "
          f"(relative {max(error_hidden, error_normed) / scale:.2e})")
    assert max(error_hidden, error_normed) / scale < 2e-2, (error_hidden, error_normed)


def test_swiglu(device):
    torch.manual_seed(2)
    intermediate = 14336
    x = torch.randn(ROWS, 2 * intermediate, dtype=torch.bfloat16, device=device)
    fused = fused_swiglu(x)
    value, gate = x[:, :intermediate].float(), x[:, intermediate:].float()
    reference = value * (gate * torch.sigmoid(gate))
    error = (fused.float() - reference).abs().max().item()
    relative = error / reference.abs().max().item()
    print(f"swiglu                 max_abs={error:.3e} relative={relative:.2e}")
    assert relative < 2e-2, relative


def test_qknorm_rope(device):
    torch.manual_seed(3)
    x = torch.randn(1, 128, HEADS, HEAD_DIM, dtype=torch.bfloat16, device=device)
    weight = torch.randn(HEAD_DIM, dtype=torch.bfloat16, device=device)
    cos = torch.randn(1, 128, ROTARY_DIM, dtype=torch.float32, device=device)
    sin = torch.randn(1, 128, ROTARY_DIM, dtype=torch.float32, device=device)
    eps = 1e-6

    fused = fused_qknorm_rope(x, weight, cos, sin, eps)

    def reference_one(t):
        normed = t.float() * torch.rsqrt(t.float().pow(2).mean(-1, keepdim=True) + eps)
        normed = normed * weight.float()
        first, second = normed[..., : ROTARY_DIM // 2], normed[..., ROTARY_DIM // 2: ROTARY_DIM]
        rotated = torch.cat((-second, first), dim=-1)
        # cos/sin are (1, seq, rotary_dim); broadcast over the heads dimension.
        cos_b = cos[0].unsqueeze(0).unsqueeze(2)
        sin_b = sin[0].unsqueeze(0).unsqueeze(2)
        out_rotary = normed[..., :ROTARY_DIM] * cos_b + rotated * sin_b
        return torch.cat((out_rotary, normed[..., ROTARY_DIM:]), dim=-1)

    reference = reference_one(x)
    difference = (fused.float() - reference).abs().max().item()
    scale = reference.abs().max().item()
    print(f"qknorm_partial_rope    max_abs={difference:.3e} (reference max {scale:.1f}, "
          f"relative {difference / scale:.2e})")
    # One bf16 rounding of the output; the kernel is fp32 internally.
    assert difference / scale < 1e-2, (difference, scale)


def main() -> int:
    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    print(f"device: {torch.cuda.get_device_name()} (SM{torch.cuda.get_device_capability(device)[0]}"
          f"{torch.cuda.get_device_capability(device)[1]})")
    test_rmsnorm_modulate(device)
    test_residual_gate(device)
    test_swiglu(device)
    test_qknorm_rope(device)
    print("ALL FUSION KERNELS PASS ON SM80")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
