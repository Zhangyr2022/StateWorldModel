#!/usr/bin/env python3
"""NCCL communication bandwidth probe for the local A800 topology.

Measures, with one process per GPU:
  * full-world all-to-all (the exact collective the Ulysses attention exchange
    issues ~200 times per request),
  * pair-subgroup all-to-all (the NVLink bridges: 0-1, 2-7, 3-4, 5-6),
  * a cross-pair quad subgroup (NVLink pairs stitched over PCIe), and
  * full-world all_gather (the VAE tile gather).

Run: torchrun --standalone --nproc_per_node=8 bench/topo_probe.py
"""

from __future__ import annotations

import os
import time

import torch
import torch.distributed as dist

NV_PAIRS = [(0, 1), (2, 7), (3, 4), (5, 6)]
QUADS = [(0, 1, 2, 7), (3, 4, 5, 6)]
MB = 1024**2


def time_collective(fn, *, warmup: int = 3, repeat: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    started = time.perf_counter()
    for _ in range(repeat):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - started) / repeat


def report(label: str, seconds: float, bytes_moved: int, spokesman: int | None):
    # every subgroup's lowest global rank prints, so pairs/quads that don't
    # contain rank 0 still report their measurement
    if spokesman is not None and dist.get_rank() == spokesman:
        bandwidth = bytes_moved / seconds / MB
        print(f"{label:<40} {seconds * 1000:>9.2f} ms/call   {bandwidth:>9.1f} MB/s/rank", flush=True)


def main() -> int:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl", device_id=device)
    rank, world = dist.get_rank(), dist.get_world_size()

    # ~206 MB of BF16 per rank, sized like one Ulysses QKV exchange at ~38k rows.
    rows, heads, head_dim = 4779, 56, 128
    payload = torch.randn(rows * heads * 3 * head_dim, dtype=torch.bfloat16, device=device)

    def full_all_to_all():
        out = torch.empty_like(payload)
        dist.all_to_all_single(out, payload)

    seconds = time_collective(full_all_to_all)
    report(f"world-{world} all-to-all ({payload.numel() * 2 // MB} MB/rank)", seconds, payload.numel() * 2)

    gather_out = torch.empty(world * payload.numel(), dtype=torch.bfloat16, device=device)

    def full_all_gather():
        dist.all_gather_into_tensor(gather_out, payload)

    seconds = time_collective(full_all_gather)
    report(
        f"world-{world} all-gather ({payload.numel() * 2 // MB} MB/rank)",
        seconds,
        payload.numel() * 2 * (world - 1) / world,
    )

    # Every rank must create every subgroup in the same order, member or not.
    pair_groups = {members: dist.new_group(ranks=list(members)) for members in NV_PAIRS}
    quad_groups = {members: dist.new_group(ranks=list(members)) for members in QUADS}
    dist.barrier()

    for members, group in pair_groups.items():
        if rank not in members:
            continue
        share = world // len(members)
        chunk = payload[: payload.numel() // share]

        def pair_all_to_all(chunk=chunk, group=group):
            out = torch.empty_like(chunk)
            dist.all_to_all_single(out, chunk, group=group)

        seconds = time_collective(pair_all_to_all)
        report(f"NVLink pair {list(members)} all-to-all", seconds, chunk.numel() * 2,
               spokesman=min(members))

    for members, group in quad_groups.items():
        if rank not in members:
            continue
        share = world // len(members)
        chunk = payload[: payload.numel() // share]

        def quad_all_to_all(chunk=chunk, group=group):
            out = torch.empty_like(chunk)
            dist.all_to_all_single(out, chunk, group=group)

        seconds = time_collective(quad_all_to_all)
        report(f"cross-pair quad {list(members)} all-to-all", seconds, chunk.numel() * 2,
               spokesman=min(members))

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
