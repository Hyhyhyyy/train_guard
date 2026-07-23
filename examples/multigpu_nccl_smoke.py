#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多 GPU NCCL 通信冒烟测试（torch.distributed）。

用法示例（3 张 A100）：
  torchrun --standalone --nproc_per_node=3 examples/multigpu_nccl_smoke.py

从环境变量读取 LOCAL_RANK，每进程绑定对应 GPU，对值为 local_rank+1 的张量做 all_reduce 求和。
With 3 GPUs the expected all_reduce sum is 6. Does not modify the training stack.
"""

from __future__ import annotations

import os
import sys


def main() -> int:
    """执行 NCCL all_reduce 冒烟检查。"""
    try:
        import torch
        import torch.distributed as dist
    except ImportError as exc:
        print(f"错误: 无法导入 PyTorch: {exc}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("错误: CUDA 不可用。", file=sys.stderr)
        return 2

    local_rank_raw = os.environ.get("LOCAL_RANK")
    if local_rank_raw is None:
        print(
            "错误: 未设置环境变量 LOCAL_RANK。请使用 torchrun 启动，例如：\n"
            "  torchrun --standalone --nproc_per_node=3 examples/multigpu_nccl_smoke.py",
            file=sys.stderr,
        )
        return 2

    try:
        local_rank = int(local_rank_raw)
    except ValueError:
        print(f"错误: LOCAL_RANK 无效: {local_rank_raw!r}", file=sys.stderr)
        return 2

    if local_rank < 0 or local_rank >= torch.cuda.device_count():
        print(
            f"错误: LOCAL_RANK={local_rank} 超出可用 GPU 数量 {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        return 2

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    gpu_name = torch.cuda.get_device_name(local_rank)

    # 兼容 torchrun 注入的 RANK / WORLD_SIZE / MASTER_*
    if not dist.is_available():
        print("错误: 当前 PyTorch 未编译分布式支持。", file=sys.stderr)
        return 2

    backend = "nccl"
    try:
        dist.init_process_group(backend=backend, device_id=device)
    except TypeError:
        # 旧签名无 device_id
        dist.init_process_group(backend=backend)
    except Exception as exc:  # noqa: BLE001
        print(f"错误: init_process_group(NCCL) 失败: {exc}", file=sys.stderr)
        return 1

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.tensor([float(local_rank + 1)], device=device, dtype=torch.float32)
    try:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        dist.barrier()
    except Exception as exc:  # noqa: BLE001
        print(f"错误: all_reduce/barrier 失败 (rank={rank}): {exc}", file=sys.stderr)
        try:
            dist.destroy_process_group()
        except Exception:  # noqa: BLE001
            pass
        return 1

    result = float(tensor.item())
    expected = float(sum(range(1, world_size + 1)))  # 1+2+...+world_size；3 卡时为 6
    ok = abs(result - expected) < 1e-5

    print(
        f"[rank={rank} local_rank={local_rank}] "
        f"GPU={gpu_name} device=cuda:{local_rank} "
        f"all_reduce={result} expected={expected} ok={ok}",
        flush=True,
    )

    try:
        dist.barrier()
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        print(f"错误: barrier/destroy_process_group 失败 (rank={rank}): {exc}", file=sys.stderr)
        return 1

    if not ok:
        print(
            f"错误: all_reduce 结果不正确 (rank={rank}): got={result}, expected={expected}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
