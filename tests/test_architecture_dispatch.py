# Copyright (C) 2026 Tencent.

"""Registration is shared even when the loaded architecture has no kernel."""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.realpath(list(Path(__file__).parent.glob("../build/lib.*/"))[0]))

import hpc
import pytest
import torch

_HAS_SM120_DEVICE = torch.cuda.is_available() and torch.cuda.get_device_capability(0) == (12, 0)


@pytest.mark.parametrize(
    "wrapper, operator",
    [
        ("topk", "topk_filtered"),
        ("topk_workspace_size", "topk_filtered_workspace_size"),
        ("gemm_bf16xfp32", "gemm_bf16xfp32"),
        ("fuse_moe_blockwise_mxfp4_sm120", "fuse_moe_mx"),
    ],
)
def test_common_registration(wrapper, operator):
    assert callable(getattr(hpc, wrapper))
    assert getattr(torch.ops.hpc, operator).default._schema is not None


def test_topk_fake_without_architecture_kernel():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        logits = torch.empty((1, 512), dtype=torch.float32, device="cuda")
        ke = torch.empty((1,), dtype=torch.int32, device="cuda")
        output = torch.empty((1, 512), dtype=torch.int32, device="cuda")
        valid_rows = torch.empty((1,), dtype=torch.int32, device="cuda")
        assert hpc.topk(logits, ke, output, valid_rows, top_k=512) is output


@pytest.mark.skipif(hpc.__loaded_arch__ != 120, reason="SM120 has no Top-K kernel")
@pytest.mark.skipif(not _HAS_SM120_DEVICE, reason="requires an SM120 CUDA device")
@pytest.mark.parametrize(
    "query, args",
    [
        ("topk_workspace_size", (1, 512)),
        ("topk_min_workspace_size", (512,)),
        ("topk_peak_workspace_size", (512,)),
    ],
)
def test_unsupported_workspace_query(query, args):
    operator = query.replace("topk", "topk_filtered", 1)
    with pytest.raises(
        RuntimeError, match=rf"hpc::{operator} does not run on sm120: implemented for 90, 100, 103"
    ):
        getattr(hpc, query)(*args)


@pytest.mark.skipif(hpc.__loaded_arch__ != 120, reason="SM120 has no Top-K kernel")
@pytest.mark.skipif(not _HAS_SM120_DEVICE, reason="requires an SM120 CUDA device")
@pytest.mark.parametrize("provided_workspace", [False, True])
def test_unsupported_topk_call(provided_workspace):
    logits = torch.zeros((1, 512), dtype=torch.float32, device="cuda")
    ke = torch.full((1,), 512, dtype=torch.int32, device="cuda")
    output = torch.full((1, 512), -7, dtype=torch.int32, device="cuda")
    valid_rows = torch.ones((1,), dtype=torch.int32, device="cuda")
    counters = torch.zeros((4,), dtype=torch.uint8, device="cuda") if provided_workspace else None
    workspace = torch.empty((4,), dtype=torch.uint8, device="cuda") if provided_workspace else None
    with pytest.raises(
        RuntimeError,
        match=r"hpc::topk_filtered does not run on sm120: implemented for 90, 100, 103",
    ):
        hpc.topk(logits, ke, output, valid_rows, 512, counters, workspace)
    assert torch.all(output == -7)


@pytest.mark.skipif(hpc.__loaded_arch__ != 120, reason="SM120 has no BF16 x FP32 GEMM kernel")
@pytest.mark.skipif(not _HAS_SM120_DEVICE, reason="requires an SM120 CUDA device")
def test_unsupported_gemm_call():
    x = torch.empty((16, 128), dtype=torch.bfloat16, device="cuda")
    weight = torch.empty((64, 128), dtype=torch.bfloat16, device="cuda")
    with pytest.raises(
        RuntimeError, match=r"hpc::gemm_bf16xfp32 does not run on sm120: implemented for 90,"
    ):
        hpc.gemm_bf16xfp32(x, weight, weight, 1.0 / 256)
