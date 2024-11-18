# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.retention import (chunk_retention, fused_recurrent_retention,
                               parallel_retention)


def get_abs_err(x, y):
    return (x-y).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunk(
    B: int,
    H: int,
    T: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    V = K * expand_ratio

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    h0 = torch.randn((B, H, K, V), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)

    tri, tri_ht = chunk_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((tri * do).sum() + (tri_ht * dht).sum()).backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None



@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_parallel(
    B: int,
    H: int,
    T: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    V = K * expand_ratio

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    do = torch.randn_like(v)

    tri, _ = parallel_retention(q, k, v)
    tri.backward(do)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None

@pytest.mark.parametrize("B", [4])
@pytest.mark.parametrize("H", [4])
@pytest.mark.parametrize("T", [300, 512])
@pytest.mark.parametrize("K", [32, 64, 100])
@pytest.mark.parametrize("expand_ratio", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_fused_recurrent(
    B: int,
    H: int,
    T: int,
    K: int,
    expand_ratio: int,
    dtype: torch.dtype
):
    torch.manual_seed(42)
    V = K * expand_ratio

    q = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    k = torch.randn((B, H, T, K), dtype=dtype, device='cuda').requires_grad_()
    v = torch.randn((B, H, T, V), dtype=dtype, device='cuda').requires_grad_()
    h0 = torch.randn((B, H, K, V), dtype=dtype, device='cuda').requires_grad_()

    do = torch.randn_like(v)
    dht = torch.randn_like(h0)
    ref, ref_ht = fused_recurrent_retention(q, k, v, initial_state=h0, output_final_state=True)
    ((ref * do).sum() + (ref_ht * dht).sum()).backward()
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None