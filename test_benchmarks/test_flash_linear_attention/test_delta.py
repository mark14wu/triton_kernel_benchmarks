# -*- coding: utf-8 -*-

import pytest
import torch

from fla.ops.delta_rule import chunk_delta_rule, fused_recurrent_delta_rule


def get_err_ratio(x, y):
    err = (x-y).flatten().square().mean().sqrt().item()
    base = (x).flatten().square().mean().sqrt().item()
    return err / base


@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [286, 300])
@pytest.mark.parametrize("D", [50, 100, 200])
@pytest.mark.parametrize("scale", [1, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_chunk_delta_rule(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    scale: float
):
    q = torch.randn(B, H, T, D, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, H, T, D, dtype=dtype)
    beta = torch.rand(B, H, T, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    tri, tri_ht = chunk_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone()
    )
    ((tri * do).sum() + (tri_ht * dht).sum()).backward(retain_graph=True)
    tri_dq, tri_dk, tri_dv, tri_dbeta, tri_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad

@pytest.mark.parametrize("B", [2])
@pytest.mark.parametrize("H", [2])
@pytest.mark.parametrize("T", [286, 300])
@pytest.mark.parametrize("D", [50, 100, 200])
@pytest.mark.parametrize("scale", [1, 0.5])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_fused_recurrent_delta_rule(
    B: int,
    H: int,
    T: int,
    D: int,
    dtype: torch.dtype,
    scale: float
):
    q = torch.randn(B, H, T, D, dtype=dtype)
    k = torch.nn.functional.normalize(torch.randn(B, H, T, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
    v = torch.randn(B, H, T, D, dtype=dtype)
    beta = torch.rand(B, H, T, dtype=dtype).sigmoid()
    h0 = torch.randn(B, H, D, D, dtype=torch.float32)
    q, k, v, beta, h0 = map(lambda x: x.cuda().requires_grad_(True), (q, k, v, beta, h0))
    do = torch.rand_like(v)
    dht = torch.rand_like(h0)

    ref, ref_ht = fused_recurrent_delta_rule(
        q.clone(),
        k.clone(),
        v.clone(),
        beta.clone(),
        scale=scale,
        output_final_state=True,
        initial_state=h0.clone()
    )
    ((ref * do).sum() + (ref_ht * dht).sum()).backward(retain_graph=True)
    ref_dq, ref_dk, ref_dv, ref_dbeta, ref_dh0 = q.grad, k.grad, v.grad, beta.grad, h0.grad
