# Copyright (c) 2026, FlashAttention contributors.
"""Correctness tests for the packed-QKV FA4 entry points.

The tests in this file intentionally use MHA only.  In particular, they test the
packing convention and the packed backward allocation rather than GQA packing.
Compilation-only runs use the same FakeTensorMode convention as the other CuTe
attention tests; numerical checks are skipped in that mode.
"""

import itertools
import os

import pytest
import torch

from flash_attn.cute import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
)
from flash_attn.cute.testing import is_fake_mode, maybe_fake_tensor_mode

USE_FAKE_TENSOR = int(os.getenv("FLASH_ATTENTION_FAKE_TENSOR", "0")) == 1
CUDA_AVAILABLE = torch.cuda.is_available()

pytestmark = pytest.mark.skipif(
    not CUDA_AVAILABLE and not USE_FAKE_TENSOR,
    reason="packed-QKV numerical tests require CUDA",
)


def _is_arch(*architectures):
    from flash_attn.cute.interface import _get_device_arch

    return (CUDA_AVAILABLE or USE_FAKE_TENSOR) and _get_device_arch() // 10 in architectures


def _assert_close(actual, expected, *, dtype, kind):
    """Use tolerances appropriate for a low precision fused attention kernel."""
    if is_fake_mode():
        assert actual.shape == expected.shape
        return
    assert torch.isfinite(actual).all(), kind
    assert torch.isfinite(expected).all(), f"reference {kind}"
    if kind == "lse":
        atol, rtol = 5e-3, 5e-3
    elif kind == "grad":
        atol, rtol = (4e-2, 4e-2) if dtype == torch.bfloat16 else (8e-3, 8e-3)
    else:
        atol, rtol = (2e-2, 2e-2) if dtype == torch.bfloat16 else (5e-3, 5e-3)
    torch.testing.assert_close(actual.float(), expected.float(), atol=atol, rtol=rtol)


def _assert_same_kernel(actual, expected, *, kind):
    """Packed and separate APIs should use the same numerical forward kernel."""
    if is_fake_mode():
        assert actual.shape == expected.shape
    else:
        torch.testing.assert_close(actual.float(), expected.float(), atol=5e-3, rtol=5e-3, msg=kind)


def _dense_reference(qkv, causal, softcap=0.0):
    """Independent float32 reference; this does not call any flash-attention code."""
    q, k, v = qkv.float().unbind(dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k) * (q.shape[-1] ** -0.5)
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    if causal:
        mask = torch.triu(
            torch.ones((q.shape[1], k.shape[1]), device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    out = torch.einsum("bhqk,bkhd->bqhd", scores.softmax(dim=-1), v)
    return out, lse


def _dense_concat_reference(qkv, num_heads_q, causal, softcap=0.0, seqused=None):
    qkv = qkv.float()
    q = qkv[..., :num_heads_q, :]
    num_heads_kv = (qkv.shape[-2] - num_heads_q) // 2
    k = qkv[..., num_heads_q : num_heads_q + num_heads_kv, :]
    v = qkv[..., num_heads_q + num_heads_kv :, :]
    scores = torch.einsum("bqhd,bkhd->bhqk", q, k.repeat_interleave(num_heads_q // num_heads_kv, dim=2))
    scores = scores * (q.shape[-1] ** -0.5)
    if softcap:
        scores = softcap * torch.tanh(scores / softcap)
    B, S = q.shape[:2]
    key_idx = torch.arange(S, device=q.device)
    query_idx = key_idx
    q_valid = torch.ones(B, S, device=q.device, dtype=torch.bool)
    for b in range(B):
        if seqused is not None:
            q_valid[b] = query_idx < seqused[b]
        valid = key_idx[None, :] < (seqused[b] if seqused is not None else S)
        if causal:
            valid = valid & (key_idx[None, :] <= query_idx[:, None])
        valid = valid & q_valid[b, :, None]
        scores[b] = scores[b].masked_fill(~valid, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.nan_to_num(scores.softmax(dim=-1), nan=0.0)
    values = v.repeat_interleave(num_heads_q // num_heads_kv, dim=2)
    out = torch.einsum("bhqk,bkhd->bqhd", probs, values)
    out = out * q_valid[:, :, None, None]
    lse = torch.where(q_valid[:, None, :], lse, torch.full_like(lse, float("-inf")))
    return out, lse


def _varlen_reference(qkv, cu_seqlens, causal):
    """Independent per-sequence reference for equal-length Q/K varlen MHA."""
    q, k, v = qkv.float().unbind(dim=1)
    lengths = cu_seqlens.detach().cpu().tolist()
    outputs = []
    lses = []
    for start, end in itertools.pairwise(lengths):
        if end == start:
            continue
        qb, kb, vb = q[start:end], k[start:end], v[start:end]
        scores = torch.einsum("qhd,khd->hqk", qb, kb) * (q.shape[-1] ** -0.5)
        if causal:
            mask = torch.triu(
                torch.ones((end - start, end - start), device=q.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask, float("-inf"))
        lses.append(torch.logsumexp(scores, dim=-1))
        outputs.append(torch.einsum("hqk,khd->qhd", scores.softmax(dim=-1), vb))
    if not outputs:
        return (
            torch.empty((0, q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32),
            torch.empty((q.shape[1], 0), device=q.device, dtype=torch.float32),
        )
    return torch.cat(outputs, dim=0), torch.cat(lses, dim=1)


def _dense_inputs(dtype, head_dim, *, seqlen=65, nheads=2, batch=1):
    return torch.randn(
        batch, seqlen, 3, nheads, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )


def _separate_dense_inputs(qkv):
    # Independent leaves make the three separate gradients directly comparable
    # with qkv.grad, including when qkv itself has non-standard outer strides.
    return tuple(x.detach().clone().requires_grad_() for x in qkv.unbind(dim=2))


def _separate_varlen_inputs(qkv):
    return tuple(x.detach().clone().requires_grad_() for x in qkv.unbind(dim=1))


def _check_dense_case(qkv, causal, *, lse_grad=True):
    dtype = qkv.dtype
    out_p, lse_p = flash_attn_qkvpacked_func(
        qkv, causal=causal, num_splits=1, return_lse=True
    )
    q, k, v = _separate_dense_inputs(qkv)
    out_s, lse_s = flash_attn_func(
        q, k, v, causal=causal, num_splits=1, return_lse=True
    )

    assert out_p.shape == out_s.shape == qkv.shape[:2] + qkv.shape[3:]
    assert lse_p.shape == lse_s.shape == (qkv.shape[0], qkv.shape[3], qkv.shape[1])
    # Exercise backward before returning in fake mode, so pass 1 compiles the
    # packed stores as well as forward. Use a noncontiguous upstream gradient.
    dout = torch.randn((*out_p.shape[:-1], 2 * out_p.shape[-1]), device=out_p.device, dtype=dtype)[..., ::2]
    dlse = torch.randn_like(lse_p)
    if lse_grad:
        dqkv_p = torch.autograd.grad((out_p, lse_p), qkv, (dout, dlse))[0]
        grads_s = torch.autograd.grad((out_s, lse_s), (q, k, v), (dout, dlse))
    else:
        dqkv_p = torch.autograd.grad(out_p, qkv, dout)[0]
        grads_s = torch.autograd.grad(out_s, (q, k, v), dout)
    dqkv_s = torch.stack(grads_s, dim=2)
    if is_fake_mode():
        assert dqkv_p.shape == qkv.shape
        return

    qkv_ref = qkv.detach().float().requires_grad_()
    out_ref, lse_ref = _dense_reference(qkv_ref, causal)
    _assert_same_kernel(out_p, out_s, kind="dense packed/separate output")
    _assert_same_kernel(lse_p, lse_s, kind="dense packed/separate lse")
    _assert_close(out_p, out_ref, dtype=dtype, kind="output")
    _assert_close(lse_p, lse_ref, dtype=dtype, kind="lse")
    if lse_grad:
        dqkv_ref = torch.autograd.grad((out_ref, lse_ref), qkv_ref, (dout.float(), dlse.float()))[0]
    else:
        dqkv_ref = torch.autograd.grad(out_ref, qkv_ref, dout.float())[0]
    _assert_close(dqkv_p, dqkv_s, dtype=dtype, kind="grad")
    _assert_close(dqkv_p, dqkv_ref, dtype=dtype, kind="grad")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_dense(dtype, head_dim, causal):
    """Dense packed fwd, output+LSE backward, and an independent reference."""
    torch.manual_seed(1100 + head_dim + int(causal))
    _check_dense_case(_dense_inputs(dtype, head_dim), causal)


def _check_varlen_case(qkv, cu_seqlens, max_seqlen, causal):
    dtype = qkv.dtype
    out_p, lse_p = flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, max_seqlen, causal=causal, num_splits=1, return_lse=True
    )
    q, k, v = _separate_varlen_inputs(qkv)
    out_s, lse_s = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=causal,
        num_splits=1,
        return_lse=True,
    )

    assert out_p.shape == out_s.shape == qkv.shape[:1] + qkv.shape[2:]
    assert lse_p.shape == lse_s.shape == (qkv.shape[2], qkv.shape[0])
    dout = torch.randn_like(out_p)
    dlse = torch.randn_like(lse_p)
    dqkv_p = torch.autograd.grad((out_p, lse_p), qkv, (dout, dlse))[0]
    dq_s, dk_s, dv_s = torch.autograd.grad((out_s, lse_s), (q, k, v), (dout, dlse))
    dqkv_s = torch.stack((dq_s, dk_s, dv_s), dim=1)
    if is_fake_mode():
        assert dqkv_p.shape == qkv.shape
        return

    qkv_ref = qkv.detach().float().requires_grad_()
    out_ref, lse_ref = _varlen_reference(qkv_ref, cu_seqlens, causal)
    _assert_same_kernel(out_p, out_s, kind="varlen packed/separate output")
    _assert_same_kernel(lse_p, lse_s, kind="varlen packed/separate lse")
    _assert_close(out_p, out_ref, dtype=dtype, kind="output")
    _assert_close(lse_p, lse_ref, dtype=dtype, kind="lse")
    dqkv_ref = torch.autograd.grad(
        (out_ref, lse_ref), qkv_ref, (dout.float(), dlse.float())
    )[0]
    _assert_close(dqkv_p, dqkv_s, dtype=dtype, kind="grad")
    _assert_close(dqkv_p, dqkv_ref, dtype=dtype, kind="grad")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("head_dim", [64, 96, 128])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_qkvpacked(dtype, head_dim, causal):
    """Varlen packed fwd/dQKV with an empty slot and a ragged tile tail."""
    torch.manual_seed(2200 + head_dim + int(causal))
    lengths = (65, 0, 67)
    cu_seqlens = torch.tensor(
        (0, 65, 65, 132), device="cuda", dtype=torch.int32
    )
    qkv = torch.randn(
        132, 3, 2, head_dim, device="cuda", dtype=dtype, requires_grad=True
    )
    _check_varlen_case(qkv, cu_seqlens, max(lengths), causal)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_noncontiguous_outer_stride(dtype, causal):
    """A strided packed view must not overwrite its storage guards."""
    torch.manual_seed(3300 + int(causal))
    batch, seqlen, heads, head_dim = 1, 65, 2, 64
    sentinel = 17.25
    storage = torch.full(
        (batch, 2 * seqlen + 3, 3, heads, head_dim),
        sentinel,
        device="cuda",
        dtype=dtype,
    )
    qkv_view = storage[:, 1 : 1 + 2 * seqlen : 2]
    qkv_view.copy_(torch.randn_like(qkv_view))
    storage_before = storage.clone()
    qkv = qkv_view.detach().requires_grad_(True)

    out_p, lse_p = flash_attn_qkvpacked_func(qkv, causal=causal, return_lse=True)
    q, k, v = _separate_dense_inputs(qkv)
    out_s, lse_s = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    dout = torch.randn_like(out_p)
    dqkv_p = torch.autograd.grad(out_p, qkv, dout)[0]
    dq_s, dk_s, dv_s = torch.autograd.grad(out_s, (q, k, v), dout)
    if is_fake_mode():
        assert out_p.shape == out_s.shape
        assert dqkv_p.shape == qkv.shape
        return
    _assert_same_kernel(out_p, out_s, kind="strided packed/separate output")
    _assert_same_kernel(lse_p, lse_s, kind="strided packed/separate lse")
    _assert_close(dqkv_p, torch.stack((dq_s, dk_s, dv_s), dim=2), dtype=dtype, kind="grad")
    # Both forward and backward must leave all interleaved input guards intact.
    assert torch.equal(storage, storage_before)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_qkvpacked_empty_max_seqlen_zero(dtype):
    """All-empty varlen batches, including the explicit max_seqlen=0 form."""
    qkv = torch.empty((0, 3, 2, 64), device="cuda", dtype=dtype, requires_grad=True)
    cu_seqlens = torch.zeros((3,), device="cuda", dtype=torch.int32)
    out_p, lse_p = flash_attn_varlen_qkvpacked_func(
        qkv, cu_seqlens, 0, return_lse=True
    )
    q, k, v = _separate_varlen_inputs(qkv)
    out_s, lse_s = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=0,
        max_seqlen_k=0,
        return_lse=True,
    )
    assert out_p.shape == out_s.shape == (0, 2, 64)
    assert lse_p.shape == lse_s.shape == (2, 0)
    grad_p = torch.autograd.grad(out_p, qkv, torch.empty_like(out_p))[0]
    grad_s = torch.autograd.grad(out_s, (q, k, v), torch.empty_like(out_s))
    assert grad_p.shape == qkv.shape
    if is_fake_mode():
        return
    _assert_same_kernel(out_p, out_s, kind="empty packed/separate output")
    _assert_same_kernel(lse_p, lse_s, kind="empty packed/separate lse")
    assert torch.equal(grad_p, torch.stack(grad_s, dim=1))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_lse_only_backward(dtype):
    """The returned LSE remains differentiable when its output is used alone."""
    torch.manual_seed(4400)
    qkv = _dense_inputs(dtype, 64)
    _, lse_p = flash_attn_qkvpacked_func(qkv, causal=True, return_lse=True)
    q, k, v = _separate_dense_inputs(qkv)
    _, lse_s = flash_attn_func(q, k, v, causal=True, return_lse=True)
    dlse = torch.randn_like(lse_p)
    grad_p = torch.autograd.grad(lse_p, qkv, dlse)[0]
    grad_s = torch.autograd.grad(lse_s, (q, k, v), dlse)
    if is_fake_mode():
        assert grad_p.shape == qkv.shape
        return

    qkv_ref = qkv.detach().float().requires_grad_()
    _, lse_ref = _dense_reference(qkv_ref, True)
    grad_ref = torch.autograd.grad(lse_ref, qkv_ref, dlse.float())[0]
    _assert_close(grad_p, torch.stack(grad_s, dim=2), dtype=dtype, kind="grad")
    _assert_close(grad_p, grad_ref, dtype=dtype, kind="grad")


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_cache_warmup_separate_packed_separate(causal):
    """A packed compile must not poison the separate API's warmed cache entry."""
    torch.manual_seed(5500 + int(causal))
    qkv = _dense_inputs(torch.bfloat16, 64)
    q, k, v = qkv.unbind(dim=2)
    out_first, _ = flash_attn_func(q, k, v, causal=causal, return_lse=False)
    out_packed, _ = flash_attn_qkvpacked_func(qkv, causal=causal, return_lse=False)
    out_last, _ = flash_attn_func(q, k, v, causal=causal, return_lse=False)
    assert out_first.shape == out_packed.shape == out_last.shape
    if is_fake_mode():
        return
    _assert_same_kernel(out_first, out_packed, kind="cache warmup packed output")
    _assert_same_kernel(out_first, out_last, kind="cache warmup separate output")


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_dense_and_varlen_share_autograd_function():
    """Dense and varlen wrappers are backed by one packed custom Function."""
    qkv_dense = _dense_inputs(torch.bfloat16, 64, seqlen=65)
    out_dense, _ = flash_attn_qkvpacked_func(qkv_dense, return_lse=True)
    qkv_varlen = torch.randn(
        65, 3, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    cu_seqlens = torch.tensor([0, 65], device="cuda", dtype=torch.int32)
    out_varlen, _ = flash_attn_varlen_qkvpacked_func(
        qkv_varlen, cu_seqlens, 65, return_lse=True
    )
    assert type(out_dense.grad_fn) is type(out_varlen.grad_fn)


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_concat_gqa(causal):
    """Concatenated-head GQA packing shares the regular FA4 GQA kernels."""
    dtype, D, Hq, Hkv, S = torch.bfloat16, 64, 4, 2, 33
    qkv = torch.randn(1, S, Hq + 2 * Hkv, D, device="cuda", dtype=dtype, requires_grad=True)
    out_p, lse_p = flash_attn_qkvpacked_func(
        qkv, num_heads_q=Hq, causal=causal, return_lse=True
    )
    q = qkv[..., :Hq, :]
    k = qkv[..., Hq : Hq + Hkv, :]
    v = qkv[..., Hq + Hkv :, :]
    out_s, lse_s = flash_attn_func(q, k, v, causal=causal, return_lse=True)
    assert out_p.shape == out_s.shape == (1, S, Hq, D)
    assert lse_p.shape == lse_s.shape == (1, Hq, S)
    dout, dlse = torch.randn_like(out_p), torch.randn_like(lse_p)
    grad_p = torch.autograd.grad((out_p, lse_p), qkv, (dout, dlse))[0]
    grad_s = torch.autograd.grad((out_s, lse_s), (q, k, v), (dout, dlse))
    grad_s = torch.cat(grad_s, dim=-2)
    if is_fake_mode():
        assert grad_p.shape == qkv.shape
        return
    ref_qkv = qkv.detach().float().requires_grad_()
    ref_out, ref_lse = _dense_concat_reference(ref_qkv, Hq, causal)
    grad_ref = torch.autograd.grad((ref_out, ref_lse), ref_qkv, (dout.float(), dlse.float()))[0]
    _assert_close(out_p, ref_out, dtype=dtype, kind="gqa output")
    _assert_close(lse_p, ref_lse, dtype=dtype, kind="gqa lse")
    _assert_close(grad_p, grad_s, dtype=dtype, kind="gqa grad")
    _assert_close(grad_p, grad_ref, dtype=dtype, kind="gqa reference grad")


@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_varlen_qkvpacked_concat_gqa():
    """Variable-length concatenated-head GQA uses the same packed gradient layout."""
    dtype, D, Hq, Hkv = torch.bfloat16, 64, 4, 2
    cu = torch.tensor([0, 17, 17, 40], device="cuda", dtype=torch.int32)
    qkv = torch.randn(40, Hq + 2 * Hkv, D, device="cuda", dtype=dtype, requires_grad=True)
    out_p, lse_p = flash_attn_varlen_qkvpacked_func(
        qkv, cu, 17, num_heads_q=Hq, causal=True, return_lse=True
    )
    q = qkv[..., :Hq, :]
    k = qkv[..., Hq : Hq + Hkv, :]
    v = qkv[..., Hq + Hkv :, :]
    out_s, lse_s = flash_attn_varlen_func(
        q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu,
        max_seqlen_q=17, max_seqlen_k=17, causal=True, return_lse=True
    )
    dout, dlse = torch.randn_like(out_p), torch.randn_like(lse_p)
    grad_p = torch.autograd.grad((out_p, lse_p), qkv, (dout, dlse))[0]
    grad_s = torch.cat(torch.autograd.grad((out_s, lse_s), (q, k, v), (dout, dlse)), dim=-2)
    if is_fake_mode():
        assert grad_p.shape == qkv.shape
        return
    _assert_same_kernel(out_p, out_s, kind="varlen gqa output")
    _assert_same_kernel(lse_p, lse_s, kind="varlen gqa lse")
    _assert_close(grad_p, grad_s, dtype=dtype, kind="varlen gqa grad")


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_seqused(causal):
    """Dense packed storage can carry different effective lengths per batch item."""
    dtype, D, H, S = torch.float16, 64, 2, 33
    used = torch.tensor([33, 17], device="cuda", dtype=torch.int32)
    qkv = torch.randn(2, S, 3, H, D, device="cuda", dtype=dtype, requires_grad=True)
    out_p, lse_p = flash_attn_qkvpacked_func(qkv, seqused=used, causal=causal, return_lse=True)
    q, k, v = qkv.unbind(dim=2)
    out_s, lse_s = flash_attn_varlen_func(
        q, k, v, seqused_q=used, seqused_k=used, causal=causal, return_lse=True
    )
    valid_rows = torch.arange(S, device="cuda")[None, :] < used[:, None]
    dout, dlse = torch.randn_like(out_p), torch.randn_like(lse_p)
    dout = dout.masked_fill(~valid_rows[:, :, None, None], 0)
    dlse = dlse.masked_fill(~valid_rows[:, None, :], 0)
    grad_p = torch.autograd.grad((out_p, lse_p), qkv, (dout, dlse))[0]
    grad_s = torch.stack(torch.autograd.grad((out_s, lse_s), (q, k, v), (dout, dlse)), dim=2)
    if is_fake_mode():
        assert grad_p.shape == qkv.shape
        return
    _assert_same_kernel(out_p[valid_rows], out_s[valid_rows], kind="seqused output")
    _assert_same_kernel(lse_p.permute(0, 2, 1)[valid_rows], lse_s.permute(0, 2, 1)[valid_rows], kind="seqused lse")
    for b, length in enumerate(used.tolist()):
        for component in range(3):
            _assert_close(
                grad_p[b, :length, component],
                grad_s[b, :length, component],
                dtype=dtype,
                kind="seqused grad",
            )
    # The packed wrapper canonicalizes dense suffix rows to zero.
    assert torch.count_nonzero(out_p[~valid_rows]) == 0
    assert torch.count_nonzero(lse_p.permute(0, 2, 1)[~valid_rows]) == 0


@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_softcap_sm120(causal):
    """Generated softcap backward is supported on the SM120 SM80-style path."""
    if CUDA_AVAILABLE and torch.cuda.get_device_capability()[0] != 12:
        pytest.skip("focused test targets the SM120 path")
    dtype, D = torch.bfloat16, 64
    qkv = _dense_inputs(dtype, D, seqlen=33, nheads=2)
    out_p, lse_p = flash_attn_qkvpacked_func(
        qkv, causal=causal, softcap=10.0, return_lse=True
    )
    dout, dlse = torch.randn_like(out_p), torch.randn_like(lse_p)
    grad_p = torch.autograd.grad((out_p, lse_p), qkv, (dout, dlse))[0]
    if is_fake_mode():
        assert grad_p.shape == qkv.shape
        return
    ref_qkv = qkv.detach().float().requires_grad_()
    ref_out, ref_lse = _dense_reference(ref_qkv, causal, softcap=10.0)
    grad_ref = torch.autograd.grad((ref_out, ref_lse), ref_qkv, (dout.float(), dlse.float()))[0]
    _assert_close(out_p, ref_out, dtype=dtype, kind="softcap output")
    _assert_close(lse_p, ref_lse, dtype=dtype, kind="softcap lse")
    _assert_close(grad_p, grad_ref, dtype=dtype, kind="softcap grad")


@pytest.mark.skipif(
    not _is_arch(9, 10, 11),
    reason="packed-QKV head_dim=256 is only supported on SM90/SM100/SM110",
)
@pytest.mark.parametrize("causal", [False, True], ids=["noncausal", "causal"])
@maybe_fake_tensor_mode(USE_FAKE_TENSOR)
def test_flash_attn_qkvpacked_hd256(causal):
    """The larger-head-dimension regression is kept off SM120."""
    torch.manual_seed(6600 + int(causal))
    # The dedicated SM100/110 hd256 backward does not support dLSE.
    _check_dense_case(
        _dense_inputs(torch.bfloat16, 256, seqlen=129, nheads=1), causal, lse_grad=False
    )
