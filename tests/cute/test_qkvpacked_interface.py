"""CPU-only contract tests for the packed-QKV FA4 public interfaces.

The real FA4 implementation is a CUDA/CuTeDSL kernel.  These tests deliberately
replace dispatch (and, for autograd tests, the two kernel entry points) so that
input validation and the Python autograd plumbing can be tested without a GPU.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from flash_attn.cute import input_validation, interface

DTYPE = torch.float16


def dense_qkv(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(2, 3, 3, 2, 8, dtype=DTYPE, requires_grad=requires_grad)


def varlen_qkv(*, requires_grad: bool = False) -> torch.Tensor:
    return torch.randn(6, 3, 2, 8, dtype=DTYPE, requires_grad=requires_grad)


def cu_seqlens() -> torch.Tensor:
    return torch.tensor([0, 3, 6], dtype=torch.int32)


@pytest.fixture
def dispatch(monkeypatch):
    """Patch the single packed autograd entry point used by both public APIs."""

    result = object()
    calls = []

    def apply(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(interface, "is_fake_mode", lambda: True)
    monkeypatch.setattr(interface.FlashAttnQKVPackedFunc, "apply", apply)
    return result, calls


@pytest.mark.parametrize(
    "api,varlen",
    [
        ("flash_attn_qkvpacked_func", False),
        ("flash_attn_varlen_qkvpacked_func", True),
    ],
)
def test_public_packed_apis_use_one_shared_dispatch(
    api, varlen, dispatch
):
    fn = getattr(interface, api)
    qkv = varlen_qkv() if varlen else dense_qkv()
    cu = cu_seqlens() if varlen else None
    args = (qkv, cu, 3) if varlen else (qkv,)
    result = fn(*args, softmax_scale=0.125, causal=True, window_size=(2, 1))

    assert result is dispatch[0]
    assert len(dispatch[1]) == 1
    assert len(dispatch[1][0]) == 20
    dispatched = dispatch[1][0]
    assert dispatched[0] is qkv
    assert dispatched[1] is cu
    assert dispatched[2] == (3 if varlen else None)
    assert dispatched[3:] == (
        0.125,
        True,
        (2, 1),
        None,
        0.0,
        1,
        False,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        False,
        None,
        None,
    )


def test_packed_validation_checks_type_rank_and_size_before_delegation(monkeypatch):
    calls = []

    def delegate(q, k, v, **kwargs):
        calls.append((q, k, v, kwargs))

    monkeypatch.setattr(input_validation, "validate_attention_inputs", delegate)
    qkv = varlen_qkv()
    cu = cu_seqlens()
    q, k, v = input_validation.validate_qkvpacked_inputs(
        qkv, cu_seqlens=cu, allow_cpu=True
    )
    expected_q, expected_k, expected_v = qkv.unbind(dim=-3)
    for actual, expected in zip((q, k, v), (expected_q, expected_k, expected_v)):
        assert actual.data_ptr() == expected.data_ptr()
        assert actual.stride() == expected.stride()
    assert calls and calls[0][0] is q
    assert calls[0][1] is k
    assert calls[0][2] is v
    assert calls[0][3]["cu_seqlens_q"] is cu
    assert calls[0][3]["cu_seqlens_k"] is cu
    assert calls[0][3]["allow_cpu"] is True

    calls.clear()
    for bad_qkv in (
        [],
        torch.empty(3, 2, 8, dtype=DTYPE),
        torch.empty(2, 4, 2, 8, dtype=DTYPE),
    ):
        with pytest.raises((TypeError, ValueError)):
            input_validation.validate_qkvpacked_inputs(
                bad_qkv, cu_seqlens=cu, allow_cpu=True
            )
    assert not calls


@pytest.mark.parametrize(
    "bad_qkv,error,match",
    [
        ([], TypeError, "qkv must be a torch.Tensor"),
        (torch.empty(2, 3, 2, 8, dtype=DTYPE), ValueError, "shape"),
        (torch.empty(2, 3, 4, 2, 8, dtype=DTYPE), ValueError, "size 3"),
        (torch.empty(2, 3, 3, 0, 8, dtype=DTYPE), ValueError, "positive head count"),
        (torch.empty(2, 3, 3, 2, 0, dtype=DTYPE), ValueError, "positive head count"),
        (torch.empty(2, 3, 3, 2, 8, dtype=torch.float32), TypeError, "supported dtype"),
    ],
)
def test_invalid_dense_packed_qkv(bad_qkv, error, match, dispatch):
    with pytest.raises(error, match=match):
        interface.flash_attn_qkvpacked_func(bad_qkv)
    assert not dispatch[1]


def test_concatenated_head_gqa_validation_and_dispatch(dispatch):
    qkv = torch.randn(2, 3, 8, 8, dtype=DTYPE)
    result = interface.flash_attn_qkvpacked_func(qkv, num_heads_q=4)
    assert result is dispatch[0]
    args = dispatch[1][0]
    assert len(args) == 20
    assert args[-1] == 4
    with pytest.raises(ValueError, match="positive even"):
        interface.flash_attn_qkvpacked_func(torch.randn(2, 3, 7, 8, dtype=DTYPE), num_heads_q=4)
    with pytest.raises(ValueError, match="num_heads_q"):
        interface.flash_attn_qkvpacked_func(qkv, num_heads_q=0)


def test_seqused_validation_and_dispatch(dispatch):
    qkv = dense_qkv()
    used = torch.tensor([2, 1], dtype=torch.int32)
    result = interface.flash_attn_qkvpacked_func(qkv, seqused=used)
    assert result is dispatch[0]
    args = dispatch[1][0]
    assert args[18] is used
    assert args[19] is None
    result = interface.flash_attn_qkvpacked_func(
        torch.randn(2, 3, 6, 8, dtype=DTYPE), seqused=used, num_heads_q=2
    )
    assert result is dispatch[0]
    assert dispatch[1][-1][18] is used
    assert dispatch[1][-1][19] == 2


@pytest.mark.parametrize(
    "replacement,error,match",
    [
        (None, TypeError, "cu_seqlens"),
        ([0, 3, 6], TypeError, "torch.Tensor"),
        (torch.zeros(3, dtype=torch.int64), TypeError, "torch.int32"),
        (torch.zeros(1, 3, dtype=torch.int32), ValueError, "1 dimensions"),
        (torch.zeros(1, dtype=torch.int32), ValueError, "at least two"),
        (torch.arange(5, dtype=torch.int32)[::2], ValueError, "contiguous"),
        (torch.empty(3, dtype=torch.int32, device="meta"), ValueError, "same device"),
    ],
)
def test_invalid_varlen_packed_sequence_metadata(replacement, error, match, dispatch):
    with pytest.raises(error, match=match):
        interface.flash_attn_varlen_qkvpacked_func(varlen_qkv(), replacement, 3)
    assert not dispatch[1]


@pytest.mark.parametrize(
    "max_seqlen",
    [None, -1, True, False, 1.0, torch.tensor(3)],
)
def test_varlen_max_seqlen_is_a_nonnegative_host_int(max_seqlen, dispatch):
    with pytest.raises((TypeError, ValueError), match="max_seqlen"):
        interface.flash_attn_varlen_qkvpacked_func(
            varlen_qkv(), cu_seqlens(), max_seqlen
        )
    assert not dispatch[1]


def test_zero_max_seqlen_is_allowed_and_none_cu_is_rejected(dispatch):
    qkv = torch.empty(0, 3, 2, 8, dtype=DTYPE)
    assert (
        interface.flash_attn_varlen_qkvpacked_func(qkv, torch.zeros(3, dtype=torch.int32), 0)
        is dispatch[0]
    )
    # None must not accidentally route a dense-shaped input through this API.
    for qkv in (varlen_qkv(), dense_qkv()):
        with pytest.raises(TypeError, match="cu_seqlens"):
            interface.flash_attn_varlen_qkvpacked_func(qkv, None, 3)
    assert len(dispatch[1]) == 1


def test_cpu_inputs_are_allowed_only_in_fake_mode(monkeypatch, dispatch):
    monkeypatch.setattr(interface, "is_fake_mode", lambda: False)
    with pytest.raises(ValueError, match="CUDA device"):
        interface.flash_attn_qkvpacked_func(dense_qkv())
    with pytest.raises(ValueError, match="CUDA device"):
        interface.flash_attn_varlen_qkvpacked_func(varlen_qkv(), cu_seqlens(), 3)
    assert not dispatch[1]


def test_varlen_api_rejects_block_sparse_kwargs(dispatch):
    qkv, cu = varlen_qkv(), cu_seqlens()
    with pytest.raises(TypeError):
        interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu, 3, block_sparse_tensors=object()
        )
    with pytest.raises(TypeError):
        interface.flash_attn_varlen_qkvpacked_func(
            qkv, cu, 3, block_sparse_tensors_bwd=object()
        )
    assert not dispatch[1]


def test_packed_functions_are_public_exports():
    from flash_attn import cute

    assert cute.flash_attn_qkvpacked_func is interface.flash_attn_qkvpacked_func
    assert (
        cute.flash_attn_varlen_qkvpacked_func
        is interface.flash_attn_varlen_qkvpacked_func
    )
    assert "flash_attn_qkvpacked_func" in cute.__all__
    assert "flash_attn_varlen_qkvpacked_func" in cute.__all__


def test_packed_unbind_is_zero_copy(monkeypatch):
    seen = {}

    def fake_fwd(q, k, v, **kwargs):
        seen["qkv"] = (q, k, v)
        return q.clone(), None, None, None, None

    monkeypatch.setattr(interface, "is_fake_mode", lambda: True)
    monkeypatch.setattr(interface, "_flash_attn_fwd", fake_fwd)
    qkv = dense_qkv()
    interface.flash_attn_qkvpacked_func(qkv)

    q, k, v = seen["qkv"]
    base_ptr = qkv.untyped_storage().data_ptr()
    assert all(t.untyped_storage().data_ptr() == base_ptr for t in (q, k, v))
    assert all(t._base is not None for t in (q, k, v))
    assert q.data_ptr() == qkv.data_ptr()
    assert k.data_ptr() > q.data_ptr()
    assert v.data_ptr() > k.data_ptr()


class _KernelMocks:
    def __init__(self, monkeypatch):
        self.fwd_calls = []
        self.bwd_calls = []

        def fwd(q, k, v, **kwargs):
            self.fwd_calls.append((q, k, v, kwargs))
            # A grad-enabled forward allocates LSE even when the public
            # return_lse flag is false; the autograd wrapper suppresses dlse
            # in that mode while preserving the (out, lse) return convention.
            lse_shape = (q.shape[0], q.shape[2], q.shape[1]) if q.ndim == 4 else (q.shape[1], q.shape[0])
            lse = torch.ones(lse_shape, dtype=torch.float32, device=q.device)
            return q.clone(), lse, None, None, None

        def bwd(q, k, v, out, dout, lse, **kwargs):
            self.bwd_calls.append(
                {"q": q, "k": k, "v": v, "out": out, "dout": dout, "lse": lse, **kwargs}
            )
            dq, dk, dv = kwargs["dq"], kwargs["dk"], kwargs["dv"]
            dq.fill_(1)
            dk.fill_(2)
            dv.fill_(3)
            if kwargs.get("learnable_sink") is not None:
                dsink = torch.full_like(kwargs["learnable_sink"], 7)
                return dq, dk, dv, dsink
            return dq, dk, dv

        monkeypatch.setattr(interface, "is_fake_mode", lambda: True)
        monkeypatch.setattr(interface, "_flash_attn_fwd", fwd)
        monkeypatch.setattr(interface, "_flash_attn_bwd", bwd)


def test_mocked_autograd_uses_one_packed_contiguous_grad_allocation(monkeypatch):
    mocks = _KernelMocks(monkeypatch)
    qkv = dense_qkv(requires_grad=True)
    out, lse = interface.flash_attn_qkvpacked_func(qkv)
    assert lse is not None
    assert qkv.requires_grad
    assert all(t.requires_grad for t in mocks.fwd_calls[0][:3])
    out.sum().backward()

    assert len(mocks.bwd_calls) == 1
    call = mocks.bwd_calls[0]
    assert call["q"].requires_grad
    assert call["k"].requires_grad
    assert call["v"].requires_grad
    assert call["dlse"] is None
    assert call["dout"] is not None
    assert torch.count_nonzero(call["dout"]) > 0

    grads = (call["dq"], call["dk"], call["dv"])
    base_ptr = grads[0].untyped_storage().data_ptr()
    assert all(t.untyped_storage().data_ptr() == base_ptr for t in grads)
    assert len({t.data_ptr() for t in grads}) == 3
    assert qkv.grad is not None
    assert qkv.grad.is_contiguous()
    assert tuple(qkv.grad.shape) == tuple(qkv.shape)
    assert torch.equal(qkv.grad.unbind(dim=-3)[0], torch.ones_like(qkv.grad.unbind(dim=-3)[0]))
    assert torch.equal(qkv.grad.unbind(dim=-3)[1], torch.full_like(qkv.grad.unbind(dim=-3)[1], 2))
    assert torch.equal(qkv.grad.unbind(dim=-3)[2], torch.full_like(qkv.grad.unbind(dim=-3)[2], 3))


def test_return_lse_preserves_dlse_and_zero_materializes_dout(monkeypatch):
    mocks = _KernelMocks(monkeypatch)
    qkv = dense_qkv(requires_grad=True)
    out, lse = interface.flash_attn_qkvpacked_func(qkv, return_lse=True)
    assert out.requires_grad
    assert lse is not None

    # Only LSE participates in the loss: autograd must pass dout=None to the
    # custom function, which the implementation converts to a zero dout.
    lse.sum().backward()
    assert len(mocks.bwd_calls) == 1
    call = mocks.bwd_calls[0]
    assert call["dout"] is not None
    assert torch.count_nonzero(call["dout"]) == 0
    assert call["dlse"] is not None
    assert torch.count_nonzero(call["dlse"]) > 0


def test_sink_gradient_is_returned_in_the_sink_argument_slot(monkeypatch):
    mocks = _KernelMocks(monkeypatch)
    qkv = dense_qkv(requires_grad=True)
    sink = torch.zeros(2, dtype=DTYPE, requires_grad=True)
    out, lse = interface.flash_attn_qkvpacked_func(
        qkv, learnable_sink=sink, return_lse=True
    )
    (out.sum() + lse.sum()).backward()

    assert len(mocks.bwd_calls) == 1
    assert sink.grad is not None
    assert torch.equal(sink.grad, torch.full_like(sink, 7))
    assert qkv.requires_grad


def test_modifiers_auxiliary_and_sparse_metadata_reach_both_kernels(monkeypatch):
    mocks = _KernelMocks(monkeypatch)
    qkv = dense_qkv(requires_grad=True)
    aux = [torch.ones(2)]
    score_mod = object()
    score_mod_bwd = object()
    mask_mod = object()
    sparse_fwd = SimpleNamespace(name="fwd")
    sparse_bwd = SimpleNamespace(name="bwd")
    aux_scalars = (1.25, 7)

    out, lse = interface.flash_attn_qkvpacked_func(
        qkv,
        score_mod=score_mod,
        score_mod_bwd=score_mod_bwd,
        mask_mod=mask_mod,
        aux_tensors=aux,
        aux_scalars=aux_scalars,
        block_sparse_tensors=sparse_fwd,
        block_sparse_tensors_bwd=sparse_bwd,
        return_lse=True,
    )
    (out.sum() + lse.sum()).backward()

    fwd_kwargs = mocks.fwd_calls[0][3]
    assert fwd_kwargs["score_mod"] is score_mod
    assert fwd_kwargs["mask_mod"] is mask_mod
    assert fwd_kwargs["aux_tensors"] is aux
    assert fwd_kwargs["aux_scalars"] == aux_scalars
    assert fwd_kwargs["block_sparse_tensors"] is sparse_fwd
    assert fwd_kwargs["return_lse"] is True

    bwd = mocks.bwd_calls[0]
    assert bwd["score_mod"] is score_mod
    assert bwd["score_mod_bwd"] is score_mod_bwd
    assert bwd["mask_mod"] is mask_mod
    assert len(bwd["aux_tensors"]) == 1
    assert bwd["aux_tensors"][0] is aux[0]
    assert bwd["aux_scalars"] == aux_scalars
    assert bwd["block_sparse_tensors"] is sparse_bwd
    assert bwd["dlse"] is not None


def test_varlen_metadata_reaches_forward_and_backward(monkeypatch):
    mocks = _KernelMocks(monkeypatch)
    qkv, cu = varlen_qkv(requires_grad=True), cu_seqlens()
    out, lse = interface.flash_attn_varlen_qkvpacked_func(
        qkv, cu, 5, return_lse=True
    )
    (out.sum() + lse.sum()).backward()

    fwd_kwargs = mocks.fwd_calls[0][3]
    assert fwd_kwargs["cu_seqlens_q"] is cu
    assert fwd_kwargs["cu_seqlens_k"] is cu
    assert fwd_kwargs["max_seqlen_q"] == 5
    assert fwd_kwargs["max_seqlen_k"] == 5

    bwd = mocks.bwd_calls[0]
    assert bwd["cu_seqlens_q"] is cu
    assert bwd["cu_seqlens_k"] is cu
    assert bwd["max_seqlen_q"] == 5
    assert bwd["max_seqlen_k"] == 5


@pytest.mark.parametrize("varlen", [False, True])
def test_unrequested_lse_gradient_is_not_forwarded(monkeypatch, varlen):
    mocks = _KernelMocks(monkeypatch)
    qkv = varlen_qkv(requires_grad=True) if varlen else dense_qkv(requires_grad=True)
    args = (qkv, cu_seqlens(), 3) if varlen else (qkv,)
    api = interface.flash_attn_varlen_qkvpacked_func if varlen else interface.flash_attn_qkvpacked_func
    out, lse = api(*args, return_lse=False)
    (out.sum() + lse.sum()).backward()
    assert len(mocks.bwd_calls) == 1
    assert mocks.bwd_calls[0]["dlse"] is None


@pytest.mark.parametrize("varlen", [False, True])
def test_sink_only_requires_grad(monkeypatch, varlen):
    mocks = _KernelMocks(monkeypatch)
    qkv = varlen_qkv() if varlen else dense_qkv()
    sink = torch.zeros(2, dtype=torch.float32, requires_grad=True)
    api = interface.flash_attn_varlen_qkvpacked_func if varlen else interface.flash_attn_qkvpacked_func
    args = (qkv, cu_seqlens(), 3) if varlen else (qkv,)
    out, _ = api(*args, learnable_sink=sink)
    out.sum().backward()
    assert len(mocks.bwd_calls) == 1
    torch.testing.assert_close(sink.grad, torch.full_like(sink, 7))
    assert qkv.grad is None


def test_sparse_training_requires_backward_metadata(dispatch):
    qkv = dense_qkv(requires_grad=True)
    sparse = SimpleNamespace(name="sparse")
    with pytest.raises(ValueError, match="block_sparse_tensors_bwd"):
        interface.flash_attn_qkvpacked_func(qkv, block_sparse_tensors=sparse)
    assert not dispatch[1]
    with torch.no_grad():
        assert interface.flash_attn_qkvpacked_func(qkv, block_sparse_tensors=sparse) is dispatch[0]


@pytest.mark.parametrize("varlen", [False, True])
def test_noncontiguous_input_returns_canonical_gradient(monkeypatch, varlen):
    mocks = _KernelMocks(monkeypatch)
    shape = (12, 3, 2, 8) if varlen else (2, 6, 3, 2, 8)
    storage = torch.randn(shape, dtype=DTYPE)
    qkv = (storage[::2] if varlen else storage[:, ::2]).detach().requires_grad_()
    api = interface.flash_attn_varlen_qkvpacked_func if varlen else interface.flash_attn_qkvpacked_func
    args = (qkv, cu_seqlens(), 3) if varlen else (qkv,)
    out, _ = api(*args)
    grad, = torch.autograd.grad(out.sum(), qkv, retain_graph=True)
    grad_again, = torch.autograd.grad(out.sum(), qkv)
    assert grad.is_contiguous()
    assert grad.shape == qkv.shape
    torch.testing.assert_close(grad, grad_again)
    assert len(mocks.bwd_calls) == 2
    assert mocks.bwd_calls[0]["dq"].untyped_storage().data_ptr() == grad.untyped_storage().data_ptr()


@pytest.mark.parametrize("varlen", [False, True])
def test_fake_tensor_metadata_validation(monkeypatch, dispatch, varlen):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flash_attn.cute.testing import is_fake_mode

    monkeypatch.setattr(interface, "is_fake_mode", is_fake_mode)
    with FakeTensorMode():
        qkv = varlen_qkv() if varlen else dense_qkv()
        api = interface.flash_attn_varlen_qkvpacked_func if varlen else interface.flash_attn_qkvpacked_func
        args = (qkv, cu_seqlens(), 3) if varlen else (qkv,)
        assert api(*args) is dispatch[0]
