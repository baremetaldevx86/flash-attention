"""Public API validation tests; no GPU allocation or kernel compilation needed."""

import pytest
import torch

from flash_attn.cute import interface


@pytest.fixture
def dispatch(monkeypatch):
    result = object()
    calls = []

    def apply(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(interface, "is_fake_mode", lambda: True)
    monkeypatch.setattr(interface.FlashAttnFunc, "apply", apply)
    monkeypatch.setattr(interface.FlashAttnVarlenFunc, "apply", apply)
    return result, calls


def dense_inputs(q_heads=4, kv_heads=2, head_dim_v=64):
    return {
        "q": torch.empty(2, 3, q_heads, 64, dtype=torch.float16),
        "k": torch.empty(2, 5, kv_heads, 64, dtype=torch.float16),
        "v": torch.empty(2, 5, kv_heads, head_dim_v, dtype=torch.float16),
    }


@pytest.mark.parametrize(
    "api", [interface.flash_attn_func, interface.flash_attn_varlen_func]
)
@pytest.mark.parametrize(
    "name,replacement,error,message",
    [
        ("q", [], TypeError, "q must be a torch.Tensor"),
        ("v", None, TypeError, "v must be a torch.Tensor"),
        ("q", torch.empty(3, 4, 64, dtype=torch.float16), ValueError, "cu_seqlens"),
        ("v", torch.empty(2, 5, 2, 64), TypeError, "supported dtype"),
        (
            "k",
            torch.empty(2, 5, 2, 64, dtype=torch.bfloat16),
            TypeError,
            "same dtype as v",
        ),
        (
            "k",
            torch.empty(2, 6, 2, 64, dtype=torch.float16),
            ValueError,
            "matching batch/token",
        ),
        (
            "k",
            torch.empty(2, 5, 2, 32, dtype=torch.float16),
            ValueError,
            "same head dimension",
        ),
        ("q", torch.empty(3, 3, 4, 64, dtype=torch.float16), ValueError, "batch size"),
        ("q", torch.empty(2, 3, 3, 64, dtype=torch.float16), ValueError, "divisible"),
        (
            "v",
            torch.empty(2, 5, 0, 64, dtype=torch.float16),
            ValueError,
            "positive head count",
        ),
        (
            "q",
            torch.empty(2, 3, 4, 0, dtype=torch.float16),
            ValueError,
            "positive head count",
        ),
        (
            "k",
            torch.empty(2, 5, 2, 64, dtype=torch.float16, device="meta"),
            ValueError,
            "same device",
        ),
    ],
)
def test_invalid_dense_inputs(api, name, replacement, error, message, dispatch):
    inputs = dense_inputs()
    inputs[name] = replacement
    with pytest.raises(error, match=message):
        api(**inputs)
    assert not dispatch[1]


@pytest.mark.parametrize("q_heads,kv_heads", [(4, 4), (4, 2), (4, 1)])
@pytest.mark.parametrize(
    "api", [interface.flash_attn_func, interface.flash_attn_varlen_func]
)
def test_valid_mha_gqa_mqa(api, q_heads, kv_heads, dispatch):
    # V may have a different head dimension, and last-dimension views are copied downstream.
    inputs = dense_inputs(q_heads, kv_heads, head_dim_v=32)
    inputs["q"] = torch.empty(2, 3, q_heads, 128, dtype=torch.float16)[..., ::2]
    assert api(**inputs) is dispatch[0]
    assert len(dispatch[1]) == 1


def packed_inputs():
    inputs = dense_inputs()
    inputs.update(
        q=inputs["q"].flatten(0, 1),
        k=inputs["k"].flatten(0, 1),
        v=inputs["v"].flatten(0, 1),
        cu_seqlens_q=torch.tensor([0, 3, 6], dtype=torch.int32),
        cu_seqlens_k=torch.tensor([0, 5, 10], dtype=torch.int32),
    )
    return inputs


@pytest.mark.parametrize(
    "name,replacement,error,message",
    [
        ("cu_seqlens_q", [0, 3, 6], TypeError, "torch.Tensor"),
        ("cu_seqlens_q", torch.zeros(3, dtype=torch.int64), TypeError, "torch.int32"),
        (
            "cu_seqlens_q",
            torch.zeros(1, 3, dtype=torch.int32),
            ValueError,
            "1 dimensions",
        ),
        ("cu_seqlens_q", torch.zeros(1, dtype=torch.int32), ValueError, "at least two"),
        (
            "cu_seqlens_k",
            torch.zeros(4, dtype=torch.int32),
            ValueError,
            "must have shape",
        ),
        (
            "cu_seqlens_q",
            torch.zeros(6, dtype=torch.int32)[::2],
            ValueError,
            "contiguous",
        ),
        ("seqused_k", torch.zeros(3, dtype=torch.int32), ValueError, "must have shape"),
        ("seqused_q", torch.zeros(2, dtype=torch.int64), TypeError, "torch.int32"),
    ],
)
def test_invalid_sequence_metadata(name, replacement, error, message, dispatch):
    inputs = packed_inputs()
    inputs[name] = replacement
    with pytest.raises(error, match=message):
        interface.flash_attn_varlen_func(**inputs)
    assert not dispatch[1]


@pytest.mark.parametrize("pack_q,pack_k", [(True, True), (True, False), (False, True)])
def test_valid_mixed_layouts(pack_q, pack_k, dispatch):
    inputs = packed_inputs()
    dense = dense_inputs()
    if not pack_q:
        inputs["q"] = dense["q"]
        inputs.pop("cu_seqlens_q")
    if not pack_k:
        inputs.update(k=dense["k"], v=dense["v"])
        inputs.pop("cu_seqlens_k")
    assert interface.flash_attn_varlen_func(**inputs) is dispatch[0]


def test_valid_paged_kv(dispatch):
    inputs = dense_inputs()
    inputs.update(
        k=torch.empty(7, 16, 2, 64, dtype=torch.float16),
        v=torch.empty(7, 16, 2, 64, dtype=torch.float16),
        page_table=torch.zeros(2, 4, dtype=torch.int32),
    )
    assert interface.flash_attn_varlen_func(**inputs) is dispatch[0]
    inputs["cu_seqlens_k"] = torch.zeros(3, dtype=torch.int32)
    with pytest.raises(ValueError, match="cannot be combined"):
        interface.flash_attn_varlen_func(**inputs)
    inputs.pop("cu_seqlens_k")
    inputs["page_table"] = torch.zeros(3, 4, dtype=torch.int32)
    with pytest.raises(ValueError, match="page_table must have shape"):
        interface.flash_attn_varlen_func(**inputs)


def test_mla_and_sink_validation(dispatch):
    inputs = dense_inputs(head_dim_v=512)
    inputs["qv"] = torch.empty(2, 3, 4, 512, dtype=torch.float16)
    assert interface.flash_attn_func(**inputs) is dispatch[0]
    inputs.update(q=None, k=None)
    assert interface.flash_attn_func(**inputs) is dispatch[0]
    inputs["qv"] = torch.empty(2, 3, 4, 64, dtype=torch.float16)
    with pytest.raises(ValueError, match="qv and v"):
        interface.flash_attn_func(**inputs)
    inputs = dense_inputs()
    inputs["learnable_sink"] = torch.zeros(3)
    with pytest.raises(ValueError, match="learnable_sink must have shape"):
        interface.flash_attn_func(**inputs)
    inputs["learnable_sink"] = torch.zeros(4, dtype=torch.int32)
    with pytest.raises(TypeError, match="learnable_sink must have dtype"):
        interface.flash_attn_func(**inputs)
    inputs.update(q=None, qv=None)
    with pytest.raises(ValueError, match="at least one query"):
        interface.flash_attn_func(**inputs)


@pytest.mark.parametrize(
    "api", [interface.flash_attn_func, interface.flash_attn_varlen_func]
)
def test_cpu_inputs_rejected_outside_fake_mode(api, monkeypatch, dispatch):
    monkeypatch.setattr(interface, "is_fake_mode", lambda: False)
    with pytest.raises(ValueError, match="CUDA device"):
        api(**dense_inputs())
    assert not dispatch[1]


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]
)
def test_supported_dtypes(dtype, dispatch):
    inputs = {name: tensor.to(dtype) for name, tensor in dense_inputs().items()}
    assert interface.flash_attn_func(**inputs) is dispatch[0]


def test_real_fake_tensor_mode(monkeypatch, dispatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flash_attn.cute.testing import is_fake_mode

    monkeypatch.setattr(interface, "is_fake_mode", is_fake_mode)
    with FakeTensorMode():
        assert interface.flash_attn_func(**dense_inputs()) is dispatch[0]


def test_different_cuda_devices_rejected(monkeypatch, dispatch):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from flash_attn.cute.testing import is_fake_mode

    monkeypatch.setattr(interface, "is_fake_mode", is_fake_mode)
    with FakeTensorMode():
        inputs = {
            name: torch.empty(tensor.shape, dtype=tensor.dtype, device="cuda:0")
            for name, tensor in dense_inputs().items()
        }
        inputs["k"] = torch.empty(2, 5, 2, 64, dtype=torch.float16, device="cuda:1")
        with pytest.raises(ValueError, match="same device as v"):
            interface.flash_attn_func(**inputs)
    assert not dispatch[1]
