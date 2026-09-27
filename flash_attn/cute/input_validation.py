"""Metadata-only validation for the public attention APIs."""

import torch


def validate_attention_inputs(
    q,
    k,
    v,
    *,
    qv=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    page_table=None,
    learnable_sink=None,
    allow_cpu=False,
):
    """Check layouts without reading tensor values or synchronizing a device.

    CPU tensors are allowed only for the existing FakeTensor compilation path.
    Cumulative-length contents and architecture-specific limits remain the
    responsibility of the caller and kernel dispatch, respectively.
    """
    if q is None and qv is None:
        raise ValueError("Provide q or qv; at least one query tensor is required.")
    if v is None:
        raise TypeError("v must be a torch.Tensor; provide the value tensor.")

    tensors = {"q": q, "k": k, "v": v, "qv": qv}
    metadata = {
        "cu_seqlens_q": cu_seqlens_q,
        "cu_seqlens_k": cu_seqlens_k,
        "seqused_q": seqused_q,
        "seqused_k": seqused_k,
        "page_table": page_table,
        "learnable_sink": learnable_sink,
    }
    for name, tensor in {**tensors, **metadata}.items():
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")

    for name, tensor in metadata.items():
        if tensor is None or name == "learnable_sink":
            continue
        rank = 2 if name == "page_table" else 1
        if tensor.ndim != rank:
            raise ValueError(
                f"{name} must have {rank} dimensions, got shape {tuple(tensor.shape)}."
            )
        if tensor.dtype != torch.int32:
            raise TypeError(
                f"{name} must have dtype torch.int32, got {tensor.dtype}; use {name}.to(torch.int32)."
            )
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must be contiguous in its last dimension; use {name}.contiguous()."
            )
        if name.startswith("cu_seqlens") and tensor.numel() < 2:
            raise ValueError(f"{name} must contain at least two entries (batch_size + 1).")

    if page_table is not None and cu_seqlens_k is not None:
        raise ValueError(
            "page_table cannot be combined with cu_seqlens_k; use paged or packed KV storage."
        )

    for name, tensor in tensors.items():
        if tensor is None:
            continue
        packed = cu_seqlens_q is not None if name in ("q", "qv") else cu_seqlens_k is not None
        rank = 3 if packed else 4
        layout = "(total_tokens, heads, head_dim)" if packed else "(batch, seqlen, heads, head_dim)"
        if tensor.ndim != rank:
            hint = (
                ""
                if packed
                else " For packed inputs, pass the corresponding cu_seqlens tensor to flash_attn_varlen_func."
            )
            raise ValueError(f"{name} must have shape {layout}, got {tuple(tensor.shape)}.{hint}")
        if tensor.shape[-2] <= 0 or tensor.shape[-1] <= 0:
            raise ValueError(
                f"{name} must have positive head count and head dimension, got {tuple(tensor.shape)}."
            )

    supported_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)
    if v.dtype not in supported_dtypes:
        raise TypeError(
            f"v must have dtype float16, bfloat16, float8_e4m3fn, or float8_e5m2, got {v.dtype}; convert q, k, and v to a supported dtype."
        )
    for name, tensor in tensors.items():
        if tensor is not None and tensor.dtype != v.dtype:
            raise TypeError(
                f"{name} must have the same dtype as v ({v.dtype}), got {tensor.dtype}; convert all attention inputs to the same dtype."
            )

    query = q if q is not None else qv
    batch = cu_seqlens_q.numel() - 1 if cu_seqlens_q is not None else query.shape[0]
    if k is not None:
        if k.shape[:-1] != v.shape[:-1]:
            raise ValueError(
                f"k and v must have matching batch/token and KV-head dimensions, got {tuple(k.shape)} and {tuple(v.shape)}."
            )
        if q is not None and q.shape[-1] != k.shape[-1]:
            raise ValueError(
                f"q and k must have the same head dimension, got {q.shape[-1]} and {k.shape[-1]}."
            )
    if cu_seqlens_k is None and page_table is None and v.shape[0] != batch:
        raise ValueError(
            f"v batch size must match the query batch size ({batch}), got {v.shape[0]}."
        )
    if query.shape[-2] % v.shape[-2] != 0:
        raise ValueError(
            f"Query head count ({query.shape[-2]}) must be divisible by KV head count ({v.shape[-2]}); use equal counts for MHA or a multiple for GQA/MQA."
        )
    if qv is not None:
        if q is not None and qv.shape[:-1] != q.shape[:-1]:
            raise ValueError(
                f"qv must match q in every dimension except the last, got {tuple(qv.shape)} and {tuple(q.shape)}."
            )
        if qv.shape[-1] != v.shape[-1]:
            raise ValueError(
                f"qv and v must have the same head dimension, got {qv.shape[-1]} and {v.shape[-1]}."
            )
    for name, tensor in metadata.items():
        if tensor is None:
            continue
        if name == "learnable_sink":
            expected = (query.shape[-2],)
            if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32):
                raise TypeError(
                    f"learnable_sink must have dtype float16, bfloat16, or float32, got {tensor.dtype}."
                )
        elif name == "page_table":
            expected = (batch, tensor.shape[1])
        else:
            expected = (batch + 1,) if name.startswith("cu_seqlens") else (batch,)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must have shape {expected}, got {tuple(tensor.shape)}.")

    for name, tensor in {**tensors, **metadata}.items():
        if tensor is None:
            continue
        if tensor.device != v.device:
            raise ValueError(
                f"{name} must be on the same device as v ({v.device}), got {tensor.device}; move all inputs to the same CUDA device."
            )
        if not allow_cpu and not tensor.is_cuda:
            raise ValueError(
                f"{name} must be on a CUDA device, got {tensor.device}; move the inputs with .to('cuda')."
            )


def validate_qkvpacked_inputs(
    qkv,
    *,
    cu_seqlens=None,
    seqused=None,
    num_heads_q=None,
    learnable_sink=None,
    allow_cpu=False,
):
    """Validate packed self-attention inputs and return Q/K/V views.

    Canonical MHA packing uses ``(B, S, 3, H, D)`` or
    ``(T, 3, H, D)``.  Concatenated-head GQA/MQA packing uses
    ``(B, S, Hq + 2 * Hkv, D)`` or ``(T, Hq + 2 * Hkv, D)`` and requires
    ``num_heads_q=Hq``.  ``seqused`` is supported with dense storage and is
    shared by Q and K.  Validation remains metadata-only: sequence metadata
    contents are intentionally not read on the host.
    """
    if not isinstance(qkv, torch.Tensor):
        raise TypeError(f"qkv must be a torch.Tensor, got {type(qkv).__name__}.")
    if cu_seqlens is not None and seqused is not None:
        raise ValueError("cu_seqlens and seqused cannot be combined for packed attention.")
    if num_heads_q is not None and (
        isinstance(num_heads_q, bool) or not isinstance(num_heads_q, int) or num_heads_q <= 0
    ):
        raise ValueError(f"num_heads_q must be a positive host integer, got {num_heads_q!r}.")

    concatenated = num_heads_q is not None
    varlen = cu_seqlens is not None
    if concatenated:
        expected_rank = 3 if varlen else 4
        layout = (
            "(total_tokens, heads_q + 2 * heads_kv, head_dim)"
            if varlen
            else "(batch, seqlen, heads_q + 2 * heads_kv, head_dim)"
        )
        if qkv.ndim != expected_rank:
            raise ValueError(f"qkv must have shape {layout}, got {tuple(qkv.shape)}.")
        total_heads = qkv.shape[-2]
        remaining_heads = total_heads - num_heads_q
        if remaining_heads <= 0 or remaining_heads % 2 != 0:
            raise ValueError(
                "qkv concatenated-head layout requires total_heads - num_heads_q "
                "to be a positive even number."
            )
        num_heads_kv = remaining_heads // 2
        q = qkv[..., :num_heads_q, :]
        k = qkv[..., num_heads_q : num_heads_q + num_heads_kv, :]
        v = qkv[..., num_heads_q + num_heads_kv :, :]
    else:
        expected_rank = 4 if varlen else 5
        layout = (
            "(total_tokens, 3, heads, head_dim)"
            if varlen
            else "(batch, seqlen, 3, heads, head_dim)"
        )
        if qkv.ndim != expected_rank:
            raise ValueError(f"qkv must have shape {layout}, got {tuple(qkv.shape)}.")
        if qkv.shape[-3] != 3:
            raise ValueError(
                f"qkv's packed dimension must have size 3 at axis {-3}, got {qkv.shape[-3]}."
            )
        q, k, v = qkv.unbind(dim=-3)

    validate_attention_inputs(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        seqused_q=seqused,
        seqused_k=seqused,
        learnable_sink=learnable_sink,
        allow_cpu=allow_cpu,
    )
    return q, k, v
