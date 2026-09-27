# FlashAttention-4 (CuTeDSL)

FlashAttention-4 is a CuTeDSL-based implementation of FlashAttention for Hopper and Blackwell GPUs.

## Installation

```sh
pip install flash-attn-4
```

If you're on CUDA 13, install with the `cu13` extra for best performance:

```sh
pip install "flash-attn-4[cu13]"
```

## Usage

```python
from flash_attn.cute import (
    flash_attn_func,
    flash_attn_varlen_func,
    flash_attn_qkvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)

out, lse = flash_attn_func(q, k, v, causal=True)

# Canonical MHA packed QKV: [batch, seqlen, 3, heads, head_dim]
out, lse = flash_attn_qkvpacked_func(qkv, causal=True)

# Dense storage with per-example effective lengths.
out, lse = flash_attn_qkvpacked_func(qkv, seqused=seqused, causal=True)

# Concatenated-head GQA/MQA: [batch, seqlen, Hq + 2 * Hkv, head_dim]
out, lse = flash_attn_qkvpacked_func(qkv_gqa, num_heads_q=Hq, causal=True)

# Variable-length canonical packed QKV: [total_tokens, 3, heads, head_dim]
out, lse = flash_attn_varlen_qkvpacked_func(
    qkv, cu_seqlens, max_seqlen, causal=True
)

# Variable-length concatenated-head GQA/MQA:
# [total_tokens, Hq + 2 * Hkv, head_dim]
out, lse = flash_attn_varlen_qkvpacked_func(
    qkv_gqa, cu_seqlens, max_seqlen, num_heads_q=Hq, causal=True
)
```

## Development

```sh
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
pip install -e "flash_attn/cute[dev]"       # CUDA 12.x
pip install -e "flash_attn/cute[dev,cu13]"  # CUDA 13.x (e.g. B200)
pytest tests/cute/
```
