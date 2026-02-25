from transformers.integrations.sdpa_attention import (
    sdpa_attention_forward,
    _is_torch_npu_available,
    use_gqa_in_sdpa,
    repeat_kv,
    logger,
)

import math

import torch

from gstruct.ops import clean_inner_dim as ttl_clean_inner_dim
from gstruct.ops import triu as ttl_triu
from gstruct.ops import baddbmm as ttl_baddbmm
from gstruct.ops import bmm as ttl_bmm
from gstruct.ops import activation as ttl_activation
from gstruct.ops import transpose_inner_dim as ttl_transpose_inner_dim
from gstruct import GroqBuffer, GroqMLIR, TiledMemref
from compile_ttl import compile_ttl_model
from gstruct.runner import GroqRunner
from gstruct import gstruct
import numpy as np
from typing import Union, cast

from gstruct.constants import VECTOR_SIZE, conv_np_dtype_to_dtypes, dtypes_to_np

from ttl_pytorch_utils import get_split_num
from gstruct.tiled_memref import move_dimension_to_position
from gstruct import vxm_ops


TORCH_TO_NUMPY = {
    torch.float16: np.float16,
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.uint8: np.uint8,
    torch.bool: np.bool_,
}


def sdpa_attention_forward_ttl(
    module: torch.nn.Module,
    query: Union[torch.Tensor, GroqMLIR],
    key: Union[torch.Tensor, GroqMLIR],
    value: Union[torch.Tensor, GroqMLIR],
    attention_mask: Union[torch.Tensor, GroqMLIR, None],
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:

    if isinstance(query, torch.Tensor):
        query_np = query.numpy()

        split_num = get_split_num(query_np.shape)
        tinput = TiledMemref(
            query_np.shape,
            conv_np_dtype_to_dtypes(query_np.dtype),
            ends=(split_num * VECTOR_SIZE - query_np.shape[-1],),
        )
        query_buffer = GroqBuffer.input("query", tinput)

    elif isinstance(query, GroqMLIR):
        query_buffer = query
    else:
        assert False, "query must be a torch.Tensor or GroqMLIR"

    if isinstance(key, torch.Tensor):
        key_np = key.numpy()
        split_num = get_split_num(key_np.shape)
        tinput = TiledMemref(
            key_np.shape,
            conv_np_dtype_to_dtypes(key_np.dtype),
            ends=(split_num * VECTOR_SIZE - key_np.shape[-1],),
        )
        key_buffer = GroqBuffer.input("key", tinput)

    elif isinstance(key, GroqMLIR):
        key_buffer = key
    else:
        assert False, "key must be a torch.Tensor or GroqMLIR"

    if isinstance(value, torch.Tensor):
        value_np = value.numpy()
        split_num = get_split_num(value_np.shape)
        tinput = TiledMemref(
            value_np.shape,
            conv_np_dtype_to_dtypes(value_np.dtype),
            ends=(split_num * VECTOR_SIZE - value_np.shape[-1],),
        )
        value_buffer = GroqBuffer.input("value", tinput)
    elif isinstance(value, GroqMLIR):
        value_buffer = value
    else:
        assert False, "value must be a torch.Tensor or GroqMLIR"

    query_shape = query_buffer.out_tensor_shape
    query_shape = [cast(int, dim) for dim in query_shape]

    key_shape = key_buffer.out_tensor_shape
    key_shape = [cast(int, dim) for dim in key_shape]

    if kwargs.get("output_attentions", False):
        logger.warning_once(
            "`sdpa` attention does not support `output_attentions=True`."
            " Please set your attention to `eager` if you want any of these features."
        )
    sdpa_kwargs = {}
    if hasattr(module, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key):
            key = repeat_kv(key, module.num_key_value_groups)
            value = repeat_kv(value, module.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Instead of relying on the value set in the module directly, we use the is_causal passed in kwargs if it is presented
    is_causal = (
        is_causal if is_causal is not None else getattr(module, "is_causal", True)
    )

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query_shape[2] > 1 and attention_mask is None and is_causal

    # Shapes (e.g. query.shape[2]) are tensors during jit tracing, resulting in `is_causal` being a tensor.
    # We convert it to a bool for the SDPA kernel that only accepts bools.
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # When `is_causal = False` and the `attention_mask` is not of boolean type, the Ascend NPU's SDPA interface cannot utilize the FlashAttentionScore operator，
    # and falls back to small-operator concatenation. To invoke the FlashAttentionScore, the attention_mask must be converted to boolean type.
    # This adaptation ensures the `attention_mask` meets the requirement for using FlashAttentionScore.
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            # Convert to boolean type, making sdpa to force call FlashAttentionScore to improve performance.
            attention_mask = torch.logical_not(attention_mask.bool()).to(query.device)

    # SDPA's Flash Attention (and cuDNN) kernels rely on the `is_causal` flag. However, there are certain conditions:
    # - Not in decoding phase (otherwise we want full attention on the single query token)
    # - Attention mask is not to be provided (even if it is a causal pattern)
    # - Internally, we marked this as compatible with causal, i.e. it is a decoder attention type
    #
    # Quirks on the conditionals:
    # - We avoid inline passing this to the SDPA function directly to support both torch.compile's dynamic shapes and
    #   full graph options. Otherwise, dynamic shapes are prevented from compiling.
    # - It is important to check first for the shape, otherwise compile will fail with
    #   `argument 'is_causal' must be bool, not SymBool`.
    is_causal = query_shape[2] > 1 and attention_mask is None and is_causal

    L = query_shape[-2]
    S = key_shape[-2]

    scale_factor = 1 / math.sqrt(query_shape[-1]) if scaling is None else scaling

    if is_causal:
        assert attention_mask is None
        attn_bias_np = np.full((1, S), float("-inf"), dtype=np.float32)
        attn_bias_ttl = GroqBuffer.constant(value=attn_bias_np)
        attn_bias_ttl = gstruct.concat([attn_bias_ttl] * L, concatAxis=0)
        attn_bias_ttl = ttl_triu(attn_bias_ttl, diagonal=1)

    else:
        attn_bias_np = np.zeros((1, S), dtype=TORCH_TO_NUMPY[query.dtype])
        attn_bias_ttl = GroqBuffer.constant(value=attn_bias_np)
        attn_bias_ttl = gstruct.concat([attn_bias_ttl] * L, concatAxis=0)

    if attention_mask is not None:
        assert False, "attention_mask is not supported in TTL"
        if attention_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
        else:
            attn_bias = attention_mask + attn_bias

    if sdpa_kwargs.get("enable_gqa", False):
        assert False, "enable_gqa is not supported in TTL"
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_output = ttl_baddbmm(
        attn_bias_ttl, query_buffer, key_buffer, alpha=scale_factor
    )

    attn_output = ttl_activation(attn_output, activation_function="Softmax", dim=-1)

    print("attn_output: ", attn_output.out_tmemrefs[0])

    assert dropout == 0.0, "dropout is not supported in TTL"

    value_buffer_transposed = ttl_transpose_inner_dim(value_buffer, dim=2)
    value_buffer_transposed = ttl_clean_inner_dim(value_buffer_transposed)

    attn_output = ttl_bmm(attn_output, value_buffer_transposed)

    n = len(attn_output.out_tensor_shape)
    permutation = move_dimension_to_position(n, source_dim=1, target_pos=2)
    attn_output = gstruct.transpose(attn_output, permutation)

    return attn_output, None

    attn_output_tmp = attn_output

    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()

    print("attn_output: ", attn_output_tmp.shape)
    print("attn_output_sdpa: ", attn_output.shape)

    print(attn_output_tmp)
    print(attn_output)

    assert torch.allclose(attn_output_tmp, attn_output)

    return attn_output, None

    return sdpa_attention_forward_orig(
        module, query, key, value, attention_mask, dropout, scaling, is_causal, **kwargs
    )
