from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel,
    GPT2Block,
    GPT2Attention,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import (
    create_causal_mask,
)

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import numpy as np
from typing import Optional, Callable, cast
import torch

from gstruct import gstruct, TiledMemref, GroqBuffer, dtypes
from gstruct.constants import VECTOR_SIZE, conv_np_dtype_to_dtypes
from gstruct.ops import baddbmm as ttl_baddbmm
from gstruct.runner import GroqRunner
from compile_ttl import compile_ttl_model


def get_split_num(input_shape: tuple[int, ...], inner_axis: int = -1):

    return (input_shape[inner_axis] + VECTOR_SIZE - 1) // VECTOR_SIZE


class GPT2AttentionTTL(GPT2Attention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> tuple[torch.Tensor | tuple[torch.Tensor], ...]:

        is_cross_attention = encoder_hidden_states is not None

        if past_key_values is not None:
            if isinstance(past_key_values, EncoderDecoderCache):
                is_updated = past_key_values.is_updated.get(self.layer_idx)
                if is_cross_attention:
                    # after the first generated id, we can subsequently re-use all key/value_layer from cache
                    curr_past_key_values = past_key_values.cross_attention_cache
                else:
                    curr_past_key_values = past_key_values.self_attention_cache
            else:
                curr_past_key_values = past_key_values

        print("jjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjjj")
        print(curr_past_key_values)
        print(is_cross_attention)

        assert (
            is_cross_attention == False
        ), "is_cross_attention==True is not implemented yet"
        if is_cross_attention:
            if not hasattr(self, "q_attn"):
                raise ValueError(
                    "If class is used as cross attention, the weights `q_attn` have to be defined. "
                    "Please make sure to instantiate class with `GPT2Attention(..., is_cross_attention=True)`."
                )
            query_states = self.q_attn(hidden_states)
            attention_mask = encoder_attention_mask

            # Try to get key/value states from cache if possible
            if past_key_values is not None and is_updated:
                key_states = curr_past_key_values.layers[self.layer_idx].keys
                value_states = curr_past_key_values.layers[self.layer_idx].values
            else:
                key_states, value_states = self.c_attn(encoder_hidden_states).split(
                    self.split_size, dim=2
                )
                shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
                key_states = key_states.view(shape_kv).transpose(1, 2)
                value_states = value_states.view(shape_kv).transpose(1, 2)
        else:
            # query_states, key_states, value_states = self.c_attn(hidden_states).split(
            #     self.split_size, dim=2
            # )

            print("hidden_states: ", hidden_states.shape)
            print("weight:", self.c_attn.weight.shape)

            weight_shape = self.c_attn.weight.shape
            weight_shape = (1, *weight_shape[:-1], 3, -1)
            weights = self.c_attn.weight.view(weight_shape)
            weights = weights.transpose(-2, 1).contiguous()

            bias_shape = self.c_attn.bias.shape
            bias_shape = (1, *bias_shape[:-1], 1, 3, -1)
            bias = self.c_attn.bias.view(bias_shape)
            bias = bias.transpose(-2, 1).contiguous()

            weight_transposed = (
                weights.numpy().swapaxes(-1, -2).copy().astype(np.float16)
            )
            weight_mt = GroqBuffer.constant(value=weight_transposed)

            bias = bias.numpy()
            bias_mt = GroqBuffer.constant(value=bias)

            print("weight_buffer: ", weight_mt.out_tmemrefs[0])
            print("bias_buffer: ", bias_mt.out_tmemrefs[0])

            print("weight:", weights.shape)

            hidden_states_np = hidden_states.numpy()
            hidden_states_np = np.expand_dims(hidden_states_np, -3)

            split_num = get_split_num(hidden_states_np.shape)
            tinput = TiledMemref(
                hidden_states_np.shape,
                conv_np_dtype_to_dtypes(hidden_states_np.dtype),
                ends=(split_num * VECTOR_SIZE - hidden_states_np.shape[-1],),
            )

            input_tensor_name = "image"
            hidden_states_buffer = GroqBuffer.input(input_tensor_name, tinput)

            print("bias_mt: ", bias_mt.out_tmemrefs[0])
            print("hidden_states_buffer: ", hidden_states_buffer.out_tmemrefs[0])
            print("weight_mt: ", weight_mt.out_tmemrefs[0])

            x = ttl_baddbmm(
                bias_mt,
                hidden_states_buffer,
                weight_mt,
            )

            print("x: ", x.out_tmemrefs[0])
            vectors_x = x.out_vector_shape
            print("vectors_x: ", vectors_x)

            assert (
                vectors_x[1] % 3 == 0
            ), "vectors_x[1] must be divisible by 3 (query, key, value)"
            static_sizes = [cast(int, dim) for dim in vectors_x]
            static_sizes[1] = static_sizes[1] // 3
            print("vectors: ", static_sizes)

            offset = [0] * len(vectors_x)
            strides = [1] * len(vectors_x)
            query_states = gstruct.subview(
                x,
                static_offsets=offset,
                static_sizes=static_sizes,
                static_strides=strides,
            )

            query_states = gstruct.reshape(
                query_states,
                query_states.out_tmemrefs[0].squeeze(1),
            )

            offset[1] += static_sizes[1]
            key_states = gstruct.subview(
                x,
                static_offsets=offset,
                static_sizes=static_sizes,
                static_strides=strides,
            )

            key_states = gstruct.reshape(
                key_states,
                key_states.out_tmemrefs[0].squeeze(1),
            )

            offset[1] += static_sizes[1]
            value_states = gstruct.subview(
                x,
                static_offsets=offset,
                static_sizes=static_sizes,
                static_strides=strides,
            )

            value_states = gstruct.reshape(
                value_states,
                value_states.out_tmemrefs[0].squeeze(1),
            )

            output_tensors = (query_states, key_states, value_states)
            output_tensor_names = ("query_states", "key_states", "value_states")
            basename = "attention"
            output_dir = "./attentionTTL"

            compiled_program = compile_ttl_model(
                output_tensors, output_tensor_names, basename, output_dir
            )

            with GroqRunner(timing_report=False) as runner:
                runner.upload_iop_file(
                    compiled_program["iop_file"],
                    program_name=compiled_program["program_name"],
                )

                results_groq = runner.invoke(
                    {
                        input_tensor_name: hidden_states_np,
                    }
                )

            query_states = torch.from_numpy(results_groq["query_states"])
            key_states = torch.from_numpy(results_groq["key_states"])
            value_states = torch.from_numpy(results_groq["value_states"])

            print(f"query_states.shape: {query_states.shape}")
            print(f"key_states.shape: {key_states.shape}")
            print(f"value_states.shape: {value_states.shape}")

            # ------------------------------------------------------------
            # basename = "attentionOrig"
            # output_dir = "./attentionTTLOrig"

            # concanated_tensor = self.c_attn.forward_ttl(hidden_states_buffer)
            # shape = concanated_tensor.out_tensor_shape
            # shape = [cast(int, dim) for dim in shape]

            # assert (
            #     shape[-1] % 3 == 0
            # ), "concanated_tensor.shape[-1] must be divisible by 3 (query, key, value)"

            # target_shape = tuple([*shape[:-1], 3, shape[-1] // 3])
            # ends = (
            #     target_shape[-1] + VECTOR_SIZE - 1
            # ) // VECTOR_SIZE * VECTOR_SIZE - target_shape[-1]

            # unpacked_tensor = gstruct.vector_unpack(
            #     concanated_tensor,
            #     TiledMemref(
            #         target_shape,
            #         x.out_dtype,
            #         ends=(ends,),
            #     ),
            # )

            # output_tensors = (unpacked_tensor,)
            # compiled_program = compile_ttl_model(
            #     output_tensors, output_tensor_names, basename, output_dir
            # )
            # ------------------------------------------------------------

            print("query_states: ", query_states.shape)
            print("key_states: ", key_states.shape)
            print("value_states: ", value_states.shape)

            shape_kv = (*key_states.shape[:-1], -1, self.head_dim)
            key_states = key_states.view(shape_kv).transpose(1, 2)
            value_states = value_states.view(shape_kv).transpose(1, 2)

        shape_q = (*query_states.shape[:-1], -1, self.head_dim)
        query_states = query_states.view(shape_q).transpose(1, 2)

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            # save all key/value_layer to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = curr_past_key_values.update(
                key_states,
                value_states,
                self.layer_idx,
                {"cache_position": cache_position},
            )
            # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
            if is_cross_attention:
                past_key_values.is_updated[self.layer_idx] = True

        using_eager = self.config._attn_implementation == "eager"
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[
                self.config._attn_implementation
            ]

        if using_eager and self.reorder_and_upcast_attn:
            attn_output, attn_weights = self._upcast_and_reordered_attn(
                query_states, key_states, value_states, attention_mask
            )
        else:
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=self.attn_dropout.p if self.training else 0.0,
                **kwargs,
            )

        attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output, attn_weights

        return super().forward(
            hidden_states,
            past_key_values,
            cache_position,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            output_attentions,
            **kwargs,
        )


class GPT2BlockTTL(GPT2Block):
    def __init__(self, config, layer_idx=None):
        super().__init__(config, layer_idx)

        print("Overriding GPT2Block")

        if hasattr(config, "run_ttl"):
            self.hidden_size = config.hidden_size
            self.inner_dim = (
                config.n_inner if config.n_inner is not None else 4 * self.hidden_size
            )

            self.config = config

        """
        hidden_size = config.hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        self.ln_1 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.attn = GPT2Attention(config=config, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        if config.add_cross_attention:
            self.crossattention = GPT2Attention(
                config=config, is_cross_attention=True, layer_idx=layer_idx
            )
            self.ln_cross_attn = nn.LayerNorm(
                hidden_size, eps=config.layer_norm_epsilon
            )

        self.mlp = GPT2MLP(inner_dim, config)
        """

    def forward(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ):

        if hasattr(self.config, "run_ttl") and self.config.run_ttl:

            if hidden_states is not None:
                split_num = get_split_num(hidden_states.shape)
                tinput = TiledMemref(
                    hidden_states.shape,
                    dtypes.f16,
                    ends=(split_num * VECTOR_SIZE - hidden_states.shape[-1],),
                )
                hidden_states_buffer = GroqBuffer.input("image", tinput)
            else:
                hidden_states_buffer = None

            print(hidden_states)
            print(past_key_values)
            print(cache_position)

            self.compile_ttl(
                hidden_states,
                past_key_values,
                cache_position,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                use_cache,
                output_attentions,
                **kwargs,
            )

        return super().forward(
            hidden_states,
            past_key_values,
            cache_position,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            use_cache,
            output_attentions,
            **kwargs,
        )

    def compile_ttl(
        self,
        hidden_states: tuple[torch.FloatTensor] | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = False,
        output_attentions: bool | None = False,
        **kwargs,
    ) -> (
        tuple[torch.Tensor] | tuple[torch.Tensor, tuple[torch.FloatTensor, ...]] | None
    ):
        """
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        attn_output, self_attn_weights = self.attn(
            hidden_states,
            past_key_values=past_key_values,
            cache_position=cache_position,
            attention_mask=attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            **kwargs,
        )
        # residual connection
        hidden_states = attn_output + residual

        if encoder_hidden_states is not None:
            # add one self-attention block for cross-attention
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with "
                    "cross-attention layers by setting `config.add_cross_attention=True`"
                )
            residual = hidden_states
            hidden_states = self.ln_cross_attn(hidden_states)
            cross_attn_output, cross_attn_weights = self.crossattention(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_attentions=output_attentions,
            )
            # residual connection
            hidden_states = residual + cross_attn_output

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        feed_forward_hidden_states = self.mlp(hidden_states)
        # residual connection
        hidden_states = residual + feed_forward_hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs
        """
