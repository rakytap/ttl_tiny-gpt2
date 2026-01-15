from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2Block
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import (
    create_causal_mask,
)

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import numpy as np
from typing import Optional
import torch

from gstruct import TiledMemref, GroqBuffer, dtypes
from gstruct.constants import VECTOR_SIZE


def get_split_num(input_shape: tuple[int, ...], inner_axis: int = -1):

    return (input_shape[inner_axis] + VECTOR_SIZE - 1) // VECTOR_SIZE


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
