from transformers.models.gpt2.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel,
    GPT2MLP,
    GPT2Block,
    GPT2Attention,
    GPT2Model,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
)
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache
from transformers.masking_utils import (
    create_causal_mask,
)

from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa

import numpy as np
from typing import Optional, Callable, cast, Tuple
import torch

from gstruct import gstruct, TiledMemref, GroqBuffer, dtypes, GroqMLIR, vxm_ops
from gstruct.constants import VECTOR_SIZE, conv_np_dtype_to_dtypes
from gstruct.models import qrv_from_hidden_states as ttl_qrv_from_hidden_states
from gstruct.runner import GroqRunner
from compile_ttl import compile_ttl_model

from gstruct.ops import gstruct_input_tensor

from gstruct.ops import layer_norm as layer_norm_ttl
from gstruct.ops import linear as linear_ttl

from gstruct.tiled_memref import move_dimension_to_position


USE_EXTERNAL_QRV_EVALUATION = True


def get_split_num(input_shape: tuple[int, ...], inner_axis: int = -1):

    return (input_shape[inner_axis] + VECTOR_SIZE - 1) // VECTOR_SIZE


class GPT2AttentionTTL(GPT2Attention):

    def __init__(self, config, is_cross_attention=False, layer_idx=None):
        super().__init__(config, is_cross_attention, layer_idx)

    def forward(
        self,
        hidden_states: GroqMLIR | None,
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

        assert (
            is_cross_attention == False
        ), "is_cross_attention==True is not implemented yet in TTL"

        if is_cross_attention:
            assert False, "is_cross_attention==True is not implemented yet in TTL"
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

            # hidden_states_np = hidden_states.numpy()

            if USE_EXTERNAL_QRV_EVALUATION:

                weights_np = self.c_attn.weight.numpy()
                bias_np = self.c_attn.bias.numpy()

                # split_num = get_split_num(hidden_states_np.shape)
                # tinput = TiledMemref(
                #     hidden_states_np.shape,
                #     conv_np_dtype_to_dtypes(hidden_states_np.dtype),
                #     ends=(split_num * VECTOR_SIZE - hidden_states_np.shape[-1],),
                # )

                # input_tensor_name = "image"
                # hidden_states_buffer = GroqBuffer.input(input_tensor_name, tinput)

                query_states, key_states, value_states = ttl_qrv_from_hidden_states(
                    hidden_states, self.head_dim, weights_np, bias_np
                )

            else:  # Use internal attention

                query_states, key_states, value_states = (
                    self.qrv_from_hidden_states_ttl(hidden_states)
                )

        if (past_key_values is not None and not is_cross_attention) or (
            past_key_values is not None and is_cross_attention and not is_updated
        ):
            assert False, "past_key_values is not supported in TTL"
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

        print("attention_interface: ", attention_interface)

        if using_eager and self.reorder_and_upcast_attn:
            assert (
                False
            ), "using_eager and self.reorder_and_upcast_attn is not implemented yet in TTL"
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

            shape = attn_output.out_tensor_shape
            shape = [cast(int, dim) for dim in shape]
            shape = tuple([*shape[:-2], shape[-1] * shape[-2]])
            inner_dim = shape[-1]

            out_tmemref = attn_output.out_tmemrefs[0]
            if out_tmemref.ends[0] != 0:

                tiled_memref = TiledMemref(
                    shape,
                    out_tmemref.dtype,
                    ends=(
                        (inner_dim + VECTOR_SIZE - 1) // VECTOR_SIZE * VECTOR_SIZE
                        - inner_dim,
                    ),
                    expanded=True,
                    devId=attn_output.out_tmemrefs[0].devId,
                )

                attn_output = gstruct.vector_pack(attn_output, tiled_memref)

            attn_output = self.c_proj.forward_ttl(attn_output)

            if self.resid_dropout.p > 0 and self.training:
                raise ValueError(
                    "resid_dropout.p > 0 and training is not supported in TTL"
                )
                # attn_output = self.resid_dropout(attn_output)

            return attn_output, attn_weights

            basename = "attention"
            output_dir = "./attentionTTL"

            if attn_weights is None:
                output_tensors = (attn_output,)
                output_tensor_names = ("attn_output",)

            else:
                output_tensors = (
                    attn_output,
                    attn_weights,
                )
                output_tensor_names = (
                    "attn_output",
                    "attn_weights",
                )

            compiled_program = compile_ttl_model(
                output_tensors, output_tensor_names, basename, output_dir
            )

            input_tensor_name = "image"
            Groq_input = {
                input_tensor_name: hidden_states_np,
            }

            with GroqRunner(timing_report=False) as runner:
                runner.upload_iop_file(
                    compiled_program["iop_file"],
                    program_name=compiled_program["program_name"],
                )

                results_groq = runner.invoke(Groq_input)

                attn_output = torch.from_numpy(results_groq["attn_output"])
                attn_weights = results_groq.get("attn_weights", None)
                if attn_weights is not None:
                    attn_weights = torch.from_numpy(attn_weights)

        if self.resid_dropout.p > 0 and self.training:
            raise ValueError("resid_dropout.p > 0 and training is not supported in TTL")
            # attn_output = self.resid_dropout(attn_output)

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

    def qrv_from_hidden_states_ttl(
        self, hidden_states_buffer: GroqBuffer
    ) -> Tuple[GroqMLIR, GroqMLIR, GroqMLIR]:

        concanated_tensor = self.c_attn.forward_ttl(hidden_states_buffer)

        shape = concanated_tensor.out_tensor_shape
        shape = [cast(int, dim) for dim in shape]

        assert (
            shape[-1] % 3 == 0
        ), "concanated_tensor.shape[-1] must be divisible by 3 (query, key, value)"

        target_shape = tuple(
            [*shape[:-1], 3, shape[-1] // (3 * self.head_dim), self.head_dim]
        )
        ends = (
            target_shape[-1] + VECTOR_SIZE - 1
        ) // VECTOR_SIZE * VECTOR_SIZE - target_shape[-1]

        unpacked_tensor = gstruct.vector_unpack(
            concanated_tensor,
            TiledMemref(
                target_shape,
                concanated_tensor.out_dtype,
                ends=(ends,),
            ),
        )

        unpacked_tensor = gstruct.transpose(unpacked_tensor, (0, 3, 2, 1, 4))

        vectors_x = cast(Tuple[int, ...], unpacked_tensor.out_vector_shape)

        assert (
            vectors_x[2] % 3 == 0
        ), "vectors_x[2] must be divisible by 3 (query, key, value)"
        static_sizes = [cast(int, dim) for dim in vectors_x]
        static_sizes[2] = static_sizes[2] // 3

        offset = [0] * len(vectors_x)
        strides = [1] * len(vectors_x)
        query_states = gstruct.subview(
            unpacked_tensor,
            static_offsets=offset,
            static_sizes=static_sizes,
            static_strides=strides,
        )

        query_states = gstruct.reshape(
            query_states,
            query_states.out_tmemrefs[0].squeeze(2),
        )

        offset[2] += static_sizes[2]
        key_states = gstruct.subview(
            unpacked_tensor,
            static_offsets=offset,
            static_sizes=static_sizes,
            static_strides=strides,
        )

        key_states = gstruct.reshape(
            key_states,
            key_states.out_tmemrefs[0].squeeze(2),
        )

        offset[2] += static_sizes[2]
        value_states = gstruct.subview(
            unpacked_tensor,
            static_offsets=offset,
            static_sizes=static_sizes,
            static_strides=strides,
        )

        value_states = gstruct.reshape(
            value_states,
            value_states.out_tmemrefs[0].squeeze(2),
        )

        return query_states, key_states, value_states


class GPT2MLPTTL(GPT2MLP):
    def __init__(self, intermediate_size, config):
        super().__init__(intermediate_size, config)

        # embed_dim = config.hidden_size
        # self.c_fc = Conv1D(intermediate_size, embed_dim)
        # self.c_proj = Conv1D(embed_dim, intermediate_size)
        # self.act = ACT2FN[config.activation_function]
        # self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: GroqMLIR | None) -> GroqMLIR:

        hidden_states = self.c_fc.forward_ttl(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj.forward_ttl(hidden_states)

        if self.dropout.p > 0 and self.training:
            raise ValueError("dropout.p > 0 and training is not supported in TTL")
            # hidden_states = self.dropout(hidden_states)

        return hidden_states


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
        hidden_states: GroqMLIR | None,
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

            return self.compile_ttl(
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
        hidden_states: GroqMLIR | None,
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

        residual = hidden_states
        hidden_states = layer_norm_ttl(
            hidden_states,
            self.ln_1.normalized_shape,
            self.ln_1.weight.numpy(),
            self.ln_1.bias.numpy(),
            self.ln_1.eps,
        )

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
        hidden_states = gstruct.vxm(vxm_ops.vxm_binary_addsat, attn_output, residual)

        if encoder_hidden_states is not None:
            assert False, "cross-attention is not implemented yet in TTL"
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
        hidden_states = layer_norm_ttl(
            hidden_states,
            self.ln_2.normalized_shape,
            self.ln_2.weight.numpy(),
            self.ln_2.bias.numpy(),
            self.ln_2.eps,
        )

        feed_forward_hidden_states = self.mlp(hidden_states)

        # residual connection
        hidden_states = gstruct.vxm(
            vxm_ops.vxm_binary_addsat, feed_forward_hidden_states, residual
        )

        outputs = (hidden_states,)
        if output_attentions:
            assert False, "output_attentions is not implemented yet in TTL"
            outputs += (self_attn_weights,)
            if encoder_hidden_states is not None:
                outputs += (cross_attn_weights,)

        return outputs


class GPT2ModelTTL(GPT2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> tuple | BaseModelOutputWithPastAndCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """

        print("Running GPT2Model forward pass")
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        print("output_attentions ", output_attentions)
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            assert False, "token_type_ids is not implemented yet in TTL"
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        if use_cache:
            assert False, "use_cache is not implemented yet in TTL"
            if past_key_values is None:
                past_key_values = DynamicCache(config=self.config)

            if self.config.add_cross_attention and not isinstance(
                past_key_values, EncoderDecoderCache
            ):
                past_key_values = EncoderDecoderCache(
                    past_key_values, DynamicCache(config=self.config)
                )

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)

        causal_mask = create_causal_mask(
            config=self.config,
            input_embeds=inputs_embeds,
            attention_mask=attention_mask,
            cache_position=cache_position,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        print("causal_mask: ", causal_mask)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask,
                    dtype=inputs_embeds.dtype,
                    tgt_len=input_shape[-1],
                )
            elif self._attn_implementation != "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_attention_mask = None

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            hidden_states = hidden_states + token_type_embeds

        hidden_states = self.drop(hidden_states)

        output_shape = (-1,) + input_shape[1:] + (hidden_states.size(-1),)

        all_self_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if output_attentions and self.config.add_cross_attention else None
        )

        ###########################################xx

        hidden_states_np = hidden_states.numpy()
        split_num = get_split_num(hidden_states.shape)
        tinput = TiledMemref(
            hidden_states.shape,
            dtypes.f32,
            ends=(split_num * VECTOR_SIZE - hidden_states_np.shape[-1],),
        )

        # hidden_states_ttl = GroqBuffer.input("image", tinput)
        hidden_states_ttl = gstruct_input_tensor(
            "image",
            tinput,
            byte_packed=True,
            input_packed=True,
        )

        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states_ttl,)

            outputs = block(
                hidden_states_ttl,
                (
                    past_key_values
                    if not (self.gradient_checkpointing and self.training)
                    else None
                ),
                cache_position,
                causal_mask,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )

            hidden_states_ttl = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

        # self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        hidden_states_ttl = layer_norm_ttl(
            hidden_states_ttl,
            self.ln_f.normalized_shape,
            self.ln_f.weight.numpy(),
            self.ln_f.bias.numpy(),
            self.ln_f.eps,
        )

        hidden_states_shape = hidden_states_ttl.out_tensor_shape

        hidden_states_ttl = gstruct.reshape(
            hidden_states_ttl,
            hidden_states_ttl.out_tmemrefs[0].merge_axes(
                0, len(hidden_states_shape) - 2
            ),
        )

        # hidden_states = hidden_states.view(output_shape)

        hidden_states = hidden_states_ttl

        ########################################

        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        print("returning dict", return_dict)
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )

        return (
            BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
                cross_attentions=all_cross_attentions,
            ),
            hidden_states_np,
        )


class GPT2LMHeadModelTTL(GPT2LMHeadModel):

    def __init__(self, config):
        super().__init__(config)
        # self.transformer = GPT2Model(config)
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        encoder_attention_mask: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> tuple | CausalLMOutputWithCrossAttentions:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values.get_seq_length()` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        transformer_outputs, hidden_states_np = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            cache_position=cache_position,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )

        hidden_vector_shape = hidden_states.out_vector_shape
        hidden_vector_shape = [cast(int, x) for x in hidden_vector_shape]

        slice_start = slice_indices.start
        slice_stop = slice_indices.stop
        slice_step = slice_indices.step

        if slice_step is None:
            slice_step = 1
        if slice_start is None:
            slice_start = 0
        if slice_stop is None:
            slice_stop = hidden_vector_shape[1]

        slice_length = slice_stop - slice_start
        slice_length = slice_length // slice_step

        hidden_states_subview = gstruct.subview(
            hidden_states,
            [0, slice_start, 0],
            [
                hidden_vector_shape[0],
                slice_stop - slice_start,
                hidden_vector_shape[2],
            ],
            [1, slice_step, 1],
        )

        # hidden_states_subview = hidden_states[:, slice_indices, :]

        weight_np = np.transpose(self.lm_head.weight.numpy(), (1, 0)).astype(np.float16)
        print("weight_np: ", weight_np.shape)
        print(np.isnan(weight_np).any())
        bias_np = self.lm_head.bias
        if bias_np is not None:
            bias_np = bias_np.numpy()

        # logits = linear_ttl(
        #     hidden_states_subview,
        #     weights=weight_np,
        #     bias=bias_np,
        # )
        # logits = self.lm_head(hidden_states_subview)

        basename = "GPT2LMHeadModelTTL"
        output_dir = "./GPT2LMHeadModelTTL"

        output_tensors = (hidden_states, hidden_states_subview)
        output_tensor_names = ("hidden_states", "hidden_states_subview")

        compiled_program = compile_ttl_model(
            output_tensors, output_tensor_names, basename, output_dir
        )

        input_tensor_name = "image"
        Groq_input = {
            input_tensor_name: hidden_states_np,
        }

        with GroqRunner(timing_report=False) as runner:
            runner.upload_iop_file(
                compiled_program["iop_file"],
                program_name=compiled_program["program_name"],
            )

            results_groq = runner.invoke(Groq_input)

        hidden_states = torch.from_numpy(results_groq["hidden_states"])
        hidden_states_subview = torch.from_numpy(results_groq["hidden_states_subview"])
        # logits = torch.from_numpy(results_groq["logits"])

        # slice_indices = (
        #     slice(-logits_to_keep, None)
        #     if isinstance(logits_to_keep, int)
        #     else logits_to_keep
        # )

        # hidden_states_subview = hidden_states[:, slice_indices, :]
        logits = self.lm_head(hidden_states_subview)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
