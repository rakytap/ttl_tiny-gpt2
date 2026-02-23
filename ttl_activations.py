from transformers.activations import ACT2FN, NewGELUActivation

from torch import Tensor, nn
import torch

import math

from gstruct import gstruct, TiledMemref, GroqBuffer, dtypes, GroqMLIR, vxm_ops
from gstruct.constants import VECTOR_SIZE, dtypes_to_np
from gstruct.ops import activation as ttl_activation

from gstruct.runner import GroqRunner
from compile_ttl import compile_ttl_model

import numpy as np


def get_split_num(input_shape: tuple[int, ...], inner_axis: int = -1):

    return (input_shape[inner_axis] + VECTOR_SIZE - 1) // VECTOR_SIZE


class NewGELUActivationTTL(NewGELUActivation):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://huggingface.co/papers/1606.08415
    """

    def forward(self, input_buffer: GroqMLIR) -> GroqMLIR:
        # return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))

        return ttl_activation(input_buffer, activation_function="NewGELU")
