from transformers.pytorch_utils import Conv1D
from gstruct import GroqMLIR, TiledMemref, GroqBuffer, dtypes
from gstruct.constants import VECTOR_SIZE, conv_np_dtype_to_dtypes

from gstruct.ops import addmm as ttl_addmm
from compile_ttl import compile_ttl_model
from gstruct.runner import GroqRunner
from gstruct import gstruct

from typing import Union, Optional
import numpy as np

import torch


def get_split_num(input_shape: tuple[int, ...], inner_axis: int = -1):
    return (input_shape[inner_axis] + VECTOR_SIZE - 1) // VECTOR_SIZE


class Conv1DTTL(Conv1D):
    def __init__(self, nf, nx):
        super().__init__(nf, nx)
        print("Overriding Conv1D")

    def forward(self, x):
        print("Running Conv1D forward pass")
        input_tensor_name = "image"
        output_tensors = self.forward_ttl(x, input_tensor_name)

        output_tensor_name = "ret"
        basename = "conv1d"
        output_dir = "./Conv1DTTL"

        compiled_program = compile_ttl_model(
            output_tensors, output_tensor_name, basename, output_dir
        )

        with GroqRunner(timing_report=False) as runner:
            runner.upload_iop_file(
                compiled_program["iop_file"],
                program_name=compiled_program["program_name"],
            )

            weight_transposed = (
                self.weight.numpy().swapaxes(-1, -2).copy().astype(np.float16)
            )
            results_groq = runner.invoke(
                {
                    input_tensor_name: x.numpy(),
                    "weight": weight_transposed,
                    "bias": self.bias.numpy(),
                }
            )

        ret = results_groq[output_tensor_name]
        return torch.from_numpy(ret)

        return super().forward(x)

    def forward_ttl(
        self, x: Union[torch.Tensor, GroqMLIR], input_tensor_name: str = "image"
    ):

        # size_out = x.size()[:-1] + (self.nf,)
        # x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        # x = x.view(size_out)
        # return x

        if isinstance(x, torch.Tensor):
            x = x.numpy()

            split_num = get_split_num(x.shape)
            tinput = TiledMemref(
                x.shape,
                conv_np_dtype_to_dtypes(x.dtype),
                ends=(split_num * VECTOR_SIZE - x.shape[-1],),
            )
            x_buffer = GroqBuffer.input(input_tensor_name, tinput)
        else:
            x_buffer = x

        x_buffer_shape = x_buffer.out_tensor_shape

        x_buffer = gstruct.reshape(
            x_buffer,
            x_buffer.out_tmemrefs[0].merge_axes(0, len(x_buffer_shape) - 1),
        )

        weight_transposed = (
            self.weight.numpy().swapaxes(-1, -2).copy().astype(np.float16)
        )
        split_num = get_split_num(weight_transposed.shape)

        tweight = TiledMemref(
            weight_transposed.shape,
            conv_np_dtype_to_dtypes(weight_transposed.dtype),
            ends=(split_num * VECTOR_SIZE - weight_transposed.shape[-1],),
        )
        weight_buffer = GroqBuffer.input("weight", tweight)

        bias = self.bias.numpy()
        split_num = get_split_num(bias.shape)
        tbias = TiledMemref(
            bias.shape,
            conv_np_dtype_to_dtypes(bias.dtype),
            ends=(split_num * VECTOR_SIZE - bias.shape[-1],),
        )
        bias_buffer = GroqBuffer.input("bias", tbias)

        x = ttl_addmm(
            bias_buffer,
            x_buffer,
            weight_buffer,
        )

        x = gstruct.reshape(
            x,
            TiledMemref(
                tuple([*x_buffer_shape[0:-1], self.nf]),
                x.out_dtype,
                ends=(
                    (self.nf + VECTOR_SIZE - 1) // VECTOR_SIZE * VECTOR_SIZE - self.nf,
                ),
            ),
        )

        return x
