from transformers.pytorch_utils import Conv1D
from gstruct import GroqMLIR, TiledMemref, GroqBuffer, dtypes
from gstruct.constants import VECTOR_SIZE, conv_np_dtype_to_dtypes

from gstruct.ops import linear as ttl_linear
from compile_ttl import compile_ttl_model
from gstruct.runner import GroqRunner

from typing import Union
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

            results_groq = runner.invoke({input_tensor_name: x.numpy()})
            print(results_groq)

        ret = results_groq[output_tensor_name]
        return torch.from_numpy(ret)

        ret = super().forward(x)

        print(ret)
        ff

        return super().forward(x)

    def forward_ttl(self, x: Union[torch.Tensor, GroqMLIR], input_tensor_name: str):

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

        x = ttl_linear(
            x_buffer,
            self.weight.numpy().astype(np.float16),
            self.bias.numpy(),
        )

        return x
