"""
Module for compiling convolution programs for Groq LPU (Language Processing Unit).

This module provides functions to compile encoder models using different compilation
methods (gAPI, gMLIR, or Compiler) for execution on Groq hardware accelerators.
"""

import numpy as np
import groq.api as g
from enum import Enum

from typing import Any, Union, List, Dict, Tuple

from groq_convolution.compile_lpu_convolution import (
    compile_g_api,
    compile_with_compiler,
    get_iop_stats,
)
from groq_convolution.gapi_conv1d import GroqConv1D, ResourceScopeName
from groq_convolution.gapi_pooling import GroqMaxPooling1D
from groq_convolution.constants import VECTOR_SIZE

from gstruct import GroqMLIR

import torch


def compile_ttl_model(
    model: Union[GroqMLIR, Tuple[GroqMLIR, GroqMLIR]],
    output_tensor_name: Union[str, Tuple[str, ...]] = "model_result",
    basename: str = "model",
    output_dir: str = "./modelTTL",
) -> Union[dict[str, Union[str, Any]], Any]:

    from gstruct import GroqBuffer, gstruct_to_mlir, mlir_to_iop

    try:

        if isinstance(model, tuple):

            output_buffer = [
                GroqBuffer.output(output_tensor_name[idx], model[idx])
                for idx in range(len(model))
            ]

        else:
            output_buffer = [GroqBuffer.output(output_tensor_name, model)]

        mlirtext = gstruct_to_mlir(output_buffer)

        iop_file = mlir_to_iop(
            mlirtext, basename, output_dir, is_opt=False
        )  # ; assert False

        program_name = basename

        compiled_program = {
            "iop_file": iop_file,
            "output_dir": output_dir,
            "basename": basename,
            "program_name": program_name,
        }

    except Exception as e:
        print(f"Error message: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback

        traceback.print_exc()
        return None

    return compiled_program
