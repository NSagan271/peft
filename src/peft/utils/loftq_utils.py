# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Reference code: https://github.com/yxli2123/LoftQ/blob/main/utils.py
# Reference paper: https://arxiv.org/abs/2310.08659

import logging
from typing import Union

import torch

from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from peft.utils.quantization_utils import NFQuantizerFactory


if is_bnb_available():
    import bitsandbytes as bnb


def _low_rank_decomposition(weight, reduced_rank=32):
    """
    :param weight: The matrix to decompose, of shape (H, W) :param reduced_rank: the final rank :return:
    """
    matrix_dimension = len(weight.size())
    if matrix_dimension != 2:
        raise ValueError(f"Only support 2D matrix, but your input has {matrix_dimension} dimensions.")

    # Use SVD to decompose a matrix, default full_matrices is False to save parameters
    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

    L = U @ (torch.sqrt(torch.diag(S)[:, 0:reduced_rank]))
    R = torch.sqrt(torch.diag(S)[0:reduced_rank, :]) @ Vh

    return {"L": L, "R": R, "U": U, "S": S, "Vh": Vh, "reduced_rank": reduced_rank}


@torch.no_grad()
def loftq_init(
    weight: Union[torch.Tensor, torch.nn.Parameter],
    num_bits: int,
    reduced_rank: int,
    num_iter=1,
    quantizer_factory:NFQuantizerFactory = NFQuantizerFactory(),
    device="cuda",
    log_errors=False 
):
    if num_bits not in [2, 4, 8]:
        raise ValueError("Only support 2, 4, 8 bits quantization")
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")

    out_feature, in_feature = weight.size()
    output_device = weight.device
    weight = weight.to(device)
    dtype = weight.dtype

    logging.info(
        f"Weight: ({out_feature}, {in_feature}) | Rank: {reduced_rank} "
        f"| Num Iter: {num_iter} | Num Bits: {num_bits}"
    )
    if not is_bnb_4bit_available() or num_bits in [2, 8]:
        quantizer = quantizer_factory.get_quantizer(num_bits, device=device)
        compute_device = device
    else:
        compute_device = "cuda"

    weight = weight.to(device=compute_device, dtype=torch.float32)
    res = weight.clone()

    errors = []

    best_error = float('inf')
    best_mtxs = None
    for _ in range(num_iter):
        torch.cuda.empty_cache()
        # Quantization
        if num_bits == 4 and is_bnb_4bit_available():
            qweight = bnb.nn.Params4bit(
                res.to("cpu"), requires_grad=False, compress_statistics=False, quant_type="nf4"
            ).to(compute_device)
            dequantized_weight = bnb.functional.dequantize_4bit(qweight.data, qweight.quant_state)
        else:
            quantized_weight, max_abs, shape = quantizer.quantize_block(res)
            dequantized_weight = quantizer.dequantize_block(quantized_weight, max_abs, shape)

        res = weight - dequantized_weight

        # Decompose the residual by SVD
        output = _low_rank_decomposition(res, reduced_rank=reduced_rank)
        L, R, reduced_rank = output["L"], output["R"], output["reduced_rank"]
        res = weight - torch.mm(L, R)

        error = torch.norm(weight - dequantized_weight - L@R, p="fro").item()
        errors.append(error)
        if error < best_error:
            best_mtxs = (dequantized_weight, R, L)
            best_error = error

    dequantized_weight, R, L = best_mtxs
    lora_A, lora_B = R, L

    if log_errors:
        return dequantized_weight.to(device=output_device, dtype=dtype), lora_A.to(output_device), lora_B.to(output_device), errors

    return dequantized_weight.to(device=output_device, dtype=dtype), lora_A.o(output_device), lora_B.to(output_device)
