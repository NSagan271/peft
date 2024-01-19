import torch

import logging
from typing import Union

from lplr.weight_compressors import alternating_mixed_lplr
from peft.utils.quantization_utils import NFQuantizerFactory

@torch.no_grad()
def loftq_lplr_init(
    weight: Union[torch.Tensor, torch.nn.Parameter],
    num_bits: int,
    num_bits_factors: int,
    reduced_rank: int,
    num_full_precision_factors: int = 0,
    quantizer_factory:NFQuantizerFactory = NFQuantizerFactory(),
    num_iter=1,
    num_iter_lplr=10,
    device="cuda",
    log_errors=False 
):
    if num_bits not in [2, 4, 8]:
        raise ValueError("Only support 2, 4, 8 bits quantization")
    if num_iter <= 0:
        raise ValueError("Number of iterations must be greater than 0")
    
    output_device = weight.device
    weight = weight.to(device)
    
    n, d = weight.size()
    transposed_weight = False
    if n < d:
        weight = weight.T
        transposed_weight = True
        n, d = d, n

    device = weight.device
    dtype = weight.dtype

    logging.info(
        f"Weight: ({n}, {d}) | Rank: {reduced_rank} "
        f"| Num Iter: {num_iter} | Num Bits: {num_bits}"
    )
    
    quantizer = quantizer_factory.get_quantizer(num_bits, device=device)

    weight = weight.to(device=device, dtype=torch.float32)
    res = weight

    errors = []

    best_error = float('inf')
    best_mtxs = None
    for _ in range(num_iter):
        torch.cuda.empty_cache()
        # Quantization
        dequantized_weight = quantizer.dequantize_block(*quantizer.quantize_block(res))
        res = weight - dequantized_weight

        # Decompose the residual by SVD
        mtxs, _ = alternating_mixed_lplr(
            X=res, k=reduced_rank,
            r1=num_full_precision_factors, r2=num_full_precision_factors,
            B1=num_bits_factors, B2=num_bits_factors,
            quantizer_factory=quantizer_factory, iters=num_iter_lplr
        )
        L, R = mtxs
        res = weight - torch.mm(L, R)

        error = torch.norm(weight - dequantized_weight - L@R, p="fro").item()
        errors.append(error)
        if error < best_error:
            best_mtxs = (dequantized_weight, R, L)
            best_error = error

    dequantized_weight, R, L = best_mtxs
    if transposed_weight:
        dequantized_weight = dequantized_weight.T
        L, R = R.T, L.T

    lora_A, lora_B = R.contiguous(), L.contiguous()
    dequantized_weight = dequantized_weight.contiguous()

    if log_errors:
        return dequantized_weight.to(device=output_device, dtype=dtype), lora_A.to(output_device), lora_B.to(output_device), errors

    return dequantized_weight.to(device=output_device, dtype=dtype), lora_A.to(output_device), lora_B.to(output_device)

