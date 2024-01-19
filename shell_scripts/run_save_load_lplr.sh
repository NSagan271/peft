#!/bin/bash
SAVE_DIR="./artifacts/loftq_lplr/"
python ./examples/loftq_finetuning/quantize_save_load.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.1" \
    --bits 4 \
    --iter 5 \
    --use_lplr \
    --lplr_iter 5 \
    --low_memory_quant \
    --lplr_bits 8 \
    --rank 64 \
    --save_dir $SAVE_DIR
