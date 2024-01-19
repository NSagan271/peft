#!/bin/bash
SAVE_DIR="./artifacts/loftq_lplr/"
python ./examples/loftq_finetuning/quantize_save_load.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --bits 4 \
    --iter 5 \
    --use_lplr \
    --lplr_iter 5 \
    --lplr_bits 8 \
    --rank 64 \
    --save_dir $SAVE_DIR \
    --token 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 \
    --device cuda
