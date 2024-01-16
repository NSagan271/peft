#!/bin/bash
SAVE_DIR="../artifacts/loftq/"
python ../examples/loftq_finetuning/quantize_save_load.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.1" \
    --bits 4 \
    --iter 1 \
    --rank 16 \
    --save_dir $SAVE_DIR
