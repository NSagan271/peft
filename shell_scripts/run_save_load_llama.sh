#!/bin/bash
SAVE_DIR="../artifacts/loftq/"
python ../examples/loftq_finetuning/quantize_save_load.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --bits 4 \
    --iter 5 \
    --rank 64 \
    --save_dir $SAVE_DIR

#--model_name_or_path "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf/blobs" \

