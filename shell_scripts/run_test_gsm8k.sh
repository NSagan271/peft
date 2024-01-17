#!/bin/bash

python ./scripts/test_gsm8k.py \
  --model_name_or_path ./artifacts/loftq/Llama-2-7b-hf-4bit-64rank \
  --ckpt_dir ./exp_results/gsm8k/llama-7b-loftq-test \
  --batch_size 1 \
