#!/bin/fish

export DS_SKIP_CUDA_CHECK=1; accelerate launch --config_file ./shell_scripts/nsagan_config_single_gpu.yaml scripts/test_gsm8k.py \
  --model_name_or_path ./artifacts/loftq/Llama-2-7b-hf-4bit-64rank \
  --adapter_name_or_path ./artifacts/loftq/Llama-2-7b-hf-4bit-64rank/loft_init \
  --batch_size 1 \
