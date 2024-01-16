#! /usr/bin/fish

set -x ACCELERATE_MIXED_PRECISION fp16; python examples/loftq_finetuning/train_gsm8k_llama.py \
    --model_name_or_path ./artifacts/loftq/Llama-2-7b-hf-4bit-64rank \
    --output_dir exp_results/gsm8k/llama-2-7b-hf-4bit-64rank \
    --learning_rate 1e-4  \
    --weight_decay 0.1 \
    --lr_scheduler_type cosine \
    --num_warmup_steps 100 \
    --seed 202 \
    --dataset_name gsm8k \
    --dataset_config main \
    --pad_to_max_length \
    --max_source_length 128 \
    --max_target_length 256 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --with_tracking \
    --report_to tensorboard