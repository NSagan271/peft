#!/bin/bash
# python train_gsm8k.py \
#     --model_name_or_path "../artifacts/loftq/Llama-2-7b-hf-4bit-16rank" \
#     --learning_rate 3e-4 \
#     --seed 11 \
#     --expt_name gsm8k_llama_7b_4bit_16rank_loftq \
#     --output_dir exp_results/ \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --evaluation_strategy "no" \
#     --save_strategy "epoch" \
#     --weight_decay 0.1 \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 10 \
#     --do_train \
#     --report_to tensorboard

#accelerate launch --config_file ds_zero3_cpu.yaml ../examples/loftq_finetuning/train_gsm8k_llama.py \
#    --model_name_or_path ../artifacts/loftq/Llama-2-7b-hf-4bit-16rank \
#    --output_dir exp_results/gsm8k/llama-7b-loftq-test \
#    --learning_rate 1e-4  \
#    --weight_decay 0.1 \
#    --lr_scheduler_type cosine \
#    --num_warmup_steps 100 \
#    --seed 202 \
#    --dataset_name gsm8k \
#    --dataset_config main \
#    --pad_to_max_length \
#    --max_source_length 128 \
#    --max_target_length 256 \
#    --num_train_epochs 1 \
#    --per_device_train_batch_size 4 \
#    --per_device_eval_batch_size 4 \
#    --gradient_accumulation_steps 4 \
#    --with_tracking \
#    --report_to tensorboard

accelerate launch --config_file default.yaml ../examples/loftq_finetuning/train_gsm8k_llama.py \
    --model_name_or_path ../artifacts/loftq/Llama-2-7b-hf-4bit-16rank \
    --output_dir exp_results/gsm8k/llama-7b-loftq-test \
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
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --with_tracking \
    --report_to tensorboard
