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

import argparse
import os

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from peft import LoftQConfig, LoftQLplrConfig, LoraConfig, TaskType, get_peft_model
from peft.utils.quantization_utils import NFQuantizerFactory


class Shell(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = nn.Parameter(bias, requires_grad=False)


def unwrap_model(model, sub_module_name=".base_layer"):
    sub_module_name_list = [k.split(sub_module_name)[0] for k in model.state_dict().keys() if sub_module_name in k]
    sub_module_name_set = set(sub_module_name_list)
    for name in sub_module_name_set:
        # get the parent of the submodule
        name_parent = ".".join(name.split(".")[:-1])
        name_child = name.split(".")[-1]
        sub_module = model.get_submodule(name_parent)
        print(sub_module)

        # replace with shell
        child = getattr(sub_module, name_child)
        weight = getattr(child.base_layer, "weight", None)
        bias = getattr(child.base_layer, "bias", None)
        shell = Shell(weight, bias)

        setattr(sub_module, name_child, shell)

    print("You have unwrapped the model. Use it on your own risk.")


def print_model(model, name):
    print("=" * 10 + name + "=" * 10)
    print(model)
    for name, param in model.named_parameters():
        if torch.is_tensor(param):
            if param.dtype in [torch.float32, torch.float16]:
                print(
                    name,
                    param.shape,
                    param.device,
                    param.dtype,
                    param.requires_grad,
                    param.mean().item(),
                    param.max().item(),
                )
            else:
                print(name, param.shape, param.device, param.dtype, param.requires_grad)


def arg_parse():
    parser = argparse.ArgumentParser(description="Quantize a model with LoftQ.")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="The name or path of the fp32/16 model.",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="The access token to download model from HuggingFace Hub.",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=4,
        help="The quantized bits",
    )
    parser.add_argument(
        "--iter",
        type=int,
        default=1,
        help="The alternating steps in LoftQ",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./model_zoo/loftq/",
        help="The rank of the LoRA adapter",
    )
    parser.add_argument(
        "--use_lplr",
        action='store_true',
        help='If set, the SVD step of LoftQ is replaced with LPLR'
    )
    parser.add_argument(
        "--lplr_bits",
        type=int,
        default=8,
        help='If use_lplr is set, this specifies the bit precision of the low-rank factors'
    )
    parser.add_argument(
        "--lplr_iter",
        type=int,
        default=20,
        help='If use_lplr is set, this specifies the number of alternating LPLR steps'
    )
    parser.add_argument(
        "--lplr_num_full_precision_factors",
        type=int,
        default=0,
        help='If use_lplr is set, this specifies the number of  factors to keep in full precision'
    )
    parser.add_argument(
        "--uniform_quant",
        action='store_true',
        help='Use uniform quantization instead of Normal float'
    )
    parser.add_argument(
        "--low_memory_quant",
        action='store_true',
        help='Use a lower-memory quantizer implementation'
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device on which to perform LoftQ computations."
    )
    args = parser.parse_args()
    return args


def quantize_and_save():
    args = arg_parse()

    # Download weights and configure LoRA
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
    if any(name in args.model_name_or_path.lower() for name in ["llama", "mistral", "falcon"]):
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, token=args.token, trust_remote_code=True)
        task_type = TaskType.CAUSAL_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["bart", "t5"]):
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q_proj", "k_proj", "v_proj", "fc1", "fc2", "out_proj"]

    elif any(name in args.model_name_or_path.lower() for name in ["deberta", "roberta", "bert"]):
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, token=args.token)
        task_type = TaskType.SEQ_CLS
        target_modules = ["query_proj", "key_proj", "value_proj", "dense"]  # embeddings not supported by peft
    else:
        raise NotImplementedError("Other models not supported yet.")

    # Config of LoftQ
    if args.use_lplr:
        loftq_lplr_config = LoftQLplrConfig(
            loftq_bits=args.bits, loftq_iter=args.iter,
            lplr_bits=args.lplr_bits, lplr_iter=args.lplr_iter,
            lplr_num_full_precision_factors=args.lplr_num_full_precision_factors,
            quantization_type="uniform" if args.uniform_quant else "normal",
            low_memory_quantizer=args.low_memory_quant,
            device=args.device
        )
        
        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=True,
            r=args.rank,
            lora_alpha=16 if task_type is TaskType.CAUSAL_LM else args.rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights="loftq-lplr",
            loftq_lplr_config=loftq_lplr_config,
        )
    else:
        loftq_config = LoftQConfig(
            loftq_bits=args.bits, loftq_iter=args.iter,
            quantization_type="uniform" if args.uniform_quant else "normal",
            low_memory_quantizer=args.low_memory_quant,
            device=args.device
        )

        lora_config = LoraConfig(
            task_type=task_type,
            inference_mode=True,
            r=args.rank,
            lora_alpha=16 if task_type is TaskType.CAUSAL_LM else args.rank,
            lora_dropout=0.1,
            target_modules=target_modules,
            init_lora_weights="loftq",
            loftq_config=loftq_config,
        )

    # Obtain LoftQ model
    lora_model = get_peft_model(model, lora_config)
    base_model = lora_model.get_base_model()

    # Save LoftQ model
    model_name = args.model_name_or_path.split("/")[-1] + f"-{args.bits}bit" + f"-{args.rank}rank"
    base_model_dir = os.path.join(args.save_dir, model_name)
    lora_model_dir = os.path.join(args.save_dir, model_name, "loft_init")

    # save lora adapters first
    lora_model.base_model.peft_config[
        "default"
    ].base_model_name_or_path = base_model_dir  # This can be a local path or Hub model id
    lora_model.base_model.peft_config["default"].init_lora_weights = True  # Don't apply LoftQ when loading again

    lora_model.save_pretrained(lora_model_dir)
    print_model(lora_model, "lora_model")

    # remove lora adapters and save the backbone
    unwrap_model(base_model)
    base_model.save_pretrained(base_model_dir)
    tokenizer.save_pretrained(base_model_dir)

    print_model(base_model, "base_model")

    return base_model_dir, lora_model_dir


if __name__ == "__main__":
    base_dir, lora_dir = quantize_and_save()

# example command:
# python quantize_save_load.py \
# --model_name_or_path meta-llama/Llama-2-7b-hf \
# --token XXX \
# --bits 4 --iter 5 --rank 16 \
# --save_dir ./model_zoo/loftq/
