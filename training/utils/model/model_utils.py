# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import math
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)

from transformers.deepspeed import HfDeepSpeedConfig
from .reward_model import RewardModel


def create_hf_model(model_class, model_name_or_path, tokenizer, ds_config=None, tokenizer_size=None):
    model_config = AutoConfig.from_pretrained(model_name_or_path)
    model_config.dropout = 0.0
    # , torch_dtype=torch.float16, low_cpu_mem_usage=True
    model = model_class.from_pretrained(model_name_or_path, from_tf=bool(".ckpt" in model_name_or_path), config=model_config)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    if tokenizer_size is None:
        # tokenizer_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else int(8 * math.ceil(len(tokenizer) / 8.0))
        tokenizer_size = int(8 * math.ceil(len(tokenizer) / 8.0))
    # print(f"tokenizer_size {tokenizer_size}, len(tokenizer) {len(tokenizer)}")
    model.resize_token_embeddings(tokenizer_size)
    return model


def create_critic_model(config_name_or_path, model_name_or_path, tokenizer, ds_config=None, rlhf_training=False, tokenizer_size=None, with_sft=False, bias=False):
    if config_name_or_path:
        model_config = AutoConfig.from_pretrained(model_name_or_path)
        model = RewardModel.from_pretrained(model_name_or_path, tokenizer, model_config, with_sft=with_sft, bias=bias)
    else:
        model = RewardModel.from_pretrained(model_name_or_path, tokenizer, with_sft=with_sft, bias=bias)
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.transformer.config.end_token_id = tokenizer.eos_token_id
    model.transformer.config.pad_token_id = model.config.eos_token_id
    if tokenizer_size is None:
        tokenizer_size = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else int(8 * math.ceil(len(tokenizer) / 8.0))
    model.transformer.resize_token_embeddings(tokenizer_size)
    return model
