# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import os
import json
import torch
import random
import numpy as np
import torch.distributed as dist
from transformers import set_seed, AutoTokenizer, LlamaTokenizerFast
import deepspeed
import functools
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
# from training.utils.module.lora import convert_lora_to_linear_layer


def barrier():
    if dist.is_initialized():
        dist.barrier()


def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)


def to_device(batch, device):
    output = {}
    for k, v in batch.items():
        try:
            output[k] = v.to(device)
        except:
            output[k] = v
    return output


def init_tokenizer(tokenizer_name_or_path, padding_side="right"):
    config = json.load(open(os.path.join(tokenizer_name_or_path, "tokenizer_config.json")))
    is_llama = (config["tokenizer_class"].lower() == "llamatokenizer")
    if is_llama:
        tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_name_or_path, fast_tokenizer=True, padding_side=padding_side)
        if not tokenizer.eos_token:
            tokenizer.add_special_tokens({"eos_token": "</s>", "bos_token": "<s>", "unk_token": "<unk>"})
    else:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, fast_tokenizer=True, padding_side=padding_side)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


class MovingAverage:

    def __init__(self):
        self.count = 0
        self.total = 0
        self.mean = 0

    def update(self, num):
        self.total += num
        self.count += 1
        self.mean = self.total / self.count

        return self.mean


def save_hf_format(model, tokenizer, args, sub_folder="", no_lm_head=False):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key or (no_lm_head and "lm_head" in key):
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    try:
        tokenizer.save_vocabulary(output_dir)
    except: 
        pass


def set_random_seed(seed):
    if seed is not None:
        set_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor


def get_optimizer_grouped_parameters(model,
                                     weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if (not any(nd in n
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n
                        for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay":
            0.0,
        },
    ]
    return optimizer_grouped_parameters


def _z3_params_to_fetch(param_list):
    return [
        p for p in param_list
        if hasattr(p, 'ds_id') and p.ds_status == ZeroParamStatus.NOT_AVAILABLE
    ]


def moving_average(model, model_ema, beta=0.992, device=None, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    with torch.no_grad():
        for param, param_ema in zip(model.parameters(),
                                    model_ema.parameters()):
            # TODO: use prefiltering for efficiency
            params_to_fetch = _z3_params_to_fetch([param, param_ema]) if zero_stage_3 else []
            should_gather_param = len(params_to_fetch) > 0
            with deepspeed.zero.GatheredParameters(
                    params_to_fetch, enabled=should_gather_param):
                data = param.data
                if device is not None:
                    data = data.to(device)
                param_ema.data.copy_(torch.lerp(data, param_ema.data, beta))


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0, no_lm_head=False):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)

    model_to_save = model_ema.module if hasattr(model_ema, 'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]), enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k and not (no_lm_head and "lm_head" in k):
                output_state_dict[k] = v_p
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file)
        del output_state_dict


def save_model(model, args, tokenizer, zero_stage, sub_folder, no_lm_head=False):
    print_rank_0(f'saving {sub_folder} ...')
    # model = convert_lora_to_linear_layer(model)
    if torch.distributed.get_rank() == 0:
        save_hf_format(model, tokenizer, args, sub_folder=sub_folder, no_lm_head=no_lm_head)
    torch.distributed.barrier()
    if zero_stage == 3:
        save_zero_three_model(model, global_rank=args.global_rank, save_dir=os.path.join(args.output_dir, sub_folder), zero_stage=zero_stage, no_lm_head=no_lm_head)


def save_engine(engine, args, tokenizer, step):
    save_model(engine.actor, args, tokenizer, args.actor_zero_stage, f"{step}/actor")
    if args.with_reward_model:
        save_model(engine.critic, args, tokenizer, args.critic_zero_stage, f"{step}/critic")
    if args.enable_ema:
        save_model(engine.actor_ema, args, tokenizer, args.actor_zero_stage, f"{step}/actor_ema")


def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr, *args):
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs):
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")


def hf_get_decoder_blocks(model):
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = ("h", "layers", "decoder.layers", "transformer.h", "transformer.model", "model.decoder.layers", "gpt_neox.layers", "decoder.block")
    return findattr(model, hidden_layers_attrs)


def freeze_bottom_causal_layers(model, num_layers_unfrozen=0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)


def freeze_llama_layers(model, num_layers_unfrozen):
    """Freezes the bottom transformer block layers of the specified model."""
    if num_layers_unfrozen is None:
        return
    model.embed_tokens.requires_grad_(False)
    model.norm.requires_grad_(False)
    num_layers_frozen = max(0, len(model.layers) - num_layers_unfrozen)
    for i, layer in enumerate(model.layers):
        layer.requires_grad_(False)
        if i + 1 == num_layers_frozen:
            break
