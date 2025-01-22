#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import math
import json
import random
import sys

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.data.data_utils import create_dataset, DataCollatorReward
from utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, get_optimizer_grouped_parameters, save_model, init_tokenizer, freeze_llama_layers
from utils.ds_utils import get_train_ds_config
from utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, only_optimize_lora_parameters


def parse_args():
    parser = argparse.ArgumentParser()
    # data_path: 1 critic 2 refine 3 prefer
    parser.add_argument('--data_path', nargs='*', default=[])
    parser.add_argument('--validate_dataset', nargs='*', default=[])
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--config_name_or_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--no_train_split", action='store_true')
    parser.add_argument("--num_layers_unfrozen", type=int, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial","constant", "constant_with_warmup"])
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--early_stop", type=str, default="None")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument("--data_sample", type=float, default=1.)
    parser.add_argument('--with_sft', action='store_true')
    parser.add_argument('--zero_stage', type=int, default=0)
    parser.add_argument("--lora_dim", type=int, default=0)
    parser.add_argument("--lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument('--only_optimize_lora', action='store_true')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    # add system path
    if args.system:
        for i in range(len(args.data_path)):
            if os.path.isfile(os.path.join(args.system, args.data_path[i])):
                args.data_path[i] = os.path.join(args.system, args.data_path[i])
        for i in range(len(args.validate_dataset)):
            if os.path.isfile(os.path.join(args.system, args.validate_dataset[i])):
                args.validate_dataset[i] = os.path.join(args.system, args.validate_dataset[i])
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    if args.config_name_or_path is None:
        args.config_name_or_path = args.model_name_or_path
    # Validate settings
    if args.gradient_checkpointing and args.lora_dim > 0:
        assert not args.only_optimize_lora, "--gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."
    return args


def validate(args, model, dataset, collator, device):
    if dataset is None:
        return
    eval_sampler = DistributedSampler(dataset) if args.local_rank >= 0 else SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, collate_fn=collator, sampler=eval_sampler, batch_size=args.per_device_eval_batch_size)
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    cscores = 0
    rscores = 0
    for step, batch in enumerate(eval_dataloader):
        batch = to_device(batch, device)
        with torch.no_grad():
            _, _, chosen, rejected = model(**batch)
        correct_predictions += (chosen > rejected).sum()
        total_predictions += chosen.shape[0]
        cscores += chosen.sum().float()
        rscores += rejected.sum().float()
    acc = correct_predictions / total_predictions
    cscores = cscores / total_predictions
    rscores = rscores / total_predictions
    try:
        acc = get_all_reduce_mean(acc).item()
        cscores = get_all_reduce_mean(cscores).item()
        rscores = get_all_reduce_mean(rscores).item()
    except:
        pass
    print_rank_0(f"chosen_mean_scores : {cscores}, reject_mean_scores : {rscores}, acc : {acc}", args.global_rank)
    return acc


def main():
    # 1 预处理
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = dist.get_rank()
    ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage)
    ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config['train_batch_size'] = args.per_device_train_batch_size * dist.get_world_size() * args.gradient_accumulation_steps
    set_random_seed(args.seed)
    dist.barrier()
    # 2 tokenizer和模型
    tokenizer = init_tokenizer(args.tokenizer_name_or_path)
    rm_model = create_critic_model(args.config_name_or_path, args.model_name_or_path, tokenizer, ds_config, with_sft=args.with_sft)
    if args.lora_dim > 0:
        rm_model = convert_linear_layer_to_lora(rm_model, args.lora_module_name, args.lora_dim)
        if args.only_optimize_lora:
            rm_model = only_optimize_lora_parameters(rm_model)
    print("Inited RM")
    dist.barrier()
    # 加载数据集

    train_raw_dataset = [[json.loads(line.strip()) for line in open(fn)] for fn in args.data_path]
    eval_dataset = [[json.loads(line.strip()) for line in open(fn)] for fn in args.validate_dataset]
    random.shuffle(train_raw_dataset)
    train_dataset = create_dataset(train_raw_dataset, args, 2, tokenizer)
    eval_dataset = create_dataset(eval_dataset, args, 2, tokenizer)
    print("Created prompt dataset")
    # DataLoaders creation:
    data_collator = DataCollatorReward()
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=data_collator, sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(rm_model, args.weight_decay)
    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters, lr=args.learning_rate, betas=(0.9, 0.999))
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    )
    rm_model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=rm_model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)
    if args.gradient_checkpointing:
        rm_model.gradient_checkpointing_enable()
    # Train!
    print_rank_0(f"***** Evaluating reward, Epoch 0/{args.num_train_epochs} *****", args.global_rank)
    best_valid_acc = validate(args, rm_model, eval_dataset, data_collator, device)
    early_stop = 0
    max_stops = int(args.early_stop[5:]) if len(args.early_stop) >= 5 and args.early_stop[5:].isnumeric() else -1 
    print_rank_0(f"***** Running training, frozen layers {args.num_layers_unfrozen}, early stop epochs {max_stops} *****", args.global_rank)
    for epoch in range(args.num_train_epochs):
        print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}", args.global_rank)
        rm_model.train()
        freeze_llama_layers(rm_model.module.transformer, args.num_layers_unfrozen)
        loss_10 = 0
        sft_loss_10 = 0
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        for step, batch in enumerate(train_dataloader):
            batch = to_device(batch, device)
            rm_loss, sft_loss, _, _ = rm_model(**batch, use_cache=False)
            rm_model.backward(rm_loss + sft_loss)
            rm_model.step()
            loss_10 += float(rm_loss)
            sft_loss_10 += float(sft_loss)
            if (step + 1) % 10 == 0:
                print_rank_0(f"Epoch {epoch+1}/{args.num_train_epochs}|Step {step+1}/{len(train_dataloader)}|RM loss {loss_10/10}|SFT loss {sft_loss_10/10}", args.global_rank)
                loss_10, sft_loss_10 = 0, 0
        print_rank_0(f"***** Evaluating reward, Epoch {epoch+1}/{args.num_train_epochs} *****", args.global_rank)
        acc = validate(args, rm_model, eval_dataset, data_collator, device)
        rm_model.tput_timer.update_epoch_count()
        if args.output_dir is not None:
            output = "epoch" + str(epoch)
            save_model(rm_model, args, tokenizer, args.zero_stage, output, no_lm_head=True)
        if best_valid_acc <= acc:
            best_valid_acc = acc
            early_stop = 0
        else:
            early_stop += 1
            if "valid" in args.early_stop and early_stop == int(max_stops):
                print(f"Early stop at epoch {epoch + 1}")
                return
        if args.early_stop.isnumeric() and int(args.early_stop) == epoch + 1:
            print(f"Early stop at epoch {epoch + 1}")
            return


if __name__ == "__main__":
    main()
