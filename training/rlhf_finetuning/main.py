#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import json
import os
import math
import random
import torch
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
)

import deepspeed

from ppo_trainer import DeepSpeedPPOTrainer, DeepSpeedPPOTrainerUnsupervised
from rlhf_engine import DeepSpeedRLHFEngine

import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_dataset, MiniDataset, DataCollatorRLHF
from utils.utils import print_rank_0, to_device, set_random_seed, get_all_reduce_mean, moving_average, save_engine, init_tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument('--data_path', nargs='*', default=[])
    parser.add_argument("--ptx_samples", type=int, default=0)
    parser.add_argument('--ptx_path', nargs='*', default=[])
    parser.add_argument("--ptx_coef", type=float, default=27.8)
    parser.add_argument("--ptx_decay", action='store_true')
    parser.add_argument("--actor_model_name_or_path", type=str, required=True)
    parser.add_argument("--actor_tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--critic_model_name_or_path", type=str, required=True)
    parser.add_argument("--critic_config_name_or_path", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size",type=int, default=16)
    parser.add_argument("--per_device_mini_train_batch_size", type=int, default=16)
    parser.add_argument("--generation_batch_numbers", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    parser.add_argument("--max_prompt_seq_len", type=int, default=256)
    parser.add_argument("--max_answer_seq_len", type=int, default=256)
    parser.add_argument("--actor_learning_rate", type=float, default=9.65e-6)
    parser.add_argument("--critic_learning_rate", type=float, default=5e-6)
    parser.add_argument("--actor_weight_decay", type=float, default=0.01)
    parser.add_argument("--critic_weight_decay", type=float, default=0.01)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--save_interval", type=int, default=100000)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument('--num_actor_layers_unfrozen', type=int, default=8)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--preprocessing_num_workers", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--no_train_split', action='store_true')
    parser.add_argument("--saved_experience_path", type=str, default=None)
    parser.add_argument("--num_train_steps", type=int, default=100000)
    # DeepSpeed
    parser.add_argument("--enable_hybrid_engine", action='store_true')
    parser.add_argument("--unpin_actor_parameters", action='store_true')
    parser.add_argument("--release_inference_cache", action='store_true')
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--tp_gather_partition_size", type=int, default=8)
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--offload_reference_model', action='store_true')
    parser.add_argument('--actor_zero_stage', type=int, default=3)
    parser.add_argument('--critic_zero_stage', type=int, default=0)
    parser.add_argument('--actor_gradient_checkpointing', action='store_true')
    parser.add_argument('--critic_gradient_checkpointing', action='store_true')
    ## LoRA for efficient training setting
    parser.add_argument("--actor_lora_dim", type=int, default=0)
    parser.add_argument("--actor_lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument("--critic_lora_dim", type=int, default=0)
    parser.add_argument("--critic_lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument('--only_optimize_lora', action='store_true')
    ## Make EMA as an optional feature
    parser.add_argument('--enable_ema', action='store_true')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    args.max_seq_len = args.max_prompt_seq_len + args.max_answer_seq_len
    if args.system:
        for i in range(len(args.data_path)):
            if os.path.isfile(os.path.join(args.system, args.data_path[i])):
                args.data_path[i] = os.path.join(args.system, args.data_path[i])
        for i in range(len(args.ptx_path)):
            if os.path.isfile(os.path.join(args.system, args.ptx_path[i])):
                args.ptx_path[i] = os.path.join(args.system, args.ptx_path[i])
    assert not args.offload, "zero-offload is not currently supported but coming soon!"
    if args.actor_tokenizer_name_or_path is None:
        args.actor_tokenizer_name_or_path = args.actor_model_name_or_path
    if args.critic_config_name_or_path is None:
        args.critic_config_name_or_path = args.critic_model_name_or_path
    # Validate settings
    if (args.actor_gradient_checkpointing and args.actor_lora_dim > 0) or (args.critic_gradient_checkpointing and args.critic_lora_dim > 0):
        assert not args.only_optimize_lora, "--{actor,critic}_gradient_checkpointing and --only_optimizer_lora cannot be enabled at the same time."
    if args.inference_tp_size > 1:
        assert args.actor_zero_stage == 3, "Zero stage 3 must be used to do Tensor sharding in the hybrid engine"
    return args


def create_datasets(args, tokenizer, tokenizer_right):
    dataset = [[json.loads(line.strip()) for line in open(fn)] for fn in args.data_path]
    if args.ptx_path:
        unsupervised_train_dataset = [[json.loads(line.strip()) for line in open(fn)] for fn in args.ptx_path]
        prompt_train_dataset = dataset
    else:
        unsupervised_train_dataset = dataset[:args.ptx_samples]
        prompt_train_dataset = dataset[args.ptx_samples:]
    unsupervised_train_dataset = create_dataset(unsupervised_train_dataset, args, 1, tokenizer_right)
    prompt_train_dataset = create_dataset(prompt_train_dataset, args, 3, tokenizer)

    # DataLoaders creation:
    if args.local_rank == -1:
        prompt_train_sampler = RandomSampler(prompt_train_dataset)
        if args.ptx_path:
            unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset)
    else:
        prompt_train_sampler = DistributedSampler(prompt_train_dataset)
        if args.ptx_path:
            unsupervised_train_sampler = DistributedSampler(unsupervised_train_dataset)
    prompt_train_dataloader = DataLoader(prompt_train_dataset, collate_fn=DataCollatorRLHF(), sampler=prompt_train_sampler, batch_size=args.per_device_train_batch_size)
    if args.ptx_path:
        unsupervised_train_dataloader = DataLoader(unsupervised_train_dataset, collate_fn=default_data_collator, sampler=unsupervised_train_sampler, batch_size=args.per_device_train_batch_size)
    else:
        unsupervised_train_dataloader = [None] * len(prompt_train_dataloader)  # basically a dummy dataloader

    num_update_steps_per_epoch = min(len(prompt_train_dataloader), len(unsupervised_train_dataloader)) * \
        (args.per_device_train_batch_size / args.per_device_mini_train_batch_size) * \
        args.ppo_epochs / args.gradient_accumulation_steps
    num_total_iters = args.num_train_epochs * math.ceil(num_update_steps_per_epoch)

    return prompt_train_dataloader, unsupervised_train_dataloader, min(args.num_train_steps, num_total_iters)


def main():
    args = parse_args()

    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        deepspeed.init_distributed()
    args.device = device

    args.global_rank = dist.get_rank()

    # create common tokenizer based on actor model
    tokenizer_left_padding = init_tokenizer(args.actor_tokenizer_name_or_path, padding_side="left")
    tokenizer_right_padding = init_tokenizer(args.actor_tokenizer_name_or_path)
    args.gradient_accumulation_steps_actor = args.gradient_accumulation_steps * 2
    # If passed along, set the training seed now.
    set_random_seed(args.seed)
    torch.distributed.barrier()
    prompt_train_dataloader, unsupervised_train_dataloader, num_total_iters = create_datasets(args, tokenizer_left_padding, tokenizer_right_padding)
    torch.distributed.barrier()
    # RLHF engine is responsible for creating models, loading checkpoints, ds-initialize models/optims/lr-schedulers
    rlhf_engine = DeepSpeedRLHFEngine(
        actor_model_name_or_path=args.actor_model_name_or_path,
        critic_config_name_or_path=args.critic_config_name_or_path,
        critic_model_name_or_path=args.critic_model_name_or_path,
        tokenizer=tokenizer_right_padding, num_total_iters=num_total_iters, args=args)

    ppo_trainer = DeepSpeedPPOTrainerUnsupervised if args.ptx_path else DeepSpeedPPOTrainer
    trainer = ppo_trainer(rlhf_engine, args)
    # first number is how many experience-batch to generate, second number is the training batch size, which is the micro-batch size used
    exp_mini_dataset = MiniDataset(args.generation_batch_numbers, args.per_device_mini_train_batch_size)
    unsup_mini_dataset = MiniDataset(args.generation_batch_numbers, args.per_device_mini_train_batch_size)

    # Train!
    print_rank_0("***** Running training *****", args.global_rank)
    global_steps = 0
    global_iters = 0
    rewards = []

    for epoch in range(args.num_train_epochs):
        print_rank_0(f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, Total Generation Batches {min(len(prompt_train_dataloader), len(unsupervised_train_dataloader))}", args.global_rank)
        for step, (batch_prompt, batch_unsupervised) in enumerate(zip(prompt_train_dataloader, unsupervised_train_dataloader)):
            batch_prompt = to_device(batch_prompt, device)
            if batch_unsupervised is not None:
                batch_unsupervised = to_device(batch_unsupervised, device)
                unsup_dataset = unsup_mini_dataset.add(batch_unsupervised)
            else:
                unsup_dataset = unsup_mini_dataset.add([[None] * args.per_device_train_batch_size])
            out = trainer.generate_experience(batch_prompt)
            exp_dataset = exp_mini_dataset.add(out)
            print_rank_0(f"experience dataset {len(exp_dataset)} unsup dataset {len(unsup_dataset)}", args.global_rank)

            if exp_dataset is not None:
                inner_iter = 0
                al, cl, ul = 0, 0, 0
                average_reward = 0

                if args.actor_gradient_checkpointing:
                    rlhf_engine.actor.gradient_checkpointing_enable()

                for ppo_ep in range(args.ppo_epochs):
                    for i, (exp_data, unsup_data) in enumerate(zip(exp_dataset, unsup_dataset)):
                        a, c = trainer.train_rlhf(exp_data)
                        al += float(a)
                        cl += float(c)
                        # print(f"localrank: {args.global_rank}|rewards: {exp_data['rewards']}\n", end="")
                        average_reward += exp_data["rewards"].mean()
                        if args.ptx_path:
                            ptx_coef = (args.ptx_coef * (1 - global_iters / num_total_iters)) if args.ptx_decay else args.ptx_coef
                            unsup_loss = trainer.train_unsupervised(unsup_data, ptx_coef)
                            ul += unsup_loss.item()
                        inner_iter += 1
                        global_iters += 1
                        if args.enable_ema:
                            moving_average(rlhf_engine.actor, rlhf_engine.actor_ema, zero_stage=args.actor_zero_stage)
                        print_rank_0(f'ppo_ep: {ppo_ep+1}|step: {i}|al: {a:.4f}|cl: {c:.4f}|ul: {c:.4f}', args.global_rank)
                    random.shuffle(exp_dataset)
                    random.shuffle(unsup_dataset)

                print_rank_0(f'epoch: {epoch}|step: {step}|act_loss: {al/inner_iter:.4f}|cri_loss: {cl/inner_iter:.4f}|unsuper_loss: {ul/inner_iter:.4f}',args.global_rank)
                average_reward = get_all_reduce_mean(average_reward).item()
                rewards.append(average_reward)
                print_rank_0(f"average reward score: {average_reward/inner_iter}", args.global_rank)
                print_rank_0("-------------------------------------------------------------------------------------", args.global_rank)

            global_steps += 1
            if args.actor_gradient_checkpointing:
                rlhf_engine.actor.gradient_checkpointing_disable() # args.num_train_steps
            if args.output_dir is not None and (step + 1) % args.save_interval == 0:
                save_engine(rlhf_engine, args, tokenizer_right_padding, f"epoch{epoch}step{step + 1}")
            if global_steps >= args.num_train_steps:
                break
        if args.output_dir is not None:
            save_engine(rlhf_engine, args, tokenizer_right_padding, f"epoch{epoch}")
        if global_steps >= args.num_train_steps:
            break


if __name__ == "__main__":
    main()
