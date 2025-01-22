#!/usr/bin/env python
import argparse
import os
import json
import sys
import torch
import math
import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, SchedulerType, default_data_collator
import deepspeed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from critic_trainer import DeepSpeedCriticTrainer
from dpo_trainer import DeepSpeedDPOTrainer
from engine import DeepSpeedCriticEngine
from utils.data.data_utils import CritiqueDataset, DataCollatorForCritic, DPODataset, DataCollatorDPO
from utils.utils import print_rank_0, save_model, barrier, set_random_seed, init_tokenizer
from utils.ds_utils import get_train_ds_config


if __name__ == "__main__":
    # 0 拿参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply_chat_template", action='store_true')
    parser.add_argument("--config_name_or_path", type=str, default=None)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument("--early_stop", type=int, default=-1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--kl_ctl", type=float, default=1)
    parser.add_argument("--kl_ctl_for_exo", type=float, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--loss_type", type=str, default="mse", choices=["exo", "mse", "dpo"])
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "polynomial"])
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_padding_at_beginning", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument('--ptx_path', type=str, default=None)
    parser.add_argument("--ptx_coef", type=float, default=27.8)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--show_progress", action='store_true')
    parser.add_argument("--steps_per_print", type=int, default=10)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument('--use_average', action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0.)
    # deepspeed features
    parser.add_argument('--offload', action='store_true')
    parser.add_argument('--zero_stage', type=int, default=0)
    parser.add_argument("--enable_hybrid_engine", action='store_true')
    parser.add_argument("--unpin_actor_parameters", action='store_true')
    parser.add_argument("--release_inference_cache", action='store_true')
    parser.add_argument("--inference_tp_size", type=int, default=1)
    parser.add_argument("--tp_gather_partition_size", type=int, default=8)
    parser.add_argument('--offload_reference_model', action='store_true')
    # LoRA for efficient training setting
    parser.add_argument("--lora_dim", type=int, default=0)
    parser.add_argument("--lora_module_name", type=str, default="decoder.layers.")
    parser.add_argument('--only_optimize_lora', action='store_true')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    
    # 1 预处理
    assert not args.offload, "zero-offload is not currently supported but coming soon!"
    if args.tokenizer_name_or_path is None:
        args.tokenizer_name_or_path = args.model_name_or_path
    if args.config_name_or_path is None:
        args.config_name_or_path = args.model_name_or_path
    if args.local_rank == -1:
        args.device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        deepspeed.init_distributed()
    args.global_rank = dist.get_rank()
    num_gpus = max(1, torch.cuda.device_count())
    # ds_config = get_train_ds_config(offload=args.offload, stage=args.zero_stage, steps_per_print=args.steps_per_print)
    # ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    # ds_config['train_batch_size'] = args.per_device_train_batch_size * dist.get_world_size() * args.gradient_accumulation_steps
    set_random_seed(args.seed)
    barrier()
    # 2 tokenizer
    tokenizer = init_tokenizer(args.tokenizer_name_or_path)
    print("Inited tokenizer")
    # 3 数据集
    if args.loss_type == "dpo":
        train_dataset = DPODataset(args, tokenizer)
        DataCollator = DataCollatorDPO
    else:
        train_dataset = CritiqueDataset(args, tokenizer)
        DataCollator = DataCollatorForCritic
    # 4 dataloader
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=DataCollator(), sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    num_total_iters = args.num_train_epochs * math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    print("Created prompt dataset")
    barrier()
    # 5 模型
    TrainerType = DeepSpeedDPOTrainer if args.loss_type == "dpo" else DeepSpeedCriticTrainer
    dpo_engine = DeepSpeedCriticEngine(actor_model_name_or_path=args.model_name_or_path, tokenizer=tokenizer, args=args, num_total_iters=num_total_iters)
    trainer = TrainerType(dpo_engine, args)
    print("Inited engine and trainer")
    barrier()
    # 6 训练
    print_rank_0("***** Running training *****", args.global_rank)
    if args.gradient_checkpointing:
        trainer.actor.gradient_checkpointing_enable()
    trainer.train()
    for epoch in range(args.num_train_epochs):
        if 0 < args.early_stop == epoch:
            print(f"Early stop at {epoch}")
            break
        if hasattr(train_dataloader.sampler, "set_epoch"):
            train_dataloader.sampler.set_epoch(epoch)
        print_rank_0(f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}", args.global_rank)
        inner_iter = tqdm.tqdm(train_dataloader) if args.show_progress else train_dataloader
        for step, batch in enumerate(inner_iter):
            loss = trainer.train_iter(batch)
            if args.steps_per_print > 0 and (step + 1) % args.steps_per_print == 0:
                print_rank_0(f"Loss: {loss:.4f}", args.local_rank)
        if args.output_dir is not None:
            save_model(dpo_engine.actor, args, tokenizer, args.zero_stage, str(epoch) if args.num_train_epochs > 1 else "")
