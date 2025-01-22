#!/usr/bin/env python
import argparse
import os
import json
import sys
import torch
import tqdm
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, SchedulerType, default_data_collator
import deepspeed
from dpo_trainer import DeepSpeedDPOTrainer
from dpo_engine import DeepSpeedDPOEngine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.data.data_utils import create_dataset, DataCollatorDPO
from utils.utils import print_rank_0, save_model, to_device, set_random_seed, init_tokenizer
from utils.ds_utils import get_train_ds_config


if __name__ == "__main__":
    # 0 拿参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', nargs='*', default=[])
    parser.add_argument('--ptx_path', nargs='*', default=[])
    parser.add_argument("--ptx_samples", type=int, default=0)
    parser.add_argument("--ptx_coef", type=float, default=27.8)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--tokenizer_name_or_path", type=str, default=None)
    parser.add_argument("--config_name_or_path", type=str, default=None)
    parser.add_argument("--num_padding_at_beginning", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--max_prp_len", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--kl_ctl", type=float, default=0.1)
    parser.add_argument("--show_progress", action='store_true')
    parser.add_argument("--no_train_split", action='store_true')
    parser.add_argument("--weight_decay", type=float, default=0.)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", choices=["linear", "cosine", "polynomial"])
    parser.add_argument("--system", type=str, default=None)
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--gradient_checkpointing', action='store_true')
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
    if args.system:
        for i in range(len(args.data_path)):
            if os.path.isfile(os.path.join(args.system, args.data_path[i])):
                args.data_path[i] = os.path.join(args.system, args.data_path[i])
        for i in range(len(args.ptx_path)):
            if os.path.isfile(os.path.join(args.system, args.ptx_path[i])):
                args.ptx_path[i] = os.path.join(args.system, args.ptx_path[i])
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
    set_random_seed(args.seed)
    dist.barrier()
    # 2 tokenizer
    tokenizer = init_tokenizer(args.tokenizer_name_or_path)
    print("Inited tokenizer")
    # 3 数据集
    dataset = [[json.loads(line.strip()) for line in open(fn)] for fn in args.data_path]
    if args.ptx_path:
        unsupervised_train_dataset = [[json.loads(line.strip()) for line in open(fn)] for fn in args.ptx_path]
        train_dataset = dataset
    else:
        unsupervised_train_dataset = dataset[:args.ptx_samples]
        train_dataset = dataset[args.ptx_samples:]
    unsupervised_train_dataset = create_dataset(unsupervised_train_dataset, args, 1, tokenizer)
    train_dataset = create_dataset(train_dataset, args, 4, tokenizer)
    # 4 dataloader
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    unsupervised_train_sampler = RandomSampler(unsupervised_train_dataset) if args.local_rank == -1 else DistributedSampler(unsupervised_train_dataset)
    train_dataloader = DataLoader(train_dataset, collate_fn=DataCollatorDPO(), sampler=train_sampler, batch_size=args.per_device_train_batch_size)
    unsupervised_train_dataloader = DataLoader(unsupervised_train_dataset, collate_fn=default_data_collator, sampler=unsupervised_train_sampler, batch_size=args.per_device_train_batch_size)
    num_total_iters = args.num_train_epochs * int(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_warmup_steps = args.num_warmup_steps * 2
    print("Created prompt dataset")
    dist.barrier()
    # 5 模型
    dpo_engine = DeepSpeedDPOEngine(actor_model_name_or_path=args.model_name_or_path, tokenizer=tokenizer, args=args, num_total_iters=num_total_iters)
    dpo_trainer = DeepSpeedDPOTrainer(dpo_engine, args)
    print("Inited engine and trainer")
    dist.barrier()
    # 6 训练
    print_rank_0("***** Running training *****", args.global_rank)
    if args.gradient_checkpointing:
        dpo_trainer.actor.gradient_checkpointing_enable()
    dpo_trainer.train()
    unsupervised_iterator = iter(unsupervised_train_dataloader)
    for epoch in range(args.num_train_epochs):
        tqdm.tqdm()
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
            unsupervised_train_dataloader.sampler.set_epoch(epoch)
        print_rank_0(f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader) * 2}", args.global_rank)
        for step, batch in enumerate(train_dataloader):
            loss = dpo_trainer.train_dpo(batch)
            try:
                unsup_data = next(unsupervised_iterator)
            except StopIteration:
                unsupervised_iterator = iter(unsupervised_train_dataloader)
                unsup_data = next(unsupervised_iterator)
            unsup_loss = dpo_trainer.train_unsupervised(unsup_data, args.ptx_coef) if args.ptx_path else 0
        if args.output_dir is not None:
            save_model(dpo_engine.actor, args, tokenizer, args.zero_stage, str(epoch) if args.num_train_epochs > 1 else "")
