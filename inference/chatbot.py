import argparse
import os
import time
import torch
import json
import torch.distributed as dist
import deepspeed
import math
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.ds_utils import get_eval_ds_config
from utils.utils import print_rank_0, to_device, init_tokenizer, get_all_reduce_mean
from utils.model.model_utils import create_hf_model, create_critic_model
from transformers import AutoModelForCausalLM


def remove_exist_outfile(rank, out_fn):
    if rank == 0:
        for i in range(8):
            if os.path.isfile(f"{out_fn}-{i}.txt"):
                os.remove(f"{out_fn}-{i}.txt")
    elif rank == -1:
        if os.path.isfile(f"{out_fn}.txt"):
            os.remove(f"{out_fn}.txt")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--additional_human_stopword', action='store_true')
    parser.add_argument("--config_name_or_path", type=str)
    parser.add_argument('--deepspeed', action='store_true')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--mode", type=str, default="save", choices=["generate", "ppl", "rm"])
    parser.add_argument("--tokenizer_name_or_path", type=str)
    parser.add_argument("--outfile_name_or_path", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--max_prompt_seq_len", type=int, default=384)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    args.world_size = max(torch.cuda.device_count(), 1)
    if len(args.model_name_or_path) == 1 and args.model_name_or_path[0][-4:] == "json":
        args.model_name_or_path = json.load(open(args.model_name_or_path[0]))
    if not args.tokenizer_name_or_path:
        args.tokenizer_name_or_path = args.model_name_or_path
    if not args.config_name_or_path:
        args.config_name_or_path = args.model_name_or_path
    return args


def inference_step(model, tokenizer, batch, max_new_tokens, max_length=0, num_samples=0):
    with torch.no_grad():
        if num_samples == 0:
            output = model.generate(batch.input_ids, attention_mask=batch.attention_mask, max_new_tokens=max_new_tokens)
        else:
            output = model.generate(batch.input_ids, attention_mask=batch.attention_mask, max_new_tokens=max_new_tokens, do_sample=True,
                                    num_return_sequences=num_samples, temperature=0.8, top_k=50, top_p=0.9, no_repeat_ngram_size=3)
            # print(output.shape)
    outputs = tokenizer.batch_decode(output[:, max_length:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs


def ppl_step(model, input_ids, attention_mask, action_mask):
    with torch.no_grad():
        logits = model(input_ids, attention_mask=attention_mask, return_dict=True).logits[:, :-1]
        logits = -torch.log_softmax(logits, dim=-1)
        selected = torch.gather(logits, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        ppl = (selected * action_mask).sum(-1) / action_mask.sum(-1)
    return ppl


def generate_mixin(args, dataset, tokenizer, model_name):
    model = create_hf_model(AutoModelForCausalLM, model_name, tokenizer)
    if args.local_rank >= 0:
        dist.barrier()
        model, *_ = deepspeed.initialize(model=model, config=get_eval_ds_config(offload=None, stage=3))
    else:
        model.to(args.device)
    model.eval()
    print("Finish loading model")
    for i in range(0, len(dataset), args.batch_size * args.world_size):
        if args.local_rank >= 0:
            batch = dataset[i + args.local_rank: i + args.batch_size + args.local_rank: args.world_size]
        else:
            batch = dataset[i: i + args.batch_size]

        prompt = [x['prompt'] for x in batch]
        L = len(batch)
        input_ids = tokenizer(prompt, return_tensors='pt', max_length=args.max_prompt_seq_len, padding="max_length").to(args.device)
        ans = inference_step(model, tokenizer, input_ids, args.max_new_tokens, args.max_prompt_seq_len, args.num_samples)
        prompt_and_answers = []
        # a: B*N, L, H
        for j in range(L):
            if args.num_samples == 0:
                a = ans[j]
                if "</s>" in a:
                    a = a[:a.index("</s>")]
                if args.additional_human_stopword and "\n\nHuman:" in a:
                    a = a[:a.index("\n\nHuman:")]
                prompt_and_answers.append(prompt[j] + a)
                batch[j]["greedy"] = a
            else:
                batch[j]["sample"] = []
                for s in ans[j * args.num_samples: (j + 1) * args.num_samples]:
                    if "</s>" in s:
                        s = s[:s.index("</s>")]
                    if args.additional_human_stopword and "\n\nHuman:" in s:
                        s = s[:s.index("\n\nHuman:")]
                    prompt_and_answers.append(prompt[j] + s)
                    batch[j]["sample"].append(s)
        if args.outfile_name_or_path:
            fn = f"{args.outfile_name_or_path}-{args.local_rank}.txt" if args.local_rank >= 0 else f"{args.outfile_name_or_path}.txt"
            with open(fn, "a+") as f:
                f.write("\n".join([json.dumps(b) for b in batch]) + "\n")
    print(f"Process {args.local_rank} finished")
    dist.barrier()


def ppl_mixin(args, dataset, tokenizer, model_name):
    model = create_hf_model(AutoModelForCausalLM, model_name, tokenizer)
    if args.local_rank >= 0:
        dist.barrier()
        model, *_ = deepspeed.initialize(model=model, config=get_eval_ds_config(offload=None, stage=3))
    else:
        model.to(args.device)
    model.eval()
    print("Finish loading model")
    for i in range(0, len(dataset), args.batch_size * args.world_size):
        if args.local_rank >= 0:
            batch = dataset[i + args.local_rank: i + args.batch_size + args.local_rank: args.world_size]
        else:
            batch = dataset[i: i + args.batch_size]

        prompt = [x['prompt'] for x in batch]
        prompt_and_answers = [x['prompt'] + x["greedy"] for x in batch]
        L = len(batch)
        input_ids = tokenizer(prompt_and_answers, return_tensors='pt', max_length=args.max_seq_len, padding="max_length").to(args.device)
        prompt = tokenizer(prompt, return_tensors='pt', padding="longest")
        prompt_length = prompt.attention_mask.sum(-1).tolist()
        sequence_length = input_ids.attention_mask.sum(-1).tolist()
        action_mask = torch.zeros((L, args.max_seq_len), dtype=torch.long, device=args.device)
        for j, x in enumerate(batch):
            action_mask[j][prompt_length[j] - 1: sequence_length[j] - 1] = 1
        ppl = ppl_step(model, input_ids.input_ids, input_ids.attention_mask, action_mask)
        ppl = ppl.tolist()
        for j, p in enumerate(ppl):
            batch[j]["ppl"] = p
        if args.outfile_name_or_path:
            fn = f"{args.outfile_name_or_path}-{args.local_rank}.txt" if args.local_rank >= 0 else f"{args.outfile_name_or_path}.txt"
            with open(fn, "a+") as f:
                f.write("\n".join([json.dumps(b) for b in batch]) + "\n")
    print(f"Process {args.local_rank} finished")
    dist.barrier()


def rm_mixin(args, dataset, tokenizer, model_name, config_name):
    # bias: bias=True for OpenAssist reward model, bias=False for model trained by RM step
    model = create_critic_model(config_name, model_name, tokenizer, bias=True, tokenizer_size=int(8 * math.ceil(len(tokenizer) / 8.0)))
    if args.local_rank >= 0 and args.deepspeed:
        dist.barrier()
        model, *_ = deepspeed.initialize(model=model, config=get_eval_ds_config(offload=None, stage=3))
    else:
        model.to(args.device)
    model.half()
    model.eval()
    average_scores = 0

    # Input format of OpenAssist model is f'<|prompter|>{prompt}<|endoftext|><|assistant|>{output}<|endoftext|>'
    # Input format of model trained by RM step is f'{prompt} {output}'
    for d in dataset:
        input_ids = tokenizer([f"<|prompter|>{d['prompt']}<|endoftext|><|assistant|>{d['output']}<|endoftext|>"], return_tensors='pt', max_length=args.max_seq_len, padding="max_length").to(args.device)
        score = model.forward_score(input_ids.input_ids, attention_mask=input_ids.attention_mask)[0]
        d['score'] = float(score)
        average_scores += float(score)
        fn = f"{args.outfile_name_or_path}-rm-{args.local_rank}.txt" if args.local_rank >= 0 else f"{args.outfile_name_or_path}-rm.txt"
        with open(fn, "a+") as f:
            f.write(json.dumps(d) + "\n")
    print(f"Average scores: {average_scores / len(dataset)}")

def main():
    args = parse_args()
    if args.local_rank >= 0:
        torch.cuda.set_device(args.local_rank)
        args.device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend="nccl", init_method='env://')
    elif torch.cuda.is_available():
        args.device = torch.device("cuda")
    else:
        args.device = torch.device("cpu")

    tokenizer = init_tokenizer(args.tokenizer_name_or_path, padding_side="left" if args.mode == "generate" else "right")
    raw = [json.loads(line) for line in open(args.data_path)]
    # remove_exist_outfile(args.local_rank, args.outfile_name_or_path)
    if args.mode == "rm":
        rm_mixin(args, raw, tokenizer, args.model_name_or_path, args.config_name_or_path)
    elif args.mode == "ppl":
        ppl_mixin(args, raw, tokenizer, args.model_name_or_path)
    else:
        generate_mixin(args, raw, tokenizer, args.model_name_or_path)


if __name__ == "__main__":
    main()
