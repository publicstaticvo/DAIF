# TODO: 改成在main函数中读取好数据并按照比例分割出用于PPO/DPO和用于PTX的数据，传入这里包装成数据集。
import torch
from torch.utils.data import Dataset, Subset, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
import os
import random
import math
from itertools import chain
from raw_datasets import LocalDataset


def pad_for_dpo(sentence, pad_token, length, prompt_length=None, side="right", prompt=None):
    """
    pad single sequence in DPO way
    :param sentence: sentence to pad
    :param pad_token: pad token id
    :param length: max sequence length
    :param prompt_length: length of prompt
    :param side: padding side
    :param prompt: for debug
    """
    assert prompt_length is None or length > prompt_length, f"This prompt has length {prompt_length} and is too long for {length}: {prompt}"
    if len(sentence) > length:
        pad_sequence = sentence[:length] if side == "right" else sentence[-length:]
        attention_mask = [1 for _ in range(length)]
        if prompt_length is not None:
            action_mask = [0 for _ in range(prompt_length - 1)] + [1 for _ in range(length - prompt_length)]
    else:
        p = [0 for _ in range(length - len(sentence))]
        if prompt_length is not None:
            action_mask = [0 for _ in range(prompt_length - 1)] + [1 for _ in range(len(sentence) - prompt_length)]
        if side == "right":
            pad_sequence = sentence + [pad_token for _ in range(length - len(sentence))]
            attention_mask = [1 for _ in sentence] + p
            if prompt_length is not None:
                action_mask = action_mask + p
        else:
            pad_sequence = [pad_token for _ in range(length - len(sentence))] + sentence
            if prompt_length is not None:
                action_mask = p + action_mask
            attention_mask = p + [1 for _ in sentence]
    if prompt_length is not None:
        return pad_sequence, attention_mask, action_mask
    return pad_sequence, attention_mask


class PromptDataset(Dataset):

    def __init__(self, prompt_dataset, chosen_dataset, reject_dataset, pad_token_id, train_phase):
        super().__init__()
        self.prompt_dataset = prompt_dataset
        self.chosen_dataset = chosen_dataset
        self.reject_dataset = reject_dataset
        self.pad_token_id = pad_token_id
        self.train_phase = train_phase

    def __len__(self):
        length = len(self.chosen_dataset)
        if self.train_phase == 3:
            length = len(self.prompt_dataset)
        return length

    def __getitem__(self, idx):
        if self.train_phase == 1:
            return {
                "input_ids": self.chosen_dataset[idx]["input_ids"],
                "attention_mask": self.chosen_dataset[idx]["attention_mask"],
                "labels": self.chosen_dataset[idx]["input_ids"]
            }
        elif self.train_phase == 2:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                   self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"]
        elif self.train_phase == 3:
            return self.prompt_dataset[idx]["input_ids"], self.prompt_dataset[idx]["attention_mask"]
        elif self.train_phase == 4:
            return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
                   self.chosen_dataset[idx]["action_mask"], \
                   self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"], \
                   self.reject_dataset[idx]["action_mask"]


def create_dataset(data, args, train_phase, tokenizer):
    prompt_dataset = []
    chosen_dataset = []
    reject_dataset = []
    if train_phase == 1:
        for x in data:
            chosen_sentence = x['prompt'] + x['chosen'] + tokenizer.eos_token
            input_ids = tokenizer(chosen_sentence, max_length=args.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")
            chosen_dataset.append({"input_ids": input_ids.input_ids.squeeze(0), "attention_mask": input_ids.attention_mask.squeeze(0)})

    elif train_phase == 2:
        for x in data:
            chosen_sentence = x['prompt'] + x['chosen'] + tokenizer.eos_token
            reject_sentence = x['prompt'] + x['reject'] + tokenizer.eos_token
            chosen_ids = tokenizer(chosen_sentence, max_length=args.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")
            reject_ids = tokenizer(reject_sentence, max_length=args.max_seq_len, padding="max_length", truncation=True, return_tensors="pt")
            chosen_dataset.append(chosen_ids)
            reject_dataset.append(reject_ids)

    elif train_phase == 3:
        for x in data:
            input_ids_for_generate = tokenizer(x['prompt'], max_length=args.max_prompt_seq_len, padding="max_length", truncation=True, return_tensors="pt")
            prompt_dataset.append({"input_ids": input_ids_for_generate.input_ids.squeeze(0), "attention_mask": input_ids_for_generate.attention_mask.squeeze(0)})

    elif train_phase == 4:
        pad_token_id = tokenizer.encode(tokenizer.pad_token)
        assert len(pad_token_id) == 2
        pad_token_id = pad_token_id[1]
        for x in data:
            prompt = tokenizer.encode(x['prompt'])
            chosen = tokenizer.encode(x['chosen'] + tokenizer.eos_token)
            reject = tokenizer.encode(x['reject'] + tokenizer.eos_token)
            chosen, chosen_attention_mask, chosen_action_mask = pad_for_dpo(prompt + chosen[1:], pad_token_id, args.max_seq_len, len(prompt), prompt=x['prompt'])
            reject, reject_attention_mask, reject_action_mask = pad_for_dpo(prompt + reject[1:], pad_token_id, args.max_seq_len, len(prompt), prompt=x['prompt'])
            chosen_dataset.append({"input_ids": torch.LongTensor(chosen),
                                   "attention_mask": torch.LongTensor(chosen_attention_mask),
                                   "action_mask": torch.BoolTensor(chosen_action_mask)})
            reject_dataset.append({"input_ids": torch.LongTensor(reject),
                                   "attention_mask": torch.LongTensor(reject_attention_mask),
                                   "action_mask": torch.BoolTensor(reject_action_mask)})
    return PromptDataset(prompt_dataset, chosen_dataset, reject_dataset, tokenizer.pad_token_id, train_phase)


class DataCollatorReward:

    def __call__(self, data):
        return {"input_ids": torch.cat([f[0] for f in data] + [f[2] for f in data], dim=0),
                "attention_mask": torch.cat([f[1] for f in data] + [f[3] for f in data], dim=0)}


class DataCollatorDPO:

    def __call__(self, data):
        return {"input_ids": torch.stack([f[0] for f in data] + [f[3] for f in data], dim=0),
                "attention_mask": torch.stack([f[1] for f in data] + [f[4] for f in data], dim=0),
                "action_mask": torch.stack([f[2] for f in data] + [f[5] for f in data], dim=0)}


class DataCollatorRLHF:

    def __init__(self):
        pass

    def __call__(self, data):
        return {"prompts": torch.stack([f[0] for f in data]), "prompt_att_mask": torch.stack([f[1] for f in data])}


class MiniDataset:

    def __init__(self, max_size, small_batch_size):
        self.dataset = []
        self.max_size = max_size
        self.small_batch_size = small_batch_size

    def seperate(self):
        small_dataset = []
        for large_batch in self.dataset:
            if type(large_batch) == list or type(large_batch) == tuple:
                large_size = len(large_batch[0])
            elif type(large_batch) == dict:
                large_size = len(large_batch[list(large_batch.keys())[0]])
            else:
                large_size = len(large_batch)
            for i in range(0, large_size, self.small_batch_size):
                if type(large_batch) == list or type(large_batch) == tuple:
                    small_dataset.append([x[i:i + self.small_batch_size] for x in large_batch])
                elif type(large_batch) == dict:
                    small_dataset.append({k: v[i:i + self.small_batch_size] for k, v in large_batch.items()})
                else:
                    small_dataset.append(large_batch[i:i + self.small_batch_size])
        self.free()

        return small_dataset

    def add(self, data):
        if len(self.dataset) < self.max_size:
            self.dataset.append(data)
            if len(self.dataset) == self.max_size:
                return self.seperate()
            else:
                return None
        else:
            raise ValueError("The dataset is full but we did not stop it. There is a bug in the code.")

    def free(self):
        self.dataset = []
