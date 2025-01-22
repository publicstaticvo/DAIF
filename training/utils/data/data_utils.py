import torch
import torch.distributed as dist
from torch.utils.data import Dataset
import json


def get_rank():
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def get_prompt(x, tokenizer=None, template=False):
    user_prompt = x['user_prompt'] if "user_prompt" in x else x['prompt']
    if template:
        system_prompt = x['system_prompt'] if 'system_prompt' in x else "You are a helpful assistant."
        return tokenizer.apply_chat_template([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], tokenize=False, add_generation_prompt=True)
    return user_prompt


def pad_for_dpo(sentence, pad_token, length, prompt_length, side="right", prompt=None):
    """
    pad single sequence in DPO way
    :param sentence: sentence to pad
    :param pad_token: pad token id
    :param length: max sequence length
    :param prompt_length: length of prompt
    :param side: padding side
    :param prompt: for debug
    """
    assert length > prompt_length, f"This prompt has length {prompt_length} and is too long for {length}: {prompt}"
    if len(sentence) > length:
        pad_sequence = sentence[:length] if side == "right" else sentence[-length:]
        attention_mask = [1 for _ in range(length)]
        action_mask = [0 for _ in range(prompt_length - 1)] + [1 for _ in range(length - prompt_length)]
    else:
        p = [0 for _ in range(length - len(sentence))]
        action_mask = [0 for _ in range(prompt_length - 1)] + [1 for _ in range(len(sentence) - prompt_length)]
        if side == "right":
            pad_sequence = sentence + [pad_token for _ in range(length - len(sentence))]
            attention_mask = [1 for _ in sentence] + p
            action_mask = action_mask + p
        else:
            pad_sequence = [pad_token for _ in range(length - len(sentence))] + sentence
            action_mask = p + action_mask
            attention_mask = p + [1 for _ in sentence]
    return pad_sequence, attention_mask, action_mask


class CritiqueDataset(Dataset):

    def __init__(self, args, tokenizer, dataset_name_or_path=None):
        super().__init__()
        if dataset_name_or_path is None:
            dataset_name_or_path = args.data_path
        data = [json.loads(line.strip()) for line in open(dataset_name_or_path)]

        self.data = []
        pad_token_id = tokenizer.encode(tokenizer.pad_token)
        print(f"Pad token id for {args.model_name_or_path} is {pad_token_id}")
        offset = len(pad_token_id) - 1
        pad_token_id = pad_token_id[-1]
        eos_token = "<|eot_id|>" if "Llama-3" in args.model_name_or_path else "</s>"
        print(f"EOS token for {args.model_name_or_path}: {eos_token}")

        ml = 0
        for x in data:
            prompt = tokenizer.encode(get_prompt(x, tokenizer, args.apply_chat_template))
            item = {"input_ids": [], "attention_mask": [], "action_mask": [], "reward": torch.FloatTensor(x['rewards'])}
            for t in x['text']:
                text = tokenizer.encode(t + eos_token)
                input_ids, attention_mask, action_mask = pad_for_dpo(prompt + text[offset:], pad_token_id, args.max_seq_len, len(prompt), prompt=x['prompt'])
                ml = max(ml, len(prompt) + len(text) - offset)
                item['input_ids'].append(torch.LongTensor(input_ids))
                item['attention_mask'].append(torch.LongTensor(attention_mask))
                item['action_mask'].append(torch.BoolTensor(action_mask))
            for k in ["input_ids", "attention_mask", "action_mask"]:
                item[k] = torch.stack(item[k])
            self.data.append(item)
        print(f"max_length: {ml}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DPODataset(Dataset):

    def __init__(self, args, tokenizer, dataset_name_or_path=None):
        super().__init__()
        if dataset_name_or_path is None:
            dataset_name_or_path = args.data_path
        data = [json.loads(line.strip()) for line in open(dataset_name_or_path)]

        self.chosen_dataset, self.reject_dataset = [], []
        pad_token_id = tokenizer.encode(tokenizer.pad_token)
        print(f"Pad token id for {args.model_name_or_path} is {pad_token_id}")
        offset = len(pad_token_id) - 1
        pad_token_id = pad_token_id[-1]
        eos_token = "<|eot_id|>" if "Llama-3" in args.model_name_or_path else "</s>"
        print(f"EOS token for {args.model_name_or_path}: {eos_token}")

        ml = 0
        for x in data:
            prompt = tokenizer.encode(get_prompt(x, tokenizer, args.apply_chat_template))
            chosen = tokenizer.encode(x['chosen'] + tokenizer.eos_token)
            reject = tokenizer.encode(x['reject'] + tokenizer.eos_token)
            chosen, chosen_attention_mask, chosen_action_mask = pad_for_dpo(prompt + chosen[offset:], pad_token_id, args.max_seq_len, len(prompt), prompt=x['prompt'])
            reject, reject_attention_mask, reject_action_mask = pad_for_dpo(prompt + reject[offset:], pad_token_id, args.max_seq_len, len(prompt), prompt=x['prompt'])
            ml = max([ml, len(prompt) + len(chosen) - offset, len(prompt) + len(reject) - offset])
            self.chosen_dataset.append({"input_ids": torch.LongTensor(chosen), "attention_mask": torch.LongTensor(chosen_attention_mask), "action_mask": torch.BoolTensor(chosen_action_mask)})
            self.reject_dataset.append({"input_ids": torch.LongTensor(reject), "attention_mask": torch.LongTensor(reject_attention_mask), "action_mask": torch.BoolTensor(reject_action_mask)})
        self.pad_token_id = pad_token_id

    def __len__(self):
        return len(self.chosen_dataset)

    def __getitem__(self, idx):
        return self.chosen_dataset[idx]["input_ids"], self.chosen_dataset[idx]["attention_mask"], \
               self.chosen_dataset[idx]["action_mask"], \
               self.reject_dataset[idx]["input_ids"], self.reject_dataset[idx]["attention_mask"], \
               self.reject_dataset[idx]["action_mask"]


class DataCollatorForCritic:

    def __call__(self, data):
        return_value = {"lengths": [len(x['input_ids']) for x in data]}
        for k in ["input_ids", "attention_mask", "action_mask", "reward"]:
            return_value[k] = torch.concat([x[k] for x in data], dim=0)
        return return_value


class DataCollatorDPO:

    def __call__(self, data):
        return {"input_ids": torch.stack([f[0] for f in data] + [f[3] for f in data], dim=0),
                "attention_mask": torch.stack([f[1] for f in data] + [f[4] for f in data], dim=0),
                "action_mask": torch.stack([f[2] for f in data] + [f[5] for f in data], dim=0)}
