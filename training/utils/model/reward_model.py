# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import copy
from torch import nn
from transformers import AutoModel, AutoModelForCausalLM


class RewardModel(nn.Module):

    def __init__(self, base_model, tokenizer, with_sft=False, bias=False):
        super().__init__()
        self.config = base_model.config
        self.with_sft = with_sft
        if self.with_sft:
            self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)
            self.loss_fct = nn.CrossEntropyLoss()
        if hasattr(self.config, "word_embed_proj_dim"):
            self.v_head = nn.Linear(self.config.word_embed_proj_dim, 1, bias=bias)
        else:
            self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
            self.v_head = nn.Linear(self.config.n_embd, 1, bias=bias)
        self.transformer = base_model
        self.PAD_ID = tokenizer.pad_token_id

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, tokenizer, config=None, with_sft=False, bias=False):
        ckpt = torch.load(pretrained_name_or_path + "/pytorch_model.bin")
        for key in list(ckpt.keys()):
            if "transformer." in key:
                ckpt[key.replace("transformer.", "")] = ckpt[key]
                del ckpt[key]
        rm = cls(AutoModel.from_pretrained(pretrained_name_or_path, state_dict=ckpt, config=config), tokenizer, with_sft=with_sft, bias=bias)
        for key in list(ckpt.keys()):
            if "v_head" in key or "out_proj" in key:
                if "weight" in key:
                    rm.v_head.weight.data = copy.deepcopy(ckpt[key].data)
                elif "bias" in key:
                    rm.v_head.bias.data = copy.deepcopy(ckpt[key].data)
            if with_sft and "lm_head" in key:
                rm.lm_head.weight.data = copy.deepcopy(ckpt[key].data)
        return rm

    def gradient_checkpointing_enable(self):
        self.transformer.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.transformer.gradient_checkpointing_disable()

    def get_rewards(self, input_ids, attention_mask, use_cache=False):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask, use_cache=use_cache)[0]
        rewards = self.v_head(hidden_states.to(dtype=self.v_head.weight.dtype)).squeeze(-1)
        return hidden_states, rewards

    def forward(self, input_ids=None, attention_mask=None, use_cache=False, return_dict=False):
        bs = input_ids.shape[0] // 2
        seq_len = input_ids.shape[1]
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask, use_cache=use_cache)[0]
        rewards = self.v_head(hidden_states.to(dtype=self.v_head.weight.dtype)).squeeze(-1)
        # chosen_hidden_states, chosen_rewards = self.get_rewards(input_ids[:bs], attention_mask[:bs], use_cache)
        # _, rejected_rewards = self.get_rewards(input_ids[bs:], attention_mask[bs:], use_cache)
        chosen_mean_scores = []
        rejected_mean_scores = []
        rm_loss = 0
        for i in range(bs):
            chosen_id = input_ids[i]
            rejected_id = input_ids[i + bs]
            chosen_reward = rewards[i]
            rejected_reward = rewards[i + bs]
            c_inds = (chosen_id == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
            check_divergence = (chosen_id != rejected_id).nonzero()

            if len(check_divergence) == 0:
                end_ind = rejected_reward.size(-1)
                divergence_ind = end_ind - 1
                r_ind = c_ind
            else:
                # Check if there is any padding otherwise take length of sequence
                r_inds = (rejected_id == self.PAD_ID).nonzero()
                r_ind = r_inds[0].item() if len(r_inds) > 0 else seq_len
                end_ind = max(c_ind, r_ind)
                divergence_ind = check_divergence[0]
            assert divergence_ind > 0
            c_truncated_reward = chosen_reward[divergence_ind:end_ind]
            r_truncated_reward = rejected_reward[divergence_ind:end_ind]
            chosen_mean_scores.append(chosen_reward[c_ind - 1])  #use the end score for refrnence
            rejected_mean_scores.append(rejected_reward[r_ind - 1])
            rm_loss += -torch.log(torch.sigmoid(c_truncated_reward - r_truncated_reward)).mean()

        rm_loss = rm_loss / bs
        if self.with_sft:
            logits = self.lm_head(chosen_hidden_states)
            shift_logits = logits[..., :-1, :].contiguous().view(-1, self.config.vocab_size)
            shift_labels = input_ids[:bs, 1:].contiguous().view(-1).to(shift_logits.device)
            sft_loss = self.loss_fct(shift_logits, shift_labels)
        else:
            sft_loss = 0
        chosen_mean_scores = torch.stack(chosen_mean_scores)
        rejected_mean_scores = torch.stack(rejected_mean_scores)
        if return_dict:
            return {"loss": rm_loss, 
                    "sft_loss": sft_loss, 
                    "chosen_mean_scores": chosen_mean_scores,
                    "rejected_mean_scores": rejected_mean_scores,
                    "chosen_ids": input_ids[:bs],
                    "rejected_ids": input_ids[bs:]}
        return rm_loss, sft_loss, chosen_mean_scores, rejected_mean_scores
    
    def forward_score(self, input_ids, attention_mask=None):
        hidden_states = self.transformer(input_ids, attention_mask=attention_mask)[0]
        rewards = self.v_head(hidden_states.to(dtype=self.v_head.weight.dtype)).squeeze(-1)
        return torch.gather(rewards, dim=-1, index=attention_mask.sum(-1).unsqueeze(0) - 1).squeeze(0)
        # seq_len = input_ids.shape[1]
        # chosen_mean_scores = []
        # for i in range(input_ids.shape[0]):
        #     chosen_id = input_ids[i]
        #     chosen_reward = rewards[i]
        #     c_inds = (chosen_id == self.PAD_ID).nonzero()
        #     c_ind = c_inds[0].item() if len(c_inds) > 0 else seq_len
        #     chosen_mean_scores.append(float(chosen_reward[c_ind - 1]))
        # return chosen_mean_scores

    def forward_value(self, input_ids, attention_mask=None, return_value_only=False, prompt_length=0, use_cache=False):
        transformer_outputs = self.transformer(input_ids, attention_mask=attention_mask, use_cache=use_cache) # bs, l1+l2, d
        hidden_states = transformer_outputs[0]
        values = self.v_head(hidden_states).squeeze(-1) # bs, l
        # print(f"values.shape, {values.shape}")
        if return_value_only:
            return values
        else:
            assert prompt_length > 1, "prompt_length must be greater than 1 to help select the end score"
            bs = values.size(0)
            seq_len = input_ids.shape[1]
            chosen_end_scores = []
            for i in range(bs):
                input_id = input_ids[i]
                value = values[i]
                c_inds = (input_id[prompt_length:] == self.PAD_ID).nonzero()
                # print(f"{self.PAD_ID},{input_id[prompt_length:].tolist()}")
                c_ind = c_inds[0].item() + prompt_length if len(c_inds) > 0 else seq_len
                # print(f"{c_inds.shape},{c_ind}")
                chosen_end_scores.append(value[c_ind - 1])
            return {
                "values": values,
                "chosen_end_scores": torch.stack(chosen_end_scores),
            }
