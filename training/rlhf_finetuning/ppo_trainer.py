# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import torch
import torch.nn.functional as F
import time
import sys
import os
import deepspeed
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from utils.utils import print_rank_0


def encode(tokenizer, seq, pr, pa, length):
    seq = tokenizer(seq, padding="longest", return_tensors="pt")
    input_ids = torch.cat([pr, seq.input_ids[:, 1:length + 1].to(pr.device)], dim=-1)
    attention_mask = torch.cat([pa, seq.attention_mask[:, 1:length + 1].to(pr.device)], dim=-1)
    return input_ids, attention_mask


def get_model_norm(model):
    with torch.no_grad():
        total = 0.0
        for param in model.parameters():
            should_gather = hasattr(
                param,
                'ds_id') and param.ds_status == ZeroParamStatus.NOT_AVAILABLE
            with deepspeed.zero.GatheredParameters(param, enabled=should_gather):
                total += float(param.float().norm())

    return total


def gather_log_probs(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)


class DeepSpeedPPOTrainer:

    def __init__(self, rlhf_engine, args):
        self.rlhf_engine = rlhf_engine
        self.actor_model = self.rlhf_engine.actor
        self.critic_model = self.rlhf_engine.critic
        self.ref_model = self.rlhf_engine.ref
        self.reward_model = self.rlhf_engine.reward
        self.tokenizer = self.rlhf_engine.tokenizer
        self.tokenizer_critic = self.rlhf_engine.tokenizer_critic
        self.args = args
        self.max_answer_seq_len = args.max_answer_seq_len
        self.saved_experience_path = args.saved_experience_path

        # Those value can be changed
        self.kl_ctl = 0.1
        self.clip_reward_value = 10
        self.cliprange = 0.2
        self.cliprange_value = 0.2
        self.gamma = 1.0
        self.lam = 0.95
        self.experience_count = 0

    def _generate_sequence(self, pr, pr_att_mask):
        with torch.no_grad():
            seq = self.actor_model.module.generate(pr, attention_mask=pr_att_mask, max_new_tokens=self.max_answer_seq_len)
        ans = self.tokenizer.batch_decode(seq[:, pr.shape[1]:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return seq, ans

    def generate_experience(self, prompts):
        self.eval()
        pr, pr_att_mask = prompts["prompts"], prompts["prompt_att_mask"]
        seq, ans = self._generate_sequence(pr, pr_att_mask)
        for i in range(len(ans)):
            if "</s>" in ans[i]:
                ans[i] = ans[i][:ans[i].index("</s>")]
        ans = [x if len(x) > 0 else "</s>" for x in ans]
        actor_ids, actor_attention_mask = encode(self.tokenizer, ans, pr, pr_att_mask, self.max_answer_seq_len)
        with torch.no_grad():
            output = self.actor_model(actor_ids, attention_mask=actor_attention_mask)
            output_ref = self.ref_model(actor_ids, attention_mask=actor_attention_mask)
            reward_score = self.reward_model.forward_value(actor_ids, attention_mask=actor_attention_mask,
                                                           prompt_length=pr.shape[1])['chosen_end_scores'].detach()
            values = self.critic_model.forward_value(actor_ids, attention_mask=actor_attention_mask,
                                                     prompt_length=pr.shape[1], return_value_only=True).detach()[:, :-1]
    
        if self.saved_experience_path:
            p = self.tokenizer.batch_decode(pr.tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
            for q, a, r in zip(p, ans, reward_score):
                self.experience_count += 1
                if self.saved_experience_path == "debug":
                    print_rank_0(f"-- Experience {self.experience_count} Reward: {r} -- \nInput: {q}\nOutput: {a}\n\n", torch.distributed.get_rank())
                else:
                    with open(f"{self.saved_experience_path}-{torch.distributed.get_rank()}.txt", "a+") as f:
                        f.write(f"-- Experience {self.experience_count} Reward: {r} -- \nInput: {q}\nOutput: {a}\n\n")
        logits = output.logits
        logits_ref = output_ref.logits
        self.train()

        return {
            'prompts': pr,
            'logprobs': gather_log_probs(logits[:, :-1, :], actor_ids[:, 1:]),
            'ref_logprobs': gather_log_probs(logits_ref[:, :-1, :], actor_ids[:, 1:]),
            'value': values,
            'rewards': reward_score,
            'seq_ids': actor_ids,
            'seq_attention_mask': actor_attention_mask
        }

    def compute_rewards(self, prompts, log_probs, ref_log_probs, reward_score, action_mask):
        # prompts 16,768 log_probs 16,1023 ref_log_probs 16,1023 reward_score 16 action_mask 16,256
        kl_divergence_estimate = -self.kl_ctl * (log_probs - ref_log_probs)
        rewards = kl_divergence_estimate
        start = prompts.shape[1] - 1
        # rewards 16,1023 start 767 ends N
        ends = start + action_mask.sum(1)
        reward_clip = torch.clamp(reward_score, -self.clip_reward_value, self.clip_reward_value)
        batch_size = log_probs.shape[0]
        for j in range(batch_size):
            rewards[j, start:ends[j]][-1] += reward_clip[j]
        return rewards

    def train_rlhf(self, inputs):
        # train the rlhf mode here
        prompts = inputs['prompts']
        log_probs = inputs['logprobs']
        ref_log_probs = inputs['ref_logprobs']
        reward_score = torch.sigmoid(inputs['rewards'])  # BS * L，来自reward model
        values = inputs['value']  # BS * L，来自critic model
        actor_input_ids = inputs['seq_ids']
        actor_attention_mask = inputs['seq_attention_mask']
        start = prompts.size()[-1] - 1
        action_mask = actor_attention_mask[:, start + 1:]
        old_values = values
        with torch.no_grad():
            old_rewards = self.compute_rewards(prompts, log_probs, ref_log_probs, reward_score, action_mask)
            advantages, returns = self.get_advantages_and_returns(old_values, old_rewards, start)

        ### process the new outputs
        actor_prob = self.actor_model(input_ids=actor_input_ids, attention_mask=actor_attention_mask, use_cache=False).logits
        actor_log_prob = gather_log_probs(actor_prob[:, :-1, :], actor_input_ids[:, 1:])
        actor_loss = self.actor_loss_fn(actor_log_prob[:, start:], log_probs[:, start:], advantages, action_mask)
        self.actor_model.backward(actor_loss)
        self.actor_model.step()
        value = self.critic_model.forward_value(input_ids=actor_input_ids, attention_mask=actor_attention_mask,
                                                return_value_only=True, use_cache=False)[:, :-1]
        critic_loss = self.critic_loss_fn(value[:, start:], old_values[:, start:], returns,
                                          actor_attention_mask[:, start + 1:])
        self.critic_model.backward(critic_loss)
        self.critic_model.step()
        return actor_loss, critic_loss

    def actor_loss_fn(self, logprobs, old_logprobs, advantages, mask):
        ## policy gradient loss
        log_ratio = (logprobs - old_logprobs) * mask
        ratio = torch.exp(log_ratio)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / mask.sum()
        return pg_loss

    def critic_loss_fn(self, values, old_values, returns, mask):
        ## value loss
        values_clipped = torch.clamp(values, old_values - self.cliprange_value, old_values + self.cliprange_value,)
        vf_loss1 = (values - returns)**2
        vf_loss2 = (values_clipped - returns)**2
        vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / mask.sum()
        return vf_loss

    def get_advantages_and_returns(self, values, rewards, start):
        # Adopted from https://github.com/CarperAI/trlx/blob/main/trlx/models/modeling_ppo.py#L134
        lastgaelam = 0
        advantages_reversed = []
        length = rewards.size()[-1]
        for t in reversed(range(start, length)):
            nextvalues = values[:, t + 1] if t < length - 1 else 0.0
            delta = rewards[:, t] + self.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.gamma * self.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        returns = advantages + values[:, start:]
        return advantages.detach(), returns

    def _validate_training_mode(self):
        assert self.actor_model.module.training
        assert self.critic_model.module.training

    def _validate_evaluation_mode(self):
        assert not self.actor_model.module.training
        assert not self.critic_model.module.training
        assert not self.ref_model.module.training
        assert not self.reward_model.module.training

    def train(self):
        self.actor_model.train()
        self.critic_model.train()

    def eval(self):
        self.actor_model.eval()
        self.critic_model.eval()
        self.reward_model.eval()
        self.ref_model.eval()


class DeepSpeedPPOTrainerUnsupervised(DeepSpeedPPOTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        self._validate_training_mode()
        outputs = self.actor_model(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor_model.backward(unsup_coef * loss)
        self.actor_model.step()
        return loss
