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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def gather_log_probs(logits, labels, action_mask, use_average=False):
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    if use_average:
        return (log_probs_labels.squeeze(-1) * action_mask).mean(-1)  # B
    return (log_probs_labels.squeeze(-1) * action_mask).sum(-1)  # B


def compute_mse_loss(logits, logits_ref, rewards, lengths):
    logits, logits_ref, rewards = map(lambda x: x.split(lengths, 0), [logits, logits_ref, rewards])
    losses = 0
    for logit, logit_ref, rw in zip(logits, logits_ref, rewards):
        regularization = rw.exp().mean().log()
        losses += (logit - logit_ref - rw + regularization).pow(2).mean()
    return losses / len(logits)


def compute_exo_loss(logits, logits_ref, rewards, lengths, kl_ctl_f_theta):
    logits, logits_ref, rewards = map(lambda x: x.split(lengths, 0), [logits, logits_ref, rewards])
    losses = 0
    for logit, logit_ref, rw in zip(logits, logits_ref, rewards):
        f_theta_distribution = (kl_ctl_f_theta * (logit - logit_ref)).softmax(0)
        losses += (f_theta_distribution * (f_theta_distribution.log() - rw.log_softmax(0))).sum()
    return losses / len(logits)


class DeepSpeedCriticTrainer:

    def __init__(self, engine, args):
        self.args = args
        self.engine = engine
        self.actor = self.engine.actor
        self.ref = self.engine.ref
        self.kl_ctl = args.kl_ctl
        self.kl_ctl_f = args.kl_ctl_for_exo
        self.loss_type = args.loss_type
        self.use_average = args.use_average

    def train_iter(self, batch):
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        action_mask = batch["action_mask"].cuda()
        output = self.actor(input_ids, attention_mask=attention_mask)
        output_ref = self.ref(input_ids, attention_mask=attention_mask)
        reward = batch["reward"].to(dtype=output.logits.dtype, device=output.logits.device) / self.kl_ctl
        lengths = batch['lengths']
        log_probs = gather_log_probs(output.logits[:, :-1, :], input_ids[:, 1:], action_mask, self.use_average)
        log_probs_ref = gather_log_probs(output_ref.logits[:, :-1, :], input_ids[:, 1:], action_mask, self.use_average)
        if self.loss_type == "mse":
            loss = compute_mse_loss(log_probs, log_probs_ref, reward, lengths)
        elif self.loss_type == "exo":
            loss = compute_exo_loss(log_probs, log_probs_ref, reward, lengths, self.kl_ctl_f)
        else:
            raise ValueError(f"Invalid loss type: {self.loss_type}, must be one of \"mse\", \"exo\"")
        self.actor.backward(loss)
        self.actor.step()
        return loss

    def train(self):
        self.actor.train()

    def eval(self):
        self.actor.eval()

    def train_unsupervised(self, inputs, unsup_coef):
        # Train the unsupervised model here
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = self.actor(**inputs, use_cache=False)
        loss = outputs.loss
        self.actor.backward(unsup_coef * loss)
        self.actor.step()
        return loss
