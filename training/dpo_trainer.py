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

from utils.utils import print_rank_0

import trl.trainer.dpo_trainer


def gather_log_probs(logits, labels):
    # logits: B,L-1,32000, labels: B,L-1
    log_probs = logits.log_softmax(-1)
    log_probs_labels = log_probs.gather(dim=-1, index=labels.unsqueeze(-1))
    return log_probs_labels.squeeze(-1)  # B, L-1


class DeepSpeedDPOTrainer:

    def __init__(self, dpo_engine, args):
        self.args = args
        self.dpo_engine = dpo_engine
        self.actor = self.dpo_engine.actor
        self.ref = self.dpo_engine.ref
        self.kl_ctl = args.kl_ctl
        self.steps = 0

    def train_iter(self, batch):
        # 1 输入模型
        input_ids = batch["input_ids"].cuda()
        attention_mask = batch["attention_mask"].cuda()
        output = self.actor(input_ids, attention_mask=attention_mask)
        output_ref = self.ref(input_ids, attention_mask=attention_mask)
        log_probs = gather_log_probs(output.logits[:, :-1, :], input_ids[:, 1:])
        log_probs_ref = gather_log_probs(output_ref.logits[:, :-1, :], input_ids[:, 1:])
        # 2 输入的数据一共有2B条，其中B条chosen和B条reject，将其分开
        bs = input_ids.shape[0] // 2
        chosen, reject = torch.split(log_probs, bs, 0)
        chosen_ref, reject_ref = torch.split(log_probs_ref, bs, 0)
        action_mask, reject_action_mask = torch.split(batch["action_mask"].cuda(), bs, 0)
        # 3 算loss
        chosen, chosen_ref = map(lambda x: torch.sum(x * action_mask, dim=-1), [chosen, chosen_ref])
        reject, reject_ref = map(lambda x: torch.sum(x * reject_action_mask, dim=-1), [reject, reject_ref])
        loss = -(self.kl_ctl * (chosen - chosen_ref - reject + reject_ref)).sigmoid().log().mean()
        self.steps += 1
        if self.steps == 1:
            print(chosen.shape, chosen_ref.shape, reject.shape, reject_ref.shape)
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
