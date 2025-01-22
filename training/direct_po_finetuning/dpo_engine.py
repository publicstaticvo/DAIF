import time
import torch
import deepspeed
from deepspeed.ops.adam import FusedAdam
from deepspeed.ops.adam import DeepSpeedCPUAdam
from transformers import AutoModelForCausalLM, get_scheduler
from utils.ds_utils import get_train_ds_config, get_eval_ds_config
from utils.module.lora import convert_linear_layer_to_lora, only_optimize_lora_parameters
from utils.model.model_utils import create_hf_model, create_critic_model
from utils.utils import get_optimizer_grouped_parameters, freeze_bottom_causal_layers


def log_init(model_name, stime=None):
    if torch.distributed.get_rank() == 0:
        tag = "start" if stime is None else "end"
        suffix = "ing" if stime is None else "ed"
        duration = ""
        if stime is not None:
            duration = "(duration: {:.2f}s)".format(time.time() - stime)
        msg = f"[{tag}] Initializ{suffix} {model_name} Model [{tag}] {duration}"
        stars = (90 - len(msg)) // 2
        extra_star = "*" if (90 - len(msg)) % 2 == 1 else ""
        print("*" * stars + msg + "*" * stars + extra_star)
        return time.time()


class DeepSpeedDPOEngine:

    def __init__(self, actor_model_name_or_path, tokenizer, args, num_total_iters):
        self.args = args
        self.num_total_iters = num_total_iters
        self.tokenizer = tokenizer
        self.actor = self._init_actor(actor_model_name_or_path=actor_model_name_or_path)
        # freeze_bottom_causal_layers(self.actor, args.num_actor_layers_unfrozen)
        self.ref = self._init_ref(actor_model_name_or_path=actor_model_name_or_path)
        self.ref.requires_grad_(False)
        self.actor_ema = None

    def _init_actor(self, actor_model_name_or_path):
        stime = log_init("Actor")
        # DS Config
        ds_config = get_train_ds_config(
            offload=self.args.offload,
            stage=self.args.zero_stage,
            enable_hybrid_engine=self.args.enable_hybrid_engine,
            inference_tp_size=self.args.inference_tp_size,
            release_inference_cache=self.args.release_inference_cache,
            pin_parameters=(not self.args.unpin_actor_parameters),
            tp_gather_partition_size=self.args.tp_gather_partition_size)
        ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_config['train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size()
        # Model
        actor_model = create_hf_model(AutoModelForCausalLM, actor_model_name_or_path, self.tokenizer, ds_config)
        # LoRA
        if self.args.lora_dim > 0:
            actor_model = convert_linear_layer_to_lora(actor_model, self.args.lora_module_name, self.args.lora_dim)
            if self.args.only_optimize_lora:
                actor_model = only_optimize_lora_parameters(actor_model)
        # Optimizer
        AdamOptimizer = DeepSpeedCPUAdam if self.args.offload else FusedAdam
        optim_params = get_optimizer_grouped_parameters(actor_model, self.args.weight_decay)
        optim = AdamOptimizer(optim_params, lr=self.args.learning_rate, betas=(0.9, 0.95))
        # LR Scheduler
        lr_scheduler = get_scheduler(
            name=self.args.lr_scheduler_type,
            optimizer=optim,
            num_warmup_steps=self.args.num_warmup_steps,
            num_training_steps=self.num_total_iters,
        )
        actor_engine, *_ = deepspeed.initialize(model=actor_model, optimizer=optim, lr_scheduler=lr_scheduler, config=ds_config)
        log_init("Actor", stime=stime)
        return actor_engine

    def _init_ref(self, actor_model_name_or_path):
        stime = log_init("Ref")
        ds_config = get_eval_ds_config(self.args.offload_reference_model, 3 if self.args.zero_stage == 3 else 0)
        ds_config['train_micro_batch_size_per_gpu'] = self.args.per_device_train_batch_size
        ds_config['train_batch_size'] = self.args.per_device_train_batch_size * torch.distributed.get_world_size()
        ref_model = create_hf_model(AutoModelForCausalLM, actor_model_name_or_path, self.tokenizer, ds_config)
        ref_engine, *_ = deepspeed.initialize(model=ref_model, config=ds_config)
        log_init("Ref", stime=stime)
        return ref_engine
