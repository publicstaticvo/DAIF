# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload=None,
                        stage=2,
                        steps_per_print=10,
                        enable_hybrid_engine=False,
                        inference_tp_size=1,
                        release_inference_cache=False,
                        pin_parameters=True,
                        tp_gather_partition_size=8):

    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4,
        "stage3_max_live_parameters": 3e7,
        "stage3_prefetch_bucket_size": 3e7
    }
    return {
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "steps_per_print": steps_per_print,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "opt_level": "O2",
            "loss_scale_window": 100,
            "min_loss_scale": 0
        },
        "gradient_clipping": 1.0,
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        }
    }


def get_eval_ds_config(offload, stage=0, steps_per_print=10, model_parallel=False):
    device = "cpu" if offload else "none"
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        }
    }
    return {
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True
        },
        "prescale_gradients": False,
        "wall_clock_breakdown": False,
    }
