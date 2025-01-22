#!/bin/bash

OUT=rm

deepspeed main.py \
   --data_path /TRAIN/DATA/PATH  \
   --validate_dataset /VALID/DATA/PATH  \
   --model_name_or_path /BASE/MODEL/PATH \
   --per_device_train_batch_size 16 \
   --per_device_eval_batch_size 16 \
   --max_seq_len 1280 \
   --learning_rate 1e-5 \
   --weight_decay 0.0 \
   --num_train_epochs 5 \
   --gradient_accumulation_steps 1 \
   --lr_scheduler_type linear \
   --num_warmup_steps 50 \
   --seed 42 \
   --gradient_checkpointing \
   --zero_stage 3 \
   --deepspeed \
   --output_dir $OUT \
   &> ../logs/rm/$OUT.log 2>&1