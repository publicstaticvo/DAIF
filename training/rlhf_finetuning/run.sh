#! /bin/bash

OUTPUT=ppo

deepspeed main.py \
   --data_path /DATA/PATH1 /DATA/PATH2 \
   --actor_model_name_or_path /POLICY/BASE/MODEL/PATH \
   --critic_model_name_or_path /REWARD/MODEL/PATH \
   --ptx_path /PTX/DATA/PATH \
   --ptx_coef 27.8 \
   --per_device_train_batch_size 16 \
   --per_device_mini_train_batch_size 16 \
   --generation_batch_numbers 1 \
   --ppo_epochs 2 \
   --max_answer_seq_len 512 \
   --max_prompt_seq_len 768 \
   --actor_learning_rate 5e-6 \
   --critic_learning_rate 5e-6 \
   --actor_weight_decay 0.0 \
   --critic_weight_decay 0.0 \
   --num_train_epochs 1 \
   --save_interval 50 \
   --lr_scheduler_type cosine \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 20 \
   --deepspeed --seed 42 \
   --enable_hybrid_engine \
   --actor_gradient_checkpointing \
   --critic_gradient_checkpointing \
   --actor_zero_stage 3 \
   --critic_zero_stage 3 \
   --output_dir $OUTPUT \
   &> ../logs/ppo/$OUTPUT.log 2>&1
