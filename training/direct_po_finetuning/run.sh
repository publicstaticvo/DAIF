#！ /bin/bash
OUTPUT=dpo

deepspeed main.py \
   --data_path /DATA/PATH \
   --model_name_or_path /BASE/MODEL/PATH \
   --per_device_train_batch_size 8 \
   --ptx_path /PTX/PATH \
   --ptx_coef 10 \
   --max_seq_len 1536 \
   --max_prp_len 1024 \
   --learning_rate 1e-6 \
   --weight_decay 0.0 \
   --num_train_epochs 5 \
   --lr_scheduler_type linear \
   --gradient_checkpointing \
   --gradient_accumulation_steps 1 \
   --num_warmup_steps 100 \
   --deepspeed --seed 42 \
   --zero_stage 3 \
   --output_dir $OUTPUT \
   &> ../logs/dpo/$OUTPUT.log 2>&1
