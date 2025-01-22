#!/bin/bash

SYS=/mnt/ewwe/yts/llm

python inference/chatbot.py --model_name_or_path $SYS/ckpts_daif/llama-2-7b-sft/epoch1 \
                            --data_path $SYS/data/daif/problem.txt --batch_size 8 \
                            --outfile_name_or_path $SYS/data/daif/ppl-1 --mode generate \
                            --max_prompt_seq_len 1280 --max_new_tokens 512 --num_samples 0
# sleep 1m
# python inference/chatbot.py --model_name_or_path $SYS/ckpts_daif/llama-2-7b-sft/epoch2 \
#                             --data_path $SYS/data/daif/ppl-1.txt --batch_size 4 \
#                             --outfile_name_or_path $SYS/data/daif/ppl-2 --mode generate \
#                             --max_prompt_seq_len 1280 --max_new_tokens 512 --num_samples 2
# sleep 1m
# python inference/chatbot.py --model_name_or_path $SYS/ckpts_daif/llama-2-7b-sft-wd0.001/epoch1 \
#                             --data_path $SYS/data/daif/ppl-2.txt --batch_size 4 \
#                             --outfile_name_or_path $SYS/data/daif/ppl-3 --mode generate \
#                             --max_prompt_seq_len 1280 --max_new_tokens 512 --num_samples 2
# sleep 1m
# python inference/chatbot.py --model_name_or_path $SYS/ckpts_daif/llama-2-7b-sft-wd0.001/epoch2 \
#                             --data_path $SYS/data/daif/ppl-3.txt --batch_size 4 \
#                             --outfile_name_or_path $SYS/data/daif/ppl --mode generate \
#                             --max_prompt_seq_len 1280 --max_new_tokens 512 --num_samples 2
# rm $SYS/data/daif/ppl-1.txt $SYS/data/daif/ppl-2.txt $SYS/data/daif/ppl-3.txt
