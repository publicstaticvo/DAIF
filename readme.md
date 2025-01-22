## Step 0: Installation
```commandline
pip install -r requirements.txt
```

## Step 1: Data Collection

### Step 1.1: Problem Difficulty Assessment

1. Process the download data. `./data_collect/data/origin/` stores all training data for the project. Follow the preprocessing in `data_collect/data.ipynb`.

2. Run
```commandline
python -m torch.distributed.launch --nproc_per_nodes 4 inference/chatbot.py \
       --model_name_or_path /mnt/ewwe/yts/llm/models/lmsys-vicuna-7b-v1.1 \
       --data_path /mnt/ewwe/yts/llm/daif/data_collect/data/origin/problems.txt \
       --output_path /mnt/ewwe/yts/llm/daif/data_collect/data/initial_answers/ppl \
       --mode ppl --max_prompt_seq_len 1024 --max_new_tokens 512
```
to generate initial answers. You may need answers from two distinct models `lmsys/vicuna-7b-v1.1` and `lmsys/vicuna-13b-v1.1` to collect the preference feedback. You can download those models from [huggingface.co](https://huggingface.co).

### Step 1.2: Grouping

Follow `data_collect/data.ipynb`

### Step 1.3: Request for Feedbacks

```commandline
python data/request.py --input_files /mnt/ewwe/yts/llm/daif/data/initial_answers/group.txt \
                       --output_file /mnt/ewwe/yts/llm/daif/data/feedback/feedbacks.txt \
                       --base_model vicuna-7b-v1.1,vicuna-13b-v1.1 \
                       --api_type gpt-3.5-turbo
```

### Step 1.4: Collect Raw Feedbacks

Follow `data_collect/data.ipynb` to collect raw feedbacks.

### Step 1.5: Generate Self-Improved Response for Critique Feedback

Run
```commandline
python inference/chatbot.py --model_name_or_path /mnt/ewwe/yts/llm/models/lmsys-vicuna-7b-v1.1 \
                            --data_path /mnt/ewwe/yts/llm/daif/data_collect/data/feedback/critique.txt \
                            --output_path /mnt/ewwe/yts/llm/daif/data_collect/data/feedback/improve \
                            --mode generate --max_prompt_seq_len 1536 --max_new_tokens 512
```
to generate improved answers based on critique feedback.

### Step 1.6: Set up Train and Valid Datasets for RLHF-RM and DPO

Follow `data_collect/data.ipynb`

## Step 2: Training

Remember to fill in the path of your collected data in the training script first.

To run PPO training, run `reward_model_finetuning/run.sh` first. Then, select the checkpoint you want to use for PPO training and run `rlhf_finetuning/run.sh`.

To run DPO training, just simply run `direct_po_finetuning/run.sh`.

## Step 3: Evaluation

Use your saved checkpoint to generate on test dataset

```commandline
python inference/chatbot.py --model_name_or_path /PATH/OF/SAVED/CHECKPOINT \
                            --input_file /PATH/OF/TEST/DATASET/ \
                            --additional_human_stopword \
                            --max_prompt_seq_len 768 \
                            --max_new_tokens 512 \
                            --mode generate
```

Test your results with reward model

```commandline
python inference/chatbot.py --model_name_or_path /PATH/OF/SAVED/CHECKPOINT \
                            --input_file /PATH/OF/TEST/RESULTS/ \
                            --mode rm --ADDITIONAL_ARGUMENTS
```

### Additional Arguments

- `max_new_tokens`: Max number of generated tokens
- `max_prompt_seq_len`: Max number of input prompt tokens, for `--mode` == `generate` or `ppl`
- `max_seq_len`: Max number of input tokens, for `--mode` == `rm`
- `additional_human_stopword`: Will cut off contents after "\n\nHuman:" in generation results.
