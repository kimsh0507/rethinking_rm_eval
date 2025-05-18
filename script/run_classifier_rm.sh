#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python src/inference_reward.py \
    --input_path=dataset/benchmark/RMbench_math.json \
    --save_path=results/RMbench/Eurus-RM_reward.json \
    --model_name=openbmb/Eurus-RM-7b \
    --model_type=classifier \
    --chat_template=mistral \
    --trust_remote_code \
    --batch_size=8 \
    # --num_sample=8