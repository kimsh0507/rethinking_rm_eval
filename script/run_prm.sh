#!/bin/bash

#  peiyi9979/math-shepherd-mistral-7b-prm,  ScalableMath/llemma-7b-prm-prm800k-level-1to3-hf,  GAIR/ReasonEval-7B, GAIR/ReasonEval-34B, Qwen/Qwen2.5-Math-PRM-7B
# Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B
# "Math_Shepherd", "Easy_to_hard", "ReasonEval_7B", "ReasonEval_34B", "QwenMathPRM", "Skywork_PRM"

MODEL_NAME="peiyi9979/math-shepherd-mistral-7b-prm"
SAVE_NAME="Math_Shepherd"

SAVE_PATH="results/RewardMATH/${SAVE_NAME}_reward.json"

CUDA_VISIBLE_DEVICES=0 python src/inference_reward.py \
    --input_path=dataset/benchmark/RewardMATH_direct.json \
    --save_path=$SAVE_PATH \
    --model_name=$MODEL_NAME \
    --model_type=prm \
    --num_sample=10