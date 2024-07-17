#!/bin/bash

# set -e
declare -a backbone=('sd')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
sd=42
d='tiny_imagenet'


for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        # for dr in "${delratio[@]}"
        # do
        dr=2.0
        if [ ! -f image_gen/unlearn/lora_50_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} ${d}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 accelerate launch --mixed_precision="fp16" \
            image_gen/unlearn/run_lora.py image_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=4.0
        if [ ! -f image_gen/unlearn/lora_50_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} ${d}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 accelerate launch --mixed_precision="fp16" \
            image_gen/unlearn/run_lora.py image_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=6.0
        if [ ! -f image_gen/unlearn/lora_50_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} ${d}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 accelerate launch --mixed_precision="fp16" \
            image_gen/unlearn/run_lora.py image_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=8.0
        if [ ! -f image_gen/unlearn/lora_50_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} ${d}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 accelerate launch --mixed_precision="fp16" \
            image_gen/unlearn/run_lora.py image_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=10.0
        if [ ! -f image_gen/unlearn/lora_50_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} ${d}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 accelerate launch --mixed_precision="fp16" \
            image_gen/unlearn/run_lora.py image_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
        # done
    done
done
