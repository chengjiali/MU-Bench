#!/bin/bash

# set -e
declare -a backbone=('swin-tiny' 'swin-base' 'mobilenet_v2' 'convnext-base-224')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
sd=42
d='cifar100'
b='swin-base'


for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
    # for dr in "${delratio[@]}"
    # do
        # for d in "${data[@]}"
        # do
        
        # dr=2.0
        # if [ ! -f image/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
        #     echo "Not Found"
        #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python image/unlearn/run_lora.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json
        # fi
        
        dr=4.0
        if [ ! -f image/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python image/unlearn/run_lora.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi
        
        dr=6.0
        if [ ! -f image/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 python image/unlearn/run_lora.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=8.0
        if [ ! -f image/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 python image/unlearn/run_lora.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=10.0
        if [ ! -f image/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=7 python image/unlearn/run_lora.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
        # done
    done
done
# done
