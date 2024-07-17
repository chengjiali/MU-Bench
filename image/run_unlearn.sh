#!/bin/bash

# set -e
declare -a backbone=('swin-tiny' 'swin-base' 'mobilenet_v2' 'convnext-base-224')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('1.0' '2.0' '3.0' '4.0' '5.0' '6.0' '7.0' '8.0' '9.0' '10.0')
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
        dr=1.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi
        
        dr=2.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi
        
        dr=3.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi
        
        dr=4.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi
        
        dr=5.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=6.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=5 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=7.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=8.0
        if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=7 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        # b='vit-base-patch16-224'
        # dr=9.0
        # if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
        #     echo "Not Found"
        #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        # fi

        # b='vit-base-patch16-224'
        # dr=10.0
        # if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
        #     echo "Not Found"
        #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=1 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        # fi

        # b='vit-large-patch16-224'
        # dr=9.0
        # if [ ! -f image/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
        #     echo "Not Found"
        #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python image/unlearn/run_image_classification.py image/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        # fi

        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
        # done
    done
done
# done
