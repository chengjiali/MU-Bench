#!/bin/bash

set -e
declare -a backbone=('swin-tiny' 'swin-base' 'mobilenet_v1' 'mobilenet_v2' 'convnext-base-224' 'convnext-base-224-22k')
# 'resnet-18' 'resnet-34' 'resnet-50' 'vit-base-patch16-224' 'vit-large-patch16-224'
declare -a data=('mnist' 'cifar10' 'cifar100')
declare -a seed=("42" "87" "21" "100" "13")


for b in "${backbone[@]}"
do
    for d in "${data[@]}"
    do
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=3 python train/image/run_image_classification.py train/image/configs/"${b}"/"${d}"_42.json &
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=4 python train/image/run_image_classification.py train/image/configs/"${b}"/"${d}"_87.json &
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=5 python train/image/run_image_classification.py train/image/configs/"${b}"/"${d}"_21.json &
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=6 python train/image/run_image_classification.py train/image/configs/"${b}"/"${d}"_100.json &
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=7 python train/image/run_image_classification.py train/image/configs/"${b}"/"${d}"_13.json &
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
    done
done
