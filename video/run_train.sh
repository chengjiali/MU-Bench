#!/bin/bash

set -e
declare -a backbone=('videomae-base')
s=42
d='ucf101'

for b in "${backbone[@]}"
do
    WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=4 python video/train/run_train.py video/train/configs/"${b}"/"${d}"_"${s}".json
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    wait
done
