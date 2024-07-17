#!/bin/bash

set -e
declare -a backbone=('videomae-base')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
s=42
d='ucf101'
b='videomae-base'

for m in "${method[@]}"
do
    for dr in "${delratio[@]}"
    do
        if [ ! -f video/unlearn/lora_20_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr}"
            WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=3 python video/unlearn/run_lora.py video/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json
        fi
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
    done
done
