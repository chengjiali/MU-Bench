#!/bin/bash

set -e
declare -a backbone=('videomae-base' 'videomae-large')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
s=42
d='ucf101'

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for dr in "${delratio[@]}"
        do
            if [ ! -f video/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
                echo "${b} ${m} ${dr}"
                WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=4 python video/unlearn/run_unlearn.py video/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json
            fi
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
        done
    done
done