#!/bin/bash

set -e
declare -a backbone=('t5-small')
declare -a seed=("42" "87" "21" "100" "13")
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
d='samsum'
b='t5-small'

for m in "${method[@]}"
do
    for dr in "${delratio[@]}"
    do
        if [ ! -f text_gen/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/samsum_42/all_results.json ]; then
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python text_gen/unlearn/run_lora.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/samsum_42.json &
        fi
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
    done
done
