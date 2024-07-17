#!/bin/bash

set -e
declare -a backbone=('t5-small')
declare -a seed=("42" "87" "21" "100" "13")
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')

d='samsum'

for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for dr in "${delratio[@]}"
        do
            if [ ! -f text_gen/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/dr_results.json ]; then
                echo "Not Found ${b} ${m} ${d} ${dr}"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=6 python text_gen/unlearn/run_summarization.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json 
            fi
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
        done
    done
done