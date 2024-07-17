#!/bin/bash

set -e
declare -a backbone=('bert-base' 'distilbert-base' 'electra-base')
declare -a seed=("42" "87" "21" "100" "13")
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')

d='imdb'
for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for dr in "${delratio[@]}"
        do
            if [ ! -f text/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
                echo "Not Found ${b} ${m} ${d} ${dr}"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python text/unlearn/run_cl.py text/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
            fi
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
        done
    done
done


declare -a backbone=('biobert')

d='ddi'
for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for dr in "${delratio[@]}"
        do
            if [ ! -f text/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
                echo "Not Found ${b} ${m} ${dr} ${d}"
                WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python text/unlearn/run_cl.py text/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json 
            fi
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
        done
    done
done