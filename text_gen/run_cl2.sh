#!/bin/bash

set -e
declare -a data=('samsum')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a method=('scrub')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
sd=42
d='samsum'
b='t5-large'


for m in "${method[@]}"
do
    # for dr in "${delratio[@]}"
    # do
    # dr=2.0
    # if [ ! -f text_gen/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/ood_results.json ]; then
    #     echo "Not Found"
    #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 python text_gen/unlearn/run_cl.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
    # fi

    dr=4.0
    if [ ! -f text_gen/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/ood_results.json ]; then
        echo "Not Found"
        WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4,5,6 python text_gen/unlearn/run_cl.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json 
    fi

    # dr=6.0
    # if [ ! -f text_gen/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/ood_results.json ]; then
    #     echo "Not Found"
    #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0,1,2,3 python text_gen/unlearn/run_cl.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
    # fi

    # dr=8.0
    # if [ ! -f text_gen/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/ood_results.json ]; then
    #     echo "Not Found"
    #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4,5,6,7 python text_gen/unlearn/run_cl.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
    # fi

    # dr=10.0
    # if [ ! -f text_gen/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/ood_results.json ]; then
    #     echo "Not Found"
    #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python text_gen/unlearn/run_cl.py text_gen/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
    # fi
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    wait
    # done
done
