#!/bin/bash

set -e
declare -a backbone=('wav2vec2-base' 'wav2vec2-large' 'hubert-base' 'hubert-large')
declare -a backbone=('hubert-base')
declare -a data=('superb_ks')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
d='superb_ks'


for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for dr in "${delratio[@]}"
        do
            # if [ ! -f audio/unlearn/lora_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python audio/unlearn/run_lora.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
            # fi
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
        done
    done
done
