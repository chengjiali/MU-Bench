#!/bin/bash

set -e
declare -a backbone=('wav2vec2-base' 'wav2vec2-large' 'hubert-base' 'hubert-large' 'whisper-tiny')
declare -a data=('superb_ks')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a method=('bad_teaching')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
d='superb_ks'


for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        for dr in "${delratio[@]}"
        do
            # if [ ! -f audio/unlearn/cl_checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr} ${d}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python audio/unlearn/run_cl.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json
            # fi
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            wait
        done
    done
done
