#!/bin/bash

set -e
declare -a backbone=('wav2vec2-base' 'wav2vec2-large' 'hubert-base' 'hubert-large')
declare -a data=('superb_ks')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a method=('salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
d='superb_ks'


for b in "${backbone[@]}"
do
    for m in "${method[@]}"
    do
        # for dr in "${delratio[@]}"
        # do
        dr=2.0
        if [ ! -f audio/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python audio/unlearn/run_audio_classification.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=4.0
        if [ ! -f audio/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python audio/unlearn/run_audio_classification.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi


        dr=6.0
        if [ ! -f audio/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3 python audio/unlearn/run_audio_classification.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        dr=8.0
        if [ ! -f audio/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "${b} ${m} ${dr}"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python audio/unlearn/run_audio_classification.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi


        # dr=10.0
        # if [ ! -f audio/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
        #     echo "${b} ${m} ${dr}"
        #     WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python audio/unlearn/run_audio_classification.py audio/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        # fi


        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
    done
done
