#!/bin/bash

set -e
declare -a data=('samsum')
declare -a method=('neggrad' 'random_label' 'bad_teaching' 'scrub' 'salul')
declare -a delratio=('2.0' '4.0' '6.0' '8.0' '10.0')
sd=42
d='samsum'
b='t5-large'


# for m in "${method[@]}"
# do
    # for dr in "${delratio[@]}"
    # do
b='t5-small'
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python text_gen/train/run_ood.py text_gen/train/configs/"${b}"/"${d}"_42.json &

b='t5-base'
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=2 python text_gen/train/run_ood.py text_gen/train/configs/"${b}"/"${d}"_42.json &

# b='t5-large'
# WANDB_MODE=offline CUDA_VISIBLE_DEVICES=3,4 python text_gen/train/run_ood.py text_gen/train/configs/"${b}"/"${d}"_42.json &
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
wait
    # done
# done
