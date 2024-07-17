#!/bin/bash

set -e
declare -a backbone=('biobert' 'pubmedbert-abstract' 'pubmedbert-fulltext')
declare -a seed=("42" "87" "21" "100" "13")
declare -a data=('ddi' 'chem_prot')
m='bad_teaching'
sd=42
d='imdb'


for b in "${backbone[@]}"
do
    for s in "${seed[@]}"
    do
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 python text/train/run_classification.py text/train/configs/"${b}"/ddi_"${s}".json &
        WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=2 python text/train/run_classification.py text/train/configs/"${b}"/chem_prot_"${s}".json &
        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
    done
done
