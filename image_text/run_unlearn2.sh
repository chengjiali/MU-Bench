#!/bin/bash

set -e
declare -a data=('nlvr2')
declare -a method=('bad_teaching')
declare -a delratio=('1.0' '2.0' '3.0' '4.0' '5.0' '6.0' '7.0' '8.0' '9.0' '10.0')
sd=42
d='nlvr2'
b='vilt'


# for b in "${backbone[@]}"
# do
for m in "${method[@]}"
do
    for dr in "${delratio[@]}"
    do
    # for d in "${data[@]}"
    # do
        if [ ! -f audio/unlearn/checkpoint/"${b}"/"${m}"/"${dr}"/"${d}"_42/all_results.json ]; then
            echo "Not Found"
            WANDB_MODE=offline CUDA_VISIBLE_DEVICES=4 python image_text/unlearn/run_image_text_classification.py image_text/unlearn/configs/"${b}"/"${m}"/"${dr}"/"${d}"_42.json &
        fi

        trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
        wait
    # done
    done
done
# done
