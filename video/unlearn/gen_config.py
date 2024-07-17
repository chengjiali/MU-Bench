import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['ucf101']
backbones = ['videomae-small', 'videomae-base', 'videomae-large', 'videomae-huge']
del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]

general_methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salul']
methods = [] + general_methods


def get_full_model_name(m):
    if 'videomae-' in m:
        m = 'MCG-NJU/' + m

    return m


template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "ucf101",
    "num_train_epochs": 3,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 64,
    "learning_rate": 5e-5,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    # "push_to_hub": True,
    "remove_unused_columns": False,
}

d = 'ucf101'
s = 42
for dr in del_ratio:
    for m in methods:
        for b in backbones:
            config = copy.deepcopy(template)
            out_dir = f'{b}/{m}/{dr}'
            out_name = f'{d}_{s}'
            os.makedirs(f'configs/{out_dir}', exist_ok=True)

            config['unlearn_method'] = m
            config['del_ratio'] = dr
            
            config['model_name_or_path'] = get_full_model_name(b)
            config['dataset_name'] = d

            if 'small' in b:
                config["per_device_train_batch_size"] = 16
                config["gradient_accumulation_steps"] = 2
                config["per_device_eval_batch_size"] = 128
            
            elif 'base' in b:
                config["per_device_train_batch_size"] = 8
                config["gradient_accumulation_steps"] = 4
                config["per_device_eval_batch_size"] = 64

            elif 'large' in b:
                config["per_device_train_batch_size"] = 4
                config["gradient_accumulation_steps"] = 8
                config["per_device_eval_batch_size"] = 32

            elif 'huge' in b:
                config["per_device_train_batch_size"] = 2
                config["gradient_accumulation_steps"] = 16
                config["per_device_eval_batch_size"] = 16

            config['seed'] = s
            config['output_dir'] = f'video/unlearn/checkpoint/{out_dir}/{out_name}'
            config['hub_model_id'] = f'{b}-{m}-{dr}-{d}-{s}'

            with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
                json.dump(config, f, indent=4)
