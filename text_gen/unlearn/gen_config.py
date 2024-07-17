import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['samsum']
backbones = ['t5-small', 't5-base', 't5-large', 't5-xl', 't5-xxl']
del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]

general_methods = ['retrain', 'ft', 'neggrad', 'random_label', 'fisher', 'l-codec', 'bad_teaching', 'scrub', 'salul']
methods = [] + general_methods


def get_full_model_name(m):
    if 't5-' in m:
        scale = m.split('-')[1]
        m = 'google/t5-v1_1-' + scale

    return m


template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "samsum",
    "num_train_epochs": 5,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 32,
    "learning_rate": 5e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "rougeL",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    # "push_to_hub": True,
    "remove_unused_columns": False,
    "model_name_or_path": 'google/t5-v1_1-small',

    "predict_with_generate": True,
    "source_prefix": "summarize: ",
    "max_source_length": 1024,
    "max_target_length": 128,
}

for dr in del_ratio:
    for m in methods:
        for b in backbones:
            for d in datasets:
                for s in seeds:
                    config = copy.deepcopy(template)
                    out_dir = f'{b}/{m}/{dr}'
                    out_name = f'{d}_{s}'
                    os.makedirs(f'configs/{out_dir}', exist_ok=True)

                    config['unlearn_method'] = m
                    config['del_ratio'] = dr
                    if m == 'neggrad':
                        config['learning_rate'] /= 5

                    config['num_train_epochs'] /= 2

                    if 'xxl' in b:
                        config['per_device_train_batch_size'] = 4
                        config['gradient_accumulation_steps'] = 8
                        config['per_device_eval_batch_size'] = 2

                    elif 'xl' in b:
                        config['per_device_train_batch_size'] = 16
                        config['gradient_accumulation_steps'] = 2
                        config['per_device_eval_batch_size'] = 4

                    elif 'large' in b:
                        config['per_device_train_batch_size'] = 4
                        config['gradient_accumulation_steps'] = 8
                        config['per_device_eval_batch_size'] = 8


                    config['model_name_or_path'] = get_full_model_name(b)
                    config['dataset_name'] = d

                    config['seed'] = s
                    config['output_dir'] = f'text_gen/unlearn/checkpoint/{out_dir}/{out_name}'
                    config['hub_model_id'] = f'{b}-{m}-{dr}-{d}-{s}'


                    with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
                        json.dump(config, f, indent=4)
