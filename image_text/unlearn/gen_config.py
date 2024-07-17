import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['nlvr2']
backbones = ['vilt']
del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]

general_methods = ['retrain', 'ft', 'neggrad', 'random_label', 'fisher', 'l-codec', 'bad_teaching', 'scrub', 'salul']
methods = [] + general_methods



def get_full_model_name(m, d):
    if 'vilt' in m:
        m = 'dandelin/vilt-b32-finetuned-' + d

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "nlvr2",
    "num_train_epochs": 3,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 256,
    "learning_rate": 5e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    # "push_to_hub": True,
    "ignore_mismatched_sizes": True,
    "remove_unused_columns": False,
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
                    
                    config['model_name_or_path'] = get_full_model_name(b, d)
                    config['dataset_name'] = d

                    config['seed'] = s
                    config['output_dir'] = f'image_text/unlearn/checkpoint/{out_dir}/{out_name}'
                    config['hub_model_id'] = f'{b}-{m}-{dr}-{d}-{s}'


                    with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
                        json.dump(config, f, indent=4)


