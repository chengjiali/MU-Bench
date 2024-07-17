import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['tiny_imagenet']
backbones = ['sd']

general_methods = ['retrain', 'ft', 'neggrad', 'random_label', 'fisher', 'l-codec', 'bad_teaching', 'scrub', 'salul']
methods = [] + general_methods


def get_full_model_name(m):
    if m == 'sd':
        m = 'CompVis/stable-diffusion-v1-4'

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 2,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "per_device_eval_batch_size": 2,
    "learning_rate": 1e-5,
    "warmup_steps": 0,
    # "save_total_limit": 1,
    # "load_best_model_at_end": True,
    # "metric_for_best_model": "accuracy",
    # "metric_name": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    "push_to_hub": False,
    "seed": 42,
    "use_ema": True,
    "resolution": 64, 
}


s = 42
for b in backbones:
    for d in datasets:
        config = copy.deepcopy(template)
        out_dir = f'{b}'
        out_name = f'{d}_{s}'
        os.makedirs(f'configs/{out_dir}', exist_ok=True)

        config['model_name_or_path'] = get_full_model_name(b)
        config['dataset_name'] = 'tiny_imagenet'

        config['seed'] = s
        config['output_dir'] = f'image_gen/train/checkpoint/{out_dir}/{out_name}'
        config['hub_model_id'] = f'{b}-{d}-{s}'

        with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
            json.dump(config, f, indent=4)
