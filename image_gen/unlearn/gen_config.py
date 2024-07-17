import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['tiny_imagenet']
backbones = ['sd']

del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]
methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salul']


def get_full_model_name(m):
    if m == 'sd':
        m = 'CompVis/stable-diffusion-v1-4'

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "num_train_epochs": 5,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 16,
    "per_device_eval_batch_size": 4,
    "learning_rate": 1e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "score",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    "push_to_hub": False,
    "seed": 42,
    "use_ema": True,
    "resolution": 64, 
}


s = 42
for dr in del_ratio:
    for m in methods:
        for b in backbones:
            for d in datasets:
                config = copy.deepcopy(template)
                out_dir = f'{b}/{m}/{dr}'
                out_name = f'{d}_{s}'
                os.makedirs(f'configs/{out_dir}', exist_ok=True)

                config['unlearn_method'] = m
                config['del_ratio'] = dr
                
                config['model_name_or_path'] = get_full_model_name(b)
                config['dataset_name'] = 'zh-plus/tiny-imagenet'
                config['seed'] = s

                config['seed'] = s
                config['output_dir'] = f'image_gen/unlearn/checkpoint/{out_dir}/{out_name}'
                config['hub_model_id'] = f'{b}-{m}-{dr}-{d}-{s}'

                with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
                    json.dump(config, f, indent=4)
