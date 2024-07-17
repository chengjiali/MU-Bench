import os
import copy
import json
import argparse


datasets = ['nlvr2', 'flickr30k']
backbones = ['vilt']


def get_full_model_name(m, d):
    if 'vilt' in m:
        m = 'dandelin/vilt-b32-finetuned-' + d

    elif 'resnet-' in m:
        m = 'microsoft/' + m

    elif 'swin-' in m:
        m = 'microsoft/' + m + '-patch4-window7-224'

    elif 'mobilenet_' in m:
        m = 'google/' + m + '_1.0_224'

    elif 'convnext' in m:
        m = 'facebook/' + m

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "mnist",
    "num_train_epochs": 10,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 256,
    "learning_rate": 2e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    # "push_to_hub": True,
    "ignore_mismatched_sizes": True,
    "remove_unused_columns": False,
    'seed': 42,
}


s = 42
for b in backbones:
    for d in datasets:
        config = copy.deepcopy(template)
        out_dir = f'{b}'
        out_name = f'{d}_{s}'
        os.makedirs(f'configs/{out_dir}', exist_ok=True)
        
        config['model_name_or_path'] = get_full_model_name(b, d)
        if 'vilt' in b:
            config['do_train'] = False

        if d == 'flickr30k':
            config['metric_for_best_model'] = 'mean_recall'

        config['dataset_name'] = d
        config['seed'] = s
        config['output_dir'] = f'image_text/train/checkpoint/{out_dir}/{out_name}'
        config['hub_model_id'] = f'{b}-{d}-{s}'

        with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
            json.dump(config, f, indent=4)
