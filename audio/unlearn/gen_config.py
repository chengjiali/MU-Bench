import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['superb_ks', 'superb_si']
backbones = ['wav2vec2-base', 'wav2vec2-large', 'whisper-tiny', 'whisper-base', 'whisper-small', 'hubert-base', 'hubert-large', 'hubert-xlarge']
del_ratio = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

general_methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salul']
methods = [] + general_methods


def get_full_model_name(m):
    if 'wav2vec2-' in m:
        m = 'facebook/' + m

    elif 'whisper-' in m:
        m = 'openai/' + m

    elif 'hubert-' in m:
        m = 'facebook/' + m
        if '-base' in m:
            m = m + '-ls960'
        else:
            m = m + '-ll60k'

    return m


template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "superb_si",
    "dataset_config_name": "si",
    "num_train_epochs": 5,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 256,
    "max_length_seconds": 1,
    "learning_rate": 5e-5,
    "warmup_ratio": 0.1,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_for_best_model": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    # "push_to_hub": True,
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

                    config['model_name_or_path'] = get_full_model_name(b)
                    config['dataset_name'] = 'superb'

                    if d == 'superb_si':
                        config['dataset_config_name'] = 'si'
                        config['per_device_eval_batch_size'] = 16
                    if d == 'superb_ks':
                        config['dataset_config_name'] = 'ks'

                    config['seed'] = s
                    config['output_dir'] = f'audio/unlearn/checkpoint/{out_dir}/{out_name}'
                    config['hub_model_id'] = f'{b}-{m}-{dr}-{d}-{s}'

                    # if d in ['sst2', 'cola', 'ag_news', 'imdb', 'rotten_tomatoes']:
                    #     config['max_seq_length'] = 128

                    # if d in ['qqp', 'mrpc']:
                    #     config['metric_for_best_model'] = 'f1'
                    # elif d in ['stsb']:
                    #     config['metric_for_best_model'] = 'spearmanr'

                    with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
                        json.dump(config, f, indent=4)
