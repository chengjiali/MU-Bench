import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
datasets = ['samsum']
backbones = ['t5-small', 't5-base', 't5-large', 't5-xl', 't5-xxl']

def get_full_model_name(m):
    if 't5-' in m:
        scale = m.split('-')[1]
        m = 'google/t5-v1_1-' + scale

    # elif 'roberta' in m:
    #     m = 'FacebookAI/' + m

    # elif 'distilbert-' in m:
    #     m = 'distilbert/' + m + '-uncased'

    # elif 'electra-' in m:
    #     m = 'google/' + m + '-discriminator'

    # elif 'deberta-' in m:
    #     if 'base' in m:
    #         m = 'microsoft/deberta-v3-base'
    
    # elif 'albert-' in m:
    #     m = 'albert/' + m

    # elif 'biobert' in m:
    #     m = 'dmis-lab/' + m + '-v1.1'

    # elif 'pubmedbert-abstract' in m:
    #     m = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

    # elif 'pubmedbert-fulltext' in m:
    #     m = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

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
    "per_device_eval_batch_size": 64,
    "learning_rate": 5e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    "predict_with_generate": True,
    "source_prefix": "summarize: ",
    "max_source_length": 1024,
    "max_target_length": 128,
    "metric_for_best_model": 'rougeL',
    # "push_to_hub": True,
}


s = 42
for b in backbones:
    for d in datasets:
        config = copy.deepcopy(template)
        out_dir = f'{b}'
        out_name = f'{d}_{s}'
        os.makedirs(f'configs/{out_dir}', exist_ok=True)
        
        config['model_name_or_path'] = get_full_model_name(b)
        config['dataset_name'] = d
        config['seed'] = s
        config['output_dir'] = f'text_gen/train/checkpoint/{out_dir}/{out_name}'
        config['hub_model_id'] = f'{b}-{d}-{s}'

        if 'xxl' in b:
            config['per_device_train_batch_size'] = 1
            config['gradient_accumulation_steps'] = 2

        elif 'xl' in b:
            config['per_device_train_batch_size'] = 1
            config['gradient_accumulation_steps'] = 2

        with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
            json.dump(config, f, indent=4)