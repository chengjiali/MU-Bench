import os
import copy
import json
import argparse


seeds = [42, 87, 21, 100, 13]
glue_datasets = ['sst2', 'cola', 'mnli', 'qnli', 'rte', 'qqp', 'mrpc', 'stsb']
datasets = ['ag_news', 'imdb', 'rotten_tomatoes', 'dbpedia_14']
backbones = ['bert-base', 'roberta-base', 'distilbert-base', 'electra-base', 'deberta-base', 'albert-base-v2']

def get_full_model_name(m):
    if m.startswith('bert-'):
        m = m + '-uncased'

    elif 'roberta' in m:
        m = 'FacebookAI/' + m

    elif 'distilbert-' in m:
        m = 'distilbert/' + m + '-uncased'

    elif 'electra-' in m:
        m = 'google/' + m + '-discriminator'

    elif 'deberta-' in m:
        if 'base' in m:
            m = 'microsoft/deberta-v3-base'
    
    elif 'albert-' in m:
        m = 'albert/' + m

    elif 'biobert' in m:
        m = 'dmis-lab/' + m + '-v1.1'

    elif 'pubmedbert-abstract' in m:
        m = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract'

    elif 'pubmedbert-fulltext' in m:
        m = 'microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext'

    return m

template = {
    "do_train": True,
    "do_eval": True,
    "task_name": "sst2",
    "max_seq_length": 256,
    "num_train_epochs": 5,
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
}


# glue_datasets = []
# for b in backbones:
#     for d in glue_datasets:
#         for s in seeds:
#             config = copy.deepcopy(template)
#             out_dir = f'{b}'
#             out_name = f'{d}_{s}'
#             os.makedirs(f'configs/{out_dir}', exist_ok=True)
            
#             config['model_name_or_path'] = get_full_model_name(b)
#             config['task_name'] = d
#             config['seed'] = s
#             config['output_dir'] = f'checkpoint/{out_dir}/{out_name}'
#             config['hub_model_id'] = f'{b}-{d}-{s}'

#             if d in ['sst2', 'cola', 'ag_news', 'imdb', 'rotten_tomatoes']:
#                 config['max_seq_length'] = 128

#             if d in ['qqp', 'mrpc']:
#                 config['metric_for_best_model'] = 'f1'
#             elif d in ['stsb']:
#                 config['metric_for_best_model'] = 'spearmanr'

#             with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
#                 json.dump(config, f, indent=4)


template = {
    "do_train": True,
    "do_eval": True,
    "dataset_name": "ag_news",
    "max_seq_length": 128,
    "num_train_epochs": 10,
    "logging_steps": 1000,
    "evaluation_strategy": "epoch",
    "save_strategy": "epoch",
    "per_device_train_batch_size": 32,
    "per_device_eval_batch_size": 256,
    "learning_rate": 5e-5,
    "warmup_steps": 0,
    "save_total_limit": 1,
    "load_best_model_at_end": True,
    "metric_name": "accuracy",
    "metric_for_best_model": "accuracy",
    "overwrite_output_dir": True,
    "dataloader_num_workers": 16,
    # "push_to_hub": True,
}

# for b in backbones:
#     for d in datasets:
#         for s in seeds:
#             config = copy.deepcopy(template)
#             out_dir = f'{b}'
#             out_name = f'{d}_{s}'
#             os.makedirs(f'configs/{out_dir}', exist_ok=True)
            
#             config['model_name_or_path'] = get_full_model_name(b)
#             config['dataset_name'] = d

#             if d == 'dbpedia_14':
#                 config['dataset_name'] = 'fancyzhx/dbpedia_14'
#                 config['text_column_names'] = 'title,content'

#             config['seed'] = s
#             config['output_dir'] = f'checkpoint/{out_dir}/{out_name}'
#             config['hub_model_id'] = f'{b}-{d}-{s}'

#             # if d in ['sst2', 'cola', 'ag_news', 'imdb', 'rotten_tomatoes']:
#             #     config['max_seq_length'] = 128

#             # if d in ['qqp', 'mrpc']:
#             #     config['metric_for_best_model'] = 'f1'
#             # elif d in ['stsb']:
#             #     config['metric_for_best_model'] = 'spearmanr'

#             with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
#                 json.dump(config, f, indent=4)

backbones = ['biobert', 'pubmedbert-abstract', 'pubmedbert-fulltext']
datasets = ['ddi', 'chem_prot']
for b in backbones:
    for d in datasets:
        for s in seeds:
            config = copy.deepcopy(template)
            out_dir = f'{b}'
            out_name = f'{d}_{s}'
            os.makedirs(f'configs/{out_dir}', exist_ok=True)
            
            config['model_name_or_path'] = get_full_model_name(b)
            config['dataset_name'] = d

            config['seed'] = s
            config['output_dir'] = f'text/train/checkpoint/{out_dir}/{out_name}'
            config['hub_model_id'] = f'{b}-{d}-{s}'

            # if d in ['sst2', 'cola', 'ag_news', 'imdb', 'rotten_tomatoes']:
            #     config['max_seq_length'] = 128

            # if d in ['qqp', 'mrpc']:
            #     config['metric_for_best_model'] = 'f1'
            # elif d in ['stsb']:
            #     config['metric_for_best_model'] = 'spearmanr'

            with open(f'configs/{out_dir}/{out_name}.json', 'w') as f:
                json.dump(config, f, indent=4)