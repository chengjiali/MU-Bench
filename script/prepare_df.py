import os
import sys
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict
sys.path.append(os.getcwd())
from m3u.data.base import load_image_text_dataset
from collections import Counter


def get_biore_data(name):
    train = pd.read_csv(f'./data/{name}/train.tsv', sep='\t', names=['none', 'text', 'label']).drop('none', axis=1)
    dev = pd.read_csv(f'./data/{name}/dev.tsv', sep='\t', names=['none', 'text', 'label']).drop('none', axis=1)
    test = pd.read_csv(f'./data/{name}/test.tsv', sep='\t', names=['none', 'text', 'label']).drop('none', axis=1)

    if name == 'chem_prot':
        mapping = {'false': 0, 'CPR:3': 1, 'CPR:4': 2, 'CPR:5': 3, 'CPR:6': 4, 'CPR:9': 5}
    elif name == 'ddi':
        mapping = {'DDI-false': 0, 'DDI-mechanism': 1, 'DDI-advise': 2, 'DDI-effect': 3, 'DDI-int': 4}

    train.label = train.label.apply(mapping.get)
    dev.label = dev.label.apply(mapping.get)
    test.label = test.label.apply(mapping.get)

    raw_datasets = DatasetDict({
        'train': Dataset.from_pandas(train),
        'validation': Dataset.from_pandas(dev),
        'test': Dataset.from_pandas(test),
    })
    
    return raw_datasets


def classification(data_name, label_col='label'):
    print(data_name)
    if data_name in ['ddi', 'chem_prot']:
        data = get_biore_data(data_name)
    elif data_name in ['superb_ks']:
        name, config = data_name.split('_')
        if config == 'si':
            data_dir = 'data/superb_si'
        else:
            data_dir = None
        data = load_dataset(name, config, data_dir=data_dir)

    elif data_name in ['nlvr2']:
        data = load_image_text_dataset(data_name)

    else:
        data = load_dataset(data_name)

    data = data['train'].to_pandas()
    if data_name == 'cifar100':
        data['label'] = data['fine_label']

    if data_name == 'ddi':
        num_labels = 5
    elif data_name == 'chem_prot':
        num_labels = 6
    else:
        num_labels = len(set(data[label_col]))

    if data_name in ['nlvr2']:
        mapping = {'False': 0, 'True': 1}
        data[label_col] = data[label_col].apply(mapping.get)

    train_size = len(data)

    out_file = f'm3u/df/{data_name}.txt'
    del_size = int(train_size * 1 / 100)
    del_size_per_class = max(1, int(del_size / num_labels))
    del_size = del_size_per_class * num_labels
    df_mask = np.zeros(train_size).astype(bool)
    sel_idx = []
    sel_idx_label = []

    for del_ratio in range(1, 11):
        for label_id in range(num_labels):
            indices = data[~df_mask][data[~df_mask]['label'] == label_id].index

            # There are cases where del_size_per_class > available examples
            # We defer the remainder to the final section
            idx = np.random.choice(indices, min(del_size_per_class, indices.shape[0]), replace=False)#.tolist()
            df_mask[idx] = True
            sel_idx.extend(idx.tolist())
            sel_idx_label.extend(data[label_col][idx].tolist())

        print(del_ratio, del_size, Counter(sel_idx_label))

        # If there are a few more data needs to be sampled for Df
        if df_mask.sum() < del_ratio * del_size:
            print('missing')
            remainder_size = del_ratio * del_size - df_mask.sum()
            idx = np.random.choice(data[~df_mask].index, remainder_size, replace=False)#.tolist()
            df_mask[idx] = True
            sel_idx.extend(idx.tolist())
            sel_idx_label.extend(data[label_col][idx].tolist())
        
        print(del_ratio, del_size, Counter(sel_idx_label))

        assert df_mask.sum() == del_ratio * del_size, f'Need to sample {del_ratio * del_size}, but only sampled {df_mask.sum()}'

    np.savetxt(out_file, sel_idx, fmt='%d')


def video():
    data = pd.read_csv('/data/datasets/video_datasets/ucfTrainTestlist/train.txt', sep=' ', names=['path', 'label'])
    num_labels = 101
    train_size = data.shape[0]
    # train_idx = np.arange(train_size)
    train_labels = data['label'].values

    out_dir = f'm3u/df/ucf-101'
    os.makedirs(out_dir, exist_ok=True)
    df_mask = np.zeros_like(train_labels).astype(bool)

    total_del_size = int(train_size * 1 / 100)
    del_size_per_class = max(1, int(total_del_size / num_labels))
    total_del_size = del_size_per_class * num_labels
    # df_mask = np.zeros_like(train_labels[~df_mask]).astype(bool)

    for del_ratio in range(1, 11):
        out_file = os.path.join(out_dir, f'{del_ratio}.txt')
        for label_id in range(num_labels):
            indices = data[~df_mask][data[~df_mask]['label'] == label_id].index

            # There are cases where del_size_per_class > available examples
            # We defer the remainder to the final section
            idx = np.random.choice(indices, min(del_size_per_class, indices.shape[0]), replace=False)#.tolist()
            print(del_ratio, label_id, idx)
            df_mask[idx] = True

        print(del_ratio, df_mask.sum(), del_ratio * total_del_size)
        # If there are a few more data needs to be sampled for Df
        if df_mask.sum() < del_ratio * total_del_size:
            print('missing')
            remainder_size = total_del_size - df_mask.sum()
            idx = np.random.choice(train_idx[~df_mask], remainder_size, replace=False)#.tolist()
            df_mask[idx] = True

        assert df_mask.sum() == del_ratio * total_del_size, f'Need to sample {total_del_size}, but only sampled {df_mask.sum()}'

        print(del_ratio, '%')
        np.savetxt(out_file, df_mask, fmt='%d')


def text_gen(data_name, label_col='label'):
    data = load_dataset(data_name)
    mask = np.ones(data['train'].shape[0]).astype(bool)
    mask[6054] = False
    data['train'] = Dataset.from_dict(data['train'][mask])

    train_size = data['train'].shape[0]
    train_labels = np.array(data['train'][label_col])

    out_dir = f'm3u/df/{data_name}'
    os.makedirs(out_dir, exist_ok=True)
    for del_ratio in range(1, 21):
        df_mask = np.zeros(train_size).astype(bool)

        del_ratio /= 2
        total_del_size = int(train_size * del_ratio / 100)

        out_file = os.path.join(out_dir, f'{del_ratio}.txt')
        indices = np.arange(train_size)
        idx = np.random.choice(indices, total_del_size, replace=False)#.tolist()
        df_mask[idx] = True

        assert df_mask.sum() == total_del_size, f'Need to sample {total_del_size}, but only sampled {df_mask.sum()}'

        print(data_name, del_ratio, '%')
        np.savetxt(out_file, df_mask, fmt='%d')

def main():
    np.random.seed(42)
    for d in ['imdb', 'ddi', 'cifar100', 'nlvr2', 'superb_ks']:
        classification(d)

    # for d in ["samsum"]:
    #     text_gen(d, 'summary')

    # video()


if __name__ == "__main__":
    main()