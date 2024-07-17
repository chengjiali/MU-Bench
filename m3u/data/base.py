import os
import copy
import json
import numpy as np
import datasets
from datasets import load_dataset, concatenate_datasets, interleave_datasets, DatasetDict
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset


def find_data_source(dataset_name):
    pass


def load_image_text_dataset(name):
    data = load_dataset(
        f'/data/datasets/image_text_datasets/lavis_cache/{name}/annotations', 
        data_files={
            'train': f'/data/datasets/image_text_datasets/lavis_cache/{name}/annotations/train.json',
            'validation': f'/data/datasets/image_text_datasets/lavis_cache/{name}/annotations/dev.json',
            'test': f'/data/datasets/image_text_datasets/lavis_cache/{name}/annotations/test.json',
        })

    if name == 'snli_ve':
        col_to_remove = [
            'annotator_labels', 'gold_label', 'captionID', 'pairID', 'sentence1', 'sentence2', 'Flickr30K_ID',
            'sentence1_binary_parse', 'sentence2_binary_parse', 'sentence1_parse', 'sentence2_parse']

    elif name == 'nlvr2':
        col_to_remove = []

    new_data = {}
    for k, v in data.items():
        v = v.remove_columns(col_to_remove)
            
        new_data[k] = v
    data = DatasetDict(new_data)

    return data

# def _add_instance_ids(self, key="instance_id"):
#     for idx, ann in enumerate(self.annotation):
#         ann[key] = str(idx)

def prepare_df_dr(ori_train_data, df_mask, dr_mask):
    all_idx = np.arange(ori_train_data.shape[0])
    df_data = datasets.Dataset.from_dict(ori_train_data[all_idx[df_mask]])
    dr_data = datasets.Dataset.from_dict(ori_train_data[all_idx[dr_mask]])

    assert dr_data.shape[0] < ori_train_data.shape[0]
    assert dr_data.shape[0] + df_data.shape[0] == ori_train_data.shape[0]

    return df_data, dr_data

def prepare_deletion_data(unlearn_config, ori_train_data, label_col='label', is_gen=False):

    del_idx = np.loadtxt(f'm3u/df/{unlearn_config.data_name}.txt', dtype=int)
    del_idx = del_idx[:int(unlearn_config.del_ratio / 10 * len(del_idx))]
    train_size = len(ori_train_data)
    df_mask = np.zeros(train_size, dtype=bool)
    df_mask[np.array(del_idx)] = True
    dr_mask = ~df_mask

    df_data, dr_data = prepare_df_dr(ori_train_data, df_mask, dr_mask)
    
    # if unlearn_config.unlearn_method in ['random_label', 'salul']:
    #     num_labels = len(set(ori_train_data[label_col]))#.names)
    #     correct_label = df_data[label_col]

    #     if is_gen:
    #         dr_label = dr_data[label_col]
    #         corrupted_label = [dr_label[i] for i in range(len(correct_label))]
        
    #     else:
    #         num_labels = len(set(ori_train_data[label_col]))#.names)
    #         corrupted_label = [(i-1) % num_labels for i in correct_label]

    if unlearn_config.unlearn_method in ['random_label', 'salul']:
        correct_label = df_data[label_col]

        if is_gen:
            np.random.seed(unlearn_config.random_seed)
            dr_label = dr_data[label_col]
            random_label_idx = np.random.choice(np.arange(len(dr_label)), size=len(correct_label), replace=False)
            corrupted_label = [dr_label[i] for i in random_label_idx]
        
        else:
            num_labels = len(set(ori_train_data[label_col]))#.names)
            corrupted_label = [num_labels - i - 1 for i in correct_label]

        corrupted_df_data = copy.deepcopy(df_data)
        corrupted_df_data = corrupted_df_data.rename_column(label_col, 'ori_label')
        corrupted_df_data = corrupted_df_data.add_column(label_col, corrupted_label)

        assert all([i != j for i, j in zip(corrupted_df_data[label_col], df_data[label_col])])
        concat_data = concatenate_datasets([corrupted_df_data, dr_data])

        data = concat_data

    elif unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'scrub']:
        data = dr_data

    elif unlearn_config.unlearn_method in ['neggrad']:
        data = copy.deepcopy(df_data)

    elif unlearn_config.unlearn_method in ['bad_teaching']:
        # times = dr_data.shape[0] // df_data.shape[0]
        # remainder = dr_data.shape[0] % df_data.shape[0]

        # # Repeat Df to be of same size as Dr
        # repeated_df = [df_data,] * (times+1)
        # repeated_df = concatenate_datasets(repeated_df)
        # repeated_df = HFDataset.from_dict(repeated_df[:dr_data.shape[0]])

        # # Add prefix
        # col = dr_data.column_names
        # dr_data = dr_data.rename_columns({i: f'dr_{i}' for i in col})
        # repeated_df = repeated_df.rename_columns({i: f'df_{i}' for i in col})

        # interleave_data = concatenate_datasets([dr_data, repeated_df], axis=1)
        all_idx = np.arange(dr_data.shape[0])
        sel_idx = np.random.choice(all_idx, size=df_data.shape[0], replace=False)
        dr_subset = datasets.Dataset.from_dict(ori_train_data[sel_idx])
        df_data = df_data.add_column('is_df', [1,] * df_data.shape[0])
        dr_subset = dr_subset.add_column('is_df', [0,] * dr_subset.shape[0])
        data = interleave_datasets([dr_subset, df_data], stopping_strategy='all_exhausted')

    # elif unlearn_config.unlearn_method in ['random_label']

    if unlearn_config.unlearn_method == 'salul':
        df_for_train = corrupted_df_data
    else:
        df_for_train = copy.deepcopy(df_data)

    # return HFDataset(data)
    return data, dr_data, df_data, df_for_train

def prepare_deletion_data_video(unlearn_config, ori_train_data, label_col='label', is_gen=False):
    import pandas as pd
    df_mask = np.loadtxt(f'm3u/df/{unlearn_config.data_name}/{unlearn_config.del_ratio}.txt', dtype=bool)
    dr_mask = ~df_mask

    data = copy.deepcopy(ori_train_data)
    df_data = copy.deepcopy(ori_train_data)
    dr_data = copy.deepcopy(ori_train_data)
    df_for_train = copy.deepcopy(ori_train_data)
    dr_for_eval = copy.deepcopy(ori_train_data)
    
    ori_train_data_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])
    df_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[df_mask]
    dr_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[dr_mask]
    df_for_train_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[df_mask]
    dr_for_eval_path = pd.DataFrame(ori_train_data._paths_and_labels, columns=['path', 'label'])[dr_mask][:df_mask.sum()]

    df_data._paths_and_labels = df_path.values.tolist()
    dr_data._paths_and_labels = dr_path.values.tolist()
    df_for_train._paths_and_labels = df_for_train_path.values.tolist()
    dr_for_eval._paths_and_labels = dr_for_eval_path.values.tolist()


    if unlearn_config.unlearn_method in ['random_label', 'salul']:
        num_labels = len(set(ori_train_data_path[label_col]))#.names)
        correct_label = df_path[label_col]
        corrupted_label = [(i-1) % num_labels for i in correct_label]

        corrupted_df_data = copy.deepcopy(df_data)
        corrupted_df_data_path = pd.DataFrame(corrupted_df_data._paths_and_labels, columns=['path', 'label'])
        corrupted_df_data_path['label'] = corrupted_label

        assert all([i != j for i, j in zip(corrupted_df_data_path['label'], df_path['label'])])
        concat_path = pd.concat([corrupted_df_data_path, dr_path], ignore_index=True).values.tolist()

        data._paths_and_labels = concat_path

    elif unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'scrub']:
        data = dr_data

    elif unlearn_config.unlearn_method in ['neggrad']:
        data = copy.deepcopy(df_data)

    elif unlearn_config.unlearn_method in ['bad_teaching']:
        # times = dr_data.shape[0] // df_data.shape[0]
        # remainder = dr_data.shape[0] % df_data.shape[0]

        # # Repeat Df to be of same size as Dr
        # repeated_df = [df_data,] * (times+1)
        # repeated_df = concatenate_datasets(repeated_df)
        # repeated_df = HFDataset.from_dict(repeated_df[:dr_data.shape[0]])

        # # Add prefix
        # col = dr_data.column_names
        # dr_data = dr_data.rename_columns({i: f'dr_{i}' for i in col})
        # repeated_df = repeated_df.rename_columns({i: f'df_{i}' for i in col})

        # interleave_data = concatenate_datasets([dr_data, repeated_df], axis=1)
        all_idx = np.arange(len(dr_data))
        sel_idx = np.random.choice(all_idx, size=len(df_data), replace=False)
        dr_subset = copy.deepcopy(dr_data)
        dr_subset._paths_and_labels = dr_subset._paths_and_labels[:len(df_path)]
        data = dr_subset
        # df_data = df_data.add_column('is_df', [1,] * df_data.shape[0])
        # dr_subset = dr_subset.add_column('is_df', [0,] * dr_subset.shape[0])
        # data = interleave_datasets([dr_subset, df_data], stopping_strategy='all_exhausted')

    # elif unlearn_config.unlearn_method in ['random_label']

    if unlearn_config.unlearn_method == 'salul':
        df_for_train = corrupted_df_data
    else:
        df_for_train = copy.deepcopy(df_data)

    # return HFDataset(data)
    return data, dr_data, df_data, df_for_train, dr_for_eval


class DeletionData(Dataset):
    def __init__(self, unlearn_config, ori_train_data, transform=None):
        self.unlearn_config = unlearn_config
        self.ori_train_data = ori_train_data
        self.df_mask = np.loadtxt(f'm3u/df/text/{unlearn_config.data_name}/{unlearn_config.del_ratio}.txt', dtype=bool)
        self.dr_mask = ~self.df_mask

        self.df_data, self.dr_data = self.prepare_df_dr()
        
        if self.unlearn_config.unlearn_method in ['random_label']:
            num_labels = len(set(self.ori_train_data['label']))#.names)
            correct_label = self.df_data['label']
            corrupted_label = [num_labels - i - 1 for i in correct_label]
            corrupted_df_data = copy.deepcopy(self.df_data)
            corrupted_df_data = corrupted_df_data.rename_column('label', 'ori_label')
            corrupted_df_data = corrupted_df_data.add_column('label', corrupted_label)

            assert all([i != j for i, j in zip(corrupted_df_data['label'], self.df_data['label'])])
            self.concat_data = concatenate_datasets([corrupted_df_data, self.dr_data])
        # self.subset_dr_data = self.dr_data
        # self.resample_dr()

        # # These unlearning methods use the whole Dr as training set
        # if self.unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec']:
        #     assert self.subset_dr_data.shape[0] == self.dr_data.shape[0]

        self.transform = transform

    def resample_dr(self):
        '''Sample a subset from Dr to have same size of Df, for some unlearning methods. 
            This is called every epoch to cover a wider range of Dr, without relying on the same sample
        '''

        # These methods iterate through Df, Dr simuteneously during training.
        if self.unlearn_config.unlearn_method in ['bad_teaching']:
            dr_size = self.dr_data.shape[0]
            df_size = self.df_data.shape[0]
            sel_idx = np.random.choice(np.arange(dr_size), size=df_size, replace=False)
            
            self.subset_dr_data = self.dr_data.select(sel_idx)#.reset_index(drop=True)

    def __len__(self):
        if self.unlearn_config.unlearn_method in ['retrain', 'fisher', 'l-codec', 'bad_teaching']:
            return self.dr_data.shape[0]
        elif self.unlearn_config.unlearn_method in ['random_label']:
            return self.concat_data.shape[0]
        else:
            return self.df_data.shape[0]

    def prepare_df_dr(self):
        all_idx = np.arange(self.ori_train_data.shape[0])
        df_data = datasets.Dataset.from_dict(self.ori_train_data[all_idx[self.df_mask]])
        dr_data = datasets.Dataset.from_dict(self.ori_train_data[all_idx[self.dr_mask]])

        assert dr_data.shape[0] < self.ori_train_data.shape[0]
        assert dr_data.shape[0] + df_data.shape[0] == self.ori_train_data.shape[0]

        return df_data, dr_data

    def __getitem__(self, idx):
        # dr = self.subset_dr_data[idx]
        # df = self.df_data[idx]# % self.df_data.shape[0]]  # Df is smaller than Dr. idx may be out of bound

        if self.unlearn_config.unlearn_method in ['random_label']:
            return self.concat_data[idx]

        dr = self.dr_data[idx]
        df = self.df_data[idx % self.df_data.shape[0]]  # Df is smaller than Dr. idx may be out of bound

        # data = {'dr_input': dr, 'df_input': df}
        dfdr = {}
        for k, v in df.items():
            if k == 'label':
                k = 'labels'
            dfdr['df_'+k] = v
        for k, v in dr.items():
            if k == 'label':
                k = 'labels'
            dfdr['dr_'+k] = v

        if self.unlearn_config.unlearn_method in ['neggrad']:
            return df

        elif self.unlearn_config.unlearn_method in ['bad_teaching']:
            if self.transform is not None:
                return self.transform(dfdr)
            else:
                return dfdr

        return dr


def prepare_dr_data(dataset_train_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)

    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dataset

def prepare_df_data(dataset_train_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)
    
    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return dataset


def prepare_df_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        df_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        df_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = df_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        df_for_test.text = text
        df_for_test.image = image
        df_for_test.txt2img = txt2img
        df_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        df_for_test = copy.deepcopy(dataset_train_ori)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     df_for_test = copy.deepcopy(dataset_test_ori)

    #     df_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     df_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if str(tuple(i['images'])) in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in df_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return df_for_test

def prepare_dr_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type, sample_size=None):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset_train_ori.annotation]))

        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        dr_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_train_ori.annotation if i['image'] not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        dr_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = dr_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        dr_for_test.text = text
        dr_for_test.image = image
        dr_for_test.txt2img = txt2img
        dr_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        dr_for_test = copy.deepcopy(dataset_train_ori)

        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     dr_for_test = copy.deepcopy(dataset_test_ori)

    #     dr_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     dr_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if str(tuple(i['images'])) not in df_ids_set]

        if sample_size is not None:
            anno_id = np.arange(len(dr_for_test.annotation))
            indices = np.random.choice(anno_id, sample_size, replace=False)
            dr_for_test.annotation = [dr_for_test.annotation[i] for i in indices]

        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dr_for_test