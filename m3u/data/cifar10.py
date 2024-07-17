from datasets import load_dataset


class DeletionDataForCifar10:
    def __inint__(self, config):
        self.do = load_dataset('cifar10')
        # load_dataset(
        #     data_args.dataset_name,
        #     data_args.dataset_config_name,
        #     cache_dir=model_args.cache_dir,
        #     task="image-classification",
        #     use_auth_token=True if model_args.use_auth_token else None,
        # )

# def prepare_dr_data(dataset_train_ori, cfg, data_type):
    def prepare_dr(self, ):
        raise NotImplementedError

        with open(f'Df/vl/flickr30k/image-text-pair-42.txt', 'r') as f:
            df_ids = f.readlines()
        df_ids = [i.strip() for i in df_ids]
        df_ids = df_ids[:cfg.run_cfg.df_size]
        df_ids_set = set(df_ids)

        dataset = copy.deepcopy(dataset_train_ori)

        if cfg.run_cfg.task == 'retrieval':
            num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
            dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
            num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))


