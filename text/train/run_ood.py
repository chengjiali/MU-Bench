#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for text classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
sys.path.append(os.getcwd())
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import datasets
import evaluate
import numpy as np
import pandas as pd
from datasets import Value, load_dataset, Dataset, DatasetDict

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.32.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")


logger = logging.getLogger(__name__)


model_map_rev = {
    'bert-base-uncased': 'bert-base',
    'FacebookAI/roberta-base': 'roberta-base',
    'distilbert/distilbert-base-uncased': 'distilbert-base',
    'google/electra-base-discriminator': 'electra-base',
    'microsoft/deberta-v3-base': 'deberta-base',
    'albert/albert-base-v2': 'albert-base-v2'
}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    do_regression: bool = field(
        default=None,
        metadata={
            "help": "Whether to do regression instead of classification. If None, will be inferred from the dataset."
        },
    )
    text_column_names: Optional[str] = field(
        default='text',
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file."
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    text_column_delimiter: Optional[str] = field(
        default=" ", metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    validation_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    test_split_name: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    remove_splits: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    remove_columns: Optional[str] = field(
        default=None,
        metadata={"help": "The columns to remove from the dataset. Multiple columns should be separated by commas."},
    )
    label_column_name: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The name of the label column in the input dataset or a CSV/JSON file."
                'If not specified, will use the "label" column for single/multi-label classifcation task'
            )
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    shuffle_train_dataset: bool = field(
        default=False, metadata={"help": "Whether to shuffle the train dataset or not."}
    )
    shuffle_seed: int = field(
        default=42, metadata={"help": "Random seed that will be used to shuffle the train dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    metric_name: Optional[str] = field(default=None, metadata={"help": "The metric to use for evaluation."})
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        if self.dataset_name is None:
            if self.train_file is None or self.validation_file is None:
                raise ValueError(" training/validation file or a dataset name.")

            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: str = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )


def get_label_list(raw_dataset, split="train") -> List[str]:
    """Get the list of labels from a mutli-label dataset"""

    if isinstance(raw_dataset[split]["label"][0], list):
        label_list = [label for sample in raw_dataset[split]["label"] for label in sample]
        label_list = list(set(label_list))
    else:
        label_list = raw_dataset[split].unique("label")
    # we will treat the label list as a list of string instead of int, consistent with model.config.label2id
    label_list = [str(label) for label in label_list]
    return label_list


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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    
    backbones = ['bert-base', 'bert-large', 'distilbert-base', 'electra-base', 'deberta-base', 'albert-base-v2']
    methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salul']
    del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    metric = evaluate.load('accuracy')
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=p.label_ids)
        if len(result) > 1:
            result["combined_score"] = np.mean(list(result.values())).item()
        return result

    data_collator = default_data_collator

    from m3u.trainer import UnlearningTrainer as Trainer


    todo = []
    for b in backbones:
        ckpt_dir = f'./text/train/checkpoint/{b}/imdb_42'
        if os.path.exists(f'{ckpt_dir}/ood_results.json'):
            continue

        todo.append(ckpt_dir)

    for ck in ['checkpoint', 'cl_checkpoint', 'lora_50_checkpoint']:
        for b in backbones:
            for m in methods:
                for dr in del_ratio:
                    ckpt_dir = f'./text/unlearn/{ck}/{b}/{m}/{dr}/imdb_42'
                    if not os.path.exists(ckpt_dir):
                        continue

                    if os.path.exists(f'{ckpt_dir}/ood_results.json'):
                        continue

                    todo.append(ckpt_dir)

    for ckpt_dir in todo:
        training_args = TrainingArguments(output_dir=ckpt_dir, per_device_eval_batch_size=128, dataloader_num_workers=32, report_to=[])

        training_args.remove_unused_columns = False
        training_args.overwrite_output_dir = False


        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
        model = AutoModelForSequenceClassification.from_pretrained(ckpt_dir)

        padding = "max_length"
        max_seq_length = 256

        ood_data = load_dataset('rotten_tomatoes')['test']
        
        def preprocess_function(examples):
            # Tokenize the texts
            result = tokenizer(examples["text"], padding=padding, max_length=max_seq_length, truncation=True)
            # if label_to_id is not None and "label" in examples:
            #     if is_multi_label:
            #         result["label"] = [multi_labels_to_ids(l) for l in examples["label"]]
            #     else:
            #         result["label"] = [(label_to_id[str(l)] if l != -1 else -1) for l in examples["label"]]
            return result

        ood_data = ood_data.map(
            preprocess_function,
            batched=True,
        )


        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=data_collator,
            # raw_datasets=raw_datasets
        )
        trainer.unlearning = False

        logger.info("*** OOD ***")
        metrics = trainer.evaluate(eval_dataset=ood_data, metric_key_prefix='ood')
        print(ckpt_dir)
        trainer.log_metrics("ood", metrics)
        trainer.save_metrics("ood", metrics)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()