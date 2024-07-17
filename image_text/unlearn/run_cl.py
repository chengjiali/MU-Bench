#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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

import logging
import os
import sys
import wandb
import warnings
from dataclasses import dataclass, field
from random import randint
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset

from PIL import Image
import transformers
from transformers import (
    AutoConfig,
    AutoProcessor,
    ViltForImagesAndTextClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

sys.path.append(os.getcwd())
from m3u.args import UnlearningArguments
from m3u.trainer import UnlearningTrainer
from m3u.data.base import load_image_text_dataset, prepare_deletion_data



model_map_rev = {
    'dandelin/vilt-b32-finetuned-nlvr2': 'vilt',
}
logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0")

require_version("datasets>=1.14.0", "To fix: pip install -r examples/pytorch/audio-classification/requirements.txt")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(default=None, metadata={"help": "Name of a dataset from the datasets package"})
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the training audio paths and labels."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "A file containing the validation audio paths and labels."}
    )
    train_split_name: str = field(
        default="train",
        metadata={
            "help": "The name of the training data set split to use (via the datasets library). Defaults to 'train'"
        },
    )
    eval_split_name: str = field(
        default="validation",
        metadata={
            "help": (
                "The name of the training data set split to use (via the datasets library). Defaults to 'validation'"
            )
        },
    )
    audio_column_name: str = field(
        default="audio",
        metadata={"help": "The name of the dataset column containing the audio data. Defaults to 'audio'"},
    )
    label_column_name: str = field(
        default="label", metadata={"help": "The name of the dataset column containing the labels. Defaults to 'label'"}
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
    max_length_seconds: float = field(
        default=20,
        metadata={"help": "Audio clips will be randomly cut to this length during training if the value is set."},
    )


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="facebook/wav2vec2-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from the Hub"}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    feature_extractor_name: Optional[str] = field(
        default=None, metadata={"help": "Name or path of preprocessor config."}
    )
    freeze_feature_encoder: bool = field(
        default=True, metadata={"help": "Whether to freeze the feature encoder layers of the model."}
    )
    attention_mask: bool = field(
        default=True, metadata={"help": "Whether to generate an attention mask in the feature extractor."}
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
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option "
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will "
                "execute code present on the Hub on your local machine."
            )
        },
    )
    freeze_feature_extractor: Optional[bool] = field(
        default=None, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

    def __post_init__(self):
        if not self.freeze_feature_extractor and self.freeze_feature_encoder:
            warnings.warn(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "will be removed in a future version. Use `--freeze_feature_encoder` "
                "instead. Setting `freeze_feature_encoder==True`.",
                FutureWarning,
            )
        if self.freeze_feature_extractor and not self.freeze_feature_encoder:
            raise ValueError(
                "The argument `--freeze_feature_extractor` is deprecated and "
                "should not be used in combination with `--freeze_feature_encoder`. "
                "Only make use of `--freeze_feature_encoder`."
            )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, UnlearningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, unlearn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, unlearn_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()


    if unlearn_args.data_name is None:
        if data_args.dataset_config_name is not None:
            unlearn_args.data_name = data_args.dataset_name + '_' + data_args.dataset_config_name
        else:
            unlearn_args.data_name = data_args.dataset_name
        unlearn_args.random_seed = training_args.seed
        unlearn_args.backbone = model_map_rev[model_args.model_name_or_path]
    
    training_args.metric_for_best_model = 'unlearn_overall_' + training_args.metric_for_best_model

    training_args.output_dir = training_args.output_dir.replace('checkpoint', 'cl_checkpoint')

    # Wandb
    training_args.report_to = ['wandb']
    project = 'Unlearning Benchmark'
    group = unlearn_args.data_name + '-' + unlearn_args.backbone + '-cl'
    name = unlearn_args.unlearn_method + '-' + str(unlearn_args.del_ratio) + '-' + str(unlearn_args.random_seed)
    run_id = '-'.join([unlearn_args.data_name, unlearn_args.backbone, 'cl', unlearn_args.unlearn_method, str(unlearn_args.del_ratio), str(unlearn_args.random_seed)])
    wandb.init(project=project, group=group, name=name, config=unlearn_args, id=run_id, resume='allow')
    # Log on each process the small summary:
    # logger.warning(
    #     f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
    #     + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    # )
    # logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to train from scratch."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Initialize our dataset and prepare it for the audio classification task.
    raw_datasets = load_image_text_dataset(data_args.dataset_name)

    
    # Prepare label mappings.
    # We'll include these in the model's config to get human readable labels in the Inference API.
    labels = sorted(list(set(raw_datasets["train"][data_args.label_column_name])))
    label2id, id2label = {}, {}
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label

    if data_args.dataset_name == 'nlvr2':
        label2id = {'False': 0, 'True': 1}
        id2label = {0: 'False', 1: 'True'}
    
    
    # Setting `return_attention_mask=True` is the way to get a correctly masked mean-pooling over
    # transformer outputs in the classifier, but it doesn't always lead to better accuracy
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path)

    # model_input_name = feature_extractor.model_input_names[0]

    def preprocess_one_image(batch):
        root = f'/home/public/jcheng2/lavis_cache/flickr30k/images/flickr30k-images'

        images = [Image.open(os.path.join(root, f'{i}.jpg')).convert("RGB") for i in batch['image']]
        texts = batch['sentence']
        outputs = processor(images, texts, padding=True, return_tensors='pt')

        return outputs

    def preprocess_two_image(batch):
        root = f'/home/public/jcheng2/lavis_cache/nlvr2'

        paths = [i for pair in batch['images'] for i in pair]
        images = [Image.open(os.path.join(root, i)).convert("RGB") for i in paths]

        # images1 = [Image.open(os.path.join(root, i[0])).convert("RGB") for i in batch['images']]
        # images2 = [Image.open(os.path.join(root, i[1])).convert("RGB") for i in batch['images']]
        # images = [i for pair in zip(images1, images2) for i in pair]

        texts = batch['sentence']
        bs = len(texts)
        outputs = processor(images, texts, padding=True, truncation=True, return_tensors='pt')

        # Reshape to [batch_size, num_images, C, H, W]        
        shape = outputs.pixel_values.shape
        outputs['pixel_values'] = outputs['pixel_values'].view(bs, 2, *shape[1:])
        shape = outputs.pixel_mask.shape
        outputs['pixel_mask'] = outputs['pixel_mask'].view(bs, 2, *shape[1:])
        # outputs['labels'] = [label2id[i] for i in batch['label']]
        outputs['labels'] = batch['label']

        if unlearn_args.unlearn_method in ['bad_teaching'] and 'is_df' in batch:
            outputs['is_df'] = batch['is_df']

        return outputs

    
    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)


    config = AutoConfig.from_pretrained(
        model_args.config_name or model_args.model_name_or_path,
        num_labels=len(labels),
        label2id=label2id,
        id2label=id2label,
        # finetuning_task="image-text-classification",
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # token=model_args.token,
        # trust_remote_code=model_args.trust_remote_code,
    )
    if data_args.dataset_name == 'nlvr2':
        config.num_images = 2
        preprocess = preprocess_two_image
    else:
        config.num_images = 1
        preprocess = preprocess_one_image

    def map_label(example):
        example['label'] = {'False': 0, 'True': 1}[example['label']]
        return example

    raw_datasets = raw_datasets.map(map_label)

    # Unlearning data
    raw_datasets['ori_train'] = raw_datasets['train']
    raw_datasets['train'], raw_datasets['dr'], raw_datasets['df'], raw_datasets['df_for_train'] = prepare_deletion_data(unlearn_args, raw_datasets['ori_train'])

    raw_datasets['train'].set_transform(preprocess)
    raw_datasets['validation'].set_transform(preprocess)
    raw_datasets['test'].set_transform(preprocess)
    raw_datasets['dr'].set_transform(preprocess)
    raw_datasets['df'].set_transform(preprocess)
    raw_datasets['df_for_train'].set_transform(preprocess)


    if 'vilt' in model_args.model_name_or_path:
        ori_model = ViltForImagesAndTextClassification.from_pretrained(model_args.model_name_or_path)


    # ckpt = None
    # ckpt_dir = f'./image_text/train/checkpoint/{model_map_rev[model_args.model_name_or_path]}/{data_args.dataset_name.split("/")[-1]}_42'
    # if os.path.exists(os.path.join(ckpt_dir, 'pytorch_model.bin')):
    #     ckpt = torch.load(os.path.join(ckpt_dir, 'pytorch_model.bin'), map_location='cpu')
    # elif os.path.exists(os.path.join(ckpt_dir, 'model.safetensors')):
    #     from safetensors.torch import load_file
    #     ckpt = load_file(os.path.join(ckpt_dir, 'model.safetensors'), device='cpu')
    # if ckpt is not None:
    #     ori_model.load_state_dict(ckpt, strict=False)
    # else:
    #     print('*************************************************')
    #     print('**************   CKPT empty *********************')
    #     print('*************************************************')

    # Unlearning method
    if unlearn_args.unlearn_method == 'retrain':
        model = ViltForImagesAndTextClassification.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            config=config,
            cache_dir=model_args.cache_dir,
        )
    elif unlearn_args.unlearn_method in ['neggrad', 'fisher', 'random_label', 'bad_teaching', 'scrub', 'salul']:
        model = ori_model
        if unlearn_args.unlearn_method in ['bad_teaching']:
            training_args.remove_unused_columns = False


    training_args.per_device_eval_batch_size = 64
    training_args.dataloader_num_workers = 4
    # training_args.max_steps = 1

    # Initialize our trainer
    trainer = UnlearningTrainer(
        raw_datasets=raw_datasets,
        unlearn_config=unlearn_args,
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["validation"] if training_args.do_eval else None,
        compute_metrics=compute_metrics,
    )
    trainer.use_cl = True

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.unlearn(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        if train_result is not None:
            trainer.log_metrics("train", train_result.metrics)
            trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=raw_datasets["validation"], metric_key_prefix='eval')
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

        logger.info("*** Test ***")
        metrics = trainer.evaluate(eval_dataset=raw_datasets["test"], metric_key_prefix='test')
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        logger.info("*** Dr ***")
        metrics = trainer.evaluate(eval_dataset=raw_datasets["dr"], metric_key_prefix='dr')
        trainer.log_metrics("dr", metrics)
        trainer.save_metrics("dr", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "image-text-classification",
        "dataset": data_args.dataset_name,
        "tags": ["image-text-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
