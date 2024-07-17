
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
import warnings
from dataclasses import dataclass, field
from random import randint
from typing import Optional

import datasets
import evaluate
import numpy as np
from datasets import DatasetDict, load_dataset

import transformers
from transformers import (
    AutoConfig,
    VideoMAEImageProcessor,
    AutoModelForVideoClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

import pytorchvideo.data
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)

import wandb
sys.path.append(os.getcwd())
from transformers_for_cl import VideoMAEForVideoClassification
from m3u.args import UnlearningArguments
from m3u.trainer import UnlearningTrainer
from m3u.data.base import prepare_deletion_data_video


model_map_rev = {
    'MCG-NJU/videomae-base': 'videomae-base',
    'MCG-NJU/videomae-large': 'videomae-large',
}
logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.39.0")



def random_subsample(wav: np.ndarray, max_length: float, sample_rate: int = 16000):
    """Randomly sample chunks of `max_length` seconds from the input audio"""
    sample_length = int(round(sample_rate * max_length))
    if len(wav) <= sample_length:
        return wav
    random_offset = randint(0, len(wav) - sample_length - 1)
    return wav[random_offset : random_offset + sample_length]


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    dataset_name: Optional[str] = field(default='superb', metadata={"help": "Name of a dataset from the datasets package"})
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

    unlearn_args.data_name = data_args.dataset_name
    unlearn_args.random_seed = training_args.seed
    unlearn_args.backbone = model_map_rev[model_args.model_name_or_path]
    unlearn_args.use_cl = True

    training_args.output_dir = training_args.output_dir.replace('checkpoint', 'cl_checkpoint')
    training_args.dataloader_drop_last = True
    training_args.remove_unused_columns = False

    training_args.metric_for_best_model = 'unlearn_overall_' + training_args.metric_for_best_model
    
    # Wandb
    training_args.report_to = ['wandb']
    project = 'Unlearning Benchmark'
    group = unlearn_args.data_name + '-' + unlearn_args.backbone + '-cl'
    name = unlearn_args.unlearn_method + '-' + str(unlearn_args.del_ratio) + '-' + str(unlearn_args.random_seed)
    run_id = '-'.join([unlearn_args.data_name, unlearn_args.backbone, 'cl', unlearn_args.unlearn_method, str(unlearn_args.del_ratio), str(unlearn_args.random_seed)])
    wandb.init(project=project, group=group, name=name, config=unlearn_args, id=run_id, resume='allow')


    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )

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

    image_processor = VideoMAEImageProcessor.from_pretrained(model_args.model_name_or_path)
    import pandas as pd
    label_map = pd.read_csv(f'/data/datasets/video_datasets/ucfTrainTestlist/classInd.txt', sep=' ', names=['label_id', 'label_name'])
    label2id = {i: j-1 for i, j in zip(label_map.label_name, label_map.label_id)}
    id2label = {i: label for label, i in label2id.items()}

    ckpt_dir = f'./video/train/checkpoint/{model_map_rev[model_args.model_name_or_path]}/{data_args.dataset_name.split("/")[-1]}_42'
    ori_model = VideoMAEForVideoClassification.from_pretrained(ckpt_dir)#, id2label=id2label, label2id=label2id)

    # Initialize our dataset and prepare it for the audio classification task.
    mean = image_processor.image_mean
    std = image_processor.image_std
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = ori_model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps

    train_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=256, max_size=320),
                        RandomCrop(resize_to),
                        RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        Resize(resize_to),
                    ]
                ),
            ),
        ]
    )
    
    # Unlearning data
    import torch
    from m3u.data.video import UCF101
    from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths

    train_labeled_video_paths = LabeledVideoPaths.from_path('/data/datasets/video_datasets/ucfTrainTestlist/train.txt')
    train_labeled_video_paths.path_prefix = '/data/datasets/video_datasets/UCF-101'
    
    val_labeled_video_paths = LabeledVideoPaths.from_path('/data/datasets/video_datasets/ucfTrainTestlist/test.txt')
    val_labeled_video_paths.path_prefix = '/data/datasets/video_datasets/UCF-101'

    data_path, dr_path, df_path, df_for_train_path, dr_for_eval_path = prepare_deletion_data_video(unlearn_args, train_labeled_video_paths) 
    data_path.path_prefix = '/data/datasets/video_datasets/UCF-101'
    dr_path.path_prefix = '/data/datasets/video_datasets/UCF-101'
    df_path.path_prefix = '/data/datasets/video_datasets/UCF-101'
    df_for_train_path.path_prefix = '/data/datasets/video_datasets/UCF-101'
    dr_for_eval_path.path_prefix = '/data/datasets/video_datasets/UCF-101'

    train_dataset = UCF101(
        labeled_video_paths=data_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    dr_dataset = UCF101(
        labeled_video_paths=dr_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    df_dataset = UCF101(
        labeled_video_paths=df_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        video_sampler=torch.utils.data.SequentialSampler,
        transform=val_transform,
    )

    df_for_train_dataset = UCF101(
        labeled_video_paths=df_for_train_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )
    
    val_dataset = UCF101(
        labeled_video_paths=val_labeled_video_paths,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        video_sampler=torch.utils.data.SequentialSampler,
        transform=val_transform,
    )

    dr_for_eval_dataset = UCF101(
        labeled_video_paths=dr_for_eval_path,
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        video_sampler=torch.utils.data.SequentialSampler,
        transform=val_transform,
    )

    raw_datasets = {
        'train': train_dataset,
        'eval': val_dataset,
        'df': df_dataset,
        'dr': dr_dataset,
        'df_for_train': df_for_train_dataset,
    }


    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy", cache_dir=model_args.cache_dir)

    # Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with
    # `predictions` and `label_ids` fields) and has to return a dictionary string to float.
    def compute_metrics(eval_pred):
        """Computes accuracy on a batch of predictions"""
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return metric.compute(predictions=predictions, references=eval_pred.label_ids)

    def collate_fn(examples):
        # permute to (num_frames, num_channels, height, width)
        pixel_values = torch.stack(
            [example["video"].permute(1, 0, 2, 3) for example in examples]
        )
        labels = torch.tensor([example["label"] for example in examples])
        return {"pixel_values": pixel_values, "labels": labels}

    # Initialize our trainer
    if unlearn_args.unlearn_method == 'bad_teaching':
        from m3u.trainer import BadTeachingTrainer
        training_args.per_device_train_batch_size = training_args.per_device_train_batch_size // 2
        trainer = BadTeachingTrainer(
            model=ori_model,
            args=training_args,
            unlearn_config=unlearn_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
            raw_datasets=raw_datasets,
        )

    else:
        trainer = UnlearningTrainer(
            model=ori_model,
            args=training_args,
            unlearn_config=unlearn_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
            raw_datasets=raw_datasets,
        )
    trainer.unlearning = True
    trainer.use_cl = True
    assert trainer.use_cl == True

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.unlearn(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix='eval')
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)

        logger.info("*** Dr ***")
        metrics = trainer.evaluate(eval_dataset=dr_for_eval_dataset, metric_key_prefix='train')
        trainer.log_metrics("dr", metrics)
        trainer.save_metrics("dr", metrics)

    # Write model card and (optionally) push to hub
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "tasks": "video-classification",
        "dataset": data_args.dataset_name,
        "tags": ["video-classification"],
    }
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
