
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
    VideoMAEForVideoClassification,
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

sys.path.append(os.getcwd())
from m3u.trainer import UnlearningTrainer


model_map_rev = {
    'facebook/wav2vec2-base': 'wav2vec2-base',
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
        default='ks', metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
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

    backbones = ['videomae-small', 'videomae-base', 'videomae-large', 'videomae-huge'][1:-1]
    del_ratio = [2.0, 4.0, 6.0, 8.0, 10.0]
    methods = ['neggrad', 'random_label', 'bad_teaching', 'scrub', 'salul']

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    

    # Load the accuracy metric from the datasets package
    metric = evaluate.load("accuracy")

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


    for ck in ['checkpoint', 'cl_checkpoint', 'lora_50_checkpoint']:
        for b in backbones:
            for m in methods:
                for dr in del_ratio:
                    ckpt_dir = f'./video/unlearn/{ck}/{b}/{m}/{dr}/ucf101_42'
                    if not os.path.exists(ckpt_dir):
                        continue

                    if os.path.exists(f'{ckpt_dir}/ood_results.json'):
                        continue
                    print(ck, b, m, dr)
                    training_args = TrainingArguments(output_dir=ckpt_dir, per_device_eval_batch_size=64, dataloader_num_workers=16, report_to=[])

                    training_args.remove_unused_columns = False
                    training_args.overwrite_output_dir = False


                    image_processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
                    model = VideoMAEForVideoClassification.from_pretrained(ckpt_dir)

                    # Initialize our dataset and prepare it for the audio classification task.
                    mean = image_processor.image_mean
                    std = image_processor.image_std
                    if "shortest_edge" in image_processor.size:
                        height = width = image_processor.size["shortest_edge"]
                    else:
                        height = image_processor.size["height"]
                        width = image_processor.size["width"]
                    resize_to = (height, width)

                    num_frames_to_sample = model.config.num_frames
                    sample_rate = 4
                    fps = 30
                    clip_duration = num_frames_to_sample * sample_rate / fps

                    from m3u.data.video import UCF101
                    from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths

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
                    labeled_video_paths = LabeledVideoPaths.from_path('./data/ucf101_ds.txt')
                    labeled_video_paths.path_prefix = './data/UCF101-DS/'
                    import torch
                    val_dataset = UCF101(
                        labeled_video_paths=labeled_video_paths,
                        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
                        decode_audio=False,
                        video_sampler=torch.utils.data.SequentialSampler,
                        transform=val_transform,
                    )

                    
                    # Initialize our trainer
                    trainer = UnlearningTrainer(
                        model=model,
                        args=training_args,
                        eval_dataset=val_dataset,
                        tokenizer=image_processor,
                        compute_metrics=compute_metrics,
                        data_collator=collate_fn,
                    )
                    trainer.unlearning = False


                    logger.info("*** OOD ***")
                    metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix='ood')
                    trainer.log_metrics("ood", metrics)
                    trainer.save_metrics("ood", metrics)


if __name__ == "__main__":
    main()
