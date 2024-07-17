#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import sys
from dataclasses import dataclass, field
from typing import *
import argparse
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import TrainingArguments, HfArgumentParser, CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

sys.path.append(os.getcwd())
from m3u.generative_trainer import GenerativeTrainer as Trainer
from m3u.args import UnlearningArguments
from m3u.data.base import DeletionData, prepare_deletion_data_class_unlearn

if is_wandb_available():
    import wandb


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.28.0.dev0")

logger = get_logger(__name__, log_level="INFO")

DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}

model_map_rev = {
    'CompVis/stable-diffusion-v1-4': 'sd',
}


def log_validation(vae, text_encoder, tokenizer, unet, args, accelerator, weight_dtype, epoch):
    logger.info("Running validation... ")

    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_name_or_path,
        vae=accelerator.unwrap_model(vae),
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        unet=accelerator.unwrap_model(unet),
        safety_checker=None,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.enable_xformers_memory_efficient_attention:
        pipeline.enable_xformers_memory_efficient_attention()

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    images = []
    for i in range(len(args.validation_prompts)):
        if torch.backends.mps.is_available():
            autocast_ctx = nullcontext()
        else:
            autocast_ctx = torch.autocast(accelerator.device.type)

        with autocast_ctx:
            image = pipeline(args.validation_prompts[i], num_inference_steps=20, generator=generator).images[0]

        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        elif tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompts[i]}")
                        for i, image in enumerate(images)
                    ]
                }
            )
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

    del pipeline
    torch.cuda.empty_cache()

    return images



@dataclass
class ImageGenTrainingArguments(TrainingArguments):
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    input_perturbation: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "The scale of input perturbation. Recommended 0.1."
        },
    )
    model_name_or_path: Optional[str] = field(
        default='text',
        metadata={
            "help": (
                "The name of the text column in the input dataset or a CSV/JSON file."
                'If not specified, will use the "sentence" column for single/multi-label classifcation task.'
            )
        },
    )
    revision: Optional[str] = field(
        default=None, metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    variant: str = field(
        default=None, metadata={"help": "THe delimiter to use to join text columns into a single sentence."}
    )
    train_data_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": 'The name of the train split in the input dataset. If not specified, will use the "train" split when do_train is enabled'
        },
    )
    image_column: str = field(
        default='image',
        metadata={
            "help": 'The name of the validation split in the input dataset. If not specified, will use the "validation" split when do_eval is enabled'
        },
    )
    caption_column: str = field(
        default='text',
        metadata={
            "help": 'The name of the test split in the input dataset. If not specified, will use the "test" split when do_predict is enabled'
        },
    )
    validation_prompts: Optional[str] = field(
        default=None,
        metadata={"help": "The splits to remove from the dataset. Multiple splits should be separated by commas."},
    )
    seed: int = field(
        default=42,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    resolution: int = field(
        default=512,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "},
    )
    snr_gamma: Optional[float] = field(
        default=5.0,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    use_ema: bool = field(
        default=False, metadata={"help": "A csv or a json file containing the training data."}
    )
    dream_training: bool = field(default=False, metadata={"help": "The metric to use for evaluation."})
    dream_detail_preservation: float = field(
        default=1.0, metadata={"help": "A csv or a json file containing the validation data."}
    )
    noise_offset: int = field(default=0, metadata={"help": "A csv or a json file containing the test data."})

    def __post_init__(self):
        super().__post_init__()
        # if self.dataset_name is None:
        #     if self.train_file is None or self.validation_file is None:
        #         raise ValueError(" training/validation file or a dataset name.")

        #     train_extension = self.train_file.split(".")[-1]
        #     assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        #     validation_extension = self.validation_file.split(".")[-1]
        #     assert (
        #         validation_extension == train_extension
        #     ), "`validation_file` should have the same extension (csv or json) as `train_file`."


def main():
    # See all possible arguments in src/transformers/args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ImageGenTrainingArguments, UnlearningArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args, unlearn_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        args, unlearn_args = parser.parse_args_into_dataclasses()


    args.report_to = []
    args.remove_unused_columns = False
    args.evaluation_strategy = 'none'
    args.save_strategy = 'none'
    args.metric_for_best_model = 'unlearn_overall'

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_warning()
    diffusers.utils.logging.set_verbosity_info()

    set_seed(args.seed)

    if unlearn_args.data_name is None:
        if args.dataset_config_name is not None:
            unlearn_args.data_name = args.dataset_name + '_' + args.dataset_config_name
        else:
            unlearn_args.data_name = args.dataset_name
        unlearn_args.random_seed = args.seed
        unlearn_args.backbone = model_map_rev[args.model_name_or_path]

    if unlearn_args.unlearn_method in ['bad_teaching']:
        args.dataloader_drop_last = True
        args.remove_unused_columns = False
    
    # args.metric_for_best_model = 'unlearn_overall_' + args.metric_for_best_model

    # Wandb
    # args.report_to = ['wandb']
    # project = 'Unlearning Benchmark'
    # group = unlearn_args.data_name + '-' + unlearn_args.backbone
    # name = unlearn_args.unlearn_method + '-' + str(unlearn_args.del_ratio) + '-' + str(unlearn_args.random_seed)
    # run_id = '-'.join([unlearn_args.data_name, unlearn_args.backbone, unlearn_args.unlearn_method, str(unlearn_args.del_ratio), str(unlearn_args.random_seed)])
    # wandb.init(project=project, group=group, name=name, config=unlearn_args, id=run_id, resume='allow')


    raw_datasets = load_dataset('zh-plus/tiny-imagenet')
    

    # 6. Get the column names for input/target.
    image_column = args.image_column
    caption_column = args.caption_column

    # Preprocessing the datasets.
    # We need to tokenize input captions and transform the images.
    def tokenize_captions(examples, is_train=True):
        captions = []
        for caption in examples[caption_column]:
            if isinstance(caption, str):
                captions.append(caption)
            elif isinstance(caption, (list, np.ndarray)):
                # take a random caption if there are multiple
                captions.append(random.choice(caption) if is_train else caption[0])
            else:
                raise ValueError(
                    f"Caption column `{caption_column}` should contain either strings or lists of strings."
                )
        inputs = tokenizer(
            captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids

    # Preprocessing the datasets.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(args.resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        examples["input_ids"] = tokenize_captions(examples)
        return examples


    # Unlearning data
    raw_datasets['ori_train'] = raw_datasets['train']
    raw_datasets['train'], raw_datasets['dr'], raw_datasets['df'], raw_datasets['df_for_train'] = prepare_deletion_data_class_unlearn(unlearn_args, raw_datasets['ori_train'], 'label')


    from imagenet_mapping import imagenet_mapping
    def add_text(examples):
        examples['text'] = [f'an image of {imagenet_mapping[i]}' for i in examples['label']]
        return examples
    raw_datasets = raw_datasets.map(add_text, batched=True)

    raw_datasets["train"] = raw_datasets["train"].with_transform(preprocess_train)
    raw_datasets["df_for_train"] = raw_datasets["df_for_train"].with_transform(preprocess_train)



    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack([example["input_ids"] for example in examples])
        return {"pixel_values": pixel_values, "input_ids": input_ids}


    # Model
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(
        args.model_name_or_path, subfolder="tokenizer", revision=args.revision
    )

    text_encoder = CLIPTextModel.from_pretrained(
        args.model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    vae = AutoencoderKL.from_pretrained(
        args.model_name_or_path, subfolder="vae", revision=args.revision, variant=args.variant
    )
    model = UNet2DConditionModel.from_pretrained(
        args.model_name_or_path, subfolder="unet"
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    model.train()

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = UNet2DConditionModel.from_pretrained(
            args.model_name_or_path, subfolder="unet", revision=args.revision, variant=args.variant
        )
        ema_unet = EMAModel(ema_unet.parameters(), model_cls=UNet2DConditionModel, model_config=ema_unet.config)

    model.enable_gradient_checkpointing()


    trainer = Trainer(
        args=args,
        unlearn_config=unlearn_args,
        model=model,
        vae=vae,
        text_encoder=text_encoder,
        train_dataset=raw_datasets["train"],
        eval_dataset=raw_datasets["valid"],
        noise_scheduler=noise_scheduler,
        data_collator=collate_fn,
        raw_datasets=raw_datasets,
    )
    trainer.unlearning = True

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate(eval_dataset=raw_datasets["valid"], metric_key_prefix="eval")
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    logger.info("*** Test ***")
    metrics = trainer.evaluate(eval_dataset=raw_datasets["valid"], metric_key_prefix="test")
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    logger.info("*** Dr ***")
    if unlearn_args.unlearn_method in ['bad_teaching']:
        raw_datasets['dr'] = raw_datasets['dr'].remove_columns(['is_df'])
    metrics = trainer.evaluate(eval_dataset=raw_datasets['dr'], metric_key_prefix="dr")
    trainer.log_metrics("dr", metrics)
    trainer.save_metrics("dr", metrics)


if __name__ == "__main__":
    main()