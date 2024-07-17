import os
import copy
import time
import math
import wandb
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.utils.data import DataLoader
from typing import *
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import speed_metrics

# from m3u.evaluation import Evaluator

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
)



class GenTrainer(Trainer):
    def __init__(self, **kwargs):
        if 'vae' in kwargs:
            self.vae = kwargs['vae']
            kwargs.pop('vae')
        if 'text_encoder' in kwargs:
            self.text_encoder = kwargs['text_encoder']
            kwargs.pop('text_encoder')
        if 'noise_scheduler':
            self.noise_scheduler = kwargs['noise_scheduler']
            kwargs.pop('noise_scheduler')
        # self.unlearning = True if 'unlearn_config' in kwargs else False

        # if self.unlearning:
        #     self.raw_datasets = kwargs['raw_datasets']  # Used for computing performance on Df
        #     kwargs.pop('raw_datasets')
        #     self.unlearn_config = kwargs['unlearn_config']
        #     kwargs.pop('unlearn_config')

        super().__init__(**kwargs)
        # self.num_labels = self.model.config.num_labels if self.model.config.num_labels is not None else None
        # self.unlearn_time = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        weight_dtype = torch.float32
        clean_images = inputs["input"].to(weight_dtype)
        bsz = clean_images.shape[0]

        noise = torch.randn(clean_images.shape, dtype=weight_dtype, device=clean_images.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
        ).long()

        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # with accelerator.accumulate(model):
            # Predict the noise residual
        model_output = model(noisy_images, timesteps).sample

        # if self.args.prediction_type == "epsilon":
        loss = F.mse_loss(model_output.float(), noise.float())  # this could have different weights!
        # elif self.args.prediction_type == "sample":
        #     alpha_t = _extract_into_tensor(
        #         self.noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
        #     )
        #     snr_weights = alpha_t / (1 - alpha_t)
        #     # use SNR weighting from distillation paper
        #     loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
        #     loss = loss.mean()
        # else:
        #     raise ValueError(f"Unsupported prediction type: {self.args.prediction_type}")

        return (loss, model_output) if return_outputs else loss

    def get_df_logit(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="df"):
        start_time = time.time()
        
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation on Df",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        out_path = os.path.join(self.args.output_dir, f'pred_logit_df')
        np.save(out_path, output.predictions)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        # self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", split_name=None):
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            eval_dataloader,
            description="Evaluation",
            # No point gathering the predictions if there are no metrics, otherwise we defer to
            # self.args.prediction_loss_only
            prediction_loss_only=True if self.compute_metrics is None else None,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        # if split_name is not None:
        out_path = os.path.join(self.args.output_dir, f'pred_logit_{metric_key_prefix}')
        np.save(out_path, output.predictions)

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        ## Update unlearning metrics
        # test_logit = np.load(f'checkpoint/{model_map_rev[model_args.model_name_or_path]}/{data_args.dataset_name}_{data_args.seed}/pred_logit_validation.npy')

        if self.unlearning and metric_key_prefix is not None and metric_key_prefix != 'train':
            test_logit = output.predictions
            # test_label = self.get_data(metric_key_prefix)['label']
            test_label = output.label_ids
            df_output = self.get_df_logit(self.get_data('df'))
            df_logit = df_output.predictions
            # df_label = self.get_data('df')['label']
            df_label = df_output.label_ids
            # df_label = self.get_data('df')['label' if self.unlearn_config.unlearn_method != 'random_label' else 'ori_label']

            evaluator = Evaluator(None, test_label, test_logit, df_label, df_logit, dr_label=None, dr_logit=None, df_mask=None)
            unlearn_metrics = evaluator.compute()
            unlearn_metrics[metric_key_prefix + '_' + self.args.metric_for_best_model] = unlearn_metrics[self.args.metric_for_best_model]
            unlearn_metrics['unlearn_time'] = self.unlearn_time if self.unlearn_time is not None else -1
            output.metrics.update(unlearn_metrics)


        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
