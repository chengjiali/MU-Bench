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
from transformers import Trainer, Seq2SeqTrainer
from transformers.trainer_pt_utils import nested_detach
from transformers.trainer_utils import speed_metrics

from m3u.evaluation import Evaluator, TextGenEvaluator
from . import superloss

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
from transformers.utils import logging

logger = logging.get_logger(__name__)


def get_norm(model):
    total = 0
    for n, p in model.named_parameters():
        total += torch.norm(p)

    return total

def get_children(model: torch.nn.Module):
    # get children form model!
    children = list(model.children())
    flatt_children = []
    if children == []:
        # if model has no children; model is last child! :O
        return model
    else:
       # look for children from children... to the last child!
       for child in children:
            try:
                flatt_children.extend(get_children(child))
            except TypeError:
                flatt_children.append(get_children(child))
    return flatt_children

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]


super_loss = superloss.SuperLoss('sl', lam=10, mode='avg')

def calculate_superloss(b_loss, batch):
    conf, tau, tau_adjusted = super_loss(b_loss, None, None)
    tau = [tau] * b_loss.shape[0]
    tau_adjusted = [tau_adjusted] * b_loss.shape[0]
    sl_loss = b_loss * conf

    return sl_loss


class UnlearningTrainer(Trainer):
    def __init__(self, **kwargs):
        self.unlearning = True if 'unlearn_config' in kwargs else False

        if self.unlearning:
            self.raw_datasets = kwargs['raw_datasets']  # Used for computing performance on Df
            kwargs.pop('raw_datasets')
            self.unlearn_config = kwargs['unlearn_config']
            kwargs.pop('unlearn_config')

        super().__init__(**kwargs)
        self.num_labels = self.model.config.num_labels if hasattr(self.model.config, 'num_labels') else None
        self.unlearn_time = None
        self.use_cl = True if self.unlearning and self.unlearn_config.use_cl else False

        if self.unlearning:
            self.method_specific_setup()

    def method_specific_setup(self):
        if self.unlearn_config.unlearn_method == 'bad_teaching':
            self.good_teacher = copy.deepcopy(self.model)
            self.bad_teacher = copy.deepcopy(self.model)

            layers = get_children(self.bad_teacher)
            _ = [l.reset_parameters() for l in layers if hasattr(l, 'reset_parameters')]

            for (n1, p1), (n2, p2) in zip(self.good_teacher.named_parameters(), self.bad_teacher.named_parameters()):
                assert n1 == n2
                if n1 == 'wav2vec2.masked_spec_embed':   # For wav2vec2
                    continue
                if not p1.requires_grad:
                    continue
                print(f'bad teacher {n1} same as original?', (p1 == p2).all())
                # assert not (p1 == p2).all(), f"{n1}, {n2}"
                p1.requires_grad = False
                p2.requires_grad = False

            self.good_teacher.eval()
            self.bad_teacher.eval()

        elif self.unlearn_config.unlearn_method == 'scrub':
            self.ori_model = copy.deepcopy(self.model)
            for n, p in self.ori_model.named_parameters():
                p.requires_grad = False
            self.ori_model.eval()
            self.do_max_step = True

        elif self.unlearn_config.unlearn_method == 'salul':
            self.salient_mask = None

    def get_data(self, split_name):
        if split_name == 'eval':
            if 'validation' in self.raw_datasets:
                return self.raw_datasets['validation']
            else:
                return self.raw_datasets['test']
        
        return self.raw_datasets[split_name]

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        if self.use_cl:
            return self.compute_loss_cl(model, inputs, return_outputs)
        else:
            return self.compute_loss_non_cl(model, inputs, return_outputs)

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        if self.unlearn_config.unlearn_method is None or self.unlearn_config.unlearn_method in ['retrain', 'fisher', 'random_label', 'salul']:
            # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
            if return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

            loss = calculate_superloss(loss, inputs).mean()

        elif self.unlearn_config.unlearn_method == 'neggrad':
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = calculate_superloss(loss, inputs).mean()
            loss = -1 * loss

        elif self.unlearn_config.unlearn_method == 'bad_teaching':
            if model.training:
                # Unlearned model
                input_df_mask = inputs['is_df'] == 1
                inputs.pop('is_df')
                dr_inputs = {k: v[~input_df_mask] for k, v in inputs.items()}
                # dr_inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
                if 'labels' in dr_inputs:
                    dr_inputs.pop('labels')
                if 'label_ids' in dr_inputs:
                    dr_inputs.pop('label_ids')
                
                dr_outputs = model(**dr_inputs, return_dict=True)
                with torch.no_grad():
                    good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

                df_inputs = {k: v[input_df_mask] for k, v in inputs.items()}
                # df_inputs = {k[len('df_'):]: v for k, v in inputs.items() if k.startswith('df_')}
                if 'labels' in df_inputs:
                    df_inputs.pop('labels')
                if 'label_ids' in df_inputs:
                    df_inputs.pop('label_ids')

                df_outputs = model(**df_inputs, return_dict=True)
                with torch.no_grad():
                    bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

                kl_loss = nn.KLDivLoss(reduction='none' if self.use_cl else "batchmean")

                dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
                df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)

                dr_loss = calculate_superloss(dr_loss, dr_inputs).mean()
                df_loss = calculate_superloss(df_loss, df_inputs).mean()

                loss = dr_loss + df_loss
                outputs = df_outputs

            else:
                if 'is_df' in inputs:
                    inputs.pop('is_df')
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss = loss.mean()

        elif self.unlearn_config.unlearn_method == 'scrub':
            if model.training:
                if self.do_max_step:
                    # Do max epoch on Df. Use Df as the training set. We only need prediction, not label
                    if 'labels' in inputs:
                        inputs.pop('labels')
                    if 'label_ids' in inputs:
                        inputs.pop('label_ids')
                    outputs = model(**inputs, return_dict=True)
                    with torch.no_grad():
                        ori_outputs = self.ori_model(**inputs, return_dict=True)

                    kl_loss = nn.KLDivLoss(reduction='none')
                    loss = kl_loss(outputs.logits, ori_outputs.logits)
                    loss = calculate_superloss(loss, inputs).mean()
                    loss = -1 * loss

                else:
                    # We need label for task loss
                    outputs = model(**inputs, return_dict=True)
                    with torch.no_grad():
                        ori_outputs = self.ori_model(**inputs, return_dict=True)

                    kl_loss = nn.KLDivLoss(reduction='none')
                    loss = outputs.loss + kl_loss(outputs.logits, ori_outputs.logits).mean(axis=1)
                    loss = calculate_superloss(loss, inputs).mean()

            else:
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                loss.mean()

        return (loss, outputs) if return_outputs else loss

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        """
        Compute loss wrapper for unlearning method FineTune
        """
        # Standard training
        if not self.unlearning:
            return super().compute_loss(model, inputs, return_outputs=return_outputs)

        # Unlearning
        if self.unlearn_config.unlearn_method is None or self.unlearn_config.unlearn_method in ['retrain', 'fisher', 'random_label', 'salul']:
            # inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
            if return_outputs:
                loss, outputs = super().compute_loss(model, inputs, return_outputs=return_outputs)
            else:
                loss = super().compute_loss(model, inputs, return_outputs=return_outputs)

        elif self.unlearn_config.unlearn_method == 'neggrad':
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = -1 * loss

        elif self.unlearn_config.unlearn_method == 'bad_teaching':
            if model.training:
                # Unlearned model
                input_df_mask = inputs['is_df'] == 1
                inputs.pop('is_df')
                dr_inputs = {k: v[~input_df_mask] for k, v in inputs.items()}
                # dr_inputs = {k[len('dr_'):]: v for k, v in inputs.items() if k.startswith('dr_')}
                if 'labels' in dr_inputs:
                    dr_inputs.pop('labels')
                if 'label_ids' in dr_inputs:
                    dr_inputs.pop('label_ids')
                
                dr_outputs = model(**dr_inputs, return_dict=True)
                with torch.no_grad():
                    good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

                df_inputs = {k: v[input_df_mask] for k, v in inputs.items()}
                # df_inputs = {k[len('df_'):]: v for k, v in inputs.items() if k.startswith('df_')}
                if 'labels' in df_inputs:
                    df_inputs.pop('labels')
                if 'label_ids' in df_inputs:
                    df_inputs.pop('label_ids')

                df_outputs = model(**df_inputs, return_dict=True)
                with torch.no_grad():
                    bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

                kl_loss = nn.KLDivLoss(reduction="batchmean")
                dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
                df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)
                loss = dr_loss + df_loss
                outputs = df_outputs

            else:
                if 'is_df' in inputs:
                    inputs.pop('is_df')
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        elif self.unlearn_config.unlearn_method == 'scrub':
            if model.training:
                if self.do_max_step:
                    # Do max epoch on Df. Use Df as the training set. We only need prediction, not label
                    if 'labels' in inputs:
                        inputs.pop('labels')
                    if 'label_ids' in inputs:
                        inputs.pop('label_ids')
                    outputs = model(**inputs, return_dict=True)
                    with torch.no_grad():
                        ori_outputs = self.ori_model(**inputs, return_dict=True)

                    kl_loss = nn.KLDivLoss(reduction="batchmean")
                    loss = kl_loss(outputs.logits, ori_outputs.logits)
                    loss = -1 * loss

                else:
                    # Do min epoch on Dr. We need label for task loss
                    outputs = model(**inputs, return_dict=True)
                    with torch.no_grad():
                        ori_outputs = self.ori_model(**inputs, return_dict=True)

                    kl_loss = nn.KLDivLoss(reduction="batchmean")
                    loss = outputs.loss + kl_loss(outputs.logits, ori_outputs.logits)

            else:
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        if self.unlearning and self.unlearn_config.unlearn_method == 'salul':
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            if self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                self.accelerator.backward(loss)

            # Salient Mask
            for name, param in model.named_parameters():
                if name in ['hubert.masked_spec_embed', 'wav2vec2.masked_spec_embed']:
                    continue
                if param.grad is not None:
                    param.grad *= self.salient_mask[name].to(param.grad.device)

            return loss.detach() / self.args.gradient_accumulation_steps

        else:
            return super().training_step(model, inputs)

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()

        if self.unlearn_config.unlearn_method in ['fisher', 'l-codec']:
            fisher_info_matrix = self.compute_fisher()
            self.scrub(fisher_info_matrix)
            out = None

        elif self.unlearn_config.unlearn_method in ['scrub']:
            self.args.num_train_epochs = self.args.num_train_epochs / 2

            self.train_dataset = self.raw_datasets['df_for_train']
            logger.info(f'******** Doing max step for {self.args.num_train_epochs} epochs ********')
            print(f'******** Doing max step for {self.args.num_train_epochs} epochs ********')
            super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

            self.do_max_step = False
            self.train_dataset = self.raw_datasets['train']
            logger.info(f'******** Doing min step for {self.args.num_train_epochs} epochs ********')
            print(f'******** Doing min step for {self.args.num_train_epochs} epochs ********')
            out = super().train(None, trial, ignore_keys_for_eval, **kwargs)

        elif self.unlearn_config.unlearn_method == 'salul':
            self.train_dataset = self.raw_datasets['df_for_train']
            self.salient_mask = self.compute_salient_mask()
            self.train_dataset = self.raw_datasets['train']
            out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        else:
            out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out

    def compute_salient_mask(self):
        if os.path.exists(os.path.join(self.args.output_dir, 'sal_mask_with_0.5.pt')):
            salient_mask = torch.load(os.path.join(self.args.output_dir, 'sal_mask_with_0.5.pt'))
            
        else:
            model = self.model
            model.eval()
            # self._move_model_to_device(self.model, self.args.device)
            train_loader = self.get_train_dataloader()

            gradient = {}
            length = 0
            for inputs in tqdm(train_loader, desc='Salient Map'):
                model.zero_grad()
                inputs = self._prepare_inputs(inputs)
                loss = -1 * super().compute_loss(model, inputs).mean()
                self.accelerator.backward(loss)

                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            if name in gradient:
                                gradient[name] += param.grad.data.to('cpu')
                            else:
                                gradient[name] = 0

            with torch.no_grad():
                for name in gradient:
                    gradient[name] = torch.abs_(gradient[name])

            threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            threshold_list = [0.5]

            for i in threshold_list:
                sorted_dict_positions = {}
                salient_mask = {}

                # Concatenate all tensors into a single tensor
                all_elements = - torch.cat([tensor.flatten() for tensor in gradient.values()])

                # Calculate the threshold index for the top 10% elements
                threshold_index = int(len(all_elements) * i)

                # Calculate positions of all elements
                positions = torch.argsort(all_elements)
                ranks = torch.argsort(positions)

                start_index = 0
                for key, tensor in gradient.items():
                    num_elements = tensor.numel()
                    # tensor_positions = positions[start_index: start_index + num_elements]
                    tensor_ranks = ranks[start_index : start_index + num_elements]

                    sorted_positions = tensor_ranks.reshape(tensor.shape)
                    sorted_dict_positions[key] = sorted_positions

                    # Set the corresponding elements to 1
                    threshold_tensor = torch.zeros_like(tensor_ranks)
                    threshold_tensor[tensor_ranks < threshold_index] = 1
                    threshold_tensor = threshold_tensor.reshape(tensor.shape)
                    salient_mask[key] = threshold_tensor
                    start_index += num_elements

                torch.save(salient_mask, os.path.join(self.args.output_dir, f"sal_mask_with_{i}.pt"))

        return salient_mask
        # torch.load(os.path.join(self.args.output_dir, "sal_mask_with_0.5.pt"))

    def compute_fisher(self):
        if os.path.exists(os.path.join(self.args.output_dir, 'fim.pt')):
            fisher_info_matrix = torch.load(os.path.join(self.args.output_dir, 'fim.pt'))
            
        else:
            model = self.model
            self._move_model_to_device(self.model, self.args.device)
            train_loader = self.get_train_dataloader()
            fisher_info_matrix = {}

            length = 0
            for inputs in tqdm(train_loader, desc='Fisher'):
                inputs = self._prepare_inputs(inputs)
                outputs = model(**inputs)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
                model.zero_grad()
                loss.backward(retain_graph=True)

                logits = outputs.logits.detach()
                prob = F.softmax(logits, 1)
                log_prob = F.log_softmax(logits, 1)
                # gradient = torch.autograd.grad(y, x, retain_graph=True, grad_outputs=torch.ones_like(y))[0]

                for n, p in model.named_parameters():
                    if not p.requires_grad:
                        continue

                    # Some parameters does not have gradients due to with torch.no_grad()
                    if p.grad is None:
                        continue

                    if n not in fisher_info_matrix:
                        fisher_info_matrix[n] = torch.zeros_like(p)
                    else:
                        # fisher_of_p = []
                        for _prob in prob:
                            for y in range(prob.shape[1]):
                                fisher_info_matrix[n] += _prob[y] * p.grad.detach() * p.grad.detach()
                        # fisher_info_matrix[n] += torch.stack(fisher_of_p).sum(0)

                length += logits.shape[0]

            for n, p in fisher_info_matrix.items():
                fisher_info_matrix[n] /= length

            torch.save(fisher_info_matrix, os.path.join(self.args.output_dir, 'fim.pt'))

        return fisher_info_matrix

    def scrub(self, fisher_info_matrix):
        print(f'Model parameter norm before scrubbing:', get_norm(self.model))
        def get_mean_var(n, p, is_base_dist=False, alpha=3e-6):
            '''Source: https://github.com/AdityaGolatkar/SelectiveForgetting/blob/master/Forgetting.ipynb'''
            var = copy.deepcopy(1. / (fisher_info_matrix[n] + 1e-8))
            var = var.clamp(max=1e3)
            if p.size(0) == self.num_labels:
                var = var.clamp(max=1e2)
            var = alpha * var
            
            if p.ndim > 1:
                var = var.mean(dim=1, keepdim=True).expand_as(p).clone()
            if not is_base_dist:
                mu = copy.deepcopy(p.data.clone())
            else:
                mu = copy.deepcopy(p.data.clone())
            if p.size(0) == self.num_labels:
                # Last layer
                var *= 10
            elif p.ndim == 1:
                # BatchNorm
                var *= 10
            return mu, var

        for n, p in self.model.named_parameters():
            if n not in fisher_info_matrix:
                print(f'Parameter {n} not found in fisher information matrix')
                continue
            if p.requires_grad:
                mu, var = get_mean_var(n, p, False, alpha=1e-8)
                try:
                    assert (mu == p.data).all()
                except:
                    breakpoint()
                p.data = mu + var.sqrt() * torch.empty_like(p.data).normal_()
        print(f'Model parameter norm after scrubbing:', get_norm(self.model))


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

            evaluator = Evaluator(None, test_label, test_logit, df_label, df_logit, df_mask=None)
            unlearn_metrics = evaluator.compute()
            unlearn_metrics[metric_key_prefix + '_' + self.args.metric_for_best_model] = unlearn_metrics[self.args.metric_for_best_model]
            unlearn_metrics['unlearn_time'] = self.unlearn_time if self.unlearn_time is not None else -1
            output.metrics.update(unlearn_metrics)


        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics

class BadTeachingTrainer(UnlearningTrainer):
    def prepare_dr_dataloader(self):
        train_dataset = self.raw_datasets['dr']
        data_collator = self.data_collator
        data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return iter(DataLoader(train_dataset, **dataloader_params))

    def unlearn(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs):

        self.unlearn_start_time = time.time()

        self.dr_loader = self.prepare_dr_dataloader()
        out = super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

        self.unlearn_end_time = time.time()
        self.unlearn_time = self.unlearn_end_time - self.unlearn_start_time

        return out

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.use_cl:
            return self.compute_loss_cl(model, inputs, return_outputs)
        else:
            return self.compute_loss_non_cl(model, inputs, return_outputs)

    def compute_loss_non_cl(self, model, inputs, return_outputs=False):
        if model.training:
            # Unlearned model
            df_inputs = inputs
            dr_inputs = next(self.dr_loader)
            dr_inputs = self._prepare_inputs(dr_inputs)

            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction="batchmean")
            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits)
            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def compute_loss_cl(self, model, inputs, return_outputs=False):
        if model.training:
            # Unlearned model
            df_inputs = inputs
            dr_inputs = next(self.dr_loader)
            dr_inputs = self._prepare_inputs(dr_inputs)

            if 'labels' in dr_inputs:
                dr_inputs.pop('labels')
            if 'label_ids' in dr_inputs:
                dr_inputs.pop('label_ids')
            
            dr_outputs = model(**dr_inputs, return_dict=True)
            with torch.no_grad():
                good_outputs = self.good_teacher(**dr_inputs, return_dict=True)

            if 'labels' in df_inputs:
                df_inputs.pop('labels')
            if 'label_ids' in df_inputs:
                df_inputs.pop('label_ids')

            df_outputs = model(**df_inputs, return_dict=True)
            with torch.no_grad():
                bad_outputs = self.bad_teacher(**df_inputs, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction="none")
            dr_loss = kl_loss(dr_outputs.logits, good_outputs.logits).mean(axis=1)
            df_loss = kl_loss(df_outputs.logits, bad_outputs.logits).mean(axis=1)
            dr_loss = calculate_superloss(dr_loss, dr_inputs).mean()
            df_loss = calculate_superloss(df_loss, df_inputs).mean()
            loss = dr_loss + df_loss
            outputs = df_outputs

        else:
            if 'is_df' in inputs:
                inputs.pop('is_df')
            outputs = model(**inputs)
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            loss = loss.mean()

        return (loss, outputs) if return_outputs else loss

class UnlearningSeq2SeqTrainer(UnlearningTrainer, Seq2SeqTrainer):

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **gen_kwargs):

        gen_kwargs = gen_kwargs.copy()

        # Use legacy argument setting if a) the option is not explicitly passed; and b) the argument is set in the
        # training args
        if (
            gen_kwargs.get("max_length") is None
            and gen_kwargs.get("max_new_tokens") is None
            and self.args.generation_max_length is not None
        ):
            gen_kwargs["max_length"] = self.args.generation_max_length
        if gen_kwargs.get("num_beams") is None and self.args.generation_num_beams is not None:
            gen_kwargs["num_beams"] = self.args.generation_num_beams
        # We don't want to drop samples in general
        self.gather_function = self.accelerator.gather
        self._gen_kwargs = gen_kwargs

        
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

            evaluator = TextGenEvaluator(None, test_label, test_logit, df_label, df_logit, dr_label=None, dr_pred=None, df_mask=None, tokenizer=self.tokenizer, metric_names=['rouge'])
            unlearn_metrics = evaluator.compute()
            unlearn_metrics[metric_key_prefix + '_' + self.args.metric_for_best_model] = unlearn_metrics[self.args.metric_for_best_model]
            unlearn_metrics['unlearn_time'] = self.unlearn_time if self.unlearn_time is not None else -1
            output.metrics.update(unlearn_metrics)


        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
