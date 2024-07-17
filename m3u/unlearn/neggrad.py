import torch
import torch.nn as nn


class NegGradUnlearning(nn.Module):
    def __init__(self, unlearn_config, ori_model):
        super().__init__()

        self.unlearn_config = unlearn_config
        self.ori_model = ori_model

    def forward(self, df_input_ids=None, df_attention_mask=None, df_label=None):
        ori_output = self.ori_model(df_input_ids, attention_mask=df_attention_mask, labels=df_label, return_dict=True)
        ori_output['loss'] = -1 * ori_output['loss']

        return ori_output
