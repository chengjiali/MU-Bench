import torch
import torch.nn as nn



class NegGradUnlearning(nn.Module):
    def __init__(self, unlearn_config, ori_model):
        super().__init__()

        self.unlearn_config = unlearn_config
        self.unlearned_model = ori_model
        self.good_teacher = copy.deepcopy(ori_model)
        self.bad_teacher = copy.deepcopy(ori_model)
        for layer in self.bad_teacher.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for n, p in self.good_teacher.named_parameters():
            p.requires_grad = False
        for n, p in self.bad_teacher.named_parameters():
            p.requires_grad = False   
        self.good_teacher.eval()
        self.bad_teacher.eval()

    def forward(self, df_input_ids=None, df_attention_mask=None, df_label=None, dr_input_ids=None, dr_attention_mask=None, dr_label=None):

        if self.training:
            dr_outputs = self.unlearned_model(dr_input_ids, attention_mask=dr_attention_mask, return_dict=True)
            df_outputs = self.unlearned_model(df_input_ids, attention_mask=df_attention_mask, return_dict=True)

            self.good_teacher.eval()
            self.bad_teacher.eval()
            with torch.no_grad():
                good_outputs = self.good_teacher(dr_input_ids, attention_mask=dr_attention_mask, return_dict=True)
                bad_outputs = self.bad_teacher(df_input_ids, attention_mask=df_attention_mask, return_dict=True)

            kl_loss = nn.KLDivLoss(reduction="batchmean")

            loss1 = self.unlearn_config.alpha * kl_loss(dr_outputs.logits, good_outputs.logits)
            loss2 = (1 - self.unlearn_config.alpha) * kl_loss(df_outputs.logits, bad_outputs.logits)
            loss = loss1 + loss2

            return ori_output

        else:
            