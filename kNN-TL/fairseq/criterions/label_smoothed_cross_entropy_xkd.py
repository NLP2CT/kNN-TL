# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import numpy as np
import pickle
import os

from fairseq import metrics, utils, checkpoint_utils
from fairseq.criterions import register_criterion

from .label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
)

from dataclasses import dataclass, field
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional
from fairseq.dataclass import ChoiceEnum
LOSS_TYPE = ChoiceEnum(['bi_kl', 'js', 'kl', 'fake_js', 'mse', 'js_mse', 'ffn_mse', 'knn_mse'])
@dataclass
class LabelSmoothedCrossEntropy_XKD_CriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    kd_weight: float = field(default=0.5)
    prior_tau: float = field(default=1.0)
    teacher_dir: Optional[str] = field(default=None)
    loss_type: LOSS_TYPE = field(default='kl')
    teacher_data: Optional[str] = field(default=None)


@register_criterion(
    "label_smoothed_cross_entropy_xkd",
    dataclass=LabelSmoothedCrossEntropy_XKD_CriterionConfig,
)
class LabelSmoothedCrossEntropy_XKD_Criterion(
    LabelSmoothedCrossEntropyCriterion
):
    def __init__(
        self,task, sentence_avg, label_smoothing,
        kd_weight, prior_tau, teacher_dir,
        loss_type, teacher_data,
        ):
        super().__init__(task, sentence_avg, label_smoothing)
        self.kd_weight = kd_weight
        self.prior_tau = prior_tau
        self.teacher_dir = teacher_dir
        self.loss_type = loss_type
        self.teacher_data = teacher_data
        self.hidden_mapping_loss = nn.MSELoss(reduction="none")

    def aux_forward(self, teacher, model, aux_net_input):
        aux_net_output = teacher(**aux_net_input)

        return aux_net_output

    def mean_knn_keys(self, keys, vals, distances, temperature, device):
        r"""
        How vanilla knn-mt calculates knn probs using retrieved vals and distances.
        """
        scaled_dists = - distances / temperature
        knn_weights = torch.softmax(scaled_dists, dim=-1).unsqueeze(-1)
        keys = keys*knn_weights
        keys = keys.sum(dim=-2)

        return keys

    def kd_loss(self, model, net_output, aux_src_net_output, pad_mask=None, reduce=True, non_pad_idx=None, loss_type=None, sample_id = None):
        if self.loss_type == 'js_mse':
            self.loss_type = loss_type

        if self.loss_type == 'kl':
            q = model.get_normalized_probs((net_output[0]/self.prior_tau,), log_probs=True)
            p = model.get_normalized_probs((aux_src_net_output[0]/self.prior_tau,), log_probs=False)
            kd_loss = F.kl_div(q, p, reduction='none')
            if pad_mask is not None:
                kd_loss = kd_loss.masked_fill(pad_mask, 0.0)
            if reduce:
                kd_loss = kd_loss.sum()
            return kd_loss*(self.prior_tau**2)

        elif self.loss_type == 'js' :
            p = model.get_normalized_probs((net_output[0]/self.prior_tau,), log_probs=False)
            q = model.get_normalized_probs((aux_src_net_output[0]/self.prior_tau,), log_probs=False)
            mean = (p+q)/2.0
            mean_log = mean.log()
            p_flat = model.get_normalized_probs((net_output[0]/self.prior_tau,), log_probs=False)
            q_flat = model.get_normalized_probs((aux_src_net_output[0]/self.prior_tau,), log_probs=False)
            p_mean = F.kl_div(mean_log, p_flat, reduction='none')
            q_mean = F.kl_div(mean_log, q_flat, reduction='none')
            if pad_mask is not None:
                p_mean = p_mean.masked_fill(pad_mask, 0.0)
                q_mean = q_mean.masked_fill(pad_mask, 0.0)
            if reduce:
                p_mean = p_mean.sum()
                q_mean = q_mean.sum()
            kd_loss = (p_mean + q_mean)/2.0
            return kd_loss*(self.prior_tau**2)

        elif self.loss_type == 'mse':
            p_decoder_hidden = net_output[1]['inner_states'][-1].float()
            q_decoder_hidden = aux_src_net_output[1]['inner_states'][-1]
            p_decoder_hidden = p_decoder_hidden.reshape(-1, p_decoder_hidden.size(-1))
            q_decoder_hidden = q_decoder_hidden.reshape(-1, q_decoder_hidden.size(-1))
            p_decoder_hidden = p_decoder_hidden.index_select(dim=0, index=non_pad_idx)
            q_decoder_hidden = q_decoder_hidden.index_select(dim=0, index=non_pad_idx)
            hidden_mapping_loss = self.hidden_mapping_loss(p_decoder_hidden, q_decoder_hidden)
            hidden_mapping_loss = hidden_mapping_loss.sum(dim=-1).sum(dim=-1)
            return hidden_mapping_loss

        elif self.loss_type == 'ffn_mse':
            p_decoder_hidden = net_output[1]['last_ffn_input'].float()
            q_decoder_hidden = aux_src_net_output[1]['last_ffn_input']
            p_decoder_hidden = p_decoder_hidden.reshape(-1, p_decoder_hidden.size(-1))
            q_decoder_hidden = q_decoder_hidden.reshape(-1, q_decoder_hidden.size(-1))
            p_decoder_hidden = p_decoder_hidden.index_select(dim=0, index=non_pad_idx)
            q_decoder_hidden = q_decoder_hidden.index_select(dim=0, index=non_pad_idx)
            hidden_mapping_loss = self.hidden_mapping_loss(p_decoder_hidden, q_decoder_hidden)
            hidden_mapping_loss = hidden_mapping_loss.sum(dim=-1).sum(dim=-1)
            return hidden_mapping_loss

    def forward(self, model, sample, teacher=None, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        if not model.training:
            return super().forward(model, sample, reduce=reduce)
        
        net_output = model(**sample["net_input"])
        aux_src_net_output = self.aux_forward(teacher, model, sample["aux_net_input"])
        
        target = model.get_targets(sample, net_output).unsqueeze(-1)
        padding_mask = target.eq(self.padding_idx)
        non_pad_mask = target.ne(self.padding_idx)
        non_pad_mask = non_pad_mask.view(-1,1)
        non_pad_idx = non_pad_mask.nonzero(as_tuple=True)[0]
        if self.loss_type == 'js_mse':
            kd_loss = self.kd_loss(model, net_output, aux_src_net_output, pad_mask=padding_mask, reduce=True, non_pad_idx=non_pad_idx, loss_type='js')
            mse_loss = self.kd_loss(model, net_output, aux_src_net_output, pad_mask=padding_mask, reduce=True, non_pad_idx=non_pad_idx, loss_type='mse')
        else:
            mse_loss = self.kd_loss(model, net_output, aux_src_net_output, pad_mask=padding_mask, reduce=True, non_pad_idx=non_pad_idx,sample_id = sample["id"])
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        aux_loss, aux_nll_loss = self.compute_loss(model, aux_src_net_output, sample, reduce=reduce)

        if self.loss_type == 'js_mse':
            loss = loss + self.kd_weight*700*kd_loss + self.kd_weight*mse_loss
 
        else:
            loss = loss + self.kd_weight*mse_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        kd_loss = torch.tensor(0)
        logging_output = {
            "loss": loss.data,
            "kd_loss": kd_loss.data,
            "mse_loss": mse_loss.data,
            "aux_loss": aux_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        aux_loss_sum = sum(log.get("aux_loss", 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get("kd_loss", 0) for log in logging_outputs)
        mse_loss_sum = sum(log.get("mse_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "aux_loss", aux_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mse_loss", mse_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "kd_loss", kd_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )