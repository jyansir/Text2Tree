import math
from typing import List, Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from aug_utils import naive_mixup

# L2 normalization
def l2norm(tensor):
    if isinstance(tensor, list):
        return [l2norm(t) for t in tensor]
    elif isinstance(tensor, torch.Tensor):
        return F.normalize(tensor, dim=1)
    else:
        raise TypeError


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning
    ---
    Reference: https://github.com/HobbitLong/SupContrast
    """
    def __init__(self, temperature, label_weights='mean'):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = 0.07
        assert label_weights in ['mean', 'weighted']
        self.label_weights = label_weights
    
    @staticmethod
    def supcon_mask(labels: torch.Tensor, device: torch.device, mask_type='multilabel'):
        labels = labels.contiguous().view(-1, 1)
        if mask_type == 'multiclass': # 2
            mask = torch.eq(labels, labels.T).float().to(device)
        elif mask_type == 'multilabel': # 1
            mask = torch.eq(labels, labels.T).float().to(device)
        return mask
    
    def compute_loss(self, logits: torch.Tensor, mask: torch.Tensor, batch_size: int, device: torch.device):
        # mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8) # avoid INF
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # assert not torch.isnan(loss), 'Encounter INF loss, please check there is no INF / invalid feature values'
        return loss
    
    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor = None,
        selfcon=False,
    ):
        """
        1. supervised CL if labels are given
        2. unsupervised CL if not

        Args:
            features: [cls] hidden states
        """
        device = features.device
        assert len(features.shape) == 2, 'used for [cls] token states'
        batch_size = features.shape[0]
        features = l2norm(features)

        if labels is None or selfcon:
            # self-supervised contrastive
            # assert len(features.shape) > 2 and features.shape[1] > 1, '`n_views` dim is needed'
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        else:
            # supervised
            assert labels.shape[0] == batch_size, 'label number should match batch size'
            if labels.ndim == 1:
                # multiclass
                mask = SupConLoss.supcon_mask(labels, device)
            elif labels.ndim == 2:
                # multilabels (mask for each class)
                mask = [SupConLoss.supcon_mask(labels[:, i], device) for i in range(labels.shape[1])]
        
        # compute logits
        anchor_dot_contrast = torch.div(
            features @ features.T,
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # remove affect from self-interaction ?

        if isinstance(mask, list):
            loss = None
            num_labels = len(mask)
            if self.label_weights == 'mean':
                weights = [1 / num_labels for _ in range(num_labels)]
            elif self.label_weights == 'weighted':
                weights = F.normalize(labels.sum(0).float().detach(), p=1, dim=0).cpu().tolist()
            for i in range(num_labels):
                loss = (
                    weights[i] * self.compute_loss(logits, mask[i], batch_size, device) if loss is None
                    else loss + weights[i] * self.compute_loss(logits, mask[i], batch_size, device)
                )
        else:
            loss = self.compute_loss(logits, mask, batch_size, device)
        
        return loss


class SimSurLoss(nn.Module):
    """
    Similarity Surrogate Learning Loss
    ---

    A generalized contrastive learning paradigm based on the Label Tree Structure
    """
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = 0.07
    
    def compute_loss(self, logits: torch.Tensor, mask: torch.Tensor, batch_size: int, device: torch.device):
        # mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        # compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        # assert not torch.isnan(loss), 'Encounter INF loss, please check there is no INF / invalid feature values'
        return loss

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        label_embeddings: torch.Tensor,
        mixup='none', # you can also MixUp features here
        return_dist='none', # gradient dependency for SSL and DML
    ):
        """
        Args
        ---

        `features`: [CLS] hidden states from encoder
        `labels`: labels (Ordinal for multiclass or One-hot for multilabel)
        `label_embeddings`: label semantics from label tree hierarchical embeddings
        """
        device = features.device
        assert len(features.shape) == 2, \
            'designed for [CLS] states in the paper, for usage in other scenarios please implement your SSL'
        assert mixup in ['none', 'naive', 'hidden']
        # assert return_dist in ['none', 'detach', 'grad', 'only_detach', 'only_grad']
        batch_size = features.shape[0]
        if mixup == 'naive':
            features, l, shuffled_sample_ids = naive_mixup(features, beta=4.0)
        features = l2norm(features) # normalize

        # Calculate similarity score for each sample pair
        if labels.ndim == 2: # multilabel
            assert all(labels.view(-1) <= 1) and all(labels.view(-1) >= 0), 'Invalid one-hot encoding'
            
            # fetch embeddings
            label_states = F.normalize(labels.float() @ label_embeddings, p=2, dim=1)
            # Add mixup
            if mixup != 'none':
                # you can also MixUp features here, we do not use MixUp in SSL in this paper
                label_states = l * label_states + (1 - l) * label_states[shuffled_sample_ids]
            # There may exist samples with no labels in the multilabel task, 
            # thus they should be the positive samples of themselves
            similarity_score = label_states @ label_states.T + torch.diag((labels.sum(1) == 0).float())
            # mask out zero similarity like attention score calculation
            similarity_score = similarity_score - 10000 * (similarity_score == 0).float()       
        elif labels.ndim == 1: # multiclass
            labels = labels.long()
            # fetch embeddings
            label_states = F.normalize(torch.index_select(label_embeddings, 0, labels), p=2, dim=1)
            # There must be a label for each sample in the multiclass task
            similarity_score = label_states @ label_states.T
        else:
            raise ValueError('label should be ordinal (multiclass) or one-hot (multilabel)')
        
        # compute logits
        anchor_dot_contrast = torch.div(
            features @ features.T,
            self.temperature
        )
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # final contrastive mask
        mask = torch.softmax(similarity_score, dim=1) # normalize similarity score for stability
        loss = self.compute_loss(logits, mask, batch_size, device)
        # loss = self.compute_loss(logits, mask.detach(), batch_size, device) # detach SSL

        # OLD VER.: return similarity score rather than normalized one (i.e., mask)
        # the NEW VER. is to stabilize training process
        if return_dist == 'none': # only use SSL
            return loss
        elif return_dist == 'detach': # use both but detach DML
            return loss, mask.detach()
        elif return_dist == 'grad': # use both
            return loss, mask
        elif return_dist == 'only_detach': # only use detached DML
            return mask.detach()
        elif return_dist == 'only_grad': # only use DML
            return mask
        else:
            raise TypeError('Invalid mix type')


class FocalLoss(nn.Module):
    """
    Focal Loss
    ---

    Reference: https://github.com/clcarwin/focal_loss_pytorch
    """
    def __init__(self, num_labels, alpha=None, gamma=2., task_type='multiclass'):
        super().__init__()
        assert task_type in ['multiclass', 'multilabel']
        if alpha is None:
            self.alpha = Variable(torch.ones(num_labels)) if task_type == 'multiclass' else Variable(torch.ones(num_labels) * 0.5)
        else:
            assert isinstance(alpha, float) and 0 < alpha < 1 and task_type == 'multilabel'
            self.alpha = Variable(torch.ones(num_labels) * alpha)
        self.gamma = gamma
        self.num_labels = num_labels
    
    def forward(self, logits, labels):
        b, l = logits.shape
        if labels.ndim == 1: # multiclass
            probs = logits.softmax(1)
            one_hot_labels = torch.eye(self.num_labels, device=probs.device)[labels]
            probs = (probs * one_hot_labels).sum(1)
            log_probs = probs.log() # b,
            alpha = self.alpha[labels].to(probs.device)
            loss = -alpha * (torch.pow(1 - probs, self.gamma)) * log_probs
        else: # multilabel
            probs = logits.sigmoid() # b, l
            alpha = self.alpha.view(1, -1).to(probs.device)
            loss = - alpha * torch.pow(1 - probs, self.gamma) * torch.log(probs) * labels \
                - (1 - alpha) * torch.pow(probs, self.gamma) * torch.log(1 - probs) * (1 - labels)
        return loss.mean()
