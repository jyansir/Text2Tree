import torch
import numpy as np

def naive_mixup(Xs: torch.Tensor, beta=0.5):
    """Ordinary MixUp"""
    b, d = Xs.shape
    l = np.random.beta(beta, beta)

    shuffled_sample_ids = np.random.permutation(b)

    Xs_shuffled = Xs[shuffled_sample_ids]
    Xs_mixup = l * Xs + (1 - l) * Xs_shuffled

    return Xs_mixup, l, shuffled_sample_ids

def dis_mixup(Xs: torch.Tensor, soft_distance: torch.Tensor):
    """Text2Tree: DML"""
    b, d = Xs.shape
    assert soft_distance.shape == (b,b)
    l = soft_distance / soft_distance.max(1, keepdim=True)[0]
    l = l.unsqueeze(-1)

    Xs1 = Xs.unsqueeze(1).repeat(1, b, 1)
    Xs2 = Xs.unsqueeze(0).repeat(b, 1, 1)
    Xs_mixup = l * Xs1 + (1 - l) * Xs2

    return Xs_mixup.reshape(-1, d), l.reshape(-1)