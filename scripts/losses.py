"""Long-tail classification losses."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _inverse_freq_weights(cls_num_list, device) -> torch.Tensor:
    counts = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
    weights = 1.0 / counts
    return weights / weights.sum() * len(cls_num_list)


class WeightedCELoss(nn.Module):
    """Inverse-frequency weighted cross-entropy."""

    def __init__(self, cls_num_list, device):
        super().__init__()
        self.weight = _inverse_freq_weights(cls_num_list, device)

    def forward(self, logits, target):

        per_sample = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )
        return per_sample.mean()


class FocalLoss(nn.Module):
    """Unweighted focal loss with focusing parameter ``gamma``."""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, reduction="none")
        pt = torch.exp(-ce)
        return (((1 - pt) ** self.gamma) * ce).mean()


class ClassBalancedLoss(nn.Module):
    """Class-balanced cross-entropy using effective-number weights."""

    def __init__(self, cls_num_list, device, beta: float = 0.9999):
        super().__init__()
        counts = np.asarray(cls_num_list, dtype=np.float64)
        effective_num = 1.0 - np.power(beta, counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * len(cls_num_list)
        self.weight = torch.tensor(weights, dtype=torch.float32, device=device)

    def forward(self, logits, target):

        per_sample = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none"
        )
        return per_sample.mean()


class LDAMLoss(nn.Module):
    """LDAM with class-dependent margins; DRW is excluded because it is a schedule."""

    def __init__(self, cls_num_list, device, max_m: float = 0.5, s: float = 30.0):
        super().__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(np.asarray(cls_num_list, dtype=np.float64)))
        m_list = m_list * (max_m / np.max(m_list))
        self.m_list = torch.tensor(m_list, dtype=torch.float32, device=device)
        self.s = s

    def forward(self, logits, target):
        index = torch.zeros_like(logits, dtype=torch.bool)
        index.scatter_(1, target.view(-1, 1), True)
        batch_m = self.m_list[target].view(-1, 1)
        logits_m = logits - batch_m
        adjusted = torch.where(index, logits_m, logits)
        return F.cross_entropy(self.s * adjusted, target)


class LogitAdjustedLoss(nn.Module):
    """Logit-adjusted cross-entropy with temperature ``tau``."""

    def __init__(self, cls_num_list, device, tau: float = 1.0):
        super().__init__()
        counts = torch.tensor(cls_num_list, dtype=torch.float32, device=device)
        self.log_prior = tau * torch.log(counts / counts.sum())

    def forward(self, logits, target):
        return F.cross_entropy(logits + self.log_prior, target)


LOSS_NAMES = (
    "cross_entropy",
    "weighted_ce",
    "focal",
    "class_balanced",
    "ldam",
    "logit_adjusted",
)


def build_loss(name: str, cls_num_list, device, **kwargs) -> tuple[nn.Module, dict]:
    """Build a loss and return its resolved hyperparameters."""
    if name == "cross_entropy":
        return nn.CrossEntropyLoss(), {}
    if name == "weighted_ce":
        return WeightedCELoss(cls_num_list, device), {}
    if name == "focal":
        gamma = kwargs.get("gamma", 2.0)
        return FocalLoss(gamma=gamma), {"gamma": gamma}
    if name == "class_balanced":
        beta = kwargs.get("beta", 0.9999)
        return ClassBalancedLoss(cls_num_list, device, beta=beta), {"beta": beta}
    if name == "ldam":
        max_m = kwargs.get("max_m", 0.5)
        s = kwargs.get("s", 30.0)
        return LDAMLoss(cls_num_list, device, max_m=max_m, s=s), {
            "max_m": max_m,
            "s": s,
        }
    if name == "logit_adjusted":
        tau = kwargs.get("tau", 1.0)
        return LogitAdjustedLoss(cls_num_list, device, tau=tau), {"tau": tau}
    raise ValueError(f"unknown loss {name!r}; must be one of {LOSS_NAMES}")
