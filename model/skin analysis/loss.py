import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCELoss(nn.Module):
    """Standard cross-entropy with label smoothing. Kept for backward compatibility."""

    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logit, target):
        return self.ce(logit, target)


class OrdinalSmoothLoss(nn.Module):
    """
    Cross-entropy with ordinal label smoothing.
    Concentrates smoothing mass on adjacent classes to respect grade ordering.
    """

    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, logit, target):
        B, C  = logit.shape
        half  = self.alpha / 2.0

        # Build soft labels: spread alpha/2 to neighbours
        soft = torch.zeros_like(logit).scatter_(1, target.unsqueeze(1), 1.0)
        prev_idx  = (target - 1).clamp(min=0)
        next_idx  = (target + 1).clamp(max=C - 1)
        prev_mask = (target > 0).float().unsqueeze(1)
        next_mask = (target < C - 1).float().unsqueeze(1)

        soft_label = soft * (1.0 - self.alpha)
        soft_label.scatter_add_(1, prev_idx.unsqueeze(1), prev_mask * half)
        soft_label.scatter_add_(1, next_idx.unsqueeze(1), next_mask * half)

        # Boundary classes: use half the smoothing
        boundary = (~(target > 0) | ~(target < C - 1)).float()
        soft_label[boundary.bool()] = soft[boundary.bool()] * (1.0 - self.alpha / 2.0)
        soft_label = soft_label / soft_label.sum(dim=1, keepdim=True).clamp(min=1e-8)

        return -(soft_label * F.log_softmax(logit, dim=1)).sum(dim=1).mean()


class OrdinalMSELoss(nn.Module):
    """
    Ordinal smooth CE + λ * MSE on expected grade.

    E[pred] = Σ_i (i * softmax(logit)[i])

    Args:
        alpha : ordinal smoothing strength
        lam   : MSE loss weight
    """

    def __init__(self, alpha: float = 0.15, lam: float = 0.4):
        super().__init__()
        self.lam        = lam
        self.ord_smooth = OrdinalSmoothLoss(alpha=alpha)

    def _mse(self, logit, target_f):
        C      = logit.size(1)
        grades = torch.arange(C, dtype=torch.float32, device=logit.device)
        expect = (F.softmax(logit, dim=1) * grades).sum(dim=1)
        return F.mse_loss(expect, target_f)

    def forward(self, logit, target):
        return self.ord_smooth(logit, target) + self.lam * self._mse(logit, target.float())

    def forward_soft(self, logit, soft_target):
        """CutMix support: soft_target is (B, C) float one-hot blend."""
        ce_loss  = -(soft_target * F.log_softmax(logit, dim=1)).sum(dim=1).mean()
        C        = logit.size(1)
        grades   = torch.arange(C, dtype=torch.float32, device=logit.device)
        expect   = (F.softmax(logit, dim=1) * grades).sum(dim=1)
        tgt_exp  = (soft_target * grades).sum(dim=1)
        return ce_loss + self.lam * F.mse_loss(expect, tgt_exp)


class CBFocalLoss(nn.Module):
    """Class-balanced focal loss."""

    def __init__(self, weight=None, gamma: float = 2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, logit, target):
        ce = F.cross_entropy(logit, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def build_cb_weights(dataset, task: str, device) -> torch.Tensor:
    """Compute class-balanced weights using effective number of samples."""
    from collections import defaultdict
    from config import NUM_CLASSES

    counter = defaultdict(int)
    if hasattr(dataset, "label_cache"):
        for lm in dataset.label_cache:
            counter[lm[task]] += 1
    else:
        for sample in dataset:
            counter[sample["labels"][task].item()] += 1

    n_cls  = NUM_CLASSES[task]
    counts = np.array([counter.get(c, 1) for c in range(n_cls)], dtype=np.float32)
    beta   = 0.999
    w      = (1 - beta) / (1 - np.power(beta, counts))
    w      = w / w.sum() * n_cls
    return torch.tensor(w, dtype=torch.float32, device=device)
