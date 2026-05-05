import torch
import torch.nn as nn
import torch.nn.functional as F


class OrdinalCELoss(nn.Module):
    """일반 label smoothing CE """
    def __init__(self, label_smoothing: float = 0.1):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    def forward(self, logit, target):
        return self.ce(logit, target)


class OrdinalSmoothLoss(nn.Module):
    """
    인접 클래스에 smoothing을 집중시키는 ordinal loss.
    """
    def __init__(self, alpha: float = 0.15):
        super().__init__()
        self.alpha = alpha

    def forward(self, logit, target):
        B, C = logit.shape
        soft = torch.zeros_like(logit)
        soft.scatter_(1, target.unsqueeze(1), 1.0)
        half = self.alpha / 2.0
        prev_idx  = (target - 1).clamp(min=0)
        next_idx  = (target + 1).clamp(max=C - 1)
        prev_mask = (target > 0).float().unsqueeze(1)
        next_mask = (target < C - 1).float().unsqueeze(1)
        soft_label = soft * (1.0 - self.alpha)
        soft_label.scatter_add_(1, prev_idx.unsqueeze(1), prev_mask * half)
        soft_label.scatter_add_(1, next_idx.unsqueeze(1), next_mask * half)
        boundary = (~(target > 0) | ~(target < C - 1)).float()
        soft_label[boundary.bool(), :] = soft[boundary.bool(), :] * (1.0 - self.alpha / 2.0)
        soft_label = soft_label / soft_label.sum(dim=1, keepdim=True).clamp(min=1e-8)
        log_prob = F.log_softmax(logit, dim=1)
        return -(soft_label * log_prob).sum(dim=1).mean()


class OrdinalMSELoss(nn.Module):
    """
    CE(ordinal smooth) + λ * MSE(예측 기댓값, 정답) 혼합 loss.
    
    E[pred] = Σ_i (i * softmax(logit)[i])  ← 연속 기댓값
    MSE target = float(target class index)

    alpha: ordinal smoothing 강도 (인접 클래스 smoothing)
    lam  : MSE loss 가중치
    """
    def __init__(self, alpha: float = 0.15, lam: float = 0.4):
        super().__init__()
        self.alpha     = alpha
        self.lam       = lam
        self.ord_smooth = OrdinalSmoothLoss(alpha=alpha)

    def forward(self, logit, target):
        # ── CE 부분 (ordinal smooth label) ──────────────────────────────────
        ce_loss = self.ord_smooth(logit, target)

        # ── MSE 부분 (기댓값 vs 정답) ────────────────────────────────────────
        C      = logit.size(1)
        probs  = F.softmax(logit, dim=1)                         # (B, C)
        grades = torch.arange(C, dtype=torch.float32,
                              device=logit.device)               # [0,1,...,C-1]
        expect = (probs * grades).sum(dim=1)                     # (B,)
        target_f = target.float()                                # (B,)
        mse_loss = F.mse_loss(expect, target_f)

        return ce_loss + self.lam * mse_loss

    def forward_soft(self, logit, soft_target):
        """
        CutMix용 soft label 지원.
        soft_target: (B, C) FloatTensor (one-hot blend)
        """
        # CE 부분
        log_prob = F.log_softmax(logit, dim=1)
        ce_loss  = -(soft_target * log_prob).sum(dim=1).mean()

        # MSE 부분 — soft target의 기댓값을 정답으로 사용
        C      = logit.size(1)
        probs  = F.softmax(logit, dim=1)
        grades = torch.arange(C, dtype=torch.float32, device=logit.device)
        expect      = (probs       * grades).sum(dim=1)  # 예측 기댓값
        target_exp  = (soft_target * grades).sum(dim=1)  # soft label 기댓값
        mse_loss = F.mse_loss(expect, target_exp)

        return ce_loss + self.lam * mse_loss


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
    from collections import defaultdict
    from config import NUM_CLASSES
    import numpy as np

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
    return torch.tensor(w, dtype=torch.float32).to(device)
