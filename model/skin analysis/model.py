import torch
import torch.nn as nn
import timm
from torch.utils.checkpoint import checkpoint

from config import TASK_NAMES, NUM_CLASSES, TASK_TO_FACEPART


# ── Backbone ───────────────────────────────────────────────────────────────────

class SwinStageExtractor(nn.Module):
    """
    Swin Transformer backbone that returns intermediate features from
    stage2 (384-d) and stage3 (768-d) for multi-scale attention.
    """

    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        base = timm.create_model(model_name, pretrained=pretrained, num_classes=0)

        self.patch_embed  = base.patch_embed
        self.layers       = base.layers
        self.norm         = base.norm
        self.norm2        = nn.LayerNorm(
            base.layers[2].dim_out if hasattr(base.layers[2], "dim_out") else 384
        )
        self.num_features = base.num_features  # 768
        self.stage2_dim   = self.norm2.normalized_shape[0]  # 384

    def forward(self, x, use_checkpoint: bool = False):
        def _run(layer, inp):
            if use_checkpoint:
                return checkpoint(layer, inp, use_reentrant=False)
            return layer(inp)

        x  = self.patch_embed(x)
        x  = _run(self.layers[0], x)
        x  = _run(self.layers[1], x)
        x  = _run(self.layers[2], x)
        f2 = self.norm2(x)
        if f2.dim() == 4:
            f2 = f2.flatten(1, 2)
        x  = _run(self.layers[3], x)
        f3 = self.norm(x)
        if f3.dim() == 4:
            f3 = f3.flatten(1, 2)
        return f2, f3  # (B, N2, 384), (B, N3, 768)


# ── Attention Modules ──────────────────────────────────────────────────────────

class TSA(nn.Module):
    """Token Self-Attention."""

    def __init__(self, d: int, heads: int = 8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        return x + y


class CrossAttn(nn.Module):
    """Cross-Attention from task token (query) to feature map (key/value)."""

    def __init__(self, token_dim: int, feat_dim: int, heads: int = 8):
        super().__init__()
        self.norm_q  = nn.LayerNorm(token_dim)
        self.norm_kv = nn.LayerNorm(feat_dim)
        self.proj_kv = nn.Linear(feat_dim, token_dim) if feat_dim != token_dim else nn.Identity()
        self.attn    = nn.MultiheadAttention(token_dim, heads, dropout=0.1, batch_first=True)

    def forward(self, q, kv):
        if kv.dim() == 4:
            kv = kv.view(kv.size(0), -1, kv.size(-1))
        elif kv.dim() == 2:
            kv = kv.unsqueeze(1)
        kv   = self.proj_kv(self.norm_kv(kv))
        y, _ = self.attn(self.norm_q(q), kv, kv)
        return q + y


# ── Decoder ────────────────────────────────────────────────────────────────────

class TaskDecoder(nn.Module):
    """TSA → CrossAttn(stage3) → CrossAttn(stage2)."""

    def __init__(self, token_dim: int, stage2_dim: int, stage3_dim: int):
        super().__init__()
        self.tsa = TSA(token_dim)
        self.ca3 = CrossAttn(token_dim, stage3_dim)
        self.ca2 = CrossAttn(token_dim, stage2_dim)

    def forward(self, token, feat2, feat3):
        t = self.tsa(token)
        t = self.ca3(t, feat3)
        t = self.ca2(t, feat2)
        return t  # (B, 1, token_dim)


# ── Task Token Bank ────────────────────────────────────────────────────────────

class TaskTokenBank(nn.Module):
    """Learnable per-task query tokens."""

    def __init__(self, tasks: list, dim: int):
        super().__init__()
        self.tasks  = tasks
        self.tokens = nn.ParameterDict({
            t: nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            for t in tasks
        })

    def get(self, task: str, B: int) -> torch.Tensor:
        return self.tokens[task].expand(B, -1, -1)  # (B, 1, dim)


# ── Main Model ─────────────────────────────────────────────────────────────────

class SkinModel(nn.Module):
    """
    Multi-task skin condition classification model.

    Architecture:
        SwinStageExtractor → TaskTokenBank + TaskDecoder → per-task heads

    Args:
        model_name      : timm Swin model name
        dropout         : dropout rate in classification heads
        freeze_backbone : freeze backbone weights (no gradient)
        use_checkpoint  : gradient checkpointing (saves ~40% memory, ~20% slower)
    """

    def __init__(
        self,
        model_name: str  = "swin_tiny_patch4_window7_224",
        dropout: float   = 0.3,
        freeze_backbone: bool = False,
        use_checkpoint: bool  = False,
    ):
        super().__init__()
        token_dim = 768
        self.use_checkpoint = use_checkpoint

        self.backbone      = SwinStageExtractor(model_name, pretrained=True)
        self.freeze_backbone = freeze_backbone
        stage2_dim         = self.backbone.stage2_dim   # 384
        stage3_dim         = self.backbone.num_features  # 768

        self.token_bank    = TaskTokenBank(TASK_NAMES, token_dim)
        self.token_dropout = nn.Dropout(p=0.1)
        self.alpha         = nn.Parameter(torch.zeros(len(TASK_NAMES)))

        self.decoder = TaskDecoder(token_dim, stage2_dim, stage3_dim)

        self.heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(token_dim, 512),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(512, NUM_CLASSES[t]),
            )
            for t in TASK_NAMES
        })

        self.task_fp = [TASK_TO_FACEPART[t] for t in TASK_NAMES]

        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _encode_all(self, full_face, local_crops):
        """
        Encode full face and all local crops.
        - freeze_backbone=True  : torch.no_grad, no memory overhead
        - use_checkpoint=True   : gradient checkpointing per stage
        """
        ckpt = self.use_checkpoint and not self.freeze_backbone

        if self.freeze_backbone:
            with torch.no_grad():
                g2, g3 = self.backbone(full_face, use_checkpoint=False)
                l2_list, l3_list = [], []
                for fp in range(local_crops.size(1)):
                    f2, f3 = self.backbone(local_crops[:, fp], use_checkpoint=False)
                    l2_list.append(f2)
                    l3_list.append(f3)
        else:
            g2, g3 = self.backbone(full_face, use_checkpoint=ckpt)
            l2_list, l3_list = [], []
            for fp in range(local_crops.size(1)):
                f2, f3 = self.backbone(local_crops[:, fp], use_checkpoint=ckpt)
                l2_list.append(f2)
                l3_list.append(f3)

        return g2, g3, l2_list, l3_list

    def forward(self, full_face, local_crops) -> dict:
        B = full_face.size(0)
        g2, g3, l2_list, l3_list = self._encode_all(full_face, local_crops)

        logits = {}
        for i, t in enumerate(TASK_NAMES):
            fp = self.task_fp[i]
            w  = torch.sigmoid(self.alpha[i]).view(1, 1, 1)

            # Weighted blend of global and local features
            feat2 = g2 * (1 - w) + l2_list[fp] * w  # (B, N2, 384)
            feat3 = g3 * (1 - w) + l3_list[fp] * w  # (B, N3, 768)

            token      = self.token_dropout(self.token_bank.get(t, B))
            token      = self.decoder(token, feat2, feat3)   # (B, 1, 768)
            logits[t]  = self.heads[t](token[:, 0])          # (B, C)

        return logits
