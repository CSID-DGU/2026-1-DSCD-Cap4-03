import torch
import torch.nn as nn
import timm
from torch.utils.checkpoint import checkpoint

from config import TASK_NAMES, NUM_CLASSES, TASK_TO_FACEPART


class SwinStageExtractor(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True):
        super().__init__()
        base = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.patch_embed  = base.patch_embed
        self.layers       = base.layers
        self.norm         = base.norm
        stage2_dim        = base.layers[2].dim_out if hasattr(base.layers[2], "dim_out") else 384
        self.norm2        = nn.LayerNorm(stage2_dim)
        self.num_features = base.num_features  # 768
        self.stage2_dim   = stage2_dim          # 384

    def forward(self, x, use_checkpoint=False):
        def _ckpt(layer, inp):
            return checkpoint(layer, inp, use_reentrant=False)

        run = _ckpt if use_checkpoint else lambda layer, inp: layer(inp)

        x  = self.patch_embed(x)
        x  = run(self.layers[0], x)
        x  = run(self.layers[1], x)
        x  = run(self.layers[2], x)
        f2 = self.norm2(x)
        if f2.dim() == 4:
            f2 = f2.flatten(1, 2)
        x  = run(self.layers[3], x)
        f3 = self.norm(x)
        if f3.dim() == 4:
            f3 = f3.flatten(1, 2)
        return f2, f3   # (B, N2, 384), (B, N3, 768)


class TSA(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=0.1, batch_first=True)

    def forward(self, x):
        y, _ = self.attn(x, x, x)
        return x + y


class CrossAttn(nn.Module):
    def __init__(self, token_dim, feat_dim, heads=8):
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
        kv = self.proj_kv(self.norm_kv(kv))
        q  = self.norm_q(q)
        y, _ = self.attn(q, kv, kv)
        return q + y


class TaskDecoder(nn.Module):
    def __init__(self, token_dim, stage2_dim, stage3_dim):
        super().__init__()
        self.tsa = TSA(token_dim)
        self.ca3 = CrossAttn(token_dim, stage3_dim)
        self.ca2 = CrossAttn(token_dim, stage2_dim)

    def forward(self, token, feat2, feat3):
        t = self.tsa(token)
        t = self.ca3(t, feat3)
        t = self.ca2(t, feat2)
        return t  # (B, 1, token_dim)


class TaskTokenBank(nn.Module):
    def __init__(self, tasks, dim):
        super().__init__()
        self.tasks  = tasks
        self.tokens = nn.ParameterDict({
            t: nn.Parameter(torch.randn(1, 1, dim) * 0.02)
            for t in tasks
        })

    def get(self, task, B):
        return self.tokens[task].expand(B, -1, -1)  # (B, 1, dim)


class SkinModel(nn.Module):

    def __init__(self, model_name="swin_tiny_patch4_window7_224",
                 dropout=0.3, freeze_backbone=False, use_checkpoint=False):
        super().__init__()
        token_dim = 768
        self.use_checkpoint = use_checkpoint  # gradient checkpointing 여부

        self.backbone        = SwinStageExtractor(model_name, pretrained=True)
        self.freeze_backbone = freeze_backbone
        stage2_dim           = self.backbone.stage2_dim
        stage3_dim           = self.backbone.num_features

        self.token_bank    = TaskTokenBank(TASK_NAMES, token_dim)
        self.token_dropout = nn.Dropout(p=0.1)
        self.alpha         = nn.Parameter(torch.zeros(len(TASK_NAMES)))

        self.decoder = TaskDecoder(token_dim, stage2_dim, stage3_dim)

        hidden_dim = 512
        self.heads = nn.ModuleDict({
            t: nn.Sequential(
                nn.Linear(token_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, NUM_CLASSES[t]),
            )
            for t in TASK_NAMES
        })

        self.task_fp = [TASK_TO_FACEPART[t] for t in TASK_NAMES]

        # freeze_backbone=True면 파라미터 고정
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def _encode_all(self, full_face, local_crops):
        """
        freeze_backbone=True  → no_grad (gradient 완전 차단, 메모리 최소)
        freeze_backbone=False → gradient 흐름 유지
          + use_checkpoint=True → 각 stage activation 버리고 backward 시 재계산
                                  메모리 40~50% 절감, 속도 20~30% 감소
          + use_checkpoint=False → 일반 forward (메모리 많이 필요)
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

    def forward(self, full_face, local_crops):
        B = full_face.size(0)

        g2, g3, l2_list, l3_list = self._encode_all(full_face, local_crops)

        logits = {}
        for i, t in enumerate(TASK_NAMES):
            fp = self.task_fp[i]

            w  = torch.sigmoid(self.alpha[i]).view(1, 1, 1)
            t2 = g2 * (1 - w) + l2_list[fp] * w        # (B, N2, 384)
            t3 = g3 * (1 - w) + l3_list[fp] * w        # (B, N3, 768)

            token  = self.token_bank.get(t, B)          # (B, 1, 768)
            token  = self.token_dropout(token)
            token  = self.decoder(token, t2, t3)        # (B, 1, 768)
            logits[t] = self.heads[t](token[:, 0])      # (B, NUM_CLASSES[t])

        return logits
