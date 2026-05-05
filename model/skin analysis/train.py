import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from config import TASK_NAMES, NUM_CLASSES, DATA, BASE_DIR, FACEPART_TO_TASKS
from dataset import build_loader
from model import SkinModel
from loss import OrdinalCELoss, OrdinalMSELoss, CBFocalLoss, build_cb_weights
from metrics import compute_all_metrics
from utils import (
    set_seed, ensure_dir, save_ckpt,
    compute_class_freq, compute_cb_weights,
    plot_metrics, Logger, StopWatch, now_kst,
)

warnings.filterwarnings("ignore")

TRAIN_CFG = {
    "batch_size":       32,   
    "lr":               1e-5,
    "lr_backbone":      1e-6,
    "weight_decay":     1e-2,
    "dropout":          0.2,
    "label_smoothing":  0.1,
    "epochs":           100,
    "img_size":         224,
    "local_crop_size":  224,
    "seed":             42,
    "num_workers":      8,
    "gpu_id":           0,
    "patience":         20,
    # ReduceLROnPlateau
    "rlrop_factor":     0.5,
    "rlrop_patience":   5,
    "rlrop_min_lr":     1e-7,
    # backbone freeze
    "freeze_backbone":  False,
    # layer-wise LR decay (freeze_backbone=False 일 때만 적용)
    "use_layerwise_lr": True,       # False로 바꾸면 단일 lr로 실험
    "lr_decay":         0.75,       # stage 내려갈수록 lr *= decay
                                    # stage3: 1e-6 / stage2: 7.5e-7
                                    # stage1: 5.6e-7 / stage0: 4.2e-7
                                    # patch_embed: 3.2e-7
    # CutMix
    "cutmix_p":         0.5,    # 배치당 적용 확률
    "cutmix_alpha":     1.0,    # Beta(alpha, alpha) — 1.0이면 uniform mix ratio
    # gradient accumulation: effective batch = batch_size × accum_steps

    "accum_steps":      2,
    "use_checkpoint":   True,    # gradient checkpointing (unfreeze 시 메모리 절감)
    # task별 loss weight — 1.0이 기본, 못하는 task 상향
    "task_weights": {
        "glabellus_wrinkle":    1.5,
        "forehead_wrinkle":     1.2,    # acc 0.548, wrinkle 중 가장 나은 편이지만 추가
        "lip_dryness":          1.2,
        "l_perocular_wrinkle":  1.5,
        "r_perocular_wrinkle":  1.5,
    },
}

RUN_ID     = now_kst().strftime("%y%m%d_%H")
RESULT_DIR = os.path.join(BASE_DIR, "result", RUN_ID)


# ── Layer-wise LR decay optimizer ───────────────────────────────────────────
def build_layerwise_optimizer(model, cfg):
    """
    Swin backbone 각 stage에 lr_backbone * (lr_decay ^ depth) 적용.
      depth 0 = stage3 (최상위, lr 가장 높음)
      depth 4 = patch_embed (최하위, lr 가장 낮음)
    decoder / token_bank / alpha / heads 는 cfg["lr"] 사용.

    ReduceLROnPlateau이 모든 param_group에 동일 factor를 곱하므로
    layer간 lr 비율은 scheduler 적용 후에도 자동으로 유지됨.
    """
    lr_top  = cfg["lr_backbone"]
    decay   = cfg["lr_decay"]
    lr_head = cfg["lr"]
    wd      = cfg["weight_decay"]
    bb      = model.backbone

    backbone_groups = [
        {"params": list(bb.layers[3].parameters()),   "lr": lr_top * (decay ** 0)},
        {"params": list(bb.layers[2].parameters()),   "lr": lr_top * (decay ** 1)},
        {"params": list(bb.layers[1].parameters()),   "lr": lr_top * (decay ** 2)},
        {"params": list(bb.layers[0].parameters()),   "lr": lr_top * (decay ** 3)},
        {"params": list(bb.patch_embed.parameters()), "lr": lr_top * (decay ** 4)},
        {"params": list(bb.norm.parameters()) + list(bb.norm2.parameters()),
         "lr": lr_top * (decay ** 0)},
    ]
    backbone_param_ids = {id(p) for g in backbone_groups for p in g["params"]}
    non_backbone = [p for p in model.parameters() if id(p) not in backbone_param_ids]
    all_groups = backbone_groups + [{"params": non_backbone, "lr": lr_head}]
    for g in all_groups:
        g.setdefault("weight_decay", wd)
    return torch.optim.AdamW(all_groups)


def _lr_summary(opt, logger):
    labels = [
        "backbone.stage3", "backbone.stage2", "backbone.stage1",
        "backbone.stage0", "backbone.patch_embed", "backbone.norms",
        "head/decoder/token",
    ]
    logger.info("  [Layer-wise LR]")
    for label, g in zip(labels, opt.param_groups):
        n = sum(p.numel() for p in g["params"]) / 1e6
        logger.info(f"    {label:28s}  lr={g['lr']:.2e}  params={n:.3f}M")


# ── CutMix ──────────────────────────────────────────────────────────────────
def rand_bbox(H, W, lam):
    """CutMix 박스 좌표 반환. lam은 원본 이미지 유지 비율."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h = int(H * cut_ratio)
    cut_w = int(W * cut_ratio)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(full_face, local_crops, gt, alpha, num_classes):
    """
    Args:
        full_face   : (B, 3, H, W)
        local_crops : (B, N, 3, H, W)
        gt          : dict[task] = (B,) LongTensor — hard label
        alpha       : Beta distribution 파라미터
        num_classes : dict[task] = int

    Returns:
        mixed_full, mixed_local, soft_gt
        soft_gt: dict[task] = (B, num_classes[task]) FloatTensor
    """
    B, _, H, W = full_face.shape
    lam = np.random.beta(alpha, alpha)
    x1, y1, x2, y2 = rand_bbox(H, W, lam)

    # 실제 박스 면적 기준으로 lam 재계산 (정확한 soft label을 위해)
    lam = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)

    # 셔플 인덱스 (같은 배치 내에서 pair)
    idx = torch.randperm(B, device=full_face.device)

    # full_face mix
    mixed_full = full_face.clone()
    mixed_full[:, :, y1:y2, x1:x2] = full_face[idx, :, y1:y2, x1:x2]

    # local_crops mix (N개 전체에 동일 박스 적용)
    mixed_local = local_crops.clone()
    mixed_local[:, :, :, y1:y2, x1:x2] = local_crops[idx, :, :, y1:y2, x1:x2]

    # hard label → soft label (one-hot blend)
    soft_gt = {}
    for t in TASK_NAMES:
        nc = num_classes[t]
        # one-hot
        oh_a = F.one_hot(gt[t],          num_classes=nc).float()  # (B, nc)
        oh_b = F.one_hot(gt[t][idx],     num_classes=nc).float()  # (B, nc)
        soft_gt[t] = lam * oh_a + (1.0 - lam) * oh_b             # (B, nc)

    return mixed_full, mixed_local, soft_gt, lam


def soft_ce_loss(logits, soft_target, task, loss_ord, loss_cb, cb_loss_map):
    """
    soft label용 CE (CutMix 전용):
      acne        → CB weight CE (soft label 직접 지원)
      cb_loss_map → CB weight CE + MSE (soft label 기댓값)
      나머지      → OrdinalMSELoss.forward_soft
    """
    if task == "acne":
        return loss_cb(logits, soft_target)
    elif task in cb_loss_map:
        log_prob = F.log_softmax(logits, dim=-1)
        ce  = -(soft_target * log_prob).sum(dim=-1).mean()
        C   = logits.size(1)
        probs  = torch.softmax(logits, dim=1)
        grades = torch.arange(C, dtype=torch.float32, device=logits.device)
        expect     = (probs       * grades).sum(dim=1)
        target_exp = (soft_target * grades).sum(dim=1)
        mse = F.mse_loss(expect, target_exp)
        return ce + 0.4 * mse
    else:
        if hasattr(loss_ord, "forward_soft"):
            return loss_ord.forward_soft(logits, soft_target)
        else:
            log_prob = F.log_softmax(logits, dim=-1)
            return -(soft_target * log_prob).sum(dim=-1).mean()


class EarlyStopping:
    def __init__(self, patience=5):
        self.best     = float("inf")
        self.patience = patience
        self.count    = 0

    def step(self, val):
        if val < self.best:
            self.best  = val
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def print_config(logger, device):
    logger.info("=" * 60)
    logger.info("  TRAIN CONFIG")
    logger.info("=" * 60)
    for k, v in TRAIN_CFG.items():
        logger.info(f"  {k:20s}: {v}")
    logger.info(f"  {'device':20s}: {device}")
    logger.info(f"  {'run_id':20s}: {RUN_ID}")
    logger.info(f"  {'result_dir':20s}: {RESULT_DIR}")
    logger.info("=" * 60)


def print_data_info(logger, train_loader, val_loader):
    n_train = len(train_loader.dataset)
    logger.info("=" * 60)
    logger.info("  DATA INFO")
    logger.info("=" * 60)
    logger.info(f"  train samples : {n_train:,}  ({len(train_loader)} batches)")
    logger.info(f"  valid samples : {len(val_loader.dataset):,}  ({len(val_loader)} batches, no aug)")
    logger.info(f"  batch_size    : {TRAIN_CFG['batch_size']}")
    logger.info(f"  img_size      : {TRAIN_CFG['img_size']}")
    logger.info(f"  local_size    : {TRAIN_CFG['local_crop_size']}")
    logger.info(f"  augmentation  : HorizontalFlip(p=0.5) + VerticalFlip(p=0.1) + Rotation(±10°)"
                f" + CutMix(p={TRAIN_CFG['cutmix_p']}, alpha={TRAIN_CFG['cutmix_alpha']})")
    logger.info(f"  aug 방식      : 원본 bbox crop 후 full/local 각각 독립 aug")
    logger.info("=" * 60)


def print_model_info(logger, model):
    logger.info("=" * 60)
    logger.info("  MODEL INFO")
    logger.info("=" * 60)
    logger.info(f"  backbone   : {model.backbone.__class__.__name__}")
    logger.info(f"  stage2_dim : {model.backbone.stage2_dim}")
    logger.info(f"  stage3_dim : {model.backbone.num_features}")
    logger.info(f"  freeze_backbone : {TRAIN_CFG['freeze_backbone']}")

    def count_all(m):   return sum(p.numel() for p in m.parameters()) / 1e6
    def count_train(m): return sum(p.numel() for p in m.parameters() if p.requires_grad) / 1e6

    logger.info(f"  params total    : {count_all(model):.3f} M")
    logger.info(f"  params trainable: {count_train(model):.3f} M")
    logger.info(f"    backbone    : {count_all(model.backbone):.3f} M  "
                f"({'frozen' if TRAIN_CFG['freeze_backbone'] else 'trainable'})")
    logger.info(f"    token_bank  : {count_all(model.token_bank):.3f} M")
    if hasattr(model, "decoders"):
        logger.info(f"    decoders    : {count_all(model.decoders):.3f} M  (task-specific × {len(model.decoders)})")
    else:
        logger.info(f"    decoder     : {count_all(model.decoder):.3f} M  (shared)")
    logger.info(f"    heads       : {count_all(model.heads):.3f} M")
    logger.info("\n  HEAD OUTPUT DIMS")
    for t, h in model.heads.items():
        linear = h[-1] if isinstance(h, torch.nn.Sequential) else h
        logger.info(f"    {t:30s}: {linear.in_features} → {linear.out_features}")
    logger.info("=" * 60)


def print_class_distribution(logger, loader, split):
    task_count = {t: defaultdict(int) for t in TASK_NAMES}
    ds = loader.dataset
    if hasattr(ds, "label_cache"):
        for label_map in ds.label_cache:
            for t in TASK_NAMES:
                task_count[t][label_map[t]] += 1
    else:
        for sample in ds:
            for t in TASK_NAMES:
                task_count[t][int(sample["labels"][t].item())] += 1

    n_total = len(ds)
    max_cls = max(NUM_CLASSES.values())
    logger.info(f"\n  [{split.upper()} CLASS DISTRIBUTION]  (전체 샘플 수 = {n_total:,})")
    logger.info(f"  {'fp':>3}  {'TASK':<28}  " + "  ".join(f"cls{i}" for i in range(max_cls)) + "   total")
    logger.info("  " + "-" * (80 + (max_cls - 6) * 7))

    for fp in sorted(FACEPART_TO_TASKS.keys()):
        tasks = FACEPART_TO_TASKS[fp]
        for i, t in enumerate(tasks):
            n_cls     = NUM_CLASSES[t]
            counts    = [task_count[t].get(c, 0) for c in range(max_cls)]
            total     = sum(counts[:n_cls])
            count_str = "  ".join(f"{counts[c]:5d}" if c < n_cls else f"{'':>5}" for c in range(max_cls))
            fp_str    = f"{fp:>3}" if i == 0 else "   "
            flag      = "  !!!" if total != n_total else ""
            logger.info(f"  {fp_str}  {t:<28}  {count_str}   {total:,}{flag}")
        if len(tasks) > 1:
            logger.info("  " + " " * 34 + "-" * (46 + (max_cls - 6) * 7))


def _get_task_loss(t, logit, label, loss_ord, loss_cb, cb_loss_map):
    """task별로 적절한 loss 함수 선택."""
    if t == "acne":
        return loss_cb(logit, label)
    elif t in cb_loss_map:
        # cls 편중 task: CB weight CE + MSE
        ce = cb_loss_map[t](logit, label)
        C  = logit.size(1)
        probs  = torch.softmax(logit, dim=1)
        grades = torch.arange(C, dtype=torch.float32, device=logit.device)
        expect = (probs * grades).sum(dim=1)
        mse    = torch.nn.functional.mse_loss(expect, label.float())
        return ce + 0.4 * mse
    else:
        return loss_ord(logit, label)


# task별 loss weight 전처리 — dict에 없으면 1.0
TASK_LOSS_W = {t: TRAIN_CFG["task_weights"].get(t, 1.0) for t in TASK_NAMES}


def _weighted_loss(loss_per_task: dict) -> torch.Tensor:
    """task별 loss에 TASK_LOSS_W 가중치를 곱해 합산."""
    return sum(TASK_LOSS_W[t] * v for t, v in loss_per_task.items())


def train_epoch(model, loader, opt, loss_ord, loss_cb, cb_loss_map, device, sw):
    model.train()
    if TRAIN_CFG["freeze_backbone"]:
        model.backbone.eval()

    use_cutmix  = TRAIN_CFG["cutmix_p"] > 0
    accum_steps = TRAIN_CFG.get("accum_steps", 1)
    total = 0.0
    all_t = {t: [] for t in TASK_NAMES}
    all_p = {t: [] for t in TASK_NAMES}

    opt.zero_grad()   # accum 시작 전 초기화

    for step, batch in enumerate(loader):
        with sw.section("train/data→gpu"):
            full_face   = batch["full_face"].to(device)
            local_crops = batch["local_crops"].to(device)
            gt          = {t: batch["labels"][t].to(device) for t in TASK_NAMES}

        # ── CutMix (p=cutmix_p 확률로 적용) ────────────────────────────────
        apply_cutmix = use_cutmix and (np.random.rand() < TRAIN_CFG["cutmix_p"])
        if apply_cutmix:
            full_face, local_crops, soft_gt, _ = cutmix_batch(
                full_face, local_crops, gt,
                alpha=TRAIN_CFG["cutmix_alpha"],
                num_classes=NUM_CLASSES,
            )

        with sw.section("train/forward"):
            out = model(full_face, local_crops)

        with sw.section("train/loss"):
            if apply_cutmix:
                loss = _weighted_loss({
                    t: soft_ce_loss(out[t], soft_gt[t], t, loss_ord, loss_cb, cb_loss_map)
                    for t in TASK_NAMES
                })
            else:
                loss = _weighted_loss({
                    t: _get_task_loss(t, out[t], gt[t], loss_ord, loss_cb, cb_loss_map)
                    for t in TASK_NAMES
                })
            # accum_steps로 나눠서 gradient 스케일 유지
            loss = loss / accum_steps

        with sw.section("train/backward"):
            loss.backward()

        # accum_steps마다 또는 마지막 배치에서 optimizer step
        is_last = (step + 1 == len(loader))
        if (step + 1) % accum_steps == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        total += loss.item() * accum_steps   # 원래 스케일로 복원해서 기록
        with torch.no_grad():
            for t in TASK_NAMES:
                all_t[t] += batch["labels"][t].tolist()
                all_p[t] += out[t].argmax(1).cpu().tolist()

    _, mean_m = compute_all_metrics(all_t, all_p)
    return total / len(loader), mean_m


@torch.no_grad()
def val_epoch(model, loader, loss_ord, loss_cb, cb_loss_map, device, sw):
    model.eval()
    total = 0.0
    all_t = {t: [] for t in TASK_NAMES}
    all_p = {t: [] for t in TASK_NAMES}

    for batch in loader:
        with sw.section("val/data→gpu"):
            full_face   = batch["full_face"].to(device)
            local_crops = batch["local_crops"].to(device)
            gt          = {t: batch["labels"][t].to(device) for t in TASK_NAMES}

        with sw.section("val/forward"):
            out = model(full_face, local_crops)

        with sw.section("val/loss+metric"):
            loss = _weighted_loss({
                t: _get_task_loss(t, out[t], gt[t], loss_ord, loss_cb, cb_loss_map)
                for t in TASK_NAMES
            })
            total += loss.item()
            for t in TASK_NAMES:
                all_t[t] += batch["labels"][t].tolist()
                all_p[t] += out[t].argmax(1).cpu().tolist()

    task_m, mean_m = compute_all_metrics(all_t, all_p)
    return total / len(loader), task_m, mean_m


def main():
    set_seed(TRAIN_CFG["seed"])
    device = torch.device(
        f"cuda:{TRAIN_CFG['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )
    ensure_dir(RESULT_DIR)
    logger = Logger(RESULT_DIR, run_id=RUN_ID)
    print_config(logger, device)

    aug_img_dir   = "/home/donghyun2/AUG/images"
    aug_label_dir = "/home/donghyun2/AUG/labels"

    sw_data = StopWatch()
    with sw_data.section("build train loader"):
        train_loader = build_loader(
            aug_img_dir, aug_label_dir, train=True,
            batch_size=TRAIN_CFG["batch_size"], img_size=TRAIN_CFG["img_size"],
            local_crop_size=TRAIN_CFG["local_crop_size"],
            num_workers=TRAIN_CFG["num_workers"],
        )
    with sw_data.section("build val loader"):
        val_loader = build_loader(
            DATA["valid_img"], DATA["valid_label"], train=False,
            batch_size=TRAIN_CFG["batch_size"], img_size=TRAIN_CFG["img_size"],
            local_crop_size=TRAIN_CFG["local_crop_size"],
            num_workers=TRAIN_CFG["num_workers"],
        )
    sw_data.report(logger, prefix="[Setup] ")
    print_data_info(logger, train_loader, val_loader)
    print_class_distribution(logger, train_loader, "train")
    print_class_distribution(logger, val_loader,   "valid")

    sw_model = StopWatch()
    with sw_model.section("model init + to(device)"):
        # freeze_backbone을 model 생성 시 전달
        model = SkinModel(
            dropout=TRAIN_CFG["dropout"],
            freeze_backbone=TRAIN_CFG["freeze_backbone"],
            use_checkpoint=TRAIN_CFG["use_checkpoint"],
        ).to(device)
    sw_model.report(logger, prefix="[Setup] ")

    if TRAIN_CFG["freeze_backbone"]:
        logger.info("  backbone frozen (requires_grad=False, no_grad in _encode_all)")
    else:
        logger.info("  backbone unfrozen (gradient flows through _encode_all)")

    print_model_info(logger, model)

    # acne: CB weight + label smoothing CE
    acne_w  = build_cb_weights(train_loader.dataset, "acne", device)
    loss_cb = torch.nn.CrossEntropyLoss(
        weight=acne_w, label_smoothing=TRAIN_CFG["label_smoothing"]
    )

    # lip_dryness / l_cheek_pore / r_cheek_pore:
    # cls2에 60% 편중 → CB weight로 소수 클래스 학습 강제
    cb_tasks = ["lip_dryness", "l_cheek_pore", "r_cheek_pore"]
    cb_loss_map = {
        t: torch.nn.CrossEntropyLoss(
            weight=build_cb_weights(train_loader.dataset, t, device),
            label_smoothing=TRAIN_CFG["label_smoothing"],
        )
        for t in cb_tasks
    }

    # OrdinalMSELoss: CE(ordinal smooth) + lam*MSE(기댓값)
    # alpha=0.15: 인접 클래스 smoothing 강도
    # lam=0.4:    MSE 가중치 - rmse/qwk 직접 최적화
    loss_ord = OrdinalMSELoss(alpha=0.15, lam=0.4)

    # ── optimizer: layer-wise lr or 단일 lr ────────────────────────────────
    if not TRAIN_CFG["freeze_backbone"] and TRAIN_CFG["use_layerwise_lr"]:
        opt = build_layerwise_optimizer(model, TRAIN_CFG)
        _lr_summary(opt, logger)
    else:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"  trainable params: {sum(p.numel() for p in trainable_params)/1e6:.3f} M")
        opt = torch.optim.AdamW(
            trainable_params,
            lr=TRAIN_CFG["lr"],
            weight_decay=TRAIN_CFG["weight_decay"],
        )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode     = "min",
        factor   = TRAIN_CFG["rlrop_factor"],
        patience = TRAIN_CFG["rlrop_patience"],
        min_lr   = TRAIN_CFG["rlrop_min_lr"],
        verbose  = False,
    )

    stopper       = EarlyStopping(patience=TRAIN_CFG["patience"])
    history       = {"train_loss": [], "val_loss": [],
                     "train_rmse": [], "val_rmse": [],
                     "train_acc":  [], "val_acc":  []}
    best_mae      = float("inf")    # MAE 기준 (실제 평가 지표)
    best_epoch    = 0
    best_task_m   = {}
    ckpt_path     = os.path.join(RESULT_DIR, f"best_{RUN_ID}.pt")

    logger.info("\n" + "=" * 60)
    logger.info("  START TRAINING")
    logger.info("=" * 60)

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        sw = StopWatch()
        t0 = time.perf_counter()

        logger.info(f"[Epoch {epoch:03d}/{TRAIN_CFG['epochs']}] training...")
        tr_loss, tr_mean_m = train_epoch(
            model, train_loader, opt, loss_ord, loss_cb, cb_loss_map, device, sw)

        logger.info(f"[Epoch {epoch:03d}/{TRAIN_CFG['epochs']}] validating...")
        val_loss, task_m, mean_m = val_epoch(
            model, val_loader, loss_ord, loss_cb, cb_loss_map, device, sw)

        elapsed = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(tr_mean_m["rmse"])
        history["val_rmse"].append(mean_m["rmse"])
        history["train_acc"].append(tr_mean_m["acc"])
        history["val_acc"].append(mean_m["acc"])

        logger.info(
            f"[Epoch {epoch:03d}/{TRAIN_CFG['epochs']}] "
            f"time={elapsed:.1f}s  "
            f"TRAIN loss={tr_loss:.4f}  |  "
            f"VALID loss={val_loss:.4f}  "
            f"acc={mean_m['acc']:.4f}  mae={mean_m['mae']:.4f}  "
            f"rmse={mean_m['rmse']:.4f}  qwk={mean_m['qwk']:.4f}"
        )
        sw.report(logger, prefix=f"  [Epoch {epoch:03d}] ")

        logger.info(
            f"  {'TASK':30s}  {'acc':>6}  {'mae':>6}  {'rmse':>6}  {'qwk':>6}  {'w1':>6}"
        )
        logger.info(f"  {'-' * 65}")
        for t in TASK_NAMES:
            m = task_m[t]
            logger.info(
                f"  {t:30s}  "
                f"{m['acc']:6.4f}  {m['mae']:6.4f}  "
                f"{m['rmse']:6.4f}  {m['qwk']:6.4f}  {m['w1']:6.4f}"
            )
        logger.info(
            f"  {'[MEAN]':30s}  "
            f"{mean_m['acc']:6.4f}  {mean_m['mae']:6.4f}  "
            f"{mean_m['rmse']:6.4f}  {mean_m['qwk']:6.4f}  {mean_m['w1']:6.4f}"
        )

        # ── best 체크포인트: val_mae 기준 ──────────────────────────────────
        if mean_m["mae"] < best_mae:
            best_mae = mean_m["mae"]
            best_epoch    = epoch
            best_task_m   = task_m
            save_ckpt(model, ckpt_path)
            logger.info(
                f"  → best model saved  "
                f"(epoch={best_epoch}  mae={best_mae:.4f}  val_loss={val_loss:.4f})"
            )

        # ── early stopping: val_loss 기준 ───────────────────────────────────
        if stopper.step(val_loss):
            logger.info("Early stopping triggered.")
            break

        scheduler.step(val_loss)
        cur_lr = opt.param_groups[0]["lr"]
        logger.info(f"  lr: {cur_lr:.2e}")
        plot_metrics(history, RESULT_DIR)

    logger.info("=" * 60)
    logger.info(f"  DONE  best_epoch={best_epoch}  best_mae={best_mae:.4f}")
    logger.info(f"  ckpt={ckpt_path}")
    logger.info("=" * 60)

    if best_task_m:
        logger.info(f"\n  [BEST EPOCH {best_epoch} — TASK METRICS]")
        logger.info(f"  {'TASK':30s}  {'acc':>6}  {'mae':>6}  {'rmse':>6}  {'qwk':>6}  {'w1':>6}")
        logger.info(f"  {'-' * 65}")
        import numpy as _np
        accs, maes, rmses, qwks, w1s = [], [], [], [], []
        for t in TASK_NAMES:
            m = best_task_m[t]
            accs.append(m['acc']); maes.append(m['mae'])
            rmses.append(m['rmse']); qwks.append(m['qwk']); w1s.append(m['w1'])
            logger.info(
                f"  {t:30s}  "
                f"{m['acc']:6.4f}  {m['mae']:6.4f}  "
                f"{m['rmse']:6.4f}  {m['qwk']:6.4f}  {m['w1']:6.4f}"
            )
        logger.info(
            f"  {'[MEAN]':30s}  "
            f"{_np.mean(accs):6.4f}  {_np.mean(maes):6.4f}  "
            f"{_np.mean(rmses):6.4f}  {_np.mean(qwks):6.4f}  {_np.mean(w1s):6.4f}"
        )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
