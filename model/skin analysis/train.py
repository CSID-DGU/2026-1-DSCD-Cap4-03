"""
train.py — Multi-task skin condition model training script.

Usage:
    python train.py
"""

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
from loss import OrdinalMSELoss, build_cb_weights
from metrics import compute_all_metrics
from utils import (
    set_seed, ensure_dir, save_ckpt, now_kst,
    Logger, StopWatch, plot_metrics,
)

warnings.filterwarnings("ignore")

# ── Training Config ────────────────────────────────────────────────────────────
TRAIN_CFG = {
    # Data
    "batch_size":       32,
    "img_size":         224,
    "local_crop_size":  224,
    "num_workers":      8,
    # Optimizer
    "lr":               1e-5,
    "lr_backbone":      1e-6,
    "weight_decay":     1e-2,
    "use_layerwise_lr": True,
    "lr_decay":         0.75,
    # Regularization
    "dropout":          0.2,
    "label_smoothing":  0.1,
    "cutmix_p":         0.5,
    "cutmix_alpha":     1.0,
    # Training
    "epochs":           100,
    "accum_steps":      2,
    "seed":             42,
    "gpu_id":           0,
    # Backbone
    "freeze_backbone":  False,
    "use_checkpoint":   True,
    # Scheduler
    "cosine_T0":        10,
    "cosine_T_mult":    2,
    "rlrop_min_lr":     1e-7,
    # Early stopping
    "patience":         20,
    # Per-task loss weights
    "task_weights": {
        "glabellus_wrinkle":   1.5,
        "forehead_wrinkle":    1.2,
        "lip_dryness":         1.2,
        "l_perocular_wrinkle": 1.5,
        "r_perocular_wrinkle": 1.5,
    },
}

RUN_ID      = now_kst().strftime("%y%m%d_%H")
RESULT_DIR  = os.path.join(BASE_DIR, "result", RUN_ID)
TASK_LOSS_W = {t: TRAIN_CFG["task_weights"].get(t, 1.0) for t in TASK_NAMES}


# ── Optimizer ──────────────────────────────────────────────────────────────────
def build_layerwise_optimizer(model, cfg):
    """Layer-wise LR: upper backbone layers get higher LR."""
    bb     = model.backbone
    lr_top = cfg["lr_backbone"]
    decay  = cfg["lr_decay"]
    wd     = cfg["weight_decay"]

    backbone_groups = [
        {"params": list(bb.layers[3].parameters()),   "lr": lr_top * (decay ** 0)},
        {"params": list(bb.layers[2].parameters()),   "lr": lr_top * (decay ** 1)},
        {"params": list(bb.layers[1].parameters()),   "lr": lr_top * (decay ** 2)},
        {"params": list(bb.layers[0].parameters()),   "lr": lr_top * (decay ** 3)},
        {"params": list(bb.patch_embed.parameters()), "lr": lr_top * (decay ** 4)},
        {"params": list(bb.norm.parameters()) + list(bb.norm2.parameters()),
         "lr": lr_top * (decay ** 0)},
    ]
    backbone_ids = {id(p) for g in backbone_groups for p in g["params"]}
    non_backbone = [p for p in model.parameters() if id(p) not in backbone_ids]

    all_groups = backbone_groups + [{"params": non_backbone, "lr": cfg["lr"]}]
    for g in all_groups:
        g.setdefault("weight_decay", wd)
    return torch.optim.AdamW(all_groups)


# ── CutMix ─────────────────────────────────────────────────────────────────────
def _rand_bbox(H, W, lam):
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(H * cut_ratio), int(W * cut_ratio)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2


def cutmix_batch(full_face, local_crops, gt, alpha):
    B, _, H, W = full_face.shape
    lam        = np.random.beta(alpha, alpha)
    x1, y1, x2, y2 = _rand_bbox(H, W, lam)
    lam        = 1.0 - (x2 - x1) * (y2 - y1) / (H * W)
    idx        = torch.randperm(B, device=full_face.device)

    mixed_full              = full_face.clone()
    mixed_full[:, :, y1:y2, x1:x2] = full_face[idx, :, y1:y2, x1:x2]
    mixed_local             = local_crops.clone()
    mixed_local[:, :, :, y1:y2, x1:x2] = local_crops[idx, :, :, y1:y2, x1:x2]

    soft_gt = {
        t: lam * F.one_hot(gt[t], NUM_CLASSES[t]).float()
           + (1 - lam) * F.one_hot(gt[t][idx], NUM_CLASSES[t]).float()
        for t in TASK_NAMES
    }
    return mixed_full, mixed_local, soft_gt


# ── Loss Helpers ───────────────────────────────────────────────────────────────
def _task_loss(t, logit, label, loss_ord, loss_cb, cb_loss_map):
    if t == "acne":
        return loss_cb(logit, label)
    if t in cb_loss_map:
        ce     = cb_loss_map[t](logit, label)
        grades = torch.arange(logit.size(1), dtype=torch.float32, device=logit.device)
        expect = (F.softmax(logit, dim=1) * grades).sum(dim=1)
        return ce + 0.4 * F.mse_loss(expect, label.float())
    return loss_ord(logit, label)


def _task_loss_soft(t, logit, soft_target, loss_ord, loss_cb, cb_loss_map):
    if t == "acne":
        return -(soft_target * F.log_softmax(logit, dim=-1)).sum(dim=-1).mean()
    if t in cb_loss_map:
        ce     = -(soft_target * F.log_softmax(logit, dim=-1)).sum(dim=-1).mean()
        grades = torch.arange(logit.size(1), dtype=torch.float32, device=logit.device)
        expect  = (F.softmax(logit, dim=1) * grades).sum(dim=1)
        tgt_exp = (soft_target * grades).sum(dim=1)
        return ce + 0.4 * F.mse_loss(expect, tgt_exp)
    if hasattr(loss_ord, "forward_soft"):
        return loss_ord.forward_soft(logit, soft_target)
    return -(soft_target * F.log_softmax(logit, dim=-1)).sum(dim=-1).mean()


def _weighted_loss(loss_per_task: dict) -> torch.Tensor:
    return sum(TASK_LOSS_W[t] * v for t, v in loss_per_task.items())


# ── Early Stopping ─────────────────────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience: int = 5):
        self.best     = float("inf")
        self.patience = patience
        self.count    = 0

    def step(self, val: float) -> bool:
        if val < self.best:
            self.best, self.count = val, 0
            return False
        self.count += 1
        return self.count >= self.patience


# ── Logging Helpers ────────────────────────────────────────────────────────────
def _log_config(logger, device):
    logger.info("=" * 60)
    logger.info("  TRAIN CONFIG")
    logger.info("=" * 60)
    for k, v in TRAIN_CFG.items():
        logger.info(f"  {k:20s}: {v}")
    logger.info(f"  {'device':20s}: {device}")
    logger.info(f"  {'run_id':20s}: {RUN_ID}")
    logger.info(f"  {'result_dir':20s}: {RESULT_DIR}")
    logger.info("=" * 60)


def _log_data(logger, train_loader, val_loader):
    logger.info("=" * 60)
    logger.info("  DATA INFO")
    logger.info("=" * 60)
    logger.info(f"  train: {len(train_loader.dataset):,} samples / {len(train_loader)} batches")
    logger.info(f"  valid: {len(val_loader.dataset):,} samples / {len(val_loader)} batches")
    logger.info(f"  batch_size : {TRAIN_CFG['batch_size']}")
    logger.info(f"  img_size   : {TRAIN_CFG['img_size']}")
    logger.info("=" * 60)


def _log_model(logger, model):
    logger.info("=" * 60)
    logger.info("  MODEL INFO")
    logger.info("=" * 60)
    total     = sum(p.numel() for p in model.parameters()) / 1e6
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    logger.info(f"  total params     : {total:.3f} M")
    logger.info(f"  trainable params : {trainable:.3f} M")
    logger.info(f"  freeze_backbone  : {TRAIN_CFG['freeze_backbone']}")
    logger.info("=" * 60)


def _log_lr(opt, logger):
    labels = [
        "backbone.stage3", "backbone.stage2", "backbone.stage1",
        "backbone.stage0", "backbone.patch_embed", "backbone.norms",
        "head/decoder/token",
    ]
    logger.info("  [Layer-wise LR]")
    for label, g in zip(labels, opt.param_groups):
        n = sum(p.numel() for p in g["params"]) / 1e6
        logger.info(f"    {label:28s}  lr={g['lr']:.2e}  params={n:.3f}M")


def _log_class_dist(logger, loader, split):
    task_count = {t: defaultdict(int) for t in TASK_NAMES}
    ds = loader.dataset
    for lm in (ds.label_cache if hasattr(ds, "label_cache") else
               [s["labels"] for s in ds]):
        for t in TASK_NAMES:
            task_count[t][lm[t] if isinstance(lm[t], int) else int(lm[t].item())] += 1

    n_total = len(ds)
    max_cls = max(NUM_CLASSES.values())
    logger.info(f"\n  [{split.upper()} CLASS DISTRIBUTION]  (n={n_total:,})")
    logger.info(f"  {'fp':>3}  {'TASK':<28}  " +
                "  ".join(f"cls{i}" for i in range(max_cls)) + "   total")
    logger.info("  " + "-" * (80 + (max_cls - 6) * 7))

    for fp in sorted(FACEPART_TO_TASKS):
        for i, t in enumerate(FACEPART_TO_TASKS[fp]):
            n_cls     = NUM_CLASSES[t]
            counts    = [task_count[t].get(c, 0) for c in range(max_cls)]
            total     = sum(counts[:n_cls])
            count_str = "  ".join(
                f"{counts[c]:5d}" if c < n_cls else f"{'':>5}"
                for c in range(max_cls)
            )
            fp_str = f"{fp:>3}" if i == 0 else "   "
            flag   = "  !!!" if total != n_total else ""
            logger.info(f"  {fp_str}  {t:<28}  {count_str}   {total:,}{flag}")


# ── Train / Val Loops ──────────────────────────────────────────────────────────
def train_epoch(model, loader, opt, loss_ord, loss_cb, cb_loss_map, device, sw):
    model.train()
    if TRAIN_CFG["freeze_backbone"]:
        model.backbone.eval()

    use_cutmix  = TRAIN_CFG["cutmix_p"] > 0
    accum_steps = TRAIN_CFG["accum_steps"]
    total       = 0.0
    all_t = {t: [] for t in TASK_NAMES}
    all_p = {t: [] for t in TASK_NAMES}

    opt.zero_grad()
    for step, batch in enumerate(loader):
        with sw.section("train/data→gpu"):
            full_face   = batch["full_face"].to(device)
            local_crops = batch["local_crops"].to(device)
            gt          = {t: batch["labels"][t].to(device) for t in TASK_NAMES}

        apply_cutmix = use_cutmix and (np.random.rand() < TRAIN_CFG["cutmix_p"])
        if apply_cutmix:
            full_face, local_crops, soft_gt = cutmix_batch(
                full_face, local_crops, gt, TRAIN_CFG["cutmix_alpha"]
            )

        with sw.section("train/forward"):
            out = model(full_face, local_crops)

        with sw.section("train/loss"):
            if apply_cutmix:
                loss = _weighted_loss({
                    t: _task_loss_soft(t, out[t], soft_gt[t], loss_ord, loss_cb, cb_loss_map)
                    for t in TASK_NAMES
                })
            else:
                loss = _weighted_loss({
                    t: _task_loss(t, out[t], gt[t], loss_ord, loss_cb, cb_loss_map)
                    for t in TASK_NAMES
                })
            (loss / accum_steps).backward()

        is_last = (step + 1 == len(loader))
        if (step + 1) % accum_steps == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            opt.zero_grad()

        total += loss.item()
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
                t: _task_loss(t, out[t], gt[t], loss_ord, loss_cb, cb_loss_map)
                for t in TASK_NAMES
            })
            total += loss.item()
            for t in TASK_NAMES:
                all_t[t] += batch["labels"][t].tolist()
                all_p[t] += out[t].argmax(1).cpu().tolist()

    task_m, mean_m = compute_all_metrics(all_t, all_p)
    return total / len(loader), task_m, mean_m


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    set_seed(TRAIN_CFG["seed"])
    device = torch.device(
        f"cuda:{TRAIN_CFG['gpu_id']}" if torch.cuda.is_available() else "cpu"
    )
    ensure_dir(RESULT_DIR)
    logger = Logger(RESULT_DIR, run_id=RUN_ID)
    _log_config(logger, device)

    # ── Loaders ────────────────────────────────────────────────────────────────
    sw_data = StopWatch()
    with sw_data.section("build train loader"):
        train_loader = build_loader(
            DATA["train_img"], DATA["train_label"], train=True,
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
    _log_data(logger, train_loader, val_loader)
    _log_class_dist(logger, train_loader, "train")
    _log_class_dist(logger, val_loader,   "valid")

    # ── Model ──────────────────────────────────────────────────────────────────
    sw_model = StopWatch()
    with sw_model.section("model init + to(device)"):
        model = SkinModel(
            dropout=TRAIN_CFG["dropout"],
            freeze_backbone=TRAIN_CFG["freeze_backbone"],
            use_checkpoint=TRAIN_CFG["use_checkpoint"],
        ).to(device)
    sw_model.report(logger, prefix="[Setup] ")
    _log_model(logger, model)

    # ── Loss ───────────────────────────────────────────────────────────────────
    loss_cb  = torch.nn.CrossEntropyLoss(
        weight=build_cb_weights(train_loader.dataset, "acne", device),
        label_smoothing=TRAIN_CFG["label_smoothing"],
    )
    cb_tasks    = ["lip_dryness", "l_cheek_pore", "r_cheek_pore"]
    cb_loss_map = {
        t: torch.nn.CrossEntropyLoss(
            weight=build_cb_weights(train_loader.dataset, t, device),
            label_smoothing=TRAIN_CFG["label_smoothing"],
        )
        for t in cb_tasks
    }
    loss_ord = OrdinalMSELoss(alpha=0.15, lam=0.4)

    # ── Optimizer & Scheduler ──────────────────────────────────────────────────
    if not TRAIN_CFG["freeze_backbone"] and TRAIN_CFG["use_layerwise_lr"]:
        opt = build_layerwise_optimizer(model, TRAIN_CFG)
        _log_lr(opt, logger)
    else:
        trainable = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(
            trainable, lr=TRAIN_CFG["lr"], weight_decay=TRAIN_CFG["weight_decay"]
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt,
        T_0    = TRAIN_CFG["cosine_T0"],
        T_mult = TRAIN_CFG["cosine_T_mult"],
        eta_min= TRAIN_CFG["rlrop_min_lr"],
    )

    # ── Training Loop ──────────────────────────────────────────────────────────
    stopper     = EarlyStopping(patience=TRAIN_CFG["patience"])
    history     = {
        "train_loss": [], "val_loss": [],
        "train_rmse": [], "val_rmse": [],
        "train_ad_acc":    [], "val_ad_acc":    [],
        "train_exact_acc": [], "val_exact_acc": [],
    }
    best_mae    = float("inf")
    best_epoch  = 0
    best_task_m = {}
    ckpt_path   = os.path.join(RESULT_DIR, f"best_{RUN_ID}.pt")

    logger.info("\n" + "=" * 60)
    logger.info("  START TRAINING")
    logger.info("=" * 60)

    for epoch in range(1, TRAIN_CFG["epochs"] + 1):
        sw = StopWatch()
        t0 = time.perf_counter()

        logger.info(f"[Epoch {epoch:03d}/{TRAIN_CFG['epochs']}] training...")
        tr_loss, tr_mean_m = train_epoch(
            model, train_loader, opt, loss_ord, loss_cb, cb_loss_map, device, sw
        )
        logger.info(f"[Epoch {epoch:03d}/{TRAIN_CFG['epochs']}] validating...")
        val_loss, task_m, mean_m = val_epoch(
            model, val_loader, loss_ord, loss_cb, cb_loss_map, device, sw
        )
        elapsed = time.perf_counter() - t0

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["train_rmse"].append(tr_mean_m["rmse"])
        history["val_rmse"].append(mean_m["rmse"])
        history["train_ad_acc"].append(tr_mean_m["ad_acc"])
        history["val_ad_acc"].append(mean_m["ad_acc"])
        history["train_exact_acc"].append(tr_mean_m["exact_acc"])
        history["val_exact_acc"].append(mean_m["exact_acc"])

        logger.info(
            f"[Epoch {epoch:03d}/{TRAIN_CFG['epochs']}]  {elapsed:.1f}s  "
            f"train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}  "
            f"acc={mean_m['exact_acc']:.4f}  ad_acc={mean_m['ad_acc']:.4f}  "
            f"mae={mean_m['mae']:.4f}  rmse={mean_m['rmse']:.4f}  qwk={mean_m['qwk']:.4f}"
        )
        sw.report(logger, prefix=f"  [Epoch {epoch:03d}] ")

        logger.info(
            f"  {'TASK':30s}  {'acc':>6}  {'ad_acc':>6}  {'mae':>6}  {'rmse':>6}  {'qwk':>6}"
        )
        logger.info(f"  {'-' * 72}")
        for t in TASK_NAMES:
            m = task_m[t]
            logger.info(
                f"  {t:30s}  "
                f"{m['exact_acc']:6.4f}  {m['ad_acc']:6.4f}  "
                f"{m['mae']:6.4f}  {m['rmse']:6.4f}  {m['qwk']:6.4f}"
            )
        logger.info(
            f"  {'[MEAN]':30s}  "
            f"{mean_m['exact_acc']:6.4f}  {mean_m['ad_acc']:6.4f}  "
            f"{mean_m['mae']:6.4f}  {mean_m['rmse']:6.4f}  {mean_m['qwk']:6.4f}"
        )

        if mean_m["mae"] < best_mae:
            best_mae, best_epoch, best_task_m = mean_m["mae"], epoch, task_m
            save_ckpt(model, ckpt_path)
            logger.info(
                f"  → best saved  (epoch={best_epoch}  mae={best_mae:.4f}  "
                f"acc={mean_m['exact_acc']:.4f})"
            )

        if stopper.step(val_loss):
            logger.info("Early stopping triggered.")
            break

        scheduler.step(epoch)
        logger.info(f"  lr: {opt.param_groups[0]['lr']:.2e}")
        plot_metrics(history, RESULT_DIR)

    logger.info("=" * 60)
    logger.info(f"  DONE  best_epoch={best_epoch}  best_mae={best_mae:.4f}")
    logger.info(f"  ckpt={ckpt_path}")
    logger.info("=" * 60)

    if best_task_m:
        logger.info(f"\n  [BEST EPOCH {best_epoch} — TASK METRICS]")
        logger.info(
            f"  {'TASK':30s}  {'acc':>6}  {'ad_acc':>6}  {'mae':>6}  {'rmse':>6}  {'qwk':>6}"
        )
        logger.info(f"  {'-' * 72}")
        for t in TASK_NAMES:
            m = best_task_m[t]
            logger.info(
                f"  {t:30s}  "
                f"{m['exact_acc']:6.4f}  {m['ad_acc']:6.4f}  "
                f"{m['mae']:6.4f}  {m['rmse']:6.4f}  {m['qwk']:6.4f}"
            )
        accs    = [best_task_m[t]["exact_acc"] for t in TASK_NAMES]
        ad_accs = [best_task_m[t]["ad_acc"]    for t in TASK_NAMES]
        maes    = [best_task_m[t]["mae"]        for t in TASK_NAMES]
        rmses   = [best_task_m[t]["rmse"]       for t in TASK_NAMES]
        qwks    = [best_task_m[t]["qwk"]        for t in TASK_NAMES]
        logger.info(
            f"  {'[MEAN]':30s}  "
            f"{np.mean(accs):6.4f}  {np.mean(ad_accs):6.4f}  "
            f"{np.mean(maes):6.4f}  {np.mean(rmses):6.4f}  {np.mean(qwks):6.4f}"
        )
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
