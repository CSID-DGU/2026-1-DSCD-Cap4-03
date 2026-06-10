"""
test.py

input.json format:
    {
        "image_id":    1,
        "user_id":     123,
        "image_path":  "/path/to/image.jpg",
        "uploaded_at": "2026-05-14 14:00:00"
    }

Output JSON (saved to --out_dir):
    image_id, user_id, analyzed_at,
    {metric}_grade, {metric}_score, {metric}_prob_0 ... prob_N
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from PIL import Image

from config import TASK_NAMES, NUM_CLASSES, BASE_DIR, FACEPART_BBOX
from model import SkinModel
from utils import ensure_dir, build_logger, now_kst
from img_crop import (
    resize_pil, to_normalized_tensor,
    local_crops_to_tensor, build_local_crops,
)

# ── Metric → Landmark Mapping ─────────────────────────────────────────────────
MAP_KEYS = {
    "acne":         ["acne"],
    "dryness":      ["lip_dryness"],
    "sagging":      ["chin_sagging"],
    "pore":         ["l_cheek_pore",           "r_cheek_pore"],
    "pigmentation": ["forehead_pigmentation",   "l_cheek_pigmentation", "r_cheek_pigmentation"],
    "wrinkle":      ["forehead_wrinkle",        "glabellus_wrinkle",
                     "l_perocular_wrinkle",     "r_perocular_wrinkle"],
}

METRIC_NUM_CLASSES = {
    metric: NUM_CLASSES[landmarks[0]]
    for metric, landmarks in MAP_KEYS.items()
}


# ── Aggregation ────────────────────────────────────────────────────────────────
def aggregate_max(probs_dict: dict, landmarks: list) -> torch.Tensor:
    """Select the landmark with the highest predicted grade (break ties by confidence)."""
    best_pred, best_conf, best_probs = -1, -1.0, None
    for lm in landmarks:
        probs = probs_dict[lm]
        pred  = int(probs.argmax().item())
        conf  = float(probs[pred].item())
        if pred > best_pred or (pred == best_pred and conf > best_conf):
            best_pred, best_conf, best_probs = pred, conf, probs
    return best_probs


def expected_grade(probs_np: np.ndarray, n_cls: int) -> int:
    """Rounded expected value: sum(i * prob_i)."""
    return int(round(float((probs_np * np.arange(n_cls, dtype=np.float32)).sum())))


def expected_score(probs_np: np.ndarray, n_cls: int) -> float:
    """Normalised expected value in [0, 1]: sum(i / (K - 1) * prob_i)."""
    if n_cls <= 1:
        return 0.0
    grades = np.arange(n_cls, dtype=np.float32) / float(n_cls - 1)
    return float((probs_np * grades).sum())


# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(image_path: str, img_size: int, local_crop_size: int, device):
    img           = Image.open(image_path).convert("RGB")
    full_tensor   = to_normalized_tensor(resize_pil(img, img_size)).unsqueeze(0).to(device)
    local_tensors = local_crops_to_tensor(
        build_local_crops(img, FACEPART_BBOX, local_crop_size)
    ).unsqueeze(0).to(device)
    return full_tensor, local_tensors


# ── CLI ────────────────────────────────────────────────────────────────────────
def parse_args():
    run_id  = now_kst().strftime("%y%m%d_%H")
    out_dir = os.path.join(BASE_DIR, "results", run_id)
    p = argparse.ArgumentParser(description="Skin model single-image inference")
    p.add_argument("--ckpt",            required=True,   help="Checkpoint path (.pt)")
    p.add_argument("--input",           required=True,   help="input.json path")
    p.add_argument("--out_dir",         default=out_dir, help="Output directory")
    p.add_argument("--img_size",        type=int, default=224)
    p.add_argument("--local_crop_size", type=int, default=224)
    p.add_argument("--gpu",             type=int, default=0)
    return p.parse_args()


# ── Inference ──────────────────────────────────────────────────────────────────
@torch.no_grad()
def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)

    now    = now_kst().strftime("%y%m%d_%H%M")
    logger = build_logger(args.out_dir, name="test", run_id=now)

    # Load input
    with open(args.input, "r", encoding="utf-8") as f:
        record = json.load(f)

    image_id   = record["image_id"]
    user_id    = record["user_id"]
    image_path = record["image_path"]

    logger.info(f"ckpt       : {args.ckpt}")
    logger.info(f"image_id   : {image_id}")
    logger.info(f"user_id    : {user_id}")
    logger.info(f"image_path : {image_path}")
    logger.info(f"device     : {device}")

    # Load model
    model = SkinModel(freeze_backbone=False, use_checkpoint=False).to(device)
    model.load_state_dict(
        torch.load(args.ckpt, map_location=device, weights_only=False),
        strict=False,
    )
    model.eval()
    logger.info("model loaded")

    # Preprocess & infer
    full_face, local_crops = preprocess(
        image_path, args.img_size, args.local_crop_size, device
    )
    out = model(full_face, local_crops)

    probs_dict = {
        t: torch.softmax(out[t][0], dim=0).cpu()
        for t in TASK_NAMES
    }

    # Build output
    result = {
        "image_id":    image_id,
        "user_id":     user_id,
        "analyzed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
    }
    for metric, landmarks in MAP_KEYS.items():
        probs    = aggregate_max(probs_dict, landmarks)
        n_cls    = METRIC_NUM_CLASSES[metric]
        probs_np = probs.numpy()

        result[f"{metric}_grade"] = expected_grade(probs_np, n_cls)
        result[f"{metric}_score"] = round(expected_score(probs_np, n_cls), 6)
        for c in range(n_cls):
            result[f"{metric}_prob_{c}"] = round(float(probs[c].item()), 6)

    # Save
    out_path = os.path.join(args.out_dir, f"{user_id}_{now}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    logger.info(f"saved -> {out_path}")
    logger.info(json.dumps(result, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    run(parse_args())
