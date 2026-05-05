"""
출력 CSV:
  image, {metric}_pred, {metric}_conf, {metric}_score, {metric}_prob_0 ... prob_N
"""

import os
import argparse
from pathlib import Path

import torch
import pandas as pd
import numpy as np

from config  import DATA, TASK_NAMES, NUM_CLASSES, BASE_DIR
from dataset import build_test_loader
from model   import SkinModel
from utils   import get_device, ensure_dir, build_logger, now_kst

MAP_KEYS = {
    "acne":         ["acne"],
    "dryness":      ["lip_dryness"],
    "sagging":      ["chin_sagging"],
    "pore":         ["l_cheek_pore",         "r_cheek_pore"],
    "pigmentation": ["forehead_pigmentation", "l_cheek_pigmentation", "r_cheek_pigmentation"],
    "wrinkle":      ["forehead_wrinkle",      "glabellus_wrinkle",
                     "l_perocular_wrinkle",   "r_perocular_wrinkle"],
}

METRIC_NUM_CLASSES = {
    metric: NUM_CLASSES[landmarks[0]]
    for metric, landmarks in MAP_KEYS.items()
}


def aggregate_max(probs_dict, landmarks):
    """pred(argmax) 가 가장 높은 landmark 선택. 동점이면 conf 기준."""
    best_pred, best_conf, best_probs = -1, -1.0, None
    for lm in landmarks:
        probs = probs_dict[lm]
        pred  = int(probs.argmax().item())
        conf  = float(probs[pred].item())
        if pred > best_pred or (pred == best_pred and conf > best_conf):
            best_pred, best_conf, best_probs = pred, conf, probs
    return best_pred, best_conf, best_probs


def expected_score(probs):
    """Σ(i / (K-1) × prob_i) — 0~1 연속값."""
    probs = np.asarray(probs, dtype=np.float32)
    K = len(probs)
    if K <= 1:
        return 0.0
    s = np.arange(K, dtype=np.float32) / float(K - 1)
    return float((probs * s).sum())


def parse_args():
    RUN_ID     = now_kst().strftime("%y%m%d_%H")
    RESULT_DIR = os.path.join(BASE_DIR, "result", RUN_ID)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",            required=True)
    p.add_argument("--img_dir",         default=DATA["valid_img"])
    p.add_argument("--out_dir",         default=os.path.join(RESULT_DIR, "test"))
    p.add_argument("--img_size",        type=int, default=224)
    p.add_argument("--local_crop_size", type=int, default=224)
    p.add_argument("--batch_size",      type=int, default=16)
    p.add_argument("--num_workers",     type=int, default=8)
    return p.parse_args()


@torch.no_grad()
def run(args):
    device = get_device()
    now    = now_kst().strftime("%y%m%d_%H")
    ensure_dir(args.out_dir)
    logger = build_logger(args.out_dir, name="test", run_id=now)

    logger.info(f"ckpt            : {args.ckpt}")
    logger.info(f"img_dir         : {args.img_dir}")
    logger.info(f"img_size        : {args.img_size}")
    logger.info(f"local_crop_size : {args.local_crop_size}")
    logger.info(f"device          : {device}")

    # ── Model ──────────────────────────────
    model = SkinModel().to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    model.eval()
    logger.info("model loaded")

    # ── Data ───────────────────────────────
    # TestDataset 이 FACEPART_BBOX 고정 좌표로 local crop 생성
    loader = build_test_loader(
        args.img_dir,
        batch_size      = args.batch_size,
        img_size        = args.img_size,
        local_crop_size = args.local_crop_size,
        num_workers     = args.num_workers,
    )
    logger.info(f"test samples: {len(loader.dataset)}")

    # ── Inference ──────────────────────────
    rows = []
    for imgs, local_crops, paths in loader:
        imgs        = imgs.to(device)
        local_crops = local_crops.to(device)
        out         = model(imgs, local_crops)

        for i in range(imgs.size(0)):
            row = {"image": Path(paths[i]).name}

            probs_dict = {
                t: torch.softmax(out[t][i], dim=0)
                for t in TASK_NAMES
            }

            for metric, landmarks in MAP_KEYS.items():
                pred, conf, probs = aggregate_max(probs_dict, landmarks)
                n_cls = METRIC_NUM_CLASSES[metric]
                score = expected_score(probs.cpu().numpy())

                row[f"{metric}_pred"]  = pred
                row[f"{metric}_conf"]  = round(conf, 6)
                row[f"{metric}_score"] = round(score, 6)
                for c in range(n_cls):
                    row[f"{metric}_prob_{c}"] = round(float(probs[c].item()), 6)

            rows.append(row)

    # ── 컬럼 순서 ──────────────────────────
    cols = ["image"]
    for metric in MAP_KEYS:
        n_cls = METRIC_NUM_CLASSES[metric]
        cols += [f"{metric}_pred", f"{metric}_conf", f"{metric}_score"]
        cols += [f"{metric}_prob_{c}" for c in range(n_cls)]

    df       = pd.DataFrame(rows, columns=cols)
    out_path = os.path.join(args.out_dir, f"predictions_{now}.csv")
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    logger.info(f"columns : {len(df.columns)}")
    logger.info(f"rows    : {len(df)}")
    logger.info(f"[DONE] saved → {out_path}")
    print(df.head(3).to_string())
    return df


if __name__ == "__main__":
    run(parse_args())
