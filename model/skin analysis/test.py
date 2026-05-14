# test.py 
"""
input.json 형식:
    {
        "image_id":    1,
        "user_id":     123,
        "image_path":  "/path/to/image.jpg",
        "uploaded_at": "2026-05-14 14:00:00"
    }

출력 JSON (raw_{user_id}_{분석시각}.json):
    image_id, user_id, analyzed_at,
    {metric}_grade, {metric}_score, {metric}_prob_0 ... prob_N
"""

import os
import json
import argparse
from datetime import datetime

import torch
import numpy as np
from PIL import Image

from config   import TASK_NAMES, NUM_CLASSES, BASE_DIR, FACEPART_BBOX
from model    import SkinModel
from utils    import ensure_dir, build_logger, now_kst
from img_crop import (
    resize_pil, to_normalized_tensor,
    local_crops_to_tensor, build_local_crops,
)

# ── 지표별 landmark 매핑 ───────────────────────────────────────────────────────
MAP_KEYS = {
    "acne":         ["acne"],
    "dryness":      ["lip_dryness"],
    "sagging":      ["chin_sagging"],
    "pore":         ["l_cheek_pore",          "r_cheek_pore"],
    "pigmentation": ["forehead_pigmentation",  "l_cheek_pigmentation", "r_cheek_pigmentation"],
    "wrinkle":      ["forehead_wrinkle",       "glabellus_wrinkle",
                     "l_perocular_wrinkle",    "r_perocular_wrinkle"],
}

METRIC_NUM_CLASSES = {
    metric: NUM_CLASSES[landmarks[0]]
    for metric, landmarks in MAP_KEYS.items()
}


# ── 헬퍼 함수 ─────────────────────────────────────────────────────────────────

def aggregate_max(probs_dict, landmarks):
    """argmax가 가장 높은 landmark 선택. 동점이면 conf 기준. probs Tensor 반환."""
    best_pred, best_conf, best_probs = -1, -1.0, None
    for lm in landmarks:
        probs = probs_dict[lm]
        pred  = int(probs.argmax().item())
        conf  = float(probs[pred].item())
        if pred > best_pred or (pred == best_pred and conf > best_conf):
            best_pred, best_conf, best_probs = pred, conf, probs
    return best_probs


def expected_grade(probs_np, n_cls):
    """Σ(i × prob_i) 반올림 — 이산확률기댓값 정수."""
    s = np.arange(n_cls, dtype=np.float32)
    return int(round(float((probs_np * s).sum())))


def expected_score(probs_np, n_cls):
    """Σ(i / (K-1) × prob_i) — 0~1 정규화 연속값."""
    if n_cls <= 1:
        return 0.0
    s = np.arange(n_cls, dtype=np.float32) / float(n_cls - 1)
    return float((probs_np * s).sum())


def preprocess(image_path: str, img_size: int, local_crop_size: int, device):
    """
    PIL 이미지 1장 → full_face (1,3,H,W), local_crops (1,N_fp,3,h,w)
    TestDataset.__getitem__과 동일한 전처리
    """
    img = Image.open(image_path).convert("RGB")

    full_img    = resize_pil(img, img_size)
    local_imgs  = build_local_crops(img, FACEPART_BBOX, local_crop_size)

    full_tensor   = to_normalized_tensor(full_img).unsqueeze(0).to(device)
    local_tensors = local_crops_to_tensor(local_imgs).unsqueeze(0).to(device)

    return full_tensor, local_tensors


# ── 인자 파싱 ─────────────────────────────────────────────────────────────────

def parse_args():
    RUN_ID  = now_kst().strftime("%y%m%d_%H")
    OUT_DIR = os.path.join(BASE_DIR, "results", RUN_ID)
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt",            required=True, help="체크포인트 경로")
    p.add_argument("--input",           required=True, help="input.json 경로")
    p.add_argument("--out_dir",         default=OUT_DIR)
    p.add_argument("--img_size",        type=int, default=224)
    p.add_argument("--local_crop_size", type=int, default=224)
    p.add_argument("--gpu",             type=int, default=0)
    return p.parse_args()


# ── 메인 추론 ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def run(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    ensure_dir(args.out_dir)
    now    = now_kst().strftime("%y%m%d_%H%M")
    logger = build_logger(args.out_dir, name="test", run_id=now)

    # ── input.json 읽기 ────────────────────────────────────────────────────────
    with open(args.input, "r", encoding="utf-8") as f:
        record = json.load(f)

    image_id   = record["image_id"]
    user_id    = record["user_id"]
    image_path = record["image_path"]
    # 서버 배포 시 아래 두 줄로 교체:
    # import requests; from io import BytesIO
    # image = Image.open(BytesIO(requests.get(record["image_url"]).content)).convert("RGB")

    logger.info(f"ckpt       : {args.ckpt}")
    logger.info(f"image_id   : {image_id}")
    logger.info(f"user_id    : {user_id}")
    logger.info(f"image_path : {image_path}")
    logger.info(f"device     : {device}")

    # ── 모델 로드 ──────────────────────────────────────────────────────────────
    model = SkinModel(
        freeze_backbone=False,
        use_checkpoint=False,
    ).to(device)
    model.load_state_dict(
        torch.load(args.ckpt, map_location=device, weights_only=False),
        strict=False,  # decoder.norm 키 불일치 무시
    )
    model.eval()
    logger.info("model loaded")

    # ── 전처리 ────────────────────────────────────────────────────────────────
    full_face, local_crops = preprocess(
        image_path, args.img_size, args.local_crop_size, device
    )
    logger.info(f"full_face   : {tuple(full_face.shape)}")
    logger.info(f"local_crops : {tuple(local_crops.shape)}")

    # ── 추론 ──────────────────────────────────────────────────────────────────
    out = model(full_face, local_crops)

    probs_dict = {
        t: torch.softmax(out[t][0], dim=0).cpu()
        for t in TASK_NAMES
    }

    # ── 결과 구성 ─────────────────────────────────────────────────────────────
    analyzed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    file_id     = f"{user_id}_{now}"   # 예: 123_260514_142414

    raw = {
        "image_id":    image_id,
        "user_id":     user_id,
        "analyzed_at": analyzed_at,
    }

    for metric, landmarks in MAP_KEYS.items():
        probs    = aggregate_max(probs_dict, landmarks)
        n_cls    = METRIC_NUM_CLASSES[metric]
        probs_np = probs.numpy()

        raw[f"{metric}_grade"] = expected_grade(probs_np, n_cls)
        raw[f"{metric}_score"] = round(expected_score(probs_np, n_cls), 6)
        for c in range(n_cls):
            raw[f"{metric}_prob_{c}"] = round(float(probs[c].item()), 6)

    # ── JSON 저장 ─────────────────────────────────────────────────────────────
    raw_path = os.path.join(args.out_dir, f"{file_id}.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw, f, ensure_ascii=False, indent=2)

    logger.info(f"saved → {raw_path}")
    print(json.dumps(raw, indent=2, ensure_ascii=False))

    return raw


if __name__ == "__main__":
    run(parse_args())
