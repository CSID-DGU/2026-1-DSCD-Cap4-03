import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import (
    TASK_NAMES, NUM_CLASSES,
    FACEPART_TO_TASKS, TASK_TO_FACEPART, GRADE_REMAP,
)
from img_crop import (
    resize_pil, to_normalized_tensor,
    local_crops_to_tensor, build_local_crops,
)

DEVICE_MAP = {
    "01": ("digital_camera", 0),
    "02": ("smart_pad",      1),
    "03": ("smart_phone",    2),
}


# ── Augmentation ───────────────────────────────────────────────────────────────

class RandomGaussianNoise:
    """
    Adds Gaussian noise to a PIL Image.
    Simulates sensor noise and low-light shooting conditions.
    """

    def __init__(self, p: float = 0.3, mean: float = 0.0, std: float = 8.0):
        self.p    = p
        self.mean = mean
        self.std  = std

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() >= self.p:
            return img
        arr   = np.array(img, dtype=np.float32)
        noise = np.random.normal(self.mean, self.std, arr.shape).astype(np.float32)
        return Image.fromarray(np.clip(arr + noise, 0, 255).astype(np.uint8))

    def __repr__(self):
        return f"RandomGaussianNoise(p={self.p}, mean={self.mean}, std={self.std})"


# ── JSON Parsing Helpers ────────────────────────────────────────────────────────

def _find_value(obj, keys):
    """Recursively search for any of the given keys in a nested dict/list."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                return v
            out = _find_value(v, keys)
            if out is not None:
                return out
    elif isinstance(obj, list):
        for item in obj:
            out = _find_value(item, keys)
            if out is not None:
                return out
    return None


def _parse_bbox(val):
    if val is None:
        return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        return [int(v) for v in val]
    if isinstance(val, dict):
        if all(k in val for k in ["x1", "y1", "x2", "y2"]):
            return [int(val["x1"]), int(val["y1"]), int(val["x2"]), int(val["y2"])]
        if all(k in val for k in ["x", "y", "w", "h"]):
            x1, y1 = int(val["x"]), int(val["y"])
            return [x1, y1, x1 + int(val["w"]), y1 + int(val["h"])]
    return None


def _extract_bbox(json_obj: dict):
    for key in ["bbox", "b_box", "box", "rect", "rectangle", "face_bbox", "roi"]:
        val  = _find_value(json_obj, [key])
        bbox = _parse_bbox(val)
        if bbox is not None:
            return bbox
    return None


def _acne_count_to_grade(count: int) -> int:
    if count == 0:   return 0
    elif count <= 3: return 1
    elif count <= 7: return 2
    else:            return 3


def _extract_acne(json_obj: dict) -> int:
    ann        = json_obj.get("annotations", {})
    acne_count = ann.get("acne_count", json_obj.get("acne_count", None))
    if acne_count is not None:
        try:
            return _acne_count_to_grade(int(acne_count))
        except Exception:
            pass
    acne = ann.get("acne", None)
    if acne is None:
        return 0
    if isinstance(acne, list):
        return _acne_count_to_grade(len(acne))
    try:
        return _acne_count_to_grade(int(acne))
    except Exception:
        return 0


def extract_task_label(json_obj: dict, task_name: str) -> int:
    if task_name == "acne":
        return _extract_acne(json_obj)

    value = _find_value(
        json_obj,
        [task_name, "grade", "label", "class", "target", "score", "severity"],
    )
    if isinstance(value, dict):
        for subk in ["grade", "label", "class", "score", "severity", task_name]:
            if subk in value:
                value = value[subk]
                break
    try:
        value = int(value)
    except Exception:
        value = 0

    if task_name in GRADE_REMAP:
        value = GRADE_REMAP[task_name].get(value, value)
    return max(0, min(value, NUM_CLASSES[task_name] - 1))


# ── Filename Parsing ───────────────────────────────────────────────────────────

def _is_front_image(image_key: str) -> bool:
    parts = image_key.split("_")
    return len(parts) >= 3 and parts[2].upper() == "F"


def _parse_subject_id(image_key: str) -> str:
    return image_key.split("_")[0]


def _parse_device_info(image_key: str):
    parts = image_key.split("_")
    if len(parts) < 2:
        return "unknown", -1
    return DEVICE_MAP.get(parts[1], ("unknown", -1))


def _facepart_idx_from_json_path(json_path: str):
    stem = Path(json_path).stem
    fp   = stem.split("_")[-1]
    try:
        return int(fp)
    except Exception:
        return None


# ── Dataset ────────────────────────────────────────────────────────────────────

class SkinDataset(Dataset):

    def __init__(self, img_dir, label_dir, train=True, img_size=224, local_crop_size=224):
        self.img_dir         = Path(img_dir)
        self.label_dir       = Path(label_dir)
        self.train           = train
        self.img_size        = img_size
        self.local_crop_size = local_crop_size
        self.aug             = None  # set to transforms.Compose([...]) to enable

        # 1. Build image index (front-view only)
        self.image_map = {}
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
            for p in self.img_dir.rglob(ext):
                key = p.stem
                if _is_front_image(key):
                    self.image_map[key] = str(p)

        # 2. Build JSON index: prefix → {fp_idx: json_path}
        self.json_map = defaultdict(dict)
        for jp in self.label_dir.rglob("*.json"):
            stem   = jp.stem
            parts  = stem.split("_")
            prefix = "_".join(parts[:3])
            fp_idx = _facepart_idx_from_json_path(str(jp))
            if fp_idx is not None:
                self.json_map[prefix][fp_idx] = str(jp)

        # 3. Parse JSONs and build cached samples
        self.samples     = []
        self.label_cache = []
        self.bbox_cache  = []

        for image_key, img_path in self.image_map.items():
            prefix      = "_".join(image_key.split("_")[:3])
            fp_json_map = self.json_map.get(prefix, {})
            if not fp_json_map:
                continue

            label_map, bbox_map = {}, {}
            for fp_idx, json_path in fp_json_map.items():
                with open(json_path, "r", encoding="utf-8") as f:
                    ann = json.load(f)
                bbox_map[fp_idx] = _extract_bbox(ann)
                for task in TASK_NAMES:
                    if TASK_TO_FACEPART[task] == fp_idx:
                        label_map[task] = extract_task_label(ann, task)

            for task in TASK_NAMES:
                label_map.setdefault(task, 0)

            self.samples.append((image_key, img_path))
            self.label_cache.append(label_map)
            self.bbox_cache.append(bbox_map)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_key, img_path = self.samples[idx]
        label_map = self.label_cache[idx]
        bbox_map  = self.bbox_cache[idx]

        img             = Image.open(img_path).convert("RGB")
        local_imgs_raw  = build_local_crops(img, bbox_map, self.local_crop_size)

        if self.aug is not None:
            full_img   = self.aug(resize_pil(img, self.img_size))
            local_imgs = [self.aug(crop) for crop in local_imgs_raw]
        else:
            full_img   = resize_pil(img, self.img_size)
            local_imgs = local_imgs_raw

        device_name, device_id = _parse_device_info(image_key)

        return {
            "full_face":   to_normalized_tensor(full_img),
            "local_crops": local_crops_to_tensor(local_imgs),
            "labels": {
                t: torch.tensor(label_map[t], dtype=torch.long)
                for t in TASK_NAMES
            },
            "image_key":   image_key,
            "image_path":  img_path,
            "subject_id":  _parse_subject_id(image_key),
            "device_id":   torch.tensor(device_id, dtype=torch.long),
            "device_name": device_name,
        }


class TestDataset(Dataset):
    """Inference-only dataset using fixed FACEPART_BBOX coordinates."""

    def __init__(self, img_dir, img_size=224, local_crop_size=224):
        self.imgs = [
            p for p in Path(img_dir).rglob("*.jpg")
            if "checkpoint" not in p.name
        ]
        self.img_size        = img_size
        self.local_crop_size = local_crop_size

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        from config import FACEPART_BBOX
        path = self.imgs[idx]
        img  = Image.open(path).convert("RGB")

        full_img   = resize_pil(img, self.img_size)
        local_imgs = build_local_crops(img, FACEPART_BBOX, self.local_crop_size)

        return (
            to_normalized_tensor(full_img),
            local_crops_to_tensor(local_imgs),
            str(path),
        )


# ── Collate & Loader ───────────────────────────────────────────────────────────

def collate_fn(batch):
    return {
        "full_face":   torch.stack([b["full_face"]   for b in batch]),
        "local_crops": torch.stack([b["local_crops"] for b in batch]),
        "labels": {
            t: torch.stack([b["labels"][t] for b in batch])
            for t in TASK_NAMES
        },
        "device_id":   torch.stack([b["device_id"]  for b in batch]),
        "image_key":   [b["image_key"]   for b in batch],
        "image_path":  [b["image_path"]  for b in batch],
        "subject_id":  [b["subject_id"]  for b in batch],
        "device_name": [b["device_name"] for b in batch],
    }


def build_loader(
    img_dir, label_dir, train=True,
    batch_size=16, img_size=224, local_crop_size=224, num_workers=8,
) -> DataLoader:
    ds = SkinDataset(
        img_dir, label_dir,
        train=train, img_size=img_size, local_crop_size=local_crop_size,
    )
    return DataLoader(
        ds, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True,
        drop_last=train, collate_fn=collate_fn,
    )


def build_test_loader(
    img_dir, batch_size=16, img_size=224, local_crop_size=224, num_workers=8,
) -> DataLoader:
    ds = TestDataset(img_dir, img_size=img_size, local_crop_size=local_crop_size)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
