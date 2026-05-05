import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.tensor(arr, dtype=torch.float32)


def normalize_tensor(x: torch.Tensor) -> torch.Tensor:
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def resize_pil(img: Image.Image, size: int) -> Image.Image:
    return img.resize((size, size), Image.BILINEAR)


def _parse_bbox(val) -> list:
    if val is None:
        return None
    if isinstance(val, float) and np.isnan(val):
        return None
    if isinstance(val, str):
        try:
            val = json.loads(val)
        except Exception:
            return None
    if isinstance(val, (list, tuple)) and len(val) == 4:
        return [int(v) for v in val]
    if isinstance(val, dict):
        if all(k in val for k in ["x1", "y1", "x2", "y2"]):
            return [int(val["x1"]), int(val["y1"]),
                    int(val["x2"]), int(val["y2"])]
        if all(k in val for k in ["x", "y", "w", "h"]):
            x1, y1 = int(val["x"]), int(val["y"])
            return [x1, y1, x1 + int(val["w"]), y1 + int(val["h"])]
    return None


def safe_crop(img: Image.Image, bbox) -> Image.Image:
    """bbox=None 이면 전체 이미지 반환."""
    bbox = _parse_bbox(bbox)
    if bbox is None:
        return img
    img_w, img_h = img.size
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), img_w - 1))
    y1 = max(0, min(int(y1), img_h - 1))
    x2 = max(x1 + 1, min(int(x2), img_w))
    y2 = max(y1 + 1, min(int(y2), img_h))
    return img.crop((x1, y1, x2, y2))


def build_local_crops(img: Image.Image,
                      bbox_map: dict,
                      local_crop_size: int) -> list:
    """
    face-part 0~8 각각 crop → resize.
    bbox_map[fp] = None 이면 전체 이미지 사용.
    """
    crops = []
    for fp in range(9):
        bbox = bbox_map.get(fp, None)
        crop = safe_crop(img, bbox)
        crop = resize_pil(crop, local_crop_size)
        crops.append(crop)
    return crops


def to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    return normalize_tensor(pil_to_tensor(img))


def local_crops_to_tensor(local_imgs: list) -> torch.Tensor:
    return torch.stack([to_normalized_tensor(im) for im in local_imgs], dim=0)
