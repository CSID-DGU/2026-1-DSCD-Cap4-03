import os
import random
import logging
import time
from collections import defaultdict
from datetime import datetime, timezone, timedelta

import numpy as np
import torch
import matplotlib.pyplot as plt

KST = timezone(timedelta(hours=9))


# ──────────────────────────────────────────────
# 기본
# ──────────────────────────────────────────────

def now_kst() -> datetime:
    return datetime.now(KST)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_ckpt(model: torch.nn.Module, path: str):
    torch.save(model.state_dict(), path)


def get_device(gpu_id: int = 0) -> torch.device:
    return torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# 클래스 불균형
# ──────────────────────────────────────────────

def compute_class_freq(dataset, task: str = "acne") -> dict:
    # label_cache 가 있으면 이미지 I/O 없이 바로 집계
    counter = defaultdict(int)
    if hasattr(dataset, "label_cache"):
        for label_map in dataset.label_cache:
            counter[label_map[task]] += 1
    else:
        for sample in dataset:
            counter[sample["labels"][task].item()] += 1
    return counter


def compute_cb_weights(freq_dict: dict, beta: float = 0.999) -> torch.Tensor:
    classes = sorted(freq_dict.keys())
    counts  = np.array([freq_dict[c] for c in classes], dtype=np.float32)
    weights = (1 - beta) / (1 - np.power(beta, counts))
    weights = weights / weights.sum() * len(classes)
    return torch.tensor(weights, dtype=torch.float32)


# ──────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────

def plot_metrics(history: dict, save_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    # ── Loss ──────────────────────────────────
    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"],   label="val")
    plt.legend()
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "loss.png"))
    plt.close()

    # ── RMSE ──────────────────────────────────
    plt.figure()
    plt.plot(epochs, history["train_rmse"], label="train")
    plt.plot(epochs, history["val_rmse"],   label="val")
    plt.legend()
    plt.title("RMSE")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "rmse.png"))
    plt.close()

    # ── ACC ───────────────────────────────────
    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"],   label="val")
    plt.legend()
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "acc.png"))
    plt.close()


# ──────────────────────────────────────────────
# Logger
# ──────────────────────────────────────────────

class _KSTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        ct = datetime.fromtimestamp(record.created, tz=KST)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime("%Y-%m-%d %H:%M:%S")


def build_logger(log_dir: str, name: str = "train", run_id: str = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if logger.handlers:
        return logger

    ensure_dir(log_dir)
    suffix   = f"_{run_id}" if run_id else ""
    log_file = os.path.join(log_dir, f"{name}{suffix}.log")

    formatter = _KSTFormatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    sh = logging.StreamHandler()
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


class Logger:
    def __init__(self, save_dir: str, run_id: str = None):
        run_id       = run_id or now_kst().strftime("%y%m%d%H")
        self._logger = build_logger(save_dir, name="train", run_id=run_id)

    def info(self, msg: str):
        self._logger.info(msg)


# ──────────────────────────────────────────────
# StopWatch
# ──────────────────────────────────────────────

class StopWatch:
    def __init__(self):
        self._times:   dict  = defaultdict(float)
        self._current: str   = None
        self._t0:      float = None

    def start(self, name: str):
        self._current = name
        self._t0      = time.perf_counter()

    def stop(self):
        if self._current and self._t0 is not None:
            self._times[self._current] += time.perf_counter() - self._t0
            self._current = None

    def section(self, name: str):
        return _Section(self, name)

    def reset(self):
        self._times.clear()

    def report(self, logger, prefix: str = ""):
        total = sum(self._times.values())
        lines = []
        for name, t in self._times.items():
            pct = (t / total * 100) if total > 0 else 0
            lines.append(f"    {name:<26s}: {t:6.2f}s  ({pct:4.1f}%)")
        logger.info(f"{prefix}Timing breakdown (total={total:.2f}s)")
        for l in lines:
            logger.info(l)


class _Section:
    def __init__(self, sw: StopWatch, name: str):
        self._sw   = sw
        self._name = name

    def __enter__(self):
        self._sw.start(self._name)
        return self

    def __exit__(self, *_):
        self._sw.stop()
