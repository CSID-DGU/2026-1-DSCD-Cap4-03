import numpy as np
from sklearn.metrics import cohen_kappa_score, mean_squared_error
from scipy.stats import pearsonr

from config import TASK_NAMES


def _safe(fn, a, b, **kw):
    try:
        return fn(a, b, **kw)
    except Exception:
        return 0.0


def task_metrics(y_true, y_pred) -> dict:
    yt = np.array(y_true)
    yp = np.array(y_pred)
    return {
        "exact_acc": float(np.mean(yt == yp)),
        "ad_acc":    float(np.mean(np.abs(yt - yp) <= 1)),
        "mae":       float(np.mean(np.abs(yt - yp))),
        "rmse":      float(np.sqrt(_safe(mean_squared_error, yt, yp))),
        "qwk":       _safe(cohen_kappa_score, yt, yp, weights="quadratic")
                     if len(np.unique(yt)) > 1 else 0.0,
        "corr":      _safe(lambda a, b: pearsonr(a, b)[0], yt, yp)
                     if len(np.unique(yt)) > 1 else 0.0,
    }


def compute_all_metrics(all_t: dict, all_p: dict) -> tuple[dict, dict]:
    task_m = {t: task_metrics(all_t[t], all_p[t]) for t in TASK_NAMES}
    keys   = task_m[TASK_NAMES[0]].keys()
    mean_m = {k: float(np.mean([task_m[t][k] for t in TASK_NAMES])) for k in keys}
    return task_m, mean_m
