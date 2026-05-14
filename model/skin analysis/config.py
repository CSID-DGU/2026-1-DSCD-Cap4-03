import os
from datetime import datetime, timezone, timedelta

# ── Timezone ───────────────────────────────────────────────────────────────────
KST = timezone(timedelta(hours=9))

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = "/home/donghyun2/decs_jupyter_lab/skin_model4"
RESULT_DIR = os.path.join(BASE_DIR, "result")

DATA = {
    "train_img":   "/home/donghyun2/TS",
    "train_label": "/home/donghyun2/TL",
    "valid_img":   "/home/donghyun2/VS",
    "valid_label": "/home/donghyun2/VL",
}

# ── Tasks ──────────────────────────────────────────────────────────────────────
TASK_NAMES = [
    "acne",
    "forehead_pigmentation", "l_cheek_pigmentation", "r_cheek_pigmentation",
    "forehead_wrinkle", "glabellus_wrinkle", "l_perocular_wrinkle", "r_perocular_wrinkle",
    "l_cheek_pore", "r_cheek_pore",
    "lip_dryness",
    "chin_sagging",
]

NUM_CLASSES = {
    "acne":                  4,
    "forehead_pigmentation": 6,
    "l_cheek_pigmentation":  6,
    "r_cheek_pigmentation":  6,
    "forehead_wrinkle":      6,  # cls6 → cls5 merged
    "glabellus_wrinkle":     6,  # cls6 → cls5 merged
    "l_perocular_wrinkle":   6,  # cls6 → cls5 merged
    "r_perocular_wrinkle":   6,  # cls6 → cls5 merged
    "l_cheek_pore":          5,
    "r_cheek_pore":          5,
    "lip_dryness":           5,
    "chin_sagging":          6,
}

ORDINAL_TASKS = set(TASK_NAMES) - {"acne"}

# ── Face Part Mapping ──────────────────────────────────────────────────────────
FACEPART_TO_TASKS = {
    0: ["acne"],
    1: ["forehead_pigmentation", "forehead_wrinkle"],
    2: ["glabellus_wrinkle"],
    3: ["l_perocular_wrinkle"],
    4: ["r_perocular_wrinkle"],
    5: ["l_cheek_pore", "l_cheek_pigmentation"],
    6: ["r_cheek_pore", "r_cheek_pigmentation"],
    7: ["lip_dryness"],
    8: ["chin_sagging"],
}

TASK_TO_FACEPART = {
    task: fp
    for fp, tasks in FACEPART_TO_TASKS.items()
    for task in tasks
}

# ── Grade Remapping ────────────────────────────────────────────────────────────
GRADE_REMAP = {
    "forehead_wrinkle":    {6: 5},
    "glabellus_wrinkle":   {6: 5},
    "l_perocular_wrinkle": {6: 5},
    "r_perocular_wrinkle": {6: 5},
    "chin_sagging":        {6: 5},
    "l_cheek_pore":        {5: 4},
    "r_cheek_pore":        {5: 4},
}

# ── Fixed BBoxes for Inference (fp=0: full image) ─────────────────────────────
FACEPART_BBOX = {
    0: None,
    1: (91,  75,  309, 165),
    2: (140, 188, 246, 263),
    3: (337, 233, 386, 300),
    4: (21,  233, 63,  300),
    5: (35,  300, 140, 413),
    6: (246, 300, 351, 413),
    7: (140, 435, 260, 495),
    8: (175, 488, 246, 540),
}
