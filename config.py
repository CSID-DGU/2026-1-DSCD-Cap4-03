import os
from datetime import datetime, timezone, timedelta

KST        = timezone(timedelta(hours=9))
BASE_DIR   = "/home/donghyun2/decs_jupyter_lab/skin_model2"
RUN_ID     = datetime.now(KST).strftime("%y%m%d_%H")
RESULT_DIR = os.path.join(BASE_DIR, "result", RUN_ID)

DATA = {
    "train_img":   "/home/donghyun2/TS",
    "train_label": "/home/donghyun2/TL",
    "valid_img":   "/home/donghyun2/VS",
    "valid_label": "/home/donghyun2/VL",
}

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
    "forehead_wrinkle":      6,  # 7 → 6: cls6 → cls5로 병합
    "glabellus_wrinkle":     6,  # 7 → 6: cls6 → cls5로 병합
    "l_perocular_wrinkle":   6,  # 7 → 6: cls6 → cls5로 병합
    "r_perocular_wrinkle":   6,  # 7 → 6: cls6 → cls5로 병합
    "l_cheek_pore":          5,
    "r_cheek_pore":          5,
    "lip_dryness":           5,
    "chin_sagging":          6,
}

ORDINAL_TASKS = set(TASK_NAMES) - {"acne"}

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

GRADE_REMAP = {
    # wrinkle 4개: cls6 → cls5 병합 (7클래스 → 6클래스)
    "forehead_wrinkle":      {6: 5},
    "glabellus_wrinkle":     {6: 5},
    "l_perocular_wrinkle":   {6: 5},
    "r_perocular_wrinkle":   {6: 5},
    # 기존 remap 유지
    "chin_sagging":          {6: 5},
    "l_cheek_pore":          {5: 4},
    "r_cheek_pore":          {5: 4},
}

# test 시 고정 bbox. fp=0(acne)은 None → 전체 이미지 사용
FACEPART_BBOX = {
    0: None,
    1: (91, 75, 309, 165),
    2: (140, 188, 246, 263),
    3: (337, 233, 386, 300),
    4: (21, 233, 63, 300),
    5: (35, 300, 140, 413),
    6: (246, 300, 351, 413),
    7: (140, 435, 260, 495),
    8: (175, 488, 246, 540),
}
