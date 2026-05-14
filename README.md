# Skin Analysis Model

Swin Transformer 기반 멀티태스크 피부 상태 분석 모델  
얼굴 9개 영역(face part)에서 12개 피부 지표를 동시에 분류

---

## 분석 지표

| 카테고리 | 태스크 | 클래스 수 |
|----------|--------|-----------|
| 여드름 | `acne` | 4 |
| 색소침착 | `forehead_pigmentation`, `l/r_cheek_pigmentation` | 6 |
| 주름 | `forehead_wrinkle`, `glabellus_wrinkle`, `l/r_perocular_wrinkle` | 6 |
| 모공 | `l/r_cheek_pore` | 5 |
| 건조 | `lip_dryness` | 5 |
| 처짐 | `chin_sagging` | 6 |

---

## 모델 구조

```
SkinModel
├── SwinStageExtractor   # Swin-T 백본: stage2(384d) + stage3(768d) 피처 추출
├── TaskTokenBank        # 태스크별 learnable query token
├── TaskDecoder          # TSA → CrossAttn(stage3) → CrossAttn(stage2)
└── heads (ModuleDict)   # 태스크별 분류 헤드 (Linear-GELU-Dropout-Linear)
```

전역(full face) + 지역(local crop) 피처를 학습 가능한 `alpha`로 가중 혼합

---

## 파일 구조

```
skin_model/
├── config.py    # 태스크 정의, 클래스 수, face part bbox 등
├── dataset.py   # SkinDataset, TestDataset, DataLoader 빌더
├── img_crop.py  # PIL 전처리, bbox crop, 정규화
├── model.py     # SkinModel 전체 구조
├── loss.py      # OrdinalCELoss, OrdinalMSELoss, CBFocalLoss
├── metrics.py   # exact_acc, ad_acc, MAE, RMSE, QWK
├── train.py     # 학습 루프
├── test.py      # 단일 이미지 추론
├── utils.py     # 유틸 모음
└── logger.py    # KST 기준 파일+콘솔 로거
```
---

## 손실 함수

| 태스크 | 손실 함수 |
|--------|-----------|
| `acne` | Class-balanced CrossEntropy |
| `lip_dryness`, `l/r_cheek_pore` | CB-CE + 0.4 × MSE |
| 나머지 ordinal 태스크 | OrdinalMSELoss (ordinal smooth CE + 0.4 × MSE) |

---

## 평가 지표

- **Exact Acc**: 정확히 일치한 비율
- **Adjacent Acc (ad_acc)**: ±1 이내 비율
- **MAE**: 평균 절대 오차 (최적화 기준)
- **RMSE**: 평균 제곱근 오차
- **QWK**: Quadratic Weighted Kappa

---

## 데이터 구조

```
/TS/   # 이미지: {subject_id}_{device}_{F/S}_*.jpg  (F = front만 사용)
/TL/   # 라벨:  {subject_id}_{device}_{F/S}_{fp_idx}.json  (fp_idx: 0~8)
```

촬영 기기: `01` 디지털카메라, `02` 스마트패드, `03` 스마트폰
