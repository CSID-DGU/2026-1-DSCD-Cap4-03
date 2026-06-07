# ROUPLE API SPEC

2026-1 데이터사이언스캡스톤디자인 3조 Cap4

> **이 문서의 목적**
> 프론트엔드 구현 기준으로 백엔드가 만들어야 할 API를 정확히 명세한 문서입니다.
> 요청/응답의 **필드명, 타입, 필수 여부**를 모두 프론트 코드 기준으로 맞췄습니다.
> Mock API 기준이므로 실제 DB/모델 연동 시 이 구조를 유지하면서 내부 로직만 교체하면 됩니다.


---

## 1. 공통 규칙

### Base URL

```
로컬: http://127.0.0.1:8000
```

### 인증 방식

로그인/회원가입 성공 후 서버가 `access_token`을 응답합니다.
프론트는 이 토큰을 `localStorage`에 저장하고, 이후 모든 요청에 아래 헤더를 포함합니다.

```
Authorization: Bearer {access_token}
```

**인증이 필요한 엔드포인트:** 인증 API(`/auth/*`) 제외 전체

### 응답 형식

- 성공: `200 OK`, `201 Created`
- 잘못된 요청: `400 Bad Request`, `422 Unprocessable Entity`
- 인증 실패: `401 Unauthorized`
- 권한 없음: `403 Forbidden`
- 데이터 없음: `404 Not Found`
- 중복: `409 Conflict`

에러 응답 형식 (모든 에러에 통일):
```json
{
  "detail": "에러 메시지"
}
```

### 날짜 형식

- 날짜만: `"YYYY-MM-DD"` (예: `"2002-01-15"`)
- 날짜+시간: ISO 8601, `"YYYY-MM-DDTHH:MM:SS+00:00"` (예: `"2026-05-15T10:00:00+00:00"`)


---

## 2. 화면 기준 API 매핑

| 화면 | 엔드포인트 |
|------|-----------|
| `LoginPage` | `POST /auth/login` |
| `SignupPage` | `POST /auth/signup` |
| `UserInfoPage` | `PATCH /users/me/profile`, `PUT /users/me/allergies` |
| `DiagnosisPage` | `POST /files/presign`, `POST /images` |
| `AnalysisHistoryPage` | `GET /users/me/skin-analysis` |
| `AnalysisResultPage` | `POST /skin-analysis`, `POST /skin-analysis/summaries`, `GET /skin-analysis/{result_id}` |
| `BudgetPage` | API 없음 (프론트 전용 상태, 예산값을 RoutinePage로 전달) |
| `RoutinePage` | `POST /recommendations`, `GET /recommendations/{session_id}`, `POST /recommendation-explanations`, `POST /routines/{session_id}/save` |
| `RoutineHistoryPage` | `GET /users/me/routines`, `GET /users/me/skin-analysis` |
| `MyPage` | `GET /users/me`, `PATCH /users/me/profile`, `PUT /users/me/allergies`, `GET /users/me/allergies`, `GET /users/me/wishlist` |
| `ProductListPage` | `GET /products` |
| `ProductDetailPage` | `GET /products/{product_id}`, `POST /wishlist/{product_id}`, `DELETE /wishlist/{product_id}` |


---

## 3. 전체 화면 흐름 및 API 호출 순서

```
[회원가입/로그인]
  SignupPage: POST /auth/signup
  LoginPage:  POST /auth/login
      ↓ (access_token 발급, localStorage 저장)

[사용자 정보 입력]
  UserInfoPage: PATCH /users/me/profile
                PUT  /users/me/allergies

[피부 진단]
  DiagnosisPage:
    1. POST /files/presign        → S3 presigned URL 발급
    2. S3에 직접 업로드 (PUT, 프론트가 직접 S3로 보냄)
    3. POST /images               → 이미지 메타데이터 저장 → image_id 획득
    4. navigate('/analysis', { state: { image_id, imageUrl } })

[분석 결과]
  AnalysisResultPage:
    1. POST /skin-analysis        → image_id 전송 → 모델 실행 → result_id 획득
    2. POST /skin-analysis/summaries → result_id 전송 → LLM 요약 생성
    3. GET  /skin-analysis/{result_id} → 최종 결합 결과 조회 (차트 + 코멘트 전부)

[루틴 추천]
  BudgetPage: 예산 선택 (로컬 state, API 없음)
      ↓ navigate('/routine/result', { state: { resultId, budget } })
  RoutinePage:
    1. POST /recommendations      → result_id + 예산 전송 → 루틴 추천 실행
    2. POST /recommendation-explanations → session_id 전송 → LLM 설명 생성
    3. GET  /recommendations/{session_id} → 결과 재조회 (선택)
    4. POST /routines/{session_id}/save  → 마음에 드는 루틴 저장
```


---

## 4. 인증 API

### `POST /auth/signup` — 회원가입

**프론트 SignupPage가 보내는 값:**
- 이름, 닉네임, 이메일, 비밀번호, 비밀번호 확인 입력폼

**요청 본문:**
```json
{
  "user_name": "심고은",
  "nickname": "cap4user",
  "email": "test@example.com",
  "password": "password123",
  "login_type": "local"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `user_name` | string | ✅ | 실명 (이름 입력폼) |
| `nickname` | string | ✅ | 닉네임 |
| `email` | string | ✅ | 이메일 |
| `password` | string | ✅ | 비밀번호 (서버에서 해시 저장) |
| `login_type` | string | ✅ | 현재는 항상 `"local"` |

**응답 `201`:**
```json
{
  "access_token": "mock-token-1",
  "token_type": "bearer",
  "user_id": 1,
  "nickname": "cap4user"
}
```

**에러:**
- `409`: 이미 사용 중인 이메일

---

### `POST /auth/login` — 로그인

**요청 본문:**
```json
{
  "email": "test@example.com",
  "password": "password123"
}
```

**응답 `200`:**
```json
{
  "access_token": "mock-token-1",
  "token_type": "bearer",
  "user_id": 1,
  "nickname": "cap4user"
}
```

**에러:**
- `401`: 이메일 또는 비밀번호 불일치


---

## 5. 사용자 API

### `GET /users/me` — 내 정보 조회

**MyPage에서 사용. 탭: 내 정보 / 찜목록 (2개)**

> ⚠️ 분석기록·저장루틴 탭은 MyPage에서 제거됨. 분석기록은 `AnalysisHistoryPage`, 루틴은 `RoutineHistoryPage`에서 별도 관리.

**응답 `200`:**
```json
{
  "user_id": 1,
  "email": "test@example.com",
  "user_name": "이서은",
  "nickname": "cap4user",
  "login_type": "local",
  "gender": "female",
  "birth": "2002-01-15",
  "skin_type": "건성",
  "skin_concerns": ["acne", "dryness", "pore"]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `user_id` | int | 사용자 ID |
| `email` | string | 이메일 |
| `user_name` | string | 실명 |
| `nickname` | string | 닉네임 |
| `gender` | string | `"female"` 또는 `"male"` (프론트 표시: 여성/남성) |
| `birth` | string | `"YYYY-MM-DD"` |
| `skin_type` | string | 한국어 그대로 (예: `"건성"`, `"지성"`, `"복합성"`, `"중성"`, `"수부지"`, `"모름"`) |
| `skin_concerns` | string[] | 피부 고민 id 배열 (아래 코드표 참고) |

**skin_concerns 코드표:**
```
acne, wrinkle, brightening, sebum, dryness,
redness, dark_circle, atopy, sensitive, pore, flushing, keratin, none
```

---

### `PATCH /users/me/profile` — 내 정보 수정

**UserInfoPage 완료 버튼 클릭 시 호출**

**요청 본문:**
```json
{
  "gender": "female",
  "birth": "2002-01-15",
  "skin_type": "건성",
  "skin_concerns": ["acne", "dryness", "pore"]
}
```

> ⚠️ `nickname`, `user_name`은 현재 프론트 UserInfoPage에서 수정 불가 (MyPage 내 정보 탭에서만 수정). 향후 추가 예정.

**응답 `200`:** `GET /users/me`와 동일 형식

---

### `GET /users/me/allergies` — 알레르기 조회

**MyPage 내 정보 탭 진입 시 현재 알레르기 설정 불러오기**

**응답 `200`:**
```json
{
  "allergy_categories": ["fragrance", "preservative"],
  "allergy_ingredient_ids": [222, 542, 3219, 4053]
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `allergy_categories` | string[] | 저장된 카테고리 코드 목록 (없으면 빈 배열) |
| `allergy_ingredient_ids` | int[] | 저장된 성분 ID 목록 (없으면 빈 배열) |

> ⚠️ 현재 미구현. 백엔드 추가 필요.

---

### `PUT /users/me/allergies` — 알레르기 저장

**UserInfoPage의 AllergySelector 컴포넌트 결과값을 그대로 전송**

**AllergySelector 선택 결과 구조:**
- `allergy_categories`: 선택한 카테고리 코드 배열
- `allergy_ingredient_ids`: 선택한 개별 성분의 DB id 배열 (integer)

**카테고리 코드표:**
| 코드 | 표시 이름 |
|------|----------|
| `fragrance` | 향료/퍼퓸 |
| `preservative` | 보존제 |
| `metal` | 금속 |
| `plant_essential_oil` | 식물/에센셜오일 성분 |

**요청 본문:**
```json
{
  "allergy_categories": ["fragrance", "preservative"],
  "allergy_ingredient_ids": [222, 542, 3219, 4053]
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `allergy_categories` | string[] | ✅ | 선택한 카테고리 코드 목록 (없으면 빈 배열) |
| `allergy_ingredient_ids` | int[] | ✅ | 선택한 성분의 ingredient_id 목록 (없으면 빈 배열) |

> ingredient_id는 AllergySelector.tsx에 하드코딩된 값과 DB의 INGREDIENT 테이블 id가 일치해야 합니다.
> (예: fragrance의 Linalool = id 2963, Benzyl Alcohol = id 542 등)

**응답 `200`:**
```json
{
  "user_id": 1,
  "allergy_categories": ["fragrance", "preservative"],
  "allergy_ingredient_ids": [222, 542, 3219, 4053],
  "saved_count": 4
}
```


---

## 6. 파일 / 이미지 API

### `POST /files/presign` — S3 presigned URL 발급

**DiagnosisPage에서 이미지 크롭 완료 후 S3 업로드 전에 호출**

**크롭 고정 스펙 (프론트 하드코딩):**
- 크기: **480 × 640px**
- 형식: **JPEG, 압축률 90%**
- 예상 파일명: `skin.jpg`

**요청 본문:**
```json
{
  "file_name": "skin.jpg",
  "mime_type": "image/jpeg",
  "file_size": 245123
}
```

**응답 `200`:**
```json
{
  "upload_url": "https://s3.ap-northeast-2.amazonaws.com/bucket/...?X-Amz-Signature=...",
  "public_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/users/1/images/abc123.jpg",
  "s3_key": "users/1/images/abc123.jpg",
  "expires_in": 900
}
```

> 프론트는 `upload_url`로 PUT 요청을 직접 보내 S3에 업로드합니다.
> 업로드 완료 후 `public_url`과 `s3_key`를 사용해 `POST /images`를 호출합니다.

---

### `POST /images` — 이미지 메타데이터 저장

**S3 업로드 완료 직후 호출. 응답의 `image_id`를 이후 분석 API에 사용합니다.**

**요청 본문:**
```json
{
  "storage_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/users/1/images/abc123.jpg",
  "s3_key": "users/1/images/abc123.jpg",
  "original_file_name": "skin.jpg",
  "mime_type": "image/jpeg",
  "file_size": 245123,
  "crop_data": {
    "x": 40,
    "y": 60,
    "width": 480,
    "height": 640
  },
  "upload_status": "UPLOADED"
}
```

**응답 `201`:**
```json
{
  "image_id": 3,
  "user_id": 1,
  "storage_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/users/1/images/abc123.jpg",
  "s3_key": "users/1/images/abc123.jpg",
  "uploaded_at": "2026-05-15T10:00:00+00:00"
}
```

> 프론트는 응답에서 **`image_id`** 와 **`storage_url`** 만 사용합니다.
> `image_id`는 `POST /skin-analysis`에, `storage_url`은 AnalysisResultPage 사진 표시에 사용합니다.


---

## 7. 피부분석 API

### 분석 흐름 요약

```
POST /skin-analysis            → 모델 실행, result_id 반환
POST /skin-analysis/summaries  → LLM 실행, 코멘트 생성
GET  /skin-analysis/{result_id} → 위 두 결과 합쳐서 프론트에 전달
```

---

### `POST /skin-analysis` — 피부분석 실행

**요청 본문:**
```json
{
  "image_id": 3
}
```

**응답 `201`:**
```json
{
  "result_id": 7,
  "image_id": 3,
  "user_id": 1,
  "analyzed_at": "2026-05-15T10:05:00+00:00",
  "model_version": "mock-skin-analysis-v1",
  "analysis_status": "SUCCESS"
}
```

> 프론트는 `result_id`만 사용해서 바로 `POST /skin-analysis/summaries`를 호출합니다.

---

### `POST /skin-analysis/summaries` — LLM 요약 생성

**요청 본문:**
```json
{
  "result_id": 7
}
```

**응답 `201`:**

> ⚠️ 기존 스펙의 `summary_text / concern_text / care_tip_text` 구조는 프론트에서 사용하지 않습니다.
> 아래 구조로 변경이 필요합니다.

```json
{
  "result_id": 7,
  "llm_model": "mock-llm-analysis-v1",
  "prompt_version": "skin_v1",
  "summary_comment": "전반적으로 모공과 건조 지표가 상대적으로 조금 더 눈에 띄어 피지·모공 관리와 수분 유지에 신경 쓰는 방향이 적합합니다.",
  "indicator_comments": {
    "acne": "트러블 지표는 비교적 낮은 편이라 저자극 세안과 진정 케어로 피지 밸런스를 안정적으로 유지하는 방향이 적합합니다.",
    "dryness": "건조 지표는 살짝 신경 쓰이는 수준이므로 수분 공급과 보습막 형성을 통해 피부 장벽과 수분 유지를 꾸준히 챙기는 것이 좋습니다.",
    "sagging": "처짐 지표는 낮은 편이라 현재는 무리한 리프팅 케어보다는 탄력 유지를 위한 기본적인 보습과 생활 관리에 집중하면 충분합니다.",
    "pore": "모공 지표가 상대적으로 더 도드라져 피지 조절과 모공 케어, 각질 정돈을 함께 해주면 피부결 관리에 도움이 됩니다.",
    "pigmentation": "색소침착 지표는 낮은 편이지만 피부톤 관리를 위해 자외선 차단과 가벼운 브라이트닝 케어를 꾸준히 이어가는 것이 좋습니다.",
    "wrinkle": "주름 지표는 비교적 낮아 잔주름 예방 중심으로 보습과 탄력 관리를 유지하면 좋습니다."
  },
  "generated_at": "2026-05-15T10:06:00+00:00"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `summary_comment` | string | 전체 피부 종합 코멘트 (1~3문장) |
| `indicator_comments` | object | 6개 지표별 개별 코멘트 객체 |
| `indicator_comments.acne` | string | 트러블 지표 코멘트 |
| `indicator_comments.dryness` | string | 건조 지표 코멘트 |
| `indicator_comments.sagging` | string | 처짐 지표 코멘트 |
| `indicator_comments.pore` | string | 모공 지표 코멘트 |
| `indicator_comments.pigmentation` | string | 색소침착 지표 코멘트 |
| `indicator_comments.wrinkle` | string | 주름 지표 코멘트 |

---

### `GET /skin-analysis/{result_id}` — 분석 결과 전체 조회

**AnalysisResultPage가 최종적으로 호출하는 엔드포인트. 모델 점수 + LLM 코멘트 + 이미지 URL을 합쳐서 응답해야 합니다.**

**응답 `200`:**
```json
{
  "result_id": 7,
  "user_id": 1,
  "image_id": 3,
  "model_name": "mock-skin-analysis-v1",
  "prompt_version": "skin_v1",
  "analyzed_at": "2026-05-15T10:05:00+00:00",
  "generated_at": "2026-05-15T10:06:00+00:00",

  "summary_comment": "전반적으로 모공과 건조 지표가 상대적으로 조금 더 눈에 띄어 피지·모공 관리와 수분 유지에 신경 쓰는 방향이 적합합니다.",
  "indicator_comments": {
    "acne": "트러블 지표는 비교적 낮은 편이라 저자극 세안과 진정 케어로 유지하는 방향이 적합합니다.",
    "dryness": "건조 지표는 살짝 신경 쓰이는 수준이므로 수분 공급과 보습막 형성을 꾸준히 챙겨주세요.",
    "sagging": "처짐 지표는 낮아 기본적인 보습과 생활 관리면 충분합니다.",
    "pore": "모공 지표가 도드라져 피지 조절과 모공 케어를 함께 해주세요.",
    "pigmentation": "색소침착은 낮지만 자외선 차단을 꾸준히 이어가세요.",
    "wrinkle": "주름 지표는 낮아 예방 중심으로 보습 관리를 유지하세요."
  },

  "image_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/users/1/images/abc123.jpg",
  "skin_type": "건성",

  "raw_metrics": {
    "acne": 1,
    "dryness": 2,
    "sagging": 1,
    "pore": 2,
    "pigmentation": 1,
    "wrinkle": 1
  },
  "display_scores": {
    "acne": 26,
    "dryness": 38,
    "sagging": 10,
    "pore": 41,
    "pigmentation": 28,
    "wrinkle": 21
  }
}
```

**프론트가 사용하는 필드 설명:**

| 필드 | 타입 | 어디에 표시 |
|------|------|------------|
| `summary_comment` | string | "AI 한마디" 박스 |
| `indicator_comments.{key}` | string | 각 지표 카드 하단 설명 |
| `image_url` | string | 분석 결과 페이지 프로필 사진 |
| `skin_type` | string | 피부 타입 뱃지 (예: 건성) |
| `raw_metrics.{key}` | int | 등급 계산용 (아래 등급 기준표 참고) |
| `display_scores.{key}` | int (0~100) | 레이더 차트 + 지표별 바 차트 |

**raw_metrics 등급 기준표 (프론트에 하드코딩됨, 참고용):**
```
acne:         0 → 양호 / 1 → 보통 / 2,3 → 개선필요
dryness:      0,1 → 양호 / 2 → 보통 / 3,4 → 개선필요
sagging:      0,1 → 양호 / 2,3 → 보통 / 4,5 → 개선필요
pore:         0,1 → 양호 / 2,3 → 보통 / 4,5 → 개선필요
pigmentation: 0 → 양호 / 1,2,3 → 보통 / 4,5 → 개선필요
wrinkle:      0,1 → 양호 / 2,3 → 보통 / 4,5,6 → 개선필요
```

---

### `GET /users/me/skin-analysis` — 분석 기록 목록 조회

**AnalysisHistoryPage, RoutineHistoryPage, MyPage(분석기록 탭)에서 사용**

**응답 `200`:**
```json
{
  "items": [
    {
      "result_id": 7,
      "image_id": 3,
      "analyzed_at": "2026-05-15T10:05:00+00:00",
      "skin_type": "건성",
      "image_url": "https://bucket.s3.ap-northeast-2.amazonaws.com/users/1/images/abc123.jpg",
      "ai_comment": "모공과 건조 지표에 집중한 루틴이 필요해요."
    }
  ]
}
```

| 필드 | 타입 | 어디에 표시 |
|------|------|------------|
| `result_id` | int | 상세 조회 시 사용 |
| `image_url` | string | 목록 카드의 썸네일 이미지 |
| `analyzed_at` | string | 분석 날짜 표시 |
| `skin_type` | string | 피부 타입 뱃지 |
| `ai_comment` | string | 카드 하단 짧은 요약 (1문장 정도) |

> 최신순 정렬 필요 (내림차순). 프론트에서 `[...results].reverse()`로 뒤집어 표시합니다.


---

## 8. 추천 API

### 예산 입력 구조 (BudgetPage → RoutinePage 전달)

BudgetPage에서 선택한 예산은 router state로 RoutinePage에 전달됩니다.
RoutinePage는 이 값을 `POST /recommendations` 요청에 포함합니다.

**BudgetPage 카테고리 ↔ API 필드 매핑:**

| 프론트 카테고리 | API 필드 | 설명 |
|---------------|---------|------|
| 전체 | `total_budget` | 전체 합산 예산 상한 |
| 토너 | `toner_budget` | 토너 슬롯 예산 |
| 에멀젼 | `emulsion_budget` | 에멀젼 슬롯 예산 |
| 앰플 | `ampoule_budget` | 앰플/세럼 슬롯 예산 |
| 크림 | `cream_budget` | 크림 슬롯 예산 |

값이 `null`이면 해당 카테고리 예산 제한 없음을 의미합니다.

---

### `POST /recommendations` — 루틴 추천 실행

**요청 본문:**
```json
{
  "result_id": 7,
  "image_id": 3,
  "total_budget": 150000,
  "toner_budget": null,
  "emulsion_budget": null,
  "ampoule_budget": 40000,
  "cream_budget": 30000
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `result_id` | int | ✅ | 피부분석 결과 ID |
| `image_id` | int | ✅ | 이미지 ID |
| `total_budget` | int \| null | ✅ | 전체 예산 (null이면 제한 없음) |
| `toner_budget` | int \| null | ✅ | 토너 예산 (null이면 제한 없음) |
| `emulsion_budget` | int \| null | ✅ | 에멀젼 예산 (null이면 제한 없음) |
| `ampoule_budget` | int \| null | ✅ | 앰플 예산 (null이면 제한 없음) |
| `cream_budget` | int \| null | ✅ | 크림 예산 (null이면 제한 없음) |

**응답 `201`:**

> ⚠️ 기존 스펙에 없던 필드들이 많습니다. 프론트 RoutinePage가 필요로 하는 전체 구조입니다.

```json
{
  "session_id": 5,
  "user_id": 1,
  "result_id": 7,
  "session_status": "SUCCESS",
  "routines": [
    {
      "routine_id": "r001",
      "type": "best",
      "label": "AI BEST 루틴",
      "routine_time": "both",
      "total_cost": 161000,
      "duration": 5,
      "ai_description": "AM+PM 공용 루틴으로 토너, 에멀젼, 앰플, 크림 4단계로 구성되어 보습 강화와 모공 관리에 맞춰 추천되었습니다.",
      "products": [
        {
          "product_id": 1001,
          "step": 1,
          "application_guide": "세안 후 화장솜이나 손으로 얼굴 전체에 부드럽게 펴 발라주세요. 가볍게 두드려 흡수시켜 주세요.",
          "time_tag": null
        },
        {
          "product_id": 1002,
          "step": 2,
          "application_guide": "토너 흡수 후 적당량을 손에 덜어 얼굴에 펴 발라주세요.",
          "time_tag": null
        }
      ]
    },
    {
      "routine_id": "r002",
      "type": "budget",
      "label": "가성비 루틴",
      "routine_time": "pm",
      "total_cost": 87000,
      "duration": 4,
      "ai_description": "PM 전용 루틴으로 저자극 성분 위주로 구성해 취침 전 집중 보습과 피부 장벽 강화에 초점을 맞췄습니다.",
      "products": [
        {
          "product_id": 1003,
          "step": 1,
          "application_guide": "저녁 세안 후 화장솜이나 손으로 얼굴 전체에 부드럽게 펴 발라주세요.",
          "time_tag": "pm"
        }
      ]
    }
  ]
}
```

**프론트가 사용하는 루틴 필드 설명:**

| 필드 | 타입 | 어디에 표시 |
|------|------|------------|
| `type` | `"best"` \| `"budget"` | 탭 전환 (🏆 AI 추천 / 💸 가성비) |
| `label` | string | 루틴 이름 표시 |
| `routine_time` | `"am"` \| `"pm"` \| `"both"` | 시간대 뱃지 (🌅 AM전용 / 🌙 PM전용 / 🌅🌙 AM+PM) |
| `total_cost` | int | 총 비용 표시 (예: 161,000원) |
| `duration` | int | 소요 시간 표시 (예: 5분) |
| `ai_description` | string | "AI 루틴 추천 이유" 박스 |
| `products[].step` | int | STEP 번호 표시 |
| `products[].application_guide` | string | 바르는 법 카드 |
| `products[].time_tag` | `"am"` \| `"pm"` \| null | AM/PM 전용 뱃지 (null이면 공용) |

> 루틴은 반드시 **2개** (best + budget) 반환해야 합니다. 프론트 탭이 2개 고정입니다.

---

### `GET /recommendations/{session_id}` — 추천 결과 재조회

`POST /recommendations` 응답과 동일한 구조.

---

### `POST /recommendation-explanations` — LLM 추천 설명 생성

**요청 본문:**
```json
{
  "session_id": 5
}
```

**응답 `201`:**
```json
{
  "session_id": 5,
  "llm_model": "mock-llm-recommendation-v1",
  "prompt_version": "recommendation-explanation-v1",
  "summary_text": "현재 피부 상태를 기준으로 진정, 보습, 장벽 케어 중심 루틴을 구성했습니다.",
  "usage_guide_text": "토너부터 에센스, 로션, 크림 순서로 사용하고 자극감이 있으면 빈도를 줄이세요.",
  "warning_text": "피부 자극이 느껴질 경우 새로운 제품은 한 번에 여러 개 추가하지 마세요."
}
```

> 현재 프론트 RoutinePage는 `POST /recommendations` 응답의 `ai_description` 필드를 직접 사용합니다.
> 이 엔드포인트의 응답은 향후 추가 팝업 또는 모달에서 사용할 예정입니다.


---

## 9. 루틴 저장 API

### `POST /routines/{session_id}/save` — 루틴 저장

**RoutinePage에서 "루틴 저장" 버튼 클릭 시 호출**

**경로 파라미터:**
- `session_id`: 저장할 추천 세션 ID

**요청 본문:**
```json
{
  "routine_type": "best"
}
```

| 필드 | 타입 | 설명 |
|------|------|------|
| `routine_type` | `"best"` \| `"budget"` | 어떤 루틴을 저장할지 |

**응답 `201`:**
```json
{
  "saved_routine_id": 11,
  "session_id": 5,
  "routine_type": "best",
  "saved_at": "2026-05-15T10:20:00+00:00"
}
```

---

### `GET /users/me/routines` — 저장된 루틴 목록

**RoutineHistoryPage, MyPage(저장된루틴 탭)에서 사용**

**응답 `200`:**
```json
{
  "items": [
    {
      "saved_routine_id": 11,
      "session_id": 5,
      "routine_type": "best",
      "label": "AI BEST 루틴",
      "routine_time": "both",
      "total_cost": 161000,
      "duration": 5,
      "saved_at": "2026-05-15T10:20:00+00:00",
      "products": [
        {
          "product_id": 1001,
          "step": 1,
          "product_name": "제모스 토너",
          "brand_name": "유리아쥬",
          "category": "토너",
          "price": 29000,
          "image_url": "https://..."
        }
      ]
    }
  ]
}
```

**프론트가 사용하는 필드:**

| 필드 | 어디에 표시 |
|------|------------|
| `label` | 루틴 이름 |
| `routine_type` | 뱃지 (🏆 AI BEST / 💸 가성비) |
| `total_cost` | 총 비용 |
| `duration` | 소요 시간 |
| `products[].image_url` | 제품 썸네일 |
| `products[].product_name` | 제품명 |
| `products[].brand_name` | 브랜드명 |
| `products[].price` | 제품 가격 |


---

## 10. 상품 / 찜 API

### `GET /products` — 상품 목록 조회

**ProductListPage에서 카테고리 탭 필터링에 사용**

**쿼리 파라미터 (선택):**

| 파라미터 | 타입 | 예시 | 설명 |
|---------|------|------|------|
| `category` | string | `토너` | 카테고리 필터 (없으면 전체) |

**프론트 카테고리 탭 (표시명 → API 파라미터):**

| 탭 표시명 | `category` 파라미터 | 설명 |
|---------|-----------------|------|
| 전체 | (파라미터 없음) | 전체 조회 |
| 토너 | `Toner`, `Toner Pads` | 두 카테고리 병렬 조회 후 합산 |
| 에멀젼 | `Emulsions` | |
| 에센스/앰플 | `Essences/Ampoules/Serums` | |
| 크림/젤 | `Cream/Gel` | |

**옵션 루틴 카테고리 (성별 기반, RoutinePage · RoutineHistoryPage에서 사용):**

| 성별 | `category` 파라미터 |
|------|-----------------|
| 여성 (`female`) | `Balms/Multi-balms`, `Eye Treatments`, `Facial Oils` |
| 남성 (`male`) | `Shaving Products`, `All-In-One` |

**응답 `200`:**
```json
[
  {
    "product_id": 1001,
    "brand_name": "유리아쥬",
    "product_name": "제모스 토너",
    "category": "토너",
    "price": 29000,
    "image_url": "https://incidecoder-content.storage.googleapis.com/.../product_front.jpeg",
    "tags": ["수분", "토닝", "히알루론산"]
  }
]
```

**프론트가 목록 카드에 표시하는 필드:** `image_url`, `brand_name`, `product_name`, `category`, `price`, `tags`

---

### `GET /products/{product_id}` — 상품 상세 조회

**ProductDetailPage에서 사용. 목록보다 훨씬 많은 필드가 필요합니다.**

> ⚠️ 기존 스펙에 없던 필드들입니다. 반드시 추가해야 합니다.

**응답 `200`:**
```json
{
  "product_id": 1001,
  "brand_name": "유리아쥬",
  "product_name": "제모스 토너",
  "category": "토너",
  "price": 29000,
  "image_url": "https://incidecoder-content.storage.googleapis.com/.../product_front.jpeg",
  "tags": ["수분", "토닝", "히알루론산"],
  "ingredients": ["블루 히알루론산", "히알루론산 5종", "알로에베라"],
  "pros": ["풍부한 수분감", "레이어링 가능", "산뜻한 마무리"],
  "cons": ["건성 피부엔 추가 보습 필요"],
  "how_to_use": "세안 후 손바닥에 적당량을 덜어 얼굴 전체에 흡수시켜 주세요.",
  "apply_time": "30초"
}
```

**ProductDetailPage가 사용하는 모든 필드:**

| 필드 | 어디에 표시 |
|------|------------|
| `image_url` | 제품 대표 이미지 |
| `brand_name` | 브랜드명 |
| `product_name` | 제품명 (h1) |
| `category` | 카테고리 뱃지 |
| `price` | 가격 |
| `tags` | # 태그 목록 |
| `ingredients` | 🧪 주요 성분 |
| `pros` | 👍 장점 목록 |
| `cons` | 👎 단점 목록 |
| `how_to_use` | 📋 사용 방법 |
| `apply_time` | ⏱ 평균 흡수 시간 |

> `GET /products`와 `GET /products/{product_id}` 모두 `product_id`는 **정수**입니다.
> 프론트에서 라우트 파라미터로 `:id`를 사용하며, API 호출 시 `/products/1001` 형태로 전달합니다.

---

### `POST /wishlist/{product_id}` — 찜 추가

**응답 `201`:**
```json
{
  "product_id": 1001,
  "saved": true
}
```

---

### `DELETE /wishlist/{product_id}` — 찜 삭제

**응답 `200`:**
```json
{
  "product_id": 1001,
  "saved": false
}
```

---

### `GET /users/me/wishlist` — 내 찜 목록

**MyPage 찜목록 탭, ProductListPage 찜 버튼 상태 초기화에 사용**

**응답 `200`:**
```json
{
  "items": [
    {
      "product_id": 1001,
      "brand_name": "유리아쥬",
      "product_name": "제모스 토너",
      "category": "토너",
      "price": 29000,
      "image_url": "https://..."
    }
  ]
}
```


---

## 11. 현재 구현 상태

| 항목 | 상태 |
|------|------|
| 프론트 전체 화면 UI | ✅ Mock 데이터 기반 완성 |
| 프론트 인증 | ⚠️ localStorage 플래그만 (토큰 없음) |
| 프론트 API 연동 | ❌ 전부 Mock, 미연동 |
| S3 업로드 | ⚠️ 코드 작성됨, 주석 처리 상태 |
| 백엔드 FastAPI | ❓ |
| 피부분석 모델 | ✅ 거의 완성 |
| 추천 모델 | ✅ 거의 완성 |
| LLM | ❌ 미연동 |


---

## 12. 백엔드 개발 시 주의사항

1. **필드명은 snake_case** 로 통일 (프론트에서 camelCase로 변환해서 사용)

2. **카테고리 이름은 영어**로 저장/응답. 프론트에서 한국어로 변환하여 표시.
   - 필수: `"Toner"`, `"Toner Pads"`, `"Emulsions"`, `"Essences/Ampoules/Serums"`, `"Cream/Gel"`
   - 여성 옵션: `"Balms/Multi-balms"`, `"Eye Treatments"`, `"Facial Oils"`
   - 남성 옵션: `"Shaving Products"`, `"All-In-One"`

3. **피부 타입은 한국어** (`"건성"`, `"지성"`, `"중성"`, `"복합성"`, `"수부지"`, `"모름"`)
   **성별은 영어** (`"female"`, `"male"`), 프론트에서 여성/남성으로 표시

4. **`GET /skin-analysis/{result_id}`** 는 모델 점수 + LLM 코멘트 + 이미지 URL을 합친 응답이어야 합니다. 프론트가 단 하나의 API로 AnalysisResultPage 전체를 렌더링합니다.

5. **루틴은 반드시 2개** (type: best, type: budget) 반환해야 합니다.

6. **AllergySelector의 ingredient_id**는 DB `INGREDIENT` 테이블의 실제 id와 일치해야 합니다. [AllergySelector.tsx](front/src/components/AllergySelector.tsx) 파일에 id 목록이 하드코딩되어 있습니다.

7. **`POST /recommendations` 응답**에 이미 `ai_description`이 포함되어야 합니다 (LLM 결과). `POST /recommendation-explanations`는 별도 추가 설명용입니다.

8. 분석 결과 목록 (`GET /users/me/skin-analysis`)의 **`ai_comment`** 는 `summary_comment`를 1문장으로 요약한 짧은 버전입니다.


---

## 13. 다음 수정 예정

1. 실제 S3 연동 후 presigned URL 방식 검증
2. 실제 모델 입출력 기준으로 `raw_metrics` / `display_scores` 범위 확정
3. 실제 LLM 프롬프트 버전 관리 반영
4. 에러 응답 상세 케이스 추가
5. 페이지네이션 필요 여부 검토 (상품 목록, 분석 기록)
