# ROUPLE 개발 플로우

2026-1 데이터사이언스캡스톤디자인
3조 Cap4
현민영, 이서은, 심고은, 서동현

## 1. 목적

- 팀 공통 개발 흐름 정리
- 백엔드 / 프론트 / 모델 연동 순서 정리
- 변경 가능성이 큰 모델, DB, API를 무리 없이 연결하기 위한 기준 정리

현재 전제:

- 프론트엔드: React
- 백엔드: FastAPI
- 이미지 업로드: 프론트에서 S3 직접 업로드 방식
- 피부분석 모델: 거의 완성 상태, 일부 수정 가능
- 추천 모델: 거의 완성 상태, 일부 수정 가능
- LLM: 피부분석 뒤, 추천 뒤에 각각 연결 예정
- DB 스키마: 초안 상태, 변경 가능
- 사용자 입력값 중 피부타입 / 피부고민 / 알러지는 선택 입력 기준


## 2. 전체 흐름

1. 회원가입 / 로그인
2. 프로필 입력
3. 얼굴 이미지 업로드 및 크롭
4. 프론트에서 S3 직접 업로드
5. 백엔드에서 이미지 메타데이터 저장
6. 피부분석 모델 실행
7. 피부분석 결과 및 LLM 요약 확인
8. 내 피부에 맞는 루틴 보기 진입
9. 추천 모델 실행
10. 추천 결과 및 LLM 설명 확인
11. 루틴 저장
12. 제품 탐색 / 제품 상세 / 찜
13. 마이페이지에서 분석 결과 / 저장 루틴 / 찜 목록 확인


## 2-1. 프론트 기준 화면 구성

- `MainPage`: 메인 화면
- `LoginPage`: 로그인
- `SignupPage`: 회원가입
- `UserInfoPage`: 사용자 정보 입력 / 수정
- `DiagnosisPage`: 이미지 업로드 / 크롭 / 업로드 시작
- `AnalysisResultPage`: 피부분석 결과 및 요약 확인
- `RoutinePage`: 추천 루틴 및 추천 설명 확인
- `MyPage`: 내 정보 / 분석 기록 / 추천 기록
- `ProductListPage`: 화장품 목록
- `ProductDetailPage`: 화장품 상세

백엔드 연동 우선순위:

1. `LoginPage`, `SignupPage`
2. `UserInfoPage`
3. `DiagnosisPage`
4. `AnalysisResultPage`
5. `RoutinePage`
6. `MyPage`
7. `ProductListPage`, `ProductDetailPage`


## 3. 개발 원칙

- 화면 흐름 우선 고정
- API 계약 우선 고정
- Mock 응답으로 전체 흐름 먼저 연결
- 실제 모델은 나중에 교체
- 모델 로직 변경과 API 응답 형식 분리
- DB는 최소 구조 우선 사용
- 분석 결과 페이지와 루틴 페이지 분리 기준
- 사용자 선택 입력 누락 시에도 서비스 동작 가능 기준


## 4. 지금 할 일

1. 화면 흐름 최종 확인
2. API 명세 초안 작성
3. 최소 DB 구조 확정
4. FastAPI 백엔드 골격 생성
5. S3 presign + 이미지 메타데이터 저장 흐름 구현
6. Mock 피부분석 API 구현
7. Mock 분석 LLM API 구현
8. Mock 추천 API 구현
9. Mock 추천 설명 LLM API 구현
10. 프론트 연동
11. 이후 실제 모델 / 실제 LLM으로 교체


## 5. 최소 API 범위

- `POST /auth/signup`
- `POST /auth/login`
- `GET /users/me`
- `PATCH /users/me/profile`
- `PUT /users/me/allergies`
- `POST /files/presign`
- `POST /images`
- `POST /skin-analysis`
- `GET /skin-analysis/{result_id}`
- `POST /skin-analysis/summaries`
- `POST /recommendations`
- `GET /recommendations/{session_id}`
- `POST /recommendation-explanations`
- `GET /products`
- `GET /products/{product_id}`
- `POST /wishlist/{product_id}`
- `DELETE /wishlist/{product_id}`
- `POST /routines/{session_id}/save`
- `GET /users/me/wishlist`
- `GET /users/me/routines`
- `GET /users/me/skin-analysis`

화면 기준 API 연결:

- `LoginPage`, `SignupPage` -> 인증 API
- `UserInfoPage` -> 사용자 정보 / 알러지 API
- `DiagnosisPage` -> `POST /files/presign`, `POST /images`
- `AnalysisResultPage` -> `POST /skin-analysis`, `GET /skin-analysis/{result_id}`, `POST /skin-analysis/summaries`
- `RoutinePage` -> `POST /recommendations`, `GET /recommendations/{session_id}`, `POST /recommendation-explanations`
- `MyPage` -> 사용자 정보 / 분석 이력 / 추천 이력 API
- `ProductListPage`, `ProductDetailPage` -> 상품 목록 / 상세 / 위시리스트 API


## 6. DB 기준

현재 `schema.sql` 기준 핵심 테이블:

- 사용자: `USER`, `USER_PROFILE`, `USER_ALLERGY`, `USER_WISHLIST`
- 이미지/분석: `USER_IMAGE`, `SKIN_ANALYSIS_RESULT`
- 제품: `PRODUCT`, `PRODUCT_REVIEW`, `INGREDIENT`, `PRODUCT_INGREDIENT`, `INGREDIENT_CONFLICT`
- 추천: `RECOMMENDATION_CANDIDATE`, `RECOMMENDATION_SESSION`, `RECOMMENDATION_ROUTINE`, `RECOMMENDATION_ITEM`

사전 적재 대상 테이블:

- `PRODUCT`
- `PRODUCT_REVIEW`
- `INGREDIENT`
- `PRODUCT_INGREDIENT`
- `INGREDIENT_CONFLICT`

운영 중 생성 테이블:

- `USER`
- `USER_PROFILE`
- `USER_ALLERGY`
- `USER_WISHLIST`
- `USER_IMAGE`
- `SKIN_ANALYSIS_RESULT`
- `RECOMMENDATION_CANDIDATE`
- `RECOMMENDATION_SESSION`
- `RECOMMENDATION_ROUTINE`
- `RECOMMENDATION_ITEM`

추가 필요 가능 테이블:

- `SAVED_ROUTINE`
- `SAVED_ROUTINE_ITEM`

### 현재 스키마 기준 그대로 우선 사용 가능 영역

- 사용자 기본 정보 저장
- 알러지 저장
- 위시리스트 저장
- 업로드 이미지 저장
- 피부 점수 저장
- 추천 후보 / 추천 세션 / 루틴 / 루틴 아이템 저장

### 보완 권장 영역

#### `USER`

- `password` 길이 확장 필요
- 해시 비밀번호 저장 기준 필요

권장:

- `password VARCHAR(255)`

#### `USER_IMAGE`

현재 컬럼:

- `image_id`
- `user_id`
- `storage_url`
- `uploaded_at`

추가 권장 컬럼:

- `s3_key`
- `original_file_name`
- `mime_type`
- `file_size`
- `crop_data`
- `upload_status`

이유:

- S3 파일 추적 필요
- 프론트 크롭 정보 저장 필요
- 재분석 / 디버깅 시 원인 추적 필요

#### `SKIN_ANALYSIS_RESULT`

현재 컬럼:

- 6개 피부 점수
- `image_id`
- `user_id`
- `analyzed_at`

추가 권장 컬럼:

- `model_version`
- `analysis_status`
- `raw_result_json`
- `summary_status`
- `summary_text`

이유:

- 모델 버전 관리 필요
- 실패 / 성공 상태 구분 필요
- 원본 추론값 저장 필요
- LLM 요약 결과 저장 필요

#### `RECOMMENDATION_SESSION`

현재 컬럼:

- `user_id`
- `image_id`
- `result_id`
- `strict_budget`
- `total_budget`
- `budget_check_passed`
- `session_status`
- `failure_reason`

추가 권장 컬럼:

- `total_budget_min`
- `total_budget_max`
- `request_json`
- `model_version`
- `llm_status`

이유:

- 예산 min/max 분리 필요
- 요청 원본 저장 필요
- 추천 모델 버전 기록 필요
- 추천 설명 LLM 상태 관리 필요

#### 저장 루틴 기능

노션 기준 요구사항:

- 추천 루틴 저장 기능 필요
- 필터 적용 여부와 관계없이 저장 가능 필요
- 저장 시 제품명 / 브랜드 / 카테고리 보존 필요
- 마이페이지에서 저장 루틴 조회 가능 필요

권장 방식:

- `RECOMMENDATION_SESSION`은 추천 실행 로그 용도 유지
- 별도 저장 테이블로 루틴 스냅샷 보관

권장 테이블:

- `SAVED_ROUTINE`
- `SAVED_ROUTINE_ITEM`

예시 컬럼:

`SAVED_ROUTINE`
- `saved_routine_id`
- `user_id`
- `session_id`
- `routine_rank`
- `routine_label`
- `saved_at`

`SAVED_ROUTINE_ITEM`
- `saved_routine_item_id`
- `saved_routine_id`
- `slot_order`
- `category`
- `product_id`
- `brand_name`
- `product_name`

#### 추천 설명 저장 방식

선택지 1:

- `RECOMMENDATION_SESSION`에 설명 텍스트 컬럼 추가

선택지 2:

- 별도 테이블 추가

권장:

- 별도 테이블 `RECOMMENDATION_EXPLANATION`

예시 컬럼:

- `explanation_id`
- `session_id`
- `llm_model`
- `prompt_version`
- `summary_text`
- `usage_guide_text`
- `warning_text`
- `created_at`

#### 분석 요약 저장 방식

권장:

- 별도 테이블 `ANALYSIS_SUMMARY`

예시 컬럼:

- `summary_id`
- `result_id`
- `llm_model`
- `prompt_version`
- `summary_text`
- `concern_text`
- `care_tip_text`
- `created_at`


## 7. 로그 기준

### 피부분석 모델 로그

남길 항목:

- `request_id`
- `user_id`
- `image_id`
- `s3_key` 또는 `storage_url`
- `model_version`
- `input_image_size`
- `crop_data`
- `started_at`
- `finished_at`
- `latency_ms`
- `acne_score`
- `dryness_score`
- `pore_score`
- `wrinkle_score`
- `pigmentation_score`
- `sagging_score`
- `status`
- `error_message`

목적:

- 추론 실패 원인 추적
- 모델 버전별 결과 비교
- 잘못된 입력 이미지 확인

### 피부분석 LLM 로그

남길 항목:

- `request_id`
- `result_id`
- `llm_model`
- `prompt_version`
- `input_score_json`
- `started_at`
- `finished_at`
- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `status`
- `error_message`

목적:

- 요약 품질 추적
- 프롬프트 버전 관리
- 토큰 비용 추적

### 추천 모델 로그

남길 항목:

- `request_id`
- `session_id`
- `user_id`
- `image_id`
- `result_id`
- `model_version`
- `candidate_count`
- `hard_filter_input_count`
- `hard_filter_output_count`
- `drop_count_by_reason`
- `rerank_topk`
- `routine_count`
- `budget_check_passed`
- `failure_reason`
- `started_at`
- `finished_at`
- `latency_ms`
- `status`
- `error_message`
- `saved_routine_count`

추가 권장 로그:

- top 후보 상품 id 목록
- 최종 베스트 루틴 상품 id 목록
- 최종 가성비 루틴 상품 id 목록
- 충돌 검사 결과

목적:

- 추천 품질 분석
- 필터링 구간 병목 파악
- 예산 / 충돌 / 후보 부족 원인 추적

### 추천 LLM 로그

남길 항목:

- `request_id`
- `session_id`
- `llm_model`
- `prompt_version`
- `input_routine_json`
- `started_at`
- `finished_at`
- `latency_ms`
- `prompt_tokens`
- `completion_tokens`
- `status`
- `error_message`

목적:

- 추천 설명 품질 관리
- 토큰 비용 추적
- 설명 생성 실패 원인 추적


## 8. 마일스톤

### 1차

- FastAPI 실행
- 사용자 / 이미지 업로드 흐름 연결

### 2차

- Mock 피부분석 / Mock 추천 / Mock LLM 연결

### 3차

- 실제 피부분석 모델 연결
- 실제 추천 모델 연결

### 4차

- 실제 LLM 연결
- 프론트 전체 흐름 통합
- 저장 루틴 / 찜 / 마이페이지 흐름 통합

### 5차

- AWS 배포
- 운영 테스트


## 9. 첫 번째 성공 기준

`사용자가 이미지를 업로드하고, 피부분석 결과와 요약을 받고, 최종 루틴 추천과 설명 결과를 확인하는 전체 흐름이 한 번 끝까지 동작하는 상태`


## 10. 이후 수정 예정

- 프론트 현재 화면 흐름 반영
- 실제 백엔드 폴더 구조 반영
- API 명세 문서와 연결
- 최종 DB 수정안 반영
