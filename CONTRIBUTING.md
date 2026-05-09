# DSCD 협업 규칙

## 1. 저장소 구조
프로젝트는 아래 6개 최상위 폴더를 기준으로 관리한다.

```text
2026-1-DSCD/
├─ frontend/    # 화면(UI)
├─ backend/     # API, 인증, DB 접근, 모델 호출
├─ model/       # 피부분석/추천/루틴 AI 로직
├─ db/          # 스키마, 적재 SQL, 마이그레이션
├─ docs/        # 문서(명세, 회의결정, 규칙)
└─ scripts/     # 실행 스크립트
```

### 파일 배치 원칙
- `frontend`: 페이지/컴포넌트/프론트 API 호출 코드만 둔다.
- `backend`: 라우터, 서비스, 레포지토리, DB 세션 코드를 둔다.
- `model`: 학습/추론/임베딩/재랭킹 등 AI 코드를 둔다.
- `db`: `schema.sql`, `load_*.sql`, `migration_*.sql`를 둔다.
- `docs`: API 명세, 아키텍처, 규칙 문서를 둔다.
- `scripts`: 실행용 스크립트만 둔다.

### 현재 코드 이동 기준
- `3_Recomendation/Embedding_DB.py` -> `model/retrieval/embedding_db.py`
- `4_DB/schema_ver2.sql` -> `db/schema_ver2.sql`

## 2. 브랜치 전략

### 고정 브랜치
- `main`: 배포 가능한 안정 브랜치
- `develop`: 개발 통합 브랜치

### 작업 브랜치
- `feature/*`: 기능 개발
- `fix/*`: 버그 수정
- `docs/*`: 문서 수정
- `chore/*`: 설정/자동화/잡무

### 브랜치 이름 규칙
- 형식: `타입/기능명`
- 담당자 이름/이니셜은 브랜치명에 넣지 않는다.

### 브랜치 이름 예시
- `feature/model-retrieval-topk`
- `feature/backend-recommendation-api`
- `feature/frontend-skin-report`
- `fix/backend-empty-result`
- `docs/api-spec-update`

## 3. 개발 흐름
1. 이슈 생성 또는 기존 이슈 선택
2. `develop`에서 작업 브랜치 생성
3. 코드 수정 및 커밋
4. `develop` 대상으로 PR 생성
5. 리뷰 승인 + CI 통과 확인
6. 머지

## 4. 머지/리뷰 규칙
- `main`, `develop` 직접 push 금지
- PR로만 머지
- 최소 1명 승인 필수
- CI 실패 시 머지 금지
- PR 본문에 관련 이슈 번호 기재 (`Closes #번호`)

## 5. 커밋 메시지 규칙
아래 형식을 사용한다.

```text
type(scope): message
```

### type 목록
- `feat`: 기능 추가
- `fix`: 버그 수정
- `refactor`: 리팩토링
- `docs`: 문서
- `test`: 테스트
- `chore`: 기타 설정/관리

### type 사용 기준
- `feat`: 사용자 관점에서 새로운 기능이 생길 때 사용한다.
  예) 신규 API 추가, 추천 로직 신규 도입, 화면 기능 추가
- `fix`: 기존 기능의 오동작/오류를 수정할 때 사용한다.
  예) 빈 결과 처리 버그, 예외 처리 누락 수정
- `refactor`: 기능 변경 없이 코드 구조/위치/가독성을 개선할 때 사용한다.
  예) 파일 이동, 함수 분리, 중복 코드 정리
- `docs`: 문서만 수정할 때 사용한다.
  예) README, API 명세, 협업 규칙 업데이트
- `test`: 테스트 코드/테스트 설정만 변경할 때 사용한다.
  예) 단위 테스트 추가, 테스트 케이스 보강
- `chore`: 빌드/설정/의존성/자동화 등 유지보수성 작업에 사용한다.
  예) CI 설정, `.gitignore` 수정, 패키지 버전 업데이트

### 예시
- `feat(model): add retrieval top-k pipeline`
- `fix(backend): handle empty analysis result`
- `docs(project): update branch policy`

## 6. PR 체크리스트
- 변경 목적이 명확한가?
- 테스트 방법/결과를 적었는가?
- API/DB 변경 시 문서를 함께 수정했는가?
- 비밀정보(.env, 키, 비밀번호)를 커밋하지 않았는가?

## 7. 데이터/보안 규칙
- `.env` 실제 값 커밋 금지
- 대용량 산출물(`model/artifacts`, raw 데이터)은 Git 추적 금지
- 개인정보 포함 샘플은 마스킹 후 사용

## 8. 역할 경계
- `frontend`: 화면/사용자 입력/결과 표시
- `backend`: 인증/비즈니스 로직/DB/모델 API 연동
- `model`: 분석/추천/루틴 계산
- `db`: 스키마 및 데이터 적재 관리

경계가 애매하면 `backend`가 조정하고, 모델 로직은 `model`로 유지한다.

