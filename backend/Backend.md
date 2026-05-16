# ROUPLE Backend

## 개요

- ROUPLE 백엔드 FastAPI 프로젝트 골격
- 프론트 연동용 Mock API 구현 완료 상태
- 현재 목표: Mock 저장소에서 실제 DB 구조로 전환 준비

## 현재 상태

- Swagger 기준 주요 API 테스트 완료
- 인증 / 사용자 정보 / 이미지 / 피부 분석 / 추천 / 찜 흐름 확인 완료
- 현재 일부 기능은 메모리 저장소 사용 중
- 사용자 / 프로필 / 알러지 / 이미지 저장부터 DB 전환 시작 상태

## 기술 선택

- ORM 방식: SQLAlchemy 사용
- MySQL 드라이버: PyMySQL 사용
- 이유: 모델 정의, 세션 관리, 이후 테이블 확장 및 관계 처리 용이

## 폴더 구조

```text
backend/
  app/
    main.py
    api/
    core/
    db/
    models/
    schemas/
    services/
    workflows/
    utils/
```

## DB 관련 추가 파일

- `app/core/config.py`: DB URL 설정
- `app/db/base.py`: SQLAlchemy Base
- `app/db/session.py`: engine / session 생성
- `app/models/user.py`: `USER`, `USER_PROFILE`, `USER_ALLERGY`, `USER_IMAGE` 모델
- `app/services/db_user.py`: 사용자 / 프로필 / 알러지 / 이미지 DB 로직

## 실행 방식

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## 개발용 DB 동작 방식

- 기본값: 로컬 SQLite 사용
- 목적: MySQL 없을 때도 백엔드 실행 유지
- 개발 DB 파일: `backend/rouple_dev.db`

## MySQL 전환 방식

`.env` 기준 설정 필요

```env
USE_LOCAL_SQLITE=false
MYSQL_HOST=127.0.0.1
MYSQL_PORT=3306
MYSQL_USER=root
MYSQL_PASSWORD=비밀번호
MYSQL_DB=Rouple_db
DB_AUTO_CREATE_TABLES=true
```

또는

```env
DATABASE_URL=mysql+pymysql://root:비밀번호@127.0.0.1:3306/Rouple_db?charset=utf8mb4
```

## schema.sql 반영 내용

- `USER.password` 길이 `VARCHAR(255)`로 확장
- `USER_IMAGE`에 아래 컬럼 추가
  - `s3_key`
  - `original_file_name`
  - `mime_type`
  - `file_size`
  - `crop_data`
  - `upload_status`

## 현재 DB 전환 완료 범위

- `POST /auth/signup`
- `POST /auth/login`
- `GET /users/me`
- `PATCH /users/me/profile`
- `PUT /users/me/allergies`
- `POST /images`

## S3 실제 연동 상태

- `POST /files/presign` 실제 boto3 presigned PUT URL 발급 방식 반영
- 필요 설정
  - `AWS_ACCESS_KEY_ID`
  - `AWS_SECRET_ACCESS_KEY`
  - `AWS_SESSION_TOKEN` 선택
  - `S3_BUCKET`
  - `S3_REGION`
  - `S3_PUBLIC_BASE_URL` 선택

## S3 사용 전 확인 사항

1. S3 버킷 CORS 설정 필요
2. 업로드된 이미지가 브라우저에서 보여야 하면 버킷 또는 해당 prefix의 읽기 정책 필요
3. 프론트는 `upload_url`로 `PUT` 업로드 수행 필요
4. 업로드 성공 후 `public_url`을 `POST /images`에 저장 필요

## 아직 Mock 유지 중인 범위

- 피부 분석 결과 저장
- 피부 분석 요약 저장
- 추천 세션 / 루틴 / 아이템 저장
- 찜 / 저장 루틴 / 분석 기록 일부 조회

## 다음 작업

1. MySQL 실제 연결 확인
2. `schema.sql`로 DB 생성
3. 회원 / 프로필 / 이미지 API DB 저장 테스트
4. 피부 분석 결과 테이블 ORM 추가
5. 추천 관련 테이블 ORM 추가
6. 메모리 저장소 제거 범위 확대

## 접속 정보

- Base URL: `http://127.0.0.1:8000`
- Swagger: `http://127.0.0.1:8000/docs`
- API 명세: `backend/API_SPEC.md`
