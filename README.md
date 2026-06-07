<h1>2026-1-DSCD-Cap4-3</h1>

<h2>ROUPLE</h2>

> AI 피부 분석 기반 개인 맞춤형 스킨케어 루틴 추천 시스템  
> AI Skin Analysis-Based Personalized Skincare Routine Recommendation System

**ROUPLE**은 사용자의 얼굴 이미지를 분석해 피부 상태를 진단하고, 개인 피부 지표와 사용자 프로필을 기반으로 맞춤형 스킨케어 루틴을 추천하는 AI 기반 스킨케어 서비스입니다.

---

## 팀 정보

| 팀원 | 이름 | 학과 |
| --- | --- |
| 팀장 | 현민영 | 통계학과 |
| 팀원 | 이서은 | 통계학과 |
| 팀원 | 심고은 | 통계학과 |
| 팀원 | 서동현 | 산업시스템공학과 |

## 시연 영상

[ROUPLE 시연 영상 바로가기](https://www.youtube.com/watch?v=Y65TdlWhK5Q)



---

## 서비스 개요

기존 화장품 추천 서비스는 사용자의 피부 상태를 정량적으로 반영하기 어렵고, 추천 이유나 루틴 구성 근거가 부족하다는 한계가 있습니다. ROUPLE은 얼굴 이미지 기반 피부 분석, 화장품 성분 지식그래프, 임베딩 기반 후보 검색, LLM 설명 생성을 결합하여 사용자에게 더 개인화된 스킨케어 경험을 제공합니다.

## 주요 기능

| 기능 | 설명 |
| --- | --- |
| 스킨 리포트 | 얼굴 이미지를 분석해 6개 피부 지표를 산출하고 AI 피부 리포트를 제공합니다. |
| 루틴 리포트 | 피부 분석 결과, 사용자 프로필, 예산 조건을 반영해 Best/Value 루틴을 추천합니다. |
| 화장대 리포트 | 사용자가 보유한 화장품의 피부 적합도를 분석하고, 부족한 루틴 단계를 보완 추천합니다. |
| 제품 탐색 | 카테고리별 화장품을 탐색하고 위시리스트 및 내 화장대에 등록할 수 있습니다. |

## 핵심 모델 및 알고리즘

### 1. 피부 분석 모델

- 얼굴 이미지와 부위별 crop 이미지를 함께 활용
- Swin-Tiny Transformer 기반 멀티태스크 모델
- 여드름, 건조, 처짐, 모공, 색소침착, 주름의 6개 피부 지표 예측

### 2. 스킨케어 루틴 추천

- BGE-M3 임베딩 기반 카테고리별 Top-20 후보 제품 추출
- Neo4j Knowledge Graph 기반 피부 고민-기능-성분-제품 관계 재정렬
- Rule/SMILES 기반 성분 충돌 검사
- Beam Search 기반 단계별 루틴 조합 생성

### 3. 내 화장대 기반 추천

- 보유 화장품과 최신 피부 분석 결과를 기반으로 Skin Match 점수 산출
- 적합한 보유 제품은 고정 제품으로 활용
- 부족한 루틴 단계만 신규 제품으로 보완 추천
- LLM을 활용해 제품별 적합 이유와 사용 가이드를 생성

---

## 서비스 흐름

```text
사용자 얼굴 이미지
        ↓
S3 이미지 업로드
        ↓
피부 분석 모델
        ↓
6개 피부 지표 산출
        ↓
임베딩 기반 후보 제품 검색
        ↓
Knowledge Graph 기반 재정렬
        ↓
Rule Filtering + Beam Search
        ↓
맞춤형 스킨케어 루틴 추천
        ↓
LLM 기반 설명 생성
```

```text
내 화장대 제품 + 최신 피부 분석 결과
        ↓
Skin Match 점수 산출
        ↓
보유 제품 적합성 판단
        ↓
고정 제품 및 부족 단계 탐지
        ↓
부족 단계 보완 추천
        ↓
내 화장대 기반 루틴 생성
```

---

## 기술 스택

| 구분 | 기술 |
| --- | --- |
| Frontend | React, Vite, TypeScript |
| Backend | FastAPI, SQLAlchemy |
| Database | MySQL |
| Knowledge Graph | Neo4j |
| Storage | AWS S3 |
| Deployment | Docker, Docker Compose, AWS EC2 |
| Skin Analysis | Swin-Tiny Transformer |
| Recommendation | BGE-M3, Knowledge Graph, Beam Search |
| LLM | 피부 리포트, 루틴 설명, 화장대 설명 생성 |

## 시스템 구성

```text
frontend/   React 기반 사용자 화면
backend/    FastAPI 기반 API 서버
model/      피부 분석, 추천, 내 화장대, LLM 모델 로직
DB/         로컬 데이터 적재 및 전처리 리소스
scripts/    배포 및 시각화 보조 스크립트
```

---

## 로컬 실행 방법

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev -- --host 0.0.0.0
```

---

## Docker 배포

### 이미지 빌드 및 푸시

```powershell
docker build -f backend/Dockerfile -t myhyun1123/rouple-backend:latest .
docker push myhyun1123/rouple-backend:latest

docker build -f frontend/Dockerfile -t myhyun1123/rouple-frontend:latest .
docker push myhyun1123/rouple-frontend:latest
```

### EC2 배포

```bash
cd ~/rouple-deploy
docker compose pull
docker compose up -d
docker compose ps
```

---

## LLM 캐싱

LLM 결과는 재조회 시 비용과 지연을 줄이기 위해 JSON 파일로 저장합니다.

- 피부 분석 요약
- 루틴 추천 설명
- 내 화장대 제품별 적합성 설명
- 내 화장대 기반 루틴 설명

---

## 보안 및 주의사항

다음 파일은 Git에 커밋하지 않습니다.

- `.env`
- AWS Access Key
- PEM 키 파일
- DB dump 파일
- 모델 가중치 및 대용량 임베딩 산출물
- LLM 캐시 JSON
- 로컬 전용 `schema.sql`
