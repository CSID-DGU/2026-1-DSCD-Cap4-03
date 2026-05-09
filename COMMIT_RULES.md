# GitHub 커밋 규칙

## 1. 커밋 메시지 형식
아래 형식을 기본으로 사용한다.

```text
type(scope): message
```

- `type`: 작업 성격
- `scope`: 작업 영역(선택)
- `message`: 변경 요약 (한 줄)

예시:
- `feat(model): 임베딩 추천 로직 추가`
- `fix(backend): 빈 결과 처리 오류 수정`
- `docs(project): 커밋 규칙 문서 추가`

## 2. type 정의
- `feat`: 기능 추가
- `fix`: 버그 수정
- `refactor`: 기능 변화 없는 구조 개선
- `docs`: 문서 변경
- `test`: 테스트 코드/케이스 추가 및 수정
- `chore`: 설정, 의존성, CI, 기타 관리 작업

## 3. scope 권장값
- `model`
- `backend`
- `frontend`
- `db`
- `docs`
- `project`

예시:
- `feat(model): graph rag 후보 점수 계산 추가`
- `fix(db): schema load 쿼리 오타 수정`

## 4. 작성 규칙
- 제목은 50자 내외로 간결하게 작성
- 한 커밋에는 하나의 목적만 담기
- 기능/버그/문서를 한 커밋에 섞지 않기
- "수정", "업데이트" 같은 모호한 문구만 쓰지 않기

나쁜 예:
- `update`
- `수정`
- `feat: 이것저것 변경`

좋은 예:
- `feat(model): 추천 top-k 저장 컬럼 추가`
- `fix(backend): 사용자 이미지 없는 경우 400 응답`

## 5. 커밋 단위 기준
- PR 기준이 아니라 "의미 있는 변경 단위"로 커밋
- 파일 이동/이름 변경은 가능하면 별도 커밋
- 대규모 리팩토링은 기능 변경과 분리해서 커밋

## 6. 브랜치와 커밋 연결
- 브랜치: `feature/...`, `fix/...`, `docs/...`
- 커밋: 브랜치 목적과 일치해야 함

예시:
- 브랜치: `feature/model-recommendation`
- 커밋:
  - `feat(model): 임베딩 검색 파이프라인 추가`
  - `refactor(model): retrieval 함수 분리`

## 7. 추천 워크플로우
1. 브랜치 생성
2. 작업 후 `git status`로 변경 확인
3. 관련 파일만 `git add`
4. 규칙에 맞는 메시지로 `git commit`
5. `git push` 후 PR 생성

## 8. 자주 쓰는 커밋 메시지 예시
- `feat(model): Embedding 코드 추가`
- `feat(model): graph rag retriever 초안 추가`
- `fix(model): 코사인 유사도 계산 버그 수정`
- `refactor(model): config 상수 모듈로 이동`
- `docs(project): CONTRIBUTING 브랜치 규칙 업데이트`
- `chore(project): gitignore 정리`
