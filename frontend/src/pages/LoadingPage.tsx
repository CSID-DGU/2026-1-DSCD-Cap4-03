import { useEffect, useState, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { analysisApi } from '../api/analysis';
import { routineApi } from '../api/routine';
import { vanityApi } from '../api/vanity';
import { Microscope, Sparkles, CheckCircle } from 'lucide-react';
import './LoadingPage.css';

type LoadingType = 'analysis' | 'routine' | 'vanity_routine';

interface StepConfig {
  message: string;
  detail: string;
}

const STEPS: Record<LoadingType, StepConfig[]> = {
  analysis: [
    { message: '피부 사진 전송 중',    detail: '이미지를 서버로 보내고 있어요' },
    { message: 'AI 피부 분석 중',      detail: '피부 상태를 6가지 지표로 측정하고 있어요' },
    { message: 'AI 코멘트 작성 중',    detail: '분석 결과를 바탕으로 설명을 생성하고 있어요' },
  ],
  routine: [
    { message: '피부 분석 결과 확인 중', detail: '내 피부 상태를 불러오고 있어요' },
    { message: '맞춤 제품 탐색 중',     detail: '수천 개 제품 중 내 피부에 맞는 것을 찾고 있어요' },
    { message: '최적 루틴 구성 중',     detail: '예산과 피부 타입을 고려해 루틴을 조합하고 있어요' },
    { message: 'AI 추천 이유 작성 중',  detail: '루틴 설명과 사용법을 작성하고 있어요' },
  ],
  vanity_routine: [
    { message: '화장대 제품 확인 중',    detail: '내 화장대 제품을 불러오고 있어요' },
    { message: '맞춤 루틴 구성 중',      detail: '내 피부에 맞는 최적 루틴을 조합하고 있어요' },
    { message: 'AI 루틴 코멘트 생성 중', detail: '맞춤형 루틴 설명을 작성하고 있어요' },
  ],
};

const META: Record<LoadingType, { IconEl: ReactNode; title: string; doneTitle: string }> = {
  analysis:       { IconEl: <Microscope size={32} color="#7c3aed" />, title: 'AI 피부 분석',       doneTitle: '분석 완료!' },
  routine:        { IconEl: <Sparkles   size={32} color="#7c3aed" />, title: '맞춤 루틴 생성',     doneTitle: '루틴 완성!' },
  vanity_routine: { IconEl: <Sparkles   size={32} color="#7c3aed" />, title: '화장대 루틴 생성',   doneTitle: '루틴 완성!' },
};

export default function LoadingPage() {
  const navigate = useNavigate();
  const location = useLocation();

  const type: LoadingType = location.state?.type ?? 'analysis';
  const steps = STEPS[type];
  const meta = META[type];

  // state를 effect 밖에서 추출 → 의존성 배열 경고 방지
  const imageId: number    = location.state?.image_id  ?? 1;
  const imageUrl: string   = location.state?.imageUrl  ?? '';
  const resultId: number   = location.state?.resultId  ?? 1;
  const routineImageId: number = location.state?.imageId ?? 1;
  const budget = location.state?.budget ?? {};
  const vanityRoutineBody = location.state?.vanity_routine_body ?? null;

  const [currentStep, setCurrentStep] = useState(0);
  const [doneSteps, setDoneSteps] = useState<number[]>([]);
  const [isDone, setIsDone] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const advance = (i: number) => setCurrentStep(i);
  const complete = (i: number) => setDoneSteps((prev) => [...prev, i]);

  useEffect(() => {
    let cancelled = false;

    const runAnalysis = async () => {

      // step 0: 전송 확인 (이미 DiagnosisPage에서 완료)
      advance(0);
      await delay(600);
      if (cancelled) return;
      complete(0);

      // step 1: 피부분석 모델 실행
      advance(1);
      const { result_id } = await analysisApi.run({ image_id: imageId });
      if (cancelled) return;
      complete(1);

      // step 2: LLM 요약 생성
      advance(2);
      await analysisApi.createSummary({ result_id });
      if (cancelled) return;
      complete(2);

      setIsDone(true);
      await delay(800);
      if (!cancelled) navigate('/analysis', { state: { result_id, imageUrl } });
    };

    const runRoutine = async () => {
      // step 0: 결과 확인
      advance(0);
      await delay(400);
      if (cancelled) return;
      complete(0);

      // step 1-2: 추천 모델 실행
      advance(1);
      const rec = await routineApi.recommend({
        result_id:       resultId,
        image_id:        routineImageId,
        total_budget_min:    budget.total_min    ?? null,
        total_budget_max:    budget.total_max    ?? null,
        toner_budget_min:    budget.toner_min    ?? null,
        toner_budget_max:    budget.toner_max    ?? null,
        emulsion_budget_min: budget.emulsion_min ?? null,
        emulsion_budget_max: budget.emulsion_max ?? null,
        ampoule_budget_min:  budget.ampoule_min  ?? null,
        ampoule_budget_max:  budget.ampoule_max  ?? null,
        cream_budget_min:    budget.cream_min    ?? null,
        cream_budget_max:    budget.cream_max    ?? null,
      });
      const { session_id } = rec;
      if (cancelled) return;
      complete(1);

      advance(2);
      await delay(400);
      if (cancelled) return;
      complete(2);

      // step 3: LLM 추천 이유 생성
      advance(3);
      const explanation = await routineApi.createExplanation({ session_id });
      if (cancelled) return;
      complete(3);

      setIsDone(true);
      await delay(800);
      if (!cancelled) navigate('/routine/result', {
        state: {
          session_id,
          resultId,
          imageId: routineImageId,
          explanationRoutines: explanation.routines ?? [],
          budgetFallbackApplied: rec.budget_fallback_applied,
          budgetMessage: rec.budget_message,
        },
      });
    };

    const runVanityRoutine = async () => {
      advance(0);
      await delay(400);
      if (cancelled) return;
      complete(0);

      advance(1);
      const result = await vanityApi.runRoutine(vanityRoutineBody);
      if (cancelled) return;
      complete(1);

      advance(2);
      await delay(600);
      if (cancelled) return;
      complete(2);

      setIsDone(true);
      await delay(800);
      if (!cancelled) navigate('/vanity/routine', { state: { result } });
    };

    const run = async () => {
      try {
        if (type === 'analysis') await runAnalysis();
        else if (type === 'vanity_routine') await runVanityRoutine();
        else await runRoutine();
      } catch (err) {
        if (!cancelled) setError((err as Error).message || '처리 중 오류가 발생했어요.');
      }
    };

    run();
    return () => { cancelled = true; };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="lp-page">
      <div className="lp-card">
        <div className={`lp-icon-wrap ${isDone ? 'done' : ''}`}>
          <span className="lp-icon">{isDone ? <CheckCircle size={32} color="#22c55e" /> : meta.IconEl}</span>
          {!isDone && !error && <div className="lp-spinner" />}
        </div>

        <h1 className="lp-title">
          {error ? '오류가 발생했어요' : isDone ? meta.doneTitle : meta.title}
        </h1>

        {!error && (
          <p className="lp-sub">
            {isDone ? '잠시 후 결과 페이지로 이동합니다' : '잠시만 기다려주세요. 1~2분 정도 소요됩니다!'}
          </p>
        )}

        {!error && (
          <div className="lp-steps">
            {steps.map((step, idx) => {
              const isDoneStep = doneSteps.includes(idx);
              const isActive   = currentStep === idx && !isDone;
              const isPending  = idx > currentStep && !isDone;
              return (
                <div key={idx} className={`lp-step ${isDoneStep ? 'done' : ''} ${isActive ? 'active' : ''} ${isPending ? 'pending' : ''}`}>
                  <div className="lp-step-icon">
                    {isDoneStep ? '✓' : isActive ? <span className="lp-dot-spin" /> : idx + 1}
                  </div>
                  <div className="lp-step-text">
                    <div className="lp-step-message">{step.message}</div>
                    {isActive && <div className="lp-step-detail">{step.detail}</div>}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {error && (
          <div className="lp-error">
            <p className="lp-error-text">{error}</p>
            <button className="lp-retry-btn" onClick={() => navigate(-1)}>← 이전으로 돌아가기</button>
          </div>
        )}
      </div>
    </div>
  );
}

function delay(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
