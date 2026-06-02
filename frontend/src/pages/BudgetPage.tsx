import { useState, useRef, useEffect, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Wallet, Droplets, Leaf, Sparkles, Package } from 'lucide-react';
import './BudgetPage.css';

type Category = '전체' | '토너' | '에멀젼' | '앰플' | '크림';
type Step = { label: string; value: number | null };
type RangeIdx = { lo: number; hi: number };

const INDIVIDUAL_STEPS: Step[] = [
  { label: '0원',   value: 0     },
  { label: '1만원', value: 10000 },
  { label: '3만원', value: 30000 },
  { label: '5만원', value: 50000 },
  { label: 'MAX',   value: null  },
];

const TOTAL_STEPS: Step[] = [
  { label: '0원',   value: 0      },
  { label: '5만원', value: 50000  },
  { label: '10만원', value: 100000 },
  { label: '20만원', value: 200000 },
  { label: 'MAX',   value: null   },
];

const STEPS: Record<Category, Step[]> = {
  전체: TOTAL_STEPS,
  토너: INDIVIDUAL_STEPS,
  에멀젼: INDIVIDUAL_STEPS,
  앰플: INDIVIDUAL_STEPS,
  크림: INDIVIDUAL_STEPS,
};

function StepSlider({ steps, lo, hi, onChange }: {
  steps: Step[];
  lo: number;
  hi: number;
  onChange: (lo: number, hi: number) => void;
}) {
  const last = steps.length - 1;
  const pct = (i: number) => `${(i / last) * 100}%`;

  const trackRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef({ lo, hi, onChange, dragging: null as 'lo' | 'hi' | null });

  useEffect(() => { stateRef.current.lo = lo; }, [lo]);
  useEffect(() => { stateRef.current.hi = hi; }, [hi]);
  useEffect(() => { stateRef.current.onChange = onChange; }, [onChange]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const { dragging, lo, hi, onChange } = stateRef.current;
      if (!dragging || !trackRef.current) return;
      const { left, width } = trackRef.current.getBoundingClientRect();
      const ratio = Math.max(0, Math.min(1, (e.clientX - left) / width));
      const step = Math.round(ratio * last);
      if (dragging === 'lo') onChange(Math.min(step, hi), hi);
      else onChange(lo, Math.max(step, lo));
    };
    const onUp = () => { stateRef.current.dragging = null; };
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, [last]);

  const startDrag = (handle: 'lo' | 'hi') => (e: React.MouseEvent) => {
    e.preventDefault();
    stateRef.current.dragging = handle;
  };

  const handleClick = (i: number) => {
    if (stateRef.current.dragging !== null) return;
    const { lo, hi, onChange } = stateRef.current;
    const dLo = Math.abs(i - lo);
    const dHi = Math.abs(i - hi);
    if (dLo <= dHi) onChange(Math.min(i, hi), hi);
    else onChange(lo, Math.max(i, lo));
  };

  return (
    <div className="bgt-slider" style={{ userSelect: 'none' }}>
      <div className="bgt-track-area" ref={trackRef}>
        <div className="bgt-track" />
        <div
          className="bgt-fill"
          style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }}
        />
        {steps.map((_, i) => {
          const isLo = i === lo;
          const isHi = i === hi;
          const inRange = i > lo && i < hi;
          return (
            <button
              key={i}
              type="button"
              className={`bgt-dot${inRange ? ' in-range' : ''}${isLo ? ' handle handle-lo' : ''}${isHi ? ' handle handle-hi' : ''}`}
              style={{ left: pct(i) }}
              onMouseDown={isLo ? startDrag('lo') : isHi ? startDrag('hi') : undefined}
              onClick={() => handleClick(i)}
            />
          );
        })}
      </div>
      <div className="bgt-labels-row">
        {steps.map((step, i) => (
          <button
            key={i}
            type="button"
            className={`bgt-tick-label${i === lo || i === hi ? ' active' : ''}${i > lo && i < hi ? ' in-range' : ''}`}
            onClick={() => handleClick(i)}
          >
            {step.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function BudgetPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const resultId = location.state?.resultId;
  const imageId = location.state?.imageId;

  const initRange = (cat: Category): RangeIdx => ({ lo: 0, hi: STEPS[cat].length - 1 });
  const [rangeIdx, setRangeIdx] = useState<Record<Category, RangeIdx>>({
    전체: initRange('전체'), 토너: initRange('토너'), 에멀젼: initRange('에멀젼'),
    앰플: initRange('앰플'), 크림: initRange('크림'),
  });
  const setRange = (cat: Category, lo: number, hi: number) =>
    setRangeIdx((p) => ({ ...p, [cat]: { lo, hi } }));

  const getMinMax = (cat: Category) => {
    const { lo, hi } = rangeIdx[cat];
    const steps = STEPS[cat];
    if (lo === 0 && hi === steps.length - 1) return { min: null, max: null };
    const min = lo === 0 ? null : steps[lo].value;
    const max = hi === steps.length - 1 ? null : steps[hi].value;
    return { min, max };
  };

  const getChipLabel = (cat: Category) => {
    const { lo, hi } = rangeIdx[cat];
    const steps = STEPS[cat];
    if (lo === 0 && hi === steps.length - 1) return '설정 안함';
    if (lo === 0) return `${steps[hi].label} 이하`;
    if (hi === steps.length - 1) return `${steps[lo].label} 이상`;
    return `${steps[lo].label} ~ ${steps[hi].label}`;
  };

  const handleNext = () => {
    const b = (cat: Category) => getMinMax(cat);
    navigate('/loading', {
      state: {
        type: 'routine', resultId, imageId,
        budget: {
          total_min:    b('전체').min,   total_max:    b('전체').max,
          toner_min:    b('토너').min,   toner_max:    b('토너').max,
          emulsion_min: b('에멀젼').min, emulsion_max: b('에멀젼').max,
          ampoule_min:  b('앰플').min,  ampoule_max:  b('앰플').max,
          cream_min:    b('크림').min,   cream_max:    b('크림').max,
        },
      },
    });
  };

  const CAT_ICONS: Record<Category, ReactNode> = {
    전체: <Wallet size={15} />, 토너: <Droplets size={15} />, 에멀젼: <Leaf size={15} />,
    앰플: <Sparkles size={15} />, 크림: <Package size={15} />,
  };

  return (
    <div className="budget-page">
      <div className="budget-banner">
        <div className="budget-banner-inner">
          <div className="budget-banner-badge">ROUTINE SETTING</div>
          <h1 className="budget-title">어느 정도 예산을 생각하고 계세요?</h1>
          <p className="budget-sub">카테고리별 예산 범위를 설정하면 그 안에서 최적의 제품을 추천해드려요. 건너뛰어도 괜찮아요.</p>
        </div>
      </div>

      <div className="budget-body">
        <div className="budget-section budget-section--full">
          <div className="bgt-head">
            <div className="bgt-cat-name">{CAT_ICONS['전체']} <span>전체 예산</span></div>
            <span className={`bgt-val-chip${getChipLabel('전체') !== '설정 안함' ? ' set' : ''}`}>{getChipLabel('전체')}</span>
          </div>
          <StepSlider steps={STEPS['전체']} lo={rangeIdx['전체'].lo} hi={rangeIdx['전체'].hi}
            onChange={(lo, hi) => setRange('전체', lo, hi)} />
        </div>

        {(['토너', '에멀젼', '앰플', '크림'] as Category[]).map((cat) => (
          <div className="budget-section" key={cat}>
            <div className="bgt-head">
              <div className="bgt-cat-name">{CAT_ICONS[cat]} <span>{cat}</span></div>
              <span className={`bgt-val-chip${getChipLabel(cat) !== '설정 안함' ? ' set' : ''}`}>{getChipLabel(cat)}</span>
            </div>
            <StepSlider steps={STEPS[cat]} lo={rangeIdx[cat].lo} hi={rangeIdx[cat].hi}
              onChange={(lo, hi) => setRange(cat, lo, hi)} />
          </div>
        ))}

        <div className="budget-submit-wrap">
          <button className="btn-budget-next" onClick={handleNext}>루틴 추천 받기 →</button>
        </div>
      </div>
    </div>
  );
}
