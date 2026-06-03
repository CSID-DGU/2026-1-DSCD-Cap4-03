import { useEffect, useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { vanityApi, type VanityProduct } from '../api/vanity';
import { productApi } from '../api/product';
import { FlaskConical, Check, ChevronLeft, Clock, Package, AlertTriangle } from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import './VanityRoutinePage.css';
import './BudgetPage.css';

const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼', 'Essences/Ampoules/Serums': '에센스/앰플/세럼',
  'Cream/Gel': '크림/젤', 'Eye Treatments': '아이크림',
  'Balms/Multi-balms': '멀티밤', 'Facial Oils': '페이셜 오일', 'Face Mists': '미스트',
};

type Category = '전체' | '토너' | '에멀젼' | '앰플' | '크림';

const PRODUCT_CAT_TO_BUDGET: Record<string, Category> = {
  'Toner': '토너', 'Toner Pads': '토너',
  'Emulsions': '에멀젼',
  'Essences/Ampoules/Serums': '앰플',
  'Cream/Gel': '크림',
};

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
  { label: '0원',    value: 0      },
  { label: '5만원',  value: 50000  },
  { label: '10만원', value: 100000 },
  { label: '20만원', value: 200000 },
  { label: 'MAX',    value: null   },
];

const STEPS: Record<Category, Step[]> = {
  전체: TOTAL_STEPS, 토너: INDIVIDUAL_STEPS, 에멀젼: INDIVIDUAL_STEPS,
  앰플: INDIVIDUAL_STEPS, 크림: INDIVIDUAL_STEPS,
};

function StepSlider({ steps, lo, hi, onChange, disabled }: {
  steps: Step[];
  lo: number;
  hi: number;
  onChange: (lo: number, hi: number) => void;
  disabled?: boolean;
}) {
  const last = steps.length - 1;
  const pct = (i: number) => `${(i / last) * 100}%`;
  const trackRef = useRef<HTMLDivElement>(null);
  const stateRef = useRef({ lo, hi, onChange, dragging: null as 'lo' | 'hi' | null });

  useEffect(() => { stateRef.current.lo = lo; }, [lo]);
  useEffect(() => { stateRef.current.hi = hi; }, [hi]);
  useEffect(() => { stateRef.current.onChange = onChange; }, [onChange]);

  useEffect(() => {
    if (disabled) return;
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
  }, [last, disabled]);

  const startDrag = (handle: 'lo' | 'hi') => (e: React.MouseEvent) => {
    if (disabled) return;
    e.preventDefault();
    stateRef.current.dragging = handle;
  };

  const handleClick = (i: number) => {
    if (disabled || stateRef.current.dragging !== null) return;
    const { lo, hi, onChange } = stateRef.current;
    const dLo = Math.abs(i - lo);
    const dHi = Math.abs(i - hi);
    if (dLo <= dHi) onChange(Math.min(i, hi), hi);
    else onChange(lo, Math.max(i, lo));
  };

  return (
    <div className="bgt-slider" style={{ userSelect: 'none', opacity: disabled ? 0.35 : 1 }}>
      <div className="bgt-track-area" ref={trackRef}>
        <div className="bgt-track" />
        <div className="bgt-fill" style={{ left: pct(lo), width: `calc(${pct(hi)} - ${pct(lo)})` }} />
        {steps.map((_, i) => {
          const isLo = i === lo, isHi = i === hi, inRange = i > lo && i < hi;
          return (
            <button key={i} type="button"
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
          <button key={i} type="button"
            className={`bgt-tick-label${i === lo || i === hi ? ' active' : ''}${i > lo && i < hi ? ' in-range' : ''}`}
            onClick={() => handleClick(i)}>
            {step.label}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function VanityRoutineBudgetPage() {
  const navigate = useNavigate();

  const [products, setProducts]       = useState<VanityProduct[]>([]);
  const [selected, setSelected]       = useState<Set<number>>(new Set());
  const [hasAnalysis, setHasAnalysis] = useState(false);
  const [imageMap, setImageMap]       = useState<Record<number, string>>({});
  const [loading, setLoading]         = useState(true);

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
    const min = steps[lo].value;          // lo=0이면 0 (null 아님) — 백엔드에 0 전송
    const max = steps[hi].value;          // hi=last(MAX)이면 null
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

  useEffect(() => {
    const load = async () => {
      try {
        const [productsRes, summaryRes] = await Promise.all([
          vanityApi.getProducts(),
          vanityApi.getSummary(),
        ]);
        setProducts(productsRes.products);
        setHasAnalysis(!!summaryRes.basis_skin_result);
      } catch { /* no-op */ }
      setLoading(false);
    };
    load();
  }, []);

  useEffect(() => {
    productApi.getList()
      .then(prods => {
        const map: Record<number, string> = {};
        prods.forEach(p => { if (p.image_url) map[p.product_id] = p.image_url; });
        setImageMap(map);
      })
      .catch(() => {});
  }, []);

  const selectedProducts   = products.filter(p => selected.has(p.product_id));
  const categoryCounts     = selectedProducts.reduce<Record<string, number>>((acc, p) => {
    acc[p.category] = (acc[p.category] ?? 0) + 1; return acc;
  }, {});
  const lockedBudgetCats = new Set<Category>(
    selectedProducts
      .map(p => PRODUCT_CAT_TO_BUDGET[p.category])
      .filter((c): c is Category => !!c)
  );
  const hasConflict        = Object.values(categoryCounts).some(c => c > 1);
  const conflictCategories = Object.entries(categoryCounts)
    .filter(([, cnt]) => cnt > 1).map(([cat]) => CATEGORY_KO[cat] ?? cat);

  const toggleSelect = (id: number) => {
    setSelected(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };

  const handleRun = () => {
    if (selected.size === 0 || hasConflict) return;
    const 전체 = getMinMax('전체'), 토너 = getMinMax('토너');
    const 에멀젼 = getMinMax('에멀젼'), 앰플 = getMinMax('앰플'), 크림 = getMinMax('크림');
    navigate('/loading', {
      state: {
        type: 'vanity_routine',
        fixed_product_ids: [...selected],
        vanity_budget: {
          budget_min:   전체.min,   budget_max:   전체.max,
          toner_min:    토너.min,   toner_max:    토너.max,
          emulsion_min: 에멀젼.min, emulsion_max: 에멀젼.max,
          ampoule_min:  앰플.min,  ampoule_max:  앰플.max,
          cream_min:    크림.min,   cream_max:    크림.max,
        },
      },
    });
  };

  if (loading) return (
    <div className="vr-page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
      <LoadingSpinner text="화장대 정보를 불러오는 중이에요" />
    </div>
  );

  return (
    <div className="vr-page">
      <div className="vr-banner">
        <div className="vr-banner-inner">
          <button
            style={{ background: 'none', border: 'none', color: '#7c3aed', fontSize: 13, fontWeight: 700, cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4, padding: 0, marginBottom: 16, fontFamily: 'Noto Sans KR, sans-serif' }}
            onClick={() => navigate('/vanity')}>
            <ChevronLeft size={15} />내 화장대로
          </button>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
            <div>
              <div className="vr-badge">SKIN FIT · ROUTINE</div>
              <h1 className="vr-title">내 화장대로 루틴을 완성할 수 있을까요?</h1>
              <p className="vr-sub">쓰던 제품은 유지하고, 비어 있는 단계는 AI가 최적의 제품으로 채워드려요</p>
            </div>
            <button
              style={{ background: 'none', border: '1.5px solid #7c3aed', color: '#7c3aed', borderRadius: 8, padding: '8px 16px', fontSize: 12, fontWeight: 700, cursor: 'pointer', fontFamily: 'Noto Sans KR, sans-serif', display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0, marginTop: 8 }}
              onClick={() => navigate('/vanity/routine/history')}>
              <Clock size={13} />루틴 기록
            </button>
          </div>
        </div>
      </div>

      <div className="vr-body">

        {!hasAnalysis && (
          <div className="vr-no-analysis">
            ⚠️ 피부 분석 결과가 없어 화장대 기반 루틴 추천을 진행할 수 없어요.<br />
            먼저 피부 분석을 진행해주세요.
            <div style={{ marginTop: 12 }}>
              <button onClick={() => navigate('/diagnosis')}
                style={{ background: '#d97706', color: '#fff', border: 'none', borderRadius: 8, padding: '8px 20px', fontWeight: 700, cursor: 'pointer', fontFamily: 'Noto Sans KR, sans-serif' }}>
                피부 진단 받기 →
              </button>
            </div>
          </div>
        )}

        {hasAnalysis && (
          <div className="vr-basis-bar">
            <FlaskConical size={15} color="#7c3aed" />
            <span className="vr-basis-text">가장 최신 피부 분석 결과를 기준으로 추천해드려요</span>
          </div>
        )}

        {hasAnalysis && (
          <div className="vr-section">
            <div className="vr-section-title">STEP 1 · 고정할 제품 선택</div>
            <div className="vr-section-sub">루틴에 유지하고 싶은 제품을 선택하세요. 카테고리별 1개만 선택 가능하며, 최소 1개 이상 선택해야 해요.</div>
            {products.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '24px', color: '#9CA3AF', fontSize: 14 }}>
                화장대에 등록된 제품이 없어요.
                <div style={{ marginTop: 12 }}>
                  <button onClick={() => navigate('/vanity')}
                    style={{ background: '#7c3aed', color: '#fff', border: 'none', borderRadius: 8, padding: '8px 20px', fontWeight: 700, cursor: 'pointer', fontFamily: 'Noto Sans KR, sans-serif' }}>
                    내 화장대에서 제품 추가하기
                  </button>
                </div>
              </div>
            ) : (
              <>
                <div className="vr-select-grid">
                  {products.map(p => {
                    const isSel = selected.has(p.product_id);
                    const isCon = isSel && (categoryCounts[p.category] ?? 0) > 1;
                    return (
                      <div key={p.product_id}
                        className={`vr-select-item ${isSel ? 'selected' : ''} ${isCon ? 'conflict' : ''}`}
                        onClick={() => toggleSelect(p.product_id)}>
                        {imageMap[p.product_id] || p.image_url
                          ? <img src={imageMap[p.product_id] || p.image_url} alt={p.product_name} />
                          : <div className="vr-img-fallback"><Package size={22} color="#A78BFA" /></div>}
                        <div className="vr-select-item-brand">{p.brand_name}</div>
                        <div className="vr-select-item-name">{p.product_name}</div>
                        <div className="vr-select-item-cat">{CATEGORY_KO[p.category] ?? p.category}</div>
                        {isSel && !isCon && <div className="vr-select-check"><Check size={11} color="#fff" /></div>}
                      </div>
                    );
                  })}
                </div>
                {hasConflict && (
                  <div className="vr-conflict-warn">
                    <AlertTriangle size={14} />
                    {conflictCategories.join(', ')} 카테고리에서 2개 이상 선택됐어요.
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {hasAnalysis && (
          <div className="vr-section">
            <div className="vr-section-title">STEP 2 · 보완 제품 예산 설정</div>
            <div className="vr-section-sub">빈 루틴 단계를 채울 새 제품의 예산 범위를 설정하세요. 건너뛰어도 괜찮아요.</div>

            {/* 전체 예산 — 풀 width */}
            <div className="vrb-budget-card vrb-budget-card--full">
              <div className="bgt-head">
                <span className="vrb-cat-label">전체 예산</span>
                <span className={`bgt-val-chip${getChipLabel('전체') !== '설정 안함' ? ' set' : ''}`}>{getChipLabel('전체')}</span>
              </div>
              <StepSlider steps={STEPS['전체']} lo={rangeIdx['전체'].lo} hi={rangeIdx['전체'].hi}
                onChange={(lo, hi) => setRange('전체', lo, hi)} />
            </div>

            {/* 카테고리별 2열 */}
            <div className="vrb-budget-grid">
              {((['토너', '에멀젼', '앰플', '크림'] as Category[])).map(cat => {
                const locked = lockedBudgetCats.has(cat);
                return (
                  <div key={cat} className="vrb-budget-card">
                    <div className="bgt-head">
                      <span className="vrb-cat-label">
                        {cat}
                        {locked && <span className="vrb-locked-badge">고정됨</span>}
                      </span>
                      {!locked && (
                        <span className={`bgt-val-chip${getChipLabel(cat) !== '설정 안함' ? ' set' : ''}`}>{getChipLabel(cat)}</span>
                      )}
                    </div>
                    {locked
                      ? <p className="vrb-locked-msg">고정 제품 카테고리는 예산 설정 불필요</p>
                      : <StepSlider steps={STEPS[cat]} lo={rangeIdx[cat].lo} hi={rangeIdx[cat].hi}
                          onChange={(lo, hi) => setRange(cat, lo, hi)} />
                    }
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {hasAnalysis && products.length > 0 && (
          <div className="vr-run-wrap">
            <button className="vr-run-btn" onClick={handleRun} disabled={hasConflict || selected.size === 0}>
              화장대 루틴 추천받기 →
            </button>
            {selected.size === 0 && (
              <span style={{ fontSize: 12, color: '#9CA3AF' }}>고정할 제품을 최소 1개 이상 선택해주세요</span>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
