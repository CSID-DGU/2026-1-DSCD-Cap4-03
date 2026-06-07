import { useEffect, useState, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { routineApi, type RoutineItem } from '../api/routine';
import { analysisApi, type AnalysisResult } from '../api/analysis';
import { productApi, type ProductDetail } from '../api/product';
import { userApi } from '../api/user';
import { useAuth } from '../context/useAuth';
import {
  Sun, Moon, Trophy, Wallet, Bot, ClipboardList,
  AlertTriangle, FileText, Package, Timer, Banknote, Clock, Info,
} from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import './RoutinePage.css';

type RoutineTime = 'am' | 'pm' | 'both';

const TIME_META: Record<RoutineTime, { label: string; IconEl: ReactNode; color: string; bg: string; desc: string }> = {
  both: { label: 'AM + PM 공용', IconEl: <><Sun size={14} /><Moon size={14} /></>, color: '#7c3aed', bg: '#f5f3ff', desc: '아침·저녁 모두 사용하는 루틴이에요' },
  am:   { label: 'AM 전용',      IconEl: <Sun size={14} />,                        color: '#f59e0b', bg: '#fffbeb', desc: '아침에만 사용하는 루틴이에요' },
  pm:   { label: 'PM 전용',      IconEl: <Moon size={14} />,                       color: '#6366f1', bg: '#eef2ff', desc: '저녁에만 사용하는 루틴이에요' },
};

const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼', 'Essences/Ampoules/Serums': '에센스/앰플/세럼', 'Cream/Gel': '크림/젤',
  'Balms/Multi-balms': '멀티밤', 'Eye Treatments': '아이크림',
  'Facial Oils': '페이셜 오일', 'Shaving Products': '쉐이빙', 'All-In-One': '올인원',
  'Face Mists': '미스트',
};

const OPTIONAL_CATEGORIES = new Set([
  'Balms/Multi-balms', 'Eye Treatments', 'Facial Oils',
  'Shaving Products', 'All-In-One', 'Face Mists',
]);

const CATEGORY_DURATION: Record<string, { display: string; seconds: number }> = {
  'Toner':                    { display: '약 30초',      seconds: 30 },
  'Toner Pads':               { display: '약 30초',      seconds: 30 },
  'Emulsions':                { display: '약 30초~1분',  seconds: 45 },
  'Essences/Ampoules/Serums': { display: '약 1~2분',    seconds: 90 },
  'Cream/Gel':                { display: '약 1~2분',    seconds: 90 },
  'Face Mists':               { display: '약 10~20초',  seconds: 15 },
  'Eye Treatments':           { display: '약 1분',      seconds: 60 },
  'Balms/Multi-balms':        { display: '약 1~2분',    seconds: 90 },
  'Facial Oils':              { display: '약 1분',      seconds: 60 },
  'Shaving Products':         { display: '약 1~2분',    seconds: 90 },
  'All-In-One':               { display: '약 1~2분',    seconds: 90 },
};

const OPTIONAL_USAGE_HINT: Record<string, string> = {
  'Face Mists':        '수시로 분사',
  'Eye Treatments':    '세럼 다음에 바르기',
  'Balms/Multi-balms': '가장 마지막 단계',
  'Facial Oils':       '크림 뒤에 소량',
};

function formatBudgetRange(min: number | null | undefined, max: number | null | undefined, label: string): string | null {
  if ((min == null) && (max == null)) return null;
  if (max == null) return `${label} ${(min ?? 0).toLocaleString()}원+`;
  if (min == null || min === 0) return `${label} ~${max.toLocaleString()}원`;
  return `${label} ${min.toLocaleString()}~${max.toLocaleString()}원`;
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd} ${hh}:${min}`;
}

interface EnrichedProduct extends ProductDetail {
  applicationGuide: string;
  step: number;
  timeTag: 'am' | 'pm' | null;
}

export default function RoutinePage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { nickname } = useAuth();

  const sessionId: number  = location.state?.session_id ?? 0;
  const resultId: number   = location.state?.resultId   ?? 0;
  const budgetFallbackApplied: boolean = location.state?.budgetFallbackApplied ?? false;
  const budgetMessage: string | null = location.state?.budgetMessage ?? null;
  const explanationRoutines: {
    routine_id: number;
    routine_type: 'best' | 'value';
    routine_rank: number;
    ampm_mode: string;
    recommend_summary: string;
    ampm_comment: string;
    step_guides: { slot_order: number; category: string; usage_guide: string }[];
    strengths: string[];
    cautions: string[];
  }[] = location.state?.explanationRoutines ?? [];

  const [routines, setRoutines] = useState<RoutineItem[]>([]);
  const [productMap, setProductMap] = useState<Map<number, ProductDetail>>(new Map());
  const [skinResult, setSkinResult] = useState<AnalysisResult | null>(null);
  const [skinConcerns, setSkinConcerns] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [fetchedExplanations, setFetchedExplanations] = useState(explanationRoutines);
  const [budgetInfo, setBudgetInfo] = useState<{
    total_budget_min?: number | null; total_budget_max?: number | null;
    toner_budget_min?: number | null; toner_budget_max?: number | null;
    emulsion_budget_min?: number | null; emulsion_budget_max?: number | null;
    ampoule_budget_min?: number | null; ampoule_budget_max?: number | null;
    cream_budget_min?: number | null; cream_budget_max?: number | null;
  } | null>(null);

  const [activeType, setActiveType] = useState<'best' | 'value'>('best');

  useEffect(() => {
    userApi.getMe()
      .then((user) => setSkinConcerns(user.skin_concerns?.filter((c) => c !== 'none') ?? []))
      .catch(() => {});

    const load = async () => {
      if (!sessionId) {
        setError('루틴 세션 정보가 없어요. 저장된 루틴 목록에서 다시 시도해주세요.');
        setLoading(false);
        return;
      }
      try {
        const rec = await routineApi.getRecommendation(sessionId);
        const actualResultId = resultId || rec.result_id;
        const analysis = actualResultId ? await analysisApi.getResult(actualResultId) : null;

        setSkinResult(analysis);
        setRoutines(rec.routines);
        setBudgetInfo({
          total_budget_min:    rec.total_budget_min,
          total_budget_max:    rec.total_budget_max,
          toner_budget_min:    rec.toner_budget_min,
          toner_budget_max:    rec.toner_budget_max,
          emulsion_budget_min: rec.emulsion_budget_min,
          emulsion_budget_max: rec.emulsion_budget_max,
          ampoule_budget_min:  rec.ampoule_budget_min,
          ampoule_budget_max:  rec.ampoule_budget_max,
          cream_budget_min:    rec.cream_budget_min,
          cream_budget_max:    rec.cream_budget_max,
        });

        if (explanationRoutines.length === 0 && sessionId) {
          try {
            const expRes = await routineApi.getExplanation(sessionId);
            setFetchedExplanations(expRes.routines);
          } catch {
            // 백엔드 미지원 시 무시
          }
        }

        const allIds = [...new Set(rec.routines.flatMap((r) => r.products.map((p) => p.product_id)))];
        const details = await Promise.all(allIds.map((id) => productApi.getDetail(id)));
        const map = new Map<number, ProductDetail>();
        details.forEach((d) => map.set(d.product_id, d));
        setProductMap(map);

      } catch (err) {
        setError((err as Error).message || '루틴 정보를 불러오지 못했어요.');
      } finally {
        setLoading(false);
      }
    };
    load();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) {
    return (
      <div className="rp-page">
        <LoadingSpinner text="루틴을 불러오는 중이에요" />
      </div>
    );
  }

  if (error || routines.length === 0) {
    return (
      <div className="rp-page" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '1rem' }}>
        <p style={{ color: '#ef4444' }}>{error || '루틴 결과가 없어요.'}</p>
        <button onClick={() => navigate(-1)} style={{ padding: '0.5rem 1.5rem', borderRadius: '8px', background: '#7c3aed', color: '#fff', border: 'none', cursor: 'pointer' }}>← 돌아가기</button>
      </div>
    );
  }

  const currentRoutine = routines.find((r) => r.type === activeType) ?? routines[0];
  const timeMeta = TIME_META[currentRoutine.routine_time as RoutineTime] ?? TIME_META.both;
  const explanationRoutine = fetchedExplanations.find((r) => r.routine_type === activeType);
  const usageGuide = explanationRoutine?.ampm_comment ?? '';
  const warningText = explanationRoutine?.cautions.join('\n') ?? '';
  const stepGuideMap = new Map(
    explanationRoutine?.step_guides.map((sg) => [sg.slot_order, sg.usage_guide]) ?? []
  );

  const allEnriched: EnrichedProduct[] = currentRoutine.products
    .slice()
    .sort((a, b) => a.step - b.step)
    .map((rp) => {
      const detail = productMap.get(rp.product_id);
      if (!detail) return null;
      const guide = stepGuideMap.get(rp.step) ?? '';
      return { ...detail, applicationGuide: guide, step: rp.step, timeTag: rp.time_tag };
    })
    .filter((p): p is EnrichedProduct => p !== null);

  const enrichedProducts = allEnriched.filter((p) => !OPTIONAL_CATEGORIES.has(p.category));
  const optionalProducts  = allEnriched.filter((p) => OPTIONAL_CATEGORIES.has(p.category));

  const totalSeconds = allEnriched.reduce(
    (sum, p) => sum + (CATEGORY_DURATION[p.category]?.seconds ?? 60), 0
  );
  const totalMinutes = Math.ceil(totalSeconds / 60);

  const displayName = nickname || '내';

  return (
    <div className="rp-page">

      {/* ── HERO ── */}
      <section className="rp-hero">
        <div className="rp-hero-inner">
          <div className="rp-hero-badge">AI 맞춤 루틴</div>
          <h1 className="rp-hero-title">
            <span className="rp-hero-name">{displayName}</span>님,<br />
            <span className="rp-hero-highlight">맞춤 루틴 리포트</span>가 도착했어요
          </h1>
          {skinResult && (
            <p className="rp-hero-date">분석일 : {formatDate(skinResult.analyzed_at || skinResult.generated_at)}</p>
          )}
          <div className="rp-skin-tags">
            {skinResult?.skin_type && (
              <span className="rp-skin-tag rp-skin-tag--type">{skinResult.skin_type}</span>
            )}
            {skinConcerns.map((c) => (
              <span key={c} className="rp-skin-tag">{c}</span>
            ))}
          </div>

          {budgetInfo && (() => {
            const chips = [
              formatBudgetRange(budgetInfo.total_budget_min,    budgetInfo.total_budget_max,    '전체'),
              formatBudgetRange(budgetInfo.toner_budget_min,    budgetInfo.toner_budget_max,    '토너'),
              formatBudgetRange(budgetInfo.emulsion_budget_min, budgetInfo.emulsion_budget_max, '에멀젼'),
              formatBudgetRange(budgetInfo.ampoule_budget_min,  budgetInfo.ampoule_budget_max,  '앰플'),
              formatBudgetRange(budgetInfo.cream_budget_min,    budgetInfo.cream_budget_max,    '크림'),
            ].filter(Boolean) as string[];

            if (chips.length === 0) return null;
            return (
              <div className="rp-budget-row">
                <Wallet size={13} color="#9CA3AF" style={{ flexShrink: 0, marginTop: 1 }} />
                <span className="rp-budget-row-label">예산 조건</span>
                {chips.map((chip) => (
                  <span key={chip} className="rp-budget-chip">{chip}</span>
                ))}
              </div>
            );
          })()}
        </div>
      </section>

      {/* ── 루틴 탭 ── */}
      <div className="rp-tabs-wrap">
        <div className="rp-tabs">
          {routines.map((r) => (
            <button
              key={r.type}
              className={`rp-tab ${activeType === r.type ? 'active' : ''}`}
              onClick={() => setActiveType(r.type)}
            >
              {r.type === 'best'
                ? <><Trophy size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />AI 추천</>
                : <><Wallet size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />가성비</>}
            </button>
          ))}
        </div>
      </div>

      <div className="rp-body">

        {/* ── 예산 폴백 알림 ── */}
        {budgetFallbackApplied && budgetMessage && (
          <p className="rp-budget-fallback-notice">{budgetMessage}</p>
        )}

        {/* ── 루틴 요약 바 ── */}
        <div className="rp-summary-bar">
          <div className="rp-time-badge" style={{ background: timeMeta.bg, color: timeMeta.color }}>
            <span style={{ display: 'flex', gap: 2, alignItems: 'center' }}>{timeMeta.IconEl}</span>
            <span>{timeMeta.label}</span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <Package size={14} color="#9CA3AF" />
            <span>제품 <strong>{allEnriched.length}개</strong></span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <Timer size={14} color="#9CA3AF" />
            <span>소요 <strong>{totalMinutes}분</strong></span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <Banknote size={14} color="#9CA3AF" />
            <span>총 <strong>{currentRoutine.total_cost.toLocaleString()}원</strong></span>
          </div>
        </div>

        {/* ── 가성비 안내 문구 ── */}
        {activeType === 'value' && (
          <div className="rp-value-notice">
            <Info size={15} color="#7c3aed" style={{ flexShrink: 0, marginTop: 1 }} />
            가성비 루틴은 예산 조건과 관계없이 가장 합리적인 가격 대비 성능 조합으로 추천돼요.
          </div>
        )}

        {/* ── AI 루틴 설명 ── */}
        <div className="rp-ai-desc-box">
          <div className="rp-ai-desc-header">
            <Bot size={18} color="#7c3aed" style={{ flexShrink: 0 }} />
            <span className="rp-ai-title">AI 루틴 추천 이유</span>
            <div className="rp-time-tag" style={{ background: timeMeta.bg, color: timeMeta.color, display: 'flex', alignItems: 'center', gap: 4 }}>
              <span style={{ display: 'flex', gap: 2, alignItems: 'center' }}>{timeMeta.IconEl}</span>
              {timeMeta.desc}
            </div>
          </div>
          <p className="rp-ai-desc-text">{explanationRoutine?.recommend_summary ?? ''}</p>

          {usageGuide && (
            <div className="rp-guide-info">
              <ClipboardList size={16} color="#7c3aed" style={{ flexShrink: 0, marginTop: 1 }} />
              <div>
                <div className="rp-guide-info-label">사용 순서 가이드</div>
                <p className="rp-guide-info-text">{usageGuide}</p>
              </div>
            </div>
          )}
          {warningText && (
            <div className="rp-guide-info rp-guide-warning">
              <AlertTriangle size={16} color="#d97706" style={{ flexShrink: 0, marginTop: 1 }} />
              <div>
                <div className="rp-guide-info-label">주의사항</div>
                <p className="rp-guide-info-text">{warningText}</p>
              </div>
            </div>
          )}
        </div>

        {/* ── 필수 루틴 ── */}
        <div className="rp-routine-block">
          <div className="rp-block-header">
            <div className="rp-block-title-wrap">
              <span className="rp-block-badge required">필수</span>
              <h2 className="rp-block-title">기본 스킨케어 루틴</h2>
            </div>
            <p className="rp-block-sub">토너 · 에멀젼 · 앰플 · 크림</p>
          </div>

          <div className="rp-scroll">
            {enrichedProducts.map((product, idx) => (
              <div key={product.product_id} className="rp-product-card" onClick={() => navigate(`/products/${product.product_id}`)}>
                <div className="rp-card-header">
                  <div className="rp-product-step">STEP {idx + 1}</div>
                  <div className="rp-product-category">{CATEGORY_KO[product.category] ?? product.category}</div>
                </div>
                <img src={product.image_url} alt={product.product_name} className="rp-product-img" />
                <div className="rp-product-info">
                  <div className="rp-product-brand">{product.brand_name}</div>
                  <div className="rp-product-name">{product.product_name}</div>
                  <div className="rp-product-tags">
                    {product.timeTag && (
                      <span className={`rp-product-time-tag ${product.timeTag}`}>
                        {product.timeTag === 'am' ? 'AM전용' : 'PM전용'}
                      </span>
                    )}
                    {product.tags.filter((t) => t !== product.category).slice(0, 2).map((t) => <span key={t} className="rp-product-tag">#{t}</span>)}
                  </div>
                  <div className="rp-product-duration">
                    <Clock size={11} color="#9CA3AF" />
                    <span>{CATEGORY_DURATION[product.category]?.display ?? '약 1분'}</span>
                  </div>
                  <div className="rp-product-price">{product.price.toLocaleString()}원</div>
                </div>
                <div className="rp-guide-panel" onClick={(e) => e.stopPropagation()}>
                  <div className="rp-guide-header">
                    <FileText size={12} color="#7c3aed" />
                    <span>바르는 법</span>
                  </div>
                  <p className="rp-guide-text">{product.applicationGuide}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── 옵션 루틴 ── */}
        {optionalProducts.length > 0 && (
          <div className="rp-routine-block rp-optional-block">
            <div className="rp-block-header">
              <div className="rp-block-title-wrap">
                <span className="rp-block-badge optional">옵션</span>
                <h2 className="rp-block-title">추가 케어 루틴</h2>
              </div>
              <p className="rp-block-sub">
                {[...new Set(optionalProducts.map((p) => CATEGORY_KO[p.category] ?? p.category))].join(' · ')}
              </p>
            </div>
            <div className="rp-scroll">
              {optionalProducts.map((product) => (
                <div key={product.product_id} className="rp-product-card rp-optional-card" onClick={() => navigate(`/products/${product.product_id}`)}>
                  <div className="rp-card-header">
                    <div className="rp-product-category optional">{CATEGORY_KO[product.category] ?? product.category}</div>
                    {OPTIONAL_USAGE_HINT[product.category] && (
                      <span className="rp-optional-hint">{OPTIONAL_USAGE_HINT[product.category]}</span>
                    )}
                  </div>
                  <img src={product.image_url} alt={product.product_name} className="rp-product-img" />
                  <div className="rp-product-info">
                    <div className="rp-product-brand">{product.brand_name}</div>
                    <div className="rp-product-name">{product.product_name}</div>
                    <div className="rp-product-tags">
                      {product.timeTag && (
                        <span className={`rp-product-time-tag ${product.timeTag}`}>
                          {product.timeTag === 'am' ? 'AM전용' : 'PM전용'}
                        </span>
                      )}
                      {product.tags.filter((t) => t !== product.category).slice(0, 2).map((t) => <span key={t} className="rp-product-tag optional">#{t}</span>)}
                    </div>
                    <div className="rp-product-duration">
                      <Clock size={11} color="#9CA3AF" />
                      <span>{CATEGORY_DURATION[product.category]?.display ?? '약 1분'}</span>
                    </div>
                    <div className="rp-product-price">{product.price.toLocaleString()}원</div>
                  </div>
                  <div className="rp-guide-panel" onClick={(e) => e.stopPropagation()}>
                    <div className="rp-guide-header">
                      <FileText size={12} color="#7c3aed" />
                      <span>바르는 법</span>
                    </div>
                    <p className="rp-guide-text">{product.applicationGuide}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

      </div>

    </div>
  );
}
