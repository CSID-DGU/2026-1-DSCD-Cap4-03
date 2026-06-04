import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { vanityApi, type VanityRoutineResult } from '../api/vanity';
import { productApi } from '../api/product';
import {
  ChevronLeft, Package, Sparkles, Banknote, AlertTriangle,
  FlaskConical, Clock, FileText,
} from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import { useAuth } from '../context/useAuth';
import './RoutinePage.css';

const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼', 'Essences/Ampoules/Serums': '에센스/앰플/세럼',
  'Cream/Gel': '크림/젤', 'Eye Treatments': '아이크림',
  'Balms/Multi-balms': '멀티밤', 'Facial Oils': '페이셜 오일', 'Face Mists': '미스트',
};

const OPTIONAL_CATEGORIES = new Set([
  'Balms/Multi-balms', 'Eye Treatments', 'Facial Oils',
  'Shaving Products', 'All-In-One', 'Face Mists',
]);

const CATEGORY_STEP_ORDER: Record<string, number> = {
  'Toner': 1, 'Toner Pads': 1,
  'Emulsions': 2,
  'Essences/Ampoules/Serums': 3,
  'Cream/Gel': 4,
};

const OPTIONAL_USAGE_HINT: Record<string, string> = {
  'Face Mists':        '수시로 분사',
  'Eye Treatments':    '세럼 다음에 바르기',
  'Balms/Multi-balms': '가장 마지막 단계',
  'Facial Oils':       '크림 뒤에 소량',
};

function formatDate(str: string) {
  const d = new Date(str);
  if (isNaN(d.getTime())) return str;
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

type RoutineCardItem = { slot_order: number; category: string; product_id: number; source: string; brand_name: string; product_name: string; price: number; image_url?: string };
type LlmVanityRoutine = { overall_summary: string; step_comments: { slot_order: number; comment: string }[]; warning_comment: string } | null | undefined;

function renderCard(item: RoutineCardItem, imageMap: Record<number, string>, llmRoutine: LlmVanityRoutine, isOptional = false, onNavigate?: (id: number) => void) {
  const imgUrl      = imageMap[item.product_id] || item.image_url;
  const stepComment = llmRoutine?.step_comments?.find(c => c.slot_order === item.slot_order);
  const isVanity    = item.source === 'vanity';
  return (
    <div key={item.slot_order} className={`rp-product-card${isOptional ? ' rp-optional-card' : ''}${isVanity ? ' rp-product-card--owned' : ''}`}
      onClick={() => onNavigate?.(item.product_id)} style={{ cursor: onNavigate ? 'pointer' : 'default' }}>
      <div className="rp-card-header">
        {isOptional ? (
          <>
            <div className="rp-product-category optional">{CATEGORY_KO[item.category] ?? item.category}</div>
            {OPTIONAL_USAGE_HINT[item.category] && (
              <span className="rp-optional-hint">{OPTIONAL_USAGE_HINT[item.category]}</span>
            )}
          </>
        ) : (
          <>
            <div className="rp-product-step">STEP {CATEGORY_STEP_ORDER[item.category] ?? item.slot_order}</div>
            <div className="rp-product-category">{CATEGORY_KO[item.category] ?? item.category}</div>
          </>
        )}
      </div>
      {imgUrl
        ? <img src={imgUrl} alt={item.product_name} className="rp-product-img" />
        : <div className="rp-product-img" style={{ display:'flex', alignItems:'center', justifyContent:'center', background:'#F5F3FF' }}>
            <Package size={32} color="#A78BFA" />
          </div>
      }
      <div className="rp-product-brand">{item.brand_name}</div>
      <div className="rp-product-name">{item.product_name}</div>
      <div className="rp-product-tags">
        <span className={`rp-product-tag${isVanity ? '' : ' optional'}`}>
          {isVanity ? '내 화장품' : '추천'}
        </span>
      </div>
      <div className="rp-product-price">{item.price.toLocaleString()}원</div>
      {stepComment?.comment && (
        <div className="rp-guide-panel">
          <div className="rp-guide-header">
            <FileText size={12} color="#7c3aed" />
            <span>AI 추천 이유</span>
          </div>
          <p className="rp-guide-text">{stepComment.comment}</p>
        </div>
      )}
    </div>
  );
}

const BACK_BTN: React.CSSProperties = {
  background: 'none', border: 'none', color: '#7c3aed', fontSize: 13, fontWeight: 700,
  cursor: 'pointer', display: 'flex', alignItems: 'center', gap: 4,
  padding: 0, marginBottom: 16, fontFamily: 'Noto Sans KR, sans-serif',
};

export default function VanityRoutinePage() {
  const location = useLocation();
  const navigate  = useNavigate();
  const { nickname } = useAuth();
  const displayName = nickname || '내';

  const passedResult: VanityRoutineResult | null = location.state?.result ?? null;
  const stateBudget: Record<string, number> | null = location.state?.budgetBody ?? null;

  const getBudgetBody = (sessionId?: number): Record<string, number> | null => {
    if (stateBudget) return stateBudget;
    if (!sessionId) return null;
    try {
      const raw = localStorage.getItem(`vanity_budget_${sessionId}`);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  };

  const fmtBudgetChips = (b: Record<string, number> | null): string[] => {
    if (!b) return [];
    const fmt = (min: number | undefined, max: number | undefined, label: string) => {
      if (min == null && max == null) return null;
      if (min != null && max != null) return `${label} ${min/10000}~${max/10000}만원`;
      if (max != null) return `${label} ${max/10000}만원 이하`;
      if (min != null && min > 0) return `${label} ${min/10000}만원 이상`;
      return null;
    };
    return [
      fmt(b.toner_min,    b.toner_max,    '토너'),
      fmt(b.emulsion_min, b.emulsion_max, '에멀젼'),
      fmt(b.ampoule_min,  b.ampoule_max,  '앰플'),
      fmt(b.cream_min,    b.cream_max,    '크림'),
    ].filter((c): c is string => c !== null);
  };

  const [result, setResult]   = useState<VanityRoutineResult | null>(passedResult);
  const [imageMap, setImageMap] = useState<Record<number, string>>({});
  const [loading, setLoading]  = useState(!passedResult);

  useEffect(() => {
    productApi.getList()
      .then(prods => {
        const map: Record<number, string> = {};
        prods.forEach(p => { if (p.image_url) map[p.product_id] = p.image_url; });
        setImageMap(map);
      })
      .catch(() => {});

    if (!passedResult) {
      vanityApi.getLatestRoutine()
        .then(r => setResult(r))
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (loading) return (
    <div className="rp-page">
      <LoadingSpinner text="루틴을 불러오는 중이에요" />
    </div>
  );

  if (!result) return (
    <div className="rp-page" style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', minHeight:'60vh', gap:'1rem' }}>
      <p style={{ color:'#9CA3AF' }}>아직 화장대 루틴이 없어요.</p>
      <button onClick={() => navigate('/vanity/routine/budget')}
        style={{ padding:'10px 24px', borderRadius:8, background:'#7c3aed', color:'#fff', border:'none', cursor:'pointer', fontWeight:700, fontFamily:'Noto Sans KR, sans-serif' }}>
        루틴 추천받기 →
      </button>
    </div>
  );

  const budgetChips = fmtBudgetChips(getBudgetBody(result.recommendation_session_id));

  const items      = [...(result.routine_recommendation_results?.final_routine ?? [])].sort((a, b) => a.slot_order - b.slot_order);
  const required   = items.filter(p => !OPTIONAL_CATEGORIES.has(p.category));
  const optional   = items.filter(p => OPTIONAL_CATEGORIES.has(p.category));
  const warnings   = result.routine_recommendation_results?.warnings ?? [];
  const fixedCount = result.routine_recommendation_results?.fixed_products?.length ?? 0;
  const llmRoutine = result.llm_explanation?.vanity_routine;
  const basis      = result.basis_skin_result;

  return (
    <div className="rp-page">

      {/* ── HERO ── */}
      <section className="rp-hero">
        <div className="rp-hero-inner">
          <button style={BACK_BTN} onClick={() => navigate('/vanity')}>
            <ChevronLeft size={15} />내 화장대로
          </button>
          <div className="rp-hero-badge">SKIN FIT · ROUTINE</div>
          <h1 className="rp-hero-title">
            <span className="rp-hero-name">{displayName}</span>님,<br />
            <span className="rp-hero-highlight">화장대 루틴 리포트</span>가 도착했어요
          </h1>
          {basis?.analyzed_at && (
            <p className="rp-hero-date">피부 분석 기준: {formatDate(basis.analyzed_at)}</p>
          )}
          {result.created_at && (
            <p className="rp-hero-date">추천일: {formatDate(result.created_at)}</p>
          )}
          <div className="rp-budget-row">
            <Banknote size={13} color="#9CA3AF" style={{ flexShrink: 0, marginTop: 1 }} />
            <span className="rp-budget-row-label">예산 조건</span>
            {budgetChips.length > 0
              ? budgetChips.map((chip) => <span key={chip} className="rp-budget-chip">{chip}</span>)
              : <span className="rp-budget-row-label" style={{ color: '#9CA3AF' }}>미설정</span>
            }
          </div>
          <div style={{ display:'flex', gap:10, marginTop:16, flexWrap:'wrap' }}>
            <button onClick={() => navigate('/vanity/routine/history')}
              style={{ background:'none', border:'1.5px solid rgba(167,139,250,0.5)', color:'#835aff', fontSize:13, fontWeight:700, padding:'8px 18px', borderRadius:8, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif', display:'flex', alignItems:'center', gap:6 }}>
              <Clock size={13} />루틴 기록
            </button>
            <button onClick={() => navigate('/vanity/routine/budget')}
              style={{ background:'#835aff', border:'none', color:'#fff', fontSize:13, fontWeight:700, padding:'8px 18px', borderRadius:8, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif' }}>
              새 루틴 추천받기 →
            </button>
          </div>
        </div>
      </section>

      <div className="rp-body">

        {/* ── 요약 바 ── */}
        <div className="rp-summary-bar">
          <div className="rp-summary-item">
            <Package size={14} color="#9CA3AF" />
            <span>제품 <strong>{items.length}개</strong></span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <FlaskConical size={14} color="#9CA3AF" />
            <span>화장대 고정 <strong>{fixedCount}개</strong></span>
          </div>
        </div>

        {/* ── AI 설명 ── */}
        {llmRoutine?.overall_summary && (
          <div className="rp-ai-desc-box">
            <div className="rp-ai-desc-header">
              <Sparkles size={18} color="#7c3aed" style={{ flexShrink:0 }} />
              <span className="rp-ai-title">AI 루틴 설명</span>
            </div>
            <p className="rp-ai-desc-text">{llmRoutine.overall_summary}</p>
            {warnings.length > 0 && llmRoutine.warning_comment && (
              <div className="rp-guide-info rp-guide-warning">
                <AlertTriangle size={16} color="#d97706" style={{ flexShrink:0, marginTop:1 }} />
                <div>
                  <div className="rp-guide-info-label">주의사항</div>
                  <p className="rp-guide-info-text">{llmRoutine.warning_comment}</p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── 경고 (LLM 없을 때) ── */}
        {warnings.length > 0 && !llmRoutine?.overall_summary && (
          <div className="rp-ai-desc-box" style={{ borderLeftColor:'#d97706' }}>
            <div className="rp-ai-desc-header">
              <AlertTriangle size={18} color="#d97706" style={{ flexShrink:0 }} />
              <span className="rp-ai-title" style={{ color:'#d97706' }}>주의 성분 안내</span>
            </div>
            <ul style={{ margin:0, paddingLeft:18 }}>
              {warnings.map((w, i) => <li key={i} style={{ fontSize:13, color:'#92400E', lineHeight:1.7 }}>{w}</li>)}
            </ul>
          </div>
        )}

        {/* ── 필수 루틴 ── */}
        {required.length > 0 && (
          <div className="rp-routine-block">
            <div className="rp-block-header">
              <div className="rp-block-title-wrap">
                <span className="rp-block-badge required">필수</span>
                <h2 className="rp-block-title">기본 스킨케어 루틴</h2>
              </div>
              <p className="rp-block-sub">
                {[...new Set(required.map(p => CATEGORY_KO[p.category] ?? p.category))].join(' · ')}
              </p>
            </div>
            <div className="rp-scroll">
              {required.map(item => renderCard(item, imageMap, llmRoutine, false, (id) => navigate(`/products/${id}`)))}
            </div>
          </div>
        )}

        {/* ── 옵션 루틴 ── */}
        {optional.length > 0 && (
          <div className="rp-routine-block rp-optional-block">
            <div className="rp-block-header">
              <div className="rp-block-title-wrap">
                <span className="rp-block-badge optional">옵션</span>
                <h2 className="rp-block-title">추가 케어 루틴</h2>
              </div>
              <p className="rp-block-sub">
                {[...new Set(optional.map(p => CATEGORY_KO[p.category] ?? p.category))].join(' · ')}
              </p>
            </div>
            <div className="rp-scroll">
              {optional.map(item => renderCard(item, imageMap, llmRoutine, true, (id) => navigate(`/products/${id}`)))}
            </div>
          </div>
        )}

      </div>
    </div>
  );
}
