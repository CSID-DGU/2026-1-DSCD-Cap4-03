import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { vanityApi, type VanityProduct } from '../api/vanity';
import { productApi } from '../api/product';
import { FlaskConical, Check, ChevronLeft, Clock, Package, AlertTriangle } from 'lucide-react';
import './VanityRoutinePage.css';

const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼', 'Essences/Ampoules/Serums': '에센스/앰플/세럼',
  'Cream/Gel': '크림/젤', 'Eye Treatments': '아이크림',
  'Balms/Multi-balms': '멀티밤', 'Facial Oils': '페이셜 오일', 'Face Mists': '미스트',
};

type Category = '전체' | '토너' | '에멀젼' | '앰플' | '크림';
const CATEGORIES: Category[] = ['전체', '토너', '에멀젼', '앰플', '크림'];

const PRODUCT_CAT_TO_BUDGET: Record<string, Category> = {
  'Toner': '토너', 'Toner Pads': '토너',
  'Emulsions': '에멀젼',
  'Essences/Ampoules/Serums': '앰플',
  'Cream/Gel': '크림',
};

const TOTAL_OPTIONS = [
  { label: '설정 안함',  min: null,   max: null    },
  { label: '0~10만원',  min: 0,      max: 100000  },
  { label: '10~15만원', min: 100000, max: 150000  },
  { label: '15~20만원', min: 150000, max: 200000  },
  { label: '20만원+',   min: 200000, max: null    },
];

const CAT_OPTIONS = [
  { label: '설정 안함', min: null,  max: null  },
  { label: '0~2만원',  min: 0,     max: 20000 },
  { label: '2~3만원',  min: 20000, max: 30000 },
  { label: '3~5만원',  min: 30000, max: 50000 },
  { label: '5만원+',   min: 50000, max: null  },
];

type BudgetRange = { min: number | null; max: number | null };
const DEFAULT_BUDGETS: Record<Category, BudgetRange> = {
  '전체':  { min: null, max: null }, '토너':   { min: null, max: null },
  '에멀젼': { min: null, max: null }, '앰플':   { min: null, max: null },
  '크림':  { min: null, max: null },
};

export default function VanityRoutineBudgetPage() {
  const navigate = useNavigate();

  const [products, setProducts]       = useState<VanityProduct[]>([]);
  const [selected, setSelected]       = useState<Set<number>>(new Set());
  const [hasAnalysis, setHasAnalysis] = useState(false);
  const [budgets, setBudgets]         = useState<Record<Category, BudgetRange>>(DEFAULT_BUDGETS);
  const [imageMap, setImageMap]       = useState<Record<number, string>>({});
  const [loading, setLoading]         = useState(true);

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
  /* 선택된 제품의 카테고리에 해당하는 예산 카테고리는 비활성화 */
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
    const body: Parameters<typeof vanityApi.runRoutine>[0] = { fixed_product_ids: [...selected] };
    const bv = budgets;
    if (bv['전체'].min  !== null) body.budget_min   = bv['전체'].min;
    if (bv['전체'].max  !== null) body.budget_max   = bv['전체'].max;
    if (bv['토너'].min  !== null) body.toner_min    = bv['토너'].min;
    if (bv['토너'].max  !== null) body.toner_max    = bv['토너'].max;
    if (bv['에멀젼'].min !== null) body.emulsion_min = bv['에멀젼'].min;
    if (bv['에멀젼'].max !== null) body.emulsion_max = bv['에멀젼'].max;
    if (bv['앰플'].min  !== null) body.ampoule_min  = bv['앰플'].min;
    if (bv['앰플'].max  !== null) body.ampoule_max  = bv['앰플'].max;
    if (bv['크림'].min  !== null) body.cream_min    = bv['크림'].min;
    if (bv['크림'].max  !== null) body.cream_max    = bv['크림'].max;
    navigate('/loading', { state: { type: 'vanity_routine', vanity_routine_body: body } });
  };

  if (loading) return (
    <div className="vr-page" style={{ display:'flex', alignItems:'center', justifyContent:'center', minHeight:'60vh' }}>
      <p style={{ color: '#7c3aed' }}>불러오는 중...</p>
    </div>
  );

  return (
    <div className="vr-page">
      <div className="vr-banner">
        <div className="vr-banner-inner">
          <button
            style={{ background:'none', border:'none', color:'#7c3aed', fontSize:13, fontWeight:700, cursor:'pointer', display:'flex', alignItems:'center', gap:4, padding:0, marginBottom:16, fontFamily:'Noto Sans KR, sans-serif' }}
            onClick={() => navigate('/vanity')}>
            <ChevronLeft size={15} />내 화장대로
          </button>
          <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start' }}>
            <div>
              <div className="vr-badge">SKIN FIT · ROUTINE</div>
              <h1 className="vr-title">내 화장대로 루틴을 완성할 수 있을까요?</h1>
              <p className="vr-sub">쓰던 제품은 유지하고, 비어 있는 단계는 AI가 최적의 제품으로 채워드려요</p>
            </div>
            <button
              style={{ background:'none', border:'1.5px solid #7c3aed', color:'#7c3aed', borderRadius:8, padding:'8px 16px', fontSize:12, fontWeight:700, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif', display:'flex', alignItems:'center', gap:6, flexShrink:0, marginTop:8 }}
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
            <div style={{ marginTop:12 }}>
              <button onClick={() => navigate('/diagnosis')}
                style={{ background:'#d97706', color:'#fff', border:'none', borderRadius:8, padding:'8px 20px', fontWeight:700, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif' }}>
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
              <div style={{ textAlign:'center', padding:'24px', color:'#9CA3AF', fontSize:14 }}>
                화장대에 등록된 제품이 없어요.
                <div style={{ marginTop:12 }}>
                  <button onClick={() => navigate('/vanity')}
                    style={{ background:'#7c3aed', color:'#fff', border:'none', borderRadius:8, padding:'8px 20px', fontWeight:700, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif' }}>
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
            <div className="vr-section-sub">빈 루틴 단계를 채울 새 제품의 예산 범위를 카테고리별로 설정하세요. 설정 안 해도 괜찮아요.</div>
            <div className="vr-budget-grid">
              {CATEGORIES.map(cat => {
                const opts   = cat === '전체' ? TOTAL_OPTIONS : CAT_OPTIONS;
                const cur    = budgets[cat];
                const locked = cat !== '전체' && lockedBudgetCats.has(cat);
                return (
                  <div key={cat}>
                    <div className="vr-budget-section-label" style={{ opacity: locked ? 0.45 : 1 }}>
                      {cat}
                      {locked && (
                        <span style={{ marginLeft: 6, fontSize: 10, fontWeight: 600, color: '#7c3aed', background: '#EDE9FE', padding: '1px 7px', borderRadius: 999 }}>
                          고정됨
                        </span>
                      )}
                    </div>
                    {locked ? (
                      <div style={{ fontSize: 12, color: '#9CA3AF', padding: '4px 0' }}>고정 제품 카테고리는 예산 설정 불필요</div>
                    ) : (
                      <div className="vr-budget-options">
                        {opts.map(({ label, min, max }) => (
                          <button key={label}
                            className={`vr-budget-btn ${cur.min === min && cur.max === max ? 'selected' : ''}`}
                            onClick={() => setBudgets(prev => ({ ...prev, [cat]: { min, max } }))}>
                            {label}
                          </button>
                        ))}
                      </div>
                    )}
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
              <span style={{ fontSize:12, color:'#9CA3AF' }}>고정할 제품을 최소 1개 이상 선택해주세요</span>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
