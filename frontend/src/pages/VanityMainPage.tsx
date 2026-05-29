import { useEffect, useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  vanityApi,
  type VanitySummary,
  type SkinMatchLatest,
  type VanityRoutineResult,
  FIT_LABEL_KO,
  FIT_LABEL_CLASS,
} from '../api/vanity';
import { productApi, type ProductSummary } from '../api/product';
import {
  FlaskConical, Package,
  Plus, Trash2, Search, X, RefreshCw,
} from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import './VanityMainPage.css';

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

const FIT_LABEL_KEYS = ['excellent_match', 'good_match', 'so_so', 'weak_match', 'poor_match'] as const;

function formatDate(str: string) {
  const d = new Date(str);
  if (isNaN(d.getTime())) return str;
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

function formatDateTime(str: string) {
  const d = new Date(str);
  if (isNaN(d.getTime())) return str;
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

export default function VanityMainPage() {
  const navigate = useNavigate();

  const [summary, setSummary]                 = useState<VanitySummary | null>(null);
  const [skinMatchDetail, setSkinMatchDetail]  = useState<SkinMatchLatest | null>(null);
  const [routineDetail, setRoutineDetail]      = useState<VanityRoutineResult | null>(null);
  const [loading, setLoading]                 = useState(true);

  /* 제품 추가 모달 */
  const [showModal, setShowModal]     = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [allProducts, setAllProducts] = useState<ProductSummary[]>([]);
  const [modalLoading, setModalLoading] = useState(true);
  const [addingId, setAddingId]       = useState<number | null>(null);
  const [imageMap, setImageMap]       = useState<Record<number, string>>({});

  const load = async () => {
    let summaryData: VanitySummary | null = null;
    try {
      summaryData = await vanityApi.getSummary();
      setSummary(summaryData);
    } catch { /* no-op */ }
    try { setSkinMatchDetail(await vanityApi.getLatestSkinMatch()); } catch { /* no-op */ }
    if (summaryData?.latest_vanity_routine?.recommendation_session_id) {
      try {
        setRoutineDetail(await vanityApi.getRoutineDetail(
          summaryData.latest_vanity_routine.recommendation_session_id
        ));
      } catch { /* no-op */ }
    }
    setLoading(false);
  };

  useEffect(() => { load(); }, []);

  /* 제품 목록 선로드 → imageMap + 모달용 */
  useEffect(() => {
    productApi.getList()
      .then(prods => {
        const map: Record<number, string> = {};
        prods.forEach(p => { if (p.image_url) map[p.product_id] = p.image_url; });
        setImageMap(map);
        setAllProducts(prods);
      })
      .catch(() => {})
      .finally(() => setModalLoading(false));
  }, []);

  /* 모달 열기 */
  const openModal = () => {
    setShowModal(true);
    setSearchQuery('');
  };

  /* 제품 추가 */
  const handleAddProduct = async (product_id: number) => {
    setAddingId(product_id);
    try {
      await vanityApi.addProduct(product_id);
      setShowModal(false);
      await load();
    } catch (e) {
      alert(`제품 추가 중 오류가 발생했어요: ${e instanceof Error ? e.message : '알 수 없는 오류'}`);
    }
    setAddingId(null);
  };

  /* 제품 삭제 */
  const handleDeleteProduct = async (product_id: number, e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await vanityApi.deleteProduct(product_id);
      await load();
    } catch { /* no-op */ }
  };

  /* 이미 추가된 product_id 세트 */
  const vanityIds = useMemo(
    () => new Set((summary?.product_summary.products ?? []).map((p) => p.product_id)),
    [summary]
  );

  /* 검색 필터 */
  const filteredProducts = useMemo(() => {
    if (!searchQuery.trim()) return allProducts;
    const q = searchQuery.toLowerCase();
    return allProducts.filter(
      (p) => p.product_name.toLowerCase().includes(q) || p.brand_name.toLowerCase().includes(q)
    );
  }, [allProducts, searchQuery]);

  if (loading) return (
    <div className="vm-page">
      <LoadingSpinner />
    </div>
  );

  const products    = summary?.product_summary.products ?? [];
  const skinMatch   = summary?.latest_skin_match;
  const routine     = summary?.latest_vanity_routine;
  const basisResult = summary?.basis_skin_result;

  const matchMap = Object.fromEntries(
    (skinMatchDetail?.product_match_results ?? []).map((r) => [r.product_id, r])
  );
  const matchSummary = skinMatchDetail?.summary ?? null;

  const finalRoutine = [...(routineDetail?.routine_recommendation_results?.final_routine ?? [])]
    .sort((a, b) => a.slot_order - b.slot_order);
  const reqRoutine    = finalRoutine.filter(p => !OPTIONAL_CATEGORIES.has(p.category));
  const optRoutine    = finalRoutine.filter(p => OPTIONAL_CATEGORIES.has(p.category));
  const sortedRoutine = [...reqRoutine, ...optRoutine];

  return (
    <div className="vm-page">

      {/* ── 배너 ── */}
      <div className="vm-banner">
        <div className="vm-banner-inner">
          <div className="vm-badge">MY VANITY REPORT</div>
          <h1 className="vm-title">지금 쓰는 화장품, 내 피부에 맞나요?</h1>
          <p className="vm-sub">제품 적합성을 확인하고, 내 화장대에서 시작하는 루틴까지 완성해보세요</p>
          {basisResult ? (
            <>
              <div className="vm-analysis-chip">
                <FlaskConical size={13} />
                {basisResult.message ?? '가장 최신 피부 분석 결과를 기준으로 분석해요.'}
              </div>
              <div style={{ fontSize: 12, color: '#7c3aed', fontWeight: 600, marginTop: 6, paddingLeft: 4 }}>
                기준 분석일: {formatDate(basisResult.analyzed_at)}
              </div>
            </>
          ) : (
            <div className="vm-analysis-chip" style={{ background: '#FEF3C7', borderColor: '#FDE68A', color: '#d97706' }}>
              내 화장대 분석을 위해 먼저 피부 분석을 진행해주세요.
            </div>
          )}
        </div>
      </div>

      <div className="vm-body">

        {/* ── 1. 내 화장대 제품 + 적합성 ── */}
        <div className="vm-section">
          <div className="vm-section-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: 1, flexWrap: 'wrap' }}>
              <div className="vm-section-title">
                내 화장대
                {summary && (
                  <span style={{ fontSize: 13, fontWeight: 500, color: '#9CA3AF', marginLeft: 4 }}>
                    {summary.product_summary.total_count}개
                  </span>
                )}
              </div>
              {matchSummary && (
                <div className="vm-match-chips">
                  {FIT_LABEL_KEYS.filter((k) => (matchSummary[k] ?? 0) > 0).map((k) => (
                    <div key={k} className={`vm-match-chip ${FIT_LABEL_CLASS[k]}`}>
                      <span className="vm-match-chip-count">{matchSummary[k]}</span>
                      <span className="vm-match-chip-label">{FIT_LABEL_KO[k]}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
            <div style={{ display: 'flex', gap: 8, flexShrink: 0 }}>
              {skinMatch && (
                <button className="vm-section-btn" onClick={() => navigate('/vanity/skin-match')}>
                  자세히 보기 →
                </button>
              )}
              <button className="vm-section-btn primary" onClick={openModal}>
                <Plus size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
                제품 추가
              </button>
            </div>
          </div>

          {products.length === 0 ? (
            <div className="vm-empty-add" onClick={openModal}>
              <div className="vm-empty-add-icon"><Plus size={28} color="#A78BFA" /></div>
              <p className="vm-empty-add-text">내 화장대가 비어있어요</p>
              <p className="vm-empty-add-sub">제품을 추가하면 피부 적합성을 바로 확인할 수 있어요</p>
            </div>
          ) : (
            <>
              <div className="vm-product-scroll">
                {products.map((p) => {
                  const match  = matchMap[p.product_id];
                  const fitCls = match ? FIT_LABEL_CLASS[match.fit_label] : null;
                  const imgUrl = imageMap[p.product_id] || p.image_url;
                  return (
                    <div
                      key={p.product_id}
                      className="vm-product-card"
                      onClick={() => navigate('/vanity/skin-match')}
                    >
                      <div className="vm-card-header">
                        <div className="vm-card-cat">{CATEGORY_KO[p.category] ?? p.category}</div>
                      {match && fitCls && (
                        <div className={`vm-fit-badge ${fitCls}`}>{match.display_label}</div>
                      )}
                      </div>
                      {imgUrl
                        ? <img src={imgUrl} alt={p.product_name} className="vm-product-img" />
                        : <div className="vm-product-img" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#F5F3FF' }}>
                            <Package size={32} color="#A78BFA" />
                          </div>
                      }
                      <div className="vm-product-brand">{p.brand_name}</div>
                      <div className="vm-product-name">{p.product_name}</div>
                      <button className="vm-product-delete" onClick={(e) => handleDeleteProduct(p.product_id, e)}>
                        <Trash2 size={11} />
                      </button>
                    </div>
                  );
                })}
                <div className="vm-add-card" onClick={openModal}>
                  <Plus size={24} color="#A78BFA" />
                  <span>제품 추가</span>
                </div>
              </div>

            </>
          )}
        </div>

        {/* ── 2. 루틴 완성 ── */}
        <div className="vm-section">
          <div className="vm-section-header">
            <div className="vm-section-title">
              내 화장대에 맞는 루틴은?
            </div>
            <div style={{ display: 'flex', gap: 8 }}>
              {routine && (
                <button className="vm-section-btn" onClick={() => navigate('/vanity/routine/history')}>
                  루틴 기록 →
                </button>
              )}
              <button
                className="vm-section-btn primary"
                onClick={() => navigate('/vanity/routine/budget')}
                disabled={products.length === 0}
              >
                <RefreshCw size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />
                루틴 추천 받기
              </button>
            </div>
          </div>

          {!routine ? (
            <div className="vm-routine-empty">
              아직 루틴을 완성하지 않았어요.<br />내 화장대 제품을 고정하고, 빈 단계는 AI 추천으로 채워보세요.
            </div>
          ) : (
            <>
              <p style={{ fontSize: 12, color: '#9CA3AF', marginBottom: 12 }}>
                최신 루틴 1개를 보여드려요 · 루틴 추천일: {formatDateTime(routine.created_at)}
              </p>
              {sortedRoutine.length > 0 && (
                <div className="vm-product-scroll">
                  {sortedRoutine.map((item) => {
                    const imgUrl    = imageMap[item.product_id] || item.image_url;
                    const isOptional = OPTIONAL_CATEGORIES.has(item.category);
                    return (
                      <div key={item.slot_order} className="vm-product-card"
                        onClick={() => navigate('/vanity/routine')}>
                        <div className="vm-card-header">
                          <div className={`vm-card-step${isOptional ? ' optional' : ''}`}>
                            {isOptional ? '옵션' : `STEP ${CATEGORY_STEP_ORDER[item.category] ?? item.slot_order}`}
                          </div>
                          <div className="vm-card-cat">{CATEGORY_KO[item.category] ?? item.category}</div>
                        </div>
                        {imgUrl
                          ? <img src={imgUrl} alt={item.product_name} className="vm-product-img" />
                          : <div className="vm-product-img" style={{ display:'flex', alignItems:'center', justifyContent:'center', background:'#F5F3FF' }}>
                              <Package size={24} color="#A78BFA" />
                            </div>
                        }
                        <div className="vm-product-brand">{item.brand_name}</div>
                        <div style={{ display:'flex', alignItems:'flex-start', gap:4, flexWrap:'wrap' }}>
                          <div className="vm-product-name" style={{ flex:1 }}>{item.product_name}</div>
                          {item.source === 'vanity' && (
                            <span style={{ fontSize:9, fontWeight:700, background:'#EDE9FE', color:'#7c3aed', padding:'2px 6px', borderRadius:999, flexShrink:0, marginTop:2 }}>
                              내 화장품
                            </span>
                          )}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          )}
        </div>

      </div>

      {/* ── 제품 검색 모달 ── */}
      {showModal && (
        <div className="vm-modal-overlay" onClick={() => setShowModal(false)}>
          <div className="vm-modal" onClick={(e) => e.stopPropagation()}>

            <div className="vm-modal-header">
              <span className="vm-modal-title">제품 추가</span>
              <button className="vm-modal-close" onClick={() => setShowModal(false)}>
                <X size={18} />
              </button>
            </div>

            <div className="vm-modal-search-wrap">
              <Search size={16} color="#9CA3AF" />
              <input
                className="vm-modal-search"
                placeholder="제품명 또는 브랜드 검색"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                autoFocus
              />
              {searchQuery && (
                <button className="vm-modal-search-clear" onClick={() => setSearchQuery('')}>
                  <X size={14} />
                </button>
              )}
            </div>

            <div className="vm-modal-list">
              {modalLoading ? (
                <div className="vm-modal-empty">불러오는 중...</div>
              ) : filteredProducts.length === 0 ? (
                <div className="vm-modal-empty">검색 결과가 없어요</div>
              ) : (
                filteredProducts.map((p) => {
                  const isAdded  = vanityIds.has(p.product_id);
                  const isAdding = addingId === p.product_id;
                  return (
                    <div key={p.product_id} className="vm-modal-item">
                      {p.image_url
                        ? <img src={p.image_url} alt={p.product_name} className="vm-modal-item-img" />
                        : <div className="vm-modal-item-img vm-modal-item-img-fallback">
                            <Package size={18} color="#A78BFA" />
                          </div>
                      }
                      <div className="vm-modal-item-info">
                        <div className="vm-modal-item-brand">{p.brand_name}</div>
                        <div className="vm-modal-item-name">{p.product_name}</div>
                        <div className="vm-modal-item-meta">
                          <span className="vm-modal-item-cat">{CATEGORY_KO[p.category] ?? p.category}</span>
                          <span className="vm-modal-item-price">{p.price.toLocaleString()}원</span>
                        </div>
                      </div>
                      <button
                        className={`vm-modal-add-btn ${isAdded ? 'added' : ''}`}
                        onClick={() => !isAdded && handleAddProduct(p.product_id)}
                        disabled={isAdded || isAdding}
                      >
                        {isAdding ? '추가 중' : isAdded ? '추가됨' : '추가'}
                      </button>
                    </div>
                  );
                })
              )}
            </div>

          </div>
        </div>
      )}

    </div>
  );
}
