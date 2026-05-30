import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  vanityApi,
  type VanityRoutineHistoryItem,
  type VanityRoutineResult,
} from '../api/vanity';
import { productApi } from '../api/product';
import { ChevronLeft, ChevronRight, Package, FlaskConical } from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import './VanityRoutineHistoryPage.css';

function formatDate(str: string) {
  const d = new Date(str);
  if (isNaN(d.getTime())) return str;
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')}`;
}

export default function VanityRoutineHistoryPage() {
  const navigate = useNavigate();

  const [history, setHistory]     = useState<VanityRoutineHistoryItem[]>([]);
  const [detailMap, setDetailMap] = useState<Record<number, VanityRoutineResult>>({});
  const [imageMap, setImageMap]   = useState<Record<number, string>>({});
  const [loading, setLoading]     = useState(true);

  useEffect(() => {
    productApi.getList()
      .then(prods => {
        const map: Record<number, string> = {};
        prods.forEach(p => { if (p.image_url) map[p.product_id] = p.image_url; });
        setImageMap(map);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    const load = async () => {
      try {
        const res = await vanityApi.getRoutineHistory();
        const items = res.routines;
        setHistory(items);

        const entries = await Promise.allSettled(
          items.map(h => vanityApi.getRoutineDetail(h.recommendation_session_id))
        );
        const map: Record<number, VanityRoutineResult> = {};
        entries.forEach((r, i) => {
          if (r.status === 'fulfilled') map[items[i].recommendation_session_id] = r.value;
        });
        setDetailMap(map);
      } catch { /* no-op */ }
      setLoading(false);
    };
    load();
  }, []);

  if (loading) return (
    <div className="vrh-page">
      <LoadingSpinner text="루틴 기록을 불러오는 중이에요" />
    </div>
  );

  return (
    <div className="vrh-page">
      <div className="vrh-banner">
        <div className="vrh-banner-inner">
          <button className="vrh-back-btn" onClick={() => navigate('/vanity')}>
            <ChevronLeft size={15} />내 화장대로
          </button>
          <div className="vrh-badge">SKIN FIT · ROUTINE</div>
          <h1 className="vrh-title">루틴 기록</h1>
          <p className="vrh-sub">지금까지 추천받은 화장대 루틴을 확인해보세요</p>
        </div>
      </div>

      <div className="vrh-body">
        {history.length === 0 ? (
          <div className="vrh-empty">
            아직 루틴 기록이 없어요.<br />
            화장대 루틴을 추천받으면 여기에 기록이 남아요.
            <div style={{ marginTop: 16 }}>
              <button
                onClick={() => navigate('/vanity/routine/budget')}
                style={{ background:'#7c3aed', color:'#fff', border:'none', borderRadius:8, padding:'10px 24px', fontWeight:700, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif' }}
              >
                루틴 추천 받기 →
              </button>
            </div>
          </div>
        ) : (
          <div className="vrh-list">
            {history.map(h => {
              const detail        = detailMap[h.recommendation_session_id];
              const fixedProducts = detail?.routine_recommendation_results?.fixed_products ?? [];
              const basisDate     = detail?.basis_skin_result?.analyzed_at;
              return (
                <div
                  key={h.recommendation_session_id}
                  className="vrh-card"
                  onClick={() => detail && navigate('/vanity/routine', { state: { result: detail } })}
                  style={{ cursor: detail ? 'pointer' : 'default', opacity: detail ? 1 : 0.6 }}
                >
                  <div className="vrh-card-images">
                    {fixedProducts.length > 0
                      ? fixedProducts.slice(0, 4).map((p, idx) => {
                          const imgSrc = imageMap[p.product_id] || p.image_url;
                          return (
                            <div key={idx} className="vrh-card-img-wrap">
                              {imgSrc
                                ? <img src={imgSrc} alt={p.product_name} className="vrh-card-img" />
                                : <Package size={16} color="#A78BFA" />
                              }
                            </div>
                          );
                        })
                      : <div className="vrh-card-img-wrap"><Package size={20} color="#A78BFA" /></div>
                    }
                  </div>

                  <div className="vrh-card-info">
                    <div className="vrh-card-price">{h.total_price.toLocaleString()}원</div>
                    <div className="vrh-card-fixed">
                      {fixedProducts.length > 0
                        ? `${fixedProducts.map(p => p.product_name).join(' · ')} 고정`
                        : `내 화장대 ${h.fixed_product_count}개 고정`}
                    </div>
                    {basisDate && (
                      <div className="vrh-card-meta">
                        <FlaskConical size={11} color="#7c3aed" />
                        피부 분석 기준 {formatDate(basisDate)}
                      </div>
                    )}
                    <div className="vrh-card-meta" style={{ color:'#9CA3AF' }}>
                      추천일 {formatDate(h.created_at)}
                    </div>
                  </div>

                  <ChevronRight size={18} color="#A78BFA" style={{ flexShrink: 0 }} />
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
