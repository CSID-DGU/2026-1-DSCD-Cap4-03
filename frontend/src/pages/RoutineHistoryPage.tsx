import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { routineApi, type SavedRoutineItem } from '../api/routine';
import { analysisApi, type SkinHistoryItem } from '../api/analysis';
import { Sparkles, Heart, Trophy, Wallet } from 'lucide-react';
import './RoutineHistoryPage.css';

function formatDate(iso: string) {
  const d = new Date(iso);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd} ${hh}:${min}`;
}

export default function RoutineHistoryPage() {
  const navigate = useNavigate();

  const [savedRoutines, setSavedRoutines] = useState<SavedRoutineItem[]>([]);
  const [analysisHistory, setAnalysisHistory] = useState<SkinHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      routineApi.getHistory(),
      analysisApi.getHistory(),
    ]).then(([routineRes, analysisRes]) => {
      setSavedRoutines(routineRes.items);
      setAnalysisHistory(analysisRes.items);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  return (
    <div className="rh-page">

      <div className="rh-banner">
        <div className="rh-banner-inner">
          <div className="rh-badge">내 루틴</div>
          <h1 className="rh-title">저장된 루틴</h1>
          <p className="rh-sub">저장한 루틴을 확인하거나 새로운 루틴을 추천받아보세요</p>
        </div>
      </div>

      <div className="rh-body">

        {/* 새 루틴 추천받기 */}
        <div className="rh-new-section">
          <div className="rh-new-header">
            <div>
              <div className="rh-section-title"><Sparkles size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />새 루틴 추천받기</div>
              <p className="rh-section-sub">어떤 분석 결과를 기반으로 루틴을 추천받을까요?</p>
            </div>
          </div>

          {loading && <p style={{ color: '#7c3aed', padding: '1rem 0' }}>불러오는 중...</p>}

          {!loading && analysisHistory.length === 0 ? (
            <div className="rh-no-result">
              <p>분석 기록이 없어요. 먼저 피부 진단을 받아보세요!</p>
              <button className="rh-btn-primary" onClick={() => navigate('/diagnosis')}>피부 진단 받기</button>
            </div>
          ) : (
            <div className="rh-result-list">
              {[...analysisHistory].reverse().map((r, idx) => (
                <div
                  className="rh-result-card"
                  key={r.result_id}
                  onClick={() => navigate('/routine/budget', { state: { resultId: r.result_id, imageId: r.image_id } })}
                >
                  <div className="rh-result-card-left">
                    {r.image_url && <img src={r.image_url} alt="썸네일" className="rh-result-thumb" />}
                    {idx === 0 && <span className="rh-latest-badge">최신</span>}
                  </div>
                  <div className="rh-result-info">
                    <div className="rh-result-date-row">
                      <div className="rh-result-date">{formatDate(r.analyzed_at)}</div>
                      {r.skin_type && <span className="rh-result-skin-tag">{r.skin_type}</span>}
                    </div>
                    {r.display_scores && Object.keys(r.display_scores).length > 0 && (
                      <div className="rh-result-scores">
                        {([
                          { key: 'acne',         label: '진정' },
                          { key: 'dryness',      label: '수분' },
                          { key: 'sagging',      label: '탄력' },
                          { key: 'pore',         label: '모공' },
                          { key: 'pigmentation', label: '색소침착' },
                          { key: 'wrinkle',      label: '주름' },
                        ] as const).map(({ key, label }, i, arr) => (
                          <span key={key} className="rh-score-chip">
                            {label} <b>{Math.round((1 - (r.display_scores[key] ?? 0)) * 100)}</b>
                            {i < arr.length - 1 && <span className="rh-score-dot">·</span>}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                  <div className="rh-result-cta">
                    <span className="rh-select-btn">이 결과로 추천받기 →</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* 저장된 루틴 목록 */}
        <div className="rh-saved-section">
          <div className="rh-section-title"><Heart size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />저장된 루틴</div>
          <p className="rh-section-sub">마음에 들었던 루틴을 다시 확인해보세요</p>

          {!loading && savedRoutines.length === 0 && (
            <div className="rh-empty"><p>저장된 루틴이 없어요.</p></div>
          )}

          {savedRoutines.length > 0 && (
            <div className="rh-saved-list">
              {savedRoutines.map((routine) => {
                const sortedProducts = [...routine.products].sort((a, b) => a.step - b.step);
                const typeKey = routine.routine_type as 'best' | 'budget';

                return (
                  <div className="rh-saved-card" key={routine.saved_routine_id}>
                    <div className="rh-saved-card-header">
                      <span className={`rh-type-badge ${typeKey}`}>
                        {typeKey === 'best'
                          ? <><Trophy size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} />AI BEST</>
                          : <><Wallet size={12} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 3 }} />가성비</>}
                      </span>
                      <div className="rh-saved-meta">
                        <span>제품 {sortedProducts.length}개</span>
                        <span>·</span>
                        <span>{routine.duration}분</span>
                        <span>·</span>
                        <span className="rh-saved-cost">{routine.total_cost.toLocaleString()}원</span>
                      </div>
                    </div>

                    <div className="rh-saved-products">
                      {sortedProducts.map((p, idx) => (
                        <div
                          className="rh-saved-product"
                          key={p.product_id}
                          onClick={() => navigate(`/products/${p.product_id}`)}
                        >
                          <div className="rh-saved-step">STEP {idx + 1}</div>
                          <img src={p.image_url} alt={p.product_name} className="rh-saved-img" />
                          <div className="rh-saved-brand">{p.brand_name}</div>
                          <div className="rh-saved-name">{p.product_name}</div>
                          <div className="rh-saved-price">{p.price.toLocaleString()}원</div>
                        </div>
                      ))}
                    </div>

                  </div>
                );
              })}
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
