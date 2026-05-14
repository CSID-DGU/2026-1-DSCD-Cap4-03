import { useNavigate } from 'react-router-dom';
import { MOCK_ROUTINES, MOCK_PRODUCTS, MOCK_PAST_RESULTS } from '../mock/Mockdata';
import './RoutineHistoryPage.css';

function formatDate(iso: string) {
  const d = new Date(iso);
  return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일`;
}

export default function RoutineHistoryPage() {
  const navigate = useNavigate();
  // 나중에 API: const savedRoutines = await fetchSavedRoutines();
  const savedRoutines = MOCK_ROUTINES;
  const pastResults = MOCK_PAST_RESULTS;

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

        {/* 새 루틴 추천받기 섹션 */}
        <div className="rh-new-section">
          <div className="rh-new-header">
            <div>
              <div className="rh-section-title">✨ 새 루틴 추천받기</div>
              <p className="rh-section-sub">어떤 분석 결과를 기반으로 루틴을 추천받을까요?</p>
            </div>
          </div>

          {pastResults.length === 0 ? (
            <div className="rh-no-result">
              <p>분석 기록이 없어요. 먼저 피부 진단을 받아보세요!</p>
              <button className="rh-btn-primary" onClick={() => navigate('/diagnosis')}>
                피부 진단 받기
              </button>
            </div>
          ) : (
            <div className="rh-result-list">
              {[...pastResults].reverse().map((r, idx) => (
                <div
                  className="rh-result-card"
                  key={r.id}
                  onClick={() => navigate('/routine/budget', { state: { resultId: r.id } })}
                >
                  <div className="rh-result-card-left">
                    <img src={r.thumbnail} alt="썸네일" className="rh-result-thumb" />
                    {idx === 0 && <span className="rh-latest-badge">최신</span>}
                  </div>
                  <div className="rh-result-info">
                    <div className="rh-result-date">{formatDate(r.analyzedAt)} 분석</div>
                    <div className="rh-result-type">{r.skinType}</div>
                    <div className="rh-result-comment">{r.aiComment}</div>
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
          <div className="rh-section-title">💜 저장된 루틴</div>
          <p className="rh-section-sub">마음에 들었던 루틴을 다시 확인해보세요</p>

          {savedRoutines.length === 0 ? (
            <div className="rh-empty">
              <p>저장된 루틴이 없어요.</p>
            </div>
          ) : (
            <div className="rh-saved-list">
              {savedRoutines.map((routine) => {
                const products = routine.products
                  .sort((a, b) => a.step - b.step)
                  .map((rp) => MOCK_PRODUCTS.find((p) => p.id === rp.productId)!);

                return (
                  <div className="rh-saved-card" key={routine.id}>
                    <div className="rh-saved-card-header">
                      <span className={`rh-type-badge ${routine.type}`}>
                        {routine.type === 'best' ? '🏆 AI BEST' : '💸 가성비'}
                      </span>
                      <div className="rh-saved-meta">
                        <span>제품 {products.length}개</span>
                        <span>·</span>
                        <span>{routine.duration}분</span>
                        <span>·</span>
                        <span className="rh-saved-cost">{routine.totalCost.toLocaleString()}원</span>
                      </div>
                    </div>

                    <div className="rh-saved-products">
                      {products.map((p, idx) => (
                        <div
                          className="rh-saved-product"
                          key={p.id}
                          onClick={() => navigate(`/products/${p.id}`)}
                        >
                          <div className="rh-saved-step">STEP {idx + 1}</div>
                          <img src={p.imageUrl} alt={p.name} className="rh-saved-img" />
                          <div className="rh-saved-brand">{p.brand}</div>
                          <div className="rh-saved-name">{p.name}</div>
                          <div className="rh-saved-price">{p.price.toLocaleString()}원</div>
                        </div>
                      ))}
                    </div>

                    <button
                      className="rh-view-btn"
                      onClick={() => navigate('/routine/result', { state: { routineId: routine.id } })}
                    >
                      루틴 자세히 보기
                    </button>
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
