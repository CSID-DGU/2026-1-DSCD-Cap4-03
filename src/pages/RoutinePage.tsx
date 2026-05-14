import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MOCK_ROUTINES, MOCK_PRODUCTS, MOCK_SKIN_RESULT, MOCK_USER } from '../mock/Mockdata';
import type { RoutineTime } from '../mock/Mockdata';
import './RoutinePage.css';

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일`;
}

const TIME_META: Record<RoutineTime, { label: string; icon: string; color: string; bg: string; desc: string }> = {
  both: { label: 'AM + PM 공용',  icon: '🌅🌙', color: '#7c3aed', bg: '#f5f3ff', desc: '아침·저녁 모두 사용하는 루틴이에요' },
  am:   { label: 'AM 전용',       icon: '🌅',    color: '#f59e0b', bg: '#fffbeb', desc: '아침에만 사용하는 루틴이에요' },
  pm:   { label: 'PM 전용',       icon: '🌙',    color: '#6366f1', bg: '#eef2ff', desc: '저녁에만 사용하는 루틴이에요' },
};

// 옵션 제품 목업 (필수 카테고리 외)
const OPTIONAL_PRODUCTS = [
  {
    id: 'op001', brand: '비플레인', name: '녹두 진정 미스트', category: '미스트',
    price: 15000,
    imageUrl: 'https://placehold.co/200x200/f0fdf4/16a34a?text=미스트',
    applicationGuide: '얼굴에서 20cm 거리에서 고르게 분사한 뒤 손으로 가볍게 눌러 흡수시켜 주세요.',
    tags: ['진정', '미스트'],
  },
  {
    id: 'op002', brand: '아이오페', name: '리얼 아이크림 포 페이스', category: '아이크림',
    price: 32000,
    imageUrl: 'https://placehold.co/200x200/fef9c3/ca8a04?text=아이크림',
    applicationGuide: '눈가에 소량을 덜어 약지로 부드럽게 두드려 흡수시켜 주세요.',
    tags: ['눈가', '탄력'],
  },
  {
    id: 'op003', brand: '닥터자르트', name: '세라마이딘 멀티밤', category: '멀티밤',
    price: 28000,
    imageUrl: 'https://placehold.co/200x200/ffe4e6/e11d48?text=멀티밤',
    applicationGuide: '건조한 부위에 소량을 덜어 부드럽게 펴 바르세요.',
    tags: ['보습', '마무리'],
  },
  {
    id: 'op004', brand: '존말리', name: '로즈힙 오일', category: '오일',
    price: 55000,
    imageUrl: 'https://placehold.co/200x200/fce7f3/be185d?text=오일',
    applicationGuide: '마지막 단계에 2~3방울을 손바닥에 덜어 얼굴에 가볍게 눌러 흡수시켜 주세요.',
    tags: ['항산화', '재생'],
  },
];

export default function RoutinePage() {
  const navigate = useNavigate();
  const result = MOCK_SKIN_RESULT;
  const user = MOCK_USER;

  const [activeType, setActiveType] = useState<'best' | 'budget'>('best');
  const [savedRoutines, setSavedRoutines] = useState<Set<string>>(new Set());


  const routines = MOCK_ROUTINES;
  const currentRoutine = routines.find((r) => r.type === activeType)!;
  const timeMeta = TIME_META[currentRoutine.routineTime];

  const requiredProducts = currentRoutine.products
    .sort((a, b) => a.step - b.step)
    .map((rp) => ({
      ...MOCK_PRODUCTS.find((p) => p.id === rp.productId)!,
      applicationGuide: rp.applicationGuide,
      step: rp.step,
      timeTag: rp.timeTag,
    }));

  const toggleSave = (id: string) => {
    setSavedRoutines((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  const isSaved = savedRoutines.has(currentRoutine.id);

  return (
    <div className="rp-page">

      {/* ── HERO ── */}
      <section className="rp-hero">
        <div className="rp-hero-inner">
          <div className="rp-hero-badge">맞춤 루틴</div>
          <h1 className="rp-hero-title">
            <span className="rp-hero-name">{user.name}</span>님을 위한 루틴
          </h1>
          <p className="rp-hero-sub">AI가 분석한 피부 상태에 맞는 제품을 추천해드려요</p>
          <p className="rp-hero-date">분석일 : {formatDate(result.generated_at)}</p>
          <div className="rp-skin-type-badge">{result.skinType}</div>
        </div>
      </section>

      {/* ── 루틴 탭 (항상 2개 고정) ── */}
      <div className="rp-tabs-wrap">
        <div className="rp-tabs">
          {routines.map((r) => (
            <button
              key={r.type}
              className={`rp-tab ${activeType === r.type ? 'active' : ''}`}
              onClick={() => { setActiveType(r.type); }}
            >
              {r.type === 'best' ? '🏆 AI 추천' : '💸 가성비'}
            </button>
          ))}
        </div>
      </div>

      <div className="rp-body">

        {/* ── 루틴 요약 바 ── */}
        <div className="rp-summary-bar">
          {/* 시간대 뱃지 */}
          <div className="rp-time-badge" style={{ background: timeMeta.bg, color: timeMeta.color }}>
            <span>{timeMeta.icon}</span>
            <span>{timeMeta.label}</span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <span>📦</span>
            <span>제품 <strong>{requiredProducts.length}개</strong></span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <span>⏱️</span>
            <span>소요 <strong>{currentRoutine.duration}분</strong></span>
          </div>
          <div className="rp-summary-divider" />
          <div className="rp-summary-item">
            <span>💰</span>
            <span>총 <strong>{currentRoutine.totalCost.toLocaleString()}원</strong></span>
          </div>
          <button
            className={`rp-save-btn ${isSaved ? 'saved' : ''}`}
            onClick={() => toggleSave(currentRoutine.id)}
          >
            {isSaved ? '💜 저장됨' : '🤍 루틴 저장'}
          </button>
        </div>

        {/* ── AI 루틴 설명 ── */}
        <div className="rp-ai-desc-box">
          <div className="rp-ai-desc-header">
            <span className="rp-ai-icon">🤖</span>
            <span className="rp-ai-title">AI 루틴 추천 이유</span>
            <div className="rp-time-tag" style={{ background: timeMeta.bg, color: timeMeta.color }}>
              {timeMeta.icon} {timeMeta.desc}
            </div>
          </div>
          <p className="rp-ai-desc-text">{currentRoutine.aiDescription}</p>
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

          {/* 스텝 인디케이터 */}
          <div className="rp-steps-indicator">
            {requiredProducts.map((p, idx) => (
              <div key={p.id} className="rp-step-indicator-item">
                <div className="rp-step-dot">{idx + 1}</div>
                <span>{p.category}</span>
                {idx < requiredProducts.length - 1 && <div className="rp-step-line" />}
              </div>
            ))}
          </div>

          {/* 제품 가로 스크롤 */}
          <div className="rp-scroll">
            {requiredProducts.map((product, idx) => (
              <div key={product.id} className="rp-product-card" onClick={() => navigate(`/products/${product.id}`)}>
                {/* 상단 행: 왼쪽 STEP+카테고리, 오른쪽 시간 뱃지 */}
                <div className="rp-card-header">
                  <div className="rp-card-header-left">
                    <div className="rp-product-step">STEP {idx + 1}</div>
                    <div className="rp-product-category">{product.category}</div>
                  </div>
                  {product.timeTag && (
                    <div className={`rp-product-time-tag ${product.timeTag}`}>
                      {product.timeTag === 'am' ? '🌅 AM' : '🌙 PM'}
                    </div>
                  )}
                </div>
                <img src={product.imageUrl} alt={product.name} className="rp-product-img" />
                <div className="rp-product-brand">{product.brand}</div>
                <div className="rp-product-name">{product.name}</div>
                <div className="rp-product-tags">
                  {product.tags.slice(0, 2).map((t) => <span key={t} className="rp-product-tag">#{t}</span>)}
                </div>
                <div className="rp-product-price">{product.price.toLocaleString()}원</div>
                {/* 바르는 법 항상 노출 */}
                <div className="rp-guide-panel" onClick={(e) => e.stopPropagation()}>
                  <div className="rp-guide-header">
                    <span>📝</span>
                    <span>바르는 법</span>
                  </div>
                  <p className="rp-guide-text">{product.applicationGuide}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── 옵션 루틴 ── */}
        <div className="rp-routine-block rp-optional-block">
          <div className="rp-block-header">
            <div className="rp-block-title-wrap">
              <span className="rp-block-badge optional">옵션</span>
              <h2 className="rp-block-title">추가 케어 루틴</h2>
            </div>
            <p className="rp-block-sub">미스트 · 아이크림 · 멀티밤 · 오일</p>
          </div>

          <div className="rp-scroll">
            {OPTIONAL_PRODUCTS.map((product) => (
              <div key={product.id} className="rp-product-card rp-optional-card" onClick={() => navigate(`/products/${product.id}`)}>
                <div className="rp-product-category optional">{product.category}</div>
                <img src={product.imageUrl} alt={product.name} className="rp-product-img" />
                <div className="rp-product-brand">{product.brand}</div>
                <div className="rp-product-name">{product.name}</div>
                <div className="rp-product-tags">
                  {product.tags.slice(0, 2).map((t) => <span key={t} className="rp-product-tag optional">#{t}</span>)}
                </div>
                <div className="rp-product-price">{product.price.toLocaleString()}원</div>
                <div className="rp-guide-panel optional" onClick={(e) => e.stopPropagation()}>
                  <div className="rp-guide-header">
                    <span>📝</span>
                    <span>바르는 법</span>
                  </div>
                  <p className="rp-guide-text">{product.applicationGuide}</p>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* ── 저장 CTA ── */}
        <div className="rp-bottom-cta">
          <button
            className={`rp-full-save-btn ${isSaved ? 'saved' : ''}`}
            onClick={() => toggleSave(currentRoutine.id)}
          >
            {isSaved ? '💜 루틴 저장 완료!' : '💜 현재 루틴 저장하기'}
          </button>
        </div>

      </div>

      <footer className="rp-footer">
        <span>© 2026 ROUPLE AI 기반 맞춤형 스킨케어 솔루션</span>
        <span>개인정보처리방침 · 이용약관</span>
      </footer>
    </div>
  );
}
