import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MOCK_USER, MOCK_PAST_RESULTS, MOCK_ROUTINES, MOCK_PRODUCTS } from '../mock/Mockdata';
import './MyPage.css';

type Tab = 'info' | 'history' | 'routines' | 'wishlist';

export default function MyPage() {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<Tab>('info');
  const [wishlist] = useState<Set<string>>(new Set(['p001', 'p003']));

  const user = MOCK_USER;
  const pastResults = MOCK_PAST_RESULTS;
  const savedRoutines = MOCK_ROUTINES;
  const wishedProducts = MOCK_PRODUCTS.filter((p) => wishlist.has(p.id));

  const [form, setForm] = useState({
    name: user.name, email: user.email, skinType: user.skinType, age: user.age,
  });

  const TABS: { key: Tab; label: string; icon: string }[] = [
    { key: 'info', label: '내 정보', icon: '👤' },
    { key: 'history', label: '분석 기록', icon: '📋' },
    { key: 'routines', label: '저장된 루틴', icon: '✨' },
    { key: 'wishlist', label: '찜 목록', icon: '💜' },
  ];

  return (
    <div className="my-page">

      <div className="my-banner">
        <div className="my-banner-inner">
          <div className="my-avatar">{user.name[0]}</div>
          <div className="my-banner-info">
            <div className="my-banner-name">{user.name}님</div>
            <div className="my-banner-email">{user.email}</div>
          </div>
        </div>
      </div>

      <div className="my-tabs">
        <div className="my-tabs-inner">
          {TABS.map(({ key, label, icon }) => (
            <button key={key} className={`my-tab ${activeTab === key ? 'active' : ''}`} onClick={() => setActiveTab(key)}>
              <span>{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="my-body">

        {activeTab === 'info' && (
          <div className="my-section">
            <h2 className="my-section-title">내 정보 수정</h2>
            <div className="my-form">
              <div>
                <label className="my-label">이름</label>
                <input className="my-input" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} />
              </div>
              <div>
                <label className="my-label">이메일</label>
                <input className="my-input" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} />
              </div>
              <div>
                <label className="my-label">피부 타입</label>
                <select className="my-input" value={form.skinType} onChange={(e) => setForm({ ...form, skinType: e.target.value })}>
                  {['건성', '지성', '복합성', '중성', '민감성'].map((t) => <option key={t}>{t}</option>)}
                </select>
              </div>
              <div>
                <label className="my-label">나이</label>
                <input className="my-input" type="number" value={form.age} onChange={(e) => setForm({ ...form, age: Number(e.target.value) })} />
              </div>
              <button className="my-save-btn">저장하기</button>
            </div>
          </div>
        )}

        {activeTab === 'history' && (
          <div className="my-section">
            <h2 className="my-section-title">이전 피부 분석 결과</h2>
            {pastResults.length === 0 ? (
              <div className="my-empty">분석 기록이 없어요.</div>
            ) : (
              <div className="my-history-list">
                {pastResults.map((r) => {
                  const d = new Date(r.analyzedAt);
                  return (
                    <div className="my-history-card" key={r.id} onClick={() => navigate('/analysis', { state: { resultId: r.id } })}>
                      <img src={r.thumbnail} alt="썸네일" className="my-history-thumb" />
                      <div className="my-history-info">
                        <div className="my-history-date">{d.getFullYear()}.{d.getMonth() + 1}.{d.getDate()}</div>
                        <div className="my-history-type">{r.skinType}</div>
                        <div className="my-history-comment">{r.aiComment}</div>
                      </div>
                      <span className="my-history-arrow">›</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === 'routines' && (
          <div className="my-section">
            <h2 className="my-section-title">저장된 루틴</h2>
            {savedRoutines.length === 0 ? (
              <div className="my-empty">저장된 루틴이 없어요.</div>
            ) : (
              <div className="my-routine-list">
                {savedRoutines.map((routine) => {
                  const products = routine.products.sort((a, b) => a.step - b.step).map((rp) => MOCK_PRODUCTS.find((p) => p.id === rp.productId)!);
                  return (
                    <div className="my-routine-card" key={routine.id}>
                      <div className="my-routine-header">
                        <span className={`my-routine-badge ${routine.type}`}>{routine.type === 'best' ? '🏆 AI BEST' : '💸 가성비'}</span>
                        <span className="my-routine-cost">{routine.totalCost.toLocaleString()}원</span>
                      </div>
                      <div className="my-routine-products">
                        {products.map((p) => (
                          <div className="my-routine-product" key={p.id} onClick={() => navigate(`/products/${p.id}`)}>
                            <img src={p.imageUrl} alt={p.name} />
                            <span>{p.name}</span>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === 'wishlist' && (
          <div className="my-section">
            <h2 className="my-section-title">찜 목록</h2>
            {wishedProducts.length === 0 ? (
              <div className="my-empty">찜한 제품이 없어요.</div>
            ) : (
              <div className="my-wish-grid">
                {wishedProducts.map((p) => (
                  <div className="my-wish-card" key={p.id} onClick={() => navigate(`/products/${p.id}`)}>
                    <img src={p.imageUrl} alt={p.name} className="my-wish-img" />
                    <div className="my-wish-brand">{p.brand}</div>
                    <div className="my-wish-name">{p.name}</div>
                    <div className="my-wish-price">{p.price.toLocaleString()}원</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
