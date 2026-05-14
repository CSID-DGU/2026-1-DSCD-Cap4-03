import { useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { MOCK_PRODUCTS } from '../mock/Mockdata';
import './ProductDetailPage.css';

export default function ProductDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();
  const product = MOCK_PRODUCTS.find((p) => p.id === id);
  const [wished, setWished] = useState(false);

  if (!product) return (
    <div className="pd-not-found">
      <p>제품을 찾을 수 없어요.</p>
      <button onClick={() => navigate('/products')}>목록으로</button>
    </div>
  );

  const similar = MOCK_PRODUCTS.filter((p) => p.category === product.category && p.id !== product.id).slice(0, 4);

  return (
    <div className="pd-page">
      <button className="pd-back" onClick={() => navigate(-1)}>← 뒤로</button>

      <div className="pd-main">
        <div className="pd-main-inner">
          <div className="pd-img-wrap">
            <img src={product.imageUrl} alt={product.name} className="pd-img" />
            <button className={`pd-wish-btn ${wished ? 'wished' : ''}`} onClick={() => setWished(!wished)}>
              {wished ? '💜 찜 완료' : '🤍 찜하기'}
            </button>
          </div>
          <div className="pd-info">
            <span className="pd-category-badge">{product.category}</span>
            <div className="pd-brand">{product.brand}</div>
            <h1 className="pd-name">{product.name}</h1>
            <div className="pd-price">{product.price.toLocaleString()}원</div>
            <div className="pd-tags">
              {product.tags.map((tag) => <span key={tag} className="pd-tag">#{tag}</span>)}
            </div>
          </div>
        </div>
      </div>

      <div className="pd-body">
        <div className="pd-section">
          <h2 className="pd-section-title">🧪 주요 성분</h2>
          <div className="pd-ingredients">
            {product.ingredients.map((ing) => <span key={ing} className="pd-ingredient">{ing}</span>)}
          </div>
        </div>

        <div className="pd-section pd-procon-wrap">
          <div className="pd-procon pro">
            <h3>👍 장점</h3>
            {product.pros.map((p) => <div key={p} className="pd-procon-item">✓ {p}</div>)}
          </div>
          <div className="pd-procon con">
            <h3>👎 단점</h3>
            {product.cons.map((c) => <div key={c} className="pd-procon-item">✗ {c}</div>)}
          </div>
        </div>

        <div className="pd-section">
          <h2 className="pd-section-title">📋 사용 방법</h2>
          <p className="pd-how-to">{product.howToUse}</p>
          <div className="pd-apply-time">⏱ 평균 흡수 시간: <strong>{product.applyTime}</strong></div>
        </div>

        {similar.length > 0 && (
          <div className="pd-section">
            <h2 className="pd-section-title">🔗 비슷한 제품</h2>
            <div className="pd-similar-grid">
              {similar.map((sp) => (
                <div className="pd-similar-card" key={sp.id} onClick={() => navigate(`/products/${sp.id}`)}>
                  <img src={sp.imageUrl} alt={sp.name} className="pd-similar-img" />
                  <div className="pd-similar-brand">{sp.brand}</div>
                  <div className="pd-similar-name">{sp.name}</div>
                  <div className="pd-similar-price">{sp.price.toLocaleString()}원</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
