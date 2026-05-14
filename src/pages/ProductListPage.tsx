import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { MOCK_PRODUCTS } from '../mock/Mockdata';
import './ProductListPage.css';

const CATEGORIES = ['전체', '토너', '에멀젼', '앰플', '크림'] as const;
type Category = typeof CATEGORIES[number];

export default function ProductListPage() {
  const navigate = useNavigate();
  const [activeCategory, setActiveCategory] = useState<Category>('전체');
  const [wishlist, setWishlist] = useState<Set<string>>(new Set());

  const filtered = activeCategory === '전체'
    ? MOCK_PRODUCTS
    : MOCK_PRODUCTS.filter((p) => p.category === activeCategory);

  const toggleWish = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setWishlist((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  };

  return (
    <div className="product-list-page">

      <div className="pl-banner">
        <div className="pl-banner-inner">
          <div className="pl-banner-badge">PRODUCTS</div>
          <h1 className="pl-banner-title">전체 제품</h1>
          <p className="pl-banner-sub">내 피부에 맞는 제품을 찾아보세요</p>
        </div>
      </div>

      <div className="pl-body">
        <div className="pl-tabs">
          {CATEGORIES.map((cat) => (
            <button
              key={cat}
              className={`pl-tab ${activeCategory === cat ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat)}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="pl-grid">
          {filtered.map((product) => (
            <div className="pl-card" key={product.id} onClick={() => navigate(`/products/${product.id}`)}>
              <div className="pl-card-img-wrap">
                <img src={product.imageUrl} alt={product.name} className="pl-card-img" />
                <button className={`pl-wish-btn ${wishlist.has(product.id) ? 'wished' : ''}`} onClick={(e) => toggleWish(product.id, e)}>
                  {wishlist.has(product.id) ? '💜' : '🤍'}
                </button>
                <span className="pl-category-badge">{product.category}</span>
              </div>
              <div className="pl-card-body">
                <div className="pl-card-brand">{product.brand}</div>
                <div className="pl-card-name">{product.name}</div>
                <div className="pl-card-tags">
                  {product.tags.slice(0, 2).map((tag) => <span key={tag} className="pl-tag">#{tag}</span>)}
                </div>
                <div className="pl-card-price">{product.price.toLocaleString()}원</div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
