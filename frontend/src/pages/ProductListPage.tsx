import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { productApi, type ProductSummary } from '../api/product';
import './ProductListPage.css';

type CategoryKey = '전체' | '토너' | '에멀젼' | '에센스/앰플' | '크림/젤';

const CATEGORY_CONFIG: { key: CategoryKey; apiCategories: string[] }[] = [
  { key: '전체',        apiCategories: [] },
  { key: '토너',        apiCategories: ['Toner', 'Toner Pads'] },
  { key: '에멀젼',      apiCategories: ['Emulsions'] },
  { key: '에센스/앰플', apiCategories: ['Essences/Ampoules/Serums'] },
  { key: '크림/젤',     apiCategories: ['Cream/Gel'] },
];

export const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼',
  'Essences/Ampoules/Serums': '에센스/앰플/세럼',
  'Cream/Gel': '크림/젤',
  'Balms/Multi-balms': '멀티밤', 'Eye Treatments': '아이크림',
  'Facial Oils': '페이셜 오일', 'Shaving Products': '쉐이빙', 'All-In-One': '올인원',
};

export default function ProductListPage() {
  const navigate = useNavigate();

  const [activeCategory, setActiveCategory] = useState<CategoryKey>('전체');
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [wishlist, setWishlist] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    const config = CATEGORY_CONFIG.find((c) => c.key === activeCategory)!;
    const fetch = config.apiCategories.length === 0
      ? productApi.getList()
      : Promise.all(config.apiCategories.map((cat) => productApi.getList(cat))).then((r) => r.flat());
    fetch
      .then(setProducts)
      .catch(() => setProducts([]))
      .finally(() => setLoading(false));
  }, [activeCategory]);

  // 초기 찜 목록 로드
  useEffect(() => {
    productApi.getWishlist()
      .then((res) => setWishlist(new Set(res.items.map((p) => p.product_id))))
      .catch(() => {});
  }, []);

  const toggleWish = async (productId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const wished = wishlist.has(productId);
    // 낙관적 업데이트
    setWishlist((prev) => {
      const next = new Set(prev);
      wished ? next.delete(productId) : next.add(productId);
      return next;
    });
    try {
      wished
        ? await productApi.removeWishlist(productId)
        : await productApi.addWishlist(productId);
    } catch {
      // 실패 시 롤백
      setWishlist((prev) => {
        const next = new Set(prev);
        wished ? next.add(productId) : next.delete(productId);
        return next;
      });
    }
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
          {CATEGORY_CONFIG.map(({ key }) => (
            <button
              key={key}
              className={`pl-tab ${activeCategory === key ? 'active' : ''}`}
              onClick={() => setActiveCategory(key)}
            >
              {key}
            </button>
          ))}
        </div>

        {loading ? (
          <p style={{ textAlign: 'center', color: '#7c3aed', padding: '3rem' }}>불러오는 중...</p>
        ) : products.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#9ca3af', padding: '3rem' }}>제품이 없어요.</p>
        ) : (
          <div className="pl-grid">
            {products.map((product) => (
              <div className="pl-card" key={product.product_id} onClick={() => navigate(`/products/${product.product_id}`)}>
                <div className="pl-card-img-wrap">
                  <img src={product.image_url} alt={product.product_name} className="pl-card-img" />
                  <button
                    className={`pl-wish-btn ${wishlist.has(product.product_id) ? 'wished' : ''}`}
                    onClick={(e) => toggleWish(product.product_id, e)}
                  >
                    {wishlist.has(product.product_id) ? '💜' : '🤍'}
                  </button>
                  <span className="pl-category-badge">{CATEGORY_KO[product.category] ?? product.category}</span>
                </div>
                <div className="pl-card-body">
                  <div className="pl-card-brand">{product.brand_name}</div>
                  <div className="pl-card-name">{product.product_name}</div>
                  <div className="pl-card-tags">
                    {product.tags.slice(0, 2).map((tag) => <span key={tag} className="pl-tag">#{tag}</span>)}
                  </div>
                  <div className="pl-card-price">{product.price.toLocaleString()}원</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
