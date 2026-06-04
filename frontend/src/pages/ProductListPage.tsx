import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { productApi, type ProductSummary } from '../api/product';
import { Heart, Search, X } from 'lucide-react';
import './ProductListPage.css';
import LoadingSpinner from '../components/common/LoadingSpinner';

type CategoryKey =
  | '전체' | '토너' | '에멀젼' | '에센스/앰플' | '크림/젤'
  | '멀티밤' | '아이크림' | '페이셜 오일' | '쉐이빙' | '올인원' | '미스트';

const CATEGORY_CONFIG: { key: CategoryKey; apiCategories: string[] }[] = [
  { key: '전체',       apiCategories: [] },
  { key: '토너',       apiCategories: ['Toner', 'Toner Pads'] },
  { key: '에멀젼',     apiCategories: ['Emulsions'] },
  { key: '에센스/앰플', apiCategories: ['Essences/Ampoules/Serums'] },
  { key: '크림/젤',    apiCategories: ['Cream/Gel'] },
  { key: '멀티밤',     apiCategories: ['Balms/Multi-balms'] },
  { key: '아이크림',   apiCategories: ['Eye Treatments'] },
  { key: '페이셜 오일', apiCategories: ['Facial Oils'] },
  { key: '쉐이빙',     apiCategories: ['Shaving Products'] },
  { key: '올인원',     apiCategories: ['All-In-One'] },
  { key: '미스트',     apiCategories: ['Face Mists'] },
];

export const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼',
  'Essences/Ampoules/Serums': '에센스/앰플/세럼',
  'Cream/Gel': '크림/젤',
  'Balms/Multi-balms': '멀티밤', 'Eye Treatments': '아이크림',
  'Facial Oils': '페이셜 오일', 'Shaving Products': '쉐이빙',
  'All-In-One': '올인원', 'Face Mists': '미스트',
};

export default function ProductListPage() {
  const navigate = useNavigate();

  const PAGE_SIZE = 30;

  const [activeCategory, setActiveCategory] = useState<CategoryKey>('전체');
  const [products, setProducts] = useState<ProductSummary[]>([]);
  const [wishlist, setWishlist] = useState<Set<number>>(new Set());
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [searchQuery, setSearchQuery] = useState('');

  useEffect(() => {
    setLoading(true);
    setPage(1);
    setSearchQuery('');
    const config = CATEGORY_CONFIG.find((c) => c.key === activeCategory)!;
    const fetch = config.apiCategories.length === 0
      ? productApi.getList()
      : Promise.all(config.apiCategories.map((cat) => productApi.getList(cat))).then((r) => r.flat());
    fetch
      .then(setProducts)
      .catch(() => setProducts([]))
      .finally(() => setLoading(false));
  }, [activeCategory]);

  const filteredProducts = searchQuery.trim() === ''
    ? products
    : products.filter((p) => {
        const q = searchQuery.toLowerCase();
        return (
          p.product_name.toLowerCase().includes(q) ||
          p.brand_name.toLowerCase().includes(q)
        );
      });

  // 초기 찜 목록 로드
  useEffect(() => {
    productApi.getWishlist()
      .then((res) => setWishlist(new Set(res.items.map((p) => p.product_id))))
      .catch(() => {});
  }, []);

  const toggleWish = async (productId: number, e: React.MouseEvent) => {
    e.stopPropagation();
    const wished = wishlist.has(productId);
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
          <div className="pl-banner-badge">PRODUCT DATABASE</div>
          <h1 className="pl-banner-title">어떤 제품을 찾고 계세요?</h1>
          <p className="pl-banner-sub">피부 타입, 카테고리, 키워드로 내 피부에 맞는 제품을 찾아보세요</p>
        </div>
      </div>

      <div className="pl-body">
        <div className="pl-filter-row">
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

          <div className="pl-search-wrap">
            <Search size={16} className="pl-search-icon" />
            <input
              className="pl-search-input"
              type="text"
              placeholder="제품명 또는 브랜드 검색"
              value={searchQuery}
              onChange={(e) => { setSearchQuery(e.target.value); setPage(1); }}
            />
            {searchQuery && (
              <button className="pl-search-clear" onClick={() => { setSearchQuery(''); setPage(1); }}>
                <X size={14} />
              </button>
            )}
          </div>
        </div>

        {loading ? (
          <LoadingSpinner text="제품을 불러오는 중이에요" />
        ) : filteredProducts.length === 0 ? (
          <p style={{ textAlign: 'center', color: '#9ca3af', padding: '3rem' }}>
            {searchQuery ? `"${searchQuery}" 검색 결과가 없어요.` : '제품이 없어요.'}
          </p>
        ) : (() => {
          const totalPages = Math.ceil(filteredProducts.length / PAGE_SIZE);
          const paged = filteredProducts.slice((page - 1) * PAGE_SIZE, page * PAGE_SIZE);
          return (
            <>
              <div className="pl-grid">
                {paged.map((product) => (
                  <div className="pl-card" key={product.product_id} onClick={() => navigate(`/products/${product.product_id}`)}>
                    <div className="pl-card-img-wrap">
                      <img src={product.image_url} alt={product.product_name} className="pl-card-img" />
                      <button
                        className={`pl-wish-btn ${wishlist.has(product.product_id) ? 'wished' : ''}`}
                        onClick={(e) => toggleWish(product.product_id, e)}
                      >
                        <Heart size={14} fill={wishlist.has(product.product_id) ? '#7c3aed' : 'none'} color={wishlist.has(product.product_id) ? '#7c3aed' : '#9CA3AF'} />
                      </button>
                      <span className="pl-category-badge">{CATEGORY_KO[product.category] ?? product.category}</span>
                    </div>
                    <div className="pl-card-body">
                      <div className="pl-card-brand">{product.brand_name}</div>
                      <div className="pl-card-name">{product.product_name}</div>
                      <div className="pl-card-tags">
                        {product.tags.filter((t) => t !== product.category).slice(0, 2).map((tag) => <span key={tag} className="pl-tag">#{tag}</span>)}
                      </div>
                      <div className="pl-card-price">{product.price.toLocaleString()}원</div>
                    </div>
                  </div>
                ))}
              </div>
              {totalPages > 1 && (() => {
                const WINDOW = 5;
                const half = Math.floor(WINDOW / 2);
                let start = Math.max(1, page - half);
                let end = Math.min(totalPages, start + WINDOW - 1);
                if (end - start < WINDOW - 1) start = Math.max(1, end - WINDOW + 1);
                const pageNums = Array.from({ length: end - start + 1 }, (_, i) => start + i);
                return (
                  <div className="pl-pagination">
                    <button className="pl-page-nav" onClick={() => setPage(1)} disabled={page === 1}>«</button>
                    <button className="pl-page-nav" onClick={() => setPage((p) => Math.max(1, p - 1))} disabled={page === 1}>‹ Prev</button>
                    {pageNums.map((p) => (
                      <button key={p} className={`pl-page-num ${p === page ? 'active' : ''}`} onClick={() => setPage(p)}>{p}</button>
                    ))}
                    <button className="pl-page-nav" onClick={() => setPage((p) => Math.min(totalPages, p + 1))} disabled={page === totalPages}>Next ›</button>
                    <button className="pl-page-nav" onClick={() => setPage(totalPages)} disabled={page === totalPages}>»</button>
                  </div>
                );
              })()}
            </>
          );
        })()}
      </div>
    </div>
  );
}
