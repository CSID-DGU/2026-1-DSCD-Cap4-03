import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { productApi, type ProductDetail, type ProductSummary } from '../api/product';
import { Heart, Microscope, ThumbsUp, ThumbsDown, ClipboardList, Timer, LayoutGrid } from 'lucide-react';
import './ProductDetailPage.css';

export default function ProductDetailPage() {
  const { id } = useParams();
  const navigate = useNavigate();

  const [product, setProduct] = useState<ProductDetail | null>(null);
  const [similar, setSimilar] = useState<ProductSummary[]>([]);
  const [wished, setWished] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const productId = Number(id);

  useEffect(() => {
    if (!productId) { setError('잘못된 제품 ID예요.'); setLoading(false); return; }

    const load = async () => {
      try {
        const detail = await productApi.getDetail(productId);
        setProduct(detail);
        try {
          const list = await productApi.getList(detail.category);
          setSimilar(list.filter((p) => p.product_id !== productId).slice(0, 4));
        } catch { /* 비슷한 제품 실패는 무시 */ }
      } catch (err) {
        setError((err as Error).message || '제품 정보를 불러오지 못했어요.');
      } finally {
        setLoading(false);
      }
    };

    load();
    productApi.getWishlist()
      .then((res) => setWished(res.items.some((p) => p.product_id === productId)))
      .catch(() => {});
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [productId]);

  const toggleWish = async () => {
    const prev = wished;
    setWished(!prev);
    try {
      prev
        ? await productApi.removeWishlist(productId)
        : await productApi.addWishlist(productId);
    } catch {
      setWished(prev); // 실패 시 롤백
    }
  };

  if (loading) {
    return <div className="pd-not-found"><p style={{ color: '#7c3aed' }}>불러오는 중...</p></div>;
  }

  if (error || !product) {
    return (
      <div className="pd-not-found">
        <p>{error || '제품을 찾을 수 없어요.'}</p>
        <button onClick={() => navigate('/products')}>목록으로</button>
      </div>
    );
  }

  return (
    <div className="pd-page">
      <button className="pd-back" onClick={() => navigate(-1)}>← 뒤로</button>

      <div className="pd-main">
        <div className="pd-main-inner">
          <div className="pd-img-wrap">
            <img src={product.image_url} alt={product.product_name} className="pd-img" />
            <button className={`pd-wish-btn ${wished ? 'wished' : ''}`} onClick={toggleWish}>
              <Heart size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />{wished ? '찜 완료' : '찜하기'}
            </button>
          </div>
          <div className="pd-info">
            <span className="pd-category-badge">{product.category}</span>
            <div className="pd-brand">{product.brand_name}</div>
            <h1 className="pd-name">{product.product_name}</h1>
            <div className="pd-price">{product.price.toLocaleString()}원</div>
            <div className="pd-tags">
              {product.tags.map((tag) => <span key={tag} className="pd-tag">#{tag}</span>)}
            </div>
          </div>
        </div>
      </div>

      <div className="pd-body">
        <div className="pd-section">
          <h2 className="pd-section-title"><Microscope size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />주요 성분</h2>
          <div className="pd-ingredients">
            {product.ingredients.map((ing) => <span key={ing} className="pd-ingredient">{ing}</span>)}
          </div>
        </div>

        <div className="pd-section pd-procon-wrap">
          <div className="pd-procon pro">
            <h3><ThumbsUp size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />장점</h3>
            {product.pros.map((p) => <div key={p} className="pd-procon-item">✓ {p}</div>)}
          </div>
          <div className="pd-procon con">
            <h3><ThumbsDown size={14} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />단점</h3>
            {product.cons.map((c) => <div key={c} className="pd-procon-item">✗ {c}</div>)}
          </div>
        </div>

        <div className="pd-section">
          <h2 className="pd-section-title"><ClipboardList size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />사용 방법</h2>
          <p className="pd-how-to">{product.how_to_use}</p>
          <div className="pd-apply-time"><Timer size={13} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 4 }} />평균 흡수 시간: <strong>{product.apply_time}</strong></div>
        </div>

        {similar.length > 0 && (
          <div className="pd-section">
            <h2 className="pd-section-title"><LayoutGrid size={16} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />비슷한 제품</h2>
            <div className="pd-similar-grid">
              {similar.map((sp) => (
                <div className="pd-similar-card" key={sp.product_id} onClick={() => navigate(`/products/${sp.product_id}`)}>
                  <img src={sp.image_url} alt={sp.product_name} className="pd-similar-img" />
                  <div className="pd-similar-brand">{sp.brand_name}</div>
                  <div className="pd-similar-name">{sp.product_name}</div>
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
