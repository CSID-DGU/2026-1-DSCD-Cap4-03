import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  vanityApi,
  type SkinMatchLatest,
  FIT_LABEL_KO,
  FIT_LABEL_CLASS,
  REASON_TAG_KO,
} from '../api/vanity';
import { productApi } from '../api/product';
import { FlaskConical, ChevronLeft, Package, Check, AlertTriangle } from 'lucide-react';
import LoadingSpinner from '../components/common/LoadingSpinner';
import './SkinMatchPage.css';

const CATEGORY_KO: Record<string, string> = {
  'Toner': '토너', 'Toner Pads': '토너패드',
  'Emulsions': '에멀젼', 'Essences/Ampoules/Serums': '에센스/앰플/세럼',
  'Cream/Gel': '크림/젤', 'Eye Treatments': '아이크림',
  'Balms/Multi-balms': '멀티밤', 'Facial Oils': '페이셜 오일', 'Face Mists': '미스트',
};

const CONCERN_KO: Record<string, string> = {
  acne: '여드름', pore: '모공', dryness: '건조',
  oiliness: '과잉 피지', sensitivity: '민감성', wrinkle: '주름',
  brightening: '미백', redness: '홍조', elasticity: '탄력', dark_spot: '잡티',
};

const FIT_LABEL_KEYS = ['excellent_match', 'good_match', 'so_so', 'weak_match', 'poor_match'] as const;

function formatDate(str: string) {
  const d = new Date(str);
  if (isNaN(d.getTime())) return str;
  return `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

export default function SkinMatchPage() {
  const navigate = useNavigate();

  const [latestMatch, setLatestMatch] = useState<SkinMatchLatest | null>(null);
  const [imageMap, setImageMap]       = useState<Record<number, string>>({});
  const [loading, setLoading]         = useState(true);

  useEffect(() => {
    const load = async () => {
      try { setLatestMatch(await vanityApi.getLatestSkinMatch()); } catch {}
      try {
        const prods = await productApi.getList();
        const map: Record<number, string> = {};
        prods.forEach(p => { if (p.image_url) map[p.product_id] = p.image_url; });
        setImageMap(map);
      } catch {}
      setLoading(false);
    };
    load();
  }, []);

  if (loading) return (
    <div className="sm-page">
      <LoadingSpinner />
    </div>
  );

  const displayProducts = latestMatch?.product_match_results ?? [];
  const basisInfo       = latestMatch?.basis_skin_result;
  const summary         = latestMatch?.summary;
  const llm             = latestMatch?.llm_explanation?.skin_match;

  return (
    <div className="sm-page">
      <div className="sm-banner">
        <div className="sm-banner-inner">
          <button
            style={{ background:'none', border:'none', color:'#7c3aed', fontSize:13, fontWeight:700, cursor:'pointer', display:'flex', alignItems:'center', gap:4, padding:0, marginBottom:16, fontFamily:'Noto Sans KR, sans-serif' }}
            onClick={() => navigate('/vanity')}
          >
            <ChevronLeft size={15} />내 화장대로
          </button>
          <div className="sm-badge">SKIN FIT · MATCH</div>
          <h1 className="sm-title">이 제품들, 내 피부에 잘 맞을까요?</h1>
          <p className="sm-sub">내 피부 타입과 고민을 기준으로, 지금 쓰는 제품이 얼마나 잘 맞는지 점수로 확인해보세요</p>
        </div>
      </div>

      <div className="sm-body">

        {/* 분석 기준 배지 */}
        {basisInfo && (
          <div className="sm-basis-bar">
            <FlaskConical size={15} color="#7c3aed" />
            <span className="sm-basis-text">
              {formatDate(basisInfo.analyzed_at)} 분석 결과 기준
              {basisInfo.main_concerns && basisInfo.main_concerns.length > 0 &&
                ` · ${basisInfo.main_concerns.map(c => CONCERN_KO[c] ?? c).join(', ')}`
              }
            </span>
          </div>
        )}

        {/* 결과 없음 */}
        {displayProducts.length === 0 && (
          <div className="sm-no-analysis">
            아직 스킨 매치 결과가 없어요.<br />
            내 화장대에 제품을 추가하면 자동으로 적합성 분석이 시작돼요.
            <div style={{ marginTop: 12 }}>
              <button
                onClick={() => navigate('/vanity')}
                style={{ background:'#7c3aed', color:'#fff', border:'none', borderRadius:8, padding:'8px 20px', fontWeight:700, cursor:'pointer', fontFamily:'Noto Sans KR, sans-serif' }}
              >
                내 화장대 가기 →
              </button>
            </div>
          </div>
        )}

        {displayProducts.length > 0 && (
          <>
            {/* 요약 헤더 */}
            <div className="sm-result-header">
              <div className="sm-result-title">분석 결과</div>
              <div className="sm-summary-chips">
                {FIT_LABEL_KEYS.map((key) => {
                  const cnt = summary?.[key] ?? displayProducts.filter(p => p.fit_label === key).length;
                  return cnt > 0 ? (
                    <span key={key} className={`sm-summary-chip ${FIT_LABEL_CLASS[key]}`}>
                      {FIT_LABEL_KO[key]} {cnt}
                    </span>
                  ) : null;
                })}
              </div>
            </div>

            {/* 제품별 결과 카드 */}
            <div className="sm-result-list">
              {[...displayProducts]
                .sort((a, b) => b.vanity_fit_score - a.vanity_fit_score)
                .map((item) => {
                  const cls     = FIT_LABEL_CLASS[item.fit_label] ?? 'good';
                  const imgUrl  = imageMap[item.product_id];
                  const comment = llm?.product_comments?.find(c => c.product_id === item.product_id);
                  const score   = Math.round(item.vanity_fit_score * 100);
                  return (
                    <div key={item.product_id} className="sm-result-card">
                      <div className="sm-card-main">

                        {/* 왼쪽: 이미지 */}
                        <div className="sm-card-img-col">
                          {imgUrl
                            ? <img src={imgUrl} alt={item.product_name} className="sm-card-img" />
                            : <Package size={36} color="#A78BFA" />
                          }
                        </div>

                        {/* 중간: 제품 정보 + 태그 */}
                        <div className="sm-card-content">
                          <div className="sm-result-brand">{item.brand_name}</div>
                          <div className="sm-result-name">{item.product_name}</div>
                          <div className="sm-result-cat">{CATEGORY_KO[item.category] ?? item.category}</div>

                          {(item.reason_tags.length > 0 || item.caution_tags.length > 0) && (
                            <div className="sm-tags-wrap">
                              <div className="sm-reason-label">매칭 이유</div>
                              <div className="sm-tags">
                                {item.reason_tags.map(t => (
                                  <span key={t} className="sm-tag reason">{REASON_TAG_KO[t] ?? t}</span>
                                ))}
                                {item.caution_tags.map(t => (
                                  <span key={t} className="sm-tag caution">{REASON_TAG_KO[t] ?? t}</span>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* 오른쪽: 등급 뱃지 + 큰 점수 */}
                        <div className="sm-card-score-col">
                          <div className={`sm-fit-label ${cls}`}>{item.display_label}</div>
                          <div className={`sm-fit-score ${cls}`}>{score}</div>
                        </div>

                      </div>

                      {/* LLM 코멘트: 카드 하단 전체 너비 */}
                      {comment && (
                        <div className="sm-llm-block">
                          {comment.summary && (
                            <p className="sm-llm-line sm-llm-summary">{comment.summary}</p>
                          )}
                          {comment.fit_reason && (
                            <div className="sm-llm-line sm-llm-reason" style={{ display: 'flex', alignItems: 'flex-start', gap: 5 }}>
                              <Check size={13} color="#16a34a" style={{ flexShrink: 0, marginTop: 3 }} />
                              <span>{comment.fit_reason}</span>
                            </div>
                          )}
                          {comment.caution_comment && (
                            <div className="sm-llm-line sm-llm-caution" style={{ display: 'flex', alignItems: 'flex-start', gap: 5 }}>
                              <AlertTriangle size={13} color="#d97706" style={{ flexShrink: 0, marginTop: 3 }} />
                              <span>{comment.caution_comment}</span>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
            </div>
          </>
        )}

      </div>
    </div>
  );
}
