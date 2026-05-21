import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { analysisApi, type SkinHistoryItem } from '../api/analysis';
import { Camera } from 'lucide-react';
import './AnalysisHistoryPage.css';

function formatDate(iso: string) {
  const d = new Date(iso);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd} ${hh}:${min}`;
}

export default function AnalysisHistoryPage() {
  const navigate = useNavigate();

  const [results, setResults] = useState<SkinHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    analysisApi.getHistory()
      .then((res) => setResults(res.items))
      .catch((err) => setError((err as Error).message || '기록을 불러오지 못했어요.'))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="ah-page">

      <div className="ah-banner">
        <div className="ah-banner-inner">
          <div className="ah-badge">분석 기록</div>
          <h1 className="ah-title">피부 분석 기록</h1>
          <p className="ah-sub">이전 분석 결과를 확인하거나 새로 진단받아보세요</p>
        </div>
      </div>

      <div className="ah-body">

        <div className="ah-new-card" onClick={() => navigate('/diagnosis')}>
          <div className="ah-new-icon"><Camera size={28} color="#7c3aed" /></div>
          <div>
            <div className="ah-new-title">새로 피부 진단받기</div>
            <div className="ah-new-sub">최신 피부 상태를 다시 분석해보세요</div>
          </div>
          <span className="ah-new-arrow">→</span>
        </div>

        <div className="ah-section-title">이전 분석 기록</div>

        {loading && <p style={{ textAlign: 'center', color: '#7c3aed', padding: '2rem' }}>불러오는 중...</p>}
        {error && <p style={{ textAlign: 'center', color: '#ef4444', padding: '2rem' }}>{error}</p>}

        {!loading && !error && results.length === 0 && (
          <div className="ah-empty">
            <p>아직 분석 기록이 없어요.</p>
            <button className="ah-btn-primary" onClick={() => navigate('/diagnosis')}>
              첫 피부 진단 받기
            </button>
          </div>
        )}

        {!loading && !error && results.length > 0 && (
          <div className="ah-list">
            {[...results].reverse().map((r, idx) => (
              <div
                className="ah-card"
                key={r.result_id}
                onClick={() => navigate('/analysis', { state: { result_id: r.result_id } })}
              >
                <div className="ah-card-left">
                  {r.image_url && <img src={r.image_url} alt="썸네일" className="ah-thumb" />}
                  {idx === 0 && <span className="ah-latest-badge">최신</span>}
                </div>
                <div className="ah-card-info">
                  <div className="ah-card-date-row">
                    <div className="ah-card-date">{formatDate(r.analyzed_at)}</div>
                    {r.skin_type && <span className="ah-card-skin-tag">{r.skin_type}</span>}
                  </div>
                  {r.display_scores && Object.keys(r.display_scores).length > 0 && (
                    <div className="ah-card-scores">
                      {([
                        { key: 'acne',         label: '진정' },
                        { key: 'dryness',      label: '수분' },
                        { key: 'sagging',      label: '탄력' },
                        { key: 'pore',         label: '모공' },
                        { key: 'pigmentation', label: '색소침착' },
                        { key: 'wrinkle',      label: '주름' },
                      ] as const).map(({ key, label }, i, arr) => (
                        <span key={key} className="ah-score-chip">
                          {label} <b>{Math.round((1 - (r.display_scores[key] ?? 0)) * 100)}</b>
                          {i < arr.length - 1 && <span className="ah-score-dot">·</span>}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <span className="ah-card-arrow">›</span>
              </div>
            ))}
          </div>
        )}

      </div>
    </div>
  );
}
