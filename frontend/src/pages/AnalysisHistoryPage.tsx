import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { analysisApi, type SkinHistoryItem } from '../api/analysis';
import { Camera } from 'lucide-react';
import './AnalysisHistoryPage.css';
import LoadingSpinner from '../components/common/LoadingSpinner';

const TREND_METRICS = [
  { key: 'acne',         label: '트러블' },
  { key: 'dryness',      label: '수분' },
  { key: 'sagging',      label: '탄력' },
  { key: 'pore',         label: '모공' },
  { key: 'pigmentation', label: '색소침착' },
  { key: 'wrinkle',      label: '주름' },
] as const;
type MetricKey = typeof TREND_METRICS[number]['key'];

function formatDate(iso: string) {
  const d = new Date(iso);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd} ${hh}:${min}`;
}

function formatDateShort(iso: string) {
  const d = new Date(iso);
  return `${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`;
}

export default function AnalysisHistoryPage() {
  const navigate = useNavigate();

  const [results, setResults] = useState<SkinHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeMetric, setActiveMetric] = useState<MetricKey>('acne');

  useEffect(() => {
    analysisApi.getHistory()
      .then((res) => setResults(res.items))
      .catch((err) => setError((err as Error).message || '기록을 불러오지 못했어요.'))
      .finally(() => setLoading(false));
  }, []);

  const sortedDesc = [...results].sort(
    (a, b) => new Date(b.analyzed_at).getTime() - new Date(a.analyzed_at).getTime()
  );
  const chartData = [...results]
    .sort((a, b) => new Date(a.analyzed_at).getTime() - new Date(b.analyzed_at).getTime())
    .map(h => ({
      date: formatDateShort(h.analyzed_at),
      value: Math.round((1 - (h.display_scores?.[activeMetric] ?? 0)) * 100),
    }));

  return (
    <div className="ah-page">

      <div className="ah-banner">
        <div className="ah-banner-inner">
          <div className="ah-badge">SKIN REPORT</div>
          <h1 className="ah-title">내 피부는 어떻게 변해왔을까요?</h1>
          <p className="ah-sub">지금까지의 분석 기록을 돌아보고, 오늘 다시 한번 확인해보세요</p>
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

        {loading && <LoadingSpinner text="기록을 불러오는 중이에요" />}
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
          <div className="ah-content">

            {/* 왼쪽: 기록 목록 */}
            <div className="ah-list-col">
              <div className="ah-section-title">이전 분석 기록</div>
              <div className="ah-list">
                {sortedDesc.map((r, idx) => (
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
                            { key: 'acne',         label: '트러블' },
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
            </div>

            {/* 오른쪽: 변화 추이 차트 */}
            <div className="ah-chart-outer">
              <div className="ah-section-title">피부 변화 추이</div>
              <div className="ah-chart-col">

                {/* 지표 탭 */}
                <div className="ah-trend-tabs">
                  {TREND_METRICS.map(m => (
                    <button
                      key={m.key}
                      className={`ah-trend-tab${activeMetric === m.key ? ' active' : ''}`}
                      onClick={() => setActiveMetric(m.key)}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>
                
                {/* 콤팩트 점수 요약 */}
                {chartData.length > 0 && (() => {
                  const cur   = chartData[chartData.length - 1].value;
                  const prev  = chartData.length >= 2 ? chartData[chartData.length - 2].value : null;
                  const diff  = prev !== null ? cur - prev : null;
                  const label = TREND_METRICS.find(m => m.key === activeMetric)?.label ?? '';
                  return (
                    <div className="ah-trend-stat">
                      <span className="ah-trend-stat-label">현재 {label}</span>
                      <span className="ah-trend-stat-score">{cur}점</span>
                      {diff !== null && (
                        <span className={`ah-trend-stat-change ${diff >= 0 ? 'positive' : 'negative'}`}>
                          {diff >= 0 ? '▲' : '▼'} {Math.abs(diff)}점 {diff >= 0 ? '개선' : '감소'}
                        </span>
                      )}
                    </div>
                  );
                })()}

                {/* 그래프 */}
                {results.length < 2 ? (
                  <div className="ah-trend-empty">
                    <p>분석을 <strong>{2 - results.length}회</strong> 더 하면<br />변화 추이를 볼 수 있어요</p>
                  </div>
                ) : (
                  <div className="ah-trend-chart">
                    <ResponsiveContainer width="100%" height={260}>
                      <AreaChart data={chartData} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                        <defs>
                          <linearGradient id="ahPurpleGradient" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.28} />
                            <stop offset="100%" stopColor="#a78bfa" stopOpacity={0.03} />
                          </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f0ebff" />
                        <XAxis dataKey="date" tick={{ fontSize: 11, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
                        <YAxis domain={[0, 100]} tick={{ fontSize: 11, fill: '#9CA3AF' }} axisLine={false} tickLine={false} width={28} />
                        <Tooltip
                          contentStyle={{ borderRadius: 10, border: '1px solid #e5e7eb', fontSize: 12 }}
                          formatter={(v) => [`${v}점`, TREND_METRICS.find(m => m.key === activeMetric)?.label]}
                        />
                        <Area
                          type="monotone" dataKey="value"
                          stroke="#7c3aed" strokeWidth={2.5}
                          fill="url(#ahPurpleGradient)"
                          dot={{ r: 5, fill: '#7c3aed', strokeWidth: 2, stroke: '#fff' }}
                          activeDot={{ r: 7 }}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}

              </div>
            </div>

          </div>
        )}

      </div>
    </div>
  );
}
