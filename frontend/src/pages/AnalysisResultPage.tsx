import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, PolarRadiusAxis,
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
} from 'recharts';
import { Bot, Wand2 } from 'lucide-react';
import { analysisApi, type AnalysisResult, type SkinHistoryItem } from '../api/analysis';
import { userApi } from '../api/user';
import { useAuth } from '../context/useAuth';
import LoadingSpinner from '../components/common/LoadingSpinner';
import './AnalysisResultPage.css';

const CONCERN_KO: Record<string, string> = {
  acne: '여드름', wrinkle: '주름', brightening: '미백', sebum: '피지',
  dryness: '속건조', redness: '붉은기', dark_circle: '다크서클', atopy: '아토피',
  sensitive: '민감성', pore: '모공', flushing: '홍조', keratin: '각질',
};

// ── 등급 변환 (1-score 기준, 높을수록 좋음)
function toGrade(score: number): Grade {
  return score >= 70 ? '좋음' : score >= 50 ? '보통' : score >= 30 ? '개선필요' : '집중케어';
}

type Grade = '좋음' | '보통' | '개선필요' | '집중케어';

const TREND_METRICS = [
  { key: 'acne',         label: '진정' },
  { key: 'dryness',      label: '수분' },
  { key: 'sagging',      label: '탄력' },
  { key: 'pore',         label: '모공' },
  { key: 'pigmentation', label: '색소침착' },
  { key: 'wrinkle',      label: '주름' },
] as const;
type MetricKey = typeof TREND_METRICS[number]['key'];

const GRADE_COLOR: Record<Grade, string> = {
  좋음: '#16a34a', 보통: '#eab308', 개선필요: '#d97706', 집중케어: '#dc2626',
};
const GRADE_BG: Record<Grade, string> = {
  좋음: '#f0fdf4', 보통: '#fefce8', 개선필요: '#fffbeb', 집중케어: '#fef2f2',
};

function buildMetrics(result: AnalysisResult) {
  const raw = result.display_scores;
  const comments = result.indicator_comments;

  const s = {
    acne:         Math.round((1 - raw.acne)         * 100),
    dryness:      Math.round((1 - raw.dryness)      * 100),
    sagging:      Math.round((1 - raw.sagging)      * 100),
    pore:         Math.round((1 - raw.pore)         * 100),
    pigmentation: Math.round((1 - raw.pigmentation) * 100),
    wrinkle:      Math.round((1 - raw.wrinkle)      * 100),
  };

  return [
    { key: 'acne',         label: '진정',    grade: toGrade(s.acne),         displayVal: s.acne,         desc: comments.acne },
    { key: 'dryness',      label: '수분',    grade: toGrade(s.dryness),      displayVal: s.dryness,      desc: comments.dryness },
    { key: 'sagging',      label: '탄력',    grade: toGrade(s.sagging),      displayVal: s.sagging,      desc: comments.sagging },
    { key: 'pore',         label: '모공',    grade: toGrade(s.pore),         displayVal: s.pore,         desc: comments.pore },
    { key: 'pigmentation', label: '색소침착', grade: toGrade(s.pigmentation), displayVal: s.pigmentation, desc: comments.pigmentation },
    { key: 'wrinkle',      label: '주름',    grade: toGrade(s.wrinkle),      displayVal: s.wrinkle,      desc: comments.wrinkle },
  ];
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  const mm = String(d.getMonth() + 1).padStart(2, '0');
  const dd = String(d.getDate()).padStart(2, '0');
  const hh = String(d.getHours()).padStart(2, '0');
  const min = String(d.getMinutes()).padStart(2, '0');
  return `${d.getFullYear()}-${mm}-${dd} ${hh}:${min}`;
}

export default function AnalysisResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const { nickname } = useAuth();

  const resultId: number = location.state?.result_id ?? 1;
  const passedImageUrl: string = location.state?.imageUrl ?? '';

  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<SkinHistoryItem[]>([]);
  const [activeMetric, setActiveMetric] = useState<MetricKey>('acne');
  const [skinConcerns, setSkinConcerns] = useState<string[]>([]);

  useEffect(() => {
    analysisApi.getResult(resultId)
      .then(setResult)
      .catch((err) => setError((err as Error).message || '분석 결과를 불러오지 못했어요.'))
      .finally(() => setLoading(false));

    analysisApi.getHistory()
      .then((res) => {
        const sorted = [...res.items].sort(
          (a, b) => new Date(a.analyzed_at).getTime() - new Date(b.analyzed_at).getTime()
        );
        setHistory(sorted);
      })
      .catch(() => {});

    userApi.getMe()
      .then((user) => {
        const raw = user.skin_concerns?.filter((c) => c !== 'none') ?? [];
        const unique = [...new Set(raw)];
        setSkinConcerns(unique.map((c) => CONCERN_KO[c] ?? c));
      })
      .catch(() => {});
  }, [resultId]);

  if (loading) {
    return (
      <div className="ar-page">
        <LoadingSpinner text="분석 결과를 불러오는 중이에요" />
      </div>
    );
  }

  if (error || !result) {
    return (
      <div className="ar-page" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', gap: '1rem' }}>
        <p style={{ color: '#ef4444' }}>{error || '결과를 불러올 수 없어요.'}</p>
        <button onClick={() => navigate(-1)} style={{ padding: '0.5rem 1.5rem', borderRadius: '8px', background: '#7c3aed', color: '#fff', border: 'none', cursor: 'pointer' }}>← 돌아가기</button>
      </div>
    );
  }

  const metrics = buildMetrics(result);
  const radarData = metrics.map((m) => ({ metric: m.label, value: m.displayVal, fullMark: 100 }));
  const imageUrl = result.image_url || passedImageUrl;
  const displayName = nickname || '내';
  const skinType = result.skin_type;

  const gradeCounts: Record<Grade, number> = { 좋음: 0, 보통: 0, 개선필요: 0, 집중케어: 0 };
  metrics.forEach((m) => { gradeCounts[m.grade]++; });

  return (
    <div className="ar-page">

      {/* ── HERO ── */}
      <section className="ar-hero">
        <div className="ar-hero-inner">
          <div className="ar-hero-left">
            <div className="ar-hero-badge">AI 피부 분석 완료</div>
            <h1 className="ar-hero-title">
              <span className="ar-hero-name">{displayName}</span>님,<br />
              <span className="ar-hero-highlight">스킨 리포트</span>가<br />
              도착했어요
            </h1>
            <p className="ar-hero-date">분석일 : {formatDate(result.analyzed_at || result.generated_at)}</p>
            <div className="ar-skin-tags">
              {skinType && <span className="ar-skin-tag ar-skin-tag--type">{skinType}</span>}
              {skinConcerns.map((t) => <span key={t} className="ar-skin-tag">{t}</span>)}
            </div>
            <div className="ar-hero-btns">
              <button className="ar-btn-primary" onClick={() => navigate('/routine/budget', { state: { resultId: result.result_id, imageId: result.image_id } })}>
                <Wand2 size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />루틴 추천받기
              </button>
              <button className="ar-btn-ghost" onClick={() => navigate('/analysis-history')}>
                이전 분석 보기
              </button>
            </div>
          </div>

          <div className="ar-hero-right">
            <div className="ar-photo-ring">
              <img src={imageUrl} alt="피부 사진" className="ar-photo" />
            </div>
          </div>
        </div>
      </section>

      {/* ── 피부 지표 + AI 한마디 ── */}
      <section className="ar-report-section">
        <div className="ar-section-inner">

          <div className="ar-sec-badge">피부 지표 분석</div>
          <h2 className="ar-sec-title">내 피부 지표 리포트</h2>
          <p className="ar-sec-sub">
            높은 점수일수록 피부 상태가 좋아요 &nbsp;
            <span className="ar-grade-legend">
              <span style={{ color: GRADE_COLOR['좋음'] }}>● 좋음</span>
              <span style={{ color: GRADE_COLOR['보통'] }}>● 보통</span>
              <span style={{ color: GRADE_COLOR['개선필요'] }}>● 개선필요</span>
              <span style={{ color: GRADE_COLOR['집중케어'] }}>● 집중케어</span>
            </span>
          </p>


          {/* 레이더 + 지표 카드 */}
          <div className="ar-report-body">
            <div className="ar-radar-col">
              <div className="ar-radar-wrap">
                <ResponsiveContainer width="100%" height={460}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="78%">
                    <PolarGrid stroke="#ddd6fe" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#4b5563', fontSize: 15, fontWeight: 600 }} />
                    <PolarRadiusAxis domain={[0, 100]} tick={false} axisLine={false} />
                    <Radar dataKey="value" stroke="#7c3aed" fill="#a855f7" fillOpacity={0.28} strokeWidth={2} />
                  </RadarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="ar-metric-col">
              <div className="ar-metric-grid">
                {metrics.map((m) => {
                  const color = GRADE_COLOR[m.grade];
                  const bg = GRADE_BG[m.grade];
                  return (
                    <div className="ar-metric-card" key={m.key} style={{ borderTop: `3px solid ${color}` }}>
                      <div className="ar-metric-card-top">
                        <span style={{ width: 10, height: 10, borderRadius: '50%', background: color, flexShrink: 0, display: 'inline-block' }} />
                        <div className="ar-metric-label-wrap">
                          <div className="ar-metric-name">{m.label}</div>
                          <div className="ar-metric-grade-badge" style={{ background: bg, color }}>{m.grade}</div>
                        </div>
                        <div className="ar-metric-score" style={{ color }}>{m.displayVal}</div>
                      </div>
                      <div className="ar-metric-bar-bg">
                        <div className="ar-metric-bar-fill" style={{ width: `${m.displayVal}%`, background: color }} />
                      </div>
                      <p className="ar-metric-desc">{m.desc}</p>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* AI 한마디 */}
          <div className="ar-ai-box">
            <div className="ar-ai-label"><Bot size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />AI 한마디</div>
            <p className="ar-ai-text">{result.summary_comment}</p>
          </div>

          {/* 피부 변화 추이 */}
          <div className="ar-trend-box">
            <div className="ar-trend-header">
              <div className="ar-sec-badge" style={{ marginBottom: 4 }}>변화 추이</div>
              <h3 className="ar-trend-title">피부 변화 추이</h3>
              <p className="ar-trend-sub">지표를 선택하면 날짜별 점수 변화를 확인할 수 있어요</p>
            </div>

            {/* 지표 탭 */}
            <div className="ar-trend-tabs">
              {TREND_METRICS.map((m) => (
                <button
                  key={m.key}
                  className={`ar-trend-tab${activeMetric === m.key ? ' active' : ''}`}
                  onClick={() => setActiveMetric(m.key)}
                >
                  {m.label}
                </button>
              ))}
            </div>

            {/* 점수 요약 */}
            {history.length >= 1 && (() => {
              const idx  = history.findIndex(h => h.result_id === result.result_id);
              const cur  = idx >= 0
                ? Math.round((1 - (history[idx].display_scores?.[activeMetric] ?? 0)) * 100)
                : Math.round((1 - (result.display_scores?.[activeMetric] ?? 0)) * 100);
              const prev = idx > 0
                ? Math.round((1 - (history[idx - 1].display_scores?.[activeMetric] ?? 0)) * 100)
                : null;
              const diff  = prev !== null ? cur - prev : null;
              const label = TREND_METRICS.find(m => m.key === activeMetric)?.label ?? '';
              return (
                <div className="ar-trend-stat">
                  <span className="ar-trend-stat-label">현재 {label}</span>
                  <span className="ar-trend-stat-score">{cur}점</span>
                  {diff !== null && (
                    <span className={`ar-trend-stat-change ${diff >= 0 ? 'positive' : 'negative'}`}>
                      {diff >= 0 ? '▲' : '▼'} {Math.abs(diff)}점 {diff >= 0 ? '개선' : '감소'}
                    </span>
                  )}
                </div>
              );
            })()}

            {/* 그래프 */}
            {history.length < 2 ? (
              <div className="ar-trend-empty">
                <p> 분석을 <strong>{2 - history.length}회</strong> 더 하면 변화 추이를 볼 수 있어요</p>
              </div>
            ) : (
              <div className="ar-trend-chart">
                <ResponsiveContainer width="100%" height={240}>
                  <AreaChart
                    data={history.map((h) => ({
                      date: (() => { const d = new Date(h.analyzed_at); return `${String(d.getFullYear()).slice(2)}-${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`; })(),
                      value: Math.round((1 - (h.display_scores?.[activeMetric] ?? 0)) * 100),
                    }))}
                    margin={{ top: 10, right: 24, left: 0, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="purpleAreaGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#a78bfa" stopOpacity={0.28} />
                        <stop offset="100%" stopColor="#a78bfa" stopOpacity={0.03} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f0ebff" />
                    <XAxis dataKey="date" tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 12, fill: '#9CA3AF' }} axisLine={false} tickLine={false} width={32} />
                    <Tooltip
                      contentStyle={{ borderRadius: 10, border: '1px solid #e5e7eb', fontSize: 13 }}
                      formatter={(v: unknown) => [`${v}점`, TREND_METRICS.find(m => m.key === activeMetric)?.label]}
                    />
                    <Area
                      type="monotone" dataKey="value"
                      stroke="#7c3aed" strokeWidth={2.5}
                      fill="url(#purpleAreaGradient)"
                      dot={{ r: 5, fill: '#7c3aed', strokeWidth: 2, stroke: '#fff' }}
                      activeDot={{ r: 7 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>

        </div>
      </section>

      {/* ── FOOTER CTA ── */}
      <section className="ar-cta-section">
        <div className="ar-sec-badge" style={{ background: 'rgba(255,255,255,0.2)', color: '#fff' }}>맞춤 추천</div>
        <h2>{displayName}님을 위한 루틴</h2>
        <p>AI가 분석한 피부 상태에 맞는 제품을 추천해드려요</p>
        <button className="ar-btn-white" onClick={() => navigate('/routine/budget', { state: { resultId: result.result_id, imageId: result.image_id } })}>
          <Wand2 size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />맞춤 루틴 바로 받기
        </button>
      </section>

    </div>
  );
}
