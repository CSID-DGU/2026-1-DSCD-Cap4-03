import { useLocation, useNavigate } from 'react-router-dom';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, PolarRadiusAxis,
} from 'recharts';
import { MOCK_SKIN_RESULT, MOCK_USER } from '../mock/Mockdata';
import './AnalysisResultPage.css';

// 나중에 API 연결 시: const result = await fetchSkinResult(resultId);
const result = MOCK_SKIN_RESULT;
const user = MOCK_USER;

// ── 등급 변환 함수
function getAcneGrade(v: number)         { return v === 0 ? '양호' : v === 1 ? '보통' : '개선필요'; }
function getDrynessGrade(v: number)      { return v <= 1 ? '양호' : v === 2 ? '보통' : '개선필요'; }
function getSaggingGrade(v: number)      { return v <= 1 ? '양호' : v <= 3 ? '보통' : '개선필요'; }
function getPoreGrade(v: number)         { return v <= 1 ? '양호' : v <= 3 ? '보통' : '개선필요'; }
function getPigmentationGrade(v: number) { return v === 0 ? '양호' : v <= 3 ? '보통' : '개선필요'; }
function getWrinkleGrade(v: number)      { return v <= 1 ? '양호' : v <= 3 ? '보통' : '개선필요'; }

type Grade = '양호' | '보통' | '개선필요';

const GRADE_COLOR: Record<Grade, string> = {
  양호: '#22c55e', 보통: '#f59e0b', 개선필요: '#ef4444',
};
const GRADE_BG: Record<Grade, string> = {
  양호: '#f0fdf4', 보통: '#fffbeb', 개선필요: '#fef2f2',
};

interface MetricInfo {
  key: keyof typeof result.indicator_comments;
  label: string;
  icon: string;
  grade: Grade;
  displayVal: number;
  desc: string;
}

function buildMetrics(): MetricInfo[] {
  const raw = result.rawMetrics;
  const scores = result.displayScores;
  const comments = result.indicator_comments;

  return [
    { key: 'acne',         label: '트러블',   icon: '🔴', grade: getAcneGrade(raw.acne) as Grade,              displayVal: scores.acne,         desc: comments.acne },
    { key: 'dryness',      label: '건조',     icon: '💧', grade: getDrynessGrade(raw.dryness) as Grade,         displayVal: scores.dryness,      desc: comments.dryness },
    { key: 'sagging',      label: '처짐',     icon: '✨', grade: getSaggingGrade(raw.sagging) as Grade,         displayVal: scores.sagging,      desc: comments.sagging },
    { key: 'pore',         label: '모공',     icon: '🔬', grade: getPoreGrade(raw.pore) as Grade,              displayVal: scores.pore,         desc: comments.pore },
    { key: 'pigmentation', label: '색소침착', icon: '🌫️', grade: getPigmentationGrade(raw.pigmentation) as Grade, displayVal: scores.pigmentation, desc: comments.pigmentation },
    { key: 'wrinkle',      label: '주름',     icon: '〰️', grade: getWrinkleGrade(raw.wrinkle) as Grade,         displayVal: scores.wrinkle,      desc: comments.wrinkle },
  ];
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일`;
}

export default function AnalysisResultPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const imageUrl = location.state?.imageUrl ?? result.imageUrl;

  const metrics = buildMetrics();
  const radarData = metrics.map((m) => ({ metric: m.label, value: m.displayVal, fullMark: 100 }));
  const skinTags = [result.skinType, '모공', '건조'];

  const gradeCounts: Record<Grade, number> = { 양호: 0, 보통: 0, 개선필요: 0 };
  metrics.forEach((m) => { gradeCounts[m.grade]++; });

  return (
    <div className="ar-page">

      {/* ── HERO ── */}
      <section className="ar-hero">
        <div className="ar-hero-inner">
          <div className="ar-hero-left">
            <div className="ar-hero-badge">AI 피부 분석 완료</div>
            <h1 className="ar-hero-title">
              <span className="ar-hero-name">{user.name}</span>님,<br />
              <em>피부 리포트</em>가<br />
              도착했어요
            </h1>
            <p className="ar-hero-date">분석일 : {formatDate(result.generated_at)}</p>
            <div className="ar-skin-tags">
              {skinTags.map((t) => <span key={t} className="ar-skin-tag">{t}</span>)}
            </div>
            <div className="ar-hero-btns">
              <button className="ar-btn-primary" onClick={() => navigate('/routine/budget', { state: { resultId: result.result_id } })}>
                🌿 루틴 추천받기
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

      {/* ── 위험 지표 + AI 한마디 ── */}
      <section className="ar-report-section">
        <div className="ar-section-inner">

          <div className="ar-sec-badge ar-badge-danger">위험 지표 분석</div>
          <h2 className="ar-sec-title">내 피부 위험지표 리포트</h2>
          <p className="ar-sec-sub">
            높은 등급일수록 피부 상태가 심각해요 &nbsp;
            <span className="ar-grade-legend">
              <span style={{ color: '#22c55e' }}>● 양호</span>
              <span style={{ color: '#f59e0b' }}>● 보통</span>
              <span style={{ color: '#ef4444' }}>● 개선필요</span>
            </span>
          </p>

          {/* 등급 요약 */}
          <div className="ar-grade-summary">
            {(['양호', '보통', '개선필요'] as Grade[]).map((g) => (
              <div className="ar-grade-summary-item" key={g} style={{ borderColor: GRADE_COLOR[g], background: GRADE_BG[g] }}>
                <span className="ar-grade-count" style={{ color: GRADE_COLOR[g] }}>{gradeCounts[g]}</span>
                <span className="ar-grade-label" style={{ color: GRADE_COLOR[g] }}>{g}</span>
              </div>
            ))}
          </div>

          {/* 레이더 + 지표 카드 */}
          <div className="ar-report-body">
            <div className="ar-radar-col">
              <div className="ar-radar-wrap">
                <ResponsiveContainer width="100%" height={380}>
                  <RadarChart data={radarData} cx="50%" cy="50%" outerRadius="68%">
                    <PolarGrid stroke="#ddd6fe" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: '#4b5563', fontSize: 13, fontWeight: 600 }} />
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
                        <div className="ar-metric-icon-wrap" style={{ background: bg }}>
                          <span>{m.icon}</span>
                        </div>
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

          {/* AI 한마디 — summary_comment 사용 */}
          <div className="ar-ai-box">
            <div className="ar-ai-label">🤖 AI 한마디 ✨</div>
            <p className="ar-ai-text">{result.summary_comment}</p>
            <p className="ar-ai-cta">아래 맞춤 루틴 추천을 확인해보세요 →</p>
          </div>

        </div>
      </section>

      {/* ── FOOTER CTA ── */}
      <section className="ar-cta-section">
        <div className="ar-sec-badge" style={{ background: 'rgba(255,255,255,0.2)', color: '#fff' }}>맞춤 추천</div>
        <h2>{user.name}님을 위한 루틴</h2>
        <p>AI가 분석한 피부 상태에 맞는 제품을 추천해드려요</p>
        <button className="ar-btn-white" onClick={() => navigate('/routine/budget', { state: { resultId: result.result_id } })}>
          ✨ 맞춤 루틴 바로 받기
        </button>
      </section>

      <footer className="ar-footer">
        <span>© 2026 ROUPLE AI 기반 맞춤형 스킨케어 솔루션</span>
        <span className="ar-footer-links">개인정보처리방침 · 이용약관</span>
      </footer>

    </div>
  );
}
