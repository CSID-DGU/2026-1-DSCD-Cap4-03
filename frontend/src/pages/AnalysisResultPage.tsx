import { useEffect, useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  Radar, RadarChart, PolarGrid, PolarAngleAxis, ResponsiveContainer, PolarRadiusAxis,
} from 'recharts';
import { Flame, Droplets, Layers, ScanLine, Cloud, Waves, Leaf, Sparkles, Bot } from 'lucide-react';
import { analysisApi, type AnalysisResult } from '../api/analysis';
import { useAuth } from '../context/useAuth';
import './AnalysisResultPage.css';

// ── 등급 변환 함수 (display_score 0-100 → 양호/보통/개선필요)
// grade = round(score * 등급수 / 100), 최대 등급수-1 클램프
function toGrade(score: number, levels: number) {
  return Math.min(Math.round(score * levels / 100), levels - 1);
}
function getAcneGrade(score: number): Grade {
  const g = toGrade(score, 4); // 0~3
  return g === 0 ? '양호' : g === 1 ? '보통' : '개선필요';
}
function getDrynessGrade(score: number): Grade {
  const g = toGrade(score, 5); // 0~4
  return g <= 1 ? '양호' : g === 2 ? '보통' : '개선필요';
}
function getSaggingGrade(score: number): Grade {
  const g = toGrade(score, 6); // 0~5
  return g <= 1 ? '양호' : g <= 3 ? '보통' : '개선필요';
}
function getPoreGrade(score: number): Grade {
  const g = toGrade(score, 5); // 0~4
  return g <= 1 ? '양호' : g <= 3 ? '보통' : '개선필요';
}
function getPigmentationGrade(score: number): Grade {
  const g = toGrade(score, 6); // 0~5
  return g === 0 ? '양호' : g <= 3 ? '보통' : '개선필요';
}
function getWrinkleGrade(score: number): Grade {
  const g = toGrade(score, 6); // 0~5
  return g <= 1 ? '양호' : g === 2 ? '보통' : '개선필요';
}

type Grade = '양호' | '보통' | '개선필요';

const GRADE_COLOR: Record<Grade, string> = {
  양호: '#22c55e', 보통: '#f59e0b', 개선필요: '#ef4444',
};
const GRADE_BG: Record<Grade, string> = {
  양호: '#f0fdf4', 보통: '#fffbeb', 개선필요: '#fef2f2',
};

function buildMetrics(result: AnalysisResult) {
  const raw = result.display_scores;
  const comments = result.indicator_comments;

  const s = {
    acne:         Math.round(raw.acne         * 100),
    dryness:      Math.round(raw.dryness      * 100),
    sagging:      Math.round(raw.sagging      * 100),
    pore:         Math.round(raw.pore         * 100),
    pigmentation: Math.round(raw.pigmentation * 100),
    wrinkle:      Math.round(raw.wrinkle      * 100),
  };

  return [
    { key: 'acne',         label: '트러블',   Icon: Flame,    grade: getAcneGrade(s.acne),               displayVal: s.acne,         desc: comments.acne },
    { key: 'dryness',      label: '건조',     Icon: Droplets, grade: getDrynessGrade(s.dryness),         displayVal: s.dryness,      desc: comments.dryness },
    { key: 'sagging',      label: '처짐',     Icon: Layers,   grade: getSaggingGrade(s.sagging),         displayVal: s.sagging,      desc: comments.sagging },
    { key: 'pore',         label: '모공',     Icon: ScanLine, grade: getPoreGrade(s.pore),               displayVal: s.pore,         desc: comments.pore },
    { key: 'pigmentation', label: '색소침착', Icon: Cloud,    grade: getPigmentationGrade(s.pigmentation), displayVal: s.pigmentation, desc: comments.pigmentation },
    { key: 'wrinkle',      label: '주름',     Icon: Waves,    grade: getWrinkleGrade(s.wrinkle),         displayVal: s.wrinkle,      desc: comments.wrinkle },
  ];
}

function formatDate(dateStr: string) {
  const d = new Date(dateStr);
  return `${d.getFullYear()}년 ${d.getMonth() + 1}월 ${d.getDate()}일`;
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

  useEffect(() => {
    analysisApi.getResult(resultId)
      .then(setResult)
      .catch((err) => setError((err as Error).message || '분석 결과를 불러오지 못했어요.'))
      .finally(() => setLoading(false));
  }, [resultId]);

  if (loading) {
    return (
      <div className="ar-page" style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh' }}>
        <p style={{ color: '#7c3aed', fontSize: '1.1rem' }}>분석 결과를 불러오는 중...</p>
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
  const imageUrl = passedImageUrl || result.image_url;
  const displayName = nickname || '내';
  const skinTags = [result.skin_type, '모공', '건조'].filter(Boolean) as string[];

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
              <span className="ar-hero-name">{displayName}</span>님,<br />
              <em>피부 리포트</em>가<br />
              도착했어요
            </h1>
            <p className="ar-hero-date">분석일 : {formatDate(result.analyzed_at || result.generated_at)}</p>
            <div className="ar-skin-tags">
              {skinTags.map((t) => <span key={t} className="ar-skin-tag">{t}</span>)}
            </div>
            <div className="ar-hero-btns">
              <button className="ar-btn-primary" onClick={() => navigate('/routine/budget', { state: { resultId: result.result_id, imageId: result.image_id } })}>
                <Leaf size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />루틴 추천받기
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
                          <m.Icon size={18} color="#7c3aed" />
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

          {/* AI 한마디 */}
          <div className="ar-ai-box">
            <div className="ar-ai-label"><Bot size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />AI 한마디</div>
            <p className="ar-ai-text">{result.summary_comment}</p>
            <p className="ar-ai-cta">아래 맞춤 루틴 추천을 확인해보세요 →</p>
          </div>

        </div>
      </section>

      {/* ── FOOTER CTA ── */}
      <section className="ar-cta-section">
        <div className="ar-sec-badge" style={{ background: 'rgba(255,255,255,0.2)', color: '#fff' }}>맞춤 추천</div>
        <h2>{displayName}님을 위한 루틴</h2>
        <p>AI가 분석한 피부 상태에 맞는 제품을 추천해드려요</p>
        <button className="ar-btn-white" onClick={() => navigate('/routine/budget', { state: { resultId: result.result_id, imageId: result.image_id } })}>
          <Sparkles size={15} style={{ display: 'inline', verticalAlign: 'middle', marginRight: 6 }} />맞춤 루틴 바로 받기
        </button>
      </section>

    </div>
  );
}
