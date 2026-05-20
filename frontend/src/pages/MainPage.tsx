import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/useAuth';
import { ChevronRight } from 'lucide-react';
import './MainPage.css';

const ANALYSIS_METRICS = [
  { label: '트러블',   value: 73, color: '#ef4444' },
  { label: '건조',     value: 58, color: '#f59e0b' },
  { label: '처짐',     value: 48, color: '#f59e0b' },
  { label: '모공',     value: 65, color: '#f59e0b' },
  { label: '색소침착', value: 42, color: '#22c55e' },
  { label: '주름',     value: 23, color: '#22c55e' },
];

const ROUTINE_ITEMS = [
  { step: 1, cat: '클렌저', name: '부드러운 클렌징폼', price: '18,000원' },
  { step: 2, cat: '토너',   name: '수분 토너',         price: '25,000원' },
  { step: 3, cat: '크림',   name: '보습 크림',         price: '38,000원' },
];

const HOW_STEPS = [
  {
    num: '01',
    title: '피부타입 & 고민 설정',
    desc: '피부 타입과 주요 고민사항을 간단히 선택해주세요. 건성·지성·복합성 등 피부 타입부터 여드름·모공·주름·홍조 등 개선하고 싶은 부위를 체크하면, AI가 맞춤형 분석 기준을 자동으로 설정해 더 정확한 리포트를 드려요.',
  },
  {
    num: '02',
    title: '셀피 한 장 업로드 & 분석',
    desc: '스마트폰 카메라 한 장이면 충분해요. 밝은 조명 아래 생얼 상태로 정면을 바라보고 찍은 사진을 업로드하면, AI가 트러블·건조·처짐·모공·색소침착·주름 6가지 지표를 수치화해 개인 피부 리포트를 완성해드려요.',
  },
  {
    num: '03',
    title: '맞춤 루틴 확인',
    desc: '분석 결과와 내 예산을 함께 고려해 최적의 제품 조합을 추천해드려요. 클렌저부터 크림까지 사용 순서와 방법까지 AI가 안내하므로, 처음 스킨케어를 시작하는 분도 어렵지 않게 따라올 수 있어요.',
  },
];

/* ── SVG 일러스트: Step 01 — 피부 설정 화면 ── */
function SkinSetupVisual() {
  return (
    <svg viewBox="0 0 480 300" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="cs1" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="6" stdDeviation="12" floodColor="rgba(124,58,237,0.12)" />
        </filter>
      </defs>
      <rect x="28" y="10" width="424" height="278" rx="18" fill="white" filter="url(#cs1)" />
      <rect x="28" y="10" width="424" height="52" rx="18" fill="#7c3aed" />
      <rect x="28" y="42" width="424" height="20" fill="#7c3aed" />
      <text x="240" y="43" textAnchor="middle" fill="white" fontSize="13" fontWeight="700" fontFamily="sans-serif">내 피부 설정</text>

      <text x="50" y="80" fill="#9ca3af" fontSize="10" fontWeight="700" fontFamily="sans-serif">피부 타입</text>
      <rect x="50"  y="87" width="54" height="26" rx="13" fill="#7c3aed" />
      <text x="77"  y="105" textAnchor="middle" fill="white" fontSize="11" fontWeight="700" fontFamily="sans-serif">건성</text>
      <rect x="112" y="87" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="139" y="105" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">지성</text>
      <rect x="174" y="87" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="201" y="105" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">중성</text>
      <rect x="236" y="87" width="66" height="26" rx="13" fill="#f3f4f6" />
      <text x="269" y="105" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">복합성</text>
      <rect x="310" y="87" width="60" height="26" rx="13" fill="#f3f4f6" />
      <text x="340" y="105" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">수부지</text>

      <text x="50" y="136" fill="#9ca3af" fontSize="10" fontWeight="700" fontFamily="sans-serif">피부 고민 (복수 선택)</text>
      <rect x="50"  y="144" width="66" height="26" rx="13" fill="#7c3aed" />
      <text x="83"  y="162" textAnchor="middle" fill="white" fontSize="11" fontWeight="700" fontFamily="sans-serif">여드름</text>
      <rect x="124" y="144" width="62" height="26" rx="13" fill="#7c3aed" />
      <text x="155" y="162" textAnchor="middle" fill="white" fontSize="11" fontWeight="700" fontFamily="sans-serif">속건조</text>
      <rect x="194" y="144" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="221" y="162" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">모공</text>
      <rect x="256" y="144" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="283" y="162" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">주름</text>
      <rect x="318" y="144" width="62" height="26" rx="13" fill="#f3f4f6" />
      <text x="349" y="162" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">민감성</text>

      <rect x="50"  y="178" width="62" height="26" rx="13" fill="#f3f4f6" />
      <text x="81"  y="196" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">붉은기</text>
      <rect x="120" y="178" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="147" y="196" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">미백</text>
      <rect x="182" y="178" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="209" y="196" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">각질</text>
      <rect x="244" y="178" width="54" height="26" rx="13" fill="#f3f4f6" />
      <text x="271" y="196" textAnchor="middle" fill="#6b7280" fontSize="11" fontFamily="sans-serif">홍조</text>

      <rect x="50" y="228" width="380" height="40" rx="12" fill="#7c3aed" />
      <text x="240" y="253" textAnchor="middle" fill="white" fontSize="13" fontWeight="700" fontFamily="sans-serif">완료하기 →</text>
    </svg>
  );
}

/* ── SVG 일러스트: Step 02 — 피부 분석 레이더 차트 ── */
function AnalysisVisual() {
  // 레이더 차트: center (240,178), r=64
  // 6 axes starting from top (-90°), clockwise at 60° intervals
  // Outer hex:
  const outer = '240,114 295,146 295,210 240,242 185,210 185,146';
  // 50% grid (r=32):
  const grid50 = '240,146 268,162 268,194 240,210 212,194 212,162';
  // 25% grid (r=16):
  const grid25 = '240,162 254,170 254,186 240,194 226,186 226,170';
  // Outer axis endpoints
  const outerPts: [number, number][] = [[240,114],[295,146],[295,210],[240,242],[185,210],[185,146]];
  // Data polygon: values 73,58,48,65,42,23 (r=64)
  const data = '240,131 272,159 267,193 240,222 217,191 227,171';
  // Data dots
  const dots: [number, number][] = [[240,131],[272,159],[267,193],[240,222],[217,191],[227,171]];

  const labels = [
    { text: '트러블', x: 240, y: 98,  anchor: 'middle', val: 73, color: '#ef4444' },
    { text: '건조',   x: 310, y: 138, anchor: 'start',  val: 58, color: '#f59e0b' },
    { text: '처짐',   x: 310, y: 218, anchor: 'start',  val: 48, color: '#f59e0b' },
    { text: '모공',   x: 240, y: 260, anchor: 'middle', val: 65, color: '#f59e0b' },
    { text: '색소침착', x: 170, y: 218, anchor: 'end', val: 42, color: '#22c55e' },
    { text: '주름',   x: 170, y: 138, anchor: 'end',   val: 23, color: '#22c55e' },
  ];

  return (
    <svg viewBox="0 0 480 300" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="cs2" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="6" stdDeviation="12" floodColor="rgba(124,58,237,0.12)" />
        </filter>
        <linearGradient id="radarFill" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#7c3aed" stopOpacity="0.28" />
          <stop offset="100%" stopColor="#a78bfa" stopOpacity="0.18" />
        </linearGradient>
      </defs>

      <rect x="28" y="10" width="424" height="278" rx="18" fill="white" filter="url(#cs2)" />
      <rect x="28" y="10" width="424" height="52" rx="18" fill="#7c3aed" />
      <rect x="28" y="42" width="424" height="20" fill="#7c3aed" />
      <text x="140" y="43" textAnchor="middle" fill="white" fontSize="13" fontWeight="700" fontFamily="sans-serif">AI 피부 분석 리포트</text>

      {/* Right side: skin type info */}
      <text x="300" y="75" fill="#111" fontSize="15" fontWeight="700" fontFamily="sans-serif">건성 피부</text>
      <text x="300" y="93" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">2026. 05. 19</text>
      <line x1="290" y1="103" x2="430" y2="103" stroke="#f0f0f0" strokeWidth="1" />

      {/* Score summary */}
      {[
        { label: '개선필요', count: 4, color: '#f59e0b', y: 120 },
        { label: '양호',     count: 2, color: '#22c55e', y: 148 },
      ].map(s => (
        <g key={s.label}>
          <circle cx="300" cy={s.y} r="5" fill={s.color} opacity="0.7" />
          <text x="312" y={s.y + 4} fill="#555" fontSize="11" fontFamily="sans-serif">{s.label} {s.count}개</text>
        </g>
      ))}

      {/* Radar grid */}
      <polygon points={outer}  fill="none" stroke="#e5e7eb" strokeWidth="1" />
      <polygon points={grid50} fill="none" stroke="#e5e7eb" strokeWidth="0.7" strokeDasharray="3 2" />
      <polygon points={grid25} fill="none" stroke="#e5e7eb" strokeWidth="0.7" strokeDasharray="3 2" />

      {/* Axis lines */}
      {outerPts.map((pt, i) => (
        <line key={i} x1="240" y1="178" x2={pt[0]} y2={pt[1]} stroke="#e5e7eb" strokeWidth="0.7" />
      ))}

      {/* Data area */}
      <polygon points={data} fill="url(#radarFill)" stroke="#7c3aed" strokeWidth="2" />

      {/* Data dots */}
      {dots.map((d, i) => (
        <circle key={i} cx={d[0]} cy={d[1]} r="4" fill="#7c3aed" stroke="white" strokeWidth="1.5" />
      ))}

      {/* Axis labels */}
      {labels.map(l => (
        <g key={l.text}>
          <text
            x={l.x} y={l.y}
            textAnchor={l.anchor as 'middle' | 'start' | 'end'}
            fill={l.color}
            fontSize="10"
            fontWeight="700"
            fontFamily="sans-serif"
          >{l.text}</text>
          <text
            x={l.anchor === 'start' ? l.x + 2 : l.anchor === 'end' ? l.x - 2 : l.x}
            y={l.y + 13}
            textAnchor={l.anchor as 'middle' | 'start' | 'end'}
            fill="#9ca3af"
            fontSize="9"
            fontFamily="sans-serif"
          >{l.val}</text>
        </g>
      ))}
    </svg>
  );
}

/* ── SVG 일러스트: Step 03 — 루틴 리포트 ── */
function RoutineVisual() {
  const items = [
    { step: 1, cat: '클렌저', name: '순한 클렌징폼',  price: '18,000원' },
    { step: 2, cat: '토너',   name: '수분 진정 토너', price: '25,000원' },
    { step: 3, cat: '앰플',   name: '진정 앰플',      price: '45,000원' },
    { step: 4, cat: '크림',   name: '수분 크림',      price: '38,000원' },
  ];

  return (
    <svg viewBox="0 0 480 300" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="cs3" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="6" stdDeviation="12" floodColor="rgba(124,58,237,0.12)" />
        </filter>
      </defs>

      <rect x="28" y="10" width="424" height="278" rx="18" fill="white" filter="url(#cs3)" />
      <rect x="28" y="10" width="424" height="52" rx="18" fill="#7c3aed" />
      <rect x="28" y="42" width="424" height="20" fill="#7c3aed" />
      <text x="68" y="43" fill="white" fontSize="13" fontWeight="700" fontFamily="sans-serif">AI 추천 루틴</text>
      <text x="402" y="43" textAnchor="end" fill="rgba(255,255,255,0.75)" fontSize="11" fontFamily="sans-serif">총 126,000원</text>

      {/* Subtitle */}
      <text x="50" y="80" fill="#7c3aed" fontSize="11" fontWeight="600" fontFamily="sans-serif">건성 피부 맞춤 · 예산 범위 내 최적 조합</text>

      {/* Product rows */}
      {items.map((item, i) => {
        const ry = 105 + i * 38;
        return (
          <g key={item.step}>
            {i > 0 && (
              <line x1="50" y1={ry - 12} x2="430" y2={ry - 12} stroke="#f5f5f5" strokeWidth="1" />
            )}
            {/* STEP badge */}
            <rect x="50" y={ry - 10} width="52" height="20" rx="10" fill="#7c3aed" />
            <text x="76" y={ry + 4} textAnchor="middle" fill="white" fontSize="9" fontWeight="700" fontFamily="sans-serif">STEP {item.step}</text>
            {/* Category */}
            <text x="114" y={ry - 1} fill="#9ca3af" fontSize="9" fontFamily="sans-serif">{item.cat}</text>
            {/* Name */}
            <text x="114" y={ry + 12} fill="#111" fontSize="12" fontWeight="600" fontFamily="sans-serif">{item.name}</text>
            {/* Price */}
            <text x="402" y={ry + 12} textAnchor="end" fill="#374151" fontSize="12" fontWeight="600" fontFamily="sans-serif">{item.price}</text>
          </g>
        );
      })}

      {/* Divider + total */}
      <line x1="50" y1="252" x2="430" y2="252" stroke="#f0f0f0" strokeWidth="1" />
      <text x="50"  y="268" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">소요 약 8분</text>
      <circle cx="116" cy="265" r="2" fill="#d1d5db" />
      <text x="122" y="268" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">제품 4개</text>
      <text x="402" y="268" textAnchor="end" fill="#7c3aed" fontSize="11" fontWeight="700" fontFamily="sans-serif">총 126,000원</text>

      {/* Bottom CTA-like strip */}
      <rect x="50" y="276" width="380" height="2" rx="1" fill="#f5f3ff" />
    </svg>
  );
}

function HowVisual({ idx }: { idx: number }) {
  if (idx === 0) return <SkinSetupVisual />;
  if (idx === 1) return <AnalysisVisual />;
  return <RoutineVisual />;
}

export default function MainPage() {
  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();
  const handleStart = () => navigate(isLoggedIn ? '/diagnosis' : '/login');

  return (
    <main className="mp">

      {/* ── HERO ── */}
      <section className="mp-hero">
        <div className="mp-hero-inner">
          <div className="mp-hero-text">
            <p className="mp-eyebrow">AI Skin Analysis Platform</p>
            <h1 className="mp-hero-h1">
              내 피부를<br />AI로 읽다
            </h1>
            <p className="mp-hero-desc">
              사진 한 장으로 6가지 피부 지표를 분석하고<br />
              나에게 맞는 스킨케어 루틴을 AI가 설계합니다.
            </p>
            <div className="mp-hero-btns">
              <button className="mp-btn-white" onClick={handleStart}>
                무료로 시작하기
                <ChevronRight size={15} style={{ display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
              </button>
              <button className="mp-btn-ghost" onClick={() => navigate('/products')}>제품 둘러보기</button>
            </div>
          </div>

          {/* 사람 사진: /public/hero-photo.jpg 에 추가하세요 */}
          <div className="mp-hero-photo-wrap">
            <img
              src="/hero-photo.jpg"
              alt="스킨케어 모델"
              className="mp-hero-photo"
              onError={(e) => { e.currentTarget.style.display = 'none'; }}
            />
            {/* AI 분석 오버레이 */}
            <div className="mp-hero-overlay">
              <span className="mp-corner mp-corner--tl" />
              <span className="mp-corner mp-corner--tr" />
              <span className="mp-corner mp-corner--bl" />
              <span className="mp-corner mp-corner--br" />
              <div className="mp-hero-scan-badge">
                <span className="mp-scan-dot" />
                AI 피부 분석 중
              </div>
              <div className="mp-hero-chips">
                <div className="mp-hero-chip mp-hero-chip--red">트러블 <b>73</b></div>
                <div className="mp-hero-chip mp-hero-chip--amber">건조 <b>58</b></div>
                <div className="mp-hero-chip mp-hero-chip--green">주름 <b>23</b></div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* ── SHOWCASE ── */}
      <section className="mp-showcase">
        <div className="mp-container">
          <div className="mp-showcase-head">
            <p className="mp-eyebrow mp-eyebrow--purple">피부 분석부터 루틴까지</p>
            <h2 className="mp-showcase-h2">나에게 맞는<br />스킨케어를 찾다</h2>
            <p className="mp-showcase-sub">
              AI가 6가지 피부 지표를 분석하고, 예산에 맞는 루틴까지 한번에 제안해요.
            </p>
          </div>

          <div className="mp-cards-scatter">
            <div className="mp-scatter-card mp-scatter-card--l">
              <div className="mp-sc-top">
                <span className="mp-sc-badge mp-sc-badge--purple">AI 분석 완료</span>
                <span className="mp-sc-meta">2026. 05. 19</span>
              </div>
              <div className="mp-sc-type">건성 피부</div>
              {ANALYSIS_METRICS.map((m) => (
                <div key={m.label} className="mp-sc-metric">
                  <span className="mp-sc-ml">{m.label}</span>
                  <div className="mp-sc-bar-bg">
                    <div className="mp-sc-bar" style={{ width: `${m.value}%`, background: m.color }} />
                  </div>
                  <span className="mp-sc-mv" style={{ color: m.color }}>{m.value}</span>
                </div>
              ))}
            </div>

            <div className="mp-scatter-card mp-scatter-card--r">
              <div className="mp-sc-top">
                <span className="mp-sc-badge mp-sc-badge--green">AI 추천 루틴</span>
                <span className="mp-sc-meta">총 81,000원</span>
              </div>
              <div className="mp-sc-type">예산 범위 내 최적 조합</div>
              {ROUTINE_ITEMS.map((r) => (
                <div key={r.step} className="mp-sc-ri">
                  <div className="mp-sc-ri-step">STEP {r.step}</div>
                  <div className="mp-sc-ri-info">
                    <span className="mp-sc-ri-cat">{r.cat}</span>
                    <span className="mp-sc-ri-name">{r.name}</span>
                  </div>
                  <span className="mp-sc-ri-price">{r.price}</span>
                </div>
              ))}
              <div className="mp-sc-foot">소요 약 5분 · 제품 3개</div>
            </div>
          </div>
        </div>
      </section>

      {/* ── HOW TO ── */}
      <section className="mp-how">
        <div className="mp-container">
          <div className="mp-how-head">
            <h2 className="mp-how-h2">어떻게 하나요?</h2>
            <p className="mp-how-sub">3분이면 나만의 피부 리포트와 루틴이 완성돼요.</p>
          </div>
          <div className="mp-how-steps">
            {HOW_STEPS.map(({ num, title, desc }, idx) => (
              <div className="mp-how-card" key={num}>
                <div className="mp-how-left">
                  <span className="mp-how-num">STEP · {num}</span>
                  <h3 className="mp-how-title">{title}</h3>
                  <p className="mp-how-desc">{desc}</p>
                </div>
                <div className="mp-how-right">
                  <HowVisual idx={idx} />
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="mp-cta">
        <div className="mp-container mp-cta-inner">
          <p className="mp-eyebrow mp-eyebrow--purple">무료로 시작하기</p>
          <h2 className="mp-cta-h2">내 피부에 맞는 루틴,<br />지금 바로 찾아볼까요?</h2>
          <p className="mp-cta-desc">회원가입 후 즉시 시작할 수 있어요. 3분이면 충분합니다.</p>
          <button className="mp-btn-purple" onClick={handleStart}>
            무료로 시작하기
            <ChevronRight size={15} style={{ display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
          </button>
        </div>
      </section>

    </main>
  );
}
