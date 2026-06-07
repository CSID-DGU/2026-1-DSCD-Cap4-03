import { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/useAuth';
import { ChevronRight } from 'lucide-react';
import './MainPage.css';

const ANALYSIS_METRICS = [
  { label: '트러블', value: 73, color: '#ef4444' },
  { label: '수분',     value: 58, color: '#f59e0b' },
  { label: '탄력',     value: 48, color: '#f59e0b' },
  { label: '모공',     value: 65, color: '#f59e0b' },
  { label: '색소침착', value: 42, color: '#22c55e' },
  { label: '주름',     value: 23, color: '#22c55e' },
];

const ROUTINE_ITEMS = [
  { step: 1, cat: '토너', name: '수분 토너', price: '18,000원' },
  { step: 2, cat: '세럼',   name: '진정 세럼',         price: '25,000원' },
  { step: 3, cat: '크림',   name: '보습 크림',         price: '38,000원' },
];

const HOW_STEPS = [
  {
    num: '01',
    title: '당신의 피부를 알려주세요',
    desc: '피부 타입과 고민을 간단하게 선택하면\n당신의 피부에 맞는 분석이 시작됩니다.\n더 개인화된 스킨케어를 위한 첫 단계입니다.',
  },
  {
    num: '02',
    title: 'AI가 피부를 정밀하게 분석합니다',
    desc: '피부 사진을 업로드하면\nAI가 피부 상태와 특징을 세밀하게 분석합니다.\n보이지 않던 피부 데이터를 더 정확하게 이해합니다.',
  },
  {
    num: '03',
    title: '당신만의 스킨케어 루틴을 제안합니다',
    desc: '분석 결과와 피부 고민을 바탕으로\n당신에게 가장 잘 어울리는 화장품과 루틴을 추천합니다.\n매일의 스킨케어를 더 스마트하게 경험해보세요.',
  },
];

/* SVG illustration: Step 01 */
function SkinSetupVisual() {
  return (
    <svg viewBox="0 0 480 300" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="cs1" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="4" stdDeviation="10" floodColor="rgba(0,0,0,0.08)" />
        </filter>
      </defs>

      <rect x="28" y="10" width="424" height="278" rx="10" fill="white" filter="url(#cs1)" />

      <text x="50" y="40" fill="#111827" fontSize="14" fontWeight="700" fontFamily="sans-serif">내 피부 설정</text>
      <text x="50" y="56" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">피부 타입과 고민을 선택해주세요</text>
      <line x1="28" y1="65" x2="452" y2="65" stroke="#f3f4f6" strokeWidth="1" />

      <text x="50" y="83" fill="#6b7280" fontSize="10" fontWeight="700" fontFamily="sans-serif">피부 타입</text>
      <rect x="50" y="90" width="50" height="24" rx="6" fill="#f5f3ff" stroke="#c4b5fd" strokeWidth="1.5" />
      <text x="75" y="106" textAnchor="middle" fill="#7c3aed" fontSize="11" fontWeight="700" fontFamily="sans-serif">건성</text>
      <rect x="108" y="90" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="133" y="106" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">지성</text>
      <rect x="166" y="90" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="191" y="106" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">중성</text>
      <rect x="224" y="90" width="62" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="255" y="106" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">복합성</text>
      <rect x="294" y="90" width="62" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="325" y="106" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">수부지</text>

      <text x="50" y="132" fill="#6b7280" fontSize="10" fontWeight="700" fontFamily="sans-serif">피부 고민 (복수 선택)</text>
      <rect x="50" y="139" width="62" height="24" rx="6" fill="#f5f3ff" stroke="#c4b5fd" strokeWidth="1.5" />
      <text x="81" y="155" textAnchor="middle" fill="#7c3aed" fontSize="11" fontWeight="700" fontFamily="sans-serif">여드름</text>
      <rect x="120" y="139" width="62" height="24" rx="6" fill="#f5f3ff" stroke="#c4b5fd" strokeWidth="1.5" />
      <text x="151" y="155" textAnchor="middle" fill="#7c3aed" fontSize="11" fontWeight="700" fontFamily="sans-serif">속건조</text>
      <rect x="190" y="139" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="215" y="155" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">모공</text>
      <rect x="248" y="139" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="273" y="155" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">주름</text>
      <rect x="306" y="139" width="62" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="337" y="155" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">민감성</text>

      <rect x="50" y="171" width="62" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="81" y="187" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">붉은기</text>
      <rect x="120" y="171" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="145" y="187" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">미백</text>
      <rect x="178" y="171" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="203" y="187" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">각질</text>
      <rect x="236" y="171" width="50" height="24" rx="6" fill="#f9fafb" stroke="#e5e7eb" strokeWidth="1.5" />
      <text x="261" y="187" textAnchor="middle" fill="#9ca3af" fontSize="11" fontFamily="sans-serif">홍조</text>

      <rect x="50" y="220" width="380" height="40" rx="8" fill="#7c3aed" />
      <text x="240" y="245" textAnchor="middle" fill="white" fontSize="13" fontWeight="700" fontFamily="sans-serif">완료하기 →</text>
    </svg>
  );
}

/* SVG illustration: Step 02 */
function AnalysisVisual() {
  const LABELS = ['트러블', '수분', '탄력', '모공', '색소침착', '주름'];
  const VALUES = [0.73, 0.58, 0.48, 0.65, 0.42, 0.23];
  const cx = 349, cy = 192, R = 68;
  const N = 6;

  const pt = (i: number, r: number) => ({
    x: cx + r * Math.cos(-Math.PI / 2 + (2 * Math.PI * i) / N),
    y: cy + r * Math.sin(-Math.PI / 2 + (2 * Math.PI * i) / N),
  });

  const hexPoly = (r: number) =>
    Array.from({ length: N }, (_, i) => { const p = pt(i, r); return `${p.x},${p.y}`; }).join(' ');

  const dataPoly = VALUES.map((v, i) => { const p = pt(i, R * v); return `${p.x},${p.y}`; }).join(' ');

  const ANCHORS = ['middle', 'start', 'start', 'middle', 'end', 'end'] as const;

  return (
    <svg viewBox="0 0 490 318" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg" xmlnsXlink="http://www.w3.org/1999/xlink">
      <defs>
        <filter id="cs2a" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="10" floodColor="rgba(0,0,0,0.08)" />
        </filter>
        <filter id="cs2b" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="6" stdDeviation="14" floodColor="rgba(0,0,0,0.10)" />
        </filter>
        <clipPath id="faceClip">
          <rect x="22" y="67" width="243" height="197" rx="8" />
        </clipPath>
      </defs>

      {/* 왼쪽 카드 */}
      <rect x="12" y="10" width="263" height="268" rx="10" fill="white" filter="url(#cs2a)" />
      <text x="30" y="38" fill="#111827" fontSize="12" fontWeight="700" fontFamily="sans-serif">피부 사진 분석</text>
      <text x="30" y="52" fill="#9ca3af" fontSize="9" fontFamily="sans-serif">AI 스캔 중...</text>
      <line x1="12" y1="60" x2="275" y2="60" stroke="#f3f4f6" strokeWidth="1" />

      {/* 사진 영역 배경 */}
      <rect x="22" y="67" width="243" height="197" rx="8" fill="#f9f7ff" />

      {/* 인물 사진 */}
      <foreignObject x="22" y="67" width="243" height="197" clipPath="url(#faceClip)">
        <img
          src="/how-step02-face.png"
          alt="피부 분석"
          style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block', borderRadius: '8px' }}
        />
      </foreignObject>

      {/* 격자 오버레이 */}
      {[90, 115, 140, 165, 190, 215, 240].map(y => (
        <line key={`h${y}`} x1="22" y1={y} x2="265" y2={y} stroke="#7c3aed" strokeWidth="0.5" strokeDasharray="4 4" opacity="0.2" />
      ))}
      {[60, 90, 120, 143, 168, 200, 240].map(x => (
        <line key={`v${x}`} x1={x} y1="67" x2={x} y2="264" stroke="#7c3aed" strokeWidth="0.5" strokeDasharray="4 4" opacity="0.2" />
      ))}

      {/* 코너 브라켓 */}
      <path d="M30 78 L30 68 L40 68" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M256 78 L256 68 L246 68" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M30 254 L30 264 L40 264" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M256 254 L256 264 L246 264" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />

      {/* 스캔 중 뱃지 */}
      <rect x="82" y="70" width="102" height="18" rx="9" fill="rgba(124,58,237,0.75)" />
      <text x="133" y="82" textAnchor="middle" fill="white" fontSize="8" fontWeight="700" fontFamily="sans-serif">AI 피부 분석 중</text>

      {/* 오른쪽 리포트 카드 */}
      <rect x="218" y="40" width="263" height="268" rx="10" fill="white" filter="url(#cs2b)" />
      <text x="236" y="68" fill="#111827" fontSize="12" fontWeight="700" fontFamily="sans-serif">AI 피부 리포트</text>
      <text x="236" y="82" fill="#9ca3af" fontSize="9" fontFamily="sans-serif">건성 피부 · 2026.05.19</text>
      <line x1="218" y1="90" x2="481" y2="90" stroke="#f3f4f6" strokeWidth="1" />

      {/* 레이더 차트 배경 육각형 */}
      {[0.33, 0.67, 1].map(f => (
        <polygon key={f} points={hexPoly(R * f)} fill="none" stroke="#e5e7eb" strokeWidth="1" />
      ))}

      {/* 축선 */}
      {Array.from({ length: N }, (_, i) => {
        const p = pt(i, R);
        return <line key={i} x1={cx} y1={cy} x2={p.x} y2={p.y} stroke="#e5e7eb" strokeWidth="1" />;
      })}

      {/* 데이터 폴리곤 */}
      <polygon points={dataPoly} fill="rgba(124,58,237,0.18)" stroke="#7c3aed" strokeWidth="1.5" />

      {/* 데이터 점 */}
      {VALUES.map((v, i) => {
        const p = pt(i, R * v);
        return <circle key={i} cx={p.x} cy={p.y} r="3" fill="#7c3aed" stroke="white" strokeWidth="1" />;
      })}

      {/* 레이블 */}
      {LABELS.map((label, i) => {
        const p = pt(i, R + 14);
        return (
          <text key={i} x={p.x} y={p.y + 3} textAnchor={ANCHORS[i]}
            fill="#374151" fontSize="9" fontFamily="sans-serif">{label}</text>
        );
      })}

      <line x1="218" y1="283" x2="481" y2="283" stroke="#f3f4f6" strokeWidth="1" />
      <circle cx="238" cy="298" r="3" fill="#ef4444" opacity="0.8" />
      <text x="248" y="302" fill="#6b7280" fontSize="9" fontFamily="sans-serif">개선필요 1개</text>
      <circle cx="308" cy="298" r="3" fill="#f59e0b" opacity="0.8" />
      <text x="318" y="302" fill="#6b7280" fontSize="9" fontFamily="sans-serif">보통 3개</text>
      <circle cx="358" cy="298" r="3" fill="#22c55e" opacity="0.8" />
      <text x="366" y="302" fill="#6b7280" fontSize="9" fontFamily="sans-serif">좋음 2개</text>
    </svg>
  );
}

/* SVG illustration: Step 03 */
function RoutineVisual() {
  const items = [
    { step: 1, cat: '토너', name: '수분 진정 토너',  price: '18,000원' },
    { step: 2, cat: '에멀젼',   name: '수분 에멀젼', price: '25,000원' },
    { step: 3, cat: '앰플',   name: '진정 앰플',      price: '45,000원' },
    { step: 4, cat: '크림',   name: '보습 크림',      price: '38,000원' },
  ];

  return (
    <svg viewBox="0 0 480 300" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="cs3" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="4" stdDeviation="10" floodColor="rgba(0,0,0,0.08)" />
        </filter>
      </defs>

      <rect x="28" y="10" width="424" height="278" rx="10" fill="white" filter="url(#cs3)" />

      <text x="50" y="40" fill="#111827" fontSize="14" fontWeight="700" fontFamily="sans-serif">AI 추천 루틴</text>
      <text x="50" y="56" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">건성 피부 맞춤 · 예산 범위 내 최적 조합</text>
      <line x1="28" y1="65" x2="452" y2="65" stroke="#f3f4f6" strokeWidth="1" />

      {items.map((item, i) => {
        const ry = 87 + i * 40;
        return (
          <g key={item.step}>
            {i > 0 && (
              <line x1="50" y1={ry - 12} x2="430" y2={ry - 12} stroke="#f5f5f5" strokeWidth="1" />
            )}
            <rect x="50" y={ry - 10} width="52" height="20" rx="4" fill="#e8e4ff" />
            <text x="76" y={ry + 4} textAnchor="middle" fill="#7c3aed" fontSize="9" fontWeight="700" fontFamily="sans-serif">STEP {item.step}</text>
            <text x="114" y={ry - 1} fill="#9ca3af" fontSize="9" fontFamily="sans-serif">{item.cat}</text>
            <text x="114" y={ry + 12} fill="#111827" fontSize="12" fontWeight="600" fontFamily="sans-serif">{item.name}</text>
            <text x="430" y={ry + 12} textAnchor="end" fill="#374151" fontSize="12" fontWeight="500" fontFamily="sans-serif">{item.price}</text>
          </g>
        );
      })}

      <line x1="50" y1="247" x2="430" y2="247" stroke="#e5e7eb" strokeWidth="1" />
      <text x="50"  y="263" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">소요 약 8분</text>
      <circle cx="116" cy="260" r="2" fill="#d1d5db" />
      <text x="122" y="263" fill="#9ca3af" fontSize="10" fontFamily="sans-serif">제품 4개</text>
      <text x="430" y="263" textAnchor="end" fill="#7c3aed" fontSize="11" fontWeight="700" fontFamily="sans-serif">총 126,000원</text>
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

  useEffect(() => {
    const els = document.querySelectorAll('.mp-reveal');
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add('mp-visible');
            observer.unobserve(e.target);
          }
        });
      },
      { threshold: 0.12 }
    );
    els.forEach((el) => observer.observe(el));
    return () => observer.disconnect();
  }, []);

  return (
    <main className="mp">

      {/* HERO */}
      <section className="mp-hero">
        <div className="mp-hero-photo-wrap">
          <img
            src="/hero-photo.png"
            alt="스킨케어 모델"
            className="mp-hero-photo"
            onError={(e) => { e.currentTarget.style.display = 'none'; }}
          />
        </div>

        <div className="mp-hero-inner">
          <div className="mp-hero-text">
            <h1 className="mp-hero-h1">
              피부는 모두 다르기에<br />케어도 달라야 합니다
            </h1>
            <p className="mp-hero-desc">
              피부 분석부터 맞춤 화장품 추천까지,<br />
              당신에게 맞는 스킨케어 루틴을 AI가 설계합니다.
            </p>
            <div className="mp-hero-btns">
              <button className="mp-btn-white" onClick={handleStart}>
                피부 분석 시작하기
                <ChevronRight size={15} style={{ display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* SHOWCASE */}
      <section className="mp-showcase">
        <div className="mp-container">
          <div className="mp-showcase-head mp-reveal">
            <p className="mp-eyebrow mp-eyebrow--purple">피부 분석부터 루틴까지</p>
            <h2 className="mp-showcase-h2">나에게 맞는<br />스킨케어 루틴을 찾다</h2>
            <p className="mp-showcase-sub">
              AI가 6가지 피부 지표를 분석하고, 예산에 맞는 스킨케어 루틴까지 한번에 제안해요.
            </p>
          </div>

          <div className="mp-cards-scatter">
            <div className="mp-scatter-card mp-scatter-card--l mp-reveal mp-reveal-d1">
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

            <div className="mp-scatter-card mp-scatter-card--r mp-reveal mp-reveal-d2">
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

      {/* HOW TO */}
      <section className="mp-how">
        <div className="mp-container">
          <div className="mp-how-head mp-reveal">
            <h2 className="mp-how-h2">어떻게 하나요?</h2>
            <p className="mp-how-sub">3분이면 나만의 피부 리포트와 루틴이 완성돼요.</p>
          </div>
          <div className="mp-how-steps">
            {HOW_STEPS.map(({ num, title, desc }, idx) => (
              <div className={`mp-how-card mp-reveal mp-reveal-d${idx + 1}`} key={num}>
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

      {/* CTA */}
      <section className="mp-cta">
        <div className="mp-container mp-cta-inner mp-reveal">
          <p className="mp-eyebrow mp-eyebrow--purple">AI 피부 분석 시작하기</p>
          <h2 className="mp-cta-h2">내 피부에 맞는 루틴,<br />지금 바로 찾아볼까요?</h2>
          <p className="mp-cta-desc">회원가입 후 즉시 시작할 수 있어요. 3분이면 충분합니다.</p>
          <button className="mp-btn-purple" onClick={handleStart}>
            AI 피부 분석 시작하기
            <ChevronRight size={15} style={{ display: 'inline', verticalAlign: 'middle', marginLeft: 4 }} />
          </button>
        </div>
      </section>

    </main>
  );
}
