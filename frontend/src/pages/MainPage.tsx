import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/useAuth';
import { ChevronRight } from 'lucide-react';
import './MainPage.css';

const ANALYSIS_METRICS = [
  { label: '진정',   value: 73, color: '#ef4444' },
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
    title: '피부타입 & 고민 설정',
    desc: '피부 타입과 주요 고민사항을 간단히 선택해주세요. 건성·지성·복합성 등 피부 타입부터 여드름·모공·주름·홍조 등 개선하고 싶은 부위를 체크하면, AI가 맞춤형 분석 기준을 자동으로 설정해 더 정확한 리포트를 드려요.',
  },
  {
    num: '02',
    title: '셀피 한 장 업로드 & 분석',
    desc: '스마트폰 카메라 한 장이면 충분해요. 밝은 조명 아래 생얼 상태로 정면을 바라보고 찍은 사진을 업로드하면, AI가 진정·수분·탄력·모공·색소침착·주름 6가지 지표를 수치화해 개인 피부 리포트를 완성해드려요.',
  },
  {
    num: '03',
    title: '맞춤 루틴 확인',
    desc: '분석 결과와 내 예산을 함께 고려해 최적의 제품 조합을 추천해드려요. 클렌저부터 크림까지 사용 순서와 방법까지 AI가 안내하므로, 처음 스킨케어를 시작하는 분도 어렵지 않게 따라올 수 있어요.',
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
  const metrics = [
    { label: '진정', value: 73, color: '#ef4444' },
    { label: '수분',   value: 58, color: '#f59e0b' },
    { label: '탄력',   value: 48, color: '#f59e0b' },
    { label: '모공',   value: 65, color: '#f59e0b' },
    { label: '색소침착',   value: 42, color: '#22c55e' },
    { label: '주름',   value: 23, color: '#22c55e' },
  ];
  const BAR_W = 130;
  const BAR_X = 292;

  return (
    <svg viewBox="0 0 490 318" className="mp-how-svg" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <filter id="cs2a" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="4" stdDeviation="10" floodColor="rgba(0,0,0,0.08)" />
        </filter>
        <filter id="cs2b" x="-20%" y="-20%" width="140%" height="140%">
          <feDropShadow dx="0" dy="6" stdDeviation="14" floodColor="rgba(0,0,0,0.10)" />
        </filter>
      </defs>

      <rect x="12" y="10" width="263" height="268" rx="10" fill="white" filter="url(#cs2a)" />
      <text x="30" y="38" fill="#111827" fontSize="12" fontWeight="700" fontFamily="sans-serif">피부 사진 분석</text>
      <text x="30" y="52" fill="#9ca3af" fontSize="9" fontFamily="sans-serif">AI 스캔 중...</text>
      <line x1="12" y1="60" x2="275" y2="60" stroke="#f3f4f6" strokeWidth="1" />

      <rect x="22" y="67" width="243" height="197" rx="8" fill="#f9f7ff" />
      <ellipse cx="143" cy="162" rx="60" ry="76" fill="#fde8d8" />
      <ellipse cx="125" cy="147" rx="8" ry="5" fill="#f3c4a0" />
      <ellipse cx="161" cy="147" rx="8" ry="5" fill="#f3c4a0" />
      <ellipse cx="125" cy="145" rx="5" ry="3" fill="#4b2e20" opacity="0.65" />
      <ellipse cx="161" cy="145" rx="5" ry="3" fill="#4b2e20" opacity="0.65" />
      <path d="M132 168 Q143 176 154 168" stroke="#c8856a" strokeWidth="1.5" fill="none" strokeLinecap="round" />

      {[90, 115, 140, 165, 190, 215, 240].map(y => (
        <line key={`h${y}`} x1="22" y1={y} x2="265" y2={y} stroke="#7c3aed" strokeWidth="0.5" strokeDasharray="4 4" opacity="0.15" />
      ))}
      {[60, 90, 120, 143, 168, 200, 240].map(x => (
        <line key={`v${x}`} x1={x} y1="67" x2={x} y2="264" stroke="#7c3aed" strokeWidth="0.5" strokeDasharray="4 4" opacity="0.15" />
      ))}

      <path d="M30 78 L30 68 L40 68" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M256 78 L256 68 L246 68" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M30 254 L30 264 L40 264" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />
      <path d="M256 254 L256 264 L246 264" stroke="rgba(124,58,237,0.7)" strokeWidth="2" fill="none" strokeLinecap="round" />

      <rect x="82" y="70" width="102" height="18" rx="9" fill="rgba(124,58,237,0.75)" />
      <text x="133" y="82" textAnchor="middle" fill="white" fontSize="8" fontWeight="700" fontFamily="sans-serif">AI 피부 분석 중</text>

      {([[143,110],[108,142],[178,142],[123,178],[163,178],[143,220]] as [number,number][]).map(([x,y],i) => (
        <circle key={i} cx={x} cy={y} r="3.5" fill="#7c3aed" stroke="white" strokeWidth="1.5" opacity="0.65" />
      ))}

      <rect x="218" y="40" width="263" height="268" rx="10" fill="white" filter="url(#cs2b)" />
      <text x="236" y="68" fill="#111827" fontSize="12" fontWeight="700" fontFamily="sans-serif">AI 피부 리포트</text>
      <text x="236" y="82" fill="#9ca3af" fontSize="9" fontFamily="sans-serif">건성 피부 · 2026.05.19</text>
      <line x1="218" y1="90" x2="481" y2="90" stroke="#f3f4f6" strokeWidth="1" />

      {metrics.map((m, i) => {
        const ry = 108 + i * 30;
        const barW = Math.round(BAR_W * m.value / 100);
        return (
          <g key={m.label}>
            <text x="236" y={ry + 5} fill="#374151" fontSize="10" fontFamily="sans-serif">{m.label}</text>
            <rect x={BAR_X} y={ry - 5} width={BAR_W} height={8} rx="4" fill="#f3f4f6" />
            <rect x={BAR_X} y={ry - 5} width={barW} height={8} rx="4" fill={m.color} opacity="0.85" />
            <text x={BAR_X + BAR_W + 8} y={ry + 5} fill={m.color} fontSize="10" fontWeight="700" fontFamily="sans-serif">{m.value}%</text>
          </g>
        );
      })}

      <line x1="218" y1="283" x2="481" y2="283" stroke="#f3f4f6" strokeWidth="1" />
      <circle cx="238" cy="298" r="3" fill="#ef4444" opacity="0.8" />
      <text x="246" y="302" fill="#6b7280" fontSize="9" fontFamily="sans-serif">개선필요 4개</text>
      <circle cx="328" cy="298" r="3" fill="#22c55e" opacity="0.8" />
      <text x="336" y="302" fill="#6b7280" fontSize="9" fontFamily="sans-serif">양호 2개</text>
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
            <rect x="50" y={ry - 10} width="52" height="20" rx="4" fill="#f3f4f6" />
            <text x="76" y={ry + 4} textAnchor="middle" fill="#374151" fontSize="9" fontWeight="700" fontFamily="sans-serif">STEP {item.step}</text>
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
          <div className="mp-showcase-head">
            <p className="mp-eyebrow mp-eyebrow--purple">피부 분석부터 루틴까지</p>
            <h2 className="mp-showcase-h2">나에게 맞는<br />스킨케어 루틴을 찾다</h2>
            <p className="mp-showcase-sub">
              AI가 6가지 피부 지표를 분석하고, 예산에 맞는 스킨케어 루틴까지 한번에 제안해요.
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

      {/* HOW TO */}
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

      {/* CTA */}
      <section className="mp-cta">
        <div className="mp-container mp-cta-inner">
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
