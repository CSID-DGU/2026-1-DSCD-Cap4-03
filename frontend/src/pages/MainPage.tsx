import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/useAuth';
import './MainPage.css';


const FEATURES = [
  { icon: '📸', cls: 'purple', title: '사진 한 장으로', desc: '스마트폰으로 찍은 셀피 한 장만 있으면 돼요. 가이드라인이 정확한 분석을 도와줘요.' },
  { icon: '🔬', cls: 'pink',   title: '6가지 피부 지표 분석', desc: '수분도, 색소침착, 탄력도, 모공, 주름, 여드름을 정밀하게 측정해 개인화된 리포트를 드려요.' },
  { icon: '✨', cls: 'green',  title: '맞춤 루틴 추천', desc: '분석 결과를 바탕으로 나에게 딱 맞는 스킨케어 순서와 제품을 추천해드려요.' },
];

const STEPS = [
  { num: '01', title: '피부타입 & 고민 설정', desc: '피부 타입과 주요 고민을 선택해 더 정확한 분석을 받아보세요.' },
  { num: '02', title: '피부 사진 업로드', desc: '밝은 조명에서 정면 사진을 찍어 업로드하세요. 얼굴 가이드라인이 도와드려요.' },
  { num: '03', title: 'AI 피부 분석', desc: '6가지 지표를 AI가 정밀하게 분석해 개인화된 리포트를 드려요.' },
  { num: '04', title: '맞춤 루틴 추천', desc: '분석 결과를 기반으로 나만의 스킨케어 루틴과 제품을 추천해 드려요.' },
];

const METRICS = [
  { val: 72, label: '수분도' },
  { val: 45, label: '색소침착' },
  { val: 68, label: '탄력도' },
  { val: 55, label: '모공' },
  { val: 80, label: '주름' },
  { val: 60, label: '여드름' },
];

export default function MainPage() {
  const navigate = useNavigate();
  const { isLoggedIn } = useAuth();

  const handleStart = () => {
    if (isLoggedIn) {
      navigate('/diagnosis');
    } else {
      navigate('/login');
    }
  };

  return (
    <main className="main-page">

      {/* ── Hero ── */}
      <section className="hero">
        <div className="hero-badge">
          <span className="badge-dot" />
          AI 피부 분석 서비스
        </div>
        <h1 className="hero-title">
          당신의 피부,<br />
          <span className="highlight">AI가 분석해드려요</span>
        </h1>
        <p className="hero-sub">
          사진 한 장으로 6가지 피부 지표를 분석하고<br />
          나만의 루틴을 찾아보세요
        </p>
        <div className="hero-cta">
          <button className="btn-primary" onClick={handleStart}>
            피부 진단 시작하기 →
          </button>
        </div>
      </section>

      {/* ── Feature 카드 ── */}
      <section className="sec">
        <span className="sec-badge">ROUPLE 추천법</span>
        <h2 className="section-title">
          AI 기반 정밀 피부 분석으로<br />
          <em>나만의 루틴</em>을 찾아보세요
        </h2>
        <p className="section-sub">
          스킨케어는 맞춤이 전부예요.<br />
          내 피부에 꼭 맞는 제품과 루틴을 AI가 찾아드릴게요.
        </p>
        <div className="feature-cards">
          {FEATURES.map(({ icon, cls, title, desc }) => (
            <div key={title} className="fcard">
              <div className={`fcard-icon ${cls}`}>{icon}</div>
              <h3>{title}</h3>
              <p>{desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── 스텝 ── */}
      <section className="sec-gray">
        <span className="sec-badge">이용 방법</span>
        <h2 className="section-title">딱 4단계면 끝이에요</h2>
        <p className="section-sub">3분이면 나만의 스킨케어 루틴이 완성돼요</p>
        <div className="steps">
          {STEPS.map(({ num, title, desc }) => (
            <div key={num} className="step-row">
              <div className="step-circle">{num}</div>
              <div>
                <div className="step-title">{title}</div>
                <p className="step-desc">{desc}</p>
              </div>
            </div>
          ))}
        </div>
        <div className="step-cta">
          <button className="btn-purple" onClick={handleStart}>
            지금 바로 시작하기 →
          </button>
        </div>
      </section>

      {/* ── 메트릭 ── */}
      <section className="sec">
        <span className="sec-badge">피부의 모든 것을 정밀하게 분석해요</span>
        <h2 className="section-title">
          모든 피부 지표를 담아,<br />
          AI가 스킨케어를 설계해요
        </h2>
        <p className="section-sub">
          수분, 탄력, 색소침착 등 6가지 지표를 담은 리포트로<br />
          나에게 딱 맞는 제품과 루틴을 찾아드릴게요
        </p>
        <div className="metric-grid">
          {METRICS.map(({ val, label }) => (
            <div key={label} className="mcard">
              <div className="mcard-val">{val}</div>
              <div className="mcard-label">{label}</div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA ── */}
      <section className="cta-section">
        <span className="sec-badge" style={{ background: 'rgba(255,255,255,0.2)', color: '#fff' }}>
          지금 바로 피부 진단받기
        </span>
        <h2>내 피부에 딱 맞는<br />루틴을 찾아볼까요?</h2>
        <p>
          ROUPLE을 통해 피부 진단을 받아보세요<br />
          건강하고 깨끗한 피부로의 루틴을 시작하세요
        </p>
        <div className="cta-btns">
          <button className="btn-white-sm" onClick={() => navigate('/signup')}>
            회원가입하러 가기
          </button>
        </div>
      </section>

      {/* ── 푸터 ── */}
      <footer className="footer">
        <div>
          <div className="footer-logo">ROUPLE</div>
          <div className="footer-desc">AI 기반 피부 분석 및 스킨케어 루틴 추천 서비스</div>
        </div>
        <div style={{ display: 'flex', gap: '40px' }}>
          <div className="footer-col">
            <h4>서비스</h4>
            <a onClick={() => navigate('/diagnosis')}>피부 진단</a>
            <a onClick={() => navigate('/routine')}>루틴 추천</a>
            <a onClick={() => navigate('/products')}>제품 찾기</a>
          </div>
          <div className="footer-col">
            <h4>정보</h4>
            <a>개인정보처리방침</a>
            <a>서비스 이용약관</a>
            <a>고객센터</a>
          </div>
        </div>
      </footer>
      <div className="footer-bottom">© 2026 ROUPLE. All rights reserved.</div>

    </main>
  );
}