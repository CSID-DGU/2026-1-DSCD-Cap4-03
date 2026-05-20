import { Link } from 'react-router-dom';
import './Footer.css';

export default function Footer() {
  return (
    <footer className="site-footer">

      <div className="site-footer-top">
        <div className="site-footer-brand">
          <span className="site-footer-logo">ROUPLE</span>
          <p className="site-footer-desc">AI 기반 맞춤형 스킨케어 솔루션</p>
        </div>

        <div className="site-footer-col">
          <div className="site-footer-col-title">고객지원</div>
          <Link to="/privacy" className="site-footer-link">개인정보처리방침</Link>
          <Link to="/terms" className="site-footer-link">이용약관</Link>
        </div>
      </div>

      <div className="site-footer-bottom">
        <span className="site-footer-copy">© 2026 ROUPLE. All rights reserved.</span>
        <div className="site-footer-bottom-links">
          <Link to="/privacy" className="site-footer-bottom-link">개인정보처리방침</Link>
          <span className="site-footer-dot">·</span>
          <Link to="/terms" className="site-footer-bottom-link">이용약관</Link>
        </div>
      </div>

    </footer>
  );
}
