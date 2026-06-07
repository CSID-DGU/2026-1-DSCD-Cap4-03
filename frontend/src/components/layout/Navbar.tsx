import { Link, useLocation, useNavigate } from 'react-router-dom';
import { UserCircle } from 'lucide-react';
import { useAuth } from '../../context/useAuth';
import './Navbar.css';

export default function Navbar() {
  const { pathname } = useLocation();
  const { isLoggedIn, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="navbar">
      <div className="nav-left">
        <Link to="/" className="logo">ROUPLE</Link>
      </div>

      <div className="nav-center">
        <Link
          to="/analysis-history"
          className={`nav-link ${pathname === '/analysis-history' || pathname === '/analysis' || pathname === '/diagnosis' || pathname === '/loading' ? 'active' : ''}`}
        >
          피부 리포트
        </Link>

        <Link
          to="/routine-history"
          className={`nav-link ${pathname === '/routine-history' || pathname.startsWith('/routine') ? 'active' : ''}`}
        >
          루틴 리포트
        </Link>

        <Link
          to="/vanity"
          className={`nav-link ${pathname === '/vanity' || pathname.startsWith('/vanity') ? 'active' : ''}`}
        >
          화장대 리포트
        </Link>

        <Link
          to="/products"
          className={`nav-link ${pathname === '/products' || pathname.startsWith('/products') ? 'active' : ''}`}
        >
          제품 찾기
        </Link>

      </div>

      <div className="nav-right">
        {isLoggedIn && (
          <Link to="/mypage" className="nav-link nav-icon-link" aria-label="마이페이지" title="마이페이지">
            <UserCircle size={18} strokeWidth={2.3} />
          </Link>
        )}
        {isLoggedIn ? (
          <span className="nav-login" onClick={handleLogout}>로그아웃</span>
        ) : (
          <Link to="/login" className="nav-login">로그인</Link>
        )}
      </div>
    </nav>
  );
}
