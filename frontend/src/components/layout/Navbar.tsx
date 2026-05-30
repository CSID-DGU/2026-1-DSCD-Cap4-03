import { Link, useLocation, useNavigate } from 'react-router-dom';
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

      {/* 로고 */}
      <div className="nav-left">
        <Link to="/" className="logo">ROUPLE</Link>
      </div>

      {/* 가운데 */}
      <div className="nav-center">
        <Link
          to="/analysis-history"
          className={`nav-link ${pathname === '/analysis-history' || pathname === '/analysis' || pathname === '/diagnosis' || pathname === '/loading' ? 'active' : ''}`}
        >
          스킨 리포트
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
          제품 탐색
        </Link>
      </div>

      {/* 오른쪽 */}
      <div className="nav-right">
        <Link to="/mypage" className="nav-link">마이페이지</Link>

        {isLoggedIn ? (
          <span className="nav-login" onClick={handleLogout}>로그아웃</span>
        ) : (
          <Link to="/login" className="nav-login">로그인</Link>
        )}
      </div>

    </nav>
  );
}
