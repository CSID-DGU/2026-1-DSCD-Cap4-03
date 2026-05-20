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
          to="/diagnosis"
          className={`nav-link ${pathname === '/diagnosis' ? 'active' : ''}`}
        >
          피부 진단 시작하기
        </Link>

        <Link
          to="/analysis-history"
          className={`nav-link ${pathname === '/analysis-history' || pathname === '/analysis' ? 'active' : ''}`}
        >
          피부 분석 결과
        </Link>

        <Link
          to="/routine-history"
          className={`nav-link ${pathname === '/routine-history' || pathname.startsWith('/routine') ? 'active' : ''}`}
        >
          내 루틴 보기
        </Link>

        <Link
          to="/products"
          className={`nav-link ${pathname === '/products' || pathname.startsWith('/products') ? 'active' : ''}`}
        >
          제품 찾기
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
