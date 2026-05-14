import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/useAuth';

export default function LoginPage() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLogin = () => {
    login();
    navigate('/');
  };
  return (
    <div className="auth-page">
      <div className="auth-container">
        <h2>로그인</h2>

        <input placeholder="아이디" className="auth-input" />
        <input type="password" placeholder="비밀번호" className="auth-input" />

        <button className="btn-primary" onClick={handleLogin}>로그인</button>

        {/*
        <button className="btn-google">Google로 로그인</button> */}
        
        <p className="auth-switch">
          회원이 아니신가요?{' '}
          <span onClick={() => navigate('/signup')}>회원가입</span>
        </p>
      </div>
    </div>
  );
}