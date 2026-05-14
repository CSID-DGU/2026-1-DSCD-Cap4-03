import { useNavigate } from 'react-router-dom';
import '../styles/Auth.css';

export default function SignupPage() {
  const navigate = useNavigate();

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h2>회원가입</h2>

        <input placeholder="이름" className="auth-input" />
        <input placeholder="닉네임" className="auth-input" />
        <input placeholder="이메일" className="auth-input" />
        <input type="password" placeholder="비밀번호" className="auth-input" />
        <input type="password" placeholder="비밀번호 확인" className="auth-input" />

        <button className="btn-primary" onClick={() => navigate('/userinfo')}>
          회원가입
        </button>
      </div>
    </div>
  );
}