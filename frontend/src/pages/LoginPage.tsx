import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/useAuth';
import { authApi } from '../api/auth';

export default function LoginPage() {
  const navigate = useNavigate();
  const location = useLocation();
  const { login } = useAuth();
  const from: string = location.state?.from ?? '/';

  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleLogin = async () => {
    if (!email || !password) { setError('이메일과 비밀번호를 입력해주세요.'); return; }
    setError('');
    setLoading(true);
    try {
      const res = await authApi.login({ email, password });
      login(res.access_token, res.user_id, res.nickname);
      navigate(from, { replace: true });
    } catch (err) {
      setError((err as Error).message || '로그인에 실패했어요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h2>로그인</h2>
        <input
          placeholder="이메일"
          className="auth-input"
          type="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
        />
        <input
          type="password"
          placeholder="비밀번호"
          className="auth-input"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleLogin()}
        />
        {error && <p style={{ color: '#ef4444', fontSize: '0.85rem', margin: '0' }}>{error}</p>}
        <button className="btn-primary" onClick={handleLogin} disabled={loading}>
          {loading ? '로그인 중...' : '로그인'}
        </button>
        <p className="auth-switch">
          회원이 아니신가요?{' '}
          <span onClick={() => navigate('/signup')}>회원가입</span>
        </p>
      </div>
    </div>
  );
}
