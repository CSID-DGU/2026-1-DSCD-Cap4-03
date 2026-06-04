import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Eye, EyeOff } from 'lucide-react';
import { useAuth } from '../context/useAuth';
import { authApi } from '../api/auth';
import '../styles/Auth.css';

export default function SignupPage() {
  const navigate = useNavigate();
  const { login } = useAuth();

  const [form, setForm] = useState({
    user_name: '',
    nickname: '',
    email: '',
    password: '',
    passwordConfirm: '',
  });
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showPasswordConfirm, setShowPasswordConfirm] = useState(false);

  const set = (field: string) => (e: React.ChangeEvent<HTMLInputElement>) =>
    setForm((prev) => ({ ...prev, [field]: e.target.value }));

  const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

  const handleSignup = async () => {
    if (!form.user_name || !form.email || !form.password) {
      setError('이름, 이메일, 비밀번호는 필수입니다.'); return;
    }
    if (!EMAIL_REGEX.test(form.email)) {
      setError('올바른 이메일 형식으로 입력해주세요. (예: example@gmail.com)'); return;
    }
    if (form.password !== form.passwordConfirm) {
      setError('비밀번호가 일치하지 않아요.'); return;
    }
    setError('');
    setLoading(true);
    try {
      const res = await authApi.signup({
        user_name: form.user_name,
        nickname: form.nickname || form.user_name,
        email: form.email,
        password: form.password,
      });
      login(res.access_token, res.user_id, res.nickname);
      navigate('/userinfo');
    } catch (err) {
      const msg = (err as Error).message || '';
      if (msg.toLowerCase().includes('email') || msg.includes('이메일') || msg.includes('already') || msg.includes('duplicate')) {
        setError('이미 가입된 이메일이에요. 다른 이메일을 사용해주세요.');
      } else {
        setError(msg || '회원가입에 실패했어요.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h2>회원가입</h2>
        <input placeholder="이름" className="auth-input" value={form.user_name} onChange={set('user_name')} />
        <input placeholder="닉네임(한글로 별명을 작성해주세요)" className="auth-input" value={form.nickname} onChange={set('nickname')} />
        <input placeholder="이메일" className="auth-input" type="email" value={form.email} onChange={set('email')} />

        <div className="auth-pw-wrap">
          <input
            type={showPassword ? 'text' : 'password'}
            placeholder="비밀번호"
            className="auth-input auth-input--pw"
            value={form.password}
            onChange={set('password')}
          />
          <button type="button" className="auth-pw-toggle" onClick={() => setShowPassword(v => !v)}>
            {showPassword ? <EyeOff size={17} color="#9CA3AF" /> : <Eye size={17} color="#9CA3AF" />}
          </button>
        </div>

        <div className="auth-pw-wrap">
          <input
            type={showPasswordConfirm ? 'text' : 'password'}
            placeholder="비밀번호 확인"
            className="auth-input auth-input--pw"
            value={form.passwordConfirm}
            onChange={set('passwordConfirm')}
          />
          <button type="button" className="auth-pw-toggle" onClick={() => setShowPasswordConfirm(v => !v)}>
            {showPasswordConfirm ? <EyeOff size={17} color="#9CA3AF" /> : <Eye size={17} color="#9CA3AF" />}
          </button>
        </div>

        {error && <p style={{ color: '#ef4444', fontSize: '0.85rem', margin: '0' }}>{error}</p>}
        <button className="btn-primary" onClick={handleSignup} disabled={loading}>
          {loading ? '가입 중...' : '회원가입'}
        </button>
      </div>
    </div>
  );
}
