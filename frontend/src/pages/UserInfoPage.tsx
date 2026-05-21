import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import AllergySelector, { buildAllergyItems } from '../components/AllergySelector';
import type { AllergySelectorValue } from '../components/AllergySelector';
import { userApi } from '../api/user';
import './UserInfoPage.css';

const SKIN_TYPES = ['건성', '지성', '중성', '복합성', '수부지', '모름'];

const CONCERNS = [
  { id: 'acne',        label: '여드름' },
  { id: 'wrinkle',     label: '주름' },
  { id: 'brightening', label: '미백' },
  { id: 'sebum',       label: '피지' },
  { id: 'dryness',     label: '속건조' },
  { id: 'redness',     label: '붉은기' },
  { id: 'dark_circle', label: '다크서클' },
  { id: 'atopy',       label: '아토피' },
  { id: 'sensitive',   label: '민감성' },
  { id: 'pore',        label: '모공' },
  { id: 'flushing',    label: '홍조' },
  { id: 'keratin',     label: '각질' },
  { id: 'none',        label: '해당사항 없음' },
];

function Section({ step, title, sub, children }: {
  step: number; title: string; sub?: string; children: React.ReactNode;
}) {
  return (
    <div className="ui-section">
      <div className="ui-section-label">
        <div className="ui-step-num">{step}</div>
        <div>
          <div className="ui-section-title">{title}</div>
          {sub && <div className="ui-section-sub">{sub}</div>}
        </div>
      </div>
      <div className="ui-section-body">{children}</div>
    </div>
  );
}

export default function UserInfoPage() {
  const navigate = useNavigate();

  const [gender, setGender] = useState('');
  const [birthdate, setBirthdate] = useState('');
  const [skinType, setSkinType] = useState('');
  const [concerns, setConcerns] = useState<string[]>([]);
  const [allergy, setAllergy] = useState<AllergySelectorValue>({ categories: [], ingredientIds: [] });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleBirthdate = (e: React.ChangeEvent<HTMLInputElement>) => {
    const digits = e.target.value.replace(/\D/g, '').slice(0, 8);
    let formatted = digits;
    if (digits.length > 6) formatted = digits.slice(0, 4) + '-' + digits.slice(4, 6) + '-' + digits.slice(6);
    else if (digits.length > 4) formatted = digits.slice(0, 4) + '-' + digits.slice(4);
    setBirthdate(formatted);
  };

  const handleConcern = (id: string) => {
    if (id === 'none') { setConcerns(['none']); return; }
    const filtered = concerns.filter((c) => c !== 'none');
    setConcerns(filtered.includes(id) ? filtered.filter((c) => c !== id) : [...filtered, id]);
  };

  const handleSubmit = async () => {
    setError('');
    setLoading(true);
    try {
      const concernLabels = concerns
        .filter((id) => id !== 'none')
        .map((id) => CONCERNS.find((c) => c.id === id)?.label ?? id);

      await userApi.updateProfile({
        gender: gender || undefined,
        birth: birthdate || undefined,
        skin_type: skinType || undefined,
        skin_concerns: concernLabels.length ? concernLabels : undefined,
      });

      const allergyItems = buildAllergyItems(allergy);
      if (allergyItems.length > 0) {
        await userApi.updateAllergies({ allergy_items: allergyItems });
      }

      navigate('/');
    } catch (err) {
      setError((err as Error).message || '저장에 실패했어요. 다시 시도해주세요.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ui-page">
      <div className="ui-banner">
        <div className="ui-banner-inner">
          <div className="ui-badge">내 피부 설정</div>
          <h1 className="ui-title">추가 정보를 입력해주세요</h1>
          <p className="ui-sub">더 정확한 피부 분석과 맞춤 루틴 추천을 위해 필요해요</p>
        </div>
      </div>

      <div className="ui-body">
        <div className="ui-card">
          <Section step={1} title="성별">
            <div className="ui-btn-group">
              {[{ val: 'female', label: '여성' }, { val: 'male', label: '남성' }].map((g) => (
                <button key={g.val} type="button"
                  className={`ui-select-btn ${gender === g.val ? 'active' : ''}`}
                  onClick={() => setGender(g.val)}>
                  {g.label}
                </button>
              ))}
            </div>
          </Section>

          <div className="ui-divider" />

          <Section step={2} title="생년월일">
            <input type="text" className="ui-input" value={birthdate} onChange={handleBirthdate} placeholder="yyyy-mm-dd" maxLength={10} />
          </Section>

          <div className="ui-divider" />

          <Section step={3} title="피부 타입" sub="본인의 평소 피부 타입을 선택해주세요">
            <div className="ui-btn-group">
              {SKIN_TYPES.map((t) => (
                <button key={t} type="button"
                  className={`ui-select-btn ${skinType === t ? 'active' : ''}`}
                  onClick={() => setSkinType(t)}>{t}</button>
              ))}
            </div>
          </Section>

          <div className="ui-divider" />

          <Section step={4} title="피부 고민" sub="복수 선택 가능해요">
            <div className="ui-concern-grid">
              {CONCERNS.map((c) => (
                <button key={c.id} type="button"
                  className={`ui-concern-btn ${concerns.includes(c.id) ? 'active' : ''}`}
                  onClick={() => handleConcern(c.id)}>
                  <span>{c.label}</span>
                </button>
              ))}
            </div>
          </Section>

          <div className="ui-divider" />

          <Section step={5} title="알레르기 성분 (선택)" sub="정확히 알고 있는 성분만 선택해주세요">
            <AllergySelector value={allergy} onChange={setAllergy} />
          </Section>

          {error && <p style={{ color: '#ef4444', textAlign: 'center', fontSize: '0.9rem' }}>{error}</p>}

          <div className="ui-submit-row">
            <div className="ui-summary">
              {skinType && <span className="ui-summary-tag">{skinType}</span>}
              {concerns.filter((c) => c !== 'none').map((c) => (
                <span key={c} className="ui-summary-tag">
                  {CONCERNS.find((x) => x.id === c)?.label}
                </span>
              ))}
              {allergy.categories.length > 0 && (
                <span className="ui-summary-tag allergy">알레르기 {allergy.ingredientIds.length}개 선택</span>
              )}
            </div>
            <button className="ui-submit-btn" onClick={handleSubmit} disabled={loading}>
              {loading ? '저장 중...' : '완료하기 →'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
