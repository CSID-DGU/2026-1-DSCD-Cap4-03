import { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import './BudgetPage.css';

const CATEGORIES = ['전체', '토너', '에멀젼', '앰플', '크림'] as const;
type Category = typeof CATEGORIES[number];

const BUDGET_OPTIONS: Record<Category, { label: string; value: number | null }[]> = {
  전체: [
    { label: '설정 안함', value: null },
    { label: '~5만원', value: 50000 },
    { label: '~10만원', value: 100000 },
    { label: '~15만원', value: 150000 },
    { label: '~20만원', value: 200000 },
    { label: '20만원+', value: 999999 },
  ],
  토너: [
    { label: '설정 안함', value: null },
    { label: '~1만원', value: 10000 },
    { label: '~2만원', value: 20000 },
    { label: '~3만원', value: 30000 },
    { label: '3만원+', value: 999999 },
  ],
  에멀젼: [
    { label: '설정 안함', value: null },
    { label: '~1만원', value: 10000 },
    { label: '~2만원', value: 20000 },
    { label: '~3만원', value: 30000 },
    { label: '3만원+', value: 999999 },
  ],
  앰플: [
    { label: '설정 안함', value: null },
    { label: '~1만원', value: 10000 },
    { label: '~2만원', value: 20000 },
    { label: '~3만원', value: 30000 },
    { label: '3만원+', value: 999999 },
  ],
  크림: [
    { label: '설정 안함', value: null },
    { label: '~1만원', value: 10000 },
    { label: '~2만원', value: 20000 },
    { label: '~3만원', value: 30000 },
    { label: '3만원+', value: 999999 },
  ],
};

type BudgetState = Record<Category, number | null>;

export default function BudgetPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const resultId = location.state?.resultId;

  const [budget, setBudget] = useState<BudgetState>({
    전체: null, 토너: null, 에멀젼: null, 앰플: null, 크림: null,
  });

  const handleSelect = (cat: Category, val: number | null) => {
    setBudget((prev) => ({ ...prev, [cat]: val }));
  };

  const handleNext = () => {
    navigate('/routine/result', { state: { resultId, budget } });
  };

  const CAT_ICONS: Record<Category, string> = {
    전체: '💰', 토너: '💧', 에멀젼: '🌿', 앰플: '✨', 크림: '🧴',
  };

  return (
    <div className="budget-page">
      <div className="budget-banner">
        <div className="budget-banner-inner">
          <div className="budget-banner-badge">BUDGET SETTING</div>
          <h1 className="budget-title">예산을 설정해주세요</h1>
          <p className="budget-sub">설정하지 않아도 괜찮아요. 원하는 항목만 골라보세요.</p>
        </div>
      </div>

      <div className="budget-body">
        {CATEGORIES.map((cat) => (
          <div className="budget-section" key={cat}>
            <div className="budget-cat-label">{CAT_ICONS[cat]} {cat === '전체' ? '전체 예산' : cat}</div>
            <div className="budget-options">
              {BUDGET_OPTIONS[cat].map(({ label, value }) => (
                <label key={label} className={`budget-radio ${budget[cat] === value ? 'selected' : ''}`}>
                  <input type="radio" name={cat} value={String(value)} checked={budget[cat] === value} onChange={() => handleSelect(cat, value)} />
                  {label}
                </label>
              ))}
            </div>
          </div>
        ))}
        <div className="budget-submit-wrap">
          <button className="btn-budget-next" onClick={handleNext}>루틴 추천 받기 →</button>
        </div>
      </div>
    </div>
  );
}
