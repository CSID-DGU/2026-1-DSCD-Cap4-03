import { useState, type ReactNode } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Wallet, Droplets, Leaf, Sparkles, Package } from 'lucide-react';
import './BudgetPage.css';

const CATEGORIES = ['전체', '토너', '에멀젼', '앰플', '크림'] as const;
type Category = typeof CATEGORIES[number];

const BUDGET_OPTIONS: Record<Category, { label: string; min: number | null; max: number | null }[]> = {
  전체: [
    { label: '설정 안함',  min: null,   max: null   },
    { label: '0~10만원',   min: 0,      max: 100000 },
    { label: '10~15만원',  min: 100000, max: 150000 },
    { label: '15~20만원',  min: 150000, max: 200000 },
    { label: '20만원+',    min: 200000, max: null   },
  ],
  토너: [
    { label: '설정 안함', min: null,  max: null  },
    { label: '0~2만원',   min: 0, max: 20000 },
    { label: '2~3만원',   min: 20000, max: 30000 },
    { label: '3~5만원',   min: 30000, max: 50000 },
    { label: '5만원+',    min: 50000, max: null  },
  ],
  에멀젼: [
    { label: '설정 안함', min: null,  max: null  },
    { label: '0~2만원',   min: 0, max: 20000 },
    { label: '2~3만원',   min: 20000, max: 30000 },
    { label: '3~5만원',   min: 30000, max: 50000 },
    { label: '5만원+',    min: 50000, max: null  },
  ],
  앰플: [
    { label: '설정 안함', min: null,  max: null  },
    { label: '0~2만원',   min: 0, max: 20000 },
    { label: '2~3만원',   min: 20000, max: 30000 },
    { label: '3~5만원',   min: 30000, max: 50000 },
    { label: '5만원+',    min: 50000, max: null  },
  ],
  크림: [
    { label: '설정 안함', min: null,  max: null  },
    { label: '0~2만원',   min: 0, max: 20000 },
    { label: '2~3만원',   min: 20000, max: 30000 },
    { label: '3~5만원',   min: 30000, max: 50000 },
    { label: '5만원+',    min: 50000, max: null  },
  ],
};

type BudgetRange = { min: number | null; max: number | null };
type BudgetState = Record<Category, BudgetRange>;

export default function BudgetPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const resultId = location.state?.resultId;
  const imageId = location.state?.imageId;

  const NONE: BudgetRange = { min: null, max: null };

  const [budget, setBudget] = useState<BudgetState>({
    전체: NONE, 토너: NONE, 에멀젼: NONE, 앰플: NONE, 크림: NONE,
  });

  const handleSelect = (cat: Category, min: number | null, max: number | null) => {
    setBudget((prev) => ({ ...prev, [cat]: { min, max } }));
  };

  const isSelected = (cat: Category, min: number | null, max: number | null) =>
    budget[cat].min === min && budget[cat].max === max;

  const handleNext = () => {
    navigate('/loading', {
      state: {
        type: 'routine', resultId, imageId,
        budget: {
          total_min:    budget['전체'].min,
          total_max:    budget['전체'].max,
          toner_min:    budget['토너'].min,
          toner_max:    budget['토너'].max,
          emulsion_min: budget['에멀젼'].min,
          emulsion_max: budget['에멀젼'].max,
          ampoule_min:  budget['앰플'].min,
          ampoule_max:  budget['앰플'].max,
          cream_min:    budget['크림'].min,
          cream_max:    budget['크림'].max,
        },
      },
    });
  };

  const CAT_ICONS: Record<Category, ReactNode> = {
    전체: <Wallet size={15} />, 토너: <Droplets size={15} />, 에멀젼: <Leaf size={15} />, 앰플: <Sparkles size={15} />, 크림: <Package size={15} />,
  };

  return (
    <div className="budget-page">
      <div className="budget-banner">
        <div className="budget-banner-inner">
          <div className="budget-banner-badge">ROUTINE SETTING</div>
          <h1 className="budget-title">어느 정도 예산을 생각하고 계세요?</h1>
          <p className="budget-sub">카테고리별 예산을 설정하면 그 안에서 최적의 제품을 추천해드려요. 건너뛰어도 괜찮아요.</p>
        </div>
      </div>

      <div className="budget-body">
        {CATEGORIES.map((cat) => (
          <div className="budget-section" key={cat}>
            <div className="budget-cat-label" style={{ display: 'flex', alignItems: 'center', gap: 6 }}>{CAT_ICONS[cat]} {cat === '전체' ? '전체 예산' : cat}</div>
            <div className="budget-options">
              {BUDGET_OPTIONS[cat].map(({ label, min, max }) => (
                <label key={label} className={`budget-radio ${isSelected(cat, min, max) ? 'selected' : ''}`}>
                  <input type="radio" name={cat} value={label} checked={isSelected(cat, min, max)} onChange={() => handleSelect(cat, min, max)} />
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
