import { useState } from 'react';
import '../styles/Auth.css';

export default function UserInfoPage() {
  const skinTypes = ['건성','지성','중성','복합성','수부지','모름'];

  const concernsList = [
    '여드름','주름','미백','피지','속건조','붉은기',
    '다크서클','아토피','민감성','모공','홍조','각질','해당사항 없음'
  ];

  const [selectedSkinType, setSelectedSkinType] = useState<string | null>(null);
  const [selectedConcerns, setSelectedConcerns] = useState<string[]>([]);

  const [allergy, setAllergy] = useState('');
  const [customAllergy, setCustomAllergy] = useState('');

  // 피부 고민 선택 로직
  const handleConcernClick = (item: string) => {
    if (item === '해당사항 없음') {
      setSelectedConcerns(['해당사항 없음']);
      return;
    }

    let updated = selectedConcerns.filter(c => c !== '해당사항 없음');

    if (updated.includes(item)) {
      updated = updated.filter(c => c !== item);
    } else {
      updated.push(item);
    }

    setSelectedConcerns(updated);
  };

  return (
    <div className="auth-page">
      <div className="auth-container">
        <h2>추가 정보 입력</h2>

        {/* 성별 */}
        <div className="form-group">
          <label>성별</label>
          <select className="auth-input">
            <option value="">선택</option>
            <option value="male">남성</option>
            <option value="female">여성</option>
          </select>
        </div>

        {/* 생년월일 */}
        <div className="form-group">
          <label>생년월일</label>
          <input type="date" className="auth-input" />
        </div>

        {/* 피부 타입 */}
        <div className="form-group">
          <label>피부 타입</label>
          <div className="button-group">
            {skinTypes.map(type => (
              <button
                key={type}
                type="button"
                className={`select-btn ${selectedSkinType === type ? 'active' : ''}`}
                onClick={() => setSelectedSkinType(type)}
              >
                {type}
              </button>
            ))}
          </div>
        </div>

        {/* 피부 고민 */}
        <div className="form-group">
          <label>피부 고민</label>
          <div className="button-group">
            {concernsList.map(item => (
              <button
                key={item}
                type="button"
                className={`select-btn ${selectedConcerns.includes(item) ? 'active' : ''}`}
                onClick={() => handleConcernClick(item)}
              >
                {item}
              </button>
            ))}
          </div>
        </div>

        {/* 알러지 */}
        <div className="form-group">
          <label>알러지 (선택)</label>

          <select
            className="auth-input"
            value={allergy}
            onChange={(e) => setAllergy(e.target.value)}
          >
            <option value="">선택</option>
            <option value="향료">향료</option>
            <option value="알코올">알코올</option>
            <option value="에센셜오일">에센셜오일</option>
            <option value="파라벤">파라벤</option>
            <option value="기타">기타</option>
          </select>

          {allergy === '기타' && (
            <input
              className="auth-input"
              placeholder="알러지 입력"
              value={customAllergy}
              onChange={(e) => setCustomAllergy(e.target.value)}
            />
          )}
        </div>

        <button className="btn-primary">완료</button>
      </div>
    </div>
  );
}