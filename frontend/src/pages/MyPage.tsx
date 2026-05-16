import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { userApi, type UserProfile } from '../api/user';
import { productApi, type ProductSummary } from '../api/product';
import { useAuth } from '../context/useAuth';
import AllergySelector, { type AllergySelectorValue } from '../components/AllergySelector';
import './MyPage.css';

type Tab = 'info' | 'wishlist';

const TABS: { key: Tab; label: string; icon: string }[] = [
  { key: 'info',     label: '내 정보', icon: '👤' },
  { key: 'wishlist', label: '찜 목록', icon: '💜' },
];

const SKIN_TYPES = ['건성', '지성', '복합성', '중성', '민감성', '수부지', '모름'];
const GENDER_LABEL: Record<string, string> = { female: '여성', male: '남성' };

export default function MyPage() {
  const navigate = useNavigate();
  const { logout } = useAuth();

  const [activeTab, setActiveTab] = useState<Tab>('info');
  const [isEditing, setIsEditing] = useState(false);

  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [formName, setFormName] = useState('');
  const [formGender, setFormGender] = useState('');
  const [formSkinType, setFormSkinType] = useState('');
  const [allergy, setAllergy] = useState<AllergySelectorValue>({ categories: [], ingredientIds: [] });
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  const [wishlist, setWishlist] = useState<ProductSummary[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      userApi.getMe(),
      productApi.getWishlist(),
    ]).then(([user, wish]) => {
      setProfile(user);
      setFormName(user.nickname || user.user_name);
      setFormGender(user.gender || '');
      setFormSkinType(user.skin_type || '');
      setWishlist(wish.items);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleEdit = () => {
    setSaveMsg('');
    setIsEditing(true);
  };

  const handleCancel = () => {
    // 폼을 현재 저장된 profile로 되돌림
    setFormName(profile?.nickname || profile?.user_name || '');
    setFormGender(profile?.gender || '');
    setFormSkinType(profile?.skin_type || '');
    setAllergy({ categories: [], ingredientIds: [] });
    setSaveMsg('');
    setIsEditing(false);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg('');
    try {
      const [updated] = await Promise.all([
        userApi.updateProfile({
          nickname: formName,
          gender: formGender || undefined,
          skin_type: formSkinType || undefined,
        }),
        userApi.updateAllergies({
          allergy_categories: allergy.categories,
          allergy_ingredient_ids: allergy.ingredientIds,
        }),
      ]);
      setProfile(updated);
      setSaveMsg('저장되었어요!');
      setIsEditing(false);
    } catch {
      setSaveMsg('저장에 실패했어요.');
    } finally {
      setSaving(false);
    }
  };

  const displayName = profile?.nickname || profile?.user_name || '사용자';

  return (
    <div className="my-page">

      <div className="my-banner">
        <div className="my-banner-inner">
          <div className="my-avatar">{displayName[0]}</div>
          <div className="my-banner-info">
            <div className="my-banner-name">{displayName}님</div>
            <div className="my-banner-email">{profile?.email ?? ''}</div>
          </div>
        </div>
      </div>

      <div className="my-tabs">
        <div className="my-tabs-inner">
          {TABS.map(({ key, label, icon }) => (
            <button key={key} className={`my-tab ${activeTab === key ? 'active' : ''}`} onClick={() => setActiveTab(key)}>
              <span>{icon}</span>
              <span>{label}</span>
            </button>
          ))}
        </div>
      </div>

      <div className="my-body">

        {/* ── 내 정보 ── */}
        {activeTab === 'info' && (
          <div className="my-section">
            <h2 className="my-section-title">내 정보</h2>
            {loading ? (
              <p style={{ color: '#7c3aed' }}>불러오는 중...</p>
            ) : !isEditing ? (
              /* ── 뷰 모드 ── */
              <div className="my-form">
                <div className="my-info-row">
                  <span className="my-label">닉네임</span>
                  <span className="my-info-value">{profile?.nickname || profile?.user_name || '-'}</span>
                </div>
                <div className="my-info-row">
                  <span className="my-label">이메일</span>
                  <span className="my-info-value">{profile?.email || '-'}</span>
                </div>
                <div className="my-info-row">
                  <span className="my-label">성별</span>
                  <span className="my-info-value">{GENDER_LABEL[profile?.gender ?? ''] || '-'}</span>
                </div>
                <div className="my-info-row">
                  <span className="my-label">피부 타입</span>
                  <span className="my-info-value">{profile?.skin_type || '-'}</span>
                </div>
                {saveMsg && <p style={{ fontSize: '0.875rem', color: '#22c55e' }}>{saveMsg}</p>}
                <button className="my-save-btn" onClick={handleEdit}>수정하기</button>
                <button
                  className="my-save-btn"
                  style={{ background: '#f3f4f6', color: '#6b7280', marginTop: '0.5rem' }}
                  onClick={() => { logout(); navigate('/login'); }}
                >
                  로그아웃
                </button>
              </div>
            ) : (
              /* ── 수정 모드 ── */
              <div className="my-form">
                <div>
                  <label className="my-label">닉네임</label>
                  <input className="my-input" value={formName} onChange={(e) => setFormName(e.target.value)} />
                </div>
                <div>
                  <label className="my-label">이메일</label>
                  <input className="my-input" value={profile?.email ?? ''} disabled />
                </div>
                <div>
                  <label className="my-label">성별</label>
                  <select className="my-input" value={formGender} onChange={(e) => setFormGender(e.target.value)}>
                    <option value="">선택 안함</option>
                    <option value="female">여성</option>
                    <option value="male">남성</option>
                  </select>
                </div>
                <div>
                  <label className="my-label">피부 타입</label>
                  <select className="my-input" value={formSkinType} onChange={(e) => setFormSkinType(e.target.value)}>
                    <option value="">선택 안함</option>
                    {SKIN_TYPES.map((t) => <option key={t}>{t}</option>)}
                  </select>
                </div>
                <div>
                  <label className="my-label">알레르기 성분</label>
                  <AllergySelector value={allergy} onChange={setAllergy} />
                </div>
                {saveMsg && <p style={{ fontSize: '0.875rem', color: '#ef4444' }}>{saveMsg}</p>}
                <button className="my-save-btn" onClick={handleSave} disabled={saving}>
                  {saving ? '저장 중...' : '수정 완료'}
                </button>
                <button
                  className="my-save-btn"
                  style={{ background: '#f3f4f6', color: '#6b7280', marginTop: '0.5rem' }}
                  onClick={handleCancel}
                >
                  취소
                </button>
              </div>
            )}
          </div>
        )}

        {/* ── 찜 목록 ── */}
        {activeTab === 'wishlist' && (
          <div className="my-section">
            <h2 className="my-section-title">찜 목록</h2>
            {loading && <p style={{ color: '#7c3aed' }}>불러오는 중...</p>}
            {!loading && wishlist.length === 0 && <div className="my-empty">찜한 제품이 없어요.</div>}
            {wishlist.length > 0 && (
              <div className="my-wish-grid">
                {wishlist.map((p) => (
                  <div className="my-wish-card" key={p.product_id} onClick={() => navigate(`/products/${p.product_id}`)}>
                    <img src={p.image_url} alt={p.product_name} className="my-wish-img" />
                    <div className="my-wish-brand">{p.brand_name}</div>
                    <div className="my-wish-name">{p.product_name}</div>
                    <div className="my-wish-price">{p.price.toLocaleString()}원</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

      </div>
    </div>
  );
}
