import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { userApi, type UserProfile } from '../api/user';
import { productApi, type ProductSummary } from '../api/product';
import { useAuth } from '../context/useAuth';

import AllergySelector, { buildAllergyItems, type AllergySelectorValue } from '../components/AllergySelector';
import { User, Heart, type LucideIcon } from 'lucide-react';
import './MyPage.css';
import LoadingSpinner from '../components/common/LoadingSpinner';

type Tab = 'info' | 'wishlist';

const TABS: { key: Tab; label: string; Icon: LucideIcon }[] = [
  { key: 'info',     label: '내 정보', Icon: User },
  { key: 'wishlist', label: '찜 목록', Icon: Heart },
];

const SKIN_TYPES = ['건성', '지성', '복합성', '중성', '민감성', '수부지', '모름'];
const GENDER_LABEL: Record<string, string> = { female: '여성', male: '남성' };

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

export default function MyPage() {
  const navigate = useNavigate();
  const { logout, updateNickname } = useAuth();

  const [activeTab, setActiveTab] = useState<Tab>('info');
  const [isEditing, setIsEditing] = useState(false);

  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [formName, setFormName] = useState('');
  const [formGender, setFormGender] = useState('');
  const [formSkinType, setFormSkinType] = useState('');
  const [formConcerns, setFormConcerns] = useState<string[]>([]);
  const [allergy, setAllergy] = useState<AllergySelectorValue>({ categories: [], ingredientIds: [] });
  const [allergyLoading, setAllergyLoading] = useState(false);
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
      setFormConcerns(user.skin_concerns || []);
      setWishlist(wish.items);
    }).catch(() => {}).finally(() => setLoading(false));
  }, []);

  const handleConcern = (id: string) => {
    if (id === 'none') { setFormConcerns(['none']); return; }
    setFormConcerns((prev) => {
      const filtered = prev.filter((c) => c !== 'none');
      return filtered.includes(id) ? filtered.filter((c) => c !== id) : [...filtered, id];
    });
  };

  const handleEdit = async () => {
    setSaveMsg('');
    setFormConcerns(profile?.skin_concerns || []);
    setAllergyLoading(true);
    setIsEditing(true);
    try {
      const res = await userApi.getAllergies();
      setAllergy({ categories: res.allergy_categories as import('../components/AllergySelector').AllergyCategory[], ingredientIds: res.allergy_ingredient_ids });
    } catch {
      setAllergy({ categories: [], ingredientIds: [] });
    } finally {
      setAllergyLoading(false);
    }
  };

  const handleCancel = () => {
    setFormName(profile?.nickname || profile?.user_name || '');
    setFormGender(profile?.gender || '');
    setFormSkinType(profile?.skin_type || '');
    setFormConcerns(profile?.skin_concerns || []);
    setAllergy({ categories: [], ingredientIds: [] });
    setSaveMsg('');
    setIsEditing(false);
  };

  const handleSave = async () => {
    setSaving(true);
    setSaveMsg('');
    try {
      const concernLabels = formConcerns
        .filter((id) => id !== 'none')
        .map((id) => CONCERNS.find((c) => c.id === id)?.label ?? id);

      const allergyItems = buildAllergyItems(allergy);

      const [updated] = await Promise.all([
        userApi.updateProfile({
          nickname: formName,
          gender: formGender || undefined,
          skin_type: formSkinType || undefined,
          skin_concerns: concernLabels.length ? concernLabels : undefined,
        }),
        userApi.updateAllergies({ allergy_items: allergyItems }),
      ]);
      setProfile(updated);
      updateNickname(formName);
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
          {TABS.map(({ key, label, Icon }) => (
            <button key={key} className={`my-tab ${activeTab === key ? 'active' : ''}`} onClick={() => setActiveTab(key)}>
              <Icon size={15} />
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
              <LoadingSpinner text="정보를 불러오는 중이에요" />
            ) : !isEditing ? (
              /* ── 뷰 모드 ── */
              <>
                <div className="my-info-grid">
                  <div className="my-info-row">
                    <span className="my-info-label">닉네임</span>
                    <span className="my-info-value">{profile?.nickname || profile?.user_name || '-'}</span>
                  </div>
                  <div className="my-info-row">
                    <span className="my-info-label">이메일</span>
                    <span className="my-info-value">{profile?.email || '-'}</span>
                  </div>
                  <div className="my-info-row">
                    <span className="my-info-label">성별</span>
                    <span className={`my-info-value ${!profile?.gender ? 'empty' : ''}`}>
                      {GENDER_LABEL[profile?.gender ?? ''] || '미입력'}
                    </span>
                  </div>
                  <div className="my-info-row">
                    <span className="my-info-label">피부 타입</span>
                    <span className={`my-info-value ${!profile?.skin_type ? 'empty' : ''}`}>
                      {profile?.skin_type || '미입력'}
                    </span>
                  </div>
                  <div className="my-info-row full">
                    <span className="my-info-label">피부 고민</span>
                    {profile?.skin_concerns?.filter((c) => c !== 'none').length ? (
                      <div className="my-concern-tags">
                        {profile.skin_concerns
                          .filter((c) => c !== 'none')
                          .map((id) => (
                            <span key={id} className="my-concern-tag">
                              {CONCERNS.find((c) => c.id === id)?.label ?? id}
                            </span>
                          ))}
                      </div>
                    ) : (
                      <span className="my-info-value empty">미입력</span>
                    )}
                  </div>
                </div>
                {saveMsg && <p style={{ fontSize: '0.875rem', color: '#22c55e', marginBottom: 12 }}>{saveMsg}</p>}
                <div className="my-view-btns">
                  <button className="my-btn-primary" onClick={handleEdit}>수정하기</button>
                  <button className="my-btn-ghost" onClick={() => { logout(); navigate('/login'); }}>로그아웃</button>
                </div>
              </>
            ) : (
              /* ── 수정 모드 ── */
              <div className="my-form">
                <div className="my-field">
                  <label className="my-label">닉네임</label>
                  <input className="my-input" value={formName} onChange={(e) => setFormName(e.target.value)} />
                </div>
                <div className="my-field">
                  <label className="my-label">이메일</label>
                  <input className="my-input" value={profile?.email ?? ''} disabled />
                </div>
                <div className="my-field">
                  <label className="my-label">성별</label>
                  <select className="my-input" value={formGender} onChange={(e) => setFormGender(e.target.value)}>
                    <option value="">선택 안함</option>
                    <option value="female">여성</option>
                    <option value="male">남성</option>
                  </select>
                </div>
                <div className="my-field">
                  <label className="my-label">피부 타입</label>
                  <select className="my-input" value={formSkinType} onChange={(e) => setFormSkinType(e.target.value)}>
                    <option value="">선택 안함</option>
                    {SKIN_TYPES.map((t) => <option key={t}>{t}</option>)}
                  </select>
                </div>
                <div className="my-field">
                  <label className="my-label">피부 고민</label>
                  <div className="my-concern-grid">
                    {CONCERNS.map((c) => (
                      <button
                        key={c.id}
                        type="button"
                        className={`my-concern-btn ${formConcerns.includes(c.id) ? 'active' : ''}`}
                        onClick={() => handleConcern(c.id)}
                      >
                        {c.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="my-field">
                  <label className="my-label">알레르기 성분</label>
                  {allergyLoading
                    ? <LoadingSpinner text="알레르기 정보를 불러오는 중이에요" />
                    : <AllergySelector value={allergy} onChange={setAllergy} />
                  }
                </div>
                {saveMsg && <p style={{ fontSize: '0.875rem', color: '#ef4444' }}>{saveMsg}</p>}
                <div className="my-view-btns">
                  <button className="my-save-btn" onClick={handleSave} disabled={saving}>
                    {saving ? '저장 중...' : '수정 완료'}
                  </button>
                  <button className="my-btn-ghost" onClick={handleCancel}>취소</button>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── 찜 목록 ── */}
        {activeTab === 'wishlist' && (
          <div className="my-section">
            <h2 className="my-section-title">찜 목록</h2>
            {loading && <LoadingSpinner text="찜 목록을 불러오는 중이에요" />}
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
