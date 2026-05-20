import { useState } from 'react';
import { Check, Wind, FlaskConical, Cog, Leaf, type LucideIcon } from 'lucide-react';
import './AllergySelector.css';

// ── 타입
export type AllergyCategory = 'fragrance' | 'preservative' | 'metal' | 'plant_essential_oil';

export interface AllergySelectorValue {
  categories: AllergyCategory[];          // 1차 선택
  ingredientIds: number[];                // 2차 선택된 ingredient_id 목록
}

interface Props {
  value: AllergySelectorValue;
  onChange: (val: AllergySelectorValue) => void;
}

// ── 1차 옵션
const FIRST_LEVEL: { id: string; label: string; Icon: LucideIcon }[] = [
  { id: 'none',               label: '없음',              Icon: Check },
  { id: 'fragrance',          label: '향료/퍼퓸',          Icon: Wind },
  { id: 'preservative',       label: '보존제',             Icon: FlaskConical },
  { id: 'metal',              label: '금속',               Icon: Cog },
  { id: 'plant_essential_oil',label: '식물/에센셜오일 성분', Icon: Leaf },
];

// ── 2차 성분 목록
const SECOND_LEVEL: Record<AllergyCategory, { id: number; name: string }[]> = {
  fragrance: [
    { id: 222,  name: 'Alpha-Isomethyl Ionone' },
    { id: 320,  name: 'Amyl Cinnamal' },
    { id: 322,  name: 'Amylcinnamyl Alcohol' },
    { id: 346,  name: 'Anise Alcohol' },
    { id: 542,  name: 'Benzyl Alcohol' },
    { id: 543,  name: 'Benzyl Benzoate' },
    { id: 544,  name: 'Benzyl Cinnamate' },
    { id: 547,  name: 'Benzyl Salicylate' },
    { id: 696,  name: 'Butylphenyl Methylpropional' },
    { id: 1228, name: 'Cinnamal' },
    { id: 1242, name: 'Cinnamyl Alcohol' },
    { id: 1253, name: 'Citral' },
    { id: 1258, name: 'Citronellol' },
    { id: 1472, name: 'Coumarin' },
    { id: 1566, name: 'D-Limonene' },
    { id: 2032, name: 'Eugenol' },
    { id: 2040, name: 'Evernia Furfuracea Extract' },
    { id: 2041, name: 'Evernia Prunastri Extract' },
    { id: 2042, name: 'Farnesol' },
    { id: 2133, name: 'Geraniol' },
    { id: 2342, name: 'Hexyl Cinnamal' },
    { id: 2549, name: 'Hydroxycitronellal' },
    { id: 2562, name: 'Hydroxyisohexyl 3-Cyclohexene Carboxaldehyde' },
    { id: 2664, name: 'Isoeugenol' },
    { id: 2959, name: 'Limonene' },
    { id: 2963, name: 'Linalool' },
    { id: 3162, name: 'Methyl 2-Octynoate' },
  ],
  preservative: [
    { id: 12,   name: '2-Bromo-2-Nitropropane-1,3-Diol' },
    { id: 532,  name: 'Benzalkonium Chloride' },
    { id: 548,  name: 'Benzylparaben' },
    { id: 695,  name: 'Butylparaben' },
    { id: 1636, name: 'Diazolidinyl Urea' },
    { id: 1568, name: 'DMDM Hydantoin' },
    { id: 2006, name: 'Ethylparaben' },
    { id: 2608, name: 'Imidazolidinyl Urea' },
    { id: 2617, name: 'Iodopropynyl Butylcarbamate' },
    { id: 2644, name: 'Isobutylparaben' },
    { id: 2695, name: 'Isopropylparaben' },
    { id: 3208, name: 'Methylchloroisothiazolinone' },
    { id: 3209, name: 'Methyldibromo Glutaronitrile' },
    { id: 3217, name: 'Methylisothiazolinone' },
    { id: 3219, name: 'Methylparaben' },
    { id: 3941, name: 'Parabens' },
    { id: 4053, name: 'Phenoxyethanol' },
    { id: 4502, name: 'Propylparaben' },
    { id: 4616, name: 'Quaternium-15' },
    { id: 5121, name: 'Sodium Ethylparaben' },
    { id: 5206, name: 'Sodium Methylparaben' },
    { id: 5250, name: 'Sodium Propylparaben' },
  ],
  metal: [
    { id: 1424, name: 'Colloidal Gold' },
    { id: 2259, name: 'Gold' },
  ],
  plant_essential_oil: [
    { id: 347,  name: 'Anthemis Nobilis Flower Extract' },
    { id: 348,  name: 'Anthemis Nobilis Flower Oil' },
    { id: 398,  name: 'Arnica Montana Flower Extract' },
    { id: 399,  name: 'Arnica Montana Flower Oil' },
    { id: 837,  name: 'Calendula Officinalis Extract' },
    { id: 838,  name: 'Calendula Officinalis Flower Extract' },
    { id: 839,  name: 'Calendula Officinalis Flower Oil' },
    { id: 873,  name: 'Cananga Odorata Flower Oil' },
    { id: 875,  name: 'Cananga Odorata Leaf Oil' },
    { id: 1099, name: 'Chamomilla Recutita Flower Extract' },
    { id: 1100, name: 'Chamomilla Recutita Flower Oil' },
    { id: 1101, name: 'Chamomilla Recutita Flower/Leaf Extract' },
    { id: 1276, name: 'Citrus Aurantium Dulcis Oil' },
    { id: 1279, name: 'Citrus Aurantium Dulcis Peel Oil' },
    { id: 1280, name: 'Citrus Aurantium Dulcis Peel Oil Expressed' },
    { id: 1286, name: 'Citrus Aurantium Peel Oil Expressed' },
    { id: 1344, name: 'Citrus Sinensis Peel Oil Expressed' },
    { id: 1542, name: 'Cymbopogon Citratus Leaf Oil' },
    { id: 1546, name: 'Cymbopogon Flexuosus Herb Oil' },
    { id: 1548, name: 'Cymbopogon Martini Herb Oil' },
    { id: 1549, name: 'Cymbopogon Martini Motia Herb Oil' },
    { id: 1551, name: 'Cymbopogon Martini Sofia Herb Oil' },
    { id: 1555, name: 'Cymbopogon Winterianus Herb Oil' },
    { id: 2025, name: 'Eugenia Caryophyllus Bud Oil' },
    { id: 2028, name: 'Eugenia Caryophyllus Leaf Oil' },
    { id: 2029, name: 'Eugenia Caryophyllus Stem Oil' },
    { id: 2725, name: 'Jasminum Officinale Flower Oil' },
    { id: 2728, name: 'Jasminum Officinale Oil' },
    { id: 2920, name: 'Lavandula Angustifolia Flower/Leaf/Stem Oil' },
    { id: 2924, name: 'Lavandula Angustifolia Oil' },
    { id: 2925, name: 'Lavandula Hybrida Abrial Herb Oil' },
    { id: 2927, name: 'Lavandula Hybrida Grosso Herb Oil' },
    { id: 2928, name: 'Lavandula Hybrida Herb Oil' },
    { id: 2929, name: 'Lavandula Hybrida Oil' },
    { id: 2931, name: 'Lavandula Intermedia Flower/Leaf/Stem Oil' },
    { id: 2933, name: 'Lavandula Latifolia Herb Oil' },
    { id: 2934, name: 'Lavandula Officinalis Flower Oil' },
    { id: 2935, name: 'Lawsonia Inermis Extract' },
    { id: 3101, name: 'Melaleuca Alternifolia Leaf Oil' },
    { id: 3124, name: 'Mentha Piperita Extract' },
    { id: 3125, name: 'Mentha Piperita Flower/Leaf/Stem Extract' },
    { id: 3127, name: 'Mentha Piperita Herb Extract' },
    { id: 3128, name: 'Mentha Piperita Oil' },
    { id: 3308, name: 'Myroxylon Balsamum Pereirae Balsam Extract' },
    { id: 3309, name: 'Myroxylon Balsamum Pereirae Balsam Oil' },
    { id: 3322, name: 'Narcissus Poeticus Flower Extract' },
    { id: 4476, name: 'Propolis Wax' },
    { id: 4877, name: 'Santalum Album Wood Oil' },
    { id: 4878, name: 'Santalum Austrocaledonicum Wood Oil' },
    { id: 4880, name: 'Santalum Spicata Wood Oil' },
  ],
};

export function buildAllergyItems(value: AllergySelectorValue): { category: string; ingredient_id: number }[] {
  const result: { category: string; ingredient_id: number }[] = [];
  for (const cat of value.categories) {
    const catIds = new Set(SECOND_LEVEL[cat].map((i) => i.id));
    for (const ingId of value.ingredientIds) {
      if (catIds.has(ingId)) result.push({ category: cat, ingredient_id: ingId });
    }
  }
  return result;
}

const CATEGORY_LABELS: Record<AllergyCategory, string> = {
  fragrance:          '향료/퍼퓸 성분 선택',
  preservative:       '보존제 성분 선택',
  metal:              '금속 성분 선택',
  plant_essential_oil:'식물/에센셜오일 성분 선택',
};

export default function AllergySelector({ value, onChange }: Props) {
  const [firstChoice, setFirstChoice] = useState<string | null>(null);

  const handleFirst = (id: string) => {
    setFirstChoice(id);
    if (id === 'none' || id === 'unknown') {
      onChange({ categories: [], ingredientIds: [] });
    } else {
      const cat = id as AllergyCategory;
      const already = value.categories.includes(cat);
      const newCats = already
        ? value.categories.filter((c) => c !== cat)
        : [...value.categories, cat];
      // 해당 카테고리 해제 시 그 성분들도 제거
      const removedIds = already
        ? SECOND_LEVEL[cat].map((i) => i.id)
        : [];
      onChange({
        categories: newCats,
        ingredientIds: value.ingredientIds.filter((id) => !removedIds.includes(id)),
      });
    }
  };

  const handleIngredient = (cat: AllergyCategory, ingId: number) => {
    const has = value.ingredientIds.includes(ingId);
    onChange({
      ...value,
      ingredientIds: has
        ? value.ingredientIds.filter((i) => i !== ingId)
        : [...value.ingredientIds, ingId],
    });
  };

  const selectAll = (cat: AllergyCategory) => {
    const ids = SECOND_LEVEL[cat].map((i) => i.id);
    const allSelected = ids.every((id) => value.ingredientIds.includes(id));
    onChange({
      ...value,
      ingredientIds: allSelected
        ? value.ingredientIds.filter((id) => !ids.includes(id))
        : [...new Set([...value.ingredientIds, ...ids])],
    });
  };

  const isNoneOrUnknown = firstChoice === 'none' || firstChoice === 'unknown';
  const activeCats = value.categories;

  return (
    <div className="allergy-selector">

      {/* ── 1차 선택 ── */}
      <div className="allergy-first-grid">
        {FIRST_LEVEL.map((opt) => {
          const isActive =
            opt.id === 'none' || opt.id === 'unknown'
              ? firstChoice === opt.id
              : activeCats.includes(opt.id as AllergyCategory);
          return (
            <button
              key={opt.id}
              type="button"
              className={`allergy-first-btn ${isActive ? 'active' : ''}`}
              onClick={() => handleFirst(opt.id)}
            >
              <opt.Icon size={16} className="allergy-first-icon" />
              <span>{opt.label}</span>
            </button>
          );
        })}
      </div>

      {/* ── 2차 선택 (카테고리별 펼침) ── */}
      {!isNoneOrUnknown && activeCats.length > 0 && (
        <div className="allergy-second-wrap">
          {activeCats.map((cat) => {
            const ingredients = SECOND_LEVEL[cat];
            const selectedCount = ingredients.filter((i) =>
              value.ingredientIds.includes(i.id)
            ).length;
            const allSelected = selectedCount === ingredients.length;

            return (
              <div className="allergy-second-block" key={cat}>
                <div className="allergy-second-header">
                  <span className="allergy-second-title">{CATEGORY_LABELS[cat]}</span>
                  <div className="allergy-second-meta">
                    <span className="allergy-second-count">
                      {selectedCount}/{ingredients.length}개 선택
                    </span>
                    <button
                      type="button"
                      className="allergy-select-all"
                      onClick={() => selectAll(cat)}
                    >
                      {allSelected ? '전체 해제' : '전체 선택'}
                    </button>
                  </div>
                </div>
                <div className="allergy-ingredient-grid">
                  {ingredients.map((ing) => {
                    const checked = value.ingredientIds.includes(ing.id);
                    return (
                      <label key={ing.id} className={`allergy-ingredient ${checked ? 'checked' : ''}`}>
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => handleIngredient(cat, ing.id)}
                        />
                        <span>{ing.name}</span>
                      </label>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* 선택 없을 때 안내 */}
      {!firstChoice && (
        <p className="allergy-hint">알레르기가 없거나 모르시면 위에서 선택해주세요</p>
      )}
    </div>
  );
}
