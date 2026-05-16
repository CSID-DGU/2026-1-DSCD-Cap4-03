// ─────────────────────────────────────────
//  mockData.ts  –  나중에 API 연결 시 교체
// ─────────────────────────────────────────

export const MOCK_USER = {
  id: 1,
  name: '심고은',
  email: 'sge@example.com',
  skinType: '건성',
  age: 26,
  gender: 'female',
};

// ── API 응답 타입 (실제 서버 응답 구조 그대로)
export interface SkinAnalysisResult {
  result_id: number;
  user_id: number;
  image_id: number;
  model_name: string;
  prompt_version: string;
  generated_at: string;
  summary_comment: string;
  indicator_comments: {
    acne: string;
    dryness: string;
    sagging: string;
    pore: string;
    pigmentation: string;
    wrinkle: string;
  };
  // 프론트 전용 필드 (API 연결 시 별도 엔드포인트 or 같이 내려줄 것)
  imageUrl: string;
  skinType: string;
  rawMetrics: {
    acne: number;       // 0=양호 / 1=보통 / 2,3=개선필요
    dryness: number;    // 0,1=양호 / 2=보통 / 3,4=개선필요
    sagging: number;    // 0,1=양호 / 2,3=보통 / 4,5=개선필요
    pore: number;       // 0,1=양호 / 2,3=보통 / 4,5=개선필요
    pigmentation: number; // 0=양호 / 1,2,3=보통 / 4,5=개선필요
    wrinkle: number;    // 0,1=양호 / 2,3=보통 / 4,5,6=개선필요
  };
  displayScores: {      // 레이더/바 차트용 0~100 점수
    acne: number;
    dryness: number;
    sagging: number;
    pore: number;
    pigmentation: number;
    wrinkle: number;
  };
}

// ── 목업 데이터 (API 응답 구조 그대로 반영)
export const MOCK_SKIN_RESULT: SkinAnalysisResult = {
  result_id: 3,
  user_id: 2,
  image_id: 3,
  model_name: 'gpt-5.4-mini',
  prompt_version: 'skin_v1',
  generated_at: '2026-05-14 15:37:08',
  summary_comment: '전반적으로 큰 부담이 두드러지기보다는 비교적 고르게 관리된 편이며, 모공과 건조 지표가 상대적으로 조금 더 눈에 띄어 피지·모공 관리와 수분 유지에 신경 쓰는 방향이 적합합니다. 반면 처짐, 주름, 트러블, 색소침착 지표는 상대적으로 낮아 현재는 피부 장벽을 지키면서 자극 완화와 예방적 관리 중심으로 꾸준히 이어가는 것이 좋습니다.',
  indicator_comments: {
    acne: '트러블 지표는 비교적 낮은 편이라 과도한 관리보다는 저자극 세안과 진정 케어로 피지 밸런스를 안정적으로 유지하는 방향이 적합합니다.',
    dryness: '건조 지표는 살짝 신경 쓰이는 수준이므로 수분 공급과 보습막 형성을 통해 피부 장벽과 수분 유지를 꾸준히 챙기는 것이 좋습니다.',
    sagging: '처짐 지표는 낮은 편이라 현재는 무리한 리프팅 케어보다는 탄력 유지를 위한 기본적인 보습과 생활 관리에 집중하면 충분합니다.',
    pore: '모공 지표가 상대적으로 더 도드라져 피지 조절과 모공 케어, 각질 정돈을 함께 해주면 피부결 관리에 도움이 됩니다.',
    pigmentation: '색소침착 지표는 낮은 편이지만 피부톤 관리를 위해 자외선 차단과 가벼운 브라이트닝 케어를 꾸준히 이어가는 것이 좋습니다.',
    wrinkle: '주름 지표는 비교적 낮아 잔주름 예방 중심으로 보습과 탄력 관리, 피부 장벽 관리를 차분하게 유지하면 좋습니다.',
  },

  // 프론트 전용 (API 연결 시 교체)
  imageUrl: '/face.jpg',
  skinType: '건성',
  rawMetrics: {
    acne: 1,
    dryness: 2,
    sagging: 1,
    pore: 2,
    pigmentation: 1,
    wrinkle: 1,
  },
  displayScores: {
    acne: 26,
    dryness: 38,
    sagging: 10,
    pore: 41,
    pigmentation: 28,
    wrinkle: 21,
  },
};

// 제품 목업
export interface Product {
  id: string;
  brand: string;
  name: string;
  category: '토너' | '에멀젼' | '앰플' | '크림' | '미스트' | '아이크림' | '멀티밤' | '오일';
  price: number;
  imageUrl: string;
  ingredients: string[];
  pros: string[];
  cons: string[];
  howToUse: string;
  applyTime: string;
  tags: string[];
}
 
export const MOCK_PRODUCTS: Product[] = [
  {
    id: 'p001',
    brand: '제로이드',
    name: '인텐시브 하이드레이팅 앰플',
    category: '앰플',
    price: 18000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/c997e27e-eb16-4234-86be-875067a63fbb/products/zeroid-intensive-hydrating-ampoule/zeroid-intensive-hydrating-ampoule_front_photo_original.jpeg',
    ingredients: ['스네일 세크리션 필트레이트 96%', '나이아신아마이드', '판테놀'],
    pros: ['강력한 보습', '피부 재생 효과', '저자극 성분'],
    cons: ['달팽이 성분 거부감', '점성이 있어 흡수 시간 필요'],
    howToUse: '세안 후 토너 다음 단계에 2~3방울 취해 부드럽게 흡수시켜 주세요.',
    applyTime: '1분',
    tags: ['보습', '재생', '저자극'],
  },
  {
    id: 'p002',
    brand: '이니스프리',
    name: '그린티 씨드 세럼',
    category: '앰플',
    price: 25000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/ec354ce9-b5f8-46de-b212-de96e89a21c7/products/innisfree-green-tea-hyaluronic-skin/innisfree-green-tea-hyaluronic-skin_front_photo_original.jpeg',
    ingredients: ['제주 녹차', '히알루론산', '세라마이드'],
    pros: ['촉촉한 보습감', '산뜻한 제형', '항산화 효과'],
    cons: ['향 민감자 주의', '지성 피부엔 다소 무거울 수 있음'],
    howToUse: '토너 후 적당량을 덜어 얼굴 전체에 펴 발라 주세요.',
    applyTime: '45초',
    tags: ['보습', '항산화', '제주 녹차'],
  },
  {
    id: 'p003',
    brand: '유리아쥬',
    name: '제모스 토너',
    category: '토너',
    price: 29000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/3e56f7a6-e0f7-4b23-b0af-f77cbfe04cc4/products/uriage-xemose-moisturizing-toner/uriage-xemose-moisturizing-toner_front_photo_original.jpeg',
    ingredients: ['블루 히알루론산', '히알루론산 5종', '알로에베라'],
    pros: ['풍부한 수분감', '레이어링 가능', '산뜻한 마무리'],
    cons: ['건성 피부엔 추가 보습 필요'],
    howToUse: '세안 후 손바닥에 적당량을 덜어 얼굴 전체에 흡수시켜 주세요.',
    applyTime: '30초',
    tags: ['수분', '토닝', '히알루론산'],
  },
  {
    id: 'p004',
    brand: 'SK-II',
    name: '페이셜 트리트먼트 에센스',
    category: '토너',
    price: 89000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/858498b8-739e-47c1-aab2-d08aa4baba76/products/sk-ii-facial-treatment-essence/sk-ii-facial-treatment-essence_front_photo_original.jpeg',
    ingredients: ['갈락토미세스 발효 여과물 90%', '나이아신아마이드', '아데노신'],
    pros: ['피부결 개선', '투명감 UP', '검증된 성분'],
    cons: ['높은 가격대', '알코올 성분 포함'],
    howToUse: '세안 후 화장솜 또는 손바닥에 덜어 얼굴 전체에 발라주세요.',
    applyTime: '30초',
    tags: ['미백', '피부결', '발효'],
  },
  {
    id: 'p005',
    brand: '제로이드',
    name: '핌프로브 모이스처라이저',
    category: '크림',
    price: 32000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/990dbbd4-a58d-4a8a-8b3a-b54cce595a37/products/zeroid-pimprove-moisturizer/zeroid-pimprove-moisturizer_front_photo_original.jpeg',
    ingredients: ['아벤느 온천수', '스쿠알렌', '글리세린'],
    pros: ['민감성 피부 적합', '풍부한 보습', '저자극'],
    cons: ['지성 피부엔 다소 무거움', '향료 포함'],
    howToUse: '마지막 단계에 적당량을 취해 얼굴 전체에 펴 발라 주세요.',
    applyTime: '1분',
    tags: ['보습', '민감성', '온천수'],
  },
  {
    id: 'p006',
    brand: '닥터지',
    name: '레드 블레미쉬 클리어 수딩 크림',
    category: '크림',
    price: 22000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/9d0ba13a-a80a-4589-b5c3-8db3f0cc7993/products/dr-g-red-blemish-clear-soothing-cream/dr-g-red-blemish-clear-soothing-cream_front_photo_original.jpeg',
    ingredients: ['병풀 추출물', '티트리', '나이아신아마이드'],
    pros: ['트러블 진정', '가성비 우수', '가벼운 제형'],
    cons: ['건성 피부엔 보습 부족'],
    howToUse: '스킨케어 마지막 단계에 사용해 주세요.',
    applyTime: '45초',
    tags: ['진정', '트러블', '가성비'],
  },
  {
    id: 'p007',
    brand: '코르테',
    name: '더모 에센셜 리치 엠 로션',
    category: '에멀젼',
    price: 27000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/b972d37f-7ee5-49ea-b311-08c6e2109bcc/products/corthe-rich-m-lotion/corthe-rich-m-lotion_front_photo_original.jpeg',
    ingredients: ['세라마이드', '히알루론산', '판테놀'],
    pros: ['장벽 강화', '가벼운 텍스처', '민감성 OK'],
    cons: ['향이 없어 호불호'],
    howToUse: '토너 후 적당량을 덜어 얼굴에 고르게 펴 발라 주세요.',
    applyTime: '30초',
    tags: ['장벽', '저자극', '세라마이드'],
  },
  {
    id: 'p008',
    brand: '헤라',
    name: '블랙 스네일 에멀젼',
    category: '에멀젼',
    price: 45000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/9eb23dbf-9bb1-4ac0-a2ae-345eb6e6e38f/products/hera-age-away-collagenic-emulsion/hera-age-away-collagenic-emulsion_front_photo_original.jpeg',
    ingredients: ['블랙 스네일', '펩타이드', '레티놀'],
    pros: ['안티에이징 효과', '탄력감', '고급스러운 텍스처'],
    cons: ['레티놀 초기 자극 가능', '고가'],
    howToUse: '토너 후 적당량을 얼굴에 부드럽게 펴 발라 주세요.',
    applyTime: '45초',
    tags: ['안티에이징', '탄력', '펩타이드'],
  },
  {
    id: 'p009',
    brand: 'mixsoon',
    name: '하센티 미스트',
    category: '미스트',
    price: 45000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/15323b17-52c2-46bc-a9bf-feeb5c1c6e03/products/mixsoon-h-c-t-mist/mixsoon-h-c-t-mist_front_photo_original.jpeg',
    ingredients: ['히알루론산', '판테놀', '알로에베라'],
    pros: ['즉각 수분 공급', '산뜻한 사용감', '메이크업 위에도 사용 가능'],
    cons: ['지속력 짧음'],
    howToUse: '얼굴에서 20cm 거리에서 고르게 분사해 주세요.',
    applyTime: '20초',
    tags: ['수분', '미스트', '산뜻'],
  },
  {
    id: 'p010',
    brand: '헉슬리',
    name: '오일 라이트 앤 모어',
    category: '오일',
    price: 45000,
    imageUrl: 'https://incidecoder-content.storage.googleapis.com/ae2e51bb-56a3-4a75-8adf-388959436fb9/products/huxley-secret-of-sahara-oil-light-and-more/huxley-secret-of-sahara-oil-light-and-more_front_photo_original.jpeg',
    ingredients: ['사하라 선인장 오일', '호호바 오일', '비타민E'],
    pros: ['가벼운 오일 텍스처', '항산화 효과', '보습 마무리'],
    cons: ['오일 제형 거부감', '지성 피부 주의'],
    howToUse: '스킨케어 마지막 단계에 2~3방울을 손바닥에 덜어 얼굴에 부드럽게 눌러 흡수시켜 주세요.',
    applyTime: '30초',
    tags: ['오일', '항산화', '보습'],
  },
];
 
// 루틴 목업
export interface RoutineProduct {
  productId: string;
  step: number;
  applicationGuide: string;   // LLM 제품별 바르는 법
  timeTag?: 'am' | 'pm';      // 없으면 공용 (am+pm)
}
 
export type RoutineTime = 'am' | 'pm' | 'both';
 
export interface Routine {
  id: string;
  type: 'best' | 'budget';
  label: string;
  totalCost: number;
  duration: number;           // 분
  routineTime: RoutineTime;   // am | pm | both
  aiDescription: string;      // LLM 루틴 전체 설명
  products: RoutineProduct[];
}
 
export const MOCK_ROUTINES: Routine[] = [
  {
    id: 'r001',
    type: 'best',
    label: 'AI BEST 루틴',
    totalCost: 161000,
    duration: 5,
    routineTime: 'both',
    aiDescription: 'Best Routine은 AM+PM 공용 루틴으로 토너, 에멀젼, 앰플, 크림 4단계로 구성되어 보습 강화와 모공 관리에 맞춰 추천되었습니다.',
    products: [
      {
        productId: 'p003',
        step: 1,
        applicationGuide: '세안 후 화장솜이나 손으로 얼굴 전체에 부드럽게 펴 발라주세요. 가볍게 두드려 흡수시켜 주세요. 약 1분',
        // timeTag 없음 = AM+PM 공용
      },
      {
        productId: 'p007',
        step: 2,
        applicationGuide: '토너 흡수 후 적당량을 손에 덜어 얼굴 안쪽에서 바깥쪽으로 부드럽게 펴 발라주세요. 약 2분',
        // timeTag 없음 = AM+PM 공용
      },
      {
        productId: 'p001',
        step: 3,
        applicationGuide: '에멀젼 흡수 후 2~3방울 얼굴에 덜어 가볍게 두드려 흡수시켜 주세요. 특히 건조한 부위에 집중적으로 발라주세요.',
        
      },
      {
        productId: 'p005',
        step: 4,
        applicationGuide: '앰플 흡수 후 완두콩 크기만큼 덜어 얼굴 전체에 부드럽게 펴 발라주세요. T존은 얇게, 건조한 부위는 두껍게 발라주세요.',
        // timeTag 없음 = AM+PM 공용
      },
    ],
  },
  {
    id: 'r002',
    type: 'budget',
    label: '가성비 루틴',
    totalCost: 87000,
    duration: 4,
    routineTime: 'pm',
    aiDescription: '가성비 루틴은 PM 전용 루틴으로 Toner, Ampoules, Emulsions, Cream/Gel 카테고리로 구성되었습니다. 저자극 성분 위주로 구성해 취침 전 집중 보습과 피부 장벽 강화에 초점을 맞췄습니다.',
    products: [
      {
        productId: 'p003',
        step: 1,
        applicationGuide: '저녁 세안 후 화장솜이나 손으로 얼굴 전체에 부드럽게 펴 발라주세요. 목 부위까지 케어해 주세요.',
      },
      {
        productId: 'p002',
        step: 3,
        applicationGuide: '토너 흡수 후 2~3방울을 손에 덜어 얼굴 전체에 가볍게 두드려 흡수시켜 주세요. 건조한 부위에 한 번 더 덧발라 주세요.',
      },
      {
        productId: 'p007',
        step: 2,
        applicationGuide: '앰플 흡수 후 적당량을 손에 덜어 얼굴에 부드럽게 펴 발라주세요. 수면 중 피부 장벽을 강화해 줍니다.',
        timeTag: 'pm' as const,  // 저녁 전용
      },
      {
        productId: 'p006',
        step: 4,
        applicationGuide: '마지막 단계에서 완두콩 크기만큼 덜어 얼굴 전체에 펴 발라주세요. 트러블 부위에 살짝 더 발라 진정시켜 주세요.',
      },
    ],
  },
];
 
// 마이페이지 목업
export const MOCK_PAST_RESULTS = [
  {
    id: 'result-000',
    analyzedAt: '2026-04-01T10:15:00',
    skinType: '건성',
    thumbnail: 'https://placehold.co/80x80/f5f3ff/7c3aed?text=4월',
    aiComment: '건조함이 주요 문제였어요. 보습에 집중하세요.',
  },
  {
    id: 'result-001',
    analyzedAt: '2026-05-11T14:32:00',
    skinType: '복합성',
    thumbnail: 'https://placehold.co/80x80/ede9fe/7c3aed?text=5월',
    aiComment: '수분 보충과 자외선 차단에 집중하면 더욱 빛나는 피부를 유지할 수 있어요.',
  },
];