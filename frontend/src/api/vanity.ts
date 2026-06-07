import { api } from './client';

/* ── 2. My Products ── */
export interface VanityProduct {
  vanity_id: number;
  product_id: number;
  category: string;
  brand_name: string;
  product_name: string;
  price: number;
  created_at: string;
  image_url?: string; // 프론트 편의용 (백엔드 추가 요청 또는 별도 fetch)
}

/* ── 3. Skin Match ── */
export interface SkinMatchProductResult {
  product_id: number;
  category: string;
  brand_name: string;
  product_name: string;
  vanity_fit_score: number;
  scores: {
    concern_match_score: number;
    skin_type_bonus: number;
    review_score: number;
    irritation_penalty: number;
    vanity_fit_score: number;
  };
  fit_label: 'excellent_match' | 'good_match' | 'so_so' | 'weak_match' | 'poor_match';
  display_label: string;
  recommend_action: 'strong_keep' | 'keep' | 'neutral' | 'caution' | 'replace';
  reason_tags: string[];
  caution_tags: string[];
}

export interface LlmProductComment {
  product_id: number;
  summary: string;
  fit_reason: string;
  caution_comment: string;
  action_comment: string;
}

export interface LlmStepComment {
  slot_order: number;
  product_id: number;
  comment: string;
}

export interface LlmExplanation {
  prompt_version: string;
  generated_at: string;
  skin_match: {
    overall_summary: string;
    product_comments: LlmProductComment[];
  };
  vanity_routine: {
    overall_summary: string;
    step_comments: LlmStepComment[];
    warning_comment: string;
  } | null;
}

export interface SkinMatchResult {
  match_session_id: number;
  user_id: number;
  basis_skin_result: {
    result_id: number;
    analyzed_at: string;
    main_concerns: string[];
  };
  product_match_results: SkinMatchProductResult[];
  llm_explanation: LlmExplanation;
}

export interface SkinMatchLatest {
  match_session_id: number;
  created_at: string;
  basis_skin_result: {
    result_id: number;
    analyzed_at: string;
    main_concerns?: string[];
  };
  summary: {
    excellent_match: number;
    good_match: number;
    so_so: number;
    weak_match: number;
    poor_match: number;
  };
  product_match_results: SkinMatchProductResult[];
  llm_explanation?: LlmExplanation;
}

/* ── 4. Vanity Routine ── */
export interface VanityRoutineProduct {
  slot_order: number;
  category: string;
  product_id: number;
  source: 'vanity' | 'recommendation';
  product_score: number | null;
  brand_name: string;
  product_name: string;
  price: number;
  image_url?: string;
}

export interface VanityRoutineRecommendationResults {
  fixed_products: VanityRoutineProduct[];
  recommended_products: VanityRoutineProduct[];
  final_routine: VanityRoutineProduct[];
  warnings: string[];
  total_price: number;
}

export interface VanityRoutineResult {
  recommendation_session_id: number;
  user_id?: number;
  created_at?: string;
  basis_skin_result: {
    result_id: number;
    analyzed_at: string;
  };
  routine_recommendation_results: VanityRoutineRecommendationResults;
  llm_explanation?: LlmExplanation;
}

export interface VanityRoutineHistoryItem {
  recommendation_session_id: number;
  created_at: string;
  basis_result_id: number;
  fixed_product_count: number;
  total_price: number;
}

/* ── 6. Main Summary ── */
export interface VanitySummary {
  product_summary: {
    total_count: number;
    products: {
      product_id: number;
      brand_name: string;
      product_name: string;
      category: string;
      image_url?: string;
    }[];
  };
  latest_skin_match: {
    match_session_id: number;
    created_at: string;
    summary_text: string;
  } | null;
  latest_vanity_routine: {
    recommendation_session_id: number;
    created_at: string;
    fixed_product_count: number;
    total_price: number;
  } | null;
  basis_skin_result: {
    result_id: number;
    analyzed_at: string;
    message: string;
  } | null;
}

const VANITY_CACHE_TTL_MS = 30_000;
const vanityCache = new Map<string, { expiresAt: number; value: unknown }>();
const vanityPending = new Map<string, Promise<unknown>>();

function getCachedVanity<T>(key: string, fetcher: () => Promise<T>) {
  const now = Date.now();
  const cached = vanityCache.get(key);
  if (cached && cached.expiresAt > now) {
    return Promise.resolve(cached.value as T);
  }

  const pending = vanityPending.get(key);
  if (pending) return pending as Promise<T>;

  const request = fetcher()
    .then((value) => {
      vanityCache.set(key, { expiresAt: Date.now() + VANITY_CACHE_TTL_MS, value });
      return value;
    })
    .finally(() => {
      vanityPending.delete(key);
    });

  vanityPending.set(key, request);
  return request;
}

export function clearVanityCache() {
  vanityCache.clear();
  vanityPending.clear();
}

/* ── API ── */
export const vanityApi = {
  // 2. My Products
  getProducts: () =>
    getCachedVanity('products', () => api.get<{ products: VanityProduct[] }>('/vanity/products')),

  addProduct: (product_id: number) =>
    api.post<{ vanity_id: number; product_id: number; message: string }>('/vanity/products', { product_id })
      .then((result) => {
        clearVanityCache();
        return result;
      }),

  deleteProduct: (product_id: number) =>
    api.delete<{ message: string }>(`/vanity/products/${product_id}`)
      .then((result) => {
        clearVanityCache();
        return result;
      }),

  // 3. Skin Match
  runSkinMatch: (body: { product_ids?: number[] }) =>
    api.post<SkinMatchResult>('/vanity/skin-match', body)
      .then((result) => {
        clearVanityCache();
        return result;
      }),

  getLatestSkinMatch: () =>
    getCachedVanity('latest-skin-match', () => api.get<SkinMatchLatest>('/vanity/skin-match/latest')),

  // 4. Vanity Routine
  runRoutine: (body: {
    fixed_product_ids: number[];
    budget_min?: number; budget_max?: number;
    toner_min?: number; toner_max?: number;
    emulsion_min?: number; emulsion_max?: number;
    ampoule_min?: number; ampoule_max?: number;
    cream_min?: number; cream_max?: number;
  }) =>
    api.post<VanityRoutineResult>('/vanity/routines', body)
      .then((result) => {
        clearVanityCache();
        return result;
      }),

  getLatestRoutine: () =>
    getCachedVanity('latest-routine', () => api.get<VanityRoutineResult>('/vanity/routines/latest')),

  getRoutineHistory: () =>
    getCachedVanity('routine-history', () => api.get<{ routines: VanityRoutineHistoryItem[] }>('/vanity/routines')),

  getRoutineDetail: (recommendation_session_id: number) =>
    getCachedVanity(
      `routine-detail:${recommendation_session_id}`,
      () => api.get<VanityRoutineResult>(`/vanity/routines/${recommendation_session_id}`)
    ),

  // 6. Main Summary
  getSummary: () =>
    getCachedVanity('summary', () => api.get<VanitySummary>('/vanity/summary')),
};

/* ── 표시 라벨 매핑 ── */
export const FIT_LABEL_KO: Record<string, string> = {
  excellent_match: '아주 잘 맞아요',
  good_match:      '괜찮은 편이에요',
  so_so:           '보통이에요',
  weak_match:      '아쉬워요',
  poor_match:      '주의가 필요해요',
};

export const FIT_LABEL_CLASS: Record<string, string> = {
  excellent_match: 'great',
  good_match:      'good',
  so_so:           'soso',
  weak_match:      'bad',
  poor_match:      'bad',
};

export const REASON_TAG_KO: Record<string, string> = {
  concern_match:      '피부 고민 매칭',
  skin_type_match:    '피부 타입 매칭',
  review_match:       '리뷰 긍정',
  irritation_check:   '자극 확인 필요',
  weak_concern_match: '피부 고민 매칭 약함',
};
