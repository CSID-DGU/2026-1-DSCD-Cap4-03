import { api } from './client';

export interface RoutineProductItem {
  product_id: number;
  step: number;
  time_tag: 'am' | 'pm' | null;
}

export interface RoutineItem {
  routine_id: string;
  type: 'best' | 'value';
  label: string;
  routine_time: 'am' | 'pm' | 'both';
  total_cost: number;
  duration: number;
  products: RoutineProductItem[];
}

export interface RecommendationResponse {
  session_id: number;
  user_id: number;
  result_id: number;
  session_status: string;
  routines: RoutineItem[];
  budget_check_passed: boolean;
  budget_fallback_applied: boolean;
  budget_message: string | null;
}

export interface SavedRoutineItem {
  saved_routine_id: number;
  session_id: number;
  routine_type: string;
  label: string;
  routine_time: string;
  total_cost: number;
  duration: number;
  saved_at: string;
  products: {
    product_id: number;
    step: number;
    product_name: string;
    brand_name: string;
    category: string;
    price: number;
    image_url: string;
  }[];
}

export const routineApi = {
  recommend: (body: {
    result_id: number;
    image_id: number;
    total_budget: number | null;
    toner_budget: number | null;
    emulsion_budget: number | null;
    ampoule_budget: number | null;
    cream_budget: number | null;
  }) => api.post<RecommendationResponse>('/recommendations', body),

  getRecommendation: (sessionId: number) =>
    api.get<RecommendationResponse>(`/recommendations/${sessionId}`),

  createExplanation: (body: { session_id: number }) =>
    api.post<{
      session_id: number;
      llm_model: string;
      prompt_version: string;
      routines: {
        routine_id: number;
        routine_type: 'best' | 'value';
        routine_rank: number;
        ampm_mode: string;
        recommend_summary: string;
        ampm_comment: string;
        step_guides: {
          slot_order: number;
          category: string;
          usage_guide: string;
        }[];
        strengths: string[];
        cautions: string[];
      }[];
    }>('/recommendation-explanations', body),

  save: (sessionId: number, body: { routine_type: string }) =>
    api.post<{ saved_routine_id: number; saved_at: string }>(`/routines/${sessionId}/save`, body),

  getHistory: () =>
    api.get<{ items: SavedRoutineItem[] }>('/users/me/routines'),
};
