import { api } from './client';

// GET /skin-analysis/{result_id} 응답 (모델 점수 + LLM 코멘트 + 이미지)
export interface AnalysisResult {
  result_id: number;
  user_id: number;
  image_id: number;
  model_name: string;
  prompt_version: string;
  analyzed_at: string;
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
  image_url: string;
  skin_type: string | null;
  raw_metrics: {
    acne: number;
    dryness: number;
    sagging: number;
    pore: number;
    pigmentation: number;
    wrinkle: number;
  };
  display_scores: {
    acne: number;
    dryness: number;
    sagging: number;
    pore: number;
    pigmentation: number;
    wrinkle: number;
  };
}

// GET /users/me/skin-analysis 목록 아이템
export interface SkinHistoryItem {
  result_id: number;
  image_id: number;
  analyzed_at: string;
  skin_type: string | null;
  image_url: string | null;
  ai_comment: string;
}

export const analysisApi = {
  run: (body: { image_id: number }) =>
    api.post<{ result_id: number }>('/skin-analysis', body),

  createSummary: (body: { result_id: number }) =>
    api.post<{ result_id: number }>('/skin-analysis/summaries', body),

  getResult: (resultId: number) =>
    api.get<AnalysisResult>(`/skin-analysis/${resultId}`),

  getHistory: () =>
    api.get<{ items: SkinHistoryItem[] }>('/users/me/skin-analysis'),
};
