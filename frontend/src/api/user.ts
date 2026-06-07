import { api } from './client';

export interface UserProfile {
  user_id: number;
  email: string;
  user_name: string;
  nickname: string;
  login_type: string;
  gender: string;
  birth: string;
  skin_type: string;
  skin_concerns: string[];
}

export interface AllergyPayload {
  allergy_items: { category: string; ingredient_id: number }[];
}

export interface AllergyResponse {
  allergy_categories: string[];
  allergy_ingredient_ids: number[];
}

export const userApi = {
  getMe: () => api.get<UserProfile>('/users/me'),

  updateProfile: (body: {
    gender?: string;
    birth?: string;
    skin_type?: string;
    skin_concerns?: string[];
    nickname?: string;
    user_name?: string;
  }) => api.patch<UserProfile>('/users/me/profile', body),

  getAllergies: () => api.get<AllergyResponse>('/users/me/allergies'),

  updateAllergies: (body: AllergyPayload) =>
    api.put<{ user_id: number; saved_count: number }>('/users/me/allergies', body),
};
