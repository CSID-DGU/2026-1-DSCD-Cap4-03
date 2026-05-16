import { api } from './client';

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user_id: number;
  nickname: string;
}

export const authApi = {
  signup: (body: {
    user_name: string;
    nickname: string;
    email: string;
    password: string;
  }) => api.post<AuthResponse>('/auth/signup', { ...body, login_type: 'local' }),

  login: (body: { email: string; password: string }) =>
    api.post<AuthResponse>('/auth/login', body),
};
