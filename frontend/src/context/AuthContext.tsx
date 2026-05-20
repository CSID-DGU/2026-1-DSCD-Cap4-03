import { createContext } from 'react';

export type AuthType = {
  isLoggedIn: boolean;
  userId: number | null;
  nickname: string | null;
  login: (token: string, userId: number, nickname: string) => void;
  logout: () => void;
};

export const AuthContext = createContext<AuthType | null>(null);
