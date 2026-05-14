import { createContext } from 'react';

export type AuthType = {
  isLoggedIn: boolean;
  login: () => void;
  logout: () => void;
};

export const AuthContext = createContext<AuthType | null>(null);