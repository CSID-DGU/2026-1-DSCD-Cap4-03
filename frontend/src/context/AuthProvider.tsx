import { useState, useEffect } from 'react';
import { AuthContext } from './AuthContext';
import { userApi } from '../api/user';

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [isLoggedIn, setIsLoggedIn] = useState(
    () => !!localStorage.getItem('access_token')
  );
  const [userId, setUserId] = useState<number | null>(() => {
    const id = localStorage.getItem('user_id');
    return id ? Number(id) : null;
  });
  const [nickname, setNickname] = useState<string | null>(
    () => localStorage.getItem('nickname')
  );

  const login = (token: string, uid: number, nick: string) => {
    localStorage.setItem('access_token', token);
    localStorage.setItem('user_id', String(uid));
    localStorage.setItem('nickname', nick);
    setIsLoggedIn(true);
    setUserId(uid);
    setNickname(nick);
  };

  const logout = () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_id');
    localStorage.removeItem('nickname');
    setIsLoggedIn(false);
    setUserId(null);
    setNickname(null);
  };

  const updateNickname = (nick: string) => {
    localStorage.setItem('nickname', nick);
    setNickname(nick);
  };

  useEffect(() => {
    if (!isLoggedIn) return;
    userApi.getMe().then((profile) => {
      const fresh = profile.nickname || profile.user_name;
      if (fresh) updateNickname(fresh);
    }).catch(() => {});
  }, [isLoggedIn]);

  return (
    <AuthContext.Provider value={{ isLoggedIn, userId, nickname, login, logout, updateNickname }}>
      {children}
    </AuthContext.Provider>
  );
}
