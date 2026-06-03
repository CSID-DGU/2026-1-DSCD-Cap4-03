const BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000';

export function getToken(): string | null {
  return localStorage.getItem('access_token');
}

async function request<T>(method: string, path: string, body?: unknown): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers,
    body: body != null ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `${res.status} 오류`);
  }

  // 204 No Content
  if (res.status === 204) return undefined as T;
  return res.json();
}

async function requestMultipart<T>(path: string, formData: FormData): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const res = await fetch(`${BASE_URL}${path}`, { method: 'POST', headers, body: formData });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail ?? `${res.status} 오류`);
  }
  if (res.status === 204) return undefined as T;
  return res.json();
}

export const api = {
  get:           <T>(path: string)                        => request<T>('GET', path),
  post:          <T>(path: string, body?: unknown)        => request<T>('POST', path, body),
  patch:         <T>(path: string, body: unknown)         => request<T>('PATCH', path, body),
  put:           <T>(path: string, body: unknown)         => request<T>('PUT', path, body),
  delete:        <T>(path: string)                        => request<T>('DELETE', path),
  postMultipart: <T>(path: string, formData: FormData)    => requestMultipart<T>(path, formData),
};
