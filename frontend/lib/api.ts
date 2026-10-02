const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8080";

export type AuthResponse = { token: string; email: string; displayName: string };

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.message ?? body.error ?? `Request failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => request<{ service: string; status: string }>("/api/health"),
  pingAi: () => request<Record<string, unknown>>("/api/ping-ai"),
  register: (email: string, password: string, displayName: string) =>
    request<AuthResponse>("/api/auth/register", {
      method: "POST",
      body: JSON.stringify({ email, password, displayName }),
    }),
  login: (email: string, password: string) =>
    request<AuthResponse>("/api/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    }),
};
