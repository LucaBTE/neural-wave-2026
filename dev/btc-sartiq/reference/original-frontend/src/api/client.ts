import { SUPABASE_CONFIG_ERROR, supabase } from "../lib/supabase";

export const API_BASE_URL =
  import.meta.env.VITE_API_URL?.trim() || "http://127.0.0.1:8000";

export async function request<TResponse>(
  path: string,
  options: RequestInit = {}
): Promise<TResponse> {
  const isFormData = options.body instanceof FormData;
  const accessToken = supabase
    ? (await supabase.auth.getSession()).data.session?.access_token
    : undefined;
  const headers = new Headers(options.headers ?? {});
  if (!isFormData) {
    headers.set("Content-Type", "application/json");
  }
  if (accessToken) {
    headers.set("Authorization", `Bearer ${accessToken}`);
  }

  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers,
    ...options,
  });

  if (!response.ok) {
    const payload = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${payload}`);
  }

  if (response.status === 204) {
    return null as TResponse;
  }

  const text = await response.text();
  return text ? (JSON.parse(text) as TResponse) : (null as TResponse);
}

export function requireSupabase() {
  if (!supabase) {
    throw new Error(SUPABASE_CONFIG_ERROR);
  }

  return supabase;
}
