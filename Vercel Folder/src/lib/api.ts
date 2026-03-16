import type { RecommendationResponse, HealthResponse, EventStats } from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

export async function getRecommendations(
  userId: string,
  topK: number = 10,
  query?: string
): Promise<RecommendationResponse> {
  const res = await fetch(`${BASE}/api/recommend`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ user_id: userId, top_k: topK, query }),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || `Request failed: ${res.status}`);
  }
  return res.json();
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BASE}/api/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json();
}

export async function getEventStats(): Promise<EventStats> {
  const res = await fetch(`${BASE}/api/events`);
  if (!res.ok) throw new Error(`Event stats failed: ${res.status}`);
  return res.json();
}
