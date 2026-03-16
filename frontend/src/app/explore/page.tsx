"use client";

import { useState } from "react";
import { getRecommendations } from "@/lib/api";
import type { RecommendationResponse } from "@/lib/types";
import RecommendationCard from "@/components/explore/recommendation-card";

const PRESETS = [
  { label: "Random User", action: "random" },
  { label: "user_00001", action: "user_00001" },
  { label: "user_00042", action: "user_00042" },
  { label: "user_10000", action: "user_10000" },
];

export default function ExplorePage() {
  const [userId, setUserId] = useState("");
  const [topK, setTopK] = useState(10);
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState<RecommendationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState<number | null>(null);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    await fetchRecommendations(userId);
  };

  const fetchRecommendations = async (uid: string) => {
    if (!uid.trim()) return;
    setLoading(true);
    setError("");
    setResponse(null);
    const start = performance.now();
    try {
      const data = await getRecommendations(uid.trim(), topK, query || undefined);
      setElapsed(performance.now() - start);
      setResponse(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setLoading(false);
    }
  };

  const handlePreset = (action: string) => {
    if (action === "random") {
      const id = `user_${String(Math.floor(Math.random() * 50000)).padStart(5, "0")}`;
      setUserId(id);
      fetchRecommendations(id);
    } else {
      setUserId(action);
      fetchRecommendations(action);
    }
  };

  return (
    <div className="space-y-6 max-w-7xl">
      <div>
        <h2 className="text-2xl font-bold text-white">Recommendation Explorer</h2>
        <p className="text-sm text-[#a5b4fc]/70 mt-1">
          Enter a user ID to get personalized recommendations from the ML pipeline
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-[#1e1b4b]/50 border border-[#312e81]/40 rounded-2xl p-6">
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-medium text-[#a5b4fc]/60 mb-1.5">User ID</label>
              <input
                type="text"
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                placeholder="e.g., user_00042"
                className="w-full px-3 py-2.5 bg-[#0f172a] border border-[#312e81]/50 rounded-xl text-sm text-white placeholder:text-[#a5b4fc]/30 focus:outline-none focus:border-[#8b5cf6]/50 transition-colors"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#a5b4fc]/60 mb-1.5">
                Results (top_k): {topK}
              </label>
              <input
                type="range"
                min={1}
                max={20}
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full mt-2 accent-[#8b5cf6]"
              />
            </div>
            <div>
              <label className="block text-xs font-medium text-[#a5b4fc]/60 mb-1.5">Search Query (optional)</label>
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="e.g., wireless headphones"
                className="w-full px-3 py-2.5 bg-[#0f172a] border border-[#312e81]/50 rounded-xl text-sm text-white placeholder:text-[#a5b4fc]/30 focus:outline-none focus:border-[#8b5cf6]/50 transition-colors"
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <button
              type="submit"
              disabled={loading || !userId.trim()}
              className="px-6 py-2.5 bg-gradient-to-r from-[#8b5cf6] to-[#6366f1] text-white rounded-xl text-sm font-medium hover:from-[#7c3aed] hover:to-[#4f46e5] transition-all disabled:opacity-50 disabled:cursor-not-allowed shadow-lg shadow-purple-500/20"
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Loading...
                </span>
              ) : (
                "Get Recommendations"
              )}
            </button>
            <div className="h-6 w-px bg-[#312e81] mx-1" />
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                onClick={() => handlePreset(preset.action)}
                className="px-3 py-2 bg-[#312e81]/30 border border-[#312e81]/50 rounded-xl text-xs text-[#a5b4fc]/70 hover:text-white hover:border-[#8b5cf6]/30 transition-colors"
              >
                {preset.label}
              </button>
            ))}
          </div>
        </form>
      </div>

      {error && (
        <div className="bg-[#ef4444]/10 border border-[#ef4444]/30 rounded-xl p-4">
          <p className="text-sm text-[#ef4444]">{error}</p>
        </div>
      )}

      {response && (
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs px-3 py-1.5 rounded-xl bg-[#8b5cf6]/10 text-[#a5b4fc] border border-[#8b5cf6]/20">
            {response.model_version}
          </span>
          <span className="text-xs px-3 py-1.5 rounded-xl bg-[#312e81]/30 text-[#a5b4fc]/70 border border-[#312e81]/40">
            {response.explanation}
          </span>
          {elapsed !== null && (
            <span className="text-xs px-3 py-1.5 rounded-xl bg-[#22c55e]/10 text-[#22c55e] border border-[#22c55e]/20">
              {elapsed.toFixed(0)}ms round-trip
            </span>
          )}
          <span className="text-xs px-3 py-1.5 rounded-xl bg-[#312e81]/30 text-[#a5b4fc]/70 border border-[#312e81]/40">
            {response.items.length} items
          </span>
        </div>
      )}

      {response && response.items.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {response.items.map((item, i) => (
            <div key={item.item_id} className={`fade-in fade-in-delay-${Math.min(i + 1, 4)}`}>
              <RecommendationCard item={item} rank={i + 1} />
            </div>
          ))}
        </div>
      )}

      {response && response.items.length === 0 && (
        <div className="bg-[#1e1b4b]/50 border border-[#312e81]/40 rounded-xl p-12 text-center">
          <p className="text-[#a5b4fc]/60">No recommendations returned. The user may not exist in the dataset.</p>
        </div>
      )}
    </div>
  );
}
