"use client";

import { useState } from "react";
import { Vortex } from "@/components/ui/vortex";
import { GlowCard } from "@/components/ui/spotlight-card";
import { SplineScene } from "@/components/ui/splite";
import { Spotlight } from "@/components/ui/spotlight";
import { Card } from "@/components/ui/card";
import { getRecommendations } from "@/lib/api";
import type { RecommendedItem, Category } from "@/lib/types";
import { CATEGORY_COLORS, CATEGORY_ICONS } from "@/lib/types";

export default function DashboardPage() {
  const [userId, setUserId] = useState("");
  const [results, setResults] = useState<RecommendedItem[]>([]);
  const [loading, setLoading] = useState(false);

  const handleQuickRecommend = async (uid?: string) => {
    const id = uid || userId || `user_${String(Math.floor(Math.random() * 50000)).padStart(5, "0")}`;
    if (!uid) setUserId(id);
    setLoading(true);
    try {
      const data = await getRecommendations(id, 5);
      setResults(data.items);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl -mt-6 -mx-6">
      {/* Hero Section with Vortex */}
      <div className="w-full h-[28rem] rounded-b-3xl overflow-hidden relative">
        <Vortex
          backgroundColor="#0f172a"
          rangeY={300}
          particleCount={500}
          baseHue={260}
          baseSpeed={0.1}
          rangeSpeed={1.2}
          baseRadius={1}
          rangeRadius={2}
          className="flex items-center flex-col justify-center px-6 md:px-10 py-4 w-full h-full"
        >
          <h2 className="text-white text-3xl md:text-5xl font-bold text-center leading-tight">
            <span className="gradient-text">FlowMatch</span> AI
          </h2>
          <p className="text-[#a5b4fc] text-base md:text-xl max-w-2xl mt-4 text-center leading-relaxed">
            Production-grade recommendation engine powered by
            Two-Tower retrieval, LightGBM ranking, and MMR diversity re-ranking.
          </p>
          <div className="flex flex-col sm:flex-row items-center gap-4 mt-8">
            <a
              href="/explore"
              className="px-6 py-3 bg-gradient-to-r from-[#8b5cf6] to-[#6366f1] hover:from-[#7c3aed] hover:to-[#4f46e5] transition duration-300 rounded-xl text-white font-medium shadow-lg shadow-purple-500/25"
            >
              Try Recommendations
            </a>
            <a
              href="/pipeline"
              className="px-6 py-3 text-[#a5b4fc] hover:text-white transition border border-[#312e81] rounded-xl hover:border-[#8b5cf6]/50"
            >
              View Pipeline
            </a>
          </div>
        </Vortex>
      </div>

      <div className="px-6 space-y-8">
        {/* Stats Row with GlowCards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          {[
            { label: "Users", value: "50K", desc: "Registered users", icon: "👤", color: "blue" as const },
            { label: "Items", value: "50K", desc: "Product catalogue", icon: "📦", color: "purple" as const },
            { label: "Categories", value: "8", desc: "Product types", icon: "🏷️", color: "green" as const },
            { label: "Pipeline", value: "3-Stage", desc: "ML pipeline", icon: "⚡", color: "orange" as const },
          ].map((stat, i) => (
            <GlowCard key={stat.label} glowColor={stat.color} className={`fade-in fade-in-delay-${i + 1}`}>
              <div className="relative z-10 p-2">
                <span className="text-2xl">{stat.icon}</span>
                <p className="text-2xl font-bold text-white mt-2">{stat.value}</p>
                <p className="text-sm text-[#a5b4fc] mt-0.5">{stat.desc}</p>
              </div>
            </GlowCard>
          ))}
        </div>

        {/* Spline 3D Robot + Pipeline Info */}
        <Card className="w-full h-[500px] bg-[#1e1b4b]/60 border-[#312e81]/50 relative overflow-hidden">
          <Spotlight className="-top-40 left-0 md:left-60 md:-top-20" fill="#8b5cf6" />
          <div className="flex h-full">
            {/* Left content */}
            <div className="flex-1 p-8 relative z-10 flex flex-col justify-center">
              <p className="text-[10px] uppercase tracking-[0.2em] text-[#8b5cf6] font-semibold mb-2">
                ML Pipeline Architecture
              </p>
              <h3 className="text-3xl md:text-4xl font-bold bg-clip-text text-transparent bg-gradient-to-b from-white to-[#a5b4fc]">
                Intelligent
                <br />
                Recommendations
              </h3>
              <p className="mt-4 text-[#a5b4fc]/80 max-w-md leading-relaxed">
                3-stage pipeline processes 50K items in ~75ms. Two-Tower neural retrieval narrows to 200 candidates, LightGBM ranks with 32 features, then MMR ensures diversity.
              </p>

              <div className="flex gap-3 mt-6">
                {[
                  { stage: "Retrieval", time: "~15ms", color: "from-blue-500 to-blue-600" },
                  { stage: "Ranking", time: "~55ms", color: "from-purple-500 to-purple-600" },
                  { stage: "Re-Rank", time: "~2ms", color: "from-green-500 to-green-600" },
                ].map((s) => (
                  <div key={s.stage} className="flex items-center gap-2">
                    <div className={`w-2 h-2 rounded-full bg-gradient-to-r ${s.color}`} />
                    <span className="text-xs text-[#a5b4fc]/60">
                      {s.stage} {s.time}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Right: 3D Robot */}
            <div className="flex-1 relative">
              <SplineScene
                scene="https://prod.spline.design/kZDDjO5HuC9GJUM2/scene.splinecode"
                className="w-full h-full"
              />
            </div>
          </div>
        </Card>

        {/* Category Grid */}
        <div>
          <h3 className="text-lg font-bold text-white mb-4">Product Categories</h3>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {(["electronics", "clothing", "food", "beauty", "sports", "home", "toys", "books"] as Category[]).map(
              (cat) => (
                <div
                  key={cat}
                  className="flex items-center gap-3 p-4 rounded-xl bg-[#1e1b4b]/50 border border-[#312e81]/30 hover:border-[#8b5cf6]/30 transition-all duration-300 group"
                >
                  <div
                    className={`w-10 h-10 rounded-lg ${CATEGORY_COLORS[cat]} flex items-center justify-center text-lg`}
                  >
                    {CATEGORY_ICONS[cat]}
                  </div>
                  <div>
                    <p className="text-sm font-medium capitalize text-white group-hover:text-[#a5b4fc] transition-colors">
                      {cat}
                    </p>
                    <p className="text-xs text-[#a5b4fc]/50">~6.2K items</p>
                  </div>
                </div>
              )
            )}
          </div>
        </div>

        {/* Quick Recommend Section */}
        <div className="bg-[#1e1b4b]/40 border border-[#312e81]/40 rounded-2xl p-6">
          <h3 className="text-lg font-bold text-white mb-4">
            Quick Recommend
          </h3>
          <div className="flex gap-2 mb-4">
            <input
              type="text"
              value={userId}
              onChange={(e) => setUserId(e.target.value)}
              placeholder="e.g., user_00042"
              className="flex-1 px-4 py-2.5 bg-[#0f172a] border border-[#312e81]/50 rounded-xl text-sm text-white placeholder:text-[#a5b4fc]/30 focus:outline-none focus:border-[#8b5cf6]/50 transition-colors"
            />
            <button
              type="button"
              onClick={() => {
                const id = `user_${String(Math.floor(Math.random() * 50000)).padStart(5, "0")}`;
                setUserId(id);
                handleQuickRecommend(id);
              }}
              className="px-4 py-2.5 bg-[#312e81]/40 border border-[#312e81] rounded-xl text-sm text-[#a5b4fc] hover:text-white hover:border-[#8b5cf6]/40 transition-colors"
            >
              Random
            </button>
            <button
              onClick={() => handleQuickRecommend()}
              disabled={loading}
              className="px-5 py-2.5 bg-gradient-to-r from-[#8b5cf6] to-[#6366f1] text-white rounded-xl text-sm font-medium hover:from-[#7c3aed] hover:to-[#4f46e5] transition-all disabled:opacity-50 shadow-lg shadow-purple-500/20"
            >
              {loading ? "..." : "Go"}
            </button>
          </div>

          {results.length > 0 && (
            <div className="space-y-2">
              {results.map((item, i) => (
                <div
                  key={item.item_id}
                  className="flex items-center gap-3 p-3 rounded-xl bg-[#0f172a]/60 border border-[#312e81]/20 fade-in"
                  style={{ animationDelay: `${i * 0.08}s` }}
                >
                  <span className="text-xs font-mono text-[#a5b4fc]/40 w-5 text-right">
                    {i + 1}
                  </span>
                  <span
                    className={`text-[10px] px-2 py-0.5 rounded-full ${CATEGORY_COLORS[item.category as Category] ?? "bg-gray-500"} text-white font-medium`}
                  >
                    {item.category}
                  </span>
                  <span className="text-sm text-white flex-1 truncate">
                    {item.title}
                  </span>
                  <span className="text-xs font-mono text-[#8b5cf6]">
                    {item.score.toFixed(3)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
