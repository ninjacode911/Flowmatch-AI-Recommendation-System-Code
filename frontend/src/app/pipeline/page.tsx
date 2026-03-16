"use client";

import { useState } from "react";
import { getRecommendations } from "@/lib/api";
import type { RecommendedItem } from "@/lib/types";

const STAGES = [
  {
    id: "retrieval",
    title: "Stage 1: Retrieval",
    subtitle: "Two-Tower Model",
    color: "blue",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
      </svg>
    ),
    description:
      "The User Tower encodes the user's features (ID, cluster, age, gender) into a 256-dimensional embedding vector. This vector is used to search against 50K pre-computed item embeddings in Qdrant (ANN search) to retrieve the top 200 most similar candidates.",
    details: [
      "User Tower: user_id_emb(64) + cluster_emb(16) + features(4) → 256-d vector",
      "Item Tower: pre-computed 50K item embeddings (256-d each)",
      "ANN Search: Cosine similarity in Qdrant vector database",
      "Output: 200 candidate items with similarity scores",
    ],
    metrics: { latency: "~15ms", input: "1 user", output: "200 candidates" },
  },
  {
    id: "ranking",
    title: "Stage 2: Ranking",
    subtitle: "LightGBM LTR",
    color: "purple",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12" />
      </svg>
    ),
    description:
      "For each of the 200 candidates, the Feature Engineer computes 32 features capturing user preferences, item properties, and cross-features. LightGBM (trained with LambdaRank) scores each candidate and produces a relevance ranking.",
    details: [
      "32 features: user stats, item stats, cross-features, category preferences",
      "LambdaRank objective: directly optimizes NDCG",
      "Relevance grades: purchase=3, cart=2, click=1, view=0",
      "Output: 200 candidates with LTR scores, sorted by relevance",
    ],
    metrics: { latency: "~55ms", input: "200 candidates", output: "200 scored" },
  },
  {
    id: "reranking",
    title: "Stage 3: Re-Ranking",
    subtitle: "MMR Diversity",
    color: "green",
    icon: (
      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
      </svg>
    ),
    description:
      "Maximal Marginal Relevance (MMR) balances relevance vs. diversity. It iteratively selects items that are both relevant AND different from what's already been selected. Business rules enforce category limits and promoted items.",
    details: [
      "MMR lambda=0.7 (70% relevance, 30% diversity)",
      "Freshness boost for trending items (weight=0.05)",
      "Max 3 items from same category",
      "Output: Final top-K diverse, relevant recommendations",
    ],
    metrics: { latency: "~2ms", input: "200 scored", output: "Top-K results" },
  },
];

const COLOR_MAP: Record<string, { bg: string; text: string; border: string; glow: string }> = {
  blue: { bg: "bg-blue-500/10", text: "text-blue-400", border: "border-blue-500/20", glow: "shadow-blue-500/10" },
  purple: { bg: "bg-purple-500/10", text: "text-purple-400", border: "border-purple-500/20", glow: "shadow-purple-500/10" },
  green: { bg: "bg-green-500/10", text: "text-green-400", border: "border-green-500/20", glow: "shadow-green-500/10" },
};

export default function PipelinePage() {
  const [activeStage, setActiveStage] = useState<number | null>(null);
  const [demoUserId, setDemoUserId] = useState("");
  const [demoResults, setDemoResults] = useState<RecommendedItem[]>([]);
  const [demoLoading, setDemoLoading] = useState(false);
  const [demoStage, setDemoStage] = useState(-1);

  const runDemo = async () => {
    const uid = demoUserId.trim() || `user_${String(Math.floor(Math.random() * 50000)).padStart(5, "0")}`;
    setDemoUserId(uid);
    setDemoLoading(true);
    setDemoResults([]);

    // Animate through stages
    setDemoStage(0);
    await sleep(600);
    setDemoStage(1);
    await sleep(600);
    setDemoStage(2);
    await sleep(400);

    try {
      const data = await getRecommendations(uid, 5);
      setDemoResults(data.items);
      setDemoStage(3);
    } catch {
      setDemoStage(-1);
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className="space-y-8 max-w-7xl">
      {/* Header */}
      <div>
        <h2 className="text-2xl font-bold text-foreground">
          Pipeline Architecture
        </h2>
        <p className="text-sm text-muted mt-1">
          3-stage ML pipeline: Retrieval &rarr; Ranking &rarr; Re-Ranking
        </p>
      </div>

      {/* Pipeline Flow Diagram */}
      <div className="bg-card border border-border rounded-xl p-8">
        <div className="flex flex-col lg:flex-row items-stretch gap-4">
          {STAGES.map((stage, i) => {
            const colors = COLOR_MAP[stage.color];
            const isActive = activeStage === i || demoStage === i;
            return (
              <div key={stage.id} className="flex items-center gap-4 flex-1">
                <button
                  onClick={() => setActiveStage(activeStage === i ? null : i)}
                  className={`flex-1 p-5 rounded-xl border transition-all duration-300 cursor-pointer text-left ${
                    isActive
                      ? `${colors.bg} ${colors.border} shadow-lg ${colors.glow}`
                      : "bg-card-hover border-border hover:border-accent/20"
                  }`}
                >
                  <div className={`${colors.text} mb-3`}>{stage.icon}</div>
                  <h3 className={`text-sm font-bold ${isActive ? colors.text : "text-foreground"}`}>
                    {stage.title}
                  </h3>
                  <p className="text-xs text-muted mt-0.5">{stage.subtitle}</p>
                  <div className="flex gap-3 mt-3">
                    <span className="text-[10px] text-muted">
                      {stage.metrics.latency}
                    </span>
                    <span className="text-[10px] text-muted">
                      {stage.metrics.input} &rarr; {stage.metrics.output}
                    </span>
                  </div>
                </button>
                {i < STAGES.length - 1 && (
                  <div className="hidden lg:flex items-center">
                    <div className="relative w-8">
                      <svg className="w-8 h-8 text-muted" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                      {demoStage > i && (
                        <div className="absolute inset-0 flex items-center justify-center">
                          <div className="w-2 h-2 rounded-full bg-accent pipeline-flow" />
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Expanded stage detail */}
      {activeStage !== null && (
        <div className={`bg-card border rounded-xl p-6 fade-in ${COLOR_MAP[STAGES[activeStage].color].border}`}>
          <h3 className={`text-lg font-bold ${COLOR_MAP[STAGES[activeStage].color].text} mb-2`}>
            {STAGES[activeStage].title}: {STAGES[activeStage].subtitle}
          </h3>
          <p className="text-sm text-foreground/80 mb-4">
            {STAGES[activeStage].description}
          </p>
          <ul className="space-y-2">
            {STAGES[activeStage].details.map((detail, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-muted">
                <span className={`mt-1 w-1.5 h-1.5 rounded-full shrink-0 ${COLOR_MAP[STAGES[activeStage].color].text.replace("text-", "bg-")}`} />
                {detail}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Interactive Demo */}
      <div className="bg-card border border-border rounded-xl p-6">
        <h3 className="text-sm font-semibold text-foreground mb-4">
          Interactive Demo — Trace a Request Through the Pipeline
        </h3>
        <div className="flex gap-2 mb-4">
          <input
            type="text"
            value={demoUserId}
            onChange={(e) => setDemoUserId(e.target.value)}
            placeholder="Enter user_id or leave blank for random"
            className="flex-1 px-3 py-2.5 bg-background border border-border rounded-lg text-sm text-foreground placeholder:text-muted/50 focus:outline-none focus:border-accent"
          />
          <button
            onClick={runDemo}
            disabled={demoLoading}
            className="px-5 py-2.5 bg-accent text-white rounded-lg text-sm font-medium hover:bg-accent-light transition-colors disabled:opacity-50"
          >
            {demoLoading ? "Running..." : "Trace Pipeline"}
          </button>
        </div>

        {/* Demo progress */}
        {demoStage >= 0 && (
          <div className="flex items-center gap-2 mb-4">
            {STAGES.map((stage, i) => (
              <div key={stage.id} className="flex items-center gap-2">
                <div
                  className={`w-3 h-3 rounded-full transition-all duration-300 ${
                    demoStage > i
                      ? "bg-success"
                      : demoStage === i
                        ? "bg-accent pulse-dot"
                        : "bg-border"
                  }`}
                />
                <span
                  className={`text-xs ${demoStage >= i ? "text-foreground" : "text-muted"}`}
                >
                  {stage.subtitle}
                </span>
                {i < STAGES.length - 1 && (
                  <div className={`w-8 h-px ${demoStage > i ? "bg-success" : "bg-border"}`} />
                )}
              </div>
            ))}
            {demoStage === 3 && (
              <>
                <div className="w-8 h-px bg-success" />
                <div className="w-3 h-3 rounded-full bg-success" />
                <span className="text-xs text-success font-medium">Done!</span>
              </>
            )}
          </div>
        )}

        {/* Demo results */}
        {demoResults.length > 0 && (
          <div className="space-y-2">
            {demoResults.map((item, i) => (
              <div
                key={item.item_id}
                className="flex items-center gap-3 p-3 rounded-lg bg-card-hover fade-in"
                style={{ animationDelay: `${i * 0.1}s` }}
              >
                <span className="text-xs font-mono text-muted w-5 text-right">
                  {i + 1}
                </span>
                <span className="text-xs px-2 py-0.5 rounded bg-accent/20 text-accent-light">
                  {item.category}
                </span>
                <span className="text-sm text-foreground flex-1 truncate">
                  {item.title}
                </span>
                <span className="text-xs font-mono text-muted">
                  {item.score.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Model Details */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="bg-card border border-border rounded-xl p-5">
          <h4 className="text-sm font-semibold text-blue-400 mb-3">Two-Tower Model</h4>
          <div className="space-y-2 text-xs text-muted">
            <div className="flex justify-between"><span>Parameters</span><span className="text-foreground">6.5M</span></div>
            <div className="flex justify-between"><span>Embedding dim</span><span className="text-foreground">256-d</span></div>
            <div className="flex justify-between"><span>Loss</span><span className="text-foreground">Sampled softmax</span></div>
            <div className="flex justify-between"><span>Training</span><span className="text-foreground">GPU + AMP</span></div>
            <div className="flex justify-between"><span>Best epoch</span><span className="text-foreground">5 / 17</span></div>
          </div>
        </div>
        <div className="bg-card border border-border rounded-xl p-5">
          <h4 className="text-sm font-semibold text-purple-400 mb-3">LightGBM LTR</h4>
          <div className="space-y-2 text-xs text-muted">
            <div className="flex justify-between"><span>Objective</span><span className="text-foreground">LambdaRank</span></div>
            <div className="flex justify-between"><span>Features</span><span className="text-foreground">32</span></div>
            <div className="flex justify-between"><span>Val NDCG@10</span><span className="text-foreground">0.4169</span></div>
            <div className="flex justify-between"><span>Train users</span><span className="text-foreground">5,000</span></div>
            <div className="flex justify-between"><span>Candidates/user</span><span className="text-foreground">100</span></div>
          </div>
        </div>
        <div className="bg-card border border-border rounded-xl p-5">
          <h4 className="text-sm font-semibold text-green-400 mb-3">MMR Reranker</h4>
          <div className="space-y-2 text-xs text-muted">
            <div className="flex justify-between"><span>Lambda</span><span className="text-foreground">0.7</span></div>
            <div className="flex justify-between"><span>Freshness weight</span><span className="text-foreground">0.05</span></div>
            <div className="flex justify-between"><span>Max same category</span><span className="text-foreground">3</span></div>
            <div className="flex justify-between"><span>Similarity</span><span className="text-foreground">Cosine</span></div>
            <div className="flex justify-between"><span>Fallback</span><span className="text-foreground">Category-diverse</span></div>
          </div>
        </div>
      </div>
    </div>
  );
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
