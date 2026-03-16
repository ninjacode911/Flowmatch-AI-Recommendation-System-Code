import type { RecommendedItem, Category } from "@/lib/types";
import {
  CATEGORY_COLORS,
  CATEGORY_TEXT_COLORS,
  CATEGORY_ICONS,
} from "@/lib/types";

interface Props {
  item: RecommendedItem;
  rank: number;
}

export default function RecommendationCard({ item, rank }: Props) {
  const cat = item.category as Category;
  const textColor = CATEGORY_TEXT_COLORS[cat] ?? "text-gray-400";
  const bgColor = CATEGORY_COLORS[cat] ?? "bg-gray-500";
  const icon = CATEGORY_ICONS[cat] ?? "📦";
  const scorePercent = Math.min(item.score * 100, 100);

  return (
    <div className="bg-[#1e1b4b]/50 border border-[#312e81]/40 rounded-xl p-5 hover:border-[#8b5cf6]/30 transition-all duration-300 group">
      {/* Top row: rank + category */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs font-mono text-[#a5b4fc]/40 bg-[#312e81]/30 px-2 py-0.5 rounded">
            #{rank}
          </span>
          <span className={`text-xs px-2.5 py-0.5 rounded-full ${bgColor} text-white font-medium`}>
            {icon} {item.category}
          </span>
        </div>
        <span className={`text-sm font-mono font-bold ${textColor}`}>
          {item.score.toFixed(4)}
        </span>
      </div>

      {/* Title */}
      <h4 className="text-sm font-semibold text-white mb-1 group-hover:text-[#a5b4fc] transition-colors">
        {item.title || "Untitled Item"}
      </h4>
      <p className="text-xs text-[#a5b4fc]/40 mb-3 font-mono">{item.item_id}</p>

      {/* Score bar */}
      <div className="mb-3">
        <div className="w-full bg-[#312e81]/30 rounded-full h-1.5 overflow-hidden">
          <div className={`h-full rounded-full score-bar ${bgColor}`} style={{ width: `${scorePercent}%` }} />
        </div>
      </div>

      {/* Explanation */}
      <p className="text-xs text-[#a5b4fc]/60 leading-relaxed">{item.explanation}</p>
    </div>
  );
}
