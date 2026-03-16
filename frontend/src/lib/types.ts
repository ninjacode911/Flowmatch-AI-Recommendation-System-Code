export interface SessionEvent {
  item_id: string;
  event_type: "click" | "view" | "add_to_cart" | "purchase";
  timestamp?: number;
}

export interface RecommendationRequest {
  user_id: string;
  top_k: number;
  query?: string;
  session_events?: SessionEvent[];
}

export interface RecommendedItem {
  item_id: string;
  score: number;
  title: string;
  category: string;
  explanation: string;
}

export interface RecommendationResponse {
  user_id: string;
  items: RecommendedItem[];
  model_version: string;
  explanation: string;
}

export interface HealthResponse {
  status: string;
  service?: string;
  pipeline_loaded?: boolean;
}

export interface EventStats {
  events_sent: number;
  events_failed: number;
  buffer_size: number;
  mode: string;
}

export const CATEGORIES = [
  "electronics",
  "clothing",
  "food",
  "beauty",
  "sports",
  "home",
  "toys",
  "books",
] as const;

export type Category = (typeof CATEGORIES)[number];

export const CATEGORY_COLORS: Record<Category, string> = {
  electronics: "bg-blue-500",
  clothing: "bg-purple-500",
  food: "bg-green-500",
  beauty: "bg-pink-500",
  sports: "bg-orange-500",
  home: "bg-amber-500",
  toys: "bg-red-500",
  books: "bg-indigo-500",
};

export const CATEGORY_TEXT_COLORS: Record<Category, string> = {
  electronics: "text-blue-400",
  clothing: "text-purple-400",
  food: "text-green-400",
  beauty: "text-pink-400",
  sports: "text-orange-400",
  home: "text-amber-400",
  toys: "text-red-400",
  books: "text-indigo-400",
};

export const CATEGORY_BORDER_COLORS: Record<Category, string> = {
  electronics: "border-blue-500/30",
  clothing: "border-purple-500/30",
  food: "border-green-500/30",
  beauty: "border-pink-500/30",
  sports: "border-orange-500/30",
  home: "border-amber-500/30",
  toys: "border-red-500/30",
  books: "border-indigo-500/30",
};

export const CATEGORY_ICONS: Record<Category, string> = {
  electronics: "💻",
  clothing: "👕",
  food: "🍕",
  beauty: "💄",
  sports: "⚽",
  home: "🏠",
  toys: "🧸",
  books: "📚",
};
