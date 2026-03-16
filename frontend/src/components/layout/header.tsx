"use client";

import { useEffect, useState } from "react";
import { getHealth } from "@/lib/api";

export default function Header() {
  const [healthy, setHealthy] = useState<boolean | null>(null);

  useEffect(() => {
    const check = async () => {
      try {
        const data = await getHealth();
        setHealthy(data.status === "healthy" || data.status === "ready");
      } catch {
        setHealthy(false);
      }
    };
    check();
    const interval = setInterval(check, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="h-14 border-b border-[#312e81]/30 bg-[#0f172a]/80 backdrop-blur-sm flex items-center justify-between px-6 sticky top-0 z-40">
      <div />
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 text-sm">
          <span
            className={`w-2 h-2 rounded-full ${
              healthy === null
                ? "bg-[#a5b4fc]/40"
                : healthy
                  ? "bg-[#22c55e] pulse-dot"
                  : "bg-[#ef4444] pulse-dot"
            }`}
          />
          <span className="text-[#a5b4fc]/70 text-xs">
            {healthy === null ? "Checking..." : healthy ? "API Connected" : "API Offline"}
          </span>
        </div>
      </div>
    </header>
  );
}
