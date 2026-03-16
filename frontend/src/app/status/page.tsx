"use client";

import { useEffect, useState, useCallback } from "react";
import { getHealth, getEventStats } from "@/lib/api";
import type { HealthResponse, EventStats } from "@/lib/types";

interface ServiceStatus {
  name: string;
  port: number;
  status: "healthy" | "unhealthy" | "checking";
  description: string;
  type: "app" | "infra";
}

const SERVICES: ServiceStatus[] = [
  { name: "API Gateway", port: 8000, status: "checking", description: "Main entry point, routes requests to pipeline", type: "app" },
  { name: "Event Collector", port: 8001, status: "checking", description: "Kafka event ingestion for user actions", type: "app" },
  { name: "User Feature Service", port: 8002, status: "checking", description: "Redis-backed feature store", type: "app" },
  { name: "Ranking Service", port: 8003, status: "checking", description: "LightGBM LTR model serving", type: "app" },
  { name: "LLM Augment Service", port: 8004, status: "checking", description: "Natural language explanations & query parsing", type: "app" },
];

const INFRA = [
  { name: "PostgreSQL", port: 5432, description: "User and item metadata storage", icon: "🐘" },
  { name: "Redis", port: 6379, description: "Feature store, caching, sessions", icon: "🔴" },
  { name: "Qdrant", port: 6333, description: "Vector database for ANN search", icon: "🔷" },
  { name: "Prometheus", port: 9090, description: "Metrics collection & alerting", icon: "📊" },
  { name: "Grafana", port: 3000, description: "Dashboards & visualization", icon: "📈" },
];

export default function StatusPage() {
  const [services, setServices] = useState(SERVICES);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [eventStats, setEventStats] = useState<EventStats | null>(null);
  const [lastChecked, setLastChecked] = useState<string>("");

  const checkHealth = useCallback(async () => {
    // Check API Gateway
    try {
      const data = await getHealth();
      setHealth(data);
      setServices((prev) =>
        prev.map((s) =>
          s.name === "API Gateway"
            ? { ...s, status: "healthy" as const }
            : s
        )
      );
    } catch {
      setServices((prev) =>
        prev.map((s) =>
          s.name === "API Gateway"
            ? { ...s, status: "unhealthy" as const }
            : s
        )
      );
    }

    // Check Event Collector
    try {
      const stats = await getEventStats();
      setEventStats(stats);
      setServices((prev) =>
        prev.map((s) =>
          s.name === "Event Collector"
            ? { ...s, status: "healthy" as const }
            : s
        )
      );
    } catch {
      setServices((prev) =>
        prev.map((s) =>
          s.name === "Event Collector"
            ? { ...s, status: "unhealthy" as const }
            : s
        )
      );
    }

    // Other services -- we can only check via their respective ports
    // For now, mark them based on whether the gateway is up
    setServices((prev) =>
      prev.map((s) =>
        s.name !== "API Gateway" && s.name !== "Event Collector"
          ? { ...s, status: "checking" as const }
          : s
      )
    );

    setLastChecked(new Date().toLocaleTimeString());
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 15000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  return (
    <div className="space-y-6 max-w-7xl">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-foreground">System Status</h2>
          <p className="text-sm text-muted mt-1">
            Service health and infrastructure monitoring
          </p>
        </div>
        <div className="flex items-center gap-3">
          {lastChecked && (
            <span className="text-xs text-muted">
              Last checked: {lastChecked}
            </span>
          )}
          <button
            onClick={checkHealth}
            className="px-3 py-2 bg-card-hover border border-border rounded-lg text-sm text-muted hover:text-foreground transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Pipeline Info */}
      {health && (
        <div className="bg-card border border-border rounded-xl p-5">
          <div className="flex items-center gap-4">
            <div className={`w-3 h-3 rounded-full ${health.pipeline_loaded ? "bg-success pulse-dot" : "bg-warning"}`} />
            <div>
              <p className="text-sm font-medium text-foreground">
                Pipeline: {health.pipeline_loaded ? "Loaded & Ready" : "Not Loaded"}
              </p>
              <p className="text-xs text-muted">
                Status: {health.status}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Application Services */}
      <div>
        <h3 className="text-sm font-semibold text-foreground mb-3">
          Application Services
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {services.map((service) => (
            <div
              key={service.name}
              className="bg-card border border-border rounded-xl p-5 hover:border-accent/20 transition-colors"
            >
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-sm font-medium text-foreground">
                  {service.name}
                </h4>
                <div className="flex items-center gap-1.5">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      service.status === "healthy"
                        ? "bg-success pulse-dot"
                        : service.status === "unhealthy"
                          ? "bg-danger"
                          : "bg-muted"
                    }`}
                  />
                  <span
                    className={`text-xs ${
                      service.status === "healthy"
                        ? "text-success"
                        : service.status === "unhealthy"
                          ? "text-danger"
                          : "text-muted"
                    }`}
                  >
                    {service.status}
                  </span>
                </div>
              </div>
              <p className="text-xs text-muted mb-2">{service.description}</p>
              <p className="text-xs font-mono text-muted/60">
                Port: {service.port}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Event Statistics */}
      {eventStats && (
        <div className="bg-card border border-border rounded-xl p-5">
          <h3 className="text-sm font-semibold text-foreground mb-4">
            Event Collector Statistics
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-xs text-muted">Events Sent</p>
              <p className="text-xl font-bold text-foreground">
                {eventStats.events_sent.toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted">Events Failed</p>
              <p className="text-xl font-bold text-danger">
                {eventStats.events_failed.toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted">Buffer Size</p>
              <p className="text-xl font-bold text-foreground">
                {eventStats.buffer_size.toLocaleString()}
              </p>
            </div>
            <div>
              <p className="text-xs text-muted">Mode</p>
              <p className="text-xl font-bold text-accent-light">
                {eventStats.mode}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Infrastructure */}
      <div>
        <h3 className="text-sm font-semibold text-foreground mb-3">
          Infrastructure
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {INFRA.map((infra) => (
            <div
              key={infra.name}
              className="bg-card border border-border rounded-xl p-5 hover:border-accent/20 transition-colors"
            >
              <div className="flex items-center gap-3 mb-2">
                <span className="text-lg">{infra.icon}</span>
                <h4 className="text-sm font-medium text-foreground">
                  {infra.name}
                </h4>
              </div>
              <p className="text-xs text-muted mb-2">{infra.description}</p>
              <p className="text-xs font-mono text-muted/60">
                Port: {infra.port}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Architecture */}
      <div className="bg-card border border-border rounded-xl p-5">
        <h3 className="text-sm font-semibold text-foreground mb-3">
          Deployment Architecture
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-muted">
          <div className="space-y-1">
            <p className="text-foreground font-medium">Docker Compose (Dev)</p>
            <p>10 containers: 3 infra + 5 app + 2 monitoring</p>
          </div>
          <div className="space-y-1">
            <p className="text-foreground font-medium">Kubernetes (Prod)</p>
            <p>12 manifests, 11 pods, autoscaling ready</p>
          </div>
          <div className="space-y-1">
            <p className="text-foreground font-medium">Monitoring</p>
            <p>Prometheus scrapes RED + ML metrics, Grafana dashboards</p>
          </div>
        </div>
      </div>
    </div>
  );
}
