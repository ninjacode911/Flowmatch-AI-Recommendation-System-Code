import { NextResponse } from "next/server";

const BACKEND_URL = process.env.BACKEND_API_URL ?? "http://localhost:8000";

export async function GET() {
  try {
    const [health, ready] = await Promise.all([
      fetch(`${BACKEND_URL}/health`, { next: { revalidate: 0 } }).then((r) => r.json()),
      fetch(`${BACKEND_URL}/health/ready`, { next: { revalidate: 0 } }).then((r) => r.json()),
    ]);
    return NextResponse.json({ ...health, ...ready });
  } catch {
    return NextResponse.json(
      { status: "unhealthy", pipeline_loaded: false },
      { status: 503 }
    );
  }
}
