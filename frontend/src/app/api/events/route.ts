import { NextResponse } from "next/server";

const EVENT_URL = process.env.EVENT_COLLECTOR_URL ?? "http://localhost:8001";

export async function GET() {
  try {
    const res = await fetch(`${EVENT_URL}/events/stats`, { next: { revalidate: 0 } });
    const data = await res.json();
    return NextResponse.json(data);
  } catch {
    return NextResponse.json(
      { error: "Event collector unavailable" },
      { status: 502 }
    );
  }
}
