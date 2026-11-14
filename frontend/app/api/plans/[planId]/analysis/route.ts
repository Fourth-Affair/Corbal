import { NextResponse } from "next/server";

// Status of the plan's background analysis job: 202 while queued / running, 200 when done or failed, 404 unknown.
export async function GET(_request: Request, { params }: { params: Promise<{ planId: string }> }) {
  const { planId } = await params;
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const response = await fetch(`${backendUrl}/api/plans/${encodeURIComponent(planId)}/analysis`, { cache: "no-store" });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the analysis backend" }, { status: 502 });
  }
}
