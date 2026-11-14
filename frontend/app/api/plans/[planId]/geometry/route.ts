import { NextResponse } from "next/server";

// Rooms (polygons + purpose read from the drawing), walls and openings of an analysed plan.
export async function GET(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  const { planId } = await params;
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  const query = new URL(request.url).search;
  try {
    const response = await fetch(`${backendUrl}/api/plans/${encodeURIComponent(planId)}/geometry${query}`, {
      cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the plan service" }, { status: 502 });
  }
}
