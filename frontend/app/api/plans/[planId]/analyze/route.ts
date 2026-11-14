import { NextResponse } from "next/server";

export async function POST(
  _request: Request,
  { params }: { params: Promise<{ planId: string }> },
) {
  const { planId } = await params;
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";

  try {
    const response = await fetch(`${backendUrl}/api/plans/${encodeURIComponent(planId)}/analyze`, {
      method: "POST",
      cache: "no-store",
    });
    const body = await response.json();
    return NextResponse.json(body, { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the analysis backend" }, { status: 502 });
  }
}
