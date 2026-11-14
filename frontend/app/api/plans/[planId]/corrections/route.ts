import { NextResponse } from "next/server";

async function proxy(request: Request, planId: string, method: "GET" | "POST") {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const response = await fetch(`${backendUrl}/api/plans/${encodeURIComponent(planId)}/corrections`, {
      method,
      cache: "no-store",
      ...(method === "POST" ? { headers: { "Content-Type": "application/json" }, body: await request.text() } : {}),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the plan service" }, { status: 502 });
  }
}

export async function GET(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(request, (await params).planId, "GET");
}

export async function POST(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(request, (await params).planId, "POST");
}
