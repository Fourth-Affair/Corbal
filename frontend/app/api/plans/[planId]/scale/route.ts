import { NextResponse } from "next/server";

async function proxy(request: Request, planId: string, method: "GET" | "PUT") {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const response = await fetch(`${backendUrl}/api/plans/${encodeURIComponent(planId)}/scale`, {
      method,
      cache: "no-store",
      ...(method === "PUT" ? { headers: { "Content-Type": "application/json" }, body: await request.text() } : {}),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the plan service" }, { status: 502 });
  }
}

export async function GET(_request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(_request, (await params).planId, "GET");
}

export async function PUT(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(request, (await params).planId, "PUT");
}
