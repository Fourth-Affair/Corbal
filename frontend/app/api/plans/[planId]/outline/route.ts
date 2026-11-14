import { NextResponse } from "next/server";

async function proxy(request: Request, planId: string, method: "GET" | "PUT" | "DELETE") {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const response = await fetch(`${backendUrl}/api/plans/${encodeURIComponent(planId)}/outline`, {
      method,
      cache: "no-store",
      ...(method === "PUT" ? { headers: { "Content-Type": "application/json" }, body: await request.text() } : {}),
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the plan service" }, { status: 502 });
  }
}

export async function GET(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(request, (await params).planId, "GET");
}

export async function PUT(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(request, (await params).planId, "PUT");
}

export async function DELETE(request: Request, { params }: { params: Promise<{ planId: string }> }) {
  return proxy(request, (await params).planId, "DELETE");
}
