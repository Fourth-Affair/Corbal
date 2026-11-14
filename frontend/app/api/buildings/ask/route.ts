import { NextResponse } from "next/server";

export async function POST(request: Request) {
  const backendUrl = process.env.BACKEND_URL || "http://localhost:8000";
  try {
    const payload = await request.json();
    const response = await fetch(`${backendUrl}/api/buildings/ask`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload), cache: "no-store",
    });
    return NextResponse.json(await response.json(), { status: response.status });
  } catch {
    return NextResponse.json({ detail: "Could not reach the analysis backend" }, { status: 502 });
  }
}
