type HealthResponse = {
  status: string;
};

async function getBackendStatus(): Promise<string> {
  const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";

  try {
    const response = await fetch(`${backendUrl}/health`, { cache: "no-store" });
    if (!response.ok) return "unavailable";

    const data = (await response.json()) as HealthResponse;
    return data.status;
  } catch {
    return "unavailable";
  }
}

export default async function Home() {
  const backendStatus = await getBackendStatus();

  return (
    <main className="page">
      <section className="card">
        <p className="eyebrow">Starter app</p>
        <h1>Next.js + FastAPI</h1>
        <p className="intro">
          Your frontend and API are running as separate services in Docker Compose.
        </p>
        <div className="status-row">
          <span className={`status-dot ${backendStatus === "ok" ? "healthy" : "unavailable"}`} />
          <span>Backend status: <strong>{backendStatus}</strong></span>
        </div>
        <nav className="links" aria-label="Developer links">
          <a href="http://localhost:8000/docs">FastAPI docs <span aria-hidden="true">↗</span></a>
          <a href="http://localhost:8000/health">Health endpoint <span aria-hidden="true">↗</span></a>
        </nav>
      </section>
    </main>
  );
}
