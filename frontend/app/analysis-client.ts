// Client side of the backend's background analysis jobs (backend/app/jobs.py): POST /analyze starts or joins a job,
// GET /analysis is polled until it finishes. Used by the plan page and the building page.

type AnalysisJob = { state?: "queued" | "running" | "done" | "failed" | "unknown"; stage?: string | null; position?: number | null; elapsed_s?: number; detail?: string; graph?: unknown };

// Readable labels for the backend's analysis stages (app/profiling.py names); the first matching prefix wins
const STAGE_LABELS: Array<[string, string]> = [
  ["download", "Loading the plan"], ["decode", "Loading the plan"], ["pdf_parse", "Reading the PDF"],
  ["pipeline.perception.ocr", "Reading text"], ["pipeline.perception", "Recognising rooms and symbols"],
  ["pipeline.symbols", "Detecting doors, windows and fixtures"], ["pipeline", "Counting rooms and fixtures"],
  ["geometry.walls", "Finding walls"], ["geometry", "Tracing rooms"], ["persist", "Saving the analysis"],
];

export function analysisProgress(job: AnalysisJob): string {
  if (job.state === "queued") {
    return job.position ? `Waiting for the analyser · ${job.position} plan${job.position === 1 ? "" : "s"} ahead` : "Starting analysis…";
  }
  const label = STAGE_LABELS.find(([prefix]) => job.stage === prefix || job.stage?.startsWith(`${prefix}.`))?.[1] ?? "Analyzing floor plan";
  return `${label}…${job.elapsed_s != null ? ` · ${Math.round(job.elapsed_s)} s` : ""}`;
}

// POST starts (or joins) the plan's background analysis job, then the job is polled until it finishes.
// A 200 without "state" is an analysis saved earlier; "unknown" means the backend restarted: submit again.
export async function runAnalysis(planId: string, isCancelled: () => boolean, onProgress: (status: string) => void) {
  const base = `/api/plans/${encodeURIComponent(planId)}`;
  const read = async (request: Promise<Response>) => {
    const response = await request.catch(() => new Response("{}", { status: 502 }));
    return { response, body: await response.json().catch(() => ({})) as AnalysisJob & Record<string, unknown> };
  };
  let { response, body } = await read(fetch(`${base}/analyze`, { method: "POST" }));
  let errors = 0;
  for (;;) {
    if (isCancelled()) throw new Error("Analysis cancelled");
    if (response.ok && body.state === undefined) return body;
    if (body.state === "done") return body.graph as Record<string, unknown>;
    if (body.state === "failed") throw new Error(body.detail || "Analysis failed");
    if (body.state === "unknown") {
      ({ response, body } = await read(fetch(`${base}/analyze`, { method: "POST" })));
      continue;
    }
    if (response.status === 202) {
      errors = 0;
      onProgress(analysisProgress(body));
    } else if (++errors >= 5) {  // the proxy or backend kept failing (restart, network)
      throw new Error(body.detail || "Analysis failed");
    }
    await new Promise(resolve => setTimeout(resolve, 1500));
    ({ response, body } = await read(fetch(`${base}/analysis`, { cache: "no-store" })));
  }
}
