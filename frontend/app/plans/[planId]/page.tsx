"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type PlanRecord = {
  id: string;
  metadata?: { filename?: string };
  plan_images?: Array<{ upload_status?: string; role?: string }>;
};

export default function PlanPage({ params }: { params: Promise<{ planId: string }> }) {
  const [planId, setPlanId] = useState("");
  const [plan, setPlan] = useState<PlanRecord | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState("Preparing your floor plan…");

  useEffect(() => {
    let objectUrl: string | null = null;
    let cancelled = false;
    void params.then(async ({ planId: id }) => {
      setPlanId(id);
      const uploadError = new URLSearchParams(window.location.search).get("uploadError");
      if (uploadError) {
        setError(uploadError);
        return;
      }
      try {
        let record: PlanRecord | null = null;
        for (let attempt = 0; attempt < 90; attempt++) {
          const planResponse = await fetch(`/api/plans/${encodeURIComponent(id)}`, { cache: "no-store" });
          if (planResponse.ok) {
            record = await planResponse.json();
            const status = record?.plan_images?.find(image => image.role === "source")?.upload_status;
            if (status === "failed") throw new Error("The image upload failed. Go back and try again.");
            if (status === "uploaded") break;
          }
          if (cancelled) return;
          await new Promise(resolve => setTimeout(resolve, 1500));
        }
        if (!record || record.plan_images?.find(image => image.role === "source")?.upload_status !== "uploaded") {
          throw new Error("Upload is taking longer than expected. Please try again.");
        }
        if (cancelled) return;
        setPlan(record);
        const imageResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/image`, { cache: "no-store" });
        if (!imageResponse.ok) {
          const body = await imageResponse.json();
          throw new Error(body.detail || "Plan image could not be loaded");
        }
        objectUrl = URL.createObjectURL(await imageResponse.blob());
        if (!cancelled) {
          setImageUrl(objectUrl);
          setAnalysisStatus("Image ready · Analyzing floor plan…");
          try {
            const analysisResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/analyze`, { method: "POST" });
            const result = await analysisResponse.json();
            if (!analysisResponse.ok) throw new Error(result.detail || "Analysis failed");
            if (!cancelled) setAnalysisStatus("Floor plan ready");
          } catch {
            if (!cancelled) setAnalysisStatus("Image ready · Analysis unavailable");
          }
        }
      } catch (loadError) {
        if (!cancelled) setError(loadError instanceof Error ? loadError.message : "Plan could not be loaded");
      }
    });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [params]);

  const imageStatus = plan?.plan_images?.find(image => image.role === "source")?.upload_status;
  return (
    <main className="workspace">
      <header className="toolbar">
        <div className="plan-title" title={plan?.metadata?.filename ?? "Loading plan…"}>
          <span className="plan-title-icon" aria-hidden="true">▤</span>
          <span>{plan?.metadata?.filename ?? "Loading plan…"}</span>
        </div>
        <div className="toolbar-actions">
          <span className="connection-status"><i /> {error ?? (imageStatus === "uploaded" ? analysisStatus : "Loading saved plan…")}</span>
          <Link className="toolbar-button" href="/">Upload another plan</Link>
        </div>
      </header>
      <div className="work-area">
        <section className="drawing-panel" aria-label="Plan drawing">
          {error ? <div className="upload-empty-state"><h1>Could not load this plan</h1><p role="alert">{error}</p><Link className="toolbar-button" href="/">Back to upload</Link></div> : (
            imageUrl ? <div className="image-frame"><img className="plan-image" src={imageUrl} alt={`Floor plan: ${plan?.metadata?.filename ?? "saved plan"}`} /></div> : <div className="loading-state" role="status"><span className="loading-spinner" /><h1>Loading your floor plan</h1><p>Getting your image ready…</p></div>
          )}
        </section>
        <aside className="chat-panel" aria-label="Plan questions">
          <div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div>
          <div className="conversation" aria-live="polite"><div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>Plan loaded from your saved workspace.</p><span>Ask about quantities, rooms, and materials.</span></div></div>
          <form className="composer" onSubmit={event => event.preventDefault()}><textarea aria-label="Ask a question about the plan" placeholder="Ask about quantities, rooms, or materials…" disabled /><div className="composer-footer"><span>Analysis will be available when connected</span><button className="send-button" type="submit" disabled aria-label="Send question">↑</button></div></form>
        </aside>
      </div>
    </main>
  );
}
