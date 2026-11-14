"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

type PlanRecord = {
  id: string;
  metadata?: { filename?: string };
  plan_images?: Array<{ upload_status?: string; role?: string }>;
};

type ChatMessage = { role: "user" | "assistant"; text: string };
type PlanCitation = {
  id: string;
  bbox: [number, number, number, number];
  polygon?: Array<[number, number]> | null;
};

export default function PlanPage({ params }: { params: Promise<{ planId: string }> }) {
  const [planId, setPlanId] = useState("");
  const [plan, setPlan] = useState<PlanRecord | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState("Preparing your floor plan…");
  const [ready, setReady] = useState(false);
  const [question, setQuestion] = useState("");
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [citations, setCitations] = useState<PlanCitation[]>([]);
  const conversationRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const conversation = conversationRef.current;
    if (conversation) conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

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
            if (!cancelled) {
              if (result.width && result.height) setImageSize({ width: result.width, height: result.height });
              setAnalysisStatus("Floor plan ready");
              setReady(true);
            }
          } catch (analysisError) {
            if (!cancelled) {
              const reason = analysisError instanceof Error ? analysisError.message : "Unknown analysis error";
              setAnalysisStatus(`Analysis unavailable: ${reason}`);
            }
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

  async function submitQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const submitted = question.trim();
    if (!submitted || !ready || sending || !planId) return;
    setQuestion("");
    setCitations([]);
    setMessages(current => [...current, { role: "user", text: submitted }]);
    setSending(true);
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/ask`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: submitted, history: messages.map(({ role, text }) => ({ role, text })) }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not answer that question");
      const answer = result.answer === null ? result.explanation : `${result.answer}${result.unit ? ` ${result.unit}` : ""}. ${result.explanation}`;
      setCitations(Array.isArray(result.citations) ? result.citations : []);
      setMessages(current => [...current, { role: "assistant", text: answer }]);
    } catch (askError) {
      setCitations([]);
      const message = askError instanceof Error ? askError.message : "Could not answer that question";
      setMessages(current => [...current, { role: "assistant", text: message }]);
    } finally {
      setSending(false);
    }
  }

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
            imageUrl ? <div className="image-frame"><img className="plan-image" src={imageUrl} alt={`Floor plan: ${plan?.metadata?.filename ?? "saved plan"}`} />{imageSize && citations.length > 0 && <svg className="citation-overlay" viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="xMidYMid meet" aria-label={`Highlighted ${citations.length} supporting items`}>
              {citations.map((item, index) => {
                const [x0, y0, x1, y1] = item.bbox;
                const points = item.polygon?.map(([x, y]) => `${x},${y}`).join(" ");
                return <g className="citation-mark" key={`${item.id}-${index}`}>
                  {points ? <polygon points={points} /> : <rect x={x0} y={y0} width={Math.max(1, x1 - x0)} height={Math.max(1, y1 - y0)} />}
                  <circle cx={x0} cy={y0} r="18" />
                  <text x={x0} y={y0}>{index + 1}</text>
                </g>;
              })}
            </svg>}</div> : <div className="loading-state" role="status"><span className="loading-spinner" /><h1>Loading your floor plan</h1><p>Getting your image ready…</p></div>
          )}
        </section>
        <aside className="chat-panel" aria-label="Plan questions">
          <div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div>
          <div className="conversation" aria-live="polite" ref={conversationRef}>{messages.length === 0 && !sending ? <div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>{ready ? "Plan analysis is ready." : "Preparing the saved plan analysis…"}</p><span>Ask about quantities, rooms, and detected items.</span></div> : <>{messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${index}-${message.role}`}><span>{message.role === "user" ? "You" : "Stoa"}</span><p>{message.text}</p></div>)}{sending && <div className="chat-message assistant thinking-message" role="status"><span>Stoa</span><p><i className="thinking-spinner" aria-hidden="true" />Thinking through the plan…</p></div>}</>}</div>
          <form className="composer" onSubmit={submitQuestion}><textarea aria-label="Ask a question about the plan" placeholder="Ask about quantities, rooms, or materials…" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} disabled={!ready || sending} /><div className="composer-footer"><span>{sending ? "Thinking…" : ready ? "Answers use the analysis saved for this plan" : "Available when analysis finishes"}</span><button className="send-button" type="submit" disabled={!ready || sending || !question.trim()} aria-label="Send question">↑</button></div></form>
        </aside>
      </div>
    </main>
  );
}
