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
type ScaleCalibration = { points: [[number, number], [number, number]]; distance: number; unit: "m" | "cm" | "ft" | "in" };

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
  const [scale, setScale] = useState<ScaleCalibration | null>(null);
  const [scaleLoaded, setScaleLoaded] = useState(false);
  const [scaleMode, setScaleMode] = useState(false);
  const [measureMode, setMeasureMode] = useState(false);
  const [scalePoints, setScalePoints] = useState<Array<[number, number]>>([]);
  const [measurePoints, setMeasurePoints] = useState<Array<[number, number]>>([]);
  const [scaleDistance, setScaleDistance] = useState("");
  const [scaleUnit, setScaleUnit] = useState<ScaleCalibration["unit"]>("m");
  const [scaleSaving, setScaleSaving] = useState(false);
  const [scaleError, setScaleError] = useState<string | null>(null);
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
        try {
          const savedScale = await fetch(`/api/plans/${encodeURIComponent(id)}/scale`, { cache: "no-store" });
          if (savedScale.ok) {
            const saved = await savedScale.json();
            if (saved?.points?.length === 2) {
              setScale(saved);
              setScalePoints(saved.points);
              setScaleUnit(saved.unit);
            }
          }
        } catch {
          setScale(null);
          setScalePoints([]);
        } finally {
          setScaleLoaded(true);
        }
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
        body: JSON.stringify({ question: submitted, history: messages.map(({ role, text }) => ({ role, text })), scale }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not answer that question");
      const explanation = typeof result.explanation === "string" ? result.explanation : "";
      const needsScale = !scale && /(?:no|without|missing|lack(?:s|ing)?|unavailable|cannot|can't|unable to).{0,70}(?:scale|dimension|measurement)|(?:scale|dimension|measurement).{0,70}(?:not available|unavailable|missing|cannot be determined)/i.test(explanation);
      const answer = needsScale
        ? "Let’s set the drawing scale first. Click the two ends of a distance you know on the plan, such as a wall with a printed measurement. Then enter that real distance and unit to calibrate the drawing."
        : result.answer === null ? result.explanation : `${result.answer}${result.unit ? ` ${result.unit}` : ""}. ${result.explanation}`;
      setCitations(Array.isArray(result.citations) ? result.citations : []);
      setMessages(current => [...current, { role: "assistant", text: answer }]);
      if (needsScale) {
        startScaleCalibration();
      }
    } catch (askError) {
      setCitations([]);
      const message = askError instanceof Error ? askError.message : "Could not answer that question";
      setMessages(current => [...current, { role: "assistant", text: message }]);
    } finally {
      setSending(false);
    }
  }

  function startScaleCalibration() {
    setScaleMode(true);
    setMeasureMode(false);
    setScalePoints([]);
    setMeasurePoints([]);
    setScaleDistance("");
    setScaleUnit("m");
    setScaleError(null);
  }

  function cancelScaleCalibration() {
    setScaleMode(false);
    setScalePoints(scale?.points ?? []);
    setScaleDistance("");
    setScaleError(null);
  }

  function markScalePoint(event: React.MouseEvent<SVGSVGElement>) {
    if ((!scaleMode && !measureMode) || !imageSize) return;
    const svg = event.currentTarget;
    const point = svg.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const transform = svg.getScreenCTM();
    if (!transform) return;
    const imagePoint = point.matrixTransform(transform.inverse());
    const pixelPoint: [number, number] = [Math.max(0, Math.min(imageSize.width, imagePoint.x)), Math.max(0, Math.min(imageSize.height, imagePoint.y))];
    if (scaleMode) {
      const next = [...scalePoints.slice(0, 1), pixelPoint];
      setScalePoints(next);
      setScaleError(null);
      if (next.length === 2) setScaleMode(false);
    } else {
      const next = [...measurePoints.slice(0, 1), pixelPoint];
      setMeasurePoints(next);
      if (next.length === 2) setMeasureMode(false);
    }
  }

  async function saveScale(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (scalePoints.length !== 2 || !Number(scaleDistance) || !planId) return;
    setScaleSaving(true);
    setScaleError(null);
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/scale`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ points: scalePoints, distance: Number(scaleDistance), unit: scaleUnit }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save scale");
      setScale(result);
      setScalePoints(result.points);
      setScaleDistance(String(result.distance));
      setScaleUnit(result.unit);
    } catch (saveError) {
      setScaleError(saveError instanceof Error ? saveError.message : "Could not save scale");
    } finally { setScaleSaving(false); }
  }

  const pxPerUnit = scale ? Math.hypot(scale.points[1][0] - scale.points[0][0], scale.points[1][1] - scale.points[0][1]) / scale.distance : null;
  const measuredDistance = scale && pxPerUnit && measurePoints.length === 2
    ? Math.hypot(measurePoints[1][0] - measurePoints[0][0], measurePoints[1][1] - measurePoints[0][1]) / pxPerUnit
    : null;

  const imageStatus = plan?.plan_images?.find(image => image.role === "source")?.upload_status;
  return (
    <main className="workspace">
      <header className="toolbar">
        <div className="plan-title" title={plan?.metadata?.filename ?? "Loading plan…"}>
          <span className="plan-title-icon" aria-hidden="true">▤</span>
          <span>{plan?.metadata?.filename ?? "Loading plan…"}</span>
        </div>
        <div className="toolbar-actions">
          {ready && scaleLoaded && scale && !scaleMode && <span className="scale-status">Scale saved</span>}
          {ready && scaleLoaded && !scaleMode && <button className={`toolbar-button${scale ? "" : " primary-button"}`} type="button" onClick={startScaleCalibration}>{scale ? "Adjust" : "Set scale"}</button>}
          {ready && scaleLoaded && scaleMode && <button className="toolbar-button" type="button" onClick={cancelScaleCalibration}>Cancel scale setup</button>}
          {ready && scaleLoaded && scale && !scaleMode && <button className="toolbar-button" type="button" onClick={() => { setMeasureMode(true); setMeasurePoints([]); setScaleMode(false); }}>Measure distance</button>}
          <span className="connection-status"><i /> {error ?? (imageStatus === "uploaded" ? analysisStatus : "Loading saved plan…")}</span>
          <Link className="toolbar-button" href="/">Upload another plan</Link>
        </div>
      </header>
      <div className="work-area">
        <section className={`drawing-panel${scaleMode ? " is-calibrating" : ""}`} aria-label="Plan drawing">
          {error ? <div className="upload-empty-state"><h1>Could not load this plan</h1><p role="alert">{error}</p><Link className="toolbar-button" href="/">Back to upload</Link></div> : (
            imageUrl ? <div className="image-frame"><img className="plan-image" src={imageUrl} alt={`Floor plan: ${plan?.metadata?.filename ?? "saved plan"}`} />{imageSize && <svg className={`citation-overlay${scaleMode || measureMode ? " scale-selecting" : ""}`} viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="xMidYMid meet" onClick={markScalePoint} aria-label={scaleMode || measureMode ? "Select two points" : `Highlighted ${citations.length} supporting items`}>
              {citations.map((item, index) => {
                const [x0, y0, x1, y1] = item.bbox;
                const points = item.polygon?.map(([x, y]) => `${x},${y}`).join(" ");
                return <g className="citation-mark" key={`${item.id}-${index}`}>
                  {points ? <polygon points={points} /> : <rect x={x0} y={y0} width={Math.max(1, x1 - x0)} height={Math.max(1, y1 - y0)} />}
                  <circle cx={x0} cy={y0} r="18" /><text x={x0} y={y0}>{index + 1}</text>
                </g>;
              })}
              {scaleMode && scalePoints.length > 0 && <g className="scale-mark"><line x1={scalePoints[0][0]} y1={scalePoints[0][1]} x2={(scalePoints[1] ?? scalePoints[0])[0]} y2={(scalePoints[1] ?? scalePoints[0])[1]} />{scalePoints.map(([x, y], index) => <g key={index}><circle cx={x} cy={y} r="18"/><text x={x} y={y}>{index + 1}</text></g>)}</g>}
              {measurePoints.length > 0 && <g className="scale-mark"><line x1={measurePoints[0][0]} y1={measurePoints[0][1]} x2={(measurePoints[1] ?? measurePoints[0])[0]} y2={(measurePoints[1] ?? measurePoints[0])[1]} />{measurePoints.map(([x, y], index) => <g key={index}><circle cx={x} cy={y} r="18"/><text x={x} y={y}>{index + 1}</text></g>)}</g>}
            </svg>}{(scaleMode || measureMode) && <div className="scale-hint">{scaleMode ? (scalePoints.length === 0 ? "Step 1 of 2 · Click one end of a printed dimension" : "Step 2 of 2 · Click the other end of that dimension") : (measurePoints.length === 0 ? "Click the start of the distance to measure" : "Click the end of the distance")}</div>}{measuredDistance !== null && <div className="scale-card"><strong>Measured distance</strong><span>{measuredDistance.toFixed(2)} {scale?.unit}</span></div>}</div> : <div className="loading-state" role="status"><span className="loading-spinner" /><h1>Loading your floor plan</h1><p>Getting your image ready…</p></div>
          )}
        </section>
        <aside className="chat-panel" aria-label="Plan questions">
          {scaleMode && scalePoints.length === 2 && <form className="scale-form" onSubmit={saveScale}><div><strong>Enter the printed distance</strong><span>Use the value shown between the points on the plan, then choose its unit.</span></div><input aria-label="Known distance" type="number" min="0.001" step="any" placeholder="Distance" value={scaleDistance} onChange={event => setScaleDistance(event.target.value)} required /><select aria-label="Distance unit" value={scaleUnit} onChange={event => setScaleUnit(event.target.value as ScaleCalibration["unit"])}><option value="m">metres</option><option value="cm">centimetres</option><option value="ft">feet</option><option value="in">inches</option></select><button className="toolbar-button primary-button" disabled={scaleSaving || !scaleDistance}>{scaleSaving ? "Saving…" : "Save scale"}</button>{scaleError && <p role="alert">{scaleError}</p>}</form>}
          <div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div>
          <div className="conversation" aria-live="polite" ref={conversationRef}>{messages.length === 0 && !sending ? <div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>{ready ? "Plan analysis is ready." : "Preparing the saved plan analysis…"}</p><span>Ask about quantities, rooms, and detected items.</span></div> : <>{messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${index}-${message.role}`}><span>{message.role === "user" ? "You" : "Stoa"}</span><p>{message.text}</p></div>)}{sending && <div className="chat-message assistant thinking-message" role="status"><span>Stoa</span><p><i className="thinking-spinner" aria-hidden="true" />Thinking through the plan…</p></div>}</>}</div>
          <form className="composer" onSubmit={submitQuestion}><textarea aria-label="Ask a question about the plan" placeholder="Ask about quantities, rooms, or materials…" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} disabled={!ready || sending} /><div className="composer-footer"><span>{sending ? "Thinking…" : ready ? "Answers use the analysis saved for this plan" : "Available when analysis finishes"}</span><button className="send-button" type="submit" disabled={!ready || sending || !question.trim()} aria-label="Send question">↑</button></div></form>
        </aside>
      </div>
    </main>
  );
}
