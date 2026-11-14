"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

type PlanRecord = {
  id: string;
  metadata?: { filename?: string };
  plan_images?: Array<{ upload_status?: string; role?: string }>;
};

type ChatMessage = { role: "user" | "assistant"; text: string; evidence_ids?: string[]; feedback?: string };
type PlanCitation = {
  id: string;
  bbox: [number, number, number, number];
  polygon?: Array<[number, number]> | null;
  raw?: string | null;
  user_verified?: boolean;
};
type EvidenceItem = PlanCitation & { category: string; cls: string; text?: string | null; score?: number | null; raw?: string | null; user_verified?: boolean };
type ScaleCalibration = { points: [[number, number], [number, number]]; distance: number; unit: "m" | "cm" | "ft" | "in" };
type FloorOutline = { id: string; polygon: Array<[number, number]>; score?: number | null; scope?: "floor" | "room"; room_type?: string | null; label?: string };
const itemTypes: Record<string, string[]> = {
  room: ["bedroom", "bathroom", "wc", "sauna", "kitchen", "balcony", "other"],
  fixture: ["toilet", "sink", "double_sink", "shower", "bathtub", "sauna_stove", "sauna_bench", "other"],
  door: ["door"], window: ["window"],
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
  const [scale, setScale] = useState<ScaleCalibration | null>(null);
  const [scaleLoaded, setScaleLoaded] = useState(false);
  const [scaleMode, setScaleMode] = useState(false);
  const [measureMode, setMeasureMode] = useState(false);
  const [scalePoints, setScalePoints] = useState<Array<[number, number]>>([]);
  const [measurePoints, setMeasurePoints] = useState<Array<[number, number]>>([]);
  const [draggingMeasurePoint, setDraggingMeasurePoint] = useState<number | null>(null);
  const [scaleDistance, setScaleDistance] = useState("");
  const [scaleUnit, setScaleUnit] = useState<ScaleCalibration["unit"]>("m");
  const [scaleSaving, setScaleSaving] = useState(false);
  const [scaleError, setScaleError] = useState<string | null>(null);
  const [floorOutlines, setFloorOutlines] = useState<FloorOutline[]>([]);
  const [savedOutlines, setSavedOutlines] = useState<FloorOutline[]>([]);
  const [outlineMode, setOutlineMode] = useState(false);
  const [outlinePoints, setOutlinePoints] = useState<Array<[number, number]>>([]);
  const [outlineCandidateIndex, setOutlineCandidateIndex] = useState(0);
  const [manualOutlineTarget, setManualOutlineTarget] = useState<FloorOutline | null>(null);
  const [addOutlinePointMode, setAddOutlinePointMode] = useState(false);
  const [draggingOutlinePoint, setDraggingOutlinePoint] = useState<number | null>(null);
  const [outlineSaving, setOutlineSaving] = useState(false);
  const [outlineError, setOutlineError] = useState<string | null>(null);
  const [outlineRemoving, setOutlineRemoving] = useState(false);
  const [pendingAreaQuestion, setPendingAreaQuestion] = useState<string | null>(null);
  const [evidenceItems, setEvidenceItems] = useState<EvidenceItem[]>([]);
  const [correctionEvents, setCorrectionEvents] = useState<Array<{ action: string }>>([]);
  const [correctionMode, setCorrectionMode] = useState(false);
  const [correctionAddMode, setCorrectionAddMode] = useState(false);
  const [correctionAddPoints, setCorrectionAddPoints] = useState<Array<[number, number]>>([]);
  const [selectedEvidenceId, setSelectedEvidenceId] = useState<string | null>(null);
  const [editCategory, setEditCategory] = useState("room");
  const [editClass, setEditClass] = useState("bedroom");
  const [addCategory, setAddCategory] = useState("room");
  const [addClass, setAddClass] = useState("bedroom");
  const [correctionSaving, setCorrectionSaving] = useState(false);
  const [correctionError, setCorrectionError] = useState<string | null>(null);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
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
            if (status === "failed") throw new Error("The upload failed. Go back and try again.");
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
              setEvidenceItems(Array.isArray(result.items) ? result.items : []);
              setFloorOutlines(Array.isArray(result.floor_outlines) ? result.floor_outlines : []);
              try {
                const correctionResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/corrections`, { cache: "no-store" });
                if (correctionResponse.ok) {
                  const corrections = await correctionResponse.json();
                  setEvidenceItems(Array.isArray(corrections.items) ? corrections.items : result.items ?? []);
                  setCorrectionEvents(Array.isArray(corrections.events) ? corrections.events : []);
                }
              } catch { /* Keep the plan usable if correction history is temporarily unavailable. */ }
              try {
                const outlineResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/outline`, { cache: "no-store" });
                if (outlineResponse.ok) {
                  const outlineData = await outlineResponse.json();
                  setFloorOutlines(Array.isArray(outlineData.candidates) ? outlineData.candidates : []);
                  setSavedOutlines(Array.isArray(outlineData.saved) ? outlineData.saved : []);
                }
              } catch { /* Keep the scanned plan usable if saved outlines are temporarily unavailable. */ }
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

  async function saveCorrection(action: "add" | "update" | "delete", itemId?: string, item?: Partial<EvidenceItem>) {
    if (!planId) return;
    setCorrectionSaving(true);
    setCorrectionError(null);
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/corrections`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action, ...(itemId ? { item_id: itemId } : {}), ...(item ? { item } : {}) }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save correction");
      setEvidenceItems(result.items ?? []);
      setCorrectionEvents(result.events ?? []);
      setCorrectionAddMode(false);
      setCorrectionAddPoints([]);
      if (action === "add") setSelectedEvidenceId(null);
    } catch (saveError) {
      setCorrectionError(saveError instanceof Error ? saveError.message : "Could not save correction");
    } finally { setCorrectionSaving(false); }
  }

  async function submitFeedback(messageIndex: number, feedback: "helpful" | "incorrect" | "unclear") {
    if (!planId) return;
    const answer = messages[messageIndex];
    const question = [...messages.slice(0, messageIndex)].reverse().find(message => message.role === "user");
    if (!answer || !question) return;
    setFeedbackError(null);
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/feedback`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: question.text, answer: answer.text, feedback, evidence_ids: answer.evidence_ids ?? [] }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save feedback");
      setMessages(current => current.map((message, index) => index === messageIndex ? { ...message, feedback } : message));
    } catch (error) {
      setFeedbackError(error instanceof Error ? error.message : "Could not save feedback");
    }
  }

  function beginCorrection(item: EvidenceItem) {
    setSelectedEvidenceId(item.id);
    setEditCategory(item.category in itemTypes ? item.category : "room");
    setEditClass(itemTypes[item.category]?.includes(item.cls) ? item.cls : (itemTypes[item.category in itemTypes ? item.category : "room"]?.[0] ?? "other"));
  }

  function startCorrectionReview() {
    setCorrectionMode(true);
    setCorrectionAddMode(false);
    setCorrectionError(null);
    setOutlineMode(false);
    setScaleMode(false);
    setMeasureMode(false);
  }

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
        body: JSON.stringify({ question: submitted, history: messages.map(({ role, text }) => ({ role, text })), scale, floor_outlines: savedOutlines }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not answer that question");
      const explanation = typeof result.explanation === "string" ? result.explanation : "";
      const needsScale = !scale && /scale|known distance/i.test(explanation) && /area|dimension|measure/i.test(explanation);
      const needsOutline = Boolean(scale) && /boundary|outline/i.test(explanation) && !hasRequiredOutline(submitted);
      const answer = needsScale
        ? "Let’s set the drawing scale first. Click the two ends of a distance you know on the plan, such as a wall with a printed measurement. Then enter that real distance and unit to calibrate the drawing."
        : needsOutline
          ? getRequestedRoomType(submitted)
            ? floorOutlines.some(outline => outline.scope === "room" && outline.room_type === getRequestedRoomType(submitted)?.roomType)
              ? `I found a suggested ${getRequestedRoomType(submitted)?.label.toLowerCase()} outline. Review it, adjust it if needed, and accept it to calculate an area estimate.`
              : `I couldn’t detect a clear ${getRequestedRoomType(submitted)?.label.toLowerCase()} boundary. Draw or edit the outline, then accept it to calculate an area estimate.`
            : floorOutlines.some(outline => outline.scope !== "room")
              ? "I found a suggested floor outline. Review it, adjust it if needed, and accept it to calculate an area estimate."
              : "I couldn’t detect a clear floor boundary. Draw or edit the outline, then accept it to calculate an area estimate."
        : result.answer === null ? result.explanation : `${result.answer}${result.unit ? ` ${result.unit}` : ""}. ${result.explanation}`;
      setCitations(Array.isArray(result.citations) ? result.citations : []);
      setMessages(current => [...current, { role: "assistant", text: answer, evidence_ids: (result.citations ?? []).map((item: PlanCitation) => item.id) }]);
      if (needsScale) {
        setPendingAreaQuestion(submitted);
        startScaleCalibration();
      } else if (needsOutline) {
        setPendingAreaQuestion(submitted);
        startOutlineReview(submitted);
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

  function getRequestedRoomType(text: string) {
    const normalized = text.toLowerCase();
    const rooms = [
      { roomType: "kitchen", label: "Kitchen", terms: ["kitchen"] },
      { roomType: "bathroom", label: "Bathroom", terms: ["bathroom", "bath"] },
      { roomType: "bedroom", label: "Bedroom", terms: ["bedroom", "bed room"] },
      { roomType: "living", label: "Living room", terms: ["living room", "living area"] },
      { roomType: "balcony", label: "Balcony", terms: ["balcony"] },
      { roomType: "entry", label: "Entry", terms: ["entry", "hallway"] },
      { roomType: "storage", label: "Storage", terms: ["storage"] },
      { roomType: "garage", label: "Garage", terms: ["garage"] },
    ];
    return rooms.find(room => room.terms.some(term => normalized.includes(term))) ?? null;
  }

  function hasRequiredOutline(text: string) {
    const targetRoom = getRequestedRoomType(text);
    return targetRoom
      ? savedOutlines.some(outline => outline.scope === "room" && outline.room_type === targetRoom.roomType)
      : savedOutlines.some(outline => outline.scope !== "room");
  }

  function startOutlineReview(questionText?: string) {
    const targetRoom = questionText ? getRequestedRoomType(questionText) : null;
    const requestedOutlineIndex = targetRoom
      ? floorOutlines.findIndex(outline => outline.scope === "room" && outline.room_type === targetRoom.roomType)
      : floorOutlines.findIndex(outline => outline.scope !== "room");
    const savedTarget = targetRoom
      ? savedOutlines.find(outline => outline.scope === "room" && outline.room_type === targetRoom.roomType)
      : savedOutlines.find(outline => outline.scope !== "room");
    const index = requestedOutlineIndex >= 0 ? requestedOutlineIndex : Math.max(0, outlineCandidateIndex);
    const suggestion = requestedOutlineIndex >= 0 ? floorOutlines[index] : null;
    const manualTarget = savedTarget ?? suggestion ?? (targetRoom
      ? { id: `room-${targetRoom.roomType}-user`, polygon: [], scope: "room" as const, room_type: targetRoom.roomType, label: targetRoom.label }
      : savedOutlines[0] ?? floorOutlines[index] ?? { id: "floor-user", polygon: [], scope: "floor" as const, label: "Whole floor" });
    setManualOutlineTarget(manualTarget);
    setOutlineCandidateIndex(index);
    setOutlinePoints((manualTarget.polygon ?? []).map(([x, y]) => [x, y]));
    setOutlineMode(true);
    setAddOutlinePointMode(false);
    setScaleMode(false);
    setMeasureMode(false);
    setOutlineError(null);
  }

  function planPoint(event: React.MouseEvent<SVGSVGElement> | React.PointerEvent<SVGSVGElement>): [number, number] | null {
    if (!imageSize) return null;
    const transform = event.currentTarget.getScreenCTM();
    if (!transform) return null;
    const point = event.currentTarget.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const mapped = point.matrixTransform(transform.inverse());
    return [Math.max(0, Math.min(imageSize.width, mapped.x)), Math.max(0, Math.min(imageSize.height, mapped.y))];
  }

  function handlePlanClick(event: React.MouseEvent<SVGSVGElement>) {
    if (correctionMode && correctionAddMode) {
      const point = planPoint(event);
      if (point && imageSize) {
        if (addCategory === "room" && correctionAddPoints.length === 0) {
          setCorrectionAddPoints([point]);
          return;
        }
        const corners = addCategory === "room" ? [correctionAddPoints[0], point] : [point, point];
        const radius = 14;
        const x0 = addCategory === "room" ? Math.min(corners[0][0], corners[1][0]) : point[0] - radius;
        const y0 = addCategory === "room" ? Math.min(corners[0][1], corners[1][1]) : point[1] - radius;
        const x1 = addCategory === "room" ? Math.max(corners[0][0], corners[1][0]) : point[0] + radius;
        const y1 = addCategory === "room" ? Math.max(corners[0][1], corners[1][1]) : point[1] + radius;
        void saveCorrection("add", undefined, {
          id: "new-user-item", category: addCategory, cls: addClass,
          bbox: [Math.max(0, x0), Math.max(0, y0), Math.min(imageSize.width, Math.max(x1, x0 + 1)), Math.min(imageSize.height, Math.max(y1, y0 + 1))],
        });
      }
      return;
    }
    if (correctionMode) return;
    if (outlineMode) {
      if (addOutlinePointMode) {
        const point = planPoint(event);
        if (point) setOutlinePoints(current => [...current, point]);
        setAddOutlinePointMode(false);
      }
      return;
    }
    markScalePoint(event);
  }

  async function saveFloorOutline(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (outlinePoints.length < 3 || !planId) return;
    setOutlineSaving(true);
    setOutlineError(null);
    try {
      const target = manualOutlineTarget ?? floorOutlines[outlineCandidateIndex];
      const outline = {
        id: target?.id ?? "floor-user",
        polygon: outlinePoints,
        scope: target?.scope ?? "floor",
        room_type: target?.room_type ?? null,
        label: target?.label ?? "Whole floor",
      };
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/outline`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ outlines: [outline] }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save the floor outline");
      setSavedOutlines(result.saved ?? [outline]);
      setOutlineMode(false);
      const estimatedArea = areaM2(outlinePoints);
      if (pendingAreaQuestion && estimatedArea !== null) {
        setMessages(current => [...current, { role: "assistant", text: `Estimated ${outline.label.toLowerCase()} area: ${estimatedArea.toFixed(1)} m². This uses the scale and the outline you accepted; check the highlighted boundary before relying on it.` }]);
        setPendingAreaQuestion(null);
      }
    } catch (saveError) {
      setOutlineError(saveError instanceof Error ? saveError.message : "Could not save the floor outline");
    } finally { setOutlineSaving(false); }
  }

  async function removeSavedOutline() {
    if (!planId || savedOutlines.length === 0) return;
    setOutlineRemoving(true);
    setOutlineError(null);
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/outline`, { method: "DELETE" });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not remove the outline");
      setSavedOutlines([]);
      setOutlinePoints([]);
      setOutlineMode(false);
      setPendingAreaQuestion(null);
    } catch (removeError) {
      setOutlineError(removeError instanceof Error ? removeError.message : "Could not remove the outline");
    } finally { setOutlineRemoving(false); }
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
    let pixelPoint: [number, number] = [Math.max(0, Math.min(imageSize.width, imagePoint.x)), Math.max(0, Math.min(imageSize.height, imagePoint.y))];
    if (scaleMode) {
      const next = [...scalePoints.slice(0, 1), pixelPoint];
      setScalePoints(next);
      setScaleError(null);
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
      setScaleMode(false);
      setScalePoints(result.points);
      setScaleDistance(String(result.distance));
      setScaleUnit(result.unit);
      if (pendingAreaQuestion) startOutlineReview(pendingAreaQuestion);
    } catch (saveError) {
      setScaleError(saveError instanceof Error ? saveError.message : "Could not save scale");
    } finally { setScaleSaving(false); }
  }

  const pxPerUnit = scale ? Math.hypot(scale.points[1][0] - scale.points[0][0], scale.points[1][1] - scale.points[0][1]) / scale.distance : null;
  const measuredDistance = scale && pxPerUnit && measurePoints.length === 2
    ? Math.hypot(measurePoints[1][0] - measurePoints[0][0], measurePoints[1][1] - measurePoints[0][1]) / pxPerUnit
    : null;
  const metersPerPixel = scale
    ? scale.distance * ({ m: 1, cm: 0.01, ft: 0.3048, in: 0.0254 }[scale.unit]) /
      Math.hypot(scale.points[1][0] - scale.points[0][0], scale.points[1][1] - scale.points[0][1])
    : null;
  const areaM2 = (polygon: Array<[number, number]>) => {
    if (!metersPerPixel || polygon.length < 3) return null;
    const pixelArea = Math.abs(polygon.reduce((sum, [x, y], index) => {
      const [nextX, nextY] = polygon[(index + 1) % polygon.length];
      return sum + x * nextY - nextX * y;
    }, 0)) / 2;
    return pixelArea * metersPerPixel * metersPerPixel;
  };
  const acceptedArea = savedOutlines.reduce((sum, outline) => sum + (areaM2(outline.polygon) ?? 0), 0);
  const selectedEvidence = evidenceItems.find(item => item.id === selectedEvidenceId) ?? null;

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
          {ready && !correctionMode && <button className="toolbar-button" type="button" onClick={startCorrectionReview}>Review detections</button>}
          {correctionMode && <button className="toolbar-button" type="button" onClick={() => { setCorrectionMode(false); setCorrectionAddMode(false); setSelectedEvidenceId(null); }}>Done reviewing</button>}
          {ready && scaleLoaded && !outlineMode && <button className="toolbar-button" type="button" onClick={() => startOutlineReview()}>{savedOutlines.length ? `Edit ${savedOutlines[0].label?.toLowerCase() ?? "outline"}` : floorOutlines.length ? "Review detected outline" : "Draw outline"}</button>}
          {outlineMode && <button className="toolbar-button" type="button" onClick={() => setOutlineMode(false)}>Cancel outline</button>}
          <span className="connection-status"><i /> {error ?? (imageStatus === "uploaded" ? analysisStatus : "Loading saved plan…")}</span>
          <Link className="toolbar-button" href="/">Upload another plan</Link>
        </div>
      </header>
      <div className="work-area">
        <section className={`drawing-panel${scaleMode ? " is-calibrating" : ""}`} aria-label="Plan drawing">
          {error ? <div className="upload-empty-state"><h1>Could not load this plan</h1><p role="alert">{error}</p><Link className="toolbar-button" href="/">Back to upload</Link></div> : (
            imageUrl ? <div className="image-frame"><img className="plan-image" src={imageUrl} alt={`Floor plan: ${plan?.metadata?.filename ?? "saved plan"}`} />{imageSize && <svg className={`citation-overlay${scaleMode || measureMode || outlineMode || measurePoints.length === 2 ? " scale-selecting" : ""}${correctionMode ? " correction-active" : ""}`} viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="xMidYMid meet" onClick={handlePlanClick} onPointerMove={event => { if (draggingOutlinePoint !== null) { const point = planPoint(event); if (point) setOutlinePoints(current => current.map((value, index) => index === draggingOutlinePoint ? point : value)); } else if (draggingMeasurePoint !== null) { const point = planPoint(event); if (point) setMeasurePoints(current => current.map((value, index) => index === draggingMeasurePoint ? point : value)); } }} onPointerUp={() => { setDraggingOutlinePoint(null); setDraggingMeasurePoint(null); }} aria-label={correctionMode ? "Select a detected item to correct" : outlineMode ? "Review and edit suggested floor boundary" : scaleMode || measureMode ? "Select two points" : `Highlighted ${citations.length} supporting items`}>
              {correctionMode && evidenceItems.map(item => {
                const [x0, y0, x1, y1] = item.bbox;
                const points = item.polygon?.map(([x, y]) => `${x},${y}`).join(" ");
                return <g className={`correction-mark${selectedEvidenceId === item.id ? " selected" : ""}${item.user_verified ? " user-corrected" : ""}`} key={item.id} onClick={event => { event.stopPropagation(); beginCorrection(item); }} role="button" aria-label={`${item.category} ${item.cls}`}>
                  {points ? <polygon points={points} /> : <rect x={x0} y={y0} width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)} />}
                </g>;
              })}
              {citations.map((item, index) => {
                const [x0, y0, x1, y1] = item.bbox;
                const points = item.polygon?.map(([x, y]) => `${x},${y}`).join(" ");
                return <g className={`citation-mark${item.user_verified ? " user-corrected" : ""}`} key={`${item.id}-${index}`}>
                  {points ? <polygon points={points} /> : <rect x={x0} y={y0} width={Math.max(1, x1 - x0)} height={Math.max(1, y1 - y0)} />}
                  <circle cx={x0} cy={y0} r="18" /><text x={x0} y={y0}>{index + 1}</text>
                </g>;
              })}
              {scaleMode && scalePoints.length > 0 && <g className="scale-mark"><line x1={scalePoints[0][0]} y1={scalePoints[0][1]} x2={(scalePoints[1] ?? scalePoints[0])[0]} y2={(scalePoints[1] ?? scalePoints[0])[1]} />{scalePoints.map(([x, y], index) => <g key={index}><circle cx={x} cy={y} r="10"/><text x={x} y={y}>{index + 1}</text></g>)}</g>}
              {measurePoints.length > 0 && <g className="scale-mark measurement-mark"><line x1={measurePoints[0][0]} y1={measurePoints[0][1]} x2={(measurePoints[1] ?? measurePoints[0])[0]} y2={(measurePoints[1] ?? measurePoints[0])[1]} />{measurePoints.map(([x, y], index) => <g className="measurement-handle" key={index} onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setDraggingMeasurePoint(index); }} onPointerUp={event => { event.stopPropagation(); setDraggingMeasurePoint(null); }} onClick={event => event.stopPropagation()}><circle cx={x} cy={y} r="10"/><text x={x} y={y}>{index + 1}</text></g>)}</g>}
              {measuredDistance !== null && measurePoints.length === 2 && <g className="measurement-label" pointerEvents="none"><rect x={(measurePoints[0][0] + measurePoints[1][0]) / 2 - Math.max(52, `${measuredDistance.toFixed(2)} ${scale?.unit}`.length * 7)} y={(measurePoints[0][1] + measurePoints[1][1]) / 2 - 15} width={Math.max(104, `${measuredDistance.toFixed(2)} ${scale?.unit}`.length * 14)} height="30" rx="7"/><text x={(measurePoints[0][0] + measurePoints[1][0]) / 2} y={(measurePoints[0][1] + measurePoints[1][1]) / 2 + 5}>{measuredDistance.toFixed(2)} {scale?.unit}</text></g>}
              {(outlineMode ? outlinePoints : savedOutlines[0]?.polygon ?? []).length >= 3 && <g className="floor-outline-mark"><polygon points={(outlineMode ? outlinePoints : savedOutlines[0].polygon).map(([x, y]) => `${x},${y}`).join(" ")} />{outlineMode && outlinePoints.map(([x, y], index) => <circle key={index} cx={x} cy={y} r="11" onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setDraggingOutlinePoint(index); }} onPointerUp={event => { event.stopPropagation(); setDraggingOutlinePoint(null); }} onClick={event => event.stopPropagation()} />)}</g>}
            </svg>}{(scaleMode || measureMode || outlineMode) && <div className="scale-hint">{outlineMode ? (addOutlinePointMode ? "Click the plan to add a boundary point" : "Drag boundary points to edit the suggested outline") : scaleMode ? (scalePoints.length === 0 ? "Step 1 of 2 · Click one end of a printed dimension" : scalePoints.length === 1 ? "Step 2 of 2 · Click the other end of that dimension" : "Step complete · Enter the printed distance below") : (measurePoints.length === 0 ? "Click a start point to measure" : "Click an end point to measure")}</div>}</div> : <div className="loading-state" role="status"><span className="loading-spinner" /><h1>Loading your floor plan</h1><p>Getting your image ready…</p></div>
          )}
        </section>
        <aside className="chat-panel" aria-label="Plan questions">
          {scaleMode && scalePoints.length === 2 && <form className="scale-form" onSubmit={saveScale}><div><strong>Enter the printed distance</strong><span>Use the value shown between the points on the plan, then choose its unit.</span></div><input aria-label="Known distance" type="number" min="0.001" step="any" placeholder="Distance" value={scaleDistance} onChange={event => setScaleDistance(event.target.value)} required /><select aria-label="Distance unit" value={scaleUnit} onChange={event => setScaleUnit(event.target.value as ScaleCalibration["unit"])}><option value="m">metres</option><option value="cm">centimetres</option><option value="ft">feet</option><option value="in">inches</option></select><button className="toolbar-button primary-button" disabled={scaleSaving || !scaleDistance}>{scaleSaving ? "Saving…" : "Save scale"}</button>{scaleError && <p role="alert">{scaleError}</p>}</form>}
          {outlineMode && <form className="outline-form" onSubmit={saveFloorOutline}><div className="outline-copy"><strong>{manualOutlineTarget?.scope === "room" ? `Review ${manualOutlineTarget.label?.toLowerCase() ?? "room"} outline` : "Review whole-floor outline"}</strong><span>{manualOutlineTarget?.polygon?.length ? "Suggested from the plan analysis · Drag points to edit, then accept the boundary." : "No matching outline was detected. Click Add point and mark the boundary, then accept it."}</span></div>{floorOutlines.length > 1 && !(manualOutlineTarget?.scope === "room" && !manualOutlineTarget.polygon.length) && <label>Detected outline<select aria-label="Detected outline" value={outlineCandidateIndex} onChange={event => { const index = Number(event.target.value); setOutlineCandidateIndex(index); setManualOutlineTarget(floorOutlines[index]); setOutlinePoints((floorOutlines[index]?.polygon ?? []).map(([x, y]) => [x, y])); }}><option value={0}>{`${floorOutlines[0].label ?? "Outline 1"}${floorOutlines[0].room_type ? ` · ${floorOutlines[0].room_type}` : ""}`}</option>{floorOutlines.slice(1).map((outline, index) => <option key={outline.id} value={index + 1}>{`${outline.label ?? `Outline ${index + 2}`}${outline.room_type ? ` · ${outline.room_type}` : ""}`}</option>)}</select></label>}<div className="outline-actions"><button className="toolbar-button" type="button" onClick={() => setAddOutlinePointMode(true)}>{addOutlinePointMode ? "Click plan to add point" : "Add point"}</button><button className="toolbar-button" type="button" onClick={() => setOutlinePoints(current => current.slice(0, -1))} disabled={outlinePoints.length <= 3}>Remove last</button><button className="toolbar-button primary-button" type="submit" disabled={outlineSaving || outlinePoints.length < 3}>{outlineSaving ? "Saving…" : `Accept ${manualOutlineTarget?.label?.toLowerCase() ?? "outline"}`}</button></div>{scale && areaM2(outlinePoints) !== null && <p>Estimated area: {areaM2(outlinePoints)?.toFixed(1)} m²</p>}{!scale && <p>Set a drawing scale to calculate the area.</p>}{outlineError && <p role="alert">{outlineError}</p>}</form>}
          {!outlineMode && savedOutlines.length > 0 && <div className="area-summary"><strong>Accepted {savedOutlines[0].label?.toLowerCase() ?? "outline"}</strong><span>{acceptedArea > 0 ? `Estimated area ${acceptedArea.toFixed(1)} m²` : "Set a drawing scale to calculate its area"}</span><button className="toolbar-button" type="button" onClick={removeSavedOutline} disabled={outlineRemoving}>{outlineRemoving ? "Removing…" : "Remove outline"}</button>{outlineError && <p role="alert">{outlineError}</p>}</div>}
          {correctionMode && <section className="correction-panel" aria-label="Review detected plan items">
            <div className="correction-heading"><strong>Review detections</strong><span>{correctionEvents.length ? `${correctionEvents.length} saved correction${correctionEvents.length === 1 ? "" : "s"}` : "Model detections are shown on the plan."}</span></div>
            <div className="correction-list">{evidenceItems.filter(item => ["room", "door", "window", "fixture"].includes(item.category)).map(item => <button type="button" className={`correction-row${selectedEvidenceId === item.id ? " active" : ""}`} key={item.id} onClick={() => beginCorrection(item)}><span>{item.cls.replaceAll("_", " ") || item.category}</span><small>{item.category}{item.user_verified ? " · verified" : item.score != null ? ` · ${Math.round(item.score * 100)}%` : ""}</small></button>)}</div>
            {selectedEvidence && <div className="correction-editor"><strong>Correct selected item</strong><label>Kind<select value={editCategory} onChange={event => { const category = event.target.value; setEditCategory(category); setEditClass(itemTypes[category][0]); }}>{Object.keys(itemTypes).map(category => <option key={category} value={category}>{category}</option>)}</select></label><label>Type<select value={editClass} onChange={event => setEditClass(event.target.value)}>{(itemTypes[editCategory] ?? ["other"]).map(type => <option key={type} value={type}>{type.replaceAll("_", " ")}</option>)}</select></label><div className="correction-actions"><button className="toolbar-button primary-button" type="button" disabled={correctionSaving || (selectedEvidence.category === editCategory && selectedEvidence.cls === editClass)} onClick={() => void saveCorrection("update", selectedEvidence.id, { ...selectedEvidence, category: editCategory, cls: editClass })}>{correctionSaving ? "Saving…" : "Save correction"}</button><button className="toolbar-button" type="button" disabled={correctionSaving} onClick={() => void saveCorrection("delete", selectedEvidence.id)}>Remove detection</button></div></div>}
            <div className="correction-editor"><strong>Add a missed item</strong><label>Kind<select value={addCategory} onChange={event => { const category = event.target.value; setAddCategory(category); setAddClass(itemTypes[category]?.[0] ?? "other"); setCorrectionAddPoints([]); }}>{Object.keys(itemTypes).map(category => <option key={category} value={category}>{category}</option>)}</select></label><label>Type<select value={addClass} onChange={event => setAddClass(event.target.value)}>{(itemTypes[addCategory] ?? ["other"]).map(type => <option key={type} value={type}>{type.replaceAll("_", " ")}</option>)}</select></label><button className="toolbar-button" type="button" disabled={correctionSaving} onClick={() => { setCorrectionAddPoints([]); setCorrectionAddMode(true); }}>{correctionAddMode ? addCategory === "room" ? correctionAddPoints.length === 0 ? "Click first room corner" : "Click opposite room corner" : "Click its location on the plan" : addCategory === "room" ? "Mark missing room" : "Place missing item"}</button></div>
            {correctionError && <p className="correction-error" role="alert">{correctionError}</p>}
          </section>}
          <div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div>
          <div className="conversation" aria-live="polite" ref={conversationRef}>{messages.length === 0 && !sending ? <div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>{ready ? "Plan analysis is ready." : "Preparing the saved plan analysis…"}</p><span>Ask about quantities, rooms, and detected items.</span></div> : <>{messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${index}-${message.role}`}><span>{message.role === "user" ? "You" : "Stoa"}</span><p>{message.text}</p>{message.role === "assistant" && index > 0 && <div className="answer-feedback">{message.feedback ? <small>Thanks for the feedback.</small> : <><button type="button" onClick={() => void submitFeedback(index, "helpful")}>Helpful</button><button type="button" onClick={() => void submitFeedback(index, "incorrect")}>Needs correction</button><button type="button" onClick={() => void submitFeedback(index, "unclear")}>Unclear</button></>}</div>}</div>)}{sending && <div className="chat-message assistant thinking-message" role="status"><span>Stoa</span><p><i className="thinking-spinner" aria-hidden="true" />Thinking through the plan…</p></div>}</>}</div>
          {feedbackError && <p className="feedback-error" role="alert">{feedbackError}</p>}
          <form className="composer" onSubmit={submitQuestion}><textarea aria-label="Ask a question about the plan" placeholder="Ask about quantities, rooms, or materials…" value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} disabled={!ready || sending} /><div className="composer-footer"><span>{sending ? "Thinking…" : ready ? "Answers use the analysis saved for this plan" : "Available when analysis finishes"}</span><button className="send-button" type="submit" disabled={!ready || sending || !question.trim()} aria-label="Send question">↑</button></div></form>
        </aside>
      </div>
    </main>
  );
}
