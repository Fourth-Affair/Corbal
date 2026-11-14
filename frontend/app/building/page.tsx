"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type Floor = { id: string; name: string };
type Evidence = { id: string; bbox: [number, number, number, number]; category?: string; cls?: string; user_verified?: boolean };
type FloorAnswer = { id: string; name: string; answer?: string | number | null; unit?: string | null; width: number; height: number; citations: Evidence[] };
type Message = { role: "user" | "assistant"; text: string; floors?: FloorAnswer[] };

export default function BuildingPage() {
  const [floors, setFloors] = useState<Floor[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [question, setQuestion] = useState("");
  const [chats, setChats] = useState<Record<string, Message[]>>({});
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("");
  const [mode, setMode] = useState<"building" | "floor">("building");
  const chatKey = mode === "building" ? "building" : `floor:${selected[0] ?? "none"}`;
  const messages = chats[chatKey] ?? [];

  useEffect(() => {
    const saved = JSON.parse(localStorage.getItem("junc-building-floors") || "[]") as Floor[];
    setFloors(saved);
    setSelected(saved.map(floor => floor.id));
    setChats(JSON.parse(localStorage.getItem("junc-building-chats") || "{}") as Record<string, Message[]>);
  }, []);

  function save(next: Floor[]) {
    setFloors(next);
    localStorage.setItem("junc-building-floors", JSON.stringify(next));
  }
  function appendMessage(key: string, message: Message) {
    setChats(current => {
      const next = { ...current, [key]: [...(current[key] ?? []), message] };
      localStorage.setItem("junc-building-chats", JSON.stringify(next));
      return next;
    });
  }
  function toggle(id: string) {
    setSelected(current => current.includes(id) ? current.filter(value => value !== id) : [...current, id]);
  }

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    const text = question.trim();
    const chosen = mode === "floor" ? selected.slice(0, 1) : selected;
    if (!text || !chosen.length || busy) return;
    const targetChatKey = mode === "building" ? "building" : `floor:${chosen[0]}`;
    const chosenFloors = floors.filter(floor => chosen.includes(floor.id));
    setQuestion(""); setBusy(true); setStatus("Preparing selected floors…");
    appendMessage(targetChatKey, { role: "user", text });
    try {
      const analyzed: Record<string, { width: number; height: number }> = {};
      for (const floor of chosenFloors) {
        const analysis = await fetch(`/api/plans/${encodeURIComponent(floor.id)}/analyze`, { method: "POST" });
        if (!analysis.ok) {
          const body = await analysis.json().catch(() => ({}));
          throw new Error(body.detail || `Could not analyze ${floor.name}`);
        }
        const graph = await analysis.json();
        analyzed[floor.id] = { width: graph.width, height: graph.height };
      }
      setStatus("Checking the plans…");
      const response = mode === "building"
        ? await fetch("/api/buildings/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text, floors: chosenFloors }) })
        : await fetch(`/api/plans/${encodeURIComponent(chosen[0])}/ask`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question: text }) });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not answer that question");
      const answer = result.answer === null ? result.explanation : `${result.answer}${result.unit ? ` ${result.unit}` : ""}. ${result.explanation}`;
      const visuals: FloorAnswer[] = mode === "building"
        ? (result.floors ?? []) as FloorAnswer[]
        : [{ id: chosen[0], name: chosenFloors[0]?.name ?? "Floor", answer: result.answer, unit: result.unit,
          width: analyzed[chosen[0]]?.width ?? 1000, height: analyzed[chosen[0]]?.height ?? 1000, citations: result.citations ?? [] }];
      appendMessage(targetChatKey, { role: "assistant", text: answer, floors: visuals });
    } catch (error) {
      appendMessage(targetChatKey, { role: "assistant", text: error instanceof Error ? error.message : "Could not answer that question" });
    } finally { setBusy(false); setStatus(""); }
  }

  return <main className="workspace building-workspace">
    <header className="toolbar"><Link className="plan-title building-home-link" href="/"><span className="plan-title-icon">▤</span><span>Building workspace</span></Link><span className="building-floor-count">{floors.length} {floors.length === 1 ? "floor" : "floors"}</span></header>
    <div className="building-layout">
      <aside className="floor-sidebar">
        <div className="floor-sidebar-heading"><div><strong>Floors</strong><span>Select the plans to include</span></div><label className="toolbar-button floor-add">+ Add<input type="file" multiple accept="application/pdf,image/png,image/jpeg,image/webp" onChange={async event => {
          const files = Array.from(event.target.files ?? []); event.target.value = ""; if (!files.length || busy) return;
          setBusy(true); setStatus(`Uploading ${files.length} floor${files.length === 1 ? "" : "s"}…`);
          const added: Floor[] = [];
          for (const file of files) { const id = crypto.randomUUID(); const body = new FormData(); body.set("file", file); body.set("filename", file.name); try { const response = await fetch(`/api/plans/${id}/upload`, { method: "POST", body }); if (!response.ok) throw new Error("Upload failed"); added.push({ id, name: file.name }); } catch { appendMessage(chatKey, { role: "assistant", text: `Could not upload ${file.name}. Please try again.` }); } }
          const next = [...floors, ...added]; save(next); setSelected(current => [...new Set([...current, ...added.map(floor => floor.id)])]); setBusy(false); setStatus("");
        }} /></label></div>
        <div className="floor-list">{floors.map((floor, index) => <div className={`floor-row${selected.includes(floor.id) ? " selected" : ""}`} key={floor.id}>
          <input type="checkbox" checked={selected.includes(floor.id)} onChange={() => toggle(floor.id)} aria-label={`Include ${floor.name} in building question`} />
          <button className="floor-name" onClick={() => { setSelected([floor.id]); setMode("floor"); }}><span className="floor-index">{index + 1}</span><span title={floor.name}>{floor.name}</span></button>
          <Link className="floor-open" href={`/plans/${floor.id}`} title="Open full plan editor">↗</Link>
          <button className="floor-remove" title={`Remove ${floor.name}`} onClick={() => { const next = floors.filter(item => item.id !== floor.id); save(next); setSelected(current => current.filter(id => id !== floor.id)); }}>×</button>
        </div>)}</div>
        {!floors.length && <p className="no-floors">Add two or more plans to ask building-wide questions.</p>}
      </aside>
      <section className="building-chat">
        <header className="building-chat-header"><div><h1>{mode === "building" ? "Ask about the building" : `Ask about ${floors.find(floor => floor.id === selected[0])?.name ?? "this floor"}`}</h1><p>{mode === "building" ? "Answers use the selected floors and show a floor-by-floor breakdown." : "This question uses only the selected floor."}</p></div>
          <div className="question-mode"><button className={mode === "building" ? "active" : ""} onClick={() => setMode("building")}>Whole building</button><button className={mode === "floor" ? "active" : ""} disabled={!floors.length} onClick={() => { if (selected.length !== 1) setSelected(selected.slice(0, 1)); setMode("floor"); }}>One floor</button></div>
        </header>
        <div className="building-conversation">{!messages.length ? <div className="building-empty"><div className="chat-mark">↗</div><strong>{floors.length ? "What would you like to know?" : "Upload floor plans to get started"}</strong><span>{floors.length ? "Select floors on the left, then ask a question." : "Use Add to upload several floor plans together."}</span><div className="building-examples"><button onClick={() => setQuestion("How many toilets are there?")}>How many toilets?</button><button onClick={() => setQuestion("How many bedrooms are there?")}>How many bedrooms?</button></div></div> : messages.map((message, index) => <div key={index} className={`building-message ${message.role}`}><span>{message.role === "user" ? "You" : "Building assistant"}</span><p>{message.text}</p>{message.floors?.length ? <div className="building-evidence-grid">{message.floors.map((floor, floorIndex) => <Link className="building-evidence-card" key={`${floor.id}-${floorIndex}`} href={`/plans/${floor.id}`}><div className="building-evidence-title"><strong>{floor.name}</strong><span>{floor.answer == null ? "See floor" : `${floor.answer}${floor.unit ? ` ${floor.unit}` : ""}`}</span></div><div className="building-evidence-image"><img src={`/api/plans/${floor.id}/image`} alt={`Plan evidence for ${floor.name}`} /><svg viewBox={`0 0 ${floor.width} ${floor.height}`} preserveAspectRatio="xMidYMid meet" aria-hidden="true">{floor.citations.map((item, itemIndex) => { const [x0, y0, x1, y1] = item.bbox; return <g key={`${item.id}-${itemIndex}`}><rect x={x0} y={y0} width={Math.max(x1 - x0, 1)} height={Math.max(y1 - y0, 1)} /><circle cx={(x0 + x1) / 2} cy={(y0 + y1) / 2} r={Math.max(floor.width, floor.height) / 85} /><text x={(x0 + x1) / 2} y={(y0 + y1) / 2}>{itemIndex + 1}</text></g>; })}</svg></div><span className="building-evidence-caption">{floor.citations.length ? `${floor.citations.length} highlighted item${floor.citations.length === 1 ? "" : "s"} · open floor` : "Open floor plan"}</span></Link>)}</div> : null}</div>)}{status && <div className="building-status">{status}</div>}</div>
        <form className="composer building-composer" onSubmit={ask}><textarea value={question} onChange={event => setQuestion(event.target.value)} placeholder={mode === "building" ? "Ask about the selected floors…" : "Ask about this floor…"} disabled={!floors.length || busy} /><div className="composer-footer"><span>{mode === "building" ? `${selected.length} floors selected` : "Single floor"}</span><button className="send-button" type="submit" disabled={!question.trim() || !selected.length || busy} aria-label="Send question">↑</button></div></form>
      </section>
    </div>
  </main>;
}
