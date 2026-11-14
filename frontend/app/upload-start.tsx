"use client";

import { useRef, useState, type DragEvent } from "react";
import { useRouter } from "next/navigation";

export default function UploadStart() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  const [uploading, setUploading] = useState(false);

  async function startUpload(files?: FileList | File[]) {
    if (!files?.length || uploading) return;
    const selected = Array.from(files);
    const invalid = selected.find(file => !file.type.startsWith("image/") && file.type !== "application/pdf");
    if (invalid) { window.alert(`${invalid.name} is not a supported PDF or image.`); return; }
    setUploading(true);
    const floors: Array<{ id: string; name: string }> = [];
    for (const file of selected) {
      const id = crypto.randomUUID();
      const body = new FormData(); body.set("file", file); body.set("filename", file.name);
      try {
        const response = await fetch(`/api/plans/${id}/upload`, { method: "POST", body });
        if (!response.ok) {
          const result = await response.json().catch(() => ({}));
          throw new Error(result.detail || "Upload failed");
        }
        floors.push({ id, name: file.name });
      } catch (error) {
        window.alert(`${file.name}: ${error instanceof Error ? error.message : "Upload failed"}`);
      }
    }
    if (floors.length) {
      localStorage.setItem("junc-building-floors", JSON.stringify(floors));
      localStorage.setItem("junc-building-chats", "{}");
      router.push(`/plans/${floors[0].id}`);
    }
    setUploading(false);
  }

  function drop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    void startUpload(event.dataTransfer.files);
  }

  return (
    <main className="workspace">
      <input ref={inputRef} className="visually-hidden" type="file" multiple accept="application/pdf,image/png,image/jpeg,image/webp" onChange={event => { void startUpload(event.target.files ?? undefined); event.target.value = ""; }} />
      <header className="toolbar"><div className="plan-title"><span className="plan-title-icon" aria-hidden="true">▤</span><span>Floor plan</span></div><div className="toolbar-actions"><button className="toolbar-button" disabled={uploading} onClick={() => inputRef.current?.click()}>{uploading ? "Uploading…" : "Choose floor plan"}</button></div></header>
      <div className="work-area">
        <section className={`drawing-panel ${dragging ? "is-dragging" : ""}`} aria-label="Plan drawing" onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}>
          <div className="upload-empty-state"><div className="upload-icon" aria-hidden="true">↑</div><h1>Open a floor plan</h1><p>View the drawing, explore detected rooms, and ask questions about this floor.</p><button className="toolbar-button primary-button" disabled={uploading} onClick={() => inputRef.current?.click()}>{uploading ? "Uploading…" : "Choose floor plan"}</button><span>PDF, PNG, JPG, or WebP · you can add other floors later</span></div>
        </section>
        <aside className="chat-panel" aria-label="Plan questions"><div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div><div className="conversation"><div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>Upload a plan to get started.</p></div></div><form className="composer" onSubmit={event => event.preventDefault()}><textarea aria-label="Ask a question about the plan" placeholder="Upload a plan to ask a question" disabled /><div className="composer-footer"><span /><button className="send-button" type="submit" disabled aria-label="Send question">↑</button></div></form></aside>
      </div>
    </main>
  );
}
