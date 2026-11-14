"use client";

import { useRef, useState, type DragEvent } from "react";
import { useRouter } from "next/navigation";

export default function UploadStart() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function startUpload(file?: File) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      router.push(`/plans/${crypto.randomUUID()}?uploadError=${encodeURIComponent("Please choose a PNG or JPG image.")}`);
      return;
    }

    const planId = crypto.randomUUID();
    const body = new FormData();
    body.set("file", file);
    body.set("filename", file.name);
    void fetch(`/api/plans/${planId}/upload`, { method: "POST", body }).then(async response => {
      if (!response.ok) {
        const result = await response.json().catch(() => ({}));
        window.location.replace(`/plans/${planId}?uploadError=${encodeURIComponent(result.detail || "Image upload failed")}`);
      }
    }).catch(() => {
      window.location.replace(`/plans/${planId}?uploadError=${encodeURIComponent("Image upload failed. Check your connection and try again.")}`);
    });
    router.push(`/plans/${planId}`);
  }

  function drop(event: DragEvent<HTMLElement>) {
    event.preventDefault();
    setDragging(false);
    startUpload(event.dataTransfer.files[0]);
  }

  return (
    <main className="workspace">
      <input ref={inputRef} className="visually-hidden" type="file" accept="image/png,image/jpeg,image/webp" onChange={event => startUpload(event.target.files?.[0])} />
      <header className="toolbar"><div className="plan-title"><span className="plan-title-icon" aria-hidden="true">▤</span><span>Untitled plan</span></div><div className="toolbar-actions"><button className="toolbar-button" onClick={() => inputRef.current?.click()}>Upload plan</button></div></header>
      <div className="work-area">
        <section className={`drawing-panel ${dragging ? "is-dragging" : ""}`} aria-label="Plan drawing" onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={drop}>
          <div className="upload-empty-state"><div className="upload-icon" aria-hidden="true">↑</div><h1>Drop a floor plan here</h1><p>View the original drawing and ask questions about quantities and spaces.</p><button className="toolbar-button primary-button" onClick={() => inputRef.current?.click()}>Choose a file</button><span>PNG, JPG, or WebP</span></div>
        </section>
        <aside className="chat-panel" aria-label="Plan questions"><div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div><div className="conversation"><div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>Upload a plan to get started.</p></div></div><form className="composer" onSubmit={event => event.preventDefault()}><textarea aria-label="Ask a question about the plan" placeholder="Upload a plan to ask a question" disabled /><div className="composer-footer"><span /><button className="send-button" type="submit" disabled aria-label="Send question">↑</button></div></form></aside>
      </div>
    </main>
  );
}
