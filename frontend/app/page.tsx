"use client";

import { DragEvent, useEffect, useRef, useState } from "react";
import { createClient, type SupabaseClient } from "@supabase/supabase-js";

// The app's table/function types are managed by SQL migrations rather than generated TS types.
let supabaseClient: SupabaseClient<any> | null = null;

function getSupabaseClient(url: string, anonKey: string) {
  if (!supabaseClient) supabaseClient = createClient(url, anonKey) as SupabaseClient<any>;
  return supabaseClient;
}

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!file || !file.type.startsWith("image/")) {
      setPreview(null);
      return;
    }

    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const selectFile = async (selected?: File) => {
    if (!selected) return;
    setFile(selected);
    setUploadError(null);
    setAnalysisError(null);
    setUploadStatus(null);

    // PDFs remain available for the existing local preview. Images are uploaded to Supabase.
    if (!selected.type.startsWith("image/")) return;

    const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL;
    const supabaseAnonKey = process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;
    const bucket = process.env.NEXT_PUBLIC_SUPABASE_BUCKET || "plans";
    if (!supabaseUrl || !supabaseAnonKey) {
      setUploadError("Supabase is not configured. Add the project URL and anon key to frontend/.env.local.");
      return;
    }

    setUploadStatus("Uploading image…");
    const supabase = getSupabaseClient(supabaseUrl, supabaseAnonKey);
    const safeName = selected.name.replace(/[^a-zA-Z0-9._-]/g, "-");
    const planId = crypto.randomUUID();
    const path = `${planId}/${crypto.randomUUID()}-${safeName}`;
    const { error: recordError } = await supabase.rpc("create_plan_with_image", {
      p_plan_id: planId,
      p_bucket_id: bucket,
      p_object_path: path,
      p_role: "source",
    });
    if (recordError) {
      setUploadStatus(null);
      setUploadError(`Plan record could not be created, so the image was not uploaded: ${recordError.message}`);
      return;
    }

    const { error } = await supabase.storage.from(bucket).upload(path, selected, {
      contentType: selected.type,
      upsert: false,
    });

    if (error) {
      await supabase.rpc("set_plan_image_upload_status", {
        p_plan_id: planId,
        p_object_path: path,
        p_status: "failed",
      });
      setUploadStatus(null);
      setUploadError(`Upload failed: ${error.message}`);
      return;
    }
    const { error: statusError } = await supabase.rpc("set_plan_image_upload_status", {
      p_plan_id: planId,
      p_object_path: path,
      p_status: "uploaded",
    });
    const savedStatus = `Image uploaded to Supabase. Plan ID: ${planId}${statusError ? " (upload status confirmation pending)" : ""}`;
    setUploadStatus(`${savedStatus} Analyzing plan…`);
    try {
      const response = await fetch(`/api/plans/${planId}/analyze`, { method: "POST" });
      const result = await response.json();
      if (!response.ok) {
        throw new Error(result.detail || "Plan analysis failed");
      }
      const counts = Object.entries(result.counts ?? {})
        .map(([name, count]) => `${name.replaceAll("_", " ")}: ${count}`)
        .join(", ");
      setUploadStatus(`Analysis complete. ${counts || "Plan parsed"}${statusError ? " · Upload status pending" : ""} · ID: ${planId}`);
    } catch (error) {
      setUploadStatus(savedStatus);
      setAnalysisError(`Analysis failed: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  };

  const handleDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setIsDragging(false);
    selectFile(event.dataTransfer.files[0]);
  };

  return (
    <main className="workspace">
      <input
        ref={inputRef}
        className="visually-hidden"
        type="file"
        accept="image/png,image/jpeg,application/pdf"
        onChange={event => selectFile(event.target.files?.[0])}
      />

      <header className="toolbar">
        <div className="plan-title" title={file?.name ?? "Untitled plan"}>
          <span className="plan-title-icon" aria-hidden="true">▤</span>
          <span>{file?.name ?? "Untitled plan"}</span>
        </div>
        <div className="toolbar-actions">
          {file && <span className="connection-status"><i /> {uploadError ? "Upload issue" : analysisError ? "Analysis issue" : uploadStatus ?? "Analysis unavailable"}</span>}
          <button className="toolbar-button" onClick={() => inputRef.current?.click()}>
            {file ? "Replace plan" : "Upload plan"}
          </button>
        </div>
      </header>

      <div className="work-area">
        <section
          className={`drawing-panel ${isDragging ? "is-dragging" : ""}`}
          aria-label="Plan drawing"
          onDragOver={event => { event.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
        >
          {preview ? (
            <img className="plan-image" src={preview} alt={`Original floor plan: ${file?.name ?? "uploaded plan"}`} />
          ) : file ? (
            <div className="pdf-preview">
              <div className="pdf-icon" aria-hidden="true">PDF</div>
              <p>Original PDF selected</p>
              <span>{file.name}</span>
            </div>
          ) : (
            <div className="upload-empty-state">
              <div className="upload-icon" aria-hidden="true">↑</div>
              <h1>Drop a floor plan here</h1>
              <p>View the original drawing and ask questions about quantities and spaces.</p>
              <button className="toolbar-button primary-button" onClick={() => inputRef.current?.click()}>
                Choose a file
              </button>
              <span>PNG, JPG, or PDF</span>
            </div>
          )}
        </section>

        <aside className="chat-panel" aria-label="Plan questions">
          <div className="chat-header">
            <div>
              <h1>Ask about this plan</h1>
              <p>Questions, quantities, and room details</p>
            </div>
            {file && <button className="remove-button" onClick={() => { setFile(null); if (inputRef.current) inputRef.current.value = ""; }}>Remove</button>}
          </div>

          <div className="conversation" aria-live="polite">
            <div className="empty-conversation">
              <div className="chat-mark" aria-hidden="true">↗</div>
              <p>{file ? "Your answers will appear here." : "Upload a plan to get started."}</p>
              {file && <span>Plan analysis isn’t connected yet.</span>}
              {uploadError && <span role="alert">{uploadError}</span>}
              {analysisError && <span role="alert">{analysisError}</span>}
              {uploadStatus && <span role="status">{uploadStatus}</span>}
            </div>
          </div>

          <form className="composer" onSubmit={event => event.preventDefault()}>
            <textarea
              aria-label="Ask a question about the plan"
              placeholder={file ? "Ask about quantities, rooms, or materials…" : "Upload a plan to ask a question"}
              disabled
            />
            <div className="composer-footer">
              <span>{file ? "Analysis will be available when connected" : ""}</span>
              <button className="send-button" type="submit" disabled aria-label="Send question">↑</button>
            </div>
          </form>
        </aside>
      </div>
    </main>
  );
}
