"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";

type PlanRecord = {
  id: string;
  metadata?: { filename?: string };
  plan_images?: Array<{ upload_status?: string; role?: string }>;
};
type BuildingFloor = { id: string; name: string };

type ChatMessage = { role: "user" | "assistant"; text: string; originalText?: string; evidence_ids?: string[]; evidence?: PlanCitation[]; confidence?: string; feedback?: string; evidenceDecision?: "trusted" | "rejected" };
type PlanCitation = {
  id: string;
  category?: string;
  cls?: string | null;
  text?: string | null;
  bbox: [number, number, number, number];
  polygon?: Array<[number, number]> | null;
  raw?: string | null;
  score?: number | null;
  user_verified?: boolean;
};
type EvidenceItem = PlanCitation & { category: string; cls: string; text?: string | null; score?: number | null; raw?: string | null; user_verified?: boolean };
type EvidenceDrag = { id: string; mode: "move" | "resize"; start: [number, number]; bbox: EvidenceItem["bbox"]; polygon: EvidenceItem["polygon"]; moved: boolean };
type DrawingReference = { bbox: [number, number, number, number]; label: string; item_id?: string };
type RoomEvidence = { source: string; type: string; weight: number; detail: string; item_id?: string | null };
type RoomShape = {
  id: string;
  type: string;
  // strong: a readable label or decisive fixtures; weak: only weak evidence or a conflict; unknown: no evidence
  tier: "strong" | "weak" | "unknown";
  purposes: string[];
  evidence: RoomEvidence[];
  conflicts: string[];
  name?: string | null;
  polygon: Array<[number, number]>;
  bbox: [number, number, number, number];
  area_m2?: number | null;
  labeled_area_m2?: number | null;
  area_m2_range?: [number, number] | null; // rough, from door / fixture sizes
  size_check?: string | null;
};
type WallSegment = {
  id: string;
  p0: [number, number];
  p1: [number, number];
  length_px: number;
  thickness_px: number;
  exterior: boolean;
  length_m?: number | null;
  thickness_m?: number | null;
};
type PlanGeometry = {
  width: number;
  height: number;
  model: string;
  rooms: RoomShape[];
  wall_segments: WallSegment[];
  // the plan's own scale: printed dimension figures (with the segments they label) or areas written in the rooms
  scale?: {
    m_per_px: number;
    source: "calibration" | "written_areas" | "dimensions" | "symbols";
    warning?: string | null;
    dimensions?: Array<{ text: string; value_mm: number; p0: [number, number]; p1: [number, number] }>;
  } | null;
};

function roomWhy(room: RoomShape): string {
  const reasons = room.evidence.filter(e => e.type === room.type && e.weight > 0).sort((a, b) => b.weight - a.weight).slice(0, 3).map(e => e.detail);
  const conflict = room.conflicts.length ? ` · conflicting evidence: ${room.conflicts.join(", ")}` : "";
  const size = room.size_check ? ` · size: ${room.size_check}` : "";
  const written = room.labeled_area_m2 != null ? " · area as written on the drawing" : room.area_m2 != null ? " · area measured from the outline" : "";
  const rough = room.area_m2 == null && room.labeled_area_m2 == null && room.area_m2_range ? " · area is a rough range from door/fixture sizes; set the scale for exact sizes" : "";
  return (reasons.length ? `${reasons.join("; ")}` : "no evidence") + conflict + size + written + rough;
}

// One colour per purpose; weak rooms are dashed, unknown spaces grey and dashed
const ROOM_COLOURS: Record<string, string> = {
  bedroom: "#5b8def", living: "#e0a43b", kitchen: "#e0613b", bathroom: "#3bb3c9", wc: "#3b8fc9", sauna: "#c9733b",
  balcony: "#6dbb5f", terrace: "#6dbb5f", entry: "#9b8bd6", closet: "#b07fc0", storage: "#a0a0a0", utility: "#5fb3a0",
  dining: "#d6b13b", study: "#7f9bc0", technical: "#8a8a8a", garage: "#8a8a8a", stairs: "#b0906a", basement: "#8a8a8a",
};

function roomCaption(room: RoomShape): string {
  // the area printed on the drawing wins over one measured from the outline
  const area = room.labeled_area_m2 ?? room.area_m2;
  const what = room.purposes.length > 1 ? room.purposes.join(" + ") : room.type;
  if (area != null) return `${what} · ${area.toFixed(1)} m²`;
  if (room.area_m2_range) return `${what} · ~${Math.round(room.area_m2_range[0])}–${Math.round(room.area_m2_range[1])} m²`;
  return what;
}

function pointInPolygon(x: number, y: number, polygon: Array<[number, number]>): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

type ScaleCalibration = { points: [[number, number], [number, number]]; distance: number; unit: "m" | "cm" | "ft" | "in" };
type FloorOutline = { id: string; polygon: Array<[number, number]>; score?: number | null; scope?: "floor" | "room"; room_type?: string | null; label?: string; decision?: "accepted" | "rejected" | null };
const itemTypes: Record<string, string[]> = {
  room: ["bedroom", "bathroom", "wc", "sauna", "kitchen", "living", "balcony", "entry", "storage", "garage", "other"],
  fixture: ["toilet", "sink", "double_sink", "shower", "bathtub", "sauna_stove", "sauna_bench", "other"],
  door: ["door"], window: ["window"],
};

function itemLabel(item: Pick<EvidenceItem, "category" | "cls" | "text">) {
  const type = item.cls?.replaceAll("_", " ").trim();
  if (item.category === "room") return type || item.text || "Room";
  return type || item.category;
}

function evidenceLabel(item: EvidenceItem, items: EvidenceItem[]) {
  const base = itemLabel(item);
  const peers = items.filter(candidate => itemLabel(candidate) === base);
  return peers.length > 1 ? `${base}-${peers.indexOf(item) + 1}` : base;
}

function simplifyBoundary(points: Array<[number, number]>, maxPoints = 18): Array<[number, number]> {
  if (points.length <= maxPoints) return points;
  const distanceToSegment = (point: [number, number], start: [number, number], end: [number, number]) => {
    const dx = end[0] - start[0], dy = end[1] - start[1];
    const lengthSquared = dx * dx + dy * dy;
    const t = lengthSquared ? Math.max(0, Math.min(1, ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / lengthSquared)) : 0;
    return Math.hypot(point[0] - (start[0] + t * dx), point[1] - (start[1] + t * dy));
  };
  const simplifyOpen = (path: Array<[number, number]>, tolerance: number): Array<[number, number]> => {
    if (path.length <= 2) return path;
    let farthest = 0, farthestIndex = 0;
    for (let index = 1; index < path.length - 1; index++) {
      const distance = distanceToSegment(path[index], path[0], path[path.length - 1]);
      if (distance > farthest) { farthest = distance; farthestIndex = index; }
    }
    if (farthest <= tolerance) return [path[0], path[path.length - 1]];
    return [...simplifyOpen(path.slice(0, farthestIndex + 1), tolerance).slice(0, -1), ...simplifyOpen(path.slice(farthestIndex), tolerance)];
  };
  let split = 1, longest = 0;
  for (let index = 1; index < points.length; index++) {
    const distance = Math.hypot(points[index][0] - points[0][0], points[index][1] - points[0][1]);
    if (distance > longest) { longest = distance; split = index; }
  }
  const simplifyClosed = (tolerance: number) => {
    const first = simplifyOpen(points.slice(0, split + 1), tolerance);
    const second = simplifyOpen([...points.slice(split), points[0]], tolerance);
    return [...first.slice(0, -1), ...second.slice(0, -1)];
  };
  let tolerance = 2, result = simplifyClosed(tolerance);
  while (result.length > maxPoints && tolerance < longest) {
    tolerance *= 1.5;
    result = simplifyClosed(tolerance);
  }
  if (result.length <= maxPoints) return result;
  const stride = result.length / maxPoints;
  return Array.from({ length: maxPoints }, (_, index) => result[Math.floor(index * stride)]);
}

function outlineBounds(outline: FloorOutline): EvidenceItem["bbox"] {
  const xs = outline.polygon.map(([x]) => x);
  const ys = outline.polygon.map(([, y]) => y);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function roomBoundaryLinks(items: EvidenceItem[], outlines: FloorOutline[]) {
  const links: Record<string, string> = {};
  for (const item of items) {
    if (item.category !== "room" || item.raw === "ocr") continue;
    const typed = outlines.filter(outline => outline.scope === "room" && outline.room_type === item.cls);
    const candidates = typed.length ? typed : outlines.filter(outline => outline.scope === "room");
    const [ax0, ay0, ax1, ay1] = item.bbox;
    const best = candidates.map(outline => {
      const [bx0, by0, bx1, by1] = outlineBounds(outline);
      const overlap = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)) * Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
      const union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - overlap;
      return { outline, score: union > 0 ? overlap / union : 0 };
    }).sort((a, b) => b.score - a.score)[0];
    if (best && best.score > 0.25) links[item.id] = best.outline.id;
  }
  return links;
}

function linkRoomBoundaries(items: EvidenceItem[], outlines: FloorOutline[]) {
  const links = roomBoundaryLinks(items, outlines);
  return items.map(item => {
    const matched = outlines.find(outline => outline.id === links[item.id]);
    return matched
      ? { ...item, bbox: outlineBounds(matched), polygon: matched.polygon }
      : item;
  });
}

type CachedFloorView = {
  plan: PlanRecord;
  imageBlob: Blob;
  scale: ScaleCalibration | null;
  detectedScaleAccepted: boolean;
  imageSize: { width: number; height: number };
  evidenceItems: EvidenceItem[];
  floorOutlines: FloorOutline[];
  savedOutlines: FloorOutline[];
  correctionEvents: Array<{ action: string }>;
  geometry?: PlanGeometry | null;
};
const floorViewCache = new Map<string, CachedFloorView>();
const FLOOR_VIEW_CACHE_MAX_ITEMS = 4;
const FLOOR_VIEW_CACHE_MAX_BYTES = 64 * 1024 * 1024;

export default function PlanPage({ params }: { params: Promise<{ planId: string }> }) {
  const [planId, setPlanId] = useState("");
  const [plan, setPlan] = useState<PlanRecord | null>(null);
  const [buildingFloors, setBuildingFloors] = useState<BuildingFloor[]>([]);
  const [addingFloors, setAddingFloors] = useState(false);
  const addFloorsInput = useRef<HTMLInputElement>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [analysisStatus, setAnalysisStatus] = useState("Preparing your floor plan…");
  const [ready, setReady] = useState(false);
  const [question, setQuestion] = useState("");
  const [drawingReferences, setDrawingReferences] = useState<DrawingReference[]>([]);
  const [referenceMode, setReferenceMode] = useState(false);
  const [referenceDragStart, setReferenceDragStart] = useState<[number, number] | null>(null);
  const [referenceDragEnd, setReferenceDragEnd] = useState<[number, number] | null>(null);
  const [sending, setSending] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [imageSize, setImageSize] = useState<{ width: number; height: number } | null>(null);
  const [citations, setCitations] = useState<PlanCitation[]>([]);
  const [scale, setScale] = useState<ScaleCalibration | null>(null);
  const [detectedScaleAccepted, setDetectedScaleAccepted] = useState(false);
  const [detectedScaleSaving, setDetectedScaleSaving] = useState(false);
  const [scaleLoaded, setScaleLoaded] = useState(false);
  const [scaleMode, setScaleMode] = useState(false);
  const [measureMode, setMeasureMode] = useState(false);
  const [measurementRequested, setMeasurementRequested] = useState(false);
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
  const [editingOutlineId, setEditingOutlineId] = useState<string | null>(null);
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
  const [draggingEvidence, setDraggingEvidence] = useState<EvidenceDrag | null>(null);
  const [editCategory, setEditCategory] = useState("room");
  const [editClass, setEditClass] = useState("bedroom");
  const [addCategory, setAddCategory] = useState("room");
  const [addClass, setAddClass] = useState("bedroom");
  const [correctionSaving, setCorrectionSaving] = useState(false);
  const [correctionError, setCorrectionError] = useState<string | null>(null);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [geometry, setGeometry] = useState<PlanGeometry | null>(null);
  const [geometryError, setGeometryError] = useState<string | null>(null);
  const [showRooms, setShowRooms] = useState(false);
  const [showUncertain, setShowUncertain] = useState(false);
  const [showWalls, setShowWalls] = useState(false);
  const [animationStage, setAnimationStage] = useState<number | null>(null);
  const [animationRun, setAnimationRun] = useState(0);
  const animationStarted = useRef(false);
  const conversationRef = useRef<HTMLDivElement>(null);
  const roomBoundaryLinkRef = useRef<Record<string, string>>({});
  const evidenceItemsRef = useRef<EvidenceItem[]>([]);

  async function addFloorPlans(files: FileList | null) {
    const selected = Array.from(files ?? []);
    if (!selected.length || addingFloors) return;
    if (selected.some(file => !file.type.startsWith("image/") && file.type !== "application/pdf")) {
      window.alert("Choose PDF or image floor plans.");
      return;
    }
    setAddingFloors(true);
    const added: BuildingFloor[] = [];
    for (const file of selected) {
      const id = crypto.randomUUID();
      const body = new FormData();
      body.set("file", file);
      body.set("filename", file.name);
      try {
        const response = await fetch(`/api/plans/${id}/upload`, { method: "POST", body });
        if (!response.ok) {
          const result = await response.json().catch(() => ({}));
          throw new Error(result.detail || "Upload failed");
        }
        added.push({ id, name: file.name });
      } catch (uploadError) {
        window.alert(`${file.name}: ${uploadError instanceof Error ? uploadError.message : "Upload failed"}`);
      }
    }
    if (added.length) {
      const next = [...buildingFloors, ...added];
      setBuildingFloors(next);
      localStorage.setItem("junc-building-floors", JSON.stringify(next));
    }
    setAddingFloors(false);
  }

  function startFreshWithCurrentFloor() {
    if (!planId) return;
    const currentFloor = { id: planId, name: plan?.metadata?.filename ?? "Floor plan" };
    setBuildingFloors([currentFloor]);
    localStorage.setItem("junc-building-floors", JSON.stringify([currentFloor]));
    localStorage.setItem("junc-building-chats", "{}");
  }

  useEffect(() => {
    if (!ready || !geometry || animationStarted.current) return;
    animationStarted.current = true;
    setAnimationRun(run => run + 1);
  }, [ready, geometry]);

  useEffect(() => {
    if (!animationRun) return;
    setAnimationStage(0);
    const timers = [1100, 2200, 3400, 4900].map((delay, index) => window.setTimeout(() => {
      setAnimationStage(index === 3 ? null : index + 1);
    }, delay));
    return () => timers.forEach(window.clearTimeout);
  }, [animationRun]);

  // Room geometry once the analysis is ready; refetched when the scale changes (areas in m² depend on it)
  useEffect(() => {
    if (!ready || !planId) return;
    const cachedView = floorViewCache.get(planId);
    if (cachedView?.geometry) {
      setGeometry(cachedView.geometry);
      setGeometryError(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/geometry`, { cache: "no-store" });
        const body = await response.json();
        if (!response.ok) throw new Error(body.detail || "Rooms could not be loaded");
        if (!cancelled) {
          setGeometry(body);
          setGeometryError(null);
          const roomOutlines: FloorOutline[] = (body.rooms ?? []).map((room: RoomShape) => ({
            id: room.id, polygon: room.polygon, scope: "room", room_type: room.type,
            label: room.name || room.type.replaceAll("_", " "),
          }));
          const currentRoomIds = new Set(roomOutlines.map(outline => outline.id));
          setSavedOutlines(current => current.filter(outline => outline.scope !== "room" || currentRoomIds.has(outline.id) || outline.id.endsWith("-user")));
          roomBoundaryLinkRef.current = roomBoundaryLinks(evidenceItemsRef.current, roomOutlines);
          const linkedItems = linkRoomBoundaries(evidenceItemsRef.current, roomOutlines);
          evidenceItemsRef.current = linkedItems;
          setEvidenceItems(linkedItems);
          const currentView = floorViewCache.get(planId);
          if (currentView) {
            currentView.geometry = body;
            currentView.evidenceItems = linkedItems;
          }
        }
      } catch (geometryLoadError) {
        if (!cancelled) setGeometryError(geometryLoadError instanceof Error ? geometryLoadError.message : "Rooms could not be loaded");
      }
    })();
    return () => { cancelled = true; };
  }, [ready, planId, scale]);

  useEffect(() => {
    const conversation = conversationRef.current;
    if (conversation) conversation.scrollTo({ top: conversation.scrollHeight, behavior: "smooth" });
  }, [messages, sending]);

  useEffect(() => {
    let objectUrl: string | null = null;
    let imageBlob: Blob | null = null;
    let loadedScale: ScaleCalibration | null = null;
    let loadedDetectedScaleAccepted = false;
    let cancelled = false;
    void params.then(async ({ planId: id }) => {
      setPlanId(id);
      setError(null);
      setReady(false);
      setPlan(null);
      setImageUrl(null);
      setImageSize(null);
      setGeometry(null);
      setCitations([]);
      setScale(null);
      setDetectedScaleAccepted(false);
      setScaleLoaded(false);
      setScaleMode(false);
      setMeasureMode(false);
      setOutlineMode(false);
      setReferenceMode(false);
      setCorrectionMode(false);
      const uploadError = new URLSearchParams(window.location.search).get("uploadError");
      if (uploadError) {
        setError(uploadError);
        return;
      }
      const cachedView = floorViewCache.get(id);
      if (cachedView) {
        setError(null);
        setPlan(cachedView.plan);
        const floors = JSON.parse(localStorage.getItem("junc-building-floors") || "[]") as BuildingFloor[];
        const currentFloor = { id, name: cachedView.plan.metadata?.filename ?? "Floor plan" };
        const nextFloors = floors.some(floor => floor.id === id) ? floors : [...floors, currentFloor];
        setBuildingFloors(nextFloors);
        localStorage.setItem("junc-building-floors", JSON.stringify(nextFloors));
        setScale(cachedView.scale);
          setDetectedScaleAccepted(cachedView.detectedScaleAccepted);
        setScaleLoaded(true);
        if (cachedView.scale) {
          setScalePoints(cachedView.scale.points);
          setScaleUnit(cachedView.scale.unit);
        } else {
          setScalePoints([]);
          setScaleUnit("m");
        }
        setImageSize(cachedView.imageSize);
        setEvidenceItems(cachedView.evidenceItems);
        evidenceItemsRef.current = cachedView.evidenceItems;
        setFloorOutlines(cachedView.floorOutlines);
        setSavedOutlines(cachedView.savedOutlines);
        roomBoundaryLinkRef.current = roomBoundaryLinks(cachedView.evidenceItems, [...cachedView.floorOutlines, ...cachedView.savedOutlines]);
        setCorrectionEvents(cachedView.correctionEvents);
        setGeometry(cachedView.geometry ?? null);
        objectUrl = URL.createObjectURL(cachedView.imageBlob);
        setImageUrl(objectUrl);
        setAnalysisStatus("Loaded saved floor analysis");
        setReady(true);
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
        const existingFloors = JSON.parse(localStorage.getItem("junc-building-floors") || "[]") as BuildingFloor[];
        const currentFloor = { id, name: record.metadata?.filename ?? "Floor plan" };
        const nextFloors = existingFloors.some(floor => floor.id === id) ? existingFloors : [...existingFloors, currentFloor];
        setBuildingFloors(nextFloors);
        localStorage.setItem("junc-building-floors", JSON.stringify(nextFloors));
        try {
          const savedScale = await fetch(`/api/plans/${encodeURIComponent(id)}/scale`, { cache: "no-store" });
          if (savedScale.ok) {
            const saved = await savedScale.json();
            loadedScale = saved?.points?.length === 2 ? saved : null;
            loadedDetectedScaleAccepted = Boolean(saved?.accepted_m_per_px);
            setDetectedScaleAccepted(loadedDetectedScaleAccepted);
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
        imageBlob = await imageResponse.blob();
        objectUrl = URL.createObjectURL(imageBlob);
        if (!cancelled) {
          setImageUrl(objectUrl);
          setAnalysisStatus("Image ready · Analyzing floor plan…");
          try {
            const analysisResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/analyze`, { method: "POST" });
            const result = await analysisResponse.json();
            if (!analysisResponse.ok) throw new Error(result.detail || "Analysis failed");
            if (!cancelled) {
              let cachedEvidence: EvidenceItem[] = Array.isArray(result.items) ? result.items : [];
              let cachedFloorOutlines: FloorOutline[] = Array.isArray(result.floor_outlines) ? result.floor_outlines : [];
              let cachedSavedOutlines: FloorOutline[] = [];
              let cachedCorrectionEvents: Array<{ action: string }> = [];
              if (result.width && result.height) setImageSize({ width: result.width, height: result.height });
              setEvidenceItems(cachedEvidence);
              evidenceItemsRef.current = cachedEvidence;
              setFloorOutlines(cachedFloorOutlines);
              try {
                const correctionResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/corrections`, { cache: "no-store" });
                if (correctionResponse.ok) {
                  const corrections = await correctionResponse.json();
                  const correctedItems = Array.isArray(corrections.items) ? corrections.items : result.items ?? [];
                  cachedEvidence = correctedItems;
                  cachedCorrectionEvents = Array.isArray(corrections.events) ? corrections.events : [];
                  setEvidenceItems(correctedItems);
                  evidenceItemsRef.current = correctedItems;
                  setCorrectionEvents(cachedCorrectionEvents);
                }
              } catch { /* Keep the plan usable if correction history is temporarily unavailable. */ }
              try {
                const outlineResponse = await fetch(`/api/plans/${encodeURIComponent(id)}/outline`, { cache: "no-store" });
                if (outlineResponse.ok) {
                  const outlineData = await outlineResponse.json();
                  const candidates = Array.isArray(outlineData.candidates) ? outlineData.candidates : [];
                  const saved = Array.isArray(outlineData.saved) ? outlineData.saved.map((outline: FloorOutline) => ({ ...outline, decision: outline.decision ?? "accepted" })) : [];
                  cachedFloorOutlines = candidates;
                  cachedSavedOutlines = saved;
                  setFloorOutlines(candidates);
                  setSavedOutlines(saved);
                  roomBoundaryLinkRef.current = roomBoundaryLinks(evidenceItemsRef.current, [...candidates, ...saved]);
                  const linkedItems = linkRoomBoundaries(evidenceItemsRef.current, [...candidates, ...saved]);
                  cachedEvidence = linkedItems;
                  evidenceItemsRef.current = linkedItems;
                  setEvidenceItems(linkedItems);
                }
              } catch { /* Keep the scanned plan usable if saved outlines are temporarily unavailable. */ }
              if (imageBlob && result.width && result.height) {
                const cachedView: CachedFloorView = {
                  plan: record!, imageBlob, scale: loadedScale, detectedScaleAccepted: loadedDetectedScaleAccepted, imageSize: { width: result.width, height: result.height },
                  evidenceItems: cachedEvidence, floorOutlines: cachedFloorOutlines, savedOutlines: cachedSavedOutlines,
                  correctionEvents: cachedCorrectionEvents,
                };
                floorViewCache.set(id, cachedView);
                const cachedBytes = () => [...floorViewCache.values()].reduce((sum, view) => sum + view.imageBlob.size, 0);
                while (floorViewCache.size > FLOOR_VIEW_CACHE_MAX_ITEMS || cachedBytes() > FLOOR_VIEW_CACHE_MAX_BYTES) {
                  const oldest = floorViewCache.keys().next().value;
                  if (!oldest) break;
                  floorViewCache.delete(oldest);
                }
              }
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
      const updatedItems = result.items ?? [];
      setEvidenceItems(updatedItems);
      evidenceItemsRef.current = updatedItems;
      setCorrectionEvents(result.events ?? []);
      const cachedView = floorViewCache.get(planId);
      if (cachedView) {
        cachedView.evidenceItems = updatedItems;
        cachedView.correctionEvents = result.events ?? [];
        cachedView.geometry = null;
      }
      if (action === "update" && item?.category === "room") {
        const linkedOutlines = getAllRoomOutlines();
        const linkedId = itemId ? roomBoundaryLinkRef.current[itemId] : undefined;
        const linkedOutline = linkedOutlines.find(outline => outline.id === linkedId);
        const source = evidenceItems.find(existing => existing.id === itemId);
        const matched = linkedOutline ? { outline: linkedOutline, score: 1 } : source && linkedOutlines.map(outline => {
          const [ax0, ay0, ax1, ay1] = source.bbox;
          const [bx0, by0, bx1, by1] = outlineBounds(outline);
          const overlap = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)) * Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
          const union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - overlap;
          return { outline, score: union > 0 ? overlap / union : 0 };
        }).sort((a, b) => b.score - a.score)[0];
        if (matched && matched.score > 0.25) {
          const geometry = item.polygon?.length ? item.polygon : matched.outline.polygon;
          const bounds = item.polygon?.length ? item.bbox as EvidenceItem["bbox"] : outlineBounds(matched.outline);
          const roomType = item.cls ?? "room";
          const updatedOutline = { ...matched.outline, room_type: roomType, label: roomType.replaceAll("_", " "), polygon: geometry, decision: "accepted" as const };
          const next = [...savedOutlines.filter(outline => outline.id !== updatedOutline.id), updatedOutline];
          const outlineResponse = await fetch(`/api/plans/${encodeURIComponent(planId)}/outline`, {
            method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ outlines: next }),
          });
          const outlineResult = await outlineResponse.json();
          if (!outlineResponse.ok) throw new Error(outlineResult.detail || "Room type saved, but its boundary could not be updated");
          setSavedOutlines(outlineResult.saved ?? next);
          const cachedView = floorViewCache.get(planId);
          if (cachedView) cachedView.savedOutlines = outlineResult.saved ?? next;
          setFloorOutlines(current => current.map(outline => outline.id === updatedOutline.id ? updatedOutline : outline));
          if (itemId) roomBoundaryLinkRef.current[itemId] = updatedOutline.id;
          setEvidenceItems(current => current.map(candidate => candidate.id === itemId ? { ...candidate, bbox: bounds, polygon: geometry } : candidate));
          evidenceItemsRef.current = evidenceItemsRef.current.map(candidate => candidate.id === itemId ? { ...candidate, bbox: bounds, polygon: geometry } : candidate);
        }
      }
      setCorrectionAddMode(false);
      setCorrectionAddPoints([]);
      if (action === "add") setSelectedEvidenceId(null);
    } catch (saveError) {
      setCorrectionError(saveError instanceof Error ? saveError.message : "Could not save correction");
    } finally { setCorrectionSaving(false); }
  }

  async function submitFeedback(messageIndex: number, feedback: "helpful" | "incorrect" | "unclear" | "trusted" | "rejected") {
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
      setMessages(current => current.map((message, index) => index === messageIndex ? {
        ...message, feedback,
        ...(feedback === "trusted" ? { evidenceDecision: feedback, text: message.originalText ?? message.text } : {}),
        ...(feedback === "rejected" ? { evidenceDecision: feedback, text: "You rejected the supporting evidence, so I’m withdrawing this provisional answer. Review or correct the highlighted items before relying on a new answer." } : {}),
      } : message));
    } catch (error) {
      setFeedbackError(error instanceof Error ? error.message : "Could not save feedback");
    }
  }

  function showMessageEvidence(message: ChatMessage) {
    if (message.evidence?.length) setCitations(message.evidence);
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

  function getAllRoomOutlines() {
    const backendRooms: FloorOutline[] = (geometry?.rooms ?? []).map(room => ({
      id: room.id, polygon: room.polygon, scope: "room", room_type: room.type,
      label: room.name || room.type.replaceAll("_", " "),
    }));
    const currentRoomIds = new Set(backendRooms.map(outline => outline.id));
    const relevantSavedRooms = savedOutlines.filter(outline => outline.scope === "room" && (currentRoomIds.has(outline.id) || outline.id.endsWith("-user")));
    const byId = new Map([...backendRooms, ...relevantSavedRooms].map(outline => [outline.id, outline]));
    for (const outline of relevantSavedRooms) byId.set(outline.id, outline);
    return [...byId.values()];
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
      const recentHistory = messages.slice(-8).map(({ role, text }) => ({ role, text }));
      const requestBody = JSON.stringify({ question: submitted, history: recentHistory, scale, references: drawingReferences,
        floor_outlines: savedOutlines.filter(outline => outline.scope !== "room" || geometry?.rooms.some(room => room.id === outline.id) || outline.id.endsWith("-user")) });
      let response: Response;
      for (let attempt = 0; ; attempt++) {
        response = await fetch(`/api/plans/${encodeURIComponent(planId)}/ask`, {
          method: "POST", headers: { "Content-Type": "application/json" }, body: requestBody,
        });
        if (attempt >= 1 || ![502, 503].includes(response.status)) break;
        await new Promise(resolve => setTimeout(resolve, 700));
      }
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not answer that question");
      const explanation = typeof result.explanation === "string" ? result.explanation : "";
      const detectedScaleAvailable = Boolean(geometry?.scale && ["dimensions", "written_areas"].includes(geometry.scale.source));
      const measurementIntent = /\b(area|size|how big|how large|square metres?|square meters?|m²|measure|measurement|dimensions?)\b/i.test(submitted);
      const requestedRoom = getRequestedRoomType(submitted) ?? (/\b(it|that|this|its|their)\b/i.test(submitted)
        ? [...recentHistory].reverse().filter(turn => turn.role === "user").map(turn => getRequestedRoomType(turn.text)).find(Boolean) ?? null
        : null);
      const areaQuestionForUI = requestedRoom ? `${submitted} ${requestedRoom.label}` : submitted;
      const needsScale = measurementIntent && !scale && !detectedScaleAvailable && /scale|known distance/i.test(explanation) && /area|dimension|measure/i.test(explanation);
      // The API can use a detected scale that the user already confirmed. That
      // scale lives in detectedScaleAccepted, not in the manual `scale` state.
      // Don't hide the outline review just because the scale came from detection.
      const needsOutline = measurementIntent && /boundary|outline/i.test(explanation) && !hasRequiredOutline(areaQuestionForUI);
      const needsMeasurement = Boolean(scale) && /\b(measure|measurement|distance|length|width|wide|long)\b/i.test(submitted);
      const answer = measurementIntent && !scale && detectedScaleAvailable && result.answer === null
        ? `I found a possible scale of ${geometry!.scale!.m_per_px.toFixed(5)} metres per pixel. Confirm it in the scale prompt before I use it for measurements.`
        : needsMeasurement
        ? "I can measure that on the plan. Choose the two points you want to measure, and I’ll calculate the distance."
        : needsScale
        ? "Let’s set the drawing scale first. Click the two ends of a distance you know on the plan, such as a wall with a printed measurement. Then enter that real distance and unit to calibrate the drawing."
        : needsOutline
          ? requestedRoom
            ? floorOutlines.some(outline => outline.scope === "room" && outline.room_type === requestedRoom.roomType)
              ? `I found a suggested ${requestedRoom.label.toLowerCase()} outline. Review it, adjust it if needed, and accept it to calculate an area estimate.`
              : `I couldn’t detect a clear ${requestedRoom.label.toLowerCase()} boundary. Draw or edit the outline, then accept it to calculate an area estimate.`
            : floorOutlines.some(outline => outline.scope !== "room")
              ? "I found a suggested floor outline. Review it, adjust it if needed, and accept it to calculate an area estimate."
              : "I couldn’t detect a clear floor boundary. Draw or edit the outline, then accept it to calculate an area estimate."
        : result.answer === null ? result.explanation : `${result.answer}${result.unit ? ` ${result.unit}` : ""}. ${result.explanation}`;
      setCitations(Array.isArray(result.citations) ? result.citations : []);
      const provisional = result.confidence === "low";
      setMessages(current => [...current, { role: "assistant", text: provisional ? `Possible answer: ${answer}` : answer, ...(provisional ? { originalText: answer } : {}), evidence_ids: (result.citations ?? []).map((item: PlanCitation) => item.id), evidence: result.citations ?? [], confidence: result.confidence }]);
      setDrawingReferences([]);
      if (needsScale) {
        setPendingAreaQuestion(areaQuestionForUI);
        startScaleCalibration();
      } else if (needsOutline) {
        setPendingAreaQuestion(areaQuestionForUI);
      } else if (needsMeasurement) {
        setMeasurementRequested(true);
      }
    } catch (askError) {
      setCitations([]);
      const message = askError instanceof Error ? askError.message : "Could not answer that question";
      const userMessage = /supabase|analysis backend/i.test(message)
        ? "I couldn't load the saved plan just now. Your question is still in the chat; please try again in a moment."
        : message;
      setQuestion(submitted);
      setMessages(current => [...current, { role: "assistant", text: userMessage }]);
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

  function startMeasurement() {
    setMeasurementRequested(false);
    setMeasureMode(true);
    setMeasurePoints([]);
    setScaleMode(false);
    setOutlineMode(false);
    setCorrectionMode(false);
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
      { roomType: "bathroom", label: "Bathroom", terms: ["bathroom", "washroom", "bath"] },
      { roomType: "bedroom", label: "Bedroom", terms: ["bedroom", "bed room"] },
      { roomType: "living", label: "Living room", terms: ["living room", "livingroom", "living area"] },
      { roomType: "balcony", label: "Balcony", terms: ["balcony"] },
      { roomType: "entry", label: "Entry", terms: ["entry", "hallway"] },
      { roomType: "storage", label: "Storage", terms: ["storage"] },
      { roomType: "garage", label: "Garage", terms: ["garage"] },
      { roomType: "wc", label: "WC", terms: ["wc", "toilet room"] },
      { roomType: "sauna", label: "Sauna", terms: ["sauna"] },
    ];
    return rooms.find(room => room.terms.some(term => normalized.includes(term))) ?? null;
  }

  function hasRequiredOutline(text: string) {
    const targetRoom = getRequestedRoomType(text);
    const candidates = getReviewOutlines(text);
    if (targetRoom && candidates.length) {
      return candidates.every(outline => outline.decision === "accepted" || outline.decision === "rejected") &&
        candidates.some(outline => outline.decision === "accepted");
    }
    return targetRoom
      ? savedOutlines.some(outline => outline.scope === "room" && outline.room_type === targetRoom.roomType && outline.decision !== "rejected")
      : savedOutlines.some(outline => outline.decision === "accepted");
  }

  function getReviewOutlines(questionText?: string) {
    const targetRoom = questionText ? getRequestedRoomType(questionText) : null;
    const eligible = targetRoom
      ? getAllRoomOutlines().filter(outline => outline.room_type === targetRoom.roomType)
      : floorOutlines.filter(outline => outline.scope !== "room");
    const byId = new Map(eligible.map(outline => [outline.id, outline]));
    for (const outline of savedOutlines) {
      if ((!targetRoom && outline.scope !== "room") || (targetRoom && outline.scope === "room" && outline.room_type === targetRoom.roomType)) {
        byId.set(outline.id, outline);
      }
    }
    return [...byId.values()];
  }

  function startOutlineReview(questionText?: string, outlineId?: string) {
    const targetRoom = questionText ? getRequestedRoomType(questionText) : null;
    const candidates = getReviewOutlines(questionText);
    const roomCandidates = targetRoom ? getAllRoomOutlines() : [];
    const requestedOutlineIndex = outlineId ? candidates.findIndex(outline => outline.id === outlineId) :
      targetRoom
        ? roomCandidates.findIndex(outline => outline.room_type === targetRoom.roomType)
        : floorOutlines.findIndex(outline => outline.scope !== "room");
    const index = requestedOutlineIndex >= 0 ? requestedOutlineIndex : Math.max(0, outlineCandidateIndex);
    const savedTarget = outlineId ? savedOutlines.find(outline => outline.id === outlineId) :
      candidates.find(outline => outline.decision !== "rejected") ?? null;
    const candidatesForScope = targetRoom ? roomCandidates : floorOutlines;
    const suggestion = outlineId
      ? [...getAllRoomOutlines(), ...floorOutlines].find(outline => outline.id === outlineId)
      : requestedOutlineIndex >= 0 ? candidatesForScope[index] : null;
    const manualTarget = savedTarget ?? suggestion ?? (targetRoom
      ? { id: `room-${targetRoom.roomType}-user`, polygon: [], scope: "room" as const, room_type: targetRoom.roomType, label: targetRoom.label }
      : savedOutlines[0] ?? floorOutlines[index] ?? { id: "floor-user", polygon: [], scope: "floor" as const, label: "Whole floor" });
    setManualOutlineTarget(manualTarget);
    setEditingOutlineId(manualTarget.id);
    setOutlineCandidateIndex(index);
    setOutlinePoints(simplifyBoundary((manualTarget.polygon ?? []).map(([x, y]) => [x, y])));
    setOutlineMode(true);
    setAddOutlinePointMode(false);
    setScaleMode(false);
    setMeasureMode(false);
    setOutlineError(null);
  }

  async function saveOutlineDecision(outline: FloorOutline, decision: "accepted" | "rejected") {
    if (!planId) return;
    setOutlineError(null);
    setOutlineSaving(true);
    const updated = { ...outline, decision };
    const next = [...savedOutlines.filter(item => item.id !== outline.id), updated];
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/outline`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ outlines: next }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save the room review");
      const saved = result.saved ?? next;
      setSavedOutlines(saved);
      if (pendingAreaQuestion) {
        const reviewed = getReviewOutlines(pendingAreaQuestion).map(item => saved.find((entry: FloorOutline) => entry.id === item.id) ?? item);
        if (reviewed.length > 0 && reviewed.every(item => item.decision === "accepted" || item.decision === "rejected") && reviewed.some(item => item.decision === "accepted")) {
          const accepted = reviewed.filter(item => item.decision === "accepted");
          const areas = accepted.map(item => areaM2(item.polygon));
          if (areas.every((area): area is number => area !== null)) {
            const details = accepted.map((item, index) => `${item.label ?? "Room"} ${index + 1}: ${areas[index].toFixed(1)} m²`);
            const total = areas.reduce((sum, area) => sum + area, 0);
            setMessages(current => [...current, { role: "assistant", text: `${details.join("; ")}. Total: ${total.toFixed(1)} m².` }]);
            setPendingAreaQuestion(null);
          }
        }
      }
    } catch (error) {
      setOutlineError(error instanceof Error ? error.message : "Could not save the room review");
    } finally { setOutlineSaving(false); }
  }

  function planPoint(event: React.MouseEvent<SVGGraphicsElement> | React.PointerEvent<SVGGraphicsElement> | React.PointerEvent<SVGSVGElement>): [number, number] | null {
    if (!imageSize) return null;
    const svg = event.currentTarget instanceof SVGSVGElement ? event.currentTarget : event.currentTarget.ownerSVGElement;
    const transform = svg?.getScreenCTM();
    if (!transform) return null;
    const point = svg!.createSVGPoint();
    point.x = event.clientX;
    point.y = event.clientY;
    const mapped = point.matrixTransform(transform.inverse());
    return [Math.max(0, Math.min(imageSize.width, mapped.x)), Math.max(0, Math.min(imageSize.height, mapped.y))];
  }

  function beginEvidenceDrag(event: React.PointerEvent<SVGGraphicsElement>, item: EvidenceItem, mode: EvidenceDrag["mode"]) {
    if (!correctionMode || !imageSize) return;
    event.stopPropagation();
    event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId);
    const start = planPoint(event);
    if (!start) return;
    beginCorrection(item);
    setDraggingEvidence({ id: item.id, mode, start, bbox: item.bbox, polygon: item.polygon, moved: false });
  }

  function evidenceGeometry(drag: EvidenceDrag, point: [number, number]) {
    if (!imageSize) return null;
    const [x0, y0, x1, y1] = drag.bbox;
    let bbox: EvidenceItem["bbox"];
    if (drag.mode === "move") {
      const dx = Math.max(-x0, Math.min(imageSize.width - x1, point[0] - drag.start[0]));
      const dy = Math.max(-y0, Math.min(imageSize.height - y1, point[1] - drag.start[1]));
      bbox = [x0 + dx, y0 + dy, x1 + dx, y1 + dy];
    } else {
      bbox = [x0, y0, Math.max(x0 + 8, Math.min(imageSize.width, point[0])), Math.max(y0 + 8, Math.min(imageSize.height, point[1]))];
    }
    const polygon = drag.polygon?.map(([x, y]) => drag.mode === "move"
      ? [x + bbox[0] - x0, y + bbox[1] - y0] as [number, number]
      : [bbox[0] + (x - x0) * (bbox[2] - bbox[0]) / Math.max(1, x1 - x0), bbox[1] + (y - y0) * (bbox[3] - bbox[1]) / Math.max(1, y1 - y0)] as [number, number]);
    return { bbox, polygon };
  }

  function moveEvidenceBox(event: React.PointerEvent<SVGSVGElement>) {
    if (!draggingEvidence) return;
    const point = planPoint(event);
    if (!point) return;
    const moved = draggingEvidence.moved || Math.hypot(point[0] - draggingEvidence.start[0], point[1] - draggingEvidence.start[1]) > 2;
    if (moved && !draggingEvidence.moved) setDraggingEvidence(current => current ? { ...current, moved: true } : null);
    const geometry = evidenceGeometry(draggingEvidence, point);
    if (geometry) setEvidenceItems(current => current.map(item => item.id === draggingEvidence.id ? { ...item, ...geometry, user_verified: true } : item));
  }

  function finishEvidenceBox(event: React.PointerEvent<SVGSVGElement>) {
    if (!draggingEvidence) return;
    const drag = draggingEvidence;
    const point = planPoint(event);
    const geometry = point ? evidenceGeometry(drag, point) : null;
    setDraggingEvidence(null);
    if (!drag.moved || !geometry) return;
    const item = evidenceItems.find(candidate => candidate.id === drag.id);
    if (item) void saveCorrection("update", drag.id, { ...item, ...geometry });
  }

  function handlePlanClick(event: React.MouseEvent<SVGSVGElement>) {
    if (referenceMode && !scaleMode && !measureMode && !outlineMode && !correctionMode) {
      const point = planPoint(event);
      if (!point) return;
      if (referenceDragStart && referenceDragEnd && Math.hypot(referenceDragEnd[0] - referenceDragStart[0], referenceDragEnd[1] - referenceDragStart[1]) > 8) {
        const [sx, sy] = referenceDragStart;
        const [ex, ey] = referenceDragEnd;
        setDrawingReferences(current => [...current, { bbox: [Math.min(sx, ex), Math.min(sy, ey), Math.max(sx, ex), Math.max(sy, ey)], label: "selected area" }]);
      } else {
        const hit = evidenceItems
          .filter(item => ["fixture", "door", "window", "room"].includes(item.category))
          .map(item => {
            const [x0, y0, x1, y1] = item.bbox;
            const inside = point[0] >= x0 && point[0] <= x1 && point[1] >= y0 && point[1] <= y1;
            const dx = Math.max(x0 - point[0], 0, point[0] - x1);
            const dy = Math.max(y0 - point[1], 0, point[1] - y1);
            const near = Math.hypot(dx, dy) <= 10;
            const area = Math.max(1, (x1 - x0) * (y1 - y0));
            const priority = item.category === "room" ? 1 : 0;
            return { item, inside, near, area, priority, distance: Math.hypot(dx, dy) };
          })
          .filter(candidate => candidate.inside || candidate.near)
          .sort((a, b) => a.priority - b.priority || a.area - b.area || a.distance - b.distance)[0]?.item;
        const radius = 14;
        setDrawingReferences(current => [...current, hit
          ? { bbox: hit.bbox, label: `${hit.cls || hit.category} (${hit.category})`, item_id: hit.id }
          : { bbox: [Math.max(0, point[0] - radius), Math.max(0, point[1] - radius), Math.min(imageSize?.width ?? point[0] + radius, point[0] + radius), Math.min(imageSize?.height ?? point[1] + radius, point[1] + radius)], label: "selected point" }]);
      }
      setReferenceMode(false);
      setReferenceDragStart(null);
      setReferenceDragEnd(null);
      return;
    }
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
    if (correctionMode && !outlineMode) return;
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
        id: target?.id ?? editingOutlineId ?? "floor-user",
        polygon: outlinePoints,
        scope: target?.scope ?? "floor",
        room_type: target?.room_type ?? null,
        label: target?.label ?? "Whole floor",
        decision: "accepted" as const,
      };
      const next = [...savedOutlines.filter(item => item.id !== outline.id), outline];
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/outline`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ outlines: next }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save the floor outline");
      setSavedOutlines(result.saved ?? next);
      const cachedView = floorViewCache.get(planId);
      if (cachedView) cachedView.savedOutlines = result.saved ?? next;
      setOutlineMode(false);
      setEditingOutlineId(null);
      if (outline.scope === "room") {
        const [x0, y0, x1, y1] = outlineBounds(outline);
        const linkedEvidenceId = Object.entries(roomBoundaryLinkRef.current).find(([, outlineId]) => outlineId === outline.id)?.[0];
        const linkedEvidence = evidenceItems.find(item => item.id === linkedEvidenceId);
        const matchedRoom = linkedEvidence ? { item: linkedEvidence, score: 1 } : evidenceItems.filter(item => item.category === "room" && item.raw !== "ocr").map(item => {
          const [ax0, ay0, ax1, ay1] = item.bbox;
          const overlap = Math.max(0, Math.min(ax1, x1) - Math.max(ax0, x0)) * Math.max(0, Math.min(ay1, y1) - Math.max(ay0, y0));
          const union = (ax1 - ax0) * (ay1 - ay0) + (x1 - x0) * (y1 - y0) - overlap;
          return { item, score: union > 0 ? overlap / union : 0 };
        }).sort((a, b) => b.score - a.score)[0];
        if (matchedRoom && matchedRoom.score > 0.25) {
          void saveCorrection("update", matchedRoom.item.id, {
            ...matchedRoom.item, category: "room", cls: outline.room_type ?? matchedRoom.item.cls,
            bbox: outlineBounds(outline), polygon: outline.polygon,
          });
          roomBoundaryLinkRef.current[matchedRoom.item.id] = outline.id;
        }
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
      const cachedView = floorViewCache.get(planId);
      if (cachedView) cachedView.savedOutlines = [];
      setOutlinePoints([]);
      setOutlineMode(false);
      setEditingOutlineId(null);
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
      setDetectedScaleAccepted(false);
      const cachedView = floorViewCache.get(planId);
      if (cachedView) {
        cachedView.scale = result;
        cachedView.detectedScaleAccepted = false;
        cachedView.geometry = null;
      }
      setScaleMode(false);
      setScalePoints(result.points);
      setScaleDistance(String(result.distance));
      setScaleUnit(result.unit);
    } catch (saveError) {
      setScaleError(saveError instanceof Error ? saveError.message : "Could not save scale");
    } finally { setScaleSaving(false); }
  }

  async function confirmDetectedScale() {
    const detected = geometry?.scale;
    if (!planId || !detected || !["dimensions", "written_areas"].includes(detected.source)) return;
    setDetectedScaleSaving(true);
    setScaleError(null);
    try {
      const response = await fetch(`/api/plans/${encodeURIComponent(planId)}/scale`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accepted_m_per_px: detected.m_per_px, accepted_source: detected.source }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || "Could not save scale confirmation");
      setDetectedScaleAccepted(true);
      const cachedView = floorViewCache.get(planId);
      if (cachedView) cachedView.detectedScaleAccepted = true;
    } catch (saveError) {
      setScaleError(saveError instanceof Error ? saveError.message : "Could not save scale confirmation");
    } finally { setDetectedScaleSaving(false); }
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
    const effectiveMetersPerPixel = metersPerPixel ?? (detectedScaleAccepted ? geometry?.scale?.m_per_px ?? null : null);
    if (!effectiveMetersPerPixel || polygon.length < 3) return null;
    const pixelArea = Math.abs(polygon.reduce((sum, [x, y], index) => {
      const [nextX, nextY] = polygon[(index + 1) % polygon.length];
      return sum + x * nextY - nextX * y;
    }, 0)) / 2;
    return pixelArea * effectiveMetersPerPixel * effectiveMetersPerPixel;
  };
  const acceptedArea = savedOutlines.filter(outline => outline.decision !== "rejected")
    .reduce((sum, outline) => sum + (areaM2(outline.polygon) ?? 0), 0);
  const displayOutlines = pendingAreaQuestion
    ? getReviewOutlines(pendingAreaQuestion)
    : correctionMode ? getAllRoomOutlines().filter(outline => outline.decision !== "rejected")
      : savedOutlines.filter(outline => outline.decision !== "rejected");
  const reviewedOutlines = pendingAreaQuestion ? getReviewOutlines(pendingAreaQuestion) : [];
  const reviewedAccepted = reviewedOutlines.filter(outline => outline.decision === "accepted");
  const reviewedAcceptedArea = reviewedAccepted.reduce((sum, outline) => sum + (areaM2(outline.polygon) ?? 0), 0);
  const outlineLabelPoint = (polygon: Array<[number, number]>): [number, number] => polygon.reduce(
    ([sumX, sumY], [x, y]) => [sumX + x / polygon.length, sumY + y / polygon.length], [0, 0] as [number, number]);
  const selectedEvidence = evidenceItems.find(item => item.id === selectedEvidenceId) ?? null;
  // OCR text boxes are tiny and read as noise: they stay evidence for the backend but are not shown or reviewed
  const reviewItems = evidenceItems.filter(item => item.raw !== "ocr");
  // a cited OCR label is shown as the room it names (the space containing the text), else as its text box
  const shownCitation = (item: PlanCitation): PlanCitation => {
    if (item.raw !== "ocr" || !geometry) return item;
    const cx = (item.bbox[0] + item.bbox[2]) / 2, cy = (item.bbox[1] + item.bbox[3]) / 2;
    const room = geometry.rooms.find(room => pointInPolygon(cx, cy, room.polygon));
    return room ? { ...item, bbox: room.bbox, polygon: room.polygon } : item;
  };

  const imageStatus = plan?.plan_images?.find(image => image.role === "source")?.upload_status;
  return (
    <main className="workspace">
      <header className="toolbar">
        <input ref={addFloorsInput} className="visually-hidden" type="file" multiple accept="application/pdf,image/png,image/jpeg,image/webp" onChange={event => { void addFloorPlans(event.target.files); event.target.value = ""; }} />
        <div className="plan-title" title={plan?.metadata?.filename ?? "Loading plan…"}>
          <span className="plan-title-icon" aria-hidden="true">▤</span>
          <span>{plan?.metadata?.filename ?? "Loading plan…"}</span>
        </div>
        <div className="toolbar-actions">
          <details className="toolbar-menu floor-switcher"><summary className="toolbar-button">Building · {buildingFloors.length} {buildingFloors.length === 1 ? "floor" : "floors"}</summary><div className="toolbar-menu-popover">
            {buildingFloors.map((floor, index) => <Link key={floor.id} href={`/plans/${floor.id}`} aria-current={floor.id === planId ? "page" : undefined}>{index + 1}. {floor.name}</Link>)}
            <button type="button" onClick={() => addFloorsInput.current?.click()}>{addingFloors ? "Adding floors…" : "+ Add floor plans"}</button>
            {buildingFloors.length > 1 && <Link href="/building">Ask across floors</Link>}
            {buildingFloors.length > 1 && <button type="button" onClick={startFreshWithCurrentFloor}>Start fresh with this floor</button>}
          </div></details>
          {geometry && <details className="toolbar-menu"><summary className={`toolbar-button${showRooms || showWalls ? " is-on" : ""}`}>Overlays</summary><div className="toolbar-menu-popover" onClick={event => { if (event.target instanceof HTMLElement && event.target.closest("button")) event.currentTarget.closest("details")?.removeAttribute("open"); }}>
            <button type="button" aria-pressed={showRooms} onClick={() => setShowRooms(on => !on)}>{showRooms ? "✓ " : ""}Rooms ({geometry.rooms.filter(room => room.tier === "strong").length})</button>
            {showRooms && geometry.rooms.some(room => room.tier !== "strong") && <button type="button" aria-pressed={showUncertain} onClick={() => setShowUncertain(on => !on)}>{showUncertain ? "✓ " : ""}Uncertain spaces ({geometry.rooms.filter(room => room.tier !== "strong").length})</button>}
            {geometry.wall_segments.length > 0 && <button type="button" aria-pressed={showWalls} onClick={() => setShowWalls(on => !on)}>{showWalls ? "✓ " : ""}Walls ({geometry.wall_segments.length})</button>}
          </div></details>}
          {geometryError && <span className="scale-status" title={geometryError}>Rooms unavailable</span>}
          {ready && scaleLoaded && scale && !scaleMode && <span className="scale-status">Scale saved</span>}
          {ready && scaleLoaded && !scale && geometry?.scale && geometry.scale.source !== "calibration" && <span className="scale-status" title={`${(geometry.scale.m_per_px * 1000).toFixed(2)} mm per pixel${geometry.scale.warning ? ` · ${geometry.scale.warning}` : ""}`}>{geometry.scale.source === "dimensions" ? `Scale read from ${geometry.scale.dimensions?.length ?? 0} dimension lines` : "Scale from written room areas"}{geometry.scale.warning ? " (check)" : ""}</span>}
          {ready && scaleLoaded && !scaleMode && <button className={`toolbar-button${scale ? "" : " primary-button"}`} type="button" onClick={startScaleCalibration}>{scale ? "Adjust" : "Set scale"}</button>}
          {ready && scaleLoaded && scaleMode && <button className="toolbar-button" type="button" onClick={cancelScaleCalibration}>Cancel scale setup</button>}
          {correctionMode && <button className="toolbar-button" type="button" onClick={() => { setCorrectionMode(false); setCorrectionAddMode(false); setSelectedEvidenceId(null); }}>Done reviewing</button>}
          {referenceMode && <button className="toolbar-button" type="button" onClick={() => { setReferenceMode(false); setReferenceDragStart(null); setReferenceDragEnd(null); }}>Cancel reference</button>}
          {outlineMode && <button className="toolbar-button" type="button" onClick={() => setOutlineMode(false)}>Cancel outline</button>}
          {ready && !scaleMode && !correctionMode && !outlineMode && !referenceMode && <details className="toolbar-menu"><summary className="toolbar-button">More tools</summary><div className="toolbar-menu-popover" onClick={event => { if (event.target instanceof HTMLElement && event.target.closest("button, a")) event.currentTarget.closest("details")?.removeAttribute("open"); }}>
            {scaleLoaded && scale && <button type="button" onClick={startScaleCalibration}>Adjust scale</button>}
            {ready && geometry && <button type="button" onClick={() => setAnimationRun(run => run + 1)}>↻ Replay analysis</button>}
            <button type="button" onClick={startCorrectionReview}>Review detections</button>
            <button type="button" onClick={() => { setReferenceMode(true); setReferenceDragStart(null); setReferenceDragEnd(null); }}>Refer to drawing</button>
            {scaleLoaded && <button type="button" onClick={() => startOutlineReview()}>{savedOutlines.length ? `Edit ${savedOutlines[0].label?.toLowerCase() ?? "outline"}` : floorOutlines.length ? "Review detected outline" : "Draw outline"}</button>}
            <Link href="/">Upload another plan</Link>
          </div></details>}
          <span className="connection-status"><i /> {error ?? (imageStatus === "uploaded" ? analysisStatus : "Loading saved plan…")}</span>
        </div>
      </header>
      <div className="work-area">
        <section className={`drawing-panel${scaleMode ? " is-calibrating" : ""}`} aria-label="Plan drawing">
          {error ? <div className="upload-empty-state"><h1>Could not load this plan</h1><p role="alert">{error}</p><Link className="toolbar-button" href="/">Back to upload</Link></div> : (
            imageUrl ? <div className="image-frame"><img className="plan-image" src={imageUrl} alt={`Floor plan: ${plan?.metadata?.filename ?? "saved plan"}`} />{!ready && !error && <div className={`analysis-progress${analysisStatus.startsWith("Analysis unavailable:") ? " failed" : ""}`} role="status" aria-live="polite"><div className="analysis-progress-card">{analysisStatus.startsWith("Analysis unavailable:") ? <div className="analysis-progress-icon failed-icon">!</div> : <span className="analysis-progress-spinner" />}<h2>{analysisStatus.startsWith("Analysis unavailable:") ? "Analysis couldn’t finish" : "Analyzing your floor plan"}</h2><p>{analysisStatus.startsWith("Analysis unavailable:") ? analysisStatus.replace("Analysis unavailable: ", "") : "Finding walls, rooms, doors, windows, and fixtures. This may take a little while for detailed plans."}</p>{analysisStatus.startsWith("Analysis unavailable:") ? <button className="toolbar-button" type="button" onClick={() => window.location.reload()}>Try again</button> : <><div className="analysis-progress-track"><span /></div><small>You can keep this page open while we work.</small></>}</div></div>}{imageSize && <svg className={`citation-overlay${scaleMode || measureMode || outlineMode || measurePoints.length === 2 ? " scale-selecting" : ""}${correctionMode ? " correction-active" : ""}${referenceMode ? " reference-active" : ""}`} viewBox={`0 0 ${imageSize.width} ${imageSize.height}`} preserveAspectRatio="xMidYMid meet" onClick={handlePlanClick} onPointerDown={event => { if (referenceMode) { const point = planPoint(event); if (point) { setReferenceDragStart(point); setReferenceDragEnd(point); event.currentTarget.setPointerCapture(event.pointerId); } } }} onPointerMove={event => { if (referenceMode && referenceDragStart) { const point = planPoint(event); if (point) setReferenceDragEnd(point); } else if (draggingEvidence) moveEvidenceBox(event); else if (draggingOutlinePoint !== null) { const point = planPoint(event); if (point) setOutlinePoints(current => current.map((value, index) => index === draggingOutlinePoint ? point : value)); } else if (draggingMeasurePoint !== null) { const point = planPoint(event); if (point) setMeasurePoints(current => current.map((value, index) => index === draggingMeasurePoint ? point : value)); } }} onPointerUp={event => { finishEvidenceBox(event); setDraggingOutlinePoint(null); setDraggingMeasurePoint(null); }} aria-label={referenceMode ? "Click an item or drag to reference an area" : correctionMode ? "Select a detected item to correct" : outlineMode ? "Review and edit suggested floor boundary" : scaleMode || measureMode ? "Select two points" : `Highlighted ${citations.length} supporting items`}>
              {drawingReferences.map((reference, index) => <g className="drawing-reference-mark" key={`reference-${index}`}><rect x={reference.bbox[0]} y={reference.bbox[1]} width={Math.max(1, reference.bbox[2] - reference.bbox[0])} height={Math.max(1, reference.bbox[3] - reference.bbox[1])} rx="3"/><text x={reference.bbox[0] + 4} y={reference.bbox[1] - 4}>{index + 1}</text></g>)}
              {animationStage !== null && geometry && <g className={`analysis-animation stage-${animationStage}`} aria-hidden="true">
                {animationStage >= 0 && geometry.wall_segments.map((segment, index) => <line className="animated-wall" key={segment.id} x1={segment.p0[0]} y1={segment.p0[1]} x2={segment.p1[0]} y2={segment.p1[1]} pathLength="1" style={{ animationDelay: `${Math.min(900, index * 18)}ms` }} />)}
                {animationStage >= 1 && geometry.rooms.filter(room => room.tier === "strong").map((room, index) => <g className="animated-room" key={room.id} style={{ animationDelay: `${index * 100}ms` }}><polygon points={room.polygon.map(([x, y]) => `${x},${y}`).join(" ")} style={{ fill: ROOM_COLOURS[room.type] ?? "#7a8a80" }} /><text x={(room.bbox[0] + room.bbox[2]) / 2} y={(room.bbox[1] + room.bbox[3]) / 2}>{roomCaption(room)}</text></g>)}
                {animationStage >= 2 && evidenceItems.filter(item => item.category !== "room").map((item, index) => { const [x0, y0, x1, y1] = item.bbox; return <g className="animated-evidence" key={item.id} style={{ animationDelay: `${Math.min(1000, index * 35)}ms` }}><circle cx={(x0 + x1) / 2} cy={(y0 + y1) / 2} r={Math.max(7, Math.min(16, (x1 - x0 + y1 - y0) / 5))} /><text x={(x0 + x1) / 2} y={(y0 + y1) / 2}>{item.cls === "window" ? "W" : item.category === "door" ? "D" : "•"}</text></g>; })}
              </g>}
              {referenceMode && referenceDragStart && referenceDragEnd && <rect className="drawing-reference-preview" x={Math.min(referenceDragStart[0], referenceDragEnd[0])} y={Math.min(referenceDragStart[1], referenceDragEnd[1])} width={Math.abs(referenceDragEnd[0] - referenceDragStart[0])} height={Math.abs(referenceDragEnd[1] - referenceDragStart[1])}/>}
              {correctionMode && reviewItems.map(item => {
                const [x0, y0, x1, y1] = item.bbox;
                const points = item.polygon?.map(([x, y]) => `${x},${y}`).join(" ");
                return <g className={`correction-mark${selectedEvidenceId === item.id ? " selected" : ""}${item.user_verified ? " user-corrected" : ""}`} key={item.id} onClick={event => { event.stopPropagation(); beginCorrection(item); }} role="button" aria-label={`${item.category} ${item.cls}`}>
                  {points ? <polygon points={points} onPointerDown={event => beginEvidenceDrag(event, item, "move")} /> : <rect x={x0} y={y0} width={Math.max(2, x1 - x0)} height={Math.max(2, y1 - y0)} onPointerDown={event => beginEvidenceDrag(event, item, "move")} />}
                  {selectedEvidenceId === item.id && <rect className="correction-resize-handle" x={x1 - 9} y={y1 - 9} width="18" height="18" rx="3" onPointerDown={event => beginEvidenceDrag(event, item, "resize")} onClick={event => event.stopPropagation()} aria-label="Resize selected box" />}
                </g>;
              })}
              {geometry && (outlineMode
                ? manualOutlineTarget?.scope === "room"
                : showRooms || pendingAreaQuestion !== null) && !correctionMode && <g className="room-layer">
                {geometry.rooms.filter(room => outlineMode && manualOutlineTarget?.scope === "room"
                  ? room.id === editingOutlineId
                  : pendingAreaQuestion
                    ? room.type === getRequestedRoomType(pendingAreaQuestion)?.roomType
                    : room.tier === "strong" || showUncertain).map(room => {
                  const colour = room.tier === "unknown" ? "#7a8a80" : (ROOM_COLOURS[room.type] ?? "#7a8a80");
                  const [x0, y0, x1, y1] = room.bbox;
                  const fontSize = Math.max(11, Math.min(imageSize.width, imageSize.height) / 45);
                  const caption = room.tier === "strong" ? roomCaption(room) : room.tier === "weak" ? `${room.type}?` : "?";
                  const tooltip = room.tier === "unknown" ? "Space with unknown purpose" : `${room.tier === "weak" ? "Probably " : ""}${room.name ?? room.type}: ${roomCaption(room)} (${roomWhy(room)})`;
                  const editingRoom = outlineMode && room.id === editingOutlineId;
                  const savedBoundary = savedOutlines.find(outline => outline.id === room.id);
                  const polygon = editingRoom ? outlinePoints : savedBoundary?.polygon ?? room.polygon;
                  return <g key={room.id} className={`room-shape ${room.tier}`}>
                    <polygon points={polygon.map(([x, y]) => `${x},${y}`).join(" ")} style={{ stroke: colour, fill: room.tier === "unknown" ? "none" : colour }}>
                      <title>{tooltip}</title>
                    </polygon>
                    {!editingRoom && <text x={(x0 + x1) / 2} y={(y0 + y1) / 2} style={{ fontSize }}>{caption}</text>}
                    {editingRoom && outlinePoints.map(([x, y], pointIndex) => <g className="room-boundary-handle" key={`corner-${pointIndex}`} transform={`translate(${x} ${y})`} onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setDraggingOutlinePoint(pointIndex); }} onPointerUp={event => { event.stopPropagation(); setDraggingOutlinePoint(null); }} onClick={event => event.stopPropagation()} aria-label={`Move corner ${pointIndex + 1}`}><rect className="handle-outer" x="-5" y="-5" width="10" height="10" rx="2"/></g>)}
                    {editingRoom && addOutlinePointMode && outlinePoints.map(([x, y], pointIndex) => {
                      const [nextX, nextY] = outlinePoints[(pointIndex + 1) % outlinePoints.length];
                      const insertAt = pointIndex + 1;
                      return <g className="room-boundary-handle midpoint" key={`midpoint-${pointIndex}`} transform={`translate(${(x + nextX) / 2} ${(y + nextY) / 2})`} onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setOutlinePoints(current => [...current.slice(0, insertAt), [(x + nextX) / 2, (y + nextY) / 2], ...current.slice(insertAt)]); setDraggingOutlinePoint(insertAt); setAddOutlinePointMode(false); }} onPointerUp={event => { event.stopPropagation(); setDraggingOutlinePoint(null); }} onClick={event => event.stopPropagation()} aria-label={`Add and move boundary point ${insertAt}`}><circle className="handle-outer" r="3.5"/></g>;
                    })}
                  </g>;
                })}
              </g>}
              {geometry && showRooms && geometry.scale?.source === "dimensions" && <g className="dimension-layer">
                {(geometry.scale.dimensions ?? []).map((dim, index) => <g key={index}>
                  <line x1={dim.p0[0]} y1={dim.p0[1]} x2={dim.p1[0]} y2={dim.p1[1]} />
                  <circle cx={dim.p0[0]} cy={dim.p0[1]} r="4" /><circle cx={dim.p1[0]} cy={dim.p1[1]} r="4" />
                </g>)}
              </g>}
              {geometry && showWalls && !outlineMode && <g className="wall-layer">
                {geometry.wall_segments.map(segment => <line key={segment.id} className={segment.exterior ? "exterior" : "interior"} x1={segment.p0[0]} y1={segment.p0[1]} x2={segment.p1[0]} y2={segment.p1[1]}>
                  <title>{`${segment.exterior ? "Exterior" : "Interior"} wall · ${segment.length_m != null ? `${segment.length_m.toFixed(2)} m long, ${((segment.thickness_m ?? 0) * 100).toFixed(0)} cm thick` : `${Math.round(segment.length_px)} px long, ${Math.round(segment.thickness_px)} px thick`}`}</title>
                </line>)}
              </g>}
              {!outlineMode && citations.map((cited, index) => {
                const item = shownCitation(cited);
                const [x0, y0, x1, y1] = item.bbox;
                const points = item.polygon?.map(([x, y]) => `${x},${y}`).join(" ");
                const fullItem = evidenceItems.find(candidate => candidate.id === item.id);
                const label = fullItem ? evidenceLabel(fullItem, evidenceItems)
                  : item.category === "room" ? `${item.text || item.cls?.replaceAll("_", " ") || "Room"} boundary`
                  : item.category === "wall" ? "Wall boundary"
                  : item.raw?.replaceAll("_", " ") || "Evidence";
                const confidence = fullItem?.user_verified || item.user_verified ? "reviewed" : item.score != null ? `${Math.round(item.score * 100)}%` : null;
                const visibleLabel = confidence ? `${label} · ${confidence}` : label;
                const labelWidth = Math.max(48, Math.min(160, visibleLabel.length * 5.5 + 14));
                const labelX = Math.max(labelWidth / 2 + 3, Math.min(imageSize?.width ?? x0, (x0 + x1) / 2));
                const labelY = Math.max(11, Math.min(imageSize?.height ?? y0, y0 - 12));
                return <g className={`citation-mark${item.user_verified ? " user-corrected" : ""}`} key={`${item.id}-${index}`}>
                  {points ? <polygon points={points} /> : <rect x={x0} y={y0} width={Math.max(1, x1 - x0)} height={Math.max(1, y1 - y0)} />}
                  <g className="citation-label" aria-label={label}>
                    <rect x={labelX - labelWidth / 2} y={labelY - 9} width={labelWidth} height="18" rx="5" />
                    <text x={labelX} y={labelY + 1}>{visibleLabel.length > 25 ? `${visibleLabel.slice(0, 23)}…` : visibleLabel}</text>
                    <title>{visibleLabel}</title>
                  </g>
                </g>;
              })}
              {scaleMode && scalePoints.length > 0 && <g className="scale-mark"><line x1={scalePoints[0][0]} y1={scalePoints[0][1]} x2={(scalePoints[1] ?? scalePoints[0])[0]} y2={(scalePoints[1] ?? scalePoints[0])[1]} />{scalePoints.map(([x, y], index) => <g key={index}><circle cx={x} cy={y} r="10"/><text x={x} y={y}>{index + 1}</text></g>)}</g>}
              {measurePoints.length > 0 && <g className="scale-mark measurement-mark"><line x1={measurePoints[0][0]} y1={measurePoints[0][1]} x2={(measurePoints[1] ?? measurePoints[0])[0]} y2={(measurePoints[1] ?? measurePoints[0])[1]} />{measurePoints.map(([x, y], index) => <g className="measurement-handle" key={index} onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setDraggingMeasurePoint(index); }} onPointerUp={event => { event.stopPropagation(); setDraggingMeasurePoint(null); }} onClick={event => event.stopPropagation()}><circle cx={x} cy={y} r="10"/><text x={x} y={y}>{index + 1}</text></g>)}</g>}
              {measuredDistance !== null && measurePoints.length === 2 && <g className="measurement-label" pointerEvents="none"><rect x={(measurePoints[0][0] + measurePoints[1][0]) / 2 - Math.max(52, `${measuredDistance.toFixed(2)} ${scale?.unit}`.length * 7)} y={(measurePoints[0][1] + measurePoints[1][1]) / 2 - 15} width={Math.max(104, `${measuredDistance.toFixed(2)} ${scale?.unit}`.length * 14)} height="30" rx="7"/><text x={(measurePoints[0][0] + measurePoints[1][0]) / 2} y={(measurePoints[0][1] + measurePoints[1][1]) / 2 + 5}>{measuredDistance.toFixed(2)} {scale?.unit}</text></g>}
              {displayOutlines.filter(outline => outlineMode ? outline.id === editingOutlineId : pendingAreaQuestion || outline.scope !== "room").map((outline, index) => {
                const editing = outlineMode && outline.id === editingOutlineId;
                const polygon = editing ? outlinePoints : outline.polygon;
                if (polygon.length < 3) return null;
                const [labelX, labelY] = outlineLabelPoint(polygon);
                const area = areaM2(polygon);
                const label = area !== null ? `${area.toFixed(1)} m²` : `${outline.label ?? `Room ${index + 1}`} · scale needed`;
                const width = Math.max(104, label.length * 12);
                return <g className={`floor-outline-mark${outline.decision === "accepted" ? " accepted" : ""}${outline.decision === "rejected" ? " rejected" : ""}${editing ? " editing" : ""}`} key={outline.id}>
                  <polygon points={polygon.map(([x, y]) => `${x},${y}`).join(" ")} />
                  <g className="room-outline-label" pointerEvents="none"><rect x={labelX - width / 2} y={labelY - 18} width={width} height="36" rx="8"/><text x={labelX} y={labelY + 1}>{label}</text></g>
                  {editing && outlinePoints.map(([x, y], pointIndex) => <circle key={pointIndex} cx={x} cy={y} r="11" onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setDraggingOutlinePoint(pointIndex); }} onPointerUp={event => { event.stopPropagation(); setDraggingOutlinePoint(null); }} onClick={event => event.stopPropagation()} />)}
                </g>;
              })}
              {pendingAreaQuestion && !outlineMode && reviewedOutlines.filter(outline => outline.decision !== "accepted" && outline.polygon.length >= 3).map(outline => {
                const [x, y] = outline.polygon.reduce(([left, top], [px, py]) => [Math.min(left, px), Math.min(top, py)], [Infinity, Infinity] as [number, number]);
                return <foreignObject key={`review-controls-${outline.id}`} x={x + 8} y={y + 8} width="132" height="28" className="room-review-overlay">
                  <div className="room-review-chip" onClick={event => event.stopPropagation()}>
                    <div className="room-review-chip-actions">
                      <button className="toolbar-button" type="button" onClick={() => startOutlineReview(pendingAreaQuestion, outline.id)}>Adjust</button>
                      <button className="toolbar-button primary-button" type="button" disabled={outlineSaving || outline.decision === "accepted"} onClick={() => void saveOutlineDecision(outline, "accepted")}>{outline.decision === "accepted" ? "Accepted" : "Accept"}</button>
                      <button className="toolbar-button reject-button" type="button" disabled={outlineSaving || outline.decision === "rejected"} onClick={() => void saveOutlineDecision(outline, "rejected")}>{outline.decision === "rejected" ? "Rejected" : "Reject"}</button>
                    </div>
                  </div>
                </foreignObject>;
              })}
              {outlineMode && manualOutlineTarget?.scope !== "room" && outlinePoints.length >= 3 && !displayOutlines.some(outline => outline.id === editingOutlineId) && <g className="floor-outline-mark editing"><polygon points={outlinePoints.map(([x, y]) => `${x},${y}`).join(" ")} />{outlinePoints.map(([x, y], index) => <circle key={index} cx={x} cy={y} r="11" onPointerDown={event => { event.stopPropagation(); event.currentTarget.ownerSVGElement?.setPointerCapture(event.pointerId); setDraggingOutlinePoint(index); }} onPointerUp={event => { event.stopPropagation(); setDraggingOutlinePoint(null); }} onClick={event => event.stopPropagation()} />)}</g>}
            </svg>}{animationStage !== null && <div className="analysis-animation-caption" aria-live="polite"><strong>{["Tracing the structure", "Finding the rooms", "Spotting doors, windows & fixtures", "Analysis complete"][animationStage]}</strong><span>{["Walls are being mapped", "Rooms are taking shape", "Detected details are appearing", "Your plan is ready to explore"][animationStage]}</span></div>}{referenceMode && <div className="scale-hint">Click a detected item or drag around any area, then ask about it.</div>}{(scaleMode || measureMode) && <div className="scale-hint">{scaleMode ? (scalePoints.length === 0 ? "Step 1 of 2 · Click one end of a printed dimension" : scalePoints.length === 1 ? "Step 2 of 2 · Click the other end of that dimension" : "Step complete · Enter the printed distance below") : (measurePoints.length === 0 ? "Click a start point to measure" : "Click an end point to measure")}</div>}</div> : <div className="loading-state" role="status"><span className="loading-spinner" /><h1>Loading your floor plan</h1><p>Getting your image ready…</p></div>
          )}
        </section>
        <aside className="chat-panel" aria-label="Plan questions">
          {ready && !scale && !detectedScaleAccepted && geometry?.scale && ["dimensions", "written_areas"].includes(geometry.scale.source) && <section className="scale-form" aria-label="Confirm detected drawing scale"><div><strong>We found a drawing scale</strong><span>{geometry.scale.source === "dimensions" ? `OCR matched ${geometry.scale.dimensions?.length ?? 0} printed dimensions` : "The written room areas agree with the plan geometry"}. Estimate: {geometry.scale.m_per_px.toFixed(5)} metres per pixel{geometry.scale.warning ? ` · ${geometry.scale.warning}` : ""}. Is this right?</span></div><button className="toolbar-button primary-button" type="button" disabled={detectedScaleSaving} onClick={() => void confirmDetectedScale()}>{detectedScaleSaving ? "Saving…" : "Use this scale"}</button><button className="toolbar-button" type="button" onClick={startScaleCalibration}>Correct scale</button>{scaleError && <p role="alert">{scaleError}</p>}</section>}
          {scaleMode && scalePoints.length === 2 && <form className="scale-form" onSubmit={saveScale}><div><strong>Enter the printed distance</strong><span>Use the value shown between the points on the plan, then choose its unit.</span></div><input aria-label="Known distance" type="number" min="0.001" step="any" placeholder="Distance" value={scaleDistance} onChange={event => setScaleDistance(event.target.value)} required /><select aria-label="Distance unit" value={scaleUnit} onChange={event => setScaleUnit(event.target.value as ScaleCalibration["unit"])}><option value="m">metres</option><option value="cm">centimetres</option><option value="ft">feet</option><option value="in">inches</option></select><button className="toolbar-button primary-button" disabled={scaleSaving || !scaleDistance}>{scaleSaving ? "Saving…" : "Save scale"}</button>{scaleError && <p role="alert">{scaleError}</p>}</form>}
          {pendingAreaQuestion && <section className="room-review" aria-label="Review room areas">
            <div className="room-review-heading"><strong>Review detected {getRequestedRoomType(pendingAreaQuestion)?.label.toLowerCase() ?? "room"} areas</strong><span>Each outline is separate. Check the painted area; adjust points only where needed.</span></div>
            {reviewedOutlines.length > 0 && <p className="room-review-placement">Use the controls placed inside each room on the plan.</p>}
            {getReviewOutlines(pendingAreaQuestion).length === 0 && <p className="room-review-empty">No room outline was detected. You can draw one on the plan.</p>}
            <div className="room-review-footer">{scale || detectedScaleAccepted ? `${reviewedAccepted.length} accepted · ${reviewedAcceptedArea.toFixed(1)} m² total` : "Set a drawing scale to show room areas"}{getReviewOutlines(pendingAreaQuestion).length === 0 && <button className="toolbar-button" type="button" onClick={() => startOutlineReview(pendingAreaQuestion)}>Draw room outline</button>}</div>
            {outlineError && <p role="alert">{outlineError}</p>}
          </section>}
          {outlineMode && <form className="outline-form" onSubmit={saveFloorOutline}><div className="outline-copy"><strong>{manualOutlineTarget?.scope === "room" ? `Adjust ${manualOutlineTarget.label?.toLowerCase() ?? "room"} boundary` : "Adjust whole-floor boundary"}</strong><span>Drag a square corner handle to reshape the boundary. Select Add point to show edge handles and add a corner.</span></div><div className="outline-actions"><button className={`toolbar-button${addOutlinePointMode ? " is-on" : ""}`} type="button" onClick={() => setAddOutlinePointMode(on => !on)}>{addOutlinePointMode ? "Choose an edge" : "Add point"}</button><button className="toolbar-button" type="button" onClick={() => setOutlinePoints(current => current.slice(0, -1))} disabled={outlinePoints.length <= 3}>Remove last</button><button className="toolbar-button primary-button" type="submit" disabled={outlineSaving || outlinePoints.length < 3}>{outlineSaving ? "Saving…" : "Save boundary"}</button><button className="toolbar-button" type="button" onClick={() => { setOutlineMode(false); setEditingOutlineId(null); }}>Cancel</button></div>{(scale || detectedScaleAccepted) && areaM2(outlinePoints) !== null && <p>Area on plan: {areaM2(outlinePoints)?.toFixed(1)} m²</p>}{!scale && !detectedScaleAccepted && <p>Set a drawing scale to calculate the area.</p>}{outlineError && <p role="alert">{outlineError}</p>}</form>}
          {!pendingAreaQuestion && !outlineMode && savedOutlines.length > 0 && <div className="area-summary"><strong>Accepted outline areas</strong><span>{acceptedArea > 0 ? `Estimated total ${acceptedArea.toFixed(1)} m²` : "Set a drawing scale to calculate area"}</span><button className="toolbar-button" type="button" onClick={removeSavedOutline} disabled={outlineRemoving}>{outlineRemoving ? "Removing…" : "Remove outlines"}</button>{outlineError && <p role="alert">{outlineError}</p>}</div>}
          {correctionMode && <section className="correction-panel" aria-label="Review detected plan items">
            <div className="correction-heading"><strong>Review detections</strong><span>{correctionEvents.length ? `${correctionEvents.length} saved correction${correctionEvents.length === 1 ? "" : "s"} · used in this plan’s answers` : "Select a box, drag it to move, or drag its corner handle to resize it."}</span></div>
            {getAllRoomOutlines().length > 0 && <div className="correction-editor"><strong>Backend room boundaries</strong><span>These are the model’s suggested room shapes. Adjust a boundary to use your edit for both room answers and area estimates.</span>{getAllRoomOutlines().map((outline, index) => <div className="room-boundary-row" key={outline.id}><span>{outline.label || `Room ${index + 1}`}</span><button className="toolbar-button" type="button" onClick={() => startOutlineReview(undefined, outline.id)}>Adjust boundary</button></div>)}</div>}
            <div className="correction-list">{reviewItems.filter(item => ["room", "door", "window", "fixture"].includes(item.category)).map(item => <button type="button" className={`correction-row${selectedEvidenceId === item.id ? " active" : ""}`} key={item.id} onClick={() => beginCorrection(item)}><span>{evidenceLabel(item, evidenceItems)}</span><small>{item.text ? `${item.category} · label “${item.text}”` : item.category}{item.user_verified ? " · reviewed by you" : item.score != null ? ` · ${Math.round(item.score * 100)}% confidence` : ""}</small></button>)}</div>
            {selectedEvidence && <div className="correction-editor"><strong>Correct selected item</strong><span>Drag the box to move it, or drag its lower-right handle to resize it. If this says “toilet” but is actually a kitchen, choose room and kitchen. Save to keep the geometry and label correction with this plan.</span><label>Kind<select value={editCategory} onChange={event => { const category = event.target.value; setEditCategory(category); setEditClass(itemTypes[category][0]); }}>{Object.keys(itemTypes).map(category => <option key={category} value={category}>{category}</option>)}</select></label><label>Room or object type<select value={editClass} onChange={event => setEditClass(event.target.value)}>{(itemTypes[editCategory] ?? ["other"]).map(type => <option key={type} value={type}>{type.replaceAll("_", " ")}</option>)}</select></label><div className="correction-actions"><button className="toolbar-button primary-button" type="button" disabled={correctionSaving || (selectedEvidence.category === editCategory && selectedEvidence.cls === editClass)} onClick={() => void saveCorrection("update", selectedEvidence.id, { ...selectedEvidence, category: editCategory, cls: editClass })}>{correctionSaving ? "Saving…" : "Save correction"}</button><button className="toolbar-button" type="button" disabled={correctionSaving} onClick={() => void saveCorrection("delete", selectedEvidence.id)}>Remove detection</button></div></div>}
            <div className="correction-editor"><strong>Add a missed item</strong><label>Kind<select value={addCategory} onChange={event => { const category = event.target.value; setAddCategory(category); setAddClass(itemTypes[category]?.[0] ?? "other"); setCorrectionAddPoints([]); }}>{Object.keys(itemTypes).map(category => <option key={category} value={category}>{category}</option>)}</select></label><label>Type<select value={addClass} onChange={event => setAddClass(event.target.value)}>{(itemTypes[addCategory] ?? ["other"]).map(type => <option key={type} value={type}>{type.replaceAll("_", " ")}</option>)}</select></label><button className="toolbar-button" type="button" disabled={correctionSaving} onClick={() => { setCorrectionAddPoints([]); setCorrectionAddMode(true); }}>{correctionAddMode ? addCategory === "room" ? correctionAddPoints.length === 0 ? "Click first room corner" : "Click opposite room corner" : "Click its location on the plan" : addCategory === "room" ? "Mark missing room" : "Place missing item"}</button></div>
            {correctionError && <p className="correction-error" role="alert">{correctionError}</p>}
          </section>}
          <div className="chat-header"><div><h1>Ask about this plan</h1><p>Questions, quantities, and room details</p></div></div>
          <div className="conversation" aria-live="polite" ref={conversationRef}>{messages.length === 0 && !sending ? <div className="empty-conversation"><div className="chat-mark" aria-hidden="true">↗</div><p>{ready ? "Plan analysis is ready." : "Preparing the saved plan analysis…"}</p><span>Ask about quantities, rooms, and detected items.</span></div> : <>{messages.map((message, index) => <div className={`chat-message ${message.role}`} key={`${index}-${message.role}`}><span>{message.role === "user" ? "You" : "Stoa"}</span><p>{message.text}</p>{message.role === "assistant" && message.confidence === "low" && <><small className="answer-confidence">Lower confidence · the answer may include uncertain details. Check the highlighted items on the drawing.</small>{message.evidence?.length ? <section className="evidence-review" aria-label="Evidence to review"><strong>Items used for this answer</strong><p className="evidence-explanation">Detection confidence describes how sure the system is about each item. Answer confidence also depends on how those items support the answer.</p><button type="button" className="evidence-focus" onClick={() => showMessageEvidence(message)}>Show {message.evidence.length} item{message.evidence.length === 1 ? "" : "s"} on drawing</button><ul>{message.evidence.map((item, evidenceIndex) => <li key={`${item.id}-${evidenceIndex}`}>{itemLabel({ category: item.category ?? "item", cls: item.cls ?? "", text: item.text })}{item.text ? ` · label “${item.text}”` : ""}{item.score != null ? ` · ${Math.round(item.score * 100)}% detection confidence` : ""}</li>)}</ul>{message.evidenceDecision ? <small className={`evidence-decision ${message.evidenceDecision}`}>{message.evidenceDecision === "trusted" ? "You confirmed these items support the answer." : "You said these items do not support the answer."}</small> : <div className="evidence-actions"><button type="button" onClick={() => void submitFeedback(index, "trusted")}>These items support the answer</button><button type="button" onClick={() => void submitFeedback(index, "rejected")}>They don’t support it</button></div>}<small className="evidence-feedback-note">Your response is saved as feedback for this answer.</small></section> : null}</>}{message.role === "assistant" && index > 0 && <div className="answer-feedback">{message.feedback ? <small>Thanks for the feedback.</small> : <><button type="button" onClick={() => void submitFeedback(index, "helpful")}>Helpful</button><button type="button" onClick={() => void submitFeedback(index, "incorrect")}>Needs correction</button><button type="button" onClick={() => void submitFeedback(index, "unclear")}>Unclear</button></>}</div>}</div>)}{sending && <div className="chat-message assistant thinking-message" role="status"><span>Stoa</span><p><i className="thinking-spinner" aria-hidden="true" />Thinking through the plan…</p></div>}</>}</div>
          {feedbackError && <p className="feedback-error" role="alert">{feedbackError}</p>}
          {measurementRequested && <section className="requested-action" aria-label="Requested measurement">
            <div><strong>Measure a distance</strong><span>Choose the two points you want measured on the plan.</span></div>
            <button className="toolbar-button primary-button" type="button" onClick={startMeasurement}>Choose points</button>
          </section>}
          <form className="composer" onSubmit={submitQuestion}>{drawingReferences.length > 0 && <div className="reference-chips" aria-label="Drawing references">{drawingReferences.map((reference, index) => <span className="reference-chip" key={`reference-chip-${index}`}>{index + 1}. {reference.label}<button type="button" aria-label={`Remove reference ${index + 1}`} onClick={() => setDrawingReferences(current => current.filter((_, itemIndex) => itemIndex !== index))}>×</button></span>)}</div>}<textarea aria-label="Ask a question about the plan" placeholder={drawingReferences.length ? "Ask about the selected part…" : "Ask about quantities, rooms, or materials…"} value={question} onChange={event => setQuestion(event.target.value)} onKeyDown={event => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); event.currentTarget.form?.requestSubmit(); } }} disabled={!ready || sending} /><div className="composer-footer"><span>{sending ? "Thinking…" : ready ? "Answers use the analysis saved for this plan" : "Available when analysis finishes"}</span><button className="send-button" type="submit" disabled={!ready || sending || !question.trim()} aria-label="Send question">↑</button></div></form>
        </aside>
      </div>
    </main>
  );
}
