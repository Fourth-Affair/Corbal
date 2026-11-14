"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Point = [number, number];
type Room = {
  id: string;
  name: string;
  kind: string;
  area: number;
  polygon: Point[];
  color: string;
  confidence: number;
};

// Mock API response: geometry uses a 1000 × 650 source-image pixel space.
const floorPlan = {
  building: "Northstar House",
  address: "24 Riverside Walk, Helsinki",
  floor: "Level 02",
  source: { width: 1000, height: 650, label: "Architectural plan · rev. 03" },
  rooms: [
    { id: "R-201", name: "Open workspace", kind: "Workspace", area: 42.8, polygon: [[100,115],[440,115],[440,365],[100,365]], color: "#d8e9df", confidence: .98 },
    { id: "R-202", name: "Meeting room", kind: "Meeting room", area: 18.4, polygon: [[440,115],[640,115],[640,265],[440,265]], color: "#e8e2ce", confidence: .96 },
    { id: "R-203", name: "Focus room", kind: "Focus room", area: 10.2, polygon: [[640,115],[770,115],[770,265],[640,265]], color: "#e8e2ce", confidence: .91 },
    { id: "R-204", name: "Kitchen", kind: "Kitchen", area: 16.1, polygon: [[770,115],[900,115],[900,365],[770,365]], color: "#dce6ec", confidence: .94 },
    { id: "R-205", name: "Reception", kind: "Reception", area: 29.6, polygon: [[100,365],[360,365],[360,535],[100,535]], color: "#dce6ec", confidence: .97 },
    { id: "R-206", name: "Phone booth", kind: "Phone booth", area: 4.2, polygon: [[360,365],[440,365],[440,445],[360,445]], color: "#e8e2ce", confidence: .89 },
    { id: "R-207", name: "Lounge", kind: "Lounge", area: 22.5, polygon: [[440,265],[770,265],[770,445],[440,445]], color: "#d8e9df", confidence: .95 },
    { id: "R-208", name: "Bathroom", kind: "Bathroom", area: 14.8, polygon: [[770,365],[900,365],[900,535],[770,535]], color: "#dce6ec", confidence: .93 },
    { id: "R-209", name: "Stair & lift", kind: "Circulation", area: 12.6, polygon: [[360,445],[770,445],[770,535],[360,535]], color: "#e6e5df", confidence: .98 },
  ] as Room[],
};

function PlanDrawing({ selected, setSelected, overlay }: { selected: string; setSelected: (id: string) => void; overlay: number }) {
  const walls = [
    [100,115,900,115],[100,535,900,535],[100,115,100,535],[900,115,900,535],
    [440,115,440,535],[640,115,640,265],[770,115,770,535],[100,365,440,365],
    [440,265,770,265],[770,365,900,365],[360,365,360,535],[360,445,770,445],
  ];
  return <svg className="plan-svg" viewBox="0 0 1000 650" role="img" aria-label="Interactive floor plan">
    <defs><pattern id="grid" width="24" height="24" patternUnits="userSpaceOnUse"><path d="M 24 0 L 0 0 0 24" fill="none" stroke="#e9e9e3" strokeWidth="1" /></pattern></defs>
    <rect width="1000" height="650" fill="#f8f8f5" />
    <rect width="1000" height="650" fill="url(#grid)" />
    {floorPlan.rooms.map(room => {
      const points = room.polygon.map(p => p.join(",")).join(" ");
      const cx = room.polygon.reduce((n,p)=>n+p[0],0)/room.polygon.length;
      const cy = room.polygon.reduce((n,p)=>n+p[1],0)/room.polygon.length;
      return <g key={room.id} onClick={() => setSelected(room.id)} className="room-shape" tabIndex={0} role="button" aria-label={`Select ${room.name}`} onKeyDown={e => e.key === "Enter" && setSelected(room.id)}>
        <polygon points={points} fill={selected === room.id ? "#b9d7c5" : room.color} fillOpacity={0.76 - overlay * .45} stroke={selected === room.id ? "#276c4b" : "#9eb4a5"} strokeWidth={selected === room.id ? 4 : 2} />
        <text x={cx} y={cy-5} textAnchor="middle" className="room-label">{room.name}</text>
        <text x={cx} y={cy+18} textAnchor="middle" className="room-area">{room.area.toFixed(1)} m²</text>
      </g>;
    })}
    <g className="wall-lines">{walls.map(([x1,y1,x2,y2],i)=><line key={i} x1={x1} y1={y1} x2={x2} y2={y2} />)}</g>
    {/* Door symbols */}
    <g className="doors"><path d="M440 178 A52 52 0 0 1 492 230 M440 178 L492 178"/><path d="M640 178 A42 42 0 0 1 682 220 M640 178 L682 178"/><path d="M440 312 A44 44 0 0 0 484 268 M440 312 L484 312"/><path d="M360 410 A35 35 0 0 1 395 375 M360 410 L395 410"/><path d="M770 402 A40 40 0 0 1 810 362 M770 402 L810 402"/></g>
    {/* Furniture hints */}
    <g className="furniture"><rect x="150" y="180" width="155" height="64" rx="4"/><line x1="150" y1="212" x2="305" y2="212"/><rect x="150" y="278" width="155" height="48" rx="4"/><rect x="500" y="150" width="80" height="38" rx="18"/><circle cx="495" cy="169" r="7"/><circle cx="585" cy="169" r="7"/><circle cx="495" cy="188" r="7"/><circle cx="585" cy="188" r="7"/><rect x="515" y="317" width="180" height="52" rx="24"/><rect x="130" y="410" width="155" height="42" rx="5"/><rect x="797" y="155" width="74" height="82" rx="4"/><rect x="797" y="270" width="74" height="68" rx="4"/></g>
    <g className="scale-mark"><line x1="820" y1="590" x2="900" y2="590"/><line x1="820" y1="584" x2="820" y2="596"/><line x1="900" y1="584" x2="900" y2="596"/><text x="860" y="612" textAnchor="middle">5 m</text></g>
  </svg>;
}

function SourceDrawing() {
  const walls = [[100,115,900,115],[100,535,900,535],[100,115,100,535],[900,115,900,535],[440,115,440,535],[640,115,640,265],[770,115,770,535],[100,365,440,365],[440,265,770,265],[770,365,900,365],[360,365,360,535],[360,445,770,445]];
  return <svg className="plan-svg source-svg" viewBox="0 0 1000 650" aria-label="Mock original architectural drawing">
    <rect width="1000" height="650" fill="#f4f0e6" />
    <g className="source-grid"><path d="M60 85H940M60 565H940M75 95V550M925 95V550"/><path d="M100 95V115M440 95V115M640 95V115M770 95V115M900 95V115M100 78H900"/><path d="M80 115H100M80 365H100M80 535H100M62 115V535"/></g>
    <g className="source-dims"><text x="260" y="72">8 000</text><text x="520" y="72">2 000</text><text x="705" y="72">1 300</text><text x="865" y="72">1 300</text><text x="49" y="255" transform="rotate(-90 49 255)">4 200</text><text x="49" y="470" transform="rotate(-90 49 470)">2 800</text></g>
    {floorPlan.rooms.map(r=><polygon key={r.id} points={r.polygon.map(p=>p.join(",")).join(" ")} fill="transparent" stroke="none"/>)}
    <g className="source-walls">{walls.map(([x1,y1,x2,y2],i)=><line key={i} x1={x1} y1={y1} x2={x2} y2={y2}/>)}</g>
    <g className="source-furniture"><rect x="150" y="180" width="155" height="64"/><line x1="150" y1="212" x2="305" y2="212"/><rect x="150" y="278" width="155" height="48"/><rect x="500" y="150" width="80" height="38" rx="18"/><rect x="515" y="317" width="180" height="52" rx="24"/><rect x="130" y="410" width="155" height="42"/><rect x="797" y="155" width="74" height="82"/><rect x="797" y="270" width="74" height="68"/></g>
    <g className="source-notes"><text x="175" y="160">OPEN WORKSPACE</text><text x="484" y="143">MEETING</text><text x="790" y="145">KITCHEN</text><text x="117" y="396">RECEPTION</text><text x="820" y="577">N</text><path d="M824 570v-22m0 0-5 8m5-8 5 8"/></g>
    <g className="source-stamp"><rect x="735" y="585" width="205" height="45"/><path d="M735 600h205M830 585v45"/><text x="745" y="596">NORTHSTAR HOUSE · LEVEL 02</text><text x="745" y="618">FLOOR PLAN</text><text x="840" y="618">REV 03</text></g>
  </svg>;
}

export default function Home() {
  const [selected, setSelected] = useState("R-201");
  const [overlay, setOverlay] = useState(0.42);
  const [showComparison, setShowComparison] = useState(false);
  const [showRooms, setShowRooms] = useState(false);
  const [roomQuery, setRoomQuery] = useState("");
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<"estimate" | "review" | "room" | null>(null);
  const [waste, setWaste] = useState(10);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [uploadedPreview, setUploadedPreview] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const room = floorPlan.rooms.find(r => r.id === selected)!;
  const filteredRooms = useMemo(() => floorPlan.rooms.filter(r => `${r.name} ${r.kind}`.toLowerCase().includes(roomQuery.toLowerCase())), [roomQuery]);
  const kitchen = floorPlan.rooms.find(r => r.kind === "Kitchen")!;
  const bathroom = floorPlan.rooms.find(r => r.kind === "Bathroom")!;
  const tileCount = Math.ceil((kitchen.area * (1 + waste / 100)) / 0.36) + Math.ceil((bathroom.area * (1 + waste / 100)) / 0.36);

  useEffect(() => {
    if (!uploadedFile || !uploadedFile.type.startsWith("image/")) {
      setUploadedPreview(null);
      return;
    }
    const url = URL.createObjectURL(uploadedFile);
    setUploadedPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [uploadedFile]);

  const ask = (text: string) => {
    setQuestion(text);
    if (!text.trim()) return;
    if (/tile|flooring|floor|material|quantity|how many/i.test(text)) {
      setAnswer("estimate");
      setSelected(kitchen.id);
    }
    else if (/review|measurement|measure|confidence|check/i.test(text)) setAnswer("review");
    else {
      const found = floorPlan.rooms.find(r => `${r.name} ${r.kind}`.toLowerCase().includes(text.toLowerCase()));
      if (found) setSelected(found.id);
      setAnswer("room");
    }
  };

  const handleUpload = (file: File | undefined) => {
    if (!file) return;
    setUploadedFile(file);
    setShowComparison(false);
    setAnswer(null);
  };

  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><span/><span/><span/><span/></div><span>planform</span></div>
      <div className="workspace-label">PROJECT</div>
      <button className="building-switch"><span className="building-icon">⌂</span><span><b>{floorPlan.building}</b><small>{floorPlan.address}</small></span><span className="chevron">⌄</span></button>
      <div className="side-section"><div className="side-heading">PLANS <button aria-label="Add floor" onClick={() => fileInput.current?.click()}>＋</button></div>
        <button className="floor-link active"><span className="floor-glyph">▤</span> Level 02 <span className="floor-count">9</span></button>
        <button className="floor-link"><span className="floor-glyph">▤</span> Level 01</button>
        <button className="floor-link"><span className="floor-glyph">▤</span> Basement</button>
      </div>
      <div className="sidebar-bottom"><div className="help-card"><div className="help-icon">?</div><b>Project workspace</b><p>Review drawings and estimate materials.</p><button>Plan help <span>↗</span></button></div><div className="user-row"><div className="avatar">EL</div><span><b>Eva Lindström</b><small>Project manager</small></span><span className="more">•••</span></div></div>
    </aside>

    <section className="main-area">
      <header className="topbar"><div className="breadcrumbs">Buildings <span>/</span> {floorPlan.building} <span>/</span> <b>{floorPlan.floor}</b></div><div className="top-actions"><button className="icon-button" aria-label="Notifications">♧<i/></button><div className="top-avatar">EL</div></div></header>
      <div className="content">
        <input ref={fileInput} className="visually-hidden" type="file" accept="image/png,image/jpeg,application/pdf" onChange={e => handleUpload(e.target.files?.[0])}/>
        <div className="page-heading"><div><div className="title-kicker"><span className="live-dot"/> CONSTRUCTION PLAN</div><h1>{floorPlan.floor}</h1><p className="subheading">Ask for quantities, then check the estimate against the plan.</p></div><div className="heading-buttons"><button className="button secondary" onClick={() => fileInput.current?.click()}>Upload plan</button></div></div>
        <div className="plan-readiness"><div><span className="readiness-dot"/><span><b>{uploadedFile ? uploadedFile.name : "Sample plan ready"}</b><small>{uploadedFile ? "Selected locally · analysis connection pending" : "9 spaces identified · scale set to metres"}</small></span></div><span className="readiness-state">{uploadedFile ? "PREVIEW ONLY" : "READY TO ESTIMATE"}</span></div>
        <div className="workspace-grid">
          <section className="plan-card">
            <div className="plan-toolbar"><div><b>Floor plan</b><span className="source-tag"><i/> {uploadedFile?.name ?? floorPlan.source.label}</span></div><button className="square-button" aria-label="Developer plan tools" title="Developer plan tools" onClick={() => setShowComparison(v => !v)}>···</button></div>
            {showComparison && <div className="dev-tools"><span>Developer inspection</span><label>Source overlay <input aria-label="Original drawing opacity" type="range" min="0" max="100" value={Math.round(overlay*100)} onChange={e => setOverlay(Number(e.target.value)/100)}/><b>{Math.round(overlay*100)}%</b></label></div>}
            <div className={`canvas-wrap ${uploadedPreview ? "uploaded-canvas" : ""}`}>
              <div className="canvas-badge"><span className="badge-dot"/>{uploadedPreview ? "Local image preview" : "Detected spaces"}</div>
              {uploadedPreview ? <img className="uploaded-plan-image" src={uploadedPreview} alt="Selected floor plan preview"/> : uploadedFile ? <div className="upload-pdf-state"><b>{uploadedFile.name}</b><span>PDF selected. Plan analysis will appear when the processing service is connected.</span></div> : <div className="plan-canvas"><div className="plan-layer"><PlanDrawing selected={selected} setSelected={setSelected} overlay={0}/></div>{showComparison && <div className="blueprint-layer" style={{opacity: overlay}} aria-hidden="true"><SourceDrawing/></div>}</div>}
              {!uploadedFile && <><div className="zoom-controls"><button aria-label="Zoom in">＋</button><button aria-label="Zoom out">−</button><div/><button aria-label="Reset zoom">⌖</button></div><div className="north-marker">N <span>↑</span></div></>}
            </div>
            <div className="plan-footer"><span><span className="legend-dot"/> Detected space</span><span>{uploadedFile ? "Image stays on this device in this preview" : `${floorPlan.rooms.length} spaces · ${floorPlan.source.width} × ${floorPlan.source.height} px source`}</span><button onClick={() => setShowRooms(v => !v)}>{showRooms ? "Hide room list" : "Browse rooms"}</button></div>
            {showRooms && <div className="room-directory"><label className="room-search"><span>⌕</span><input placeholder="Search spaces" value={roomQuery} onChange={e=>setRoomQuery(e.target.value)}/></label><div className="room-list">{filteredRooms.map(r=><button key={r.id} className={`room-row ${selected===r.id?"chosen":""}`} onClick={()=>setSelected(r.id)}><span className="room-color" style={{background:r.color}}/><span className="room-row-name">{r.name}<small>{r.kind}</small></span><span className="room-row-area">{r.area.toFixed(1)} m²</span></button>)}</div></div>}
          </section>
          <aside className="assistant-panel">
            <div className="assistant-heading"><div><div className="title-kicker">PLAN ASSISTANT</div><h2>What do you need to estimate?</h2></div><span className="sample-label">SAMPLE DATA</span></div>
            <form className="assistant-form" onSubmit={e=>{e.preventDefault();ask(question)}}><label htmlFor="plan-question">Ask a question</label><textarea id="plan-question" value={question} onChange={e=>setQuestion(e.target.value)} placeholder="e.g. How many tiles for the kitchen and bathroom?" disabled={Boolean(uploadedFile)}/><button className="button primary" type="submit" disabled={Boolean(uploadedFile) || !question.trim()}>Get estimate</button></form>
            {uploadedFile ? <div className="pending-analysis"><b>Analysis service not connected</b><p>Your file is selected for this frontend preview. Connect the plan processor to identify spaces and answer questions about this drawing.</p></div> : answer === "estimate" ? <div className="estimate-result" aria-live="polite"><div className="estimate-meta"><span>QUANTITY ESTIMATE</span><span>Preview answer</span></div><strong>{tileCount} <em>tiles</em></strong><p className="estimate-caption">for the kitchen and bathroom</p><div className="estimate-rooms"><button onClick={()=>setSelected(kitchen.id)}><span>{kitchen.name}</span><b>{Math.ceil(kitchen.area*(1+waste/100)/.36)} tiles</b><small>{kitchen.area.toFixed(1)} m² · {Math.round(kitchen.confidence*100)}% recognition</small></button><button onClick={()=>setSelected(bathroom.id)}><span>{bathroom.name}</span><b>{Math.ceil(bathroom.area*(1+waste/100)/.36)} tiles</b><small>{bathroom.area.toFixed(1)} m² · {Math.round(bathroom.confidence*100)}% recognition</small></button></div><label className="waste-control">Extra for cuts and breakage <span>{waste}%</span><input aria-label="Extra tiles for cuts and breakage" type="range" min="0" max="20" step="1" value={waste} onChange={e=>setWaste(Number(e.target.value))}/></label><p className="assumption-note">Assumes 600 × 600 mm tiles and rounds up per room. Confirm site measurements before ordering.</p></div> : answer === "review" ? <div className="estimate-result" aria-live="polite"><div className="estimate-meta"><span>PLAN REVIEW</span><span>1 space to check</span></div><strong className="review-result-title">Review before estimating</strong><p className="estimate-caption">One room has lower recognition confidence in this sample.</p><div className="estimate-rooms"><button onClick={()=>setSelected("R-206")}><span>Phone booth</span><b>89% recognition · check boundary</b></button></div><p className="assumption-note">Confirm unclear boundaries before using room measurements for a material order.</p></div> : answer === "room" ? <div className="estimate-result" aria-live="polite"><div className="estimate-meta"><span>SPACE DETAILS</span><span>Preview answer</span></div><strong className="review-result-title">{room.name}: {room.area.toFixed(1)} m²</strong><p className="estimate-caption">Recognition confidence: {Math.round(room.confidence*100)}%. Select the space on the plan to inspect its location.</p></div> : <div className="suggested-questions"><span>TRY ASKING</span><button onClick={()=>ask("How many tiles for the kitchen and bathroom?")}>How many tiles for the kitchen and bathroom?</button><button onClick={()=>ask("What spaces need review?")}>Which spaces need a measurement check?</button><button onClick={()=>ask(`How large is the ${room.name.toLowerCase()}?`)}>How large is {room.name.toLowerCase()}?</button></div>}
            <div className="selected-context"><span>PLAN CONTEXT</span><b>{room.name}</b><small>{room.area.toFixed(1)} m² · {Math.round(room.confidence*100)}% recognition</small></div>
          </aside>
        </div>
        <footer className="page-footer"><span>Planform <span>·</span> Built for construction teams</span><span>Estimate results use sample data in this preview</span></footer>
      </div>
    </section>
  </main>;
}
