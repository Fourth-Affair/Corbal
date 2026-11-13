"use client";

import { useMemo, useState } from "react";

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
    { id: "R-208", name: "Services", kind: "Services", area: 14.8, polygon: [[770,365],[900,365],[900,535],[770,535]], color: "#dce6ec", confidence: .93 },
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

export default function Home() {
  const [selected, setSelected] = useState("R-201");
  const [overlay, setOverlay] = useState(0);
  const [activeTab, setActiveTab] = useState("Plan");
  const [query, setQuery] = useState("");
  const [answer, setAnswer] = useState("");
  const room = floorPlan.rooms.find(r => r.id === selected)!;
  const filteredRooms = useMemo(() => floorPlan.rooms.filter(r => `${r.name} ${r.kind}`.toLowerCase().includes(query.toLowerCase())), [query]);
  const ask = (text: string) => {
    setQuery(text);
    const found = floorPlan.rooms.find(r => `${r.name} ${r.kind}`.toLowerCase().includes(text.toLowerCase()));
    if (found) { setSelected(found.id); setAnswer(`${found.name} is on ${floorPlan.floor}. It covers ${found.area.toFixed(1)} m².`); }
    else setAnswer("This demo can answer questions about rooms and areas on this floor. Try asking about the kitchen or meeting room.");
  };
  return <main className="app-shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark"><span/><span/><span/><span/></div><span>planform</span></div>
      <div className="workspace-label">WORKSPACE</div>
      <button className="building-switch"><span className="building-icon">⌂</span><span><b>{floorPlan.building}</b><small>{floorPlan.address}</small></span><span className="chevron">⌄</span></button>
      <div className="side-section"><div className="side-heading">YOUR BUILDING <button aria-label="Add floor">＋</button></div>
        <button className="floor-link"><span className="floor-glyph">▦</span> Overview</button>
        <button className="floor-link active"><span className="floor-glyph">▤</span> Level 02 <span className="floor-count">9</span></button>
        <button className="floor-link"><span className="floor-glyph">▤</span> Level 01</button>
        <button className="floor-link"><span className="floor-glyph">▤</span> Basement</button>
      </div>
      <div className="sidebar-bottom"><div className="help-card"><div className="help-icon">✳</div><b>Need a hand?</b><p>Get more from your building data.</p><button>Explore guide <span>↗</span></button></div><div className="user-row"><div className="avatar">EL</div><span><b>Eva Lindström</b><small>Building manager</small></span><span className="more">•••</span></div></div>
    </aside>

    <section className="main-area">
      <header className="topbar"><div className="breadcrumbs">Buildings <span>/</span> {floorPlan.building} <span>/</span> <b>{floorPlan.floor}</b></div><div className="top-actions"><button className="icon-button" aria-label="Notifications">♧<i/></button><div className="top-avatar">EL</div></div></header>
      <div className="content">
        <div className="page-heading"><div><div className="title-kicker"><span className="live-dot"/> BUILDING PLAN</div><h1>{floorPlan.floor}</h1><p className="subheading">Explore spaces, understand your building.</p></div><div className="heading-buttons"><button className="button secondary">↗ <span>Share</span></button><button className="button primary">＋ <span>Add a floor plan</span></button></div></div>
        <div className="summary-row"><div className="summary-card"><div className="summary-icon green">▧</div><div><small>ROOMS DETECTED</small><strong>09 <em>spaces</em></strong></div></div><div className="summary-card"><div className="summary-icon blue">⌗</div><div><small>TOTAL FLOOR AREA</small><strong>171.2 <em>m²</em></strong></div></div><div className="summary-card"><div className="summary-icon gold">◷</div><div><small>LAST UPDATED</small><strong>Today <em>· 10:42</em></strong></div></div><div className="processing-status"><span className="check">✓</span><span><b>Plan processed</b><small>All spaces are ready to explore</small></span></div></div>
        <div className="workspace-grid">
          <section className="plan-card">
            <div className="plan-toolbar"><div className="tab-switch"><button className={activeTab === "Plan" ? "selected" : ""} onClick={()=>setActiveTab("Plan")}>Digital plan</button><button className={activeTab === "Original" ? "selected" : ""} onClick={()=>setActiveTab("Original")}>Original</button></div><div className="toolbar-right"><span className="source-tag"><i/> {floorPlan.source.label}</span><button className="square-button" title="Fit to view">⤢</button><button className="square-button" title="More options">···</button></div></div>
            <div className="canvas-wrap"><div className="canvas-badge"><span className="badge-dot"/>{activeTab === "Original" ? "Original drawing" : overlay > 0 ? "Overlay comparison" : "Digitalized plan"}</div><div className="plan-canvas"><div className="plan-layer" style={{opacity: activeTab === "Original" ? .16 : 1}}><PlanDrawing selected={selected} setSelected={setSelected} overlay={overlay}/></div><div className="blueprint-layer" style={{opacity: activeTab === "Original" ? .68 : overlay}} aria-hidden="true"><PlanDrawing selected="" setSelected={()=>{}} overlay={0}/></div></div><div className="zoom-controls"><button aria-label="Zoom in">＋</button><button aria-label="Zoom out">−</button><div/><button aria-label="Reset zoom">⌖</button></div><div className="north-marker">N <span>↑</span></div></div>
            <div className="overlay-control"><div className="overlay-copy"><span className="layers-icon">▱</span><span><b>Compare with original</b><small>Blend the source drawing with the digital plan</small></span></div><div className="slider-control"><span>OFF</span><input aria-label="Original drawing overlay opacity" type="range" min="0" max="100" value={Math.round(overlay*100)} onChange={e=>setOverlay(Number(e.target.value)/100)}/><span>ON</span></div><div className="overlay-value">{Math.round(overlay*100)}%</div></div>
            <div className="plan-footer"><span><span className="legend-dot"/> Detected room</span><span><span className="legend-line"/> Walls &amp; partitions</span><span className="footer-source">Source: {floorPlan.source.width} × {floorPlan.source.height} px</span></div>
          </section>
          <aside className="detail-card"><div className="detail-header"><div><div className="title-kicker">SPACE DETAILS</div><h2>{room.name}</h2></div><button className="square-button">···</button></div><div className="room-chip"><span className="chip-dot" style={{background:room.color}}/>{room.kind}<span className="room-id">{room.id}</span></div><div className="detail-metrics"><div><small>AREA</small><strong>{room.area.toFixed(1)} <em>m²</em></strong></div><div><small>DETECTION</small><strong>{Math.round(room.confidence*100)}<em>%</em></strong></div></div><div className="detail-divider"/><div className="detail-label">ABOUT THIS SPACE</div><p className="detail-description">A {room.kind.toLowerCase()} located on {floorPlan.floor.toLowerCase()}. Room details are based on the latest plan analysis.</p><div className="detail-label room-list-label">SPACES ON THIS FLOOR <span>{filteredRooms.length}</span></div><label className="room-search"><span>⌕</span><input placeholder="Find a room" value={query} onChange={e=>setQuery(e.target.value)}/><kbd>⌘ K</kbd></label><div className="room-list">{filteredRooms.map(r=><button key={r.id} className={`room-row ${selected===r.id?"chosen":""}`} onClick={()=>setSelected(r.id)}><span className="room-color" style={{background:r.color}}/><span className="room-row-name">{r.name}<small>{r.kind}</small></span><span className="room-row-area">{r.area.toFixed(1)} m²</span></button>)}</div></aside>
        </div>
        <section className="ask-bar"><div className="ask-spark">✳</div><div className="ask-intro"><b>Ask about this building</b><small>Get answers from your building data</small></div><form onSubmit={e=>{e.preventDefault();ask(query)}}><span>⌕</span><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Try “How large is the meeting room?”"/><button type="submit">Ask <span>↑</span></button></form><button className="suggestion" onClick={()=>ask("kitchen")}>Where’s the kitchen? <span>↗</span></button></section>
        {answer && <div className="answer-card"><div className="ask-spark">✳</div><div><b>Building assistant</b><p>{answer}</p></div><button onClick={()=>setAnswer("")}>×</button></div>}
        <footer className="page-footer"><span>Planform <span>·</span> Building intelligence</span><span>Coordinates shown in source image space</span></footer>
      </div>
    </section>
  </main>;
}
