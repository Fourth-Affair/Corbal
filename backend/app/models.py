"""API and plan-graph schemas. Keep Answer/PlanItem stable: the frontend draws overlays from them."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["floor", "room", "door", "window", "fixture", "wall", "opening", "unit"]  # unit = apartment

# Canonical classes. Rooms: bedroom, bathroom, wc, sauna, kitchen, living, balcony, entry, storage, garage, other.
# Fixtures: toilet, sink, double_sink, shower, bathtub, sauna_stove, other.
# Walls (layers API): wall = solid wall region, opening = door/window gap in a wall.


class PlanItem(BaseModel):
    """One addressable thing on a plan. Coordinates: pixels of the PNG for image plans, PDF points of page 1 for
    PDF plans (e.g. data/swiss_dwellings/*.pdf); origin top-left in both."""

    id: str
    category: Category
    cls: str = ""  # room type / fixture kind; empty for door/window/floor
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    polygon: list[tuple[float, float]] | None = None
    score: float | None = None  # model confidence; None for ground-truth labels
    text: str | None = None  # label text on the drawing (OCR at runtime, NameLabel in SVG ground truth)
    raw: str | None = None  # source-format class, e.g. SVG "Space Bath Shower" (debugging / rule fitting)
    user_verified: bool = False  # true when a person added or corrected this evidence


class PlanOutline(BaseModel):
    id: str
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=500)
    score: float | None = None
    scope: Literal["floor", "room"] = "floor"
    room_type: str | None = None
    label: str = "Floor"


class PlanGraph(BaseModel):
    image: str  # dataset-relative path, e.g. data/cubicasa5k/colorful-10052.png (or uploads/<plan id>.pdf)
    width: int
    height: int
    source: Literal["svg", "model", "pdf"]  # svg = ground-truth labels, model = detector + OCR, pdf = vector PDF parser
    items: list[PlanItem] = Field(default_factory=list)
    # model source only: per-template counts from the fusion models (items are the supporting evidence)
    counts: dict[str, int] = Field(default_factory=dict)
    floor_outlines: list[PlanOutline] = Field(default_factory=list)
    room_outlines: list[PlanOutline] = Field(default_factory=list)
    user_correction_count: int = 0  # number of append-only human correction events applied to this copy


class AskRequest(BaseModel):
    path: str
    question: str


class PlanScaleCalibration(BaseModel):
    points: tuple[tuple[float, float], tuple[float, float]]
    distance: float = Field(gt=0)
    unit: Literal["m", "cm", "ft", "in"]


class PlanOutlinesRequest(BaseModel):
    outlines: list[PlanOutline] = Field(min_length=1, max_length=20)


class PlanCorrectionRequest(BaseModel):
    action: Literal["add", "update", "delete"]
    item_id: str | None = None
    item: PlanItem | None = None


class PlanFeedbackRequest(BaseModel):
    question: str = Field(max_length=4000)
    answer: str = Field(max_length=4000)
    feedback: Literal["helpful", "incorrect", "unclear"]
    reason: str | None = Field(default=None, max_length=500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=100)


class PlanQuestionRequest(BaseModel):
    question: str
    history: list["ChatTurn"] = Field(default_factory=list, max_length=40)
    scale: PlanScaleCalibration | None = None
    floor_outlines: list[PlanOutline] = Field(default_factory=list, max_length=20)


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str = Field(max_length=4000)


class Answer(BaseModel):
    answer: int | float | str | None
    unit: str | None = None
    template: str | None = None
    confidence: Literal["high", "medium", "low", "not_found"]
    explanation: str
    citations: list[PlanItem] = Field(default_factory=list)
    answered_by: Literal["template", "llm"] = "template"  # template = router + counter, llm = Gemini agent (values still tool-computed)


class TextToken(BaseModel):
    """One OCR reading. room_type is set when the text is a recognised room label (e.g. "MH" -> bedroom)."""

    text: str
    bbox: tuple[float, float, float, float]
    score: float
    room_type: str | None = None


class Layer(BaseModel):
    """One perception layer of a plan (walls, symbols or text), in pixels of the challenge PNG."""

    image: str
    width: int
    height: int
    layer: Literal["walls", "symbols", "text"]
    model: str  # which model produced it, e.g. "wallseg", "cubicasa-net", "symbol-det.onnx", "rapidocr"
    items: list[PlanItem] = Field(default_factory=list)
    tokens: list[TextToken] = Field(default_factory=list)
    stats: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------- plan geometry (rooms + walls), stored per plan


class RoomLabel(BaseModel):
    """Text read inside a room (OCR on images, the drawing's own text on PDFs)."""

    text: str
    bbox: tuple[float, float, float, float]
    score: float | None = None
    room_type: str | None = None  # the purpose the text names (e.g. "MH" -> bedroom, "OH" -> living), if any


class RoomEvidence(BaseModel):
    """One piece of evidence for a room's type (see app.cubicasa.room_types for the weights)."""

    source: Literal["ocr", "fixture", "segmentation", "geometry", "drawing"]
    type: str  # the room type it supports
    weight: float  # score points it adds (hand-set: ocr label +6, decisive fixture +4..5, segmentation <= +1.5)
    detail: str  # human-readable, e.g. 'label "WC"', "toilet inside", "net: 62% bath pixels"
    item_id: str | None = None  # citable plan item (fixture) behind it, if any


class RoomShape(BaseModel):
    """One enclosed space. Coordinates in pixels of the plan image (PDF: points of page 1), origin top-left."""

    id: str  # citable, e.g. "space-3"
    # bedroom, bathroom, wc, sauna, kitchen, balcony, living, dining, entry, closet, storage, utility, study,
    # technical, terrace, garage, stairs, basement, or unknown
    type: str
    # strong = a readable purpose label or decisive fixtures, clear lead, no conflict (state it as fact);
    # weak = only weak evidence (2019 net pixels, ambiguous fixtures) or a conflict ("probably X");
    # unknown = no evidence (never state a purpose)
    tier: Literal["strong", "weak", "unknown"]
    purposes: list[str] = Field(default_factory=list)  # every purpose labelled inside (open plan: living + kitchen)
    scores: dict[str, float] = Field(default_factory=dict)  # top-3 type scores
    evidence: list[RoomEvidence] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)  # strong evidence pointing to another type
    name: str | None = None  # the room's own label as written, e.g. "MH" or "KEITTIÖ"
    labels: list[RoomLabel] = Field(default_factory=list)  # all text inside the room
    polygon: list[tuple[float, float]]
    bbox: tuple[float, float, float, float]
    area_px: float
    labeled_area_m2: float | None = None  # an area written in the room ("12,5 m²"), if any
    area_m2: float | None = None  # area_px through the plan scale (filled when served, not stored)
    # rough range from the symbol-size scale band, only when no better scale exists (filled when served)
    area_m2_range: tuple[float, float] | None = None
    size_check: str | None = None  # set when the size is implausible for the type (filled when served)
    neighbours: list[str] = Field(default_factory=list)  # rooms across a wall
    connections: list[str] = Field(default_factory=list)  # opening ids leading out of this room
    walls: list[str] = Field(default_factory=list)  # wall segment ids bounding the room


class WallShape(BaseModel):
    id: str  # wall-N or opening-N
    kind: Literal["wall", "opening"]
    polygon: list[tuple[float, float]]
    holes: list[list[tuple[float, float]]] = Field(default_factory=list)
    bbox: tuple[float, float, float, float]
    area_px: float
    rooms: list[str] = Field(default_factory=list)  # opening: the rooms it joins ("outside" = exterior)
    symbol: str | None = None  # opening: "door" / "window" when a detected symbol overlaps it


class WallSegment(BaseModel):
    """A straight piece of wall centreline (vectorised from the wall map)."""

    id: str  # seg-N
    p0: tuple[float, float]
    p1: tuple[float, float]
    length_px: float
    thickness_px: float
    angle_deg: float  # 0 = horizontal (left to right), 90 = vertical, in image coordinates (y down)
    orientation: Literal["horizontal", "vertical", "diagonal"]
    rooms_left: list[str] = Field(default_factory=list)  # rooms on the left of p0 -> p1 ("outside" = exterior)
    rooms_right: list[str] = Field(default_factory=list)
    exterior: bool = False  # one side is outside the building
    openings: list[str] = Field(default_factory=list)  # opening ids crossing this segment
    length_m: float | None = None  # with a plan scale (filled when served)
    thickness_m: float | None = None


class ScaleDimension(BaseModel):
    """One dimension figure used for the scale: the printed length and the segment it labels (image pixels)."""

    text: str
    value_mm: float
    bbox: tuple[float, float, float, float]  # the figure's text
    p0: tuple[float, float]  # segment end ticks
    p1: tuple[float, float]
    length_px: float


class PlanScale(BaseModel):
    """Metres per image pixel (PDF point) and where that number comes from."""

    m_per_px: float
    # calibration = the user's two points; written_areas = areas written in rooms; dimensions = printed lengths of
    # dimension lines (app.cubicasa.dimensions); symbols = rough band from the known sizes of doors / toilets /
    # sinks (app.cubicasa.size_priors), never a measurement
    source: Literal["calibration", "written_areas", "dimensions", "symbols"]
    rooms: list[str] = Field(default_factory=list)  # written_areas: the rooms whose areas agreed
    low: float | None = None  # symbols: band of plausible m/px
    high: float | None = None
    items: list[str] = Field(default_factory=list)  # symbols: the plan items used
    dimensions: list[ScaleDimension] = Field(default_factory=list)  # dimensions: the figures that agreed
    warning: str | None = None  # e.g. the scale disagrees with the symbol sizes (filled when served)


class PlanGeometry(BaseModel):
    """Rooms and walls of one plan, stored in Supabase (plan_geometry) for overlays and the answering agent."""

    version: int = 5  # bump when the content changes: older stored rows are recomputed
    # virtual walls closing door-less passages between rooms (p0, p1 in pixels), app.cubicasa.passages
    passages: list[tuple[tuple[float, float], tuple[float, float]]] = Field(default_factory=list)
    image: str
    width: int
    height: int
    model: str  # what produced the walls, e.g. "wallseg", "cubicasa-net", "pdf"
    wall_thickness_px: float | None = None
    rooms: list[RoomShape] = Field(default_factory=list)
    walls: list[WallShape] = Field(default_factory=list)
    openings: list[WallShape] = Field(default_factory=list)
    wall_segments: list[WallSegment] = Field(default_factory=list)
    scale: PlanScale | None = None  # derived from written areas when stored; the user's calibration wins when served
    symbol_scale: PlanScale | None = None  # rough band from symbol sizes (doors, toilets, ...): bounds, not a measurement
