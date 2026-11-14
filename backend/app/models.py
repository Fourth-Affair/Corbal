"""API and plan-graph schemas. Keep Answer/PlanItem stable: the frontend draws overlays from them."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["floor", "room", "door", "window", "fixture", "wall", "opening"]

# Canonical classes. Rooms: bedroom, bathroom, wc, sauna, kitchen, balcony, other.
# Fixtures: toilet, sink, double_sink, shower, bathtub, sauna_stove, other.
# Walls (layers API): wall = solid wall region, opening = door/window gap in a wall.


class PlanItem(BaseModel):
    """One addressable thing on a plan. Coordinates are pixels of the challenge PNG, origin top-left."""

    id: str
    category: Category
    cls: str = ""  # room type / fixture kind; empty for door/window/floor
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    polygon: list[tuple[float, float]] | None = None
    score: float | None = None  # model confidence; None for ground-truth labels
    text: str | None = None  # label text on the drawing (OCR at runtime, NameLabel in SVG ground truth)
    raw: str | None = None  # source-format class, e.g. SVG "Space Bath Shower" (debugging / rule fitting)


class PlanGraph(BaseModel):
    image: str  # dataset-relative path, e.g. data/cubicasa5k/colorful-10052.png
    width: int
    height: int
    source: Literal["svg", "model"]  # svg = ground-truth labels, model = detector + OCR
    items: list[PlanItem] = Field(default_factory=list)
    # model source only: per-template counts from the fusion models (items are the supporting evidence)
    counts: dict[str, int] = Field(default_factory=dict)


class AskRequest(BaseModel):
    path: str
    question: str


class Answer(BaseModel):
    answer: int | float | str | None
    unit: str | None = None
    template: str | None = None
    confidence: Literal["high", "medium", "low", "not_found"]
    explanation: str
    citations: list[PlanItem] = Field(default_factory=list)
    answered_by: Literal["template", "llm"] = "template"  # who wrote the explanation; the value is always computed


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
