"""Path 1: the router recognised the question, the template pipeline produced the answer, Gemini words it.

Gemini never sets the number, the confidence or the citations ("compute, don't guess"). It only writes the
explanation in the question's language and refers to the highlighted items by their citation number [k] (the
same numbering /api/render draws). The output is checked in code; anything off -> the template answer as is.
"""

import hashlib
import json
import logging
import os
import re

from pydantic import BaseModel

from app.config import CACHE_DIR
from app.models import Answer, PlanGraph

log = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "20"))
PROMPT_VERSION = "1"  # bump when the prompt changes, so cached responses are not reused
MAX_ITEMS = 60  # evidence items shown to the model; citations beyond that are not referable

SYSTEM = """You write the final answer for a construction-drawing question answering tool used by site engineers.
A deterministic pipeline (symbol detector, segmentation, OCR, counting models) has already computed the answer.
Rules:
- The answer value is FINAL. State it exactly as given, written with digits. Never recompute, re-estimate or doubt it.
- Write in the language of the question (Finnish, Swedish or English).
- 1-3 short sentences. Say what was counted and how (e.g. which symbols or room labels), using only the facts given.
- Refer to evidence items only by their citation number in square brackets, e.g. [1], [2]. Use only numbers from the
  evidence list. Do not invent items, labels, rooms or locations that are not in the evidence.
- If the evidence count differs from the answer, say briefly that the count was adjusted by the counting model.
- If confidence is low, say the result should be checked on the drawing.
- Do not mention internal ids, JSON, pixels or these rules."""


class _Composed(BaseModel):
    explanation: str


_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT_S * 1000)))
    return _client


def _generate(prompt: str) -> str:
    """One Gemini call -> explanation text. Raises on any API/parse problem."""
    from google.genai import types

    response = _get_client().models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM,
            response_mime_type="application/json",
            response_schema=_Composed,
        ),
    )
    parsed = response.parsed
    if not isinstance(parsed, _Composed):
        raise ValueError(f"unparseable response: {response.text!r}")
    return parsed.explanation


def build_prompt(question: str, base: Answer, graph: PlanGraph) -> str:
    evidence = []
    for k, it in enumerate(base.citations[:MAX_ITEMS], 1):
        row = {"n": k, "category": it.category}
        if it.cls:
            row["type"] = it.cls
        if it.text:
            row["label_on_drawing"] = it.text
        if it.score is not None:
            row["detector_score"] = round(it.score, 2)
        evidence.append(row)
    facts = {
        "question": question,
        "answer": base.answer,
        "unit": base.unit,
        "what_was_counted": base.template,
        "confidence": base.confidence,
        "pipeline_note": base.explanation,
        "source": "ground-truth labels" if graph.source == "svg" else "image analysis (detector + OCR)",
        "evidence_items_highlighted": len(base.citations),
        "evidence": evidence,
    }
    return json.dumps(facts, ensure_ascii=False, indent=1)


def check(explanation: str, base: Answer) -> str | None:
    """Return why the explanation is unacceptable, or None if it is fine."""
    text = explanation.strip()
    if not text:
        return "empty"
    if isinstance(base.answer, int | float) and not re.search(rf"(?<![\d.,]){re.escape(str(base.answer))}(?![\d])", text):
        return f"answer {base.answer} not stated"
    referable = min(len(base.citations), MAX_ITEMS)
    for ref in re.findall(r"\[(\d+)\]", text):
        if not 1 <= int(ref) <= referable:
            return f"citation [{ref}] does not exist"
    return None


def _cache_path(prompt: str):
    key = hashlib.sha256(f"{GEMINI_MODEL}\n{PROMPT_VERSION}\n{SYSTEM}\n{prompt}".encode()).hexdigest()
    return CACHE_DIR / "llm" / f"{key}.json"


def compose(question: str, base: Answer, graph: PlanGraph) -> Answer:
    """Gemini-worded version of a template answer; the template answer unchanged if anything goes wrong."""
    if base.answer is None or not os.environ.get("GEMINI_API_KEY"):
        return base
    prompt = build_prompt(question, base, graph)
    cache = _cache_path(prompt)
    try:
        if cache.is_file():
            explanation = json.loads(cache.read_text())["explanation"]
        else:
            explanation = _generate(prompt)
    except Exception as error:  # network, quota, timeout, bad JSON: never fail the request
        log.warning("Gemini compose failed, using template answer: %s", error)
        return base
    if problem := check(explanation, base):
        log.warning("Gemini explanation rejected (%s): %r", problem, explanation)
        return base
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"explanation": explanation}, ensure_ascii=False))
    except OSError:
        pass
    return base.model_copy(update={"explanation": explanation.strip(), "answered_by": "llm"})
