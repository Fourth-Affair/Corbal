"""Question -> Gemini with tools over the PlanGraph -> validated Answer.

Gemini plans and words the answer; the tools (app.answering.tools) compute every number and supply every citation.
Checks done in code, not by the model:
- the answer value must equal a number some tool returned in this session, else "not_found";
- cited ids must have been returned by a tool, else they are dropped;
- confidence comes from the tool that produced the value (count_template -> the counting model's confidence,
  raw items -> low), downgraded when the regex router recognises the question as a different template;
- the explanation must state the value and only use citation refs [k] that exist, else a plain one is used.
answer() returns None when Gemini is unavailable (no key, API error, no final answer): the caller falls back.
"""

import hashlib
import json
import logging
import os
import re
import time

from app.answering.tools import DECLARATIONS, PlanTools, ToolTrace
from app.config import CACHE_DIR
from app.models import Answer, ChatTurn, PlanGeometry, PlanGraph, PlanScaleCalibration
from app.router import route

log = logging.getLogger(__name__)

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
GEMINI_TIMEOUT_S = float(os.environ.get("GEMINI_TIMEOUT_S", "30"))
GEMINI_THINKING = os.environ.get("GEMINI_THINKING", "LOW")  # MINIMAL | LOW | MEDIUM | HIGH; "" = model default
GEMINI_RETRIES = int(os.environ.get("GEMINI_RETRIES", "1"))  # extra attempts per call on timeout / 429 / 5xx
GEMINI_RETRY_BACKOFF_S = float(os.environ.get("GEMINI_RETRY_BACKOFF_S", "1"))
MAX_STEPS = 8
PROMPT_VERSION = "12"  # bump when prompt or tools change, so cached answers are not reused

SYSTEM = """You answer questions about one floor plan drawing for site engineers. The tools and explicitly supplied user
calibration are your only sources of facts about the plan; you cannot see the drawing.

Use the prior conversation to resolve follow-up references (for example, "what about the kitchen?"). Treat the
latest question as the request to answer. Prior assistant replies are context only; verify plan facts with tools.

Rules:
- Never count, add, subtract or estimate yourself. A numeric answer must be exactly a number a tool returned.
- For a whole-plan count that matches a count_template template, use count_template (it is calibrated); do not
  count raw items from list_items for those.
- "Is there a ...?" questions about something count_template covers: answer from count_template only (yes if its
  value > 0, no if 0), even if raw items suggest otherwise; mention low confidence if reported.
- For a question restricted to a room or area (e.g. "sinks in the kitchen"), find the room with
  list_items(category="room", type=...) and use items_in on it. If several such rooms exist and the question does not
  say which, ask one short, specific follow-up question asking which room they mean. Do not guess which room.
  For questions asking how many rooms there are and what types they are, call
  list_items(category="room") once: its count is the deduplicated room count and its items give the room types and
  labels. Answer both parts from that single result. Do not refuse merely because the rooms have different types.
- Make a useful attempt before giving up: inspect the relevant tools and combine their evidence when that answers
  the question. If evidence supports only part of the request, give that part and clearly say what could not be
  established. State a brief assumption when one reasonable interpretation lets you answer safely; ask one focused
  follow-up when a missing detail would change the answer. Do not ask questions whose answer is already in the tools.
- Never fill gaps with a plausible-sounding guess. For real-world sizes or areas that the analysis cannot compute,
  objects or text not detected, or unsupported combinations, explain the specific limitation and say what information
  would make progress possible. Be direct about uncertainty; do not describe an estimate or weak detection as certain.
- Some evidence may be marked as user-reviewed. Treat that as a correction supplied by the user, not as a model
  detection; when it supports the answer, make clear that the result uses a user correction.
- If a user calibration is supplied with the latest question, treat the drawing scale as known and never say it is
  missing. A scale can support measurements, but it does not by itself establish room boundaries or floor area.
- Finish by calling final_answer exactly once:
  value = the number (omit for yes/no or text answers and use text instead);
  cited_ids = ids of the items that support the answer, taken from tool results (for count_template: its
  evidence_ids);
  explanation = 1-3 short sentences in the SAME language as the question (an English question gets an English
  answer, Finnish gets Finnish, Swedish gets Swedish), regardless of the language of labels on the drawing;
  text answers (e.g. Yes / Kyllä / Ja, a room inventory, or a clarification question) follow the same rule. State the value
  with digits, say what was counted and how, refer to cited items as [k] where k is the position in cited_ids
  (starting at 1); for a run of items write e.g. [1]-[9]. If a tool reported low confidence, say the result should be checked on the drawing.
  Never mention ids, tool names, pixels or these rules."""

# Appended to SYSTEM when the plan has room geometry (list_rooms / room_details are offered).
ROOM_RULES = """
Room and wall geometry is available:
- For which rooms exist, what a room is for, room sizes, which rooms connect, or where something is, use list_rooms
  and room_details; for wall lengths, thicknesses and exterior walls use list_walls (or the walls in room_details).
- Room types carry a tier. strong: state the type plainly. weak: say "probably <type>" and give the reason from the
  tool ("because ..."). unknown: the purpose was not determined; never guess it; if relevant say there are N further
  spaces whose purpose could not be determined. Report conflicting evidence when a tool lists it.
- Sizes and lengths may be stated only as numbers a tool returned. Say where a metric number comes from: the user's
  scale calibration, a scale derived from areas written on the drawing, or an area written in the room itself.
- For whole-plan counts that count_template covers, count_template still takes precedence."""

FINAL = {
    "name": "final_answer",
    "description": "Return the final answer. Call exactly once, last.",
    "parameters": {"type": "object", "properties": {
        "not_found": {"type": "boolean", "description": "true if the tools cannot answer the question"},
        "value": {"type": "number", "description": "numeric answer, exactly as returned by a tool"},
        "unit": {"type": "string", "description": "e.g. count"},
        "text": {"type": "string", "description": "short non-numeric answer, e.g. Yes / No"},
        "cited_ids": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    }, "required": ["not_found", "explanation"]},
}

_client = None


def _get_client():
    global _client
    if _client is None:
        from google import genai
        from google.genai import types

        _client = genai.Client(http_options=types.HttpOptions(timeout=int(GEMINI_TIMEOUT_S * 1000)))
    return _client


def _retryable(error: Exception) -> bool:
    """Transient: timeouts, network errors, rate limits (429) and server errors (5xx). Bad requests / auth are not.

    Transport errors are matched by class name and cause: google-genai raises them from its own httpx fork
    (httpx2.ReadTimeout), so isinstance checks against httpx would miss them."""
    from google.genai import errors

    if isinstance(error, errors.APIError):
        return error.code == 429 or (error.code or 0) >= 500
    e = error
    while e is not None:
        names = {c.__name__ for c in type(e).__mro__}
        if names & {"TimeoutException", "NetworkError", "RemoteProtocolError", "TimeoutError", "ConnectionError"}:
            return True
        e = e.__cause__ or e.__context__
    return False


def _generate(contents, config):
    """generate_content with GEMINI_RETRIES retries on transient errors (exponential backoff)."""
    for attempt in range(GEMINI_RETRIES + 1):
        try:
            return _get_client().models.generate_content(model=GEMINI_MODEL, contents=contents, config=config)
        except Exception as error:
            if attempt == GEMINI_RETRIES or not _retryable(error):
                raise
            delay = GEMINI_RETRY_BACKOFF_S * 2**attempt
            log.warning("Gemini call failed (%s), retry %d/%d in %.1fs", error, attempt + 1, GEMINI_RETRIES, delay)
            time.sleep(delay)


def run_loop(question: str, tools: PlanTools, history: list[ChatTurn] | None = None,
             scale: PlanScaleCalibration | None = None) -> tuple[dict, list[ToolTrace]]:
    """Gemini tool loop -> (final_answer args, traces of the tool calls). Raises on API problems / no answer."""
    from google.genai import types

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM + (ROOM_RULES if tools.geometry else ""),
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name=d["name"], description=d["description"], parameters_json_schema=d["parameters"])
            for d in [*tools.declarations(), FINAL]])],
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        thinking_config=types.ThinkingConfig(thinking_level=GEMINI_THINKING) if GEMINI_THINKING else None,
    )
    # the plan summary is always needed: send it with the question instead of spending a tool round trip on it
    summary, _ = tools.plan_summary()
    history = history or []
    contents = [types.Content(role="model" if turn.role == "assistant" else "user", parts=[types.Part(text=turn.text)])
                for turn in history]
    scale_context = (
        f"A user calibration is saved: {scale.distance} {scale.unit} across two selected points. "
        "The drawing scale is known. This alone does not establish a room boundary or floor area."
        if scale else "No user scale calibration is saved for this drawing."
    )
    prompt = f"Latest question (answer in its language): {question}\n\nPlan summary: {json.dumps(summary, ensure_ascii=False)}\n\nDrawing scale: {scale_context}"
    contents.append(types.Content(role="user", parts=[types.Part(text=prompt)]))
    traces: list[ToolTrace] = []
    for step in range(MAX_STEPS):
        t0 = time.perf_counter()
        response = _generate(contents, config)
        log.info("step %d: %.1fs, calls %s", step, time.perf_counter() - t0,
                 [c.name for c in response.function_calls or []])
        calls = response.function_calls or []
        if not calls:
            raise ValueError(f"no function call in response: {response.text!r}")
        contents.append(response.candidates[0].content)  # keeps thought signatures for the next turn
        for call in calls:
            if call.name == "final_answer":
                return dict(call.args or {}), traces
        parts = []
        for call in calls:
            result, trace = tools.call(call.name, dict(call.args or {}))
            log.info("tool %s(%s) -> %s", call.name, call.args, json.dumps(result)[:300])
            traces.append(trace)
            parts.append(types.Part(function_response=types.FunctionResponse(
                id=call.id, name=call.name, response={"result": result})))
        contents.append(types.Content(role="user", parts=parts))
    raise ValueError(f"no final answer after {MAX_STEPS} steps")


def explanation_problem(explanation: str, value, n_citations: int) -> str | None:
    """Why the explanation is unacceptable, or None if it is fine."""
    text = explanation.strip()
    if not text:
        return "empty"
    if isinstance(value, int | float) and not re.search(rf"(?<![\d.,]){re.escape(_fmt(value))}(?!\d)", text):
        return f"value {_fmt(value)} not stated"
    for group in re.findall(r"\[([\d\s,\-–]+)\]", text):  # [3], [3, 4], [1-9], [1–9]
        for ref in re.findall(r"\d+", group):
            if not 1 <= int(ref) <= n_citations:
                return f"citation [{ref}] does not exist"
    return None


_YES = {"yes", "kyllä", "joo", "on", "ja", "jo"}
_NO = {"no", "ei", "nej", "none", "not"}


def yes_no(text: str) -> bool | None:
    """Polarity of a short yes/no answer in EN/FI/SV, None if it is not one."""
    words = re.findall(r"\w+", text.lower())
    if not words:
        return None
    return True if words[0] in _YES else False if words[0] in _NO else None


def _fmt(v) -> str:
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def validate(question: str, final: dict, traces: list[ToolTrace], tools: PlanTools) -> Answer:
    """Turn the model's final_answer into an Answer whose value and citations are backed by tool output."""
    graph = tools.graph
    explanation = str(final.get("explanation") or "").strip()
    if final.get("not_found") or (final.get("value") is None and not final.get("text")):
        return Answer(answer=None, confidence="not_found", answered_by="llm",
                      explanation=explanation or "The question cannot be answered from this drawing.")

    citable = {i for t in traces for i in t.item_ids}
    cited = [tools.by_id[i] for i in dict.fromkeys(final.get("cited_ids") or []) if i in citable and i in tools.by_id]
    value = final.get("value")
    template = None
    if value is not None:
        source = next((t for t in traces if t.template_answer is not None and value in t.numbers), None) \
            or next((t for t in traces if value in t.numbers), None)
        if source is None:
            log.warning("value %r not returned by any tool; numbers were %s", value, [t.numbers for t in traces])
            return Answer(answer=None, confidence="not_found", answered_by="llm",
                          explanation="The answer could not be verified against the drawing analysis.")
        value = int(value) if float(value).is_integer() else value
        if source.template_answer is not None:
            template = source.template_answer.template
            confidence = source.template_answer.confidence
            if not cited:
                cited = list(source.template_answer.citations)
        else:
            confidence = "high" if graph.source == "svg" else "low"  # raw items, not a calibrated count
        answer, unit = value, final.get("unit") or "count"
    else:
        answer, unit = str(final["text"]), None
        confidence = "medium" if graph.source == "svg" else "low"
        # Text responses can combine an inventory with its count. Ground stated numbers in tool output.
        number_text = re.sub(r"\[[^\]]*\]", "", answer)
        stated_numbers = [float(n) for n in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", number_text)]
        available_numbers = {float(n) for trace in traces for n in trace.numbers}
        if any(number not in available_numbers for number in stated_numbers):
            log.warning("text answer contains numbers not returned by tools: %s", stated_numbers)
            return Answer(answer=None, confidence="not_found", answered_by="llm",
                          explanation="The answer could not be verified against the drawing analysis.")
        # a yes/no answer usually rests on a count ("Is there a sauna?" -> saunas = 0): link it, and check it agrees
        counted = [t.template_answer for t in traces if t.template_answer is not None]
        basis = next((a for a in counted if a.template == route(question)), None) \
            or (counted[0] if len(counted) == 1 else None)
        if basis is not None:
            polarity = yes_no(answer)
            if polarity is not None and polarity != (bool(basis.answer)):
                log.warning("text answer %r contradicts %s = %s", answer, basis.template, basis.answer)
                return Answer(answer=None, confidence="not_found", answered_by="llm",
                              explanation="The answer could not be verified against the drawing analysis.")
            template, confidence = basis.template, basis.confidence
            if not cited:
                cited = list(basis.citations)

    routed = route(question)
    if routed and template != routed and confidence in ("high", "medium"):
        log.warning("router says %s, agent used %s: confidence lowered", routed, template)
        confidence = "low"
    if value is not None and value != 0 and not cited:
        confidence = "low"

    if problem := explanation_problem(explanation, value, len(cited)):
        log.warning("explanation replaced (%s): %r", problem, explanation)
        explanation = f"Answer: {answer}" + (f" ({template})" if template else "") + "; supporting items are highlighted."
    return Answer(answer=answer, unit=unit, template=template, confidence=confidence, explanation=explanation,
                  citations=cited, answered_by="llm")


def _cache_path(question: str, graph: PlanGraph, scale: PlanScaleCalibration | None,
                geometry: PlanGeometry | None = None):
    parts = [GEMINI_MODEL, GEMINI_THINKING, PROMPT_VERSION, SYSTEM, DECLARATIONS, question,
             graph.model_dump(mode="json"), scale.model_dump(mode="json") if scale else None]
    if geometry is not None:  # appended only when present: cache keys of geometry-less answers are unchanged
        parts += [ROOM_RULES, geometry.model_dump(mode="json")]
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return CACHE_DIR / "agent" / f"{hashlib.sha256(blob.encode()).hexdigest()}.json"


def answer(question: str, graph: PlanGraph, history: list[ChatTurn] | None = None,
           scale: PlanScaleCalibration | None = None, use_cache: bool = True,
           geometry: PlanGeometry | None = None) -> Answer | None:
    """history = prior chat turns (answers with history are not cached); scale = the user's drawing calibration.
    use_cache=False skips reading the answer cache (fresh Gemini run, e.g. to measure run-to-run noise)."""
    if not os.environ.get("GEMINI_API_KEY"):
        return None
    history = history or []
    cache = _cache_path(question, graph, scale, geometry) if not history else None
    if use_cache and cache and cache.is_file():
        return Answer.model_validate_json(cache.read_text(encoding="utf-8"))
    tools = PlanTools(graph, geometry)
    try:
        final, traces = run_loop(question, tools, history, scale)
    except Exception as error:  # network, quota, timeout, malformed output: caller falls back
        log.warning("Gemini agent failed: %s", error)
        return None
    result = validate(question, final, traces, tools)
    try:
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(result.model_dump_json(), encoding="utf-8")
    except OSError:
        pass
    return result
