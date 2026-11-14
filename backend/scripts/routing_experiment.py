"""Can Gemini replace the regex router? Give it one tool, count(template), and compare its choice to the router.

Set 1: every unique cubicasa phrasing in questions.jsonl (router = ground truth, 100% on the eval).
Set 2: hand-written paraphrases (casual EN, Finnish slang, Swedish, compositional traps, out-of-scope) with
expected labels; "none" = the tool must not be called (unsupported question).

    uv run python scripts/routing_experiment.py [--model gemini-3.8-flash] [--repeats 3]
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from google import genai
from google.genai import types

from app.answering.tools import TEMPLATE_DEFS
from app.config import DATASET_DIR
from app.router import TEMPLATES, route

DEFS = TEMPLATE_DEFS  # what each template counts (shared with the agent tools)

SYSTEM = """You answer questions about a single floor plan image for site engineers.
You have one tool, count(template), which returns a whole-plan count computed by the drawing-analysis pipeline.
Call it only if the question asks for exactly one of its counts for the whole plan. If the question asks for something
else (areas, sizes, counts restricted to one room, combinations, yes/no about location, other objects), do NOT call
the tool; reply with text saying the question is not supported."""

TOOL = types.Tool(function_declarations=[types.FunctionDeclaration(
    name="count",
    description="Whole-plan count of one thing. Templates: " + "; ".join(f"{k} = {v}" for k, v in DEFS.items()),
    parameters_json_schema={
        "type": "object",
        "properties": {"template": {"type": "string", "enum": TEMPLATES}},
        "required": ["template"],
    },
)])

# (question, accepted labels). "none" = must not call the tool.
PARAPHRASES: list[tuple[str, set[str]]] = [
    # casual English
    ("how many windows does this place have?", {"windows"}),
    ("Number of doors in the apartment?", {"doors"}),
    ("Is this a two-storey house? How many levels are drawn?", {"floors"}),
    ("How many toilets are there? I mean the actual toilet bowls.", {"wc_seats"}),
    ("Count the wash basins and kitchen sinks for me.", {"sinks"}),
    ("How many places can you take a shower or a bath?", {"showers_baths"}),
    ("Does the flat have saunas? How many?", {"saunas"}),
    ("How many kitchenettes or kitchens are there?", {"kitchens"}),
    ("How many rooms are bedrooms?", {"bedrooms"}),
    ("Number of washrooms, excluding the separate toilets?", {"bathrooms"}),
    ("How many standalone WCs are there?", {"wc_rooms"}),
    ("How many balconies or terraces with glazing?", {"balconies"}),
    ("How many bedrooms does this unit have", {"bedrooms"}),
    ("total count of baths + showers", {"showers_baths"}),
    # Finnish, incl. slang / other words than the dataset
    ("Montako ikkunaa asunnossa on?", {"windows"}),
    ("Montako ovea tässä asunnossa on?", {"doors"}),
    ("Montako tasoa piirustuksessa on?", {"floors"}),
    ("Montako vessanpönttöä on piirretty?", {"wc_seats"}),
    ("Montako lavuaaria ja tiskiallasta on yhteensä?", {"sinks"}),
    ("Montako suihkua tai ammetta asunnossa on?", {"showers_baths"}),
    ("Onko asunnossa saunaa? Montako?", {"saunas"}),
    ("Montako keittokomeroa tai keittiötä on?", {"kitchens"}),
    ("Montako makkaria kämpässä on?", {"bedrooms"}),
    ("Montako kylppäriä on, erillisiä vessoja ei lasketa?", {"bathrooms"}),
    ("Montako erillistä vessaa asunnossa on?", {"wc_rooms"}),
    ("Montako parveketta tai lasitettua parveketta on?", {"balconies"}),
    # Swedish
    ("Hur många fönster finns på planritningen?", {"windows"}),
    ("Hur många dörrar finns det?", {"doors"}),
    ("Hur många våningar visar ritningen?", {"floors"}),
    ("Hur många toalettstolar är ritade?", {"wc_seats"}),
    ("Hur många handfat och diskhoar finns det totalt?", {"sinks"}),
    ("Hur många duschar och badkar finns det?", {"showers_baths"}),
    ("Hur många bastur finns det?", {"saunas"}),
    ("Hur många kök finns det?", {"kitchens"}),
    ("Hur många sovrum har lägenheten?", {"bedrooms"}),
    ("Hur många badrum finns det, separata WC ej medräknade?", {"bathrooms"}),
    ("Hur många separata WC finns det?", {"wc_rooms"}),
    ("Hur många balkonger finns det?", {"balconies"}),
    # traps: keywords of one template, but asking another / a composition / something unsupported
    ("What is the area of the kitchen?", {"none"}),
    ("How wide is the front door?", {"none"}),
    ("Are there windows in the bedroom?", {"none"}),
    ("How many sinks are there in the bathroom?", {"none"}),
    ("How many bathrooms have a shower?", {"none"}),
    ("Is the sauna next to the bathroom?", {"none"}),
    ("How many doors open onto the balcony?", {"none"}),
    ("Mikä on keittiön pinta-ala?", {"none"}),
    ("Hur stor är balkongen i kvadratmeter?", {"none"}),
    ("How many sinks are in the kitchen?", {"none"}),
    ("Kuinka monta ikkunaa makuuhuoneessa on?", {"none"}),
    # out of scope
    ("What is the total floor area?", {"none"}),
    ("What scale is this drawing?", {"none"}),
    ("How many radiators are shown?", {"none"}),
    ("How many rooms are there in total?", {"none"}),
    ("Kuinka monta neliötä asunnossa on?", {"none"}),
    ("Who drew this plan?", {"none"}),
    ("How many parking spaces are there?", {"none"}),
    ("Hur många rum finns det?", {"none"}),
]


def gemini_route(client, model: str, question: str) -> str:
    try:
        r = client.models.generate_content(
            model=model, contents=question,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM, tools=[TOOL],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)),
        )
    except Exception as error:
        return f"ERROR:{type(error).__name__}"
    calls = r.function_calls or []
    if not calls:
        return "none"
    if len(calls) > 1:
        return "multi:" + ",".join(str(c.args.get("template")) for c in calls)
    return str(calls[0].args.get("template"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.environ.get("GEMINI_MODEL", "gemini-3.8-flash"))
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    dataset = sorted({json.loads(line)["question"] for line in open(DATASET_DIR / "questions.jsonl")
                      if '"data/cubicasa5k/' in line})
    sets = {"dataset": [(q, {route(q)}) for q in dataset], "paraphrases": PARAPHRASES}

    client = genai.Client()
    jobs = [(name, q, ok) for name, items in sets.items() for q, ok in items for _ in range(args.repeats)]
    with ThreadPoolExecutor(args.workers) as pool:
        preds = list(pool.map(lambda j: gemini_route(client, args.model, j[1]), jobs))

    results = defaultdict(list)  # (set, question) -> [gemini preds]
    for (name, q, _), p in zip(jobs, preds):
        results[(name, q)].append(p)

    print(f"model={args.model} repeats={args.repeats}\n")
    for name, items in sets.items():
        g_ok = r_ok = stable = 0
        wrong = []
        for q, ok in items:
            ps = results[(name, q)]
            g_hits = sum(p in ok for p in ps)
            g_ok += g_hits
            stable += len(set(ps)) == 1
            r = route(q) or "none"
            r_ok += r in ok
            if g_hits < len(ps) or r not in ok:
                wrong.append((q, ok, r, Counter(ps)))
        n = len(items)
        print(f"== {name}: {n} questions")
        print(f"   gemini {g_ok}/{n * args.repeats} = {g_ok / (n * args.repeats):.1%}   "
              f"router {r_ok}/{n} = {r_ok / n:.1%}   gemini stable across repeats: {stable}/{n}")
        for q, ok, r, ps in wrong:
            print(f"   - {q!r}\n       expected {'/'.join(sorted(ok))} | router {r}{'' if r in ok else ' ✗'}"
                  f" | gemini {dict(ps)}{'' if all(p in ok for p in ps) else ' ✗'}")
        print()


if __name__ == "__main__":
    main()
