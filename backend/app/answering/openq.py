"""Open questions: questions no count template covers, with ground truth computed from the labels.

The dataset asks twelve whole-plan counts. A site engineer also asks "how many sinks are in the kitchen?", "is the
sauna next to the bathroom?" or "are there more windows than doors?". This module writes such questions for a
labelled plan (PlanGraph with source="svg") in English, Finnish and Swedish, together with the true answer, so the
benchmark (app.bench, kind "open") can measure breadth instead of rewarding a refusal.

A question is only generated where the labels give one defensible answer:
- "the kitchen" needs exactly one kitchen on the plan;
- a fixture is in a room when its centre lies in the room polygon (and in no other);
- a window belongs to a room when it runs along the room's outline; a room with a window that only clips its corner,
  or with a glazed opening shared with another indoor room (sauna glass wall), yields no window question;
- "next to" needs a clearly shared wall or a clear gap, "larger" a 1.5x difference in area.
Everything in between is skipped rather than guessed.

Families (the `family` field of a benchmark item):
  in_room_count   How many sinks are there in the kitchen?            number
  in_room_yesno   Is there a toilet in the bathroom?                  yes / no
  rooms_with      How many bedrooms have a window?                    number
  plan_yesno      Is there a sauna in this home?                      yes / no
  compare_yesno   Are there more windows than doors?                  yes / no
  arithmetic      How many toilets and sinks are there in total?      number (sum or difference of two counts)
  subtype_count   How many bathtubs are there? Do not count showers.  number
  adjacent_yesno  Is the kitchen next to the bathroom?                yes / no
  size_yesno      Is the kitchen larger than the bathroom?            yes / no
"""

import random
import statistics
from dataclasses import dataclass

from app.cubicasa.answer import count
from app.models import PlanGraph, PlanItem

LANGS = ("en", "fi", "sv")
MIN_PLANS = 5  # a (question, answer) combination must occur on this many plans to be sampled
FAMILIES = ("in_room_count", "in_room_yesno", "rooms_with", "plan_yesno", "compare_yesno", "arithmetic",
            "subtype_count", "adjacent_yesno", "size_yesno")

# --- vocabulary: every inflected form a wording needs, per language ------------------------------------------

ROOMS = {  # en: the X | fi: nominative, inessive ("in"), genitive | sv: definite, "in the X"
    "kitchen": {"en": "the kitchen", "fi": ("keittiö", "keittiössä", "keittiön"), "sv": ("köket", "i köket")},
    "bathroom": {"en": "the bathroom", "fi": ("kylpyhuone", "kylpyhuoneessa", "kylpyhuoneen"),
                 "sv": ("badrummet", "i badrummet")},
    "bedroom": {"en": "the bedroom", "fi": ("makuuhuone", "makuuhuoneessa", "makuuhuoneen"),
                "sv": ("sovrummet", "i sovrummet")},
    "living": {"en": "the living room", "fi": ("olohuone", "olohuoneessa", "olohuoneen"),
               "sv": ("vardagsrummet", "i vardagsrummet")},
    "wc": {"en": "the separate WC", "fi": ("erillinen WC", "erillisessä WC:ssä", "erillisen WC:n"),
           "sv": ("det separata WC:t", "i det separata WC:t")},
    "balcony": {"en": "the balcony", "fi": ("parveke", "parvekkeella", "parvekkeen"),
                "sv": ("balkongen", "på balkongen")},
}
ROOMS_PLURAL = {  # en plural | fi "in how many X" | sv plural
    "bathroom": {"en": "bathrooms", "fi": "kylpyhuoneessa", "sv": "badrum"},
    "bedroom": {"en": "bedrooms", "fi": "makuuhuoneessa", "sv": "sovrum"},
}
ITEMS = {  # en: plural, with article | fi: partitive singular, nominative | sv: plural, with article
    "sink": {"en": ("sinks or wash basins", "a sink or wash basin"), "fi": ("pesuallasta", "pesuallas"),
             "sv": ("handfat eller diskhoar", "ett handfat eller en diskho")},
    "toilet": {"en": ("toilets", "a toilet"), "fi": ("WC-istuinta", "WC-istuin"),
               "sv": ("toalettstolar", "en toalettstol")},
    "window": {"en": ("windows", "a window"), "fi": ("ikkunaa", "ikkuna"), "sv": ("fönster", "ett fönster")},
    "bathtub": {"en": ("bathtubs", "a bathtub"), "fi": ("kylpyammetta", "kylpyamme"),
                "sv": ("badkar", "ett badkar")},
}
COUNTED = {  # a count template as a noun. en plural | fi: partitive singular, partitive plural | sv plural
    "windows": {"en": "windows", "fi": ("ikkunaa", "ikkunoita"), "sv": "fönster"},
    "doors": {"en": "doors", "fi": ("ovea", "ovia"), "sv": "dörrar"},
    "bedrooms": {"en": "bedrooms", "fi": ("makuuhuonetta", "makuuhuoneita"), "sv": "sovrum"},
    "bathrooms": {"en": "bathrooms", "fi": ("kylpyhuonetta", "kylpyhuoneita"), "sv": "badrum"},
    "sinks": {"en": "sinks", "fi": ("pesuallasta", "pesualtaita"), "sv": "handfat och diskhoar"},
    "wc_seats": {"en": "toilets", "fi": ("WC-istuinta", "WC-istuimia"), "sv": "toalettstolar"},
}

WORDINGS = {
    "in_room_count": {
        "en": ["How many {items} are there in {room}?", "Count the {items} in {room}."],
        "fi": ["Montako {item_part} {room_in} on?", "Kuinka monta {item_part} on {room_in}?"],
        "sv": ["Hur många {items} finns det {room_in}?", "Räkna antalet {items} {room_in}."],
    },
    "in_room_yesno": {
        "en": ["Is there {a_item} in {room}?", "Does {room} have {a_item}?"],
        "fi": ["Onko {room_in} {item_part}?"],
        "sv": ["Finns det {a_item} {room_in}?"],
    },
    "rooms_with": {
        "en": ["How many {rooms} have {a_item}?", "In how many {rooms} is there {a_item}?"],
        "fi": ["Kuinka monessa {rooms} on {item_nom}?", "Monessako {rooms} on {item_nom}?"],
        "sv": ["Hur många {rooms} har {a_item}?", "I hur många {rooms} finns det {a_item}?"],
    },
    "plan_yesno": {
        "sauna": {"en": ["Is there a sauna in this home?", "Does the plan show a sauna?"],
                  "fi": ["Onko asunnossa sauna?", "Löytyykö pohjapiirustuksesta sauna?"],
                  "sv": ["Finns det en bastu i bostaden?", "Visar planritningen en bastu?"]},
        "balcony": {"en": ["Does this home have a balcony?", "Is a balcony drawn on the plan?"],
                    "fi": ["Onko asunnossa parveke?", "Onko pohjapiirustukseen piirretty parveke?"],
                    "sv": ["Har bostaden en balkong?", "Finns det en balkong på ritningen?"]},
        "bathtub": {"en": ["Is there a bathtub anywhere on the plan?", "Does this home have a bathtub?"],
                    "fi": ["Onko asunnossa kylpyamme?", "Löytyykö asunnosta kylpyamme?"],
                    "sv": ["Finns det ett badkar i bostaden?", "Har bostaden ett badkar?"]},
        "wc": {"en": ["Is there a separate WC, apart from the bathroom?", "Does this home have a separate toilet room?"],
               "fi": ["Onko asunnossa erillinen WC?", "Löytyykö asunnosta erillinen vessa?"],
               "sv": ["Finns det en separat toalett i bostaden?", "Har bostaden ett separat WC?"]},
    },
    "compare_yesno": {
        "en": ["Are there more {a} than {b}?", "Does the plan have more {a} than {b}?"],
        "fi": ["Onko {a_pl} enemmän kuin {b_pl}?"],
        "sv": ["Finns det fler {a} än {b}?"],
    },
    "arithmetic_sum": {
        "en": ["How many {a} and {b} are there in total?", "What is the combined number of {a} and {b}?"],
        "fi": ["Kuinka monta {a_sg} ja {b_sg} on yhteensä?"],
        "sv": ["Hur många {a} och {b} finns det sammanlagt?"],
    },
    "arithmetic_diff": {
        "en": ["How many more {a} than {b} are there?"],
        "fi": ["Kuinka paljon enemmän {a_pl} on kuin {b_pl}?"],
        "sv": ["Hur många fler {a} än {b} finns det?"],
    },
    "subtype_count": {
        "bathtub": {"en": ["How many bathtubs are there? Do not count showers.", "Count only the bathtubs, not the showers."],
                    "fi": ["Montako kylpyammetta asunnossa on? Suihkuja ei lasketa.", "Kuinka monta ammetta on, kun suihkuja ei lasketa mukaan?"],
                    "sv": ["Hur många badkar finns det? Räkna inte duschar."]},
        "sauna_stove": {"en": ["How many sauna stoves are drawn on the plan?"],
                        "fi": ["Montako kiuasta pohjapiirustuksessa on?"],
                        "sv": ["Hur många bastuaggregat finns på ritningen?"]},
    },
    "adjacent_yesno": {
        "en": ["Is {a} next to {b}?", "Do {a} and {b} share a wall?"],
        "fi": ["Onko {a_nom} {b_gen} vieressä?"],
        "sv": ["Ligger {a} bredvid {b}?", "Ligger {a} intill {b}?"],
    },
    "size_yesno": {
        "en": ["Is {a} larger than {b}?", "Is {a} bigger than {b}?"],
        "fi": ["Onko {a_nom} suurempi kuin {b_nom}?"],
        "sv": ["Är {a} större än {b}?"],
    },
}

IN_ROOM = [("kitchen", "sink"), ("kitchen", "window"), ("bathroom", "sink"), ("bathroom", "toilet"),
           ("bathroom", "bathtub"), ("bathroom", "window"), ("bedroom", "window"), ("living", "window"),
           ("wc", "sink"), ("wc", "window")]
ROOMS_WITH = [("bathroom", "toilet"), ("bathroom", "window"), ("bathroom", "bathtub"), ("bedroom", "window")]
COMPARE = [("windows", "doors"), ("doors", "windows"), ("bedrooms", "bathrooms"), ("sinks", "wc_seats"),
           ("wc_seats", "sinks")]
SUMS = [("bedrooms", "bathrooms"), ("wc_seats", "sinks"), ("windows", "doors")]
ADJACENT = [("kitchen", "bathroom"), ("kitchen", "bedroom"), ("bedroom", "bathroom"), ("living", "kitchen"),
            ("living", "bathroom"), ("bathroom", "wc"), ("balcony", "living"), ("bedroom", "balcony")]
LARGER = [(a, b) for pair in [("kitchen", "bathroom"), ("kitchen", "bedroom"), ("bedroom", "bathroom"),
                              ("living", "kitchen"), ("living", "bathroom"), ("bathroom", "wc")]
          for a, b in (pair, pair[::-1])]


@dataclass(frozen=True)
class Candidate:
    family: str
    key: str  # what is asked, language-independent, e.g. "kitchen/sink"
    answer: int | str  # a count, or "yes" / "no"
    slots: tuple  # arguments for question()


# --- ground truth from the labels ----------------------------------------------------------------------------


def room_kind(it: PlanItem) -> str | None:
    """Room type for these questions; the living room is read from the SVG class (the counting rules ignore it)."""
    if it.cls in ("bedroom", "bathroom", "wc", "kitchen", "balcony"):
        return it.cls
    tokens = (it.raw or "").split()
    return "living" if len(tokens) > 1 and tokens[1] == "LivingRoom" else None


def _centre(it: PlanItem):
    from shapely.geometry import Point

    return Point((it.bbox[0] + it.bbox[2]) / 2, (it.bbox[1] + it.bbox[3]) / 2)


def _fixture_is(it: PlanItem, item: str) -> bool:
    return it.category == "fixture" and (it.cls in ("sink", "double_sink") if item == "sink" else it.cls == item)


def _along(window: PlanItem, poly, door_size: float) -> float:
    """Share of the window's length that runs along the room outline (0 = elsewhere, 1 = wholly on this room)."""
    from shapely.geometry import box

    x0, y0, x1, y1 = window.bbox
    pad = 0.1 * door_size
    inter = box(x0 - pad, y0 - pad, x1 + pad, y1 + pad).intersection(poly)
    if inter.is_empty:
        return 0.0
    ix0, iy0, ix1, iy1 = inter.bounds
    return (ix1 - ix0) / (x1 - x0 + 2 * pad) if x1 - x0 >= y1 - y0 else (iy1 - iy0) / (y1 - y0 + 2 * pad)


class Plan:
    """The labelled plan as these questions see it."""

    def __init__(self, graph: PlanGraph):
        from shapely.geometry import Polygon

        self.graph = graph
        self.polys: dict[str, object] = {}
        self.outdoor: set[str] = set()
        self.rooms: dict[str, list[PlanItem]] = {}
        for it in graph.items:
            if it.category != "room" or not it.polygon or len(it.polygon) < 3:
                continue
            poly = Polygon(it.polygon)
            poly = poly if poly.is_valid else poly.buffer(0)
            if poly.is_empty or poly.area <= 0:
                continue
            self.polys[it.id] = poly
            if (it.raw or "").split()[1:2] == ["Outdoor"]:
                self.outdoor.add(it.id)
            if kind := room_kind(it):
                self.rooms.setdefault(kind, []).append(it)
        doors = [max(it.bbox[2] - it.bbox[0], it.bbox[3] - it.bbox[1]) for it in graph.items if it.category == "door"]
        self.door_size = statistics.median(doors) if doors else None  # ~0.9 m: the plan's only length unit

    def only(self, kind: str) -> PlanItem | None:
        rooms = self.rooms.get(kind, [])
        return rooms[0] if len(rooms) == 1 else None

    def n_in(self, room: PlanItem, item: str) -> int | None:
        """Items of one kind in a room, or None when the labels do not settle it."""
        poly = self.polys[room.id]
        if item == "window":
            if not self.door_size:
                return None
            n = 0
            for it in self.graph.items:
                if it.category != "window" or (share := _along(it, poly, self.door_size)) <= 0.2:
                    continue
                if share < 0.6:
                    return None  # clips the room's corner
                if any(i != room.id and i not in self.outdoor and _along(it, p, self.door_size) > 0.2
                       for i, p in self.polys.items()):
                    return None  # glazing between two indoor rooms: whose window is it?
                n += 1
            return n
        n = 0
        for it in self.graph.items:
            if _fixture_is(it, item) and poly.contains(c := _centre(it)):
                if sum(p.contains(c) for p in self.polys.values()) != 1:
                    return None  # overlapping room polygons: which room it is in is unclear
                n += 1
        return n

    def adjacent(self, a: PlanItem, b: PlanItem) -> bool | None:
        if not self.door_size:
            return None
        pa, pb = self.polys[a.id], self.polys[b.id]
        if pa.distance(pb) > 1.0 * self.door_size:
            return False
        shared = pa.buffer(0.35 * self.door_size).intersection(pb)  # reach across a wall
        if not shared.is_empty:
            x0, y0, x1, y1 = shared.bounds
            if max(x1 - x0, y1 - y0) >= 0.6 * self.door_size:
                return True
        return None  # corner contact or a thick gap: not clear either way

    def larger(self, a: PlanItem, b: PlanItem) -> bool | None:
        ra, rb = self.polys[a.id].area, self.polys[b.id].area
        return True if ra >= 1.5 * rb else False if rb >= 1.5 * ra else None


def _yn(flag: bool) -> str:
    return "yes" if flag else "no"


def candidates(graph: PlanGraph) -> list[Candidate]:
    """Every question the labels of this plan answer unambiguously."""
    plan, out = Plan(graph), []
    n = {t: count(graph, t).answer for t in ("windows", "doors", "bedrooms", "bathrooms", "sinks", "wc_seats",
                                              "saunas", "balconies", "wc_rooms")}
    fixtures = {k: sum(_fixture_is(it, k) for it in graph.items) for k in ("bathtub", "sauna_stove")}

    for room_kind_, item in IN_ROOM:
        room = plan.only(room_kind_)
        if room is None or (k := plan.n_in(room, item)) is None:
            continue
        out.append(Candidate("in_room_count", f"{room_kind_}/{item}", k, (room_kind_, item)))
        out.append(Candidate("in_room_yesno", f"{room_kind_}/{item}", _yn(k > 0), (room_kind_, item)))
    for room_kind_, item in ROOMS_WITH:
        rooms = plan.rooms.get(room_kind_, [])
        per_room = [plan.n_in(r, item) for r in rooms]
        if rooms and None not in per_room:
            out.append(Candidate("rooms_with", f"{room_kind_}/{item}", sum(k > 0 for k in per_room), (room_kind_, item)))

    for thing, present in (("sauna", n["saunas"] > 0), ("balcony", n["balconies"] > 0),
                           ("bathtub", fixtures["bathtub"] > 0), ("wc", n["wc_rooms"] > 0)):
        out.append(Candidate("plan_yesno", thing, _yn(present), (thing,)))
    for a, b in COMPARE:
        out.append(Candidate("compare_yesno", f"{a}>{b}", _yn(n[a] > n[b]), (a, b)))
    for a, b in SUMS:
        out.append(Candidate("arithmetic", f"{a}+{b}", n[a] + n[b], ("sum", a, b)))
    for a, b in (("windows", "doors"), ("doors", "windows")):
        if n[a] > n[b]:
            out.append(Candidate("arithmetic", f"{a}-{b}", n[a] - n[b], ("diff", a, b)))
    for thing in ("bathtub", "sauna_stove"):
        out.append(Candidate("subtype_count", thing, fixtures[thing], (thing,)))

    for family, pairs, test in (("adjacent_yesno", ADJACENT, plan.adjacent), ("size_yesno", LARGER, plan.larger)):
        for a, b in pairs:
            ra, rb = plan.only(a), plan.only(b)
            if ra is not None and rb is not None and (flag := test(ra, rb)) is not None:
                out.append(Candidate(family, f"{a}{'~' if family == 'adjacent_yesno' else '>'}{b}", _yn(flag), (a, b)))
    return out


# --- wording -------------------------------------------------------------------------------------------------


def question(c: Candidate, lang: str, rng: random.Random) -> str:
    """One wording of the candidate in `lang` (en / fi / sv)."""
    pick = lambda options: options[rng.randrange(len(options))]  # noqa: E731
    if c.family in ("plan_yesno", "subtype_count"):
        return pick(WORDINGS[c.family][c.slots[0]][lang])
    if c.family in ("in_room_count", "in_room_yesno"):
        room, item = ROOMS[c.slots[0]][lang], ITEMS[c.slots[1]][lang]
        fill = {"en": lambda: {"room": room, "items": item[0], "a_item": item[1]},
                "fi": lambda: {"room_in": room[1], "item_part": item[0]},
                "sv": lambda: {"room_in": room[1], "items": item[0], "a_item": item[1]}}[lang]()
        return pick(WORDINGS[c.family][lang]).format(**fill)
    if c.family == "rooms_with":
        item = ITEMS[c.slots[1]][lang]
        return pick(WORDINGS[c.family][lang]).format(rooms=ROOMS_PLURAL[c.slots[0]][lang], a_item=item[1],
                                                     item_nom=item[1])
    if c.family in ("compare_yesno", "arithmetic"):
        kind = c.family if c.family == "compare_yesno" else f"arithmetic_{c.slots[0]}"
        a, b = (COUNTED[t][lang] for t in c.slots[-2:])
        fill = {"a": a, "b": b} if lang != "fi" else {"a_sg": a[0], "b_sg": b[0], "a_pl": a[1], "b_pl": b[1]}
        return pick(WORDINGS[kind][lang]).format(**fill)
    a, b = ROOMS[c.slots[0]][lang], ROOMS[c.slots[1]][lang]  # adjacent_yesno, size_yesno
    fill = {"a_nom": a[0], "b_nom": b[0], "b_gen": b[2]} if lang == "fi" else \
        {"a": a if lang == "en" else a[0], "b": b if lang == "en" else b[0]}
    text = pick(WORDINGS[c.family][lang]).format(**fill)
    return text[0].upper() + text[1:]


# --- sampling for the benchmark ------------------------------------------------------------------------------


def sample(graphs: dict[str, PlanGraph], per_family: int, rng: random.Random) -> list[dict]:
    """per_family questions of every family over the given labelled plans, as benchmark items (without id).

    Balanced so that neither "always yes" nor "always 0" scores well: within a family the picks rotate over the
    distinct (what is asked, answer) combinations, and over plans. Combinations seen on fewer than MIN_PLANS plans
    are left out: they are mostly label oddities (a plan without a single door). Languages rotate en / fi / sv.
    """
    pool: dict[str, dict[tuple, list[tuple[str, Candidate]]]] = {f: {} for f in FAMILIES}
    for path in sorted(graphs):
        for c in candidates(graphs[path]):
            answer = min(c.answer, 3) if isinstance(c.answer, int) else c.answer  # 3 = "three or more"
            pool[c.family].setdefault((c.key, answer), []).append((path, c))
    items = []
    for family in FAMILIES:
        strata = [pool[family][k] for k in sorted(pool[family], key=str) if len(pool[family][k]) >= MIN_PLANS]
        for s in strata:
            rng.shuffle(s)
        rng.shuffle(strata)
        used: set[str] = set()
        picked = 0
        while picked < per_family and any(strata):
            for s in strata:
                if not s or picked >= per_family:
                    continue
                i = next((i for i, (p, _) in enumerate(s) if p not in used), 0)  # prefer a plan not used yet
                path, c = s.pop(i)
                used.add(path)
                lang = LANGS[picked % len(LANGS)]
                yesno = isinstance(c.answer, str)
                items.append({"kind": "open", "dataset": "open", "path": path, "question": question(c, lang, rng),
                              "answer": c.answer, "unit": None if yesno else "count",
                              "type": "yesno" if yesno else "count", "lang": lang, "template": None,
                              "family": family, "asks": c.key})
                picked += 1
    return items
