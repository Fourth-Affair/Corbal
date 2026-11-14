"""Map a natural-language question (EN/FI) to a cubicasa count template.

Order matters: several questions mention other templates' keywords
(e.g. the sinks question says "kitchen and bathroom combined").
"""

import re

_RULES: list[tuple[str, str]] = [
    ("sinks", r"\bsinks?\b|wash ?basin|allas"),
    ("showers_baths", r"shower|bathtub|suihku|kylpyamme"),
    ("wc_seats", r"toilet(?!\s+rooms?)|wc seat|wc-istuin"),
    ("bathrooms", r"bathroom|washroom|kylpy- tai pesuhuone|kylpyhuone|pesuhuone"),
    ("wc_rooms", r"wc rooms?|toilet rooms?|wc-tila"),
    ("doors", r"\bdoors?\b|\bove[ai]?\b|\boveä\b"),
    ("windows", r"window|ikkun"),
    ("floors", r"\bfloors\b|storey|kerrosta|kerrokse"),
    ("saunas", r"sauna"),
    ("kitchens", r"kitchen|keittiö"),
    ("bedrooms", r"bedroom|makuuhuone"),
    ("balconies", r"balcon|parvek"),
]
_COMPILED = [(t, re.compile(p, re.IGNORECASE)) for t, p in _RULES]

TEMPLATES = [t for t, _ in _RULES]


def route(question: str) -> str | None:
    """Return the template id, or None if the question is not a supported cubicasa count."""
    for template, pattern in _COMPILED:
        if pattern.search(question):
            return template
    return None
