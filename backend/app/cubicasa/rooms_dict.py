"""Hand-written dictionary of Finnish (and some Swedish) floor-plan room labels -> room type.

Used on raw OCR tokens. Matching is done on a normalised token (upper case, no spaces/punctuation, common
OCR confusions folded). Keys are normalised the same way.
"""

import re

ROOM_LABELS: dict[str, list[str]] = {
    "bedroom": ["MH", "MH1", "MH2", "MH3", "MH4", "MAKUUH", "MAKUUHUONE", "SOV", "SOVRUM", "SOVR", "MAKUU"],
    "bathroom": ["KH", "KPH", "PH", "PESUH", "PESUHUONE", "SH", "PSH", "KYLPYH", "KYLPYHUONE", "KYLPY", "PESU",
                 "PUKUH", "WCKH", "KHWC", "PHWC", "WCPH", "BADRUM", "BAD", "DUSCH", "TVÄTT"],
    "wc": ["WC", "WC1", "WC2", "VC", "TOAL", "TOALETT"],
    "sauna": ["SAUNA", "S", "LH", "LÖYLYH", "LÖYLYHUONE", "BASTU"],
    "kitchen": ["K", "KK", "KT", "KEITTIÖ", "KEITTIO", "AVOK", "AVOKEITTIÖ", "KEITTOKOMERO", "KOK", "KÖK",
                "KEITTIÖKOMERO", "TUPAK", "TUPAKEITTIÖ", "APUK", "APUKEITTIÖ", "KEITTIÖTILA", "RUOKAILU+K"],
    "living": ["OH", "OLOH", "OLOHUONE", "LIVING", "LIVINGROOM", "VARDAGSRUM"],
    "balcony": ["PARVEKE", "PARV", "LASITETTUPARVEKE", "LASPARV", "LASITPARV", "RANSKALPARVEKE", "PARVEKKEET",
                "PARVEKE1", "PARVEKE2", "BALKONG", "BALK", "VIHERHUONE"],
    # common non-target labels (kept so they are recognised as "room label, not ours")
    "other": ["ET", "TK", "VH", "VAR", "VARASTO", "KHH", "AK", "TUPA", "RT", "RUOK", "H", "TEKN",
              "TERASSI", "ULKOVAR", "AUTOK", "KATOS", "PORRAS", "KELLARI", "AULA", "TYÖH", "KIRJASTO", "ALK",
              "LVV", "VAATEH", "KUIVAUSH", "PYYKKIH", "SIIVK", "ULKOTILA", "PIHA", "VAR.", "ARKIH", "EHH", "RK"],
}


def normalise(token: str) -> str:
    t = token.upper().strip()
    t = t.replace("0", "O").replace("|", "I").replace("Ä", "Ä")
    t = re.sub(r"[\s.,:;'\"`´()\[\]{}\-_/\\]+", "", t)
    t = re.sub(r"\d+(M2|M²|M)?$", "", t) if not t[:2].isdigit() else t  # strip trailing area/number
    return t


_LOOKUP = {normalise(k): typ for typ, keys in ROOM_LABELS.items() for k in keys}


def room_type(token: str) -> str | None:
    """Room type for an OCR token, or None if it is not a recognised room label."""
    n = normalise(token)
    if not n or len(n) > 24:
        return None
    if n in _LOOKUP:
        return _LOOKUP[n]
    # compound labels like "KH/WC", "OH+K", "TUPAK.+HUONE": classify by the strongest part
    parts = [p for p in re.split(r"[+&]|JA", n) if p]
    if len(parts) > 1:
        types = [_LOOKUP.get(p) for p in parts]
        for pref in ("kitchen", "bathroom", "wc", "sauna", "bedroom", "balcony"):
            if pref in types:
                return pref
    if n.startswith("PARVEK") or n.startswith("LASITETTUPARV"):
        return "balcony"
    if n.startswith("MAKUU"):
        return "bedroom"
    if n.startswith("KEITTI"):
        return "kitchen"
    return None


_APT_TYPE = re.compile(r"^(\d)\s*H\s*\+", re.IGNORECASE)


def apartment_type(token: str) -> dict | None:
    """Parse apartment-type headers like '3H+K+S', '2h+kt+s' -> {'rooms': 3, 'sauna': True, 'kitchen': True}."""
    t = token.upper().replace(" ", "")
    m = _APT_TYPE.match(t)
    if not m:
        return None
    parts = t.split("+")
    return {"rooms": int(m.group(1)), "sauna": "S" in parts[1:] or "SAUNA" in parts[1:],
            "kitchen": any(p in ("K", "KK", "KT", "TK", "AVOK") for p in parts[1:])}
