"""Hand-written dictionary of Finnish (and some Swedish) floor-plan room labels -> room type.

Used on raw OCR tokens. Matching is done on a normalised token (upper case, no spaces/punctuation, common
OCR confusions folded). Keys are normalised the same way.
"""

import re
import unicodedata

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


# Purposes of the rooms the counting types lump into "other" (plan geometry / overlay only; counting is unchanged).
# Finnish abbreviations, Swedish and English words.
PURPOSE_LABELS: dict[str, list[str]] = {
    "living": ["OH", "OLOH", "OLOHUONE", "TUPA", "ARKIH", "OLESKELU", "VARDAGSRUM", "VARDRUM", "VR", "LIVING",
               "LIVINGROOM", "LOUNGE", "LIVINGAREA"],
    "dining": ["RUOK", "RT", "RUOKAILU", "RUOKAILUTILA", "MATPLATS", "MATRUM", "DINING", "DININGROOM"],
    "entry": ["ET", "ETEINEN", "EHH", "AULA", "TK", "TUULIK", "TUULIKAAPPI", "HALL", "HALLWAY", "ENTRY", "ENTRANCE",
              "FOYER", "VINDFÅNG", "CORRIDOR", "KÄYTÄVÄ"],
    "closet": ["VH", "VAATEH", "VAATEHUONE", "KLÄDKAMMARE", "KLK", "CLOSET", "WARDROBE", "WALKINCLOSET"],
    "storage": ["VAR", "VARASTO", "ULKOVAR", "SIIVK", "SIIVOUSK", "FÖRRÅD", "FRD", "STORAGE", "STORE"],
    "utility": ["KHH", "KODINHOITOH", "KODINHOITOHUONE", "PYYKKIH", "KUIVAUSH", "TVÄTTSTUGA", "TVÄTT", "LAUNDRY",
                "UTILITY"],
    "study": ["TYÖH", "TYÖHUONE", "KIRJASTO", "KIRJASTOH", "ARBETSRUM", "STUDY", "OFFICE"],
    "technical": ["TEKN", "TEKNH", "LVV", "LTO", "IV", "IVKONEH", "PANNUH", "TEKNIK", "TECHNICAL", "PLANTROOM"],
    "terrace": ["TERASSI", "TERRASS", "TERRACE", "PATIO", "ULKOTILA", "PIHA", "VERANTA"],
    "garage": ["AUTOK", "AUTOTALLI", "KATOS", "AUTOKATOS", "GARAGE", "CARPORT", "GARAGE"],
    "stairs": ["PORRAS", "PORRASH", "PORRASKÄYTÄVÄ", "TRAPPA", "STAIRS", "STAIRCASE", "STAIR"],
    "basement": ["KELLARI", "KÄLLARE", "BASEMENT", "CELLAR"],
    "bedroom": ["BEDROOM", "BED", "SOVRUM", "H", "HUONE", "ALK", "ALKOVI"],
    "kitchen": ["KITCHEN", "KITCHENETTE", "KÖK", "PENTRY"],
    "bathroom": ["BATHROOM", "BATH", "SHOWER", "SHOWERROOM", "BADRUM", "DUSCH"],
    "wc": ["TOILET", "TOILETS", "WC", "RESTROOM", "LAVATORY"],
    "sauna": ["SAUNA", "BASTU"],
    "balcony": ["BALCONY", "BALKONG"],
}

# German, French and Italian room names (printed in full on vector PDFs, e.g. the Swiss dwelling plans). A generic
# "room" (Zimmer / Pièce / Stanza) is a bedroom, like the Finnish H / HUONE above.
for _typ, _words in {
    "living": ["WOHNEN", "WOHNZIMMER", "WOHNENESSEN", "SALON", "SÉJOUR", "SEJOUR", "SÉJOURREPAS", "SEJOURREPAS",
               "SOGGIORNO", "SOGGIORNOPRANZO"],
    "dining": ["ESSEN", "ESSZIMMER", "REPAS", "SALLEÀMANGER", "SALLEAMANGER", "PRANZO", "SALADAPRANZO"],
    "entry": ["KORRIDOR", "FLUR", "DIELE", "EINGANG", "EINGANGSHALLE", "GANG", "COULOIR", "ENTRÉE", "HALLDENTRÉE",
              "DÉGAGEMENT", "DEGAGEMENT", "CORRIDOIO", "INGRESSO", "DISIMPEGNO"],
    "closet": ["ANKLEIDE", "ANKLEIDEZIMMER", "DRESSING", "GARDEROBE", "GUARDAROBA", "CABINAARMADIO"],
    "storage": ["REDUIT", "RÉDUIT", "ABSTELLRAUM", "ABSTELLKAMMER", "DÉBARRAS", "DEBARRAS", "RIPOSTIGLIO",
                "DEPOSITO"],
    "utility": ["WASCHKÜCHE", "WASCHRAUM", "BUANDERIE", "LAVANDERIA"],
    "study": ["BÜRO", "ARBEITSZIMMER", "BUREAU", "UFFICIO"],
    "technical": ["TECHNIK", "HAUSTECHNIK", "TECHNIQUE", "LOCALETECNICO"],
    "terrace": ["TERRASSE", "TERRAZZA"],
    "garage": ["AUTORIMESSA"],
    "stairs": ["TREPPE", "TREPPENHAUS", "ESCALIER", "CAGEDESCALIER", "SCALA", "VANOSCALA"],
    "basement": ["KELLER", "CAVE", "CANTINA"],
    "bedroom": ["ZIMMER", "SCHLAFZIMMER", "SCHLAFEN", "KINDERZIMMER", "PIÈCE", "PIECE", "CHAMBRE", "CHAMBREÀCOUCHER",
                "CHAMBREACOUCHER", "STANZA", "CAMERA", "CAMERADALETTO"],
    "kitchen": ["KÜCHE", "KUCHE", "CUISINE", "CUCINA"],
    "bathroom": ["BADEZIMMER", "DUSCHE", "BAIN", "SALLEDEBAIN", "SALLEDEBAINS", "SALLEDEAU", "DOUCHE", "BAGNO"],
    "wc": ["TOILETTE", "TOILETTEN", "GABINETTO"],
    "balcony": ["BALKON", "BALCON", "BALCONE", "LOGGIA"],
}.items():
    PURPOSE_LABELS[_typ] += _words


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


_PURPOSE = {normalise(k): typ for typ, keys in PURPOSE_LABELS.items() for k in keys}


def _fold(n: str) -> str:
    """Drop diacritics: OCR often reads TYÖH as TYOH, KEITTIÖ as KEITTIO."""
    return "".join(c for c in unicodedata.normalize("NFKD", n) if not unicodedata.combining(c))


# Purposes by diacritic-folded key (counting types first, finer purposes do not override them)
_FOLDED: dict[str, str] = {}
for _k, _t in [*_PURPOSE.items(), *((k, t) for k, t in _LOOKUP.items() if t != "other")]:
    _FOLDED[_fold(_k)] = _t
_MERGED_DIMENSION = re.compile(r"^\s*([^\W\d_]{2,})[\s.:]*\d{3,}\s*$")  # "VAR4956": label + dimension read as one


def _tolerant_purpose(n: str) -> str | None:
    """Purpose of a normalised token that has no exact match, allowing for common OCR damage. Plan geometry only:
    the counting features (room_type, token table) keep exact matching, their models were fitted on it."""
    f = _fold(n)
    if f in _FOLDED:
        return _FOLDED[f]
    # separator lost: "MHTYOH" = MH/TYÖH -> the first part (both parts must be labels of >= 2 letters)
    for i in range(2, len(f) - 1):
        if f[:i] in _FOLDED and f[i:] in _FOLDED:
            return _FOLDED[f[:i]]
    # a slash or bar read as a letter: "AULAI" = AULA/, "MHL" = MH|
    if len(f) >= 3 and f[-1] in "IL1" and f[:-1] in _FOLDED:
        return _FOLDED[f[:-1]]
    return None


def room_purpose(token: str) -> str | None:
    """What a room is for, from one label: the counting type when it names one, else a finer purpose (living,
    entry, closet, storage, ...). None when the token is not a room label (dimensions, furniture codes, noise).

    Tokens with 3+ digits are annotations, not labels: "H2388" is a ceiling height and "MH 12,5" an area (its
    number goes to the room's labeled area instead), but "MH2" stays bedroom 2. A label of 2+ letters fused with
    a dimension ("VAR4956") keeps its letters."""
    digits = sum(c.isdigit() for c in token)
    if digits >= 3 and not re.search(r"m2|m²", token, re.IGNORECASE):
        m = _MERGED_DIMENSION.match(token)
        return room_purpose(m.group(1)) if m else None
    t = room_type(token)
    if t and t != "other":
        return t
    n = normalise(token)
    if not n or len(n) > 24:
        return None
    if n in _PURPOSE:
        return _PURPOSE[n]
    parts = [p for p in re.split(r"[+&]|JA", n) if p]
    if len(parts) > 1:  # "OH+K", "ET/VH": the first recognised part
        for p in parts:
            if p in _PURPOSE or (_LOOKUP.get(p) not in (None, "other")):
                return _LOOKUP.get(p) if _LOOKUP.get(p) not in (None, "other") else _PURPOSE[p]
    return _tolerant_purpose(n)
