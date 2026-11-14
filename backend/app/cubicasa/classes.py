"""Detector class list shared by training export and inference. Index = YOLO class id."""

# (yolo name, PlanItem category, PlanItem cls)
YOLO_CLASSES: list[tuple[str, str, str]] = [
    ("floor", "floor", ""),
    ("door", "door", ""),
    ("window", "window", ""),
    ("toilet", "fixture", "toilet"),
    ("sink", "fixture", "sink"),
    ("double_sink", "fixture", "double_sink"),
    ("shower", "fixture", "shower"),
    ("bathtub", "fixture", "bathtub"),
    ("sauna_stove", "fixture", "sauna_stove"),
    ("room_bedroom", "room", "bedroom"),
    ("room_bathroom", "room", "bathroom"),
    ("room_wc", "room", "wc"),
    ("room_sauna", "room", "sauna"),
    ("room_kitchen", "room", "kitchen"),
    ("room_balcony", "room", "balcony"),
]

CLASS_INDEX = {(cat, cls): i for i, (_, cat, cls) in enumerate(YOLO_CLASSES)}
