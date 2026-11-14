"""Which space an area question is about. A room we cannot identify must never turn into the whole-floor area."""

import pytest

from app.main import _WHOLE_FLOOR, area_target


@pytest.mark.parametrize("question, room", [
    ("what is the floor area of the study?", "study"),  # "floor" in the question, still a room
    ("how big is the kitchen", "kitchen"),
    ("what's the size of the living room", "living"),
    ("how large is mh2?", "bedroom"),
    ("kuinka suuri on keittiön pinta-ala?", "kitchen"),
    ("how big is the laundry", "utility"),
    ("what is the area of the toilet room", "wc"),
])
def test_named_rooms(question, room):
    assert area_target(question) == (room, None)


@pytest.mark.parametrize("question, unknown", [
    ("what is the floor area of the den?", "den"),
    ("how big is the pantry in the apartment", "pantry"),
    ("how big is this room", "room"),
    ("kuinka suuri on huone?", "huone"),
    ("hur stor är rummet?", "rummet"),
])
def test_unidentified_spaces_are_not_the_floor(question, unknown):
    room, phrase = area_target(question)
    assert room is None and phrase == unknown


@pytest.mark.parametrize("question", [
    "what is the total floor area?",
    "how big is the apartment",
    "what is the area of the whole plan",
    "kuinka suuri on asunnon pinta-ala?",
])
def test_whole_floor(question):
    assert area_target(question) == (None, None) and _WHOLE_FLOOR.search(question)
