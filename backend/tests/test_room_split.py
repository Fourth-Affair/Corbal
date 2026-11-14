"""Splitting merged spaces (app.cubicasa.room_split) and splitting merged window boxes (app.cubicasa.windows)."""

import numpy as np

from app.cubicasa import room_split, windows


def _region():
    """One space: a bedroom (left) open to a living room (right), only a wall stub from the top between them."""
    lab = np.zeros((100, 200), np.int32)
    lab[10:90, 10:190] = 1
    lab[10:35, 98:102] = 0  # the stub
    return lab


def test_bedroom_open_to_living_room_is_cut_at_the_stub():
    lab, cuts = room_split.split_by_labels(_region(), [(50, 50, "bedroom"), (150, 50, "living")], t=4, min_area=500)
    assert len(np.unique(lab[lab > 0])) == 2 and len(cuts) == 1
    (x0, _), (x1, _) = cuts[0]
    assert 95 <= x0 <= 105 and x0 == x1  # vertical cut through the stub's line
    assert lab[50, 50] != lab[50, 150]


def test_open_plan_and_tiny_parts_stay_whole():
    lab, cuts = room_split.split_by_labels(_region(), [(50, 50, "kitchen"), (150, 50, "living")], t=4, min_area=500)
    assert len(np.unique(lab[lab > 0])) == 1 and not cuts
    lab, cuts = room_split.split_by_labels(_region(), [(50, 50, "bedroom"), (150, 50, "living")], t=4, min_area=8000)
    assert not cuts  # a side would be smaller than a room


def test_two_bedrooms_in_one_space_are_separated():
    lab, cuts = room_split.split_by_labels(_region(), [(50, 50, "bedroom"), (150, 50, "bedroom")], t=4, min_area=500)
    assert len(cuts) == 1


def _window(n_windows: int, double: bool = True) -> np.ndarray:
    """A window run drawn as 4 long lines with jambs; adjacent windows meet with two jamb lines."""
    ink = np.zeros((30, 200), bool)
    for y in (5, 9, 15, 19):
        ink[y, 10:190] = True
    ink[5:20, 10] = ink[5:20, 189] = True
    for k in range(1, n_windows):
        x = 10 + k * 180 // n_windows
        ink[5:20, x - 2] = True
        if double:
            ink[5:20, x + 2] = True
    return ink


def test_window_run_is_split_at_double_jambs_only():
    box = (10, 5, 190, 20)
    assert len(windows.split_box(_window(2), box)) == 2
    assert len(windows.split_box(_window(3), box)) == 3
    assert len(windows.split_box(_window(2, double=False), box)) == 1  # a sash division inside one window
    assert len(windows.split_box(_window(1), box)) == 1
