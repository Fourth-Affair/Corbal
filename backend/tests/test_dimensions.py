"""Plan scale from dimension lines: printed millimetres over the pixel span between the segment's ticks."""

import numpy as np
import pytest
from PIL import Image

from app.cubicasa import dimensions, ocr
from app.models import TextToken


@pytest.fixture(autouse=True)
def no_reread(monkeypatch):
    monkeypatch.setattr(ocr, "recognise", lambda *a, **k: ("", 0.0))


def _drawing(ticks=(100, 300, 500), y=50, skew=False):
    """A horizontal dimension line with crossing extension lines, scanned look optional (1 px drift, gaps)."""
    a = np.full((200, 600), 255, np.uint8)
    for x in range(ticks[0], ticks[-1] + 1):
        if skew and x % 37 == 0:
            continue  # gap
        a[y + (1 if skew and x > 250 else 0), x] = 0
    for x in ticks:
        a[y - 12:y + 30, x] = 0
    return Image.fromarray(a)


def tok(text, box):
    return TextToken(text=text, bbox=box, score=0.99)


@pytest.mark.parametrize("skew", [False, True])
def test_scale_from_two_agreeing_figures(skew):
    toks = [tok("2000", (170, 22, 230, 49)), tok("2 000", (370, 22, 430, 49))]
    s = dimensions.dimension_scale(_drawing(skew=skew), toks)
    assert s is not None and s.source == "dimensions"
    assert s.m_per_px == pytest.approx(0.01, rel=0.02)
    assert sorted(d.p0[0] for d in s.dimensions) == [100, 300]


def test_one_figure_or_disagreement_is_not_a_scale():
    assert dimensions.dimension_scale(_drawing(), [tok("2000", (170, 22, 230, 49))]) is None
    toks = [tok("2000", (170, 22, 230, 49)), tok("3500", (370, 22, 430, 49))]
    assert dimensions.dimension_scale(_drawing(), toks) is None


def test_text_without_a_line_or_with_a_wall_gives_nothing():
    a = np.full((200, 600), 255, np.uint8)
    a[50:62, 100:501] = 0  # a 12 px wall under the text, not a thin dimension line
    toks = [tok("2000", (170, 22, 230, 49)), tok("2000", (370, 22, 430, 49))]
    assert dimensions.dimension_scale(Image.fromarray(a), toks) is None


def test_split_vertical_figure_is_joined():
    pieces = [tok("100", (763, 278, 790, 332)), tok("3", (765, 330, 788, 347)), tok("TUPA", (429, 295, 506, 323))]
    groups = dimensions._groups(pieces)
    assert sorted(len(g) for g in groups) == [2]  # TUPA is not a number
    (g,) = groups
    readings = dimensions._readings(Image.new("RGB", (838, 793), "white"), g, dimensions._union(g), True)
    assert readings == [("3100", 3100.0)]  # vertical text reads bottom-to-top
