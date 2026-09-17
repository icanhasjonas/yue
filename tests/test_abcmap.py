from pathlib import Path

import numpy as np
import pytest

from yuecli import abcmap
from yuecli.engine import anchor_mask

FIX = Path(__file__).parent / "fixtures"
SMOKE = (FIX / "smoke.abc").read_text()


def test_smoke_grid_matches_the_measured_render():
    # First local render: 100 BPM 4/4, 37 bars; the model emitted 2220 codec frames.
    g = abcmap.grid(SMOKE)
    assert (g.bpm, g.bars, g.frames) == (100, 37, 2220)
    assert g.bar_start_frame(1) == 0
    assert g.bar_start_frame(2) == 60  # 4 quarters at 100 BPM = 2.4 s = 60 frames
    assert g.bar_start_frame(38) == 2220  # one past the last bar = end


def test_other_tempo():
    g = abcmap.grid((FIX / "upstream-example.abc").read_text())
    assert g.bpm == 88
    per_bar = round(4 * 60 / 88 * 25)
    assert g.bar_start_frame(2) == per_bar


@pytest.mark.parametrize("spec,expected", [("9-16", (9, 16)), ("9-end", (9, None)), ("9-", (9, None)), ("3", (3, 3))])
def test_parse_bars(spec, expected):
    assert abcmap.parse_bars(spec) == expected


@pytest.mark.parametrize("spec", ["0-2", "5-3", "a-b", "", "1-2-3"])
def test_parse_bars_rejects(spec):
    with pytest.raises(ValueError):
        abcmap.parse_bars(spec)


def test_bar_frames_inclusive_range():
    assert abcmap.bar_frames(SMOKE, "13-28") == (720, 1680)
    assert abcmap.bar_frames(SMOKE, "29-end") == (1680, None)
    with pytest.raises(ValueError, match="outside the score"):
        abcmap.bar_frames(SMOKE, "30-40")


def test_anchor_mask_middle_edit_keeps_head_and_tail_with_margin():
    m = anchor_mask(100, 40, 60, margin=5)
    assert m[:35].all() and not m[35:65].any() and m[65:].all()


def test_anchor_mask_tail_edit_and_extension_keep_only_the_head():
    m = anchor_mask(120, 100, None, margin=10)
    assert m[:90].all() and not m[90:].any()


def test_anchor_mask_margin_never_goes_negative():
    m = anchor_mask(50, 3, None, margin=12)
    assert not m.any()
    assert isinstance(m, np.ndarray) and m.dtype == bool
