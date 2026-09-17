"""Score time -> codec frame index.

YuE2's semantic tokens and acoustic latents both run at 25 frames per second
(48 kHz / 1920-sample VAE hop). A score with `Q:1/4=<bpm>` therefore puts bar
N at an exact frame:

    frame = quarters_before_bar * 60 / bpm * 25

Measured on the first local render (100 BPM, 4/4, 37 bars): 37 * 60 = 2220
frames, and the model emitted 2221 tokens (2220 + its end token). That is ONE
song. The model performs the score; it is not a sequencer, so a mapping this
clean is an observation to keep checking, not a guarantee -- `yue status`
prints both numbers so a drift is visible.

Bars are 1-based and inclusive, the way a musician counts: `--bars 9-16` is
eight bars starting at the ninth. `end` means the end of the song.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

from .abc_tools import parse_abc

FRAMES_PER_SECOND = 25


@dataclass(frozen=True)
class BarGrid:
    bpm: int
    starts: tuple[Fraction, ...]  # quarter-note onset of each bar, 0-based list
    total: Fraction  # quarter notes

    def frame_at(self, quarters: Fraction) -> int:
        return round(quarters * 60 * FRAMES_PER_SECOND / self.bpm)

    @property
    def frames(self) -> int:
        return self.frame_at(self.total)

    @property
    def bars(self) -> int:
        return len(self.starts)

    def bar_start_frame(self, bar: int) -> int:
        """1-based bar -> first frame. bar == bars+1 is the end of the score."""
        if bar == self.bars + 1:
            return self.frames
        if not 1 <= bar <= self.bars:
            raise ValueError(f"bar {bar} is outside the score (1..{self.bars})")
        return self.frame_at(self.starts[bar - 1])


def grid(abc: str) -> BarGrid:
    score = parse_abc(abc)
    vocal = score.voices["Vocal"]
    return BarGrid(score.bpm, tuple(start for start, _, _ in vocal.bars), vocal.time)


def parse_bars(spec: str) -> tuple[int, int | None]:
    """`9-16` -> (9, 16); `9-end` / `9-` -> (9, None); `9` -> (9, 9)."""
    match = re.fullmatch(r"\s*(\d+)\s*(?:-\s*(\d+|end)?\s*)?", spec)
    if not match:
        raise ValueError(f"`{spec}` is not a bar range; use A-B, A-end or A")
    first = int(match.group(1))
    if "-" not in spec:
        last: int | None = first
    else:
        last = None if match.group(2) in (None, "end") else int(match.group(2))
    if first < 1 or (last is not None and last < first):
        raise ValueError(f"`{spec}` is not an increasing 1-based bar range")
    return first, last


def bar_frames(abc: str, spec: str) -> tuple[int, int | None]:
    """Bar range -> [start_frame, end_frame) with None meaning the end of the song."""
    g = grid(abc)
    first, last = parse_bars(spec)
    start = g.bar_start_frame(first)
    if last is None:
        return start, None
    if last > g.bars:
        raise ValueError(f"bar {last} is outside the score (1..{g.bars})")
    return start, g.bar_start_frame(last + 1)
