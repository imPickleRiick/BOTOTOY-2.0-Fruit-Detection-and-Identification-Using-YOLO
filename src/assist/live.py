"""Live reader logic: decide WHEN to speak while the camera keeps sending frames (about 1 per second).

Speaking after every frame would be noise. The announcer speaks when:
- the same set of fruit types has been seen in STABLE_FRAMES frames in a row (the user has pointed the
  camera at the fruit for 2-3 seconds), and
- it is different from what was last said, or REPEAT_SECONDS have passed (still pointing at it).

When the camera shows no fruit for CLEAR_FRAMES frames, the scene is reset, so pointing at the same fruit
again is announced again. If a photo problem (dark, blurry) lasts STABLE_FRAMES frames while no fruit is
found, the problem is said instead (at most every QUALITY_REPEAT_SECONDS).

Only the fruit TYPES must be stable: counts and positions change slightly between frames, so the spoken
sentence uses the newest frame.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

STABLE_FRAMES = 3
CLEAR_FRAMES = 3
REPEAT_SECONDS = 15.0
QUALITY_REPEAT_SECONDS = 10.0


@dataclass
class LiveAnnouncer:
    stable_frames: int = STABLE_FRAMES
    clear_frames: int = CLEAR_FRAMES
    repeat_seconds: float = REPEAT_SECONDS
    _candidate: frozenset = field(default_factory=frozenset)
    _streak: int = 0
    _empty: int = 0
    _issue: str = ""
    _issue_streak: int = 0
    _last_said: frozenset | None = None
    _last_time: float = 0.0
    _last_quality_time: float = -1e9

    def update(self, fruit_types: set[str] | frozenset[str], sentence: str, quality_issue: str = "",
               quality_message: str = "", now: float | None = None) -> str:
        """Feed one frame's result; return the text to speak now ('' = stay quiet)."""
        now = time.monotonic() if now is None else now
        types = frozenset(fruit_types)

        if not types:
            self._empty += 1
            self._streak, self._candidate = 0, frozenset()
            if self._empty >= self.clear_frames:
                self._last_said = None  # scene cleared: the next fruit is announced fresh
            self._issue_streak = self._issue_streak + 1 if quality_issue and quality_issue == self._issue else int(bool(quality_issue))
            self._issue = quality_issue
            if (quality_issue and self._issue_streak >= self.stable_frames
                    and now - self._last_quality_time >= QUALITY_REPEAT_SECONDS):
                self._last_quality_time = now
                return quality_message
            return ""

        self._empty, self._issue_streak, self._issue = 0, 0, ""
        self._streak = self._streak + 1 if types == self._candidate else 1
        self._candidate = types
        if self._streak < self.stable_frames:
            return ""
        if types != self._last_said or now - self._last_time >= self.repeat_seconds:
            self._last_said, self._last_time = types, now
            return sentence
        return ""
