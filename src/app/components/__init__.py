"""Custom Streamlit components (plain HTML + JavaScript, no build step)."""
from __future__ import annotations

from pathlib import Path

import streamlit.components.v1 as components

_live_camera = components.declare_component("live_camera", path=str(Path(__file__).parent / "live_camera"))


def live_camera(result: dict | None, key: str, interval_ms: int = 700, rate: float = 0.95,
                max_side: int = 640) -> dict | None:
    """Live camera that sends frames and shows/speaks `result`.

    Returns the newest frame sent by the browser: {"seq": int, "frame": "data:image/jpeg;base64,..."},
    or None before the camera starts / after it stops.
    """
    return _live_camera(result=result, interval_ms=interval_ms, rate=rate, max_side=max_side,
                        key=key, default=None)
