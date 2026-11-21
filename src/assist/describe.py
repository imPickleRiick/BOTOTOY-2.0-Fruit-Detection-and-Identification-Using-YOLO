"""Turn detections into one short spoken sentence for a blind or low-vision user.

Example: "I see 3 mangoes (mangga): 2 on the left and 1 in the middle. I see 1 banana (saging), on the
right, but I am not sure."

- Position: the horizontal third of the photo that holds the box center (left / middle / right).
- Certainty: the highest confidence of that fruit type. >= 0.6 is said plainly; 0.4-0.6 adds
  "probably"; below 0.4 adds "but I am not sure".
- Order: most confident fruit type first, so the most reliable information is heard first.
- Large groups (> 5 boxes of one type) are read as "many", because counts in piles are unreliable
  (see the error review).
"""
from __future__ import annotations

from collections import defaultdict

from src.assist.fruit_names import names_for
from src.utils.detections import Detection

SURE, PROBABLY = 0.6, 0.4
MANY = 5
POSITIONS = ("on the left", "in the middle", "on the right")


def _position(det: Detection, width: int) -> int:
    x1, _, x2, _ = det.box
    center = (x1 + x2) / 2
    return min(2, max(0, int(3 * center / max(width, 1))))


def _join(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def describe(detections: list[Detection], width: int, filipino: bool = True, max_types: int = 4) -> str:
    """One sentence per fruit type (at most `max_types`), most confident first."""
    if not detections:
        return "I don't see any fruit that I know."
    by_class: dict[str, list[Detection]] = defaultdict(list)
    for d in detections:
        by_class[d.class_name].append(d)
    order = sorted(by_class, key=lambda c: -max(d.confidence for d in by_class[c]))

    sentences = []
    for cls in order[:max_types]:
        dets = by_class[cls]
        singular, plural, fil = names_for(cls)
        n = len(dets)
        name = (singular if n == 1 else plural) + (f" ({fil})" if filipino and fil and fil != singular else "")
        count = "many" if n > MANY else str(n)
        spots = [0, 0, 0]
        for d in dets:
            spots[_position(d, width)] += 1
        if n == 1:
            where = POSITIONS[spots.index(1)]
            text = f"I see 1 {name}, {where}"
        elif n > MANY or sum(1 for s in spots if s) == 1:
            used = sum(1 for s in spots if s)
            main = POSITIONS[max(range(3), key=lambda i: spots[i])]
            if used == 1:
                text = f"I see {count} {name}, {main}"
            elif used == 3:
                text = f"I see {count} {name}, across the photo"
            else:
                text = f"I see {count} {name}, mostly {main}"
        else:
            text = f"I see {count} {name}: " + _join([f"{s} {POSITIONS[i]}" for i, s in enumerate(spots) if s])
        best = max(d.confidence for d in dets)
        if best < PROBABLY:
            text += ", but I am not sure"
        elif best < SURE:
            text = text.replace("I see", "I probably see", 1)
        sentences.append(text + ".")
    if len(order) > max_types:
        sentences.append(f"There are also {len(order) - max_types} other kinds of fruit.")
    return " ".join(sentences)


def full_message(detections: list[Detection], width: int, quality_message: str = "", filipino: bool = True) -> str:
    """What the reader says: the photo problem (if any) first, then the fruits."""
    parts = []
    if quality_message:
        parts.append(quality_message)
    parts.append(describe(detections, width, filipino))
    if not detections and not quality_message:
        parts.append("Try holding the fruit closer, in good light, against a plain background.")
    return " ".join(parts)
