"""Step 9 of the OpenCV baseline: heuristic (rule-based) fruit classification.

Each region comes from one COLOR GROUP mask (red, warm, green, purple). The color group
narrows the candidates, then simple shape/color rules decide the fruit. The thresholds
live in config.yaml (baseline.rules) and were chosen from measured training-set color
statistics (outputs/baseline/calibration/), then checked on the validation split.

Decision rules (in order, first match wins):

  red    -> Watermelon if the region is triangular-ish (extent low, a slice of flesh)
            Apple otherwise
  warm   -> Banana    if strongly elongated
         -> Orange    if hue is orange (not yellow), vivid, and roughly round
         -> Pineapple if taller than wide and less saturated (brown-yellow skin)
         -> Banana    if yellow (higher hue)
         -> Orange    otherwise
  green  -> Watermelon if large and round (whole melon with green rind)
         -> Apple      if round and convex (green apple)
         -> Grape      if bumpy, ONLY when green_grapes_enabled (off by default: foliage
                       is bumpy too and caused many false positives on validation)
         -> rejected   otherwise (likely leaves / grass)
  purple -> Grape (dark grapes appear blue/purple in HSV)

The returned "confidence" is a RULE SCORE (rule reliability x region compactness),
not a calibrated probability. It is only used to rank detections.
"""
from __future__ import annotations


def classify_region(color_group: str, f: dict, rules: dict) -> tuple[str | None, str]:
    """Return (class_name or None, human-readable reason)."""
    if color_group == "red":
        if f["extent"] < rules["watermelon_flesh_max_extent"] and f["area_fraction"] >= rules["watermelon_flesh_min_area_fraction"]:
            return "Watermelon", f"red + triangular (extent {f['extent']:.2f})"
        return "Apple", f"red + compact (extent {f['extent']:.2f})"

    if color_group == "warm":
        if f["elongation"] >= rules["banana_min_elongation"]:
            return "Banana", f"warm + elongated ({f['elongation']:.1f})"
        if (f["mean_hue"] <= rules["orange_max_hue"] and f["mean_saturation"] >= rules["orange_min_saturation"]
                and f["circularity"] >= rules["round_min_circularity"]):
            return "Orange", f"orange hue {f['mean_hue']:.0f} + round ({f['circularity']:.2f})"
        if (f["height_ratio"] >= rules["pineapple_min_height_ratio"]
                and f["mean_saturation"] <= rules["pineapple_max_saturation"]):
            return "Pineapple", f"tall ({f['height_ratio']:.1f}) + dull warm color (S {f['mean_saturation']:.0f})"
        if f["mean_hue"] > rules["orange_max_hue"]:
            return "Banana", f"yellow hue {f['mean_hue']:.0f}"
        return "Orange", f"orange hue {f['mean_hue']:.0f}"

    if color_group == "green":
        if f["area_fraction"] >= rules["watermelon_min_area_fraction"] and f["circularity"] >= rules["round_min_circularity"]:
            return "Watermelon", f"large ({f['area_fraction']:.0%}) round green"
        if f["circularity"] >= rules["apple_min_circularity"] and f["solidity"] >= rules["apple_min_solidity"]:
            return "Apple", f"round green ({f['circularity']:.2f})"
        if rules.get("green_grapes_enabled") and f["solidity"] < rules["grape_max_solidity"]:
            return "Grape", f"bumpy green cluster (solidity {f['solidity']:.2f})"
        # Leaves and grass are green and irregular too; validation showed these are almost never fruit.
        return None, "irregular green region (likely foliage) - rejected"

    if color_group == "purple":
        return "Grape", f"purple/blue hue {f['mean_hue']:.0f}"

    return None, f"unknown color group '{color_group}'"


def rule_score(class_name: str, f: dict, base_confidence: dict[str, float],
               size_reference_fraction: float | None = None) -> float:
    """Rule reliability (per class, from config) x region compactness x region size.

    Validation showed that small and irregular regions are mostly background, so compact
    (high solidity) and larger regions get a higher score. Regions at least as large as
    `size_reference_fraction` of the image get the full size factor of 1.
    """
    score = base_confidence.get(class_name, 0.5) * (0.5 + 0.5 * f["solidity"])
    if size_reference_fraction:
        score *= min(1.0, (f["area_fraction"] / size_reference_fraction) ** 0.5)
    return round(score, 4)
