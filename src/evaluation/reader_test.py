"""Assistive-reader test (v2): does the Fruit Reader say the right thing?

Box metrics (mAP) judge every box. A blind user only hears the fruit NAMES, so this test measures
what the reader says, per photo:

1. Naming on the locked test split (1,432 images): for each image, the set of fruit types the
   reader names (YOLO11n, app settings, confidence 0.25) vs. the labeled set. Also compares the
   v1 six-fruit model on the same v1-source images.
2. Hard conditions: the 12 unseen photos x 11 simulated conditions (src/evaluation/hard_cases.py),
   for the v1 model and the 31-fruit model, plus whether the photo-quality check warns the user.
   NOTE: the quality thresholds were chosen on these photos, so the warning rate here shows the
   check works as designed; it is not an independent test.
3. Quality-check false alarms on the 1,432 test images (not used for tuning): how often the reader
   wrongly tells the user that a normal dataset photo is dark, blurry or overexposed.

Usage (from the project root):
    python -m src.evaluation.reader_test
"""
from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

from src.assist.quality import check_quality
from src.assist.reader import FRUIT31_CONFIG, V1_CONFIG, FruitReader
from src.evaluation.hard_cases import CONDITIONS, UNSEEN
from src.inference.image_processor import YOLO, FruitDetectionEngine
from src.utils.helpers import ensure_dir, load_config, save_json
from src.utils.image_utils import read_image

TEST = Path("data/processed_fruit31/test")
OUT = Path("outputs/reports/reader_test")
V1_SHARED = ["apple", "banana", "grape", "orange", "pineapple", "watermelon"]


def _named(engine: FruitDetectionEngine, image) -> set[str]:
    r = engine.run(image, method=YOLO)
    return {c.lower() for c, n in r.counts.items() if n}


def naming_test(classes: list[str]) -> dict:
    images = sorted((TEST / "images").iterdir())
    labels = {}
    for p in images:
        lab = TEST / "labels" / (p.stem + ".txt")
        lines = lab.read_text().split("\n") if lab.exists() else []
        labels[p.name] = {classes[int(line.split()[0])] for line in lines if line.strip()}

    e31 = FruitDetectionEngine(load_config(FRUIT31_CONFIG))
    said31 = {}
    for i, p in enumerate(images, 1):
        said31[p.name] = _named(e31, read_image(p))
        if i % 200 == 0:
            print(f"  naming test: {i}/{len(images)}", flush=True)
    e1 = FruitDetectionEngine(load_config(V1_CONFIG))
    v1_names = [p.name for p in images if p.name.startswith("v1_")]
    said1 = {n: _named(e1, read_image(TEST / "images" / n)) for n in v1_names}

    def per_class(said: dict, names: list[str], cls: str) -> dict:
        has = [n for n in names if cls in labels[n]]
        told = [n for n in names if cls in said[n]]
        return {"images": len(has),
                "named": round(sum(cls in said[n] for n in has) / max(len(has), 1), 3),
                "correct_when_said": round(sum(cls in labels[n] for n in told) / max(len(told), 1), 3)}

    names = [p.name for p in images]
    labeled = [n for n in names if labels[n]]
    return {
        "images": len(names),
        "exact_set": round(sum(said31[n] == labels[n] for n in names) / len(names), 3),
        "at_least_one_right": round(sum(bool(said31[n] & labels[n]) for n in labeled) / len(labeled), 3),
        "nothing_said": round(sum(not said31[n] for n in labeled) / len(labeled), 3),
        "per_class": {c: per_class(said31, names, c) for c in classes},
        "v1_comparison": {"images": len(v1_names),
                          "classes": {c: {"v1_model": per_class(said1, v1_names, c),
                                          "fruit31_model": per_class(said31, v1_names, c)} for c in V1_SHARED}},
    }


def hard_conditions() -> dict:
    with open(UNSEEN / "expected.csv", encoding="utf-8") as f:
        expected = {r["image"]: {c.lower() for c in r["expected_classes"].split(";")} for r in csv.DictReader(f)}
    readers = {"v1": FruitReader(load_config(V1_CONFIG), "v1"), "fruit31": FruitReader(load_config(FRUIT31_CONFIG), "31")}
    rows = []
    for name, exp in expected.items():
        original = read_image(UNSEEN / name)
        for cond, (desc, fn) in CONDITIONS.items():
            image = fn(original)
            row = {"image": name, "condition": cond, "expected": ";".join(sorted(exp))}
            for key, reader in readers.items():
                r = reader.read(image)
                found = {d.class_name.lower() for d in r.detection.detections}
                row[f"{key}_pass"] = bool(exp & found)
                row[f"{key}_wrong"] = ", ".join(sorted(found - exp))
                if key == "fruit31":
                    row["warning"] = r.quality.issues[0] if r.quality.issues else ""
                    row["spoken"] = r.message
            rows.append(row)
        print(f"  hard conditions: {name} done", flush=True)
    summary = {}
    for cond, (desc, _) in CONDITIONS.items():
        sel = [r for r in rows if r["condition"] == cond]
        summary[cond] = {"description": desc, "cases": len(sel),
                         "v1_pass": sum(r["v1_pass"] for r in sel),
                         "fruit31_pass": sum(r["fruit31_pass"] for r in sel),
                         "fruit31_wrong_type_cases": sum(bool(r["fruit31_wrong"]) for r in sel),
                         "warned": sum(bool(r["warning"]) for r in sel),
                         "failed_but_warned": sum((not r["fruit31_pass"]) and bool(r["warning"]) for r in sel),
                         "warnings": dict(Counter(r["warning"] for r in sel if r["warning"]))}
    return {"summary": summary, "cases": rows}


def quality_false_alarms() -> dict:
    images = sorted((TEST / "images").iterdir())
    issues = Counter()
    flagged = 0
    for p in images:
        q = check_quality(read_image(p))
        flagged += bool(q.issues)
        issues.update(q.issues[:1])
    return {"images": len(images), "warned": flagged, "rate": round(flagged / len(images), 3),
            "first_issue_counts": dict(issues)}


def write_report(res: dict) -> None:
    n, h, q = res["naming"], res["hard_conditions"]["summary"], res["quality_false_alarms"]
    pc = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    lines = [
        "# Assistive reader test (v2)", "",
        "Generated by `python -m src.evaluation.reader_test`. The reader is judged by **what it says**: the fruit "
        "names in its spoken sentence, per photo (YOLO11n 31-fruit model, app settings, confidence 0.25).", "",
        f"## 1. Naming on the locked test split ({n['images']:,} images)", "",
        "| Measure | Result |", "|---|---|",
        f"| At least one fruit named correctly | **{pc(n['at_least_one_right'])}** |",
        f"| Exactly the right set of fruit names (nothing missing, nothing extra) | **{pc(n['exact_set'])}** |",
        f"| Said nothing although fruit was present | {pc(n['nothing_said'])} |", "",
        "Per fruit, sorted from weakest: *named* = share of the photos containing the fruit where the reader "
        "says its name; *correct when said* = share of the times the reader says the name where the fruit "
        "is really there.", "",
        "| Fruit | Test photos | Named | Correct when said |", "|---|---|---|---|",
    ]
    for c, r in sorted(n["per_class"].items(), key=lambda kv: kv[1]["named"]):
        lines.append(f"| {c} | {r['images']} | {pc(r['named'])} | {pc(r['correct_when_said'])} |")
    v1 = n["v1_comparison"]
    lines += ["", f"### v1 model vs 31-fruit model on the same {v1['images']} v1-source test photos", "",
              "| Fruit | Photos | v1 model: named | 31-fruit: named | v1 model: correct when said | 31-fruit: correct when said |",
              "|---|---|---|---|---|---|"]
    for c, d in v1["classes"].items():
        a, b = d["v1_model"], d["fruit31_model"]
        lines.append(f"| {c} | {a['images']} | {pc(a['named'])} | {pc(b['named'])} | "
                     f"{pc(a['correct_when_said'])} | {pc(b['correct_when_said'])} |")
    tot = {k: sum(s[k] for s in h.values()) for k in ("v1_pass", "fruit31_pass", "cases", "warned")}
    lines += ["", "## 2. Hard conditions (12 unseen photos x 11 conditions)", "",
              "A case passes if the expected fruit is named. *Warned* = the photo-quality check told the user "
              "what is wrong with the photo. The thresholds were chosen on these photos, so the warning column "
              "shows the check works as designed, not an independent test (see section 3 for that).", "",
              "| Condition | v1 model | 31-fruit model | 31-fruit: wrong extra type | Warned | Failed but warned |",
              "|---|---|---|---|---|---|"]
    for s in h.values():
        lines.append(f"| {s['description']} | {s['v1_pass']}/12 | **{s['fruit31_pass']}/12** | "
                     f"{s['fruit31_wrong_type_cases']} | {s['warned']}/12 | {s['failed_but_warned']} |")
    lines += [f"| **Total** | {tot['v1_pass']}/{tot['cases']} | **{tot['fruit31_pass']}/{tot['cases']}** | | "
              f"{tot['warned']}/{tot['cases']} | |", "",
              "Warnings per condition: " + "; ".join(
                  f"{c} " + (", ".join(f"{k} {v}" for k, v in s["warnings"].items()) or "none") for c, s in h.items())
              + ". The 'overexposed' warnings on the 30-degree photos are false alarms caused by the white corners "
              "that the rotation adds, not by the photo itself.", "",
              "### What the reader said for the 12 original photos", "", "| Photo | Spoken sentence |", "|---|---|"]
    for r in res["hard_conditions"]["cases"]:
        if r["condition"] == "original":
            lines.append(f"| {r['image']} | {r['spoken']} |")
    lines += ["", f"## 3. Quality-check false alarms ({q['images']:,} test images, not used for tuning)", "",
              f"The reader warned about **{q['warned']} of {q['images']:,}** normal dataset photos "
              f"(**{pc(q['rate'])}**). First issue named: "
              + (", ".join(f"{k} {v}" for k, v in sorted(q["first_issue_counts"].items(), key=lambda kv: -kv[1])) or "none")
              + ". A warning never blocks the result: the reader still names the fruit it sees and adds the advice.", ""]
    (OUT / "READER_TEST.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    ensure_dir(OUT)
    classes = load_config(FRUIT31_CONFIG)["classes"]
    res = {"quality_false_alarms": quality_false_alarms()}
    print("quality false alarms:", res["quality_false_alarms"], flush=True)
    res["hard_conditions"] = hard_conditions()
    res["naming"] = naming_test(classes)
    save_json(res, OUT / "reader_test.json")
    write_report(res)
    n, h = res["naming"], res["hard_conditions"]["summary"]
    print(f"naming: at least one right {n['at_least_one_right']}, exact set {n['exact_set']}")
    for cond, s in h.items():
        print(f"  {cond:12s} v1 {s['v1_pass']:2d}/12  fruit31 {s['fruit31_pass']:2d}/12  warned {s['warned']:2d}/12")
    print(f"-> {OUT / 'READER_TEST.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
