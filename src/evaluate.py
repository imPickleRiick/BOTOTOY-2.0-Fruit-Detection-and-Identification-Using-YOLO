"""One-command evaluation entry point (checkpoint 4 asks for src/evaluate.py).

Runs the existing evaluation tools in order:
  1. YOLO11n metrics (src/yolo/evaluate.py)
  2. OpenCV baseline vs YOLO11n on the same images (src/evaluation/compare_methods.py)
  3. Error review by cause (src/evaluation/error_review.py)

Usage (from the project root, dataset prepared as in the README):
    python -m src.evaluate                    # validation split
    python -m src.evaluate --split test --final   # the held-out test split
"""
from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--final", action="store_true", help="required for the held-out test split")
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        print("ERROR: the test split is held out. Use --split test --final.", file=sys.stderr)
        return 1

    extra = ["--split", args.split] + (["--final"] if args.final else [])
    steps = [("YOLO11n evaluation", "src.yolo.evaluate"),
             ("Baseline vs YOLO11n", "src.evaluation.compare_methods"),
             ("Error review", "src.evaluation.error_review")]
    for title, module in steps:
        print(f"\n=== {title} ({module}) ===", flush=True)
        code = subprocess.call([sys.executable, "-m", module, *extra])
        if code != 0:
            print(f"ERROR: {module} failed with exit code {code}", file=sys.stderr)
            return code
    print(f"\nDone. Reports: outputs/yolo/eval_{args.split}/, outputs/comparisons/{args.split}/, "
          f"outputs/yolo/eval_{args.split}/error_review/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
