"""Fine-tune YOLO11n on the BOTOTOY fruit dataset (transfer learning).

Transfer learning: we start from YOLO11n weights pretrained on COCO (80 everyday object
classes, including apple, banana and orange). The backbone already knows general visual
features (edges, textures, shapes); Ultralytics replaces the detection head with a new one
for our 6 classes and fine-tunes the whole network on our training split. Model selection
uses the VALIDATION split only: Ultralytics keeps `best.pt` = the epoch with the best
validation fitness, and early stopping ends training when it stops improving.

Usage (from the project root; data/data.yaml must exist - see README "Dataset preparation"):
    python -m src.yolo.train                                   # settings from config.yaml
    python -m src.yolo.train --project /content/drive/MyDrive/BOTOTOY/runs   # Colab + Drive
    python -m src.yolo.train --resume path/to/weights/last.pt  # continue an interrupted run
    python -m src.yolo.train --epochs 1 --fraction 0.02 --imgsz 320 --device cpu   # smoke test
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import time
from pathlib import Path

from src.utils.helpers import load_config, resolve_path, save_json, system_info


def resolve_device(device: str | int) -> str | int:
    """'auto' -> first CUDA GPU if available, otherwise 'cpu'."""
    if str(device) != "auto":
        return device
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def software_info() -> dict:
    import torch
    import ultralytics
    info = {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
            "cuda_available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 1)
        info["cuda"] = torch.version.cuda
    return info


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def best_epoch_from_results(results_csv: Path) -> dict:
    """Read Ultralytics results.csv and return the epoch with the best validation mAP50-95."""
    import pandas as pd
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]
    key = "metrics/mAP50-95(B)"
    best = df.loc[df[key].idxmax()]
    return {
        "epochs_completed": int(len(df)),
        "best_epoch": int(best["epoch"]),
        "best_val_precision": round(float(best["metrics/precision(B)"]), 4),
        "best_val_recall": round(float(best["metrics/recall(B)"]), 4),
        "best_val_mAP50": round(float(best["metrics/mAP50(B)"]), 4),
        "best_val_mAP50-95": round(float(best[key]), 4),
        # cumulative epoch time logged by Ultralytics (correct even when training was resumed)
        "logged_training_minutes": round(float(df["time"].iloc[-1]) / 60, 1) if "time" in df else None,
    }


def train(cfg: dict, overrides: dict, resume: str | None = None, copy_weights: bool = True) -> Path:
    from ultralytics import YOLO

    t = {**cfg["yolo"]["train"], **{k: v for k, v in overrides.items() if v is not None}}
    t["device"] = resolve_device(t["device"])
    data_yaml = resolve_path(cfg["dataset"]["data_yaml"])
    if not resume and not data_yaml.is_file():
        raise FileNotFoundError(f"{data_yaml} not found. Run the dataset preparation steps "
                                "(download_dataset, dataset_validator, dataset_splitter --from-manifest).")

    started = time.time()
    if resume:
        print(f"Resuming training from {resume}")
        model = YOLO(resume)
        model.train(resume=True)
    else:
        model_path = resolve_path(cfg["yolo"]["model"])
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model = YOLO(str(model_path))  # downloads the pretrained checkpoint if missing
        project = Path(t["project"])
        model.train(
            data=str(data_yaml), imgsz=t["imgsz"], epochs=t["epochs"], batch=t["batch"],
            patience=t["patience"], optimizer=t["optimizer"], lr0=t["lr0"], momentum=t["momentum"],
            seed=t["seed"], deterministic=t["deterministic"], freeze=t["freeze"], workers=t["workers"],
            cache=t["cache"], device=t["device"], fraction=t.get("fraction", 1.0),
            project=str(project if project.is_absolute() else resolve_path(project)), name=t["name"],
            plots=True, val=True,
        )
    duration_min = (time.time() - started) / 60

    run_dir = Path(model.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():  # never report success without the weights file
        raise RuntimeError(f"Training finished but {best} was not created.")

    summary = {
        "run_dir": run_dir.name,
        "resumed_from": Path(resume).name if resume else None,
        "settings": {k: t[k] for k in ("imgsz", "epochs", "batch", "patience", "optimizer", "lr0", "momentum",
                                       "seed", "deterministic", "freeze", "workers", "cache")}
                    | {"device": str(t["device"]), "fraction": t.get("fraction", 1.0),
                       "pretrained": Path(cfg["yolo"]["model"]).name},
        "training_duration_minutes": round(duration_min, 1),
        "system": system_info() | software_info(),
        "best_weights": {"file": "weights/best.pt", "size_mb": round(best.stat().st_size / 1e6, 2),
                         "sha256": sha256(best)},
    }
    results_csv = run_dir / "results.csv"
    if results_csv.is_file():
        summary["validation"] = best_epoch_from_results(results_csv)
    save_json(summary, run_dir / "training_summary.json")

    if copy_weights:
        target = resolve_path(cfg["yolo"]["trained_weights"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, target)
        print(f"Copied best weights to {target}")
    print(f"\nTraining complete in {duration_min:.1f} min. Run folder: {run_dir}")
    if "validation" in summary:
        print(f"Best validation epoch: {summary['validation']}")
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", help='"auto", "cpu" or a GPU index such as 0')
    parser.add_argument("--fraction", type=float, help="use only this fraction of the training set (smoke tests)")
    parser.add_argument("--project", help="folder for run outputs (e.g. a Google Drive folder on Colab)")
    parser.add_argument("--name", help="run name inside the project folder")
    parser.add_argument("--resume", help="path to an interrupted run's weights/last.pt")
    parser.add_argument("--no-copy", action="store_true", help="do not copy best.pt to models/trained/")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides = {k: getattr(args, k) for k in ("epochs", "batch", "imgsz", "patience", "workers", "device",
                                               "fraction", "project", "name")}
    try:
        train(cfg, overrides, resume=args.resume, copy_weights=not args.no_copy)
    except Exception as exc:
        msg = str(exc)
        print(f"\nERROR: training failed: {type(exc).__name__}: {msg}", file=sys.stderr)
        if "out of memory" in msg.lower():
            print("Hint: reduce --batch (e.g. 8) or --imgsz, or use fewer --workers.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
