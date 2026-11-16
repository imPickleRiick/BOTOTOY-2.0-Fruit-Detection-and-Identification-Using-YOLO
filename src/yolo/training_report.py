"""Import a finished training run (from Colab) and build the training report.

Input: the results ZIP downloaded from the Colab notebook (or an extracted run folder)
containing results.csv, args.yaml, training_summary.json, Ultralytics plots and weights/best.pt.

What it does:
1. Extracts the ZIP into outputs/yolo/training/run/ (a fresh folder).
2. Verifies best.pt against the SHA-256 recorded on Colab, then installs it as
   models/trained/bototoy_yolo11n.pt (the path in config.yaml).
3. Plots training/validation loss curves and validation metrics per epoch.
4. Writes outputs/yolo/training/TRAINING_REPORT.md.

Usage:
    python -m src.yolo.training_report path/to/bototoy_training_results.zip
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import yaml

from src.utils.helpers import ensure_dir, load_config, load_json, resolve_path

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
TRAIN_COLOR, VAL_COLOR = "#2a78d6", "#eb6834"
METRIC_COLORS = {"mAP50": "#2a78d6", "mAP50-95": "#eb6834", "precision": "#1baf7a", "recall": "#6250d6"}


def extract_run(source: Path, dest: Path) -> Path:
    """Copy/extract the run into a fresh folder and return the folder that holds results.csv."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    if source.is_dir():
        shutil.copytree(source, dest, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as zf:
            zf.extractall(dest)  # zipfile strips absolute paths and '..' (safe extraction)
    hits = list(dest.rglob("results.csv"))
    if not hits:
        raise FileNotFoundError("results.csv not found in the training results")
    return hits[0].parent


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_curves(df: pd.DataFrame, out_dir: Path, best_epoch: int | None) -> list[str]:
    saved = []
    # Loss curves: one small panel per loss type (train vs val)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, loss in zip(axes, ("box_loss", "cls_loss", "dfl_loss")):
        ax.plot(df["epoch"], df[f"train/{loss}"], color=TRAIN_COLOR, linewidth=2, label="train")
        ax.plot(df["epoch"], df[f"val/{loss}"], color=VAL_COLOR, linewidth=2, label="val")
        _style(ax, loss, "loss")
        if best_epoch:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle="--")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Training and validation losses (dashed line = best epoch)", x=0.01, ha="left", fontsize=12)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    saved.append("loss_curves.png")

    fig, ax = plt.subplots(figsize=(9, 4))
    cols = {"mAP50": "metrics/mAP50(B)", "mAP50-95": "metrics/mAP50-95(B)",
            "precision": "metrics/precision(B)", "recall": "metrics/recall(B)"}
    for name, col in cols.items():
        ax.plot(df["epoch"], df[col], color=METRIC_COLORS[name], linewidth=2, label=name)
        ax.annotate(f"{df[col].iloc[-1]:.2f}", (df["epoch"].iloc[-1], df[col].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", fontsize=8, color=MUTED, va="center")
    if best_epoch:
        ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle="--")
    _style(ax, "Validation metrics per epoch (dashed line = best epoch)", "score")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=9, ncols=4, loc="lower right")
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(out_dir / "validation_metrics.png", dpi=130, facecolor=SURFACE)
    plt.close(fig)
    saved.append("validation_metrics.png")
    return saved


def build_report(source: Path, cfg: dict, out_dir: str = "outputs/yolo/training",
                 weights_dest: str | None = None) -> Path:
    from src.yolo.train import best_epoch_from_results, sha256

    out_dir = ensure_dir(out_dir)
    run_dir = extract_run(source, out_dir / "run")
    summary_path = run_dir / "training_summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    args = yaml.safe_load((run_dir / "args.yaml").read_text(encoding="utf-8")) if (run_dir / "args.yaml").is_file() else {}

    # --- verify and install the weights ---
    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError("weights/best.pt missing from the training results")
    digest = sha256(best)
    expected = summary.get("best_weights", {}).get("sha256")
    if expected and expected != digest:
        raise ValueError(f"best.pt checksum mismatch (expected {expected[:12]}…, got {digest[:12]}…)")
    target = resolve_path(weights_dest or cfg["yolo"]["trained_weights"])
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    for heavy in (run_dir / "weights").glob("*.pt"):  # keep only the installed copy in the repo
        heavy.unlink()

    df = pd.read_csv(run_dir / "results.csv")
    df.columns = [c.strip() for c in df.columns]
    val = best_epoch_from_results(run_dir / "results.csv")
    charts = plot_curves(df, out_dir, val["best_epoch"])

    s = summary.get("settings", {})
    sysinfo = summary.get("system", {})
    lines = ["# YOLO11n Training Report - BOTOTOY 2.0\n",
             "## Configuration\n", "| Setting | Value |", "|---|---|"]
    rows = {
        "Model": f"YOLO11n, transfer learning from COCO-pretrained `{s.get('pretrained', 'yolo11n.pt')}`",
        "Image size": args.get("imgsz", s.get("imgsz")), "Epochs (max)": args.get("epochs", s.get("epochs")),
        "Epochs completed": val["epochs_completed"], "Early-stopping patience": args.get("patience", s.get("patience")),
        "Batch size": args.get("batch", s.get("batch")), "Optimizer": f"{args.get('optimizer', s.get('optimizer'))}, "
        f"lr0 {args.get('lr0', s.get('lr0'))}, momentum {args.get('momentum', s.get('momentum'))}",
        "Seed / deterministic": f"{args.get('seed', s.get('seed'))} / {args.get('deterministic', s.get('deterministic'))}",
        "Frozen layers": args.get("freeze", s.get("freeze")) or "none (all layers fine-tuned)",
        "Augmentation": f"mosaic {args.get('mosaic')}, flip-LR {args.get('fliplr')}, HSV h/s/v "
                        f"{args.get('hsv_h')}/{args.get('hsv_s')}/{args.get('hsv_v')}, scale {args.get('scale')}",
        "Training duration": f"{val.get('logged_training_minutes') or '?'} min of epochs (Ultralytics log); "
                             f"{summary.get('training_duration_minutes', '?')} min wall-clock for the last session",
        "Hardware": f"{sysinfo.get('gpu', 'no GPU')} ({sysinfo.get('gpu_memory_gb', '?')} GB), CPU {sysinfo.get('cpu', '?')}",
        "Software": f"Python {sysinfo.get('python', '?')}, torch {sysinfo.get('torch', '?')}, "
                    f"Ultralytics {sysinfo.get('ultralytics', '?')}, CUDA {sysinfo.get('cuda', '?')}",
        "Installed weights": f"`{cfg['yolo']['trained_weights']}` ({target.stat().st_size / 1e6:.1f} MB, "
                             f"sha256 `{digest[:12]}…`, verified)",
    }
    lines += [f"| {k} | {v} |" for k, v in rows.items()]
    lines += ["", "## Best checkpoint (selected on the validation split)\n",
              f"Best epoch **{val['best_epoch']}** of {val['epochs_completed']}: validation precision "
              f"{val['best_val_precision']:.3f}, recall {val['best_val_recall']:.3f}, "
              f"mAP50 **{val['best_val_mAP50']:.3f}**, mAP50-95 **{val['best_val_mAP50-95']:.3f}** "
              "(values logged during training).\n",
              "## Curves\n"] + [f"![{c}]({c})\n" for c in charts]
    lines += ["Ultralytics' own plots (results, confusion matrix, PR curves, label statistics, training "
              "batches) are in `run/`.\n"]
    (out_dir / "TRAINING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Installed weights -> {target}")
    print(f"Best epoch {val['best_epoch']}/{val['epochs_completed']}: val mAP50 {val['best_val_mAP50']}, "
          f"mAP50-95 {val['best_val_mAP50-95']}")
    return out_dir / "TRAINING_REPORT.md"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", help="results ZIP from Colab, or an extracted run folder")
    parser.add_argument("--out", default="outputs/yolo/training", help="report folder")
    parser.add_argument("--weights-dest", default=None, help="where to install best.pt (default: config)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    try:
        report = build_report(Path(args.source), load_config(args.config), args.out, args.weights_dest)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Report: {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
