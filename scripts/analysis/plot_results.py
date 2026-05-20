#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_ORDER = ("cnn_lstm", "pure_transformer", "cnn_transformer")
MODEL_LABELS = {
    "cnn_lstm": "CNN-LSTM",
    "pure_transformer": "Pure Transformer",
    "cnn_transformer": "CNN-Transformer",
}
MODEL_COLORS = {
    "cnn_lstm": "#2f6fbb",
    "pure_transformer": "#b85252",
    "cnn_transformer": "#2f8f5b",
}
LABEL_NAMES = ("Wake", "N1", "N2", "N3", "REM")


def is_complete_run(run_dir: Path) -> bool:
    return all((run_dir / model / "history.csv").exists() for model in MODEL_ORDER) and all(
        (run_dir / model / "test_metrics.json").exists() for model in MODEL_ORDER
    )


def find_latest_complete_run(runs_dir: Path) -> Path:
    candidates = [p for p in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True) if is_complete_run(p)]
    if not candidates:
        raise FileNotFoundError(f"No complete run with all three models found under {runs_dir}")
    return candidates[0]


def find_best_complete_run(runs_dir: Path) -> Path:
    candidates: list[tuple[float, Path]] = []
    for run_dir in sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True):
        if not is_complete_run(run_dir):
            continue
        metrics_path = run_dir / "cnn_transformer" / "test_metrics.json"
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            score = float(metrics["macro_f1"])
        except Exception:
            continue
        candidates.append((score, run_dir))
    if not candidates:
        raise FileNotFoundError(f"No complete run with all three models found under {runs_dir}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def load_history(run_dir: Path) -> dict[str, pd.DataFrame]:
    histories = {}
    for model in MODEL_ORDER:
        path = run_dir / model / "history.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        histories[model] = pd.read_csv(path)
    return histories


def load_test_metrics(run_dir: Path) -> dict[str, dict]:
    metrics = {}
    for model in MODEL_ORDER:
        path = run_dir / model / "test_metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        metrics[model] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def plot_loss_curves(histories: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=False)

    for model, history in histories.items():
        label = MODEL_LABELS[model]
        color = MODEL_COLORS[model]
        axes[0].plot(history["epoch"], history["train_loss"], marker="o", linewidth=2, color=color, label=label)
        axes[1].plot(history["epoch"], history["eval_loss"], marker="o", linewidth=2, color=color, label=label)

    axes[0].set_title("Training Loss")
    axes[1].set_title("Evaluation Loss")
    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-Entropy Loss")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)

    fig.suptitle("Loss Curves Across Models", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_summary_bar(metrics: dict[str, dict], out_path: Path) -> None:
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    accuracy = [metrics[model]["accuracy"] for model in MODEL_ORDER]
    macro_f1 = [metrics[model]["macro_f1"] for model in MODEL_ORDER]

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#5b8cc0")
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#6aa57a")

    ax.set_title("Test Accuracy and Macro-F1")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)

    for bars in (bars1, bars2):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.015, f"{height:.3f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(metrics: dict[str, dict], out_path: Path) -> None:
    x = np.arange(len(LABEL_NAMES))
    width = 0.24

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    for offset, model in zip((-width, 0, width), MODEL_ORDER):
        values = [metrics[model]["per_class"][label]["f1"] for label in LABEL_NAMES]
        ax.bar(x + offset, values, width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])

    ax.set_title("Test Per-Class F1")
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_combined_test_bars(metrics: dict[str, dict], out_path: Path) -> None:
    metric_names = ("Accuracy", "Macro-F1", *LABEL_NAMES)
    values = {}
    for model in MODEL_ORDER:
        values[model] = [
            metrics[model]["accuracy"],
            metrics[model]["macro_f1"],
            *[metrics[model]["per_class"][label]["f1"] for label in LABEL_NAMES],
        ]

    x = np.arange(len(metric_names))
    width = 0.24
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    for offset, model in zip((-width, 0, width), MODEL_ORDER):
        ax.bar(x + offset, values[model], width, label=MODEL_LABELS[model], color=MODEL_COLORS[model])

    ax.set_title("Test Metrics Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_plot_values(metrics: dict[str, dict], out_path: Path) -> None:
    rows = []
    for model in MODEL_ORDER:
        row = {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "accuracy": metrics[model]["accuracy"],
            "macro_f1": metrics[model]["macro_f1"],
        }
        for label in LABEL_NAMES:
            row[f"f1_{label}"] = metrics[model]["per_class"][label]["f1"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training loss curves and test metric bar charts.")
    parser.add_argument("--run-dir", default=None, help="Run directory under runs/.")
    parser.add_argument(
        "--select",
        choices=("best", "latest"),
        default="best",
        help="When --run-dir is omitted, select the best complete run by CNN-Transformer test macro-F1 or the latest complete run.",
    )
    parser.add_argument("--out-dir", default=None, help="Output figure directory. Defaults to <run-dir>/figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    elif args.select == "latest":
        run_dir = find_latest_complete_run(PROJECT_DIR / "runs")
    else:
        run_dir = find_best_complete_run(PROJECT_DIR / "runs")
    out_dir = Path(args.out_dir).resolve() if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    histories = load_history(run_dir)
    metrics = load_test_metrics(run_dir)

    plot_loss_curves(histories, out_dir / "loss_curves.png")
    plot_summary_bar(metrics, out_dir / "test_accuracy_macro_f1.png")
    plot_per_class_f1(metrics, out_dir / "test_per_class_f1.png")
    plot_combined_test_bars(metrics, out_dir / "test_metrics_combined.png")
    write_plot_values(metrics, out_dir / "test_metric_values.csv")

    print(f"Run: {run_dir}")
    print(f"Wrote figures to: {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
