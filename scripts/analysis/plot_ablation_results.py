#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
LABEL_NAMES = ("Wake", "N1", "N2", "N3", "REM")


@dataclass(frozen=True)
class Experiment:
    key: str
    label: str
    run_dir: Path
    model_dir: str
    color: str


def load_metrics(experiment: Experiment) -> dict:
    path = experiment.run_dir / experiment.model_dir / "test_metrics.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def annotate_bars(ax, bars) -> None:
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.014,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_accuracy_macro_f1(experiments: list[Experiment], metrics: dict[str, dict], title: str, out_path: Path) -> None:
    x = np.arange(len(experiments))
    width = 0.36
    accuracy = [metrics[exp.key]["accuracy"] for exp in experiments]
    macro_f1 = [metrics[exp.key]["macro_f1"] for exp in experiments]

    fig, ax = plt.subplots(figsize=(max(8.5, 1.8 * len(experiments)), 5.0))
    bars1 = ax.bar(x - width / 2, accuracy, width, label="Accuracy", color="#5b8cc0")
    bars2 = ax.bar(x + width / 2, macro_f1, width, label="Macro-F1", color="#6aa57a")

    ax.set_title(title)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([exp.label for exp in experiments], rotation=20, ha="right")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    annotate_bars(ax, bars1)
    annotate_bars(ax, bars2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_f1(experiments: list[Experiment], metrics: dict[str, dict], title: str, out_path: Path) -> None:
    x = np.arange(len(LABEL_NAMES))
    width = min(0.8 / len(experiments), 0.22)
    offsets = (np.arange(len(experiments)) - (len(experiments) - 1) / 2) * width

    fig, ax = plt.subplots(figsize=(11.5, 5.4))
    for offset, exp in zip(offsets, experiments):
        values = [metrics[exp.key]["per_class"][label]["f1"] for label in LABEL_NAMES]
        ax.bar(x + offset, values, width, label=exp.label, color=exp.color)

    ax.set_title(title)
    ax.set_ylabel("F1 Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(LABEL_NAMES)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=min(len(experiments), 3), loc="upper center", bbox_to_anchor=(0.5, -0.12))

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_values(experiments: list[Experiment], metrics: dict[str, dict], out_path: Path) -> None:
    rows = []
    for exp in experiments:
        row = {
            "experiment": exp.key,
            "label": exp.label,
            "parameters": metrics[exp.key]["parameters"],
            "best_eval_macro_f1": metrics[exp.key]["best_eval_macro_f1"],
            "test_accuracy": metrics[exp.key]["accuracy"],
            "test_macro_f1": metrics[exp.key]["macro_f1"],
            "test_weighted_f1": metrics[exp.key]["weighted_f1"],
        }
        for label in LABEL_NAMES:
            row[f"f1_{label}"] = metrics[exp.key]["per_class"][label]["f1"]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_path, index=False)


def default_experiments() -> dict[str, Experiment]:
    runs = PROJECT_DIR / "runs"
    return {
        "lstm_only": Experiment("lstm_only", "LSTM-only", runs / "20260521_163309", "lstm_only", "#8c6bb1"),
        "cnn_lstm": Experiment("cnn_lstm", "CNN-LSTM", runs / "20260519_220130", "cnn_lstm", "#2f6fbb"),
        "pure_transformer": Experiment(
            "pure_transformer",
            "Pure Transformer",
            runs / "20260519_220130",
            "pure_transformer",
            "#b85252",
        ),
        "cnn_transformer_both": Experiment(
            "cnn_transformer_both",
            "CNN-Transformer\nBoth",
            runs / "20260519_220130",
            "cnn_transformer",
            "#2f8f5b",
        ),
        "cnn_transformer_fpz": Experiment(
            "cnn_transformer_fpz",
            "CNN-Transformer\nFpz-Cz",
            runs / "20260521_163402",
            "cnn_transformer",
            "#d08c38",
        ),
        "cnn_transformer_pz": Experiment(
            "cnn_transformer_pz",
            "CNN-Transformer\nPz-Oz",
            runs / "20260521_163454",
            "cnn_transformer",
            "#3f8f9f",
        ),
        "cnn_transformer_3l": Experiment(
            "cnn_transformer_3l",
            "CNN-Transformer\n3 layers",
            runs / "20260520_103509",
            "cnn_transformer",
            "#6b7fd7",
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot model and ablation test metrics.")
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "result"), help="Output directory for figures.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    experiments = default_experiments()
    metrics = {key: load_metrics(exp) for key, exp in experiments.items()}

    main_models = [
        experiments["cnn_lstm"],
        experiments["pure_transformer"],
        experiments["cnn_transformer_both"],
    ]
    cnn_importance = [
        experiments["lstm_only"],
        experiments["cnn_lstm"],
        experiments["pure_transformer"],
        experiments["cnn_transformer_both"],
    ]
    channel_ablation = [
        experiments["cnn_transformer_fpz"],
        experiments["cnn_transformer_pz"],
        experiments["cnn_transformer_both"],
    ]
    depth_ablation = [
        experiments["cnn_transformer_both"],
        experiments["cnn_transformer_3l"],
    ]

    plot_accuracy_macro_f1(
        main_models,
        metrics,
        "Main Model Test Accuracy and Macro-F1",
        out_dir / "main_model_accuracy_macro_f1.png",
    )
    plot_accuracy_macro_f1(
        cnn_importance,
        metrics,
        "CNN Importance Ablation",
        out_dir / "cnn_importance_accuracy_macro_f1.png",
    )
    plot_accuracy_macro_f1(
        channel_ablation,
        metrics,
        "EEG Channel Ablation",
        out_dir / "channel_ablation_accuracy_macro_f1.png",
    )
    plot_accuracy_macro_f1(
        depth_ablation,
        metrics,
        "CNN-Transformer Depth Ablation",
        out_dir / "depth_ablation_accuracy_macro_f1.png",
    )
    plot_per_class_f1(
        cnn_importance,
        metrics,
        "CNN Importance Ablation: Per-Class F1",
        out_dir / "cnn_importance_per_class_f1.png",
    )
    plot_per_class_f1(
        channel_ablation,
        metrics,
        "EEG Channel Ablation: Per-Class F1",
        out_dir / "channel_ablation_per_class_f1.png",
    )

    write_values(list(experiments.values()), metrics, out_dir / "ablation_metric_values.csv")

    print(f"Wrote ablation figures to: {out_dir}")
    for path in sorted(out_dir.glob("*ablation*.png")):
        print(path.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
