#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from src.models import build_model, count_parameters
from src.sleep_dataset import LABEL_NAMES, SleepEpochDataset, inverse_frequency_class_weights, load_split_arrays


MODEL_NAMES = ("cnn_lstm", "pure_transformer", "cnn_transformer")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_loader(dataset: SleepEpochDataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def batch_to_device(batch: tuple[torch.Tensor, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    limit_batches: int | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_samples = 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []

    for batch_idx, batch in enumerate(loader, start=1):
        if limit_batches is not None and batch_idx > limit_batches:
            break

        x, y = batch_to_device(batch, device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device.type, enabled=amp and device.type == "cuda"):
                logits = model(x)
                loss = criterion(logits, y)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

        total_loss += float(loss.item()) * y.size(0)
        total_samples += int(y.size(0))
        pred = logits.argmax(dim=1)
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=list(range(len(LABEL_NAMES))), average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    labels = list(range(len(LABEL_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = {
        LABEL_NAMES[i]: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in labels
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist(),
    }


def save_history(path: Path, history: list[dict[str, Any]]) -> None:
    fields = [
        "epoch",
        "train_loss",
        "train_accuracy",
        "train_macro_f1",
        "eval_loss",
        "eval_accuracy",
        "eval_macro_f1",
        "seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row[field] for field in fields})


def save_confusion_matrix(path: Path, matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *LABEL_NAMES])
        for label_name, row in zip(LABEL_NAMES, matrix):
            writer.writerow([label_name, *row])


def train_one_model(
    model_name: str,
    args: argparse.Namespace,
    train_dataset: SleepEpochDataset,
    eval_dataset: SleepEpochDataset,
    test_dataset: SleepEpochDataset,
    class_weights: torch.Tensor,
    device: torch.device,
    run_root: Path,
) -> dict[str, Any]:
    model_dir = run_root / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(model_name, context_size=args.context_size).to(device)
    params = count_parameters(model)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    train_loader = make_loader(train_dataset, args.batch_size, shuffle=True, num_workers=args.num_workers)
    eval_loader = make_loader(eval_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = make_loader(test_dataset, args.batch_size, shuffle=False, num_workers=args.num_workers)

    print(f"\n=== {model_name} ===")
    print(f"parameters: {params:,}")

    best_eval_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, Any]] = []
    best_ckpt = model_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_result = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
            amp=args.amp and device.type == "cuda",
            limit_batches=args.limit_train_batches,
        )
        eval_result = run_epoch(
            model=model,
            loader=eval_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
            amp=args.amp and device.type == "cuda",
            limit_batches=args.limit_eval_batches,
        )
        scheduler.step(eval_result["macro_f1"])
        seconds = time.time() - started

        row = {
            "epoch": epoch,
            "train_loss": train_result["loss"],
            "train_accuracy": train_result["accuracy"],
            "train_macro_f1": train_result["macro_f1"],
            "eval_loss": eval_result["loss"],
            "eval_accuracy": eval_result["accuracy"],
            "eval_macro_f1": eval_result["macro_f1"],
            "seconds": seconds,
        }
        history.append(row)
        print(
            f"epoch {epoch:02d} | "
            f"train loss {row['train_loss']:.4f} acc {row['train_accuracy']:.4f} mf1 {row['train_macro_f1']:.4f} | "
            f"eval loss {row['eval_loss']:.4f} acc {row['eval_accuracy']:.4f} mf1 {row['eval_macro_f1']:.4f} | "
            f"{seconds:.1f}s"
        )

        if eval_result["macro_f1"] > best_eval_f1:
            best_eval_f1 = eval_result["macro_f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_name": model_name,
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "eval_macro_f1": best_eval_f1,
                    "params": params,
                    "args": vars(args),
                },
                best_ckpt,
            )
        else:
            stale_epochs += 1

        if args.limit_train_batches is None and stale_epochs >= args.patience:
            print(f"early stopping at epoch {epoch}; best epoch {best_epoch} eval macro-F1={best_eval_f1:.4f}")
            break

    save_history(model_dir / "history.csv", history)

    checkpoint = torch.load(best_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_result = run_epoch(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        optimizer=None,
        scaler=None,
        amp=args.amp and device.type == "cuda",
        limit_batches=args.limit_eval_batches,
    )
    metrics = detailed_metrics(test_result["y_true"], test_result["y_pred"])
    metrics.update(
        {
            "model_name": model_name,
            "parameters": params,
            "best_epoch": best_epoch,
            "best_eval_macro_f1": float(best_eval_f1),
            "test_loss": float(test_result["loss"]),
        }
    )

    (model_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_confusion_matrix(model_dir / "confusion_matrix.csv", metrics["confusion_matrix"])
    np.savez_compressed(
        model_dir / "test_predictions.npz",
        y_true=test_result["y_true"],
        y_pred=test_result["y_pred"],
        label_names=np.array(LABEL_NAMES),
    )
    print(
        f"test | loss {test_result['loss']:.4f} acc {metrics['accuracy']:.4f} "
        f"macro-F1 {metrics['macro_f1']:.4f}"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train sleep-stage classification models.")
    parser.add_argument("--index-csv", default=str(PROJECT_DIR / "dataset" / "processed" / "processed_index.csv"))
    parser.add_argument("--run-dir", default=str(PROJECT_DIR / "runs"))
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--context-size", type=int, default=1, help="Odd number of consecutive epochs used as input context.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=228)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--limit-train-batches", type=int, default=None, help="Debug: train only N batches per epoch.")
    parser.add_argument("--limit-eval-batches", type=int, default=None, help="Debug: evaluate only N batches.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(0)}")

    index_csv = Path(args.index_csv).resolve()
    print(f"loading arrays from {index_csv}")
    train_arrays = load_split_arrays(index_csv, "train")
    eval_arrays = load_split_arrays(index_csv, "eval")
    test_arrays = load_split_arrays(index_csv, "test")

    print(f"train X={train_arrays.X.shape}, y={train_arrays.y.shape}")
    print(f"eval  X={eval_arrays.X.shape}, y={eval_arrays.y.shape}")
    print(f"test  X={test_arrays.X.shape}, y={test_arrays.y.shape}")

    train_dataset = SleepEpochDataset(train_arrays, context_size=args.context_size)
    eval_dataset = SleepEpochDataset(eval_arrays, context_size=args.context_size)
    test_dataset = SleepEpochDataset(test_arrays, context_size=args.context_size)
    class_weights = inverse_frequency_class_weights(train_arrays.y)
    print(f"class weights: {class_weights.numpy().round(4).tolist()} for {list(LABEL_NAMES)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.run_dir).resolve() / timestamp
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    selected_models = MODEL_NAMES if args.model == "all" else (args.model,)
    all_metrics = []
    for model_name in selected_models:
        metrics = train_one_model(
            model_name=model_name,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            test_dataset=test_dataset,
            class_weights=class_weights,
            device=device,
            run_root=run_root,
        )
        all_metrics.append(metrics)

    summary_path = run_root / "summary_metrics.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "model_name",
                "parameters",
                "best_epoch",
                "best_eval_macro_f1",
                "test_loss",
                "accuracy",
                "macro_f1",
                "weighted_f1",
            ],
        )
        writer.writeheader()
        for metrics in all_metrics:
            writer.writerow({field: metrics[field] for field in writer.fieldnames})

    print(f"\nRun directory: {run_root}")
    print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
