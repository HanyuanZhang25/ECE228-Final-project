#!/usr/bin/env python
"""
Preprocess Sleep-EDF PSG/Hypnogram pairs into model-ready tensors.

This script follows the project preprocessing plan:
  - use EEG Fpz-Cz and EEG Pz-Oz;
  - verify/resample to 100 Hz;
  - apply 0.5-40 Hz bandpass filtering;
  - perform per-recording, per-channel z-score normalization;
  - segment into 30-second epochs;
  - map hypnogram labels into Wake, N1, N2, N3, REM;
  - keep the existing strict subject-wise train/eval/test split.

Outputs:
  dataset/processed/records/<recording_id>.npz
    X: float32, shape (num_epochs, 3000, 2)
    y: int64, shape (num_epochs,)
    epoch_start_sec: int64, shape (num_epochs,)

  dataset/processed/processed_index.csv
  dataset/processed/preprocess_summary.json

Example:
  python scripts/data_get_preprocess/preprocess_sleep_edf.py

For a quick single-recording smoke test:
  python scripts/data_get_preprocess/preprocess_sleep_edf.py --limit 1
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal


CHANNELS = ("EEG Fpz-Cz", "EEG Pz-Oz")
LABEL_TO_ID = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
LABEL_NAMES = ("Wake", "N1", "N2", "N3", "REM")


@dataclass
class EdfSignalInfo:
    label: str
    physical_min: float
    physical_max: float
    digital_min: float
    digital_max: float
    samples_per_record: int
    sample_rate: float


@dataclass
class ProcessedRecord:
    split: str
    subject_id: str
    recording_id: str
    psg: str
    hypnogram: str
    output: str
    num_epochs: int
    num_dropped_epochs: int
    duration_hours: float
    label_counts: dict[str, int]


def read_ascii(handle: Any, n: int) -> str:
    return handle.read(n).decode("latin-1", errors="replace")


def read_edf_header(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        header: dict[str, Any] = {}
        header["version"] = read_ascii(handle, 8).strip()
        header["patient_id"] = read_ascii(handle, 80).strip()
        header["recording_id"] = read_ascii(handle, 80).strip()
        header["start_date"] = read_ascii(handle, 8).strip()
        header["start_time"] = read_ascii(handle, 8).strip()
        header["header_bytes"] = int(read_ascii(handle, 8).strip())
        header["reserved"] = read_ascii(handle, 44).strip()
        header["num_records"] = int(read_ascii(handle, 8).strip())
        record_duration_text = read_ascii(handle, 8).strip()
        header["record_duration_sec"] = float(record_duration_text or 0.0)
        header["num_signals"] = int(read_ascii(handle, 4).strip())

        num_signals = header["num_signals"]
        fields = [
            ("labels", 16),
            ("transducer", 80),
            ("physical_dimension", 8),
            ("physical_min", 8),
            ("physical_max", 8),
            ("digital_min", 8),
            ("digital_max", 8),
            ("prefiltering", 80),
            ("samples_per_record", 8),
            ("signal_reserved", 32),
        ]
        signals: dict[str, list[str]] = {}
        for field_name, width in fields:
            signals[field_name] = [read_ascii(handle, width).strip() for _ in range(num_signals)]
        header["signals"] = signals
    return header


def signal_infos(header: dict[str, Any]) -> list[EdfSignalInfo]:
    duration = float(header["record_duration_sec"])
    infos: list[EdfSignalInfo] = []
    signals = header["signals"]
    for i, label in enumerate(signals["labels"]):
        samples_per_record = int(signals["samples_per_record"][i])
        sample_rate = samples_per_record / duration if duration else 0.0
        infos.append(
            EdfSignalInfo(
                label=label,
                physical_min=float(signals["physical_min"][i]),
                physical_max=float(signals["physical_max"][i]),
                digital_min=float(signals["digital_min"][i]),
                digital_max=float(signals["digital_max"][i]),
                samples_per_record=samples_per_record,
                sample_rate=sample_rate,
            )
        )
    return infos


def digital_to_physical(raw: np.ndarray, info: EdfSignalInfo) -> np.ndarray:
    raw_f = raw.astype(np.float32, copy=False)
    scale = (info.physical_max - info.physical_min) / (info.digital_max - info.digital_min)
    return (raw_f - info.digital_min) * scale + info.physical_min


def read_selected_channels(psg_path: Path, channel_names: tuple[str, ...]) -> tuple[np.ndarray, float, dict[str, Any]]:
    header = read_edf_header(psg_path)
    infos = signal_infos(header)
    label_to_index = {info.label: i for i, info in enumerate(infos)}

    missing = [name for name in channel_names if name not in label_to_index]
    if missing:
        raise ValueError(f"{psg_path.name} is missing channels: {missing}")

    selected_indices = [label_to_index[name] for name in channel_names]
    selected_infos = [infos[i] for i in selected_indices]
    sample_rates = {round(info.sample_rate, 8) for info in selected_infos}
    if len(sample_rates) != 1:
        raise ValueError(f"{psg_path.name} selected channels have different sample rates: {sample_rates}")
    sample_rate = selected_infos[0].sample_rate

    num_records = int(header["num_records"])
    samples_per_record = [info.samples_per_record for info in infos]
    record_total_samples = sum(samples_per_record)
    output = [
        np.empty(num_records * info.samples_per_record, dtype=np.float32)
        for info in selected_infos
    ]

    selected_lookup = {signal_index: out_index for out_index, signal_index in enumerate(selected_indices)}
    offsets = np.cumsum([0] + samples_per_record)

    with psg_path.open("rb") as handle:
        handle.seek(int(header["header_bytes"]))
        for record_idx in range(num_records):
            record = np.fromfile(handle, dtype="<i2", count=record_total_samples)
            if record.size != record_total_samples:
                raise ValueError(f"{psg_path.name} ended unexpectedly at record {record_idx}")

            for signal_index, out_index in selected_lookup.items():
                start = offsets[signal_index]
                end = offsets[signal_index + 1]
                info = infos[signal_index]
                out_start = record_idx * info.samples_per_record
                out_end = out_start + info.samples_per_record
                output[out_index][out_start:out_end] = digital_to_physical(record[start:end], info)

    data = np.stack(output, axis=1)
    metadata = {
        "patient_id": header["patient_id"],
        "recording_id": header["recording_id"],
        "start_date": header["start_date"],
        "start_time": header["start_time"],
        "num_records": header["num_records"],
        "record_duration_sec": header["record_duration_sec"],
        "sample_rates": {info.label: info.sample_rate for info in selected_infos},
    }
    return data, sample_rate, metadata


def parse_hypnogram_annotations(hypnogram_path: Path) -> list[tuple[float, float, str]]:
    header = read_edf_header(hypnogram_path)
    with hypnogram_path.open("rb") as handle:
        handle.seek(int(header["header_bytes"]))
        text = handle.read().decode("latin-1", errors="replace")

    annotations: list[tuple[float, float, str]] = []
    for chunk in text.split("\x00"):
        if not chunk:
            continue
        parts = chunk.split("\x14")
        if not parts or "\x15" not in parts[0]:
            continue
        onset_text, duration_text = parts[0].split("\x15", 1)
        try:
            onset = float(onset_text)
            duration = float(duration_text)
        except ValueError:
            continue
        for annot in parts[1:]:
            if annot:
                annotations.append((onset, duration, annot))
    return annotations


def labels_for_epochs(
    annotations: list[tuple[float, float, str]],
    num_epochs: int,
    epoch_sec: int,
) -> np.ndarray:
    labels = np.full(num_epochs, -1, dtype=np.int64)
    for onset, duration, annotation in annotations:
        label_id = LABEL_TO_ID.get(annotation)
        if label_id is None:
            continue
        start_epoch = max(0, int(math.ceil(onset / epoch_sec - 1e-9)))
        end_epoch = min(num_epochs, int(math.floor((onset + duration) / epoch_sec + 1e-9)))
        if end_epoch > start_epoch:
            labels[start_epoch:end_epoch] = label_id
    return labels


def maybe_resample(data: np.ndarray, original_fs: float, target_fs: int) -> np.ndarray:
    if abs(original_fs - target_fs) < 1e-6:
        return data
    rounded_original = int(round(original_fs))
    gcd = math.gcd(rounded_original, target_fs)
    up = target_fs // gcd
    down = rounded_original // gcd
    return signal.resample_poly(data, up=up, down=down, axis=0).astype(np.float32, copy=False)


def bandpass_filter(data: np.ndarray, fs: int, low_hz: float, high_hz: float, order: int) -> np.ndarray:
    if low_hz <= 0 and high_hz >= fs / 2:
        return data.astype(np.float32, copy=False)
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    filtered = signal.sosfiltfilt(sos, data, axis=0)
    return filtered.astype(np.float32, copy=False)


def normalize_per_recording(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = data.mean(axis=0, keepdims=True)
    std = data.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    normalized = (data - mean) / std
    return normalized.astype(np.float32, copy=False), mean.squeeze(0), std.squeeze(0)


def epoch_signal(data: np.ndarray, labels: np.ndarray, target_fs: int, epoch_sec: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_per_epoch = target_fs * epoch_sec
    max_epochs_from_signal = data.shape[0] // points_per_epoch
    num_epochs = min(max_epochs_from_signal, labels.shape[0])
    trimmed = data[: num_epochs * points_per_epoch]
    X = trimmed.reshape(num_epochs, points_per_epoch, data.shape[1])
    y = labels[:num_epochs]
    epoch_start_sec = np.arange(num_epochs, dtype=np.int64) * epoch_sec
    valid = y >= 0
    return X[valid].astype(np.float32, copy=False), y[valid].astype(np.int64, copy=False), epoch_start_sec[valid]


def trim_wake_context(
    X: np.ndarray,
    y: np.ndarray,
    epoch_start_sec: np.ndarray,
    context_minutes: float | None,
    epoch_sec: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if context_minutes is None:
        return X, y, epoch_start_sec
    non_wake = np.flatnonzero(y != 0)
    if non_wake.size == 0:
        return X, y, epoch_start_sec
    context_epochs = int(round((context_minutes * 60) / epoch_sec))
    start = max(0, int(non_wake[0]) - context_epochs)
    end = min(len(y), int(non_wake[-1]) + context_epochs + 1)
    return X[start:end], y[start:end], epoch_start_sec[start:end]


def label_counts(y: np.ndarray) -> dict[str, int]:
    return {name: int((y == idx).sum()) for idx, name in enumerate(LABEL_NAMES)}


def process_record(
    row: dict[str, str],
    dataset_dir: Path,
    out_records_dir: Path,
    target_fs: int,
    epoch_sec: int,
    low_hz: float,
    high_hz: float,
    filter_order: int,
    wake_context_minutes: float | None,
    overwrite: bool,
) -> ProcessedRecord:
    psg_path = dataset_dir / row["psg"]
    hypnogram_path = dataset_dir / row["hypnogram"]
    output_path = out_records_dir / f"{row['recording_id']}.npz"

    if output_path.exists() and not overwrite:
        with np.load(output_path, allow_pickle=False) as cached:
            y = cached["y"]
            num_dropped_epochs = int(cached["num_dropped_epochs"])
            duration_hours = float(cached["duration_hours"])
        return ProcessedRecord(
            split=row["split"],
            subject_id=row["subject_id"],
            recording_id=row["recording_id"],
            psg=row["psg"],
            hypnogram=row["hypnogram"],
            output=str(output_path),
            num_epochs=int(y.shape[0]),
            num_dropped_epochs=num_dropped_epochs,
            duration_hours=duration_hours,
            label_counts=label_counts(y),
        )

    data, fs, metadata = read_selected_channels(psg_path, CHANNELS)
    raw_num_epochs = int(metadata["num_records"])
    data = maybe_resample(data, fs, target_fs)
    data = bandpass_filter(data, target_fs, low_hz, high_hz, filter_order)
    data, norm_mean, norm_std = normalize_per_recording(data)

    annotations = parse_hypnogram_annotations(hypnogram_path)
    labels = labels_for_epochs(annotations, raw_num_epochs, epoch_sec)
    X, y, epoch_start_sec = epoch_signal(data, labels, target_fs, epoch_sec)
    before_trim_epochs = int(y.shape[0])
    X, y, epoch_start_sec = trim_wake_context(X, y, epoch_start_sec, wake_context_minutes, epoch_sec)

    num_dropped_epochs = raw_num_epochs - before_trim_epochs
    duration_hours = raw_num_epochs * epoch_sec / 3600

    np.savez_compressed(
        output_path,
        X=X,
        y=y,
        epoch_start_sec=epoch_start_sec,
        subject_id=np.array(row["subject_id"]),
        recording_id=np.array(row["recording_id"]),
        split=np.array(row["split"]),
        channel_names=np.array(CHANNELS),
        label_names=np.array(LABEL_NAMES),
        target_fs=np.array(target_fs),
        epoch_sec=np.array(epoch_sec),
        bandpass_hz=np.array([low_hz, high_hz], dtype=np.float32),
        normalization_mean=norm_mean.astype(np.float32),
        normalization_std=norm_std.astype(np.float32),
        num_dropped_epochs=np.array(num_dropped_epochs),
        duration_hours=np.array(duration_hours),
        wake_context_minutes=np.array(-1 if wake_context_minutes is None else wake_context_minutes, dtype=np.float32),
    )

    return ProcessedRecord(
        split=row["split"],
        subject_id=row["subject_id"],
        recording_id=row["recording_id"],
        psg=row["psg"],
        hypnogram=row["hypnogram"],
        output=str(output_path),
        num_epochs=int(y.shape[0]),
        num_dropped_epochs=num_dropped_epochs,
        duration_hours=duration_hours,
        label_counts=label_counts(y),
    )


def read_split_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_index_csv(path: Path, records: list[ProcessedRecord]) -> None:
    fieldnames = [
        "split",
        "subject_id",
        "recording_id",
        "psg",
        "hypnogram",
        "output",
        "num_epochs",
        "num_dropped_epochs",
        "duration_hours",
        *[f"count_{name}" for name in LABEL_NAMES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                "split": record.split,
                "subject_id": record.subject_id,
                "recording_id": record.recording_id,
                "psg": record.psg,
                "hypnogram": record.hypnogram,
                "output": record.output,
                "num_epochs": record.num_epochs,
                "num_dropped_epochs": record.num_dropped_epochs,
                "duration_hours": f"{record.duration_hours:.6f}",
            }
            for label_name in LABEL_NAMES:
                row[f"count_{label_name}"] = record.label_counts.get(label_name, 0)
            writer.writerow(row)


def aggregate_summary(records: list[ProcessedRecord]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "label_names": list(LABEL_NAMES),
        "channels": list(CHANNELS),
        "splits": {},
    }
    for split in ("train", "eval", "test"):
        split_records = [record for record in records if record.split == split]
        counts = {name: 0 for name in LABEL_NAMES}
        for record in split_records:
            for name, count in record.label_counts.items():
                counts[name] += int(count)
        summary["splits"][split] = {
            "num_subjects": len({record.subject_id for record in split_records}),
            "num_recordings": len(split_records),
            "num_epochs": sum(record.num_epochs for record in split_records),
            "label_counts": counts,
            "recordings": [record.recording_id for record in split_records],
        }
    return summary


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    project_dir = script_dir.parent.parent
    parser = argparse.ArgumentParser(description="Preprocess Sleep-EDF files into model-ready tensors.")
    parser.add_argument("--dataset-dir", default=str(project_dir / "dataset"))
    parser.add_argument("--split-csv", default=str(project_dir / "splits" / "split_15_3_3.csv"))
    parser.add_argument("--out-dir", default=str(project_dir / "dataset" / "processed"))
    parser.add_argument("--target-fs", type=int, default=100)
    parser.add_argument("--epoch-sec", type=int, default=30)
    parser.add_argument("--low-hz", type=float, default=0.5)
    parser.add_argument("--high-hz", type=float, default=40.0)
    parser.add_argument("--filter-order", type=int, default=4)
    parser.add_argument(
        "--wake-context-minutes",
        type=float,
        default=30.0,
        help="Keep this much Wake context before the first and after the last non-Wake epoch. Use -1 to disable.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N rows for a quick smoke test.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    split_csv = Path(args.split_csv).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_records_dir = out_dir / "records"
    out_records_dir.mkdir(parents=True, exist_ok=True)

    wake_context_minutes = None if args.wake_context_minutes < 0 else args.wake_context_minutes
    rows = read_split_csv(split_csv)
    if args.limit is not None:
        rows = rows[: args.limit]

    records: list[ProcessedRecord] = []
    print(f"Dataset: {dataset_dir}")
    print(f"Split:   {split_csv}")
    print(f"Output:  {out_dir}")
    print(f"Rows:    {len(rows)}")
    print()

    for idx, row in enumerate(rows, start=1):
        print(f"[{idx:02d}/{len(rows):02d}] {row['split']:5s} {row['subject_id']} {row['recording_id']} ...", flush=True)
        record = process_record(
            row=row,
            dataset_dir=dataset_dir,
            out_records_dir=out_records_dir,
            target_fs=args.target_fs,
            epoch_sec=args.epoch_sec,
            low_hz=args.low_hz,
            high_hz=args.high_hz,
            filter_order=args.filter_order,
            wake_context_minutes=wake_context_minutes,
            overwrite=args.overwrite,
        )
        records.append(record)
        counts = ", ".join(f"{name}={record.label_counts[name]}" for name in LABEL_NAMES)
        print(f"       saved {record.num_epochs} epochs -> {Path(record.output).name} ({counts})", flush=True)

    index_path = out_dir / "processed_index.csv"
    summary_path = out_dir / "preprocess_summary.json"
    write_index_csv(index_path, records)

    summary = aggregate_summary(records)
    summary["preprocessing"] = {
        "target_fs": args.target_fs,
        "epoch_sec": args.epoch_sec,
        "channels": list(CHANNELS),
        "bandpass_hz": [args.low_hz, args.high_hz],
        "filter_order": args.filter_order,
        "normalization": "per-recording per-channel z-score",
        "wake_context_minutes": wake_context_minutes,
        "label_mapping": {k: LABEL_NAMES[v] for k, v in LABEL_TO_ID.items()},
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print(f"Wrote index:   {index_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
