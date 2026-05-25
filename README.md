# ECE228 Final Project: EEG Sleep Stage Classification

This repository contains the code for an ECE228 final project on automatic sleep-stage classification from EEG signals. The project compares three neural network architectures on a 21-subject subset of the Sleep-EDF Expanded sleep-cassette dataset:

1. **CNN-LSTM baseline**
2. **Pure Transformer ablation**
3. **CNN-Transformer proposed model**

## Project Summary

Sleep staging assigns each 30-second physiological recording segment to one of five sleep stages:

```text
Wake, N1, N2, N3, REM
```

Manual sleep scoring is time-consuming and can vary across human raters, so automatic sleep-stage classification is a useful biomedical time-series learning problem.

Following instructor feedback, the pipeline uses:

```text
EEG Fpz-Cz + EEG Pz-Oz
100 Hz sampling
0.5-40 Hz bandpass filtering
per-recording normalization
strict subject-wise train/eval/test split
CNN-Transformer instead of pure attention on raw EEG
```

## Repository Structure

```text
README.md
requirements.txt
.gitignore

scripts/
  data_get_preprocess/
    download_sleep_edf_pairs.py        download selected Sleep-EDF files
    preprocess_sleep_edf.py            convert EDF files into model-ready tensors

  training/
    train_models.py                    train, validate, and test models

  analysis/
    plot_results.py                    generate loss curves and metric bar plots
    plot_ablation_results.py           generate ablation metric plots

src/
  sleep_dataset.py                     PyTorch Dataset and split loading utilities
  models.py                            LSTM-only, CNN-LSTM, Pure Transformer, CNN-Transformer
  __init__.py

splits/
  split_15_3_3.csv                     fixed subject-wise split
  split_15_3_3.json
  train_subjects.txt
  eval_subjects.txt
  test_subjects.txt

result/
  loss_curves.png                       training/eval loss curves
  test_accuracy_macro_f1.png            accuracy and macro-F1 comparison
  test_per_class_f1.png                 per-class F1 comparison
  test_metrics_combined.png             combined result figure
  main_model_accuracy_macro_f1.png      main model accuracy and macro-F1
  cnn_importance_accuracy_macro_f1.png  CNN importance ablation
  cnn_importance_per_class_f1.png       CNN importance per-class F1
  channel_ablation_accuracy_macro_f1.png EEG channel ablation
  channel_ablation_per_class_f1.png     channel ablation per-class F1
  depth_ablation_accuracy_macro_f1.png  Transformer depth ablation
```

## Data Policy

Raw EDF files and processed `.npz` tensors are **not included** in this repository because they are large and reproducible.

Use the provided scripts to download and preprocess the data locally.

Expected local data folders after running the pipeline:

```text
dataset/                         raw EDF files
dataset/processed/records/       processed .npz files
dataset/processed/processed_index.csv
runs/                            training outputs
```

These folders are ignored by Git.

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

The main Python packages are:

```text
numpy
scipy
scikit-learn
torch
matplotlib
pandas
python-docx
```

## Full Pipeline

Run commands from the repository root.

### Step 1: Download Data

Download 21 distinct Sleep-EDF sleep-cassette subjects:

```powershell
python scripts\data_get_preprocess\download_sleep_edf_pairs.py --num-subjects 21
```

Preview which files will be downloaded:

```powershell
python scripts\data_get_preprocess\download_sleep_edf_pairs.py --num-subjects 21 --dry-run
```

Output:

```text
dataset/*.edf
```

Each subject has one PSG file and one matching Hypnogram file.

### Step 2: Preprocess Data

Convert raw EDF files into model-ready tensors:

```powershell
python scripts\data_get_preprocess\preprocess_sleep_edf.py --overwrite
```

Quick smoke test on one recording:

```powershell
python scripts\data_get_preprocess\preprocess_sleep_edf.py --limit 1 --overwrite
```

The preprocessing script:

1. Reads PSG and Hypnogram EDF files.
2. Selects `EEG Fpz-Cz` and `EEG Pz-Oz`.
3. Verifies or resamples signals to 100 Hz.
4. Applies 0.5-40 Hz bandpass filtering.
5. Applies per-recording, per-channel z-score normalization.
6. Segments signals into 30-second epochs.
7. Maps labels into five classes.
8. Merges Sleep stage 3 and Sleep stage 4 into `N3`.
9. Drops invalid labels such as `Sleep stage ?` and `Movement time`.
10. Keeps 30 minutes of Wake context before and after the sleep period.

Output:

```text
dataset/processed/records/<recording_id>.npz
dataset/processed/processed_index.csv
dataset/processed/preprocess_summary.json
```

Each `.npz` file contains:

```text
X: float32, shape = (num_epochs, 3000, 2)
y: int64,   shape = (num_epochs,)
```

Label IDs:

```text
0 = Wake
1 = N1
2 = N2
3 = N3
4 = REM
```

### Step 3: Train And Test Models

Train and test the three main models:

```powershell
python scripts\training\train_models.py --model all --epochs 30 --batch-size 64 --patience 5 --amp
```

If GPU memory is limited:

```powershell
python scripts\training\train_models.py --model all --epochs 30 --batch-size 32 --patience 5 --amp
```

Train one model only:

```powershell
python scripts\training\train_models.py --model lstm_only --epochs 30 --batch-size 64 --patience 5 --amp
python scripts\training\train_models.py --model cnn_lstm --epochs 30 --batch-size 64 --patience 5 --amp
python scripts\training\train_models.py --model pure_transformer --epochs 30 --batch-size 64 --patience 5 --amp
python scripts\training\train_models.py --model cnn_transformer --epochs 30 --batch-size 64 --patience 5 --amp
```

Run channel ablations:

```powershell
python scripts\training\train_models.py --model cnn_transformer --channels fpz_cz --epochs 30 --batch-size 64 --patience 5 --amp
python scripts\training\train_models.py --model cnn_transformer --channels pz_oz --epochs 30 --batch-size 64 --patience 5 --amp
```

Run a CNN-Transformer depth ablation:

```powershell
python scripts\training\train_models.py --model cnn_transformer --cnn-transformer-layers 3 --epochs 30 --batch-size 64 --patience 5 --amp
```

Training logic:

```text
Train set: updates model parameters
Eval set: selects the best checkpoint by macro-F1
Test set: final evaluation after training
```

The script automatically loads the best eval checkpoint and evaluates it on the test set.

Output:

```text
runs/<timestamp>/
  summary_metrics.csv
  cnn_lstm/
    best.pt
    history.csv
    test_metrics.json
    confusion_matrix.csv
    test_predictions.npz
  pure_transformer/
  cnn_transformer/
```

### Step 4: Plot Results

Generate report figures:

```powershell
python scripts\analysis\plot_results.py
```

By default, the plotting script selects the complete run with the best CNN-Transformer test macro-F1.

To plot a specific run:

```powershell
python scripts\analysis\plot_results.py --run-dir runs\<timestamp>
```

Generated figures:

```text
loss_curves.png              training/eval loss curves including LSTM-only
test_accuracy_macro_f1.png   accuracy and macro-F1 comparison including LSTM-only
test_per_class_f1.png        per-class F1 comparison including LSTM-only
test_metrics_combined.png    accuracy, macro-F1, and per-class F1 together
test_metric_values.csv       plotted values as a table
```

Generate ablation figures:

```powershell
python scripts\analysis\plot_ablation_results.py
```

The ablation plotting script writes figures to `result/`.

## Data Split

The fixed split uses 21 distinct subjects:

```text
Train: SC400-SC414, 15 subjects
Eval:  SC415-SC417, 3 subjects
Test:  SC418-SC420, 3 subjects
```

This is a strict subject-wise split:

```text
No subject appears in more than one split.
```

## Models

### LSTM-only

The LSTM-only ablation removes the CNN branch:

```text
Raw EEG patches -> LSTM -> classifier
```

This tests whether recurrent temporal modeling alone can replace CNN-based local feature extraction.

Approximate parameters:

```text
126K
```

### CNN-LSTM

The baseline follows the CNN-LSTM sleep-staging idea:

```text
CNN branch: extracts local EEG waveform features
LSTM branch: models temporal patterns
Fusion classifier: predicts sleep stage
```

Approximate parameters:

```text
207K
```

### Pure Transformer

The ablation model removes the CNN front-end:

```text
Raw EEG patches -> linear embedding -> Transformer encoder -> classifier
```

This tests whether attention alone is sufficient for EEG sleep staging.

Approximate parameters:

```text
169K
```

### CNN-Transformer

The proposed model uses:

```text
CNN front-end -> Transformer encoder -> classifier
```

The CNN first extracts local EEG waveform features and reduces temporal length before self-attention.

Approximate parameters:

```text
232K
```

## Main Metrics

The report should emphasize:

```text
Accuracy
Macro-F1
Per-class F1
Confusion matrix
```

Macro-F1 is especially important because sleep-stage labels are imbalanced, especially for `N1`.

## Optional Context Experiment

The training script supports neighboring-epoch context:

```powershell
python scripts\training\train_models.py --model cnn_transformer --epochs 30 --batch-size 64 --patience 5 --context-size 5 --amp
```

`--context-size 5` gives the model five consecutive 30-second epochs and predicts the center epoch.

The default is:

```text
--context-size 1
```

The single-epoch setting is the main fair comparison.
