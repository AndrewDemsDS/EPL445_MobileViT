# MobileViT-Based Traffic Analysis for Digital Twin Optimization

> **EPL 445 – Digital Video Processing** | University of Cyprus, Spring 2026
>
> Team: Andreas Demosthenous & Marios Olymbios

## Overview

This project implements a **frame-level traffic classification** system using a
**MobileViT** backbone. Given traffic video footage, the pipeline:

1. Extracts and crops vehicle patches from annotated frames
2. Trains a MobileViT classifier to recognise vehicle types
3. Evaluates with accuracy, F1, sensitivity, specificity, ROC/AUC
4. Runs inference on new video, producing annotated output + class counts

### Target classes

| Class        | Description               |
|-------------|---------------------------|
| `car`        | Passenger cars             |
| `motorcycle` | Motorcycles / two-wheelers |
| `bus`        | Buses                      |
| `truck`      | Trucks, vans, heavy vehicles |
| `background` | Non-vehicle patches        |

---

## Quick Start

### 1. Install dependencies

```bash
# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Install project in editable mode
pip install -e .
```

### 2. Prepare the dataset

Download and process the **UA-DETRAC** dataset:

```bash
python scripts/prepare_dataset.py --data-root data
```

This will:
- Download UA-DETRAC train images and annotations (~4 GB)
- Crop vehicle patches by class
- Generate stratified train/val/test CSV splits

> **Note:** If the download is too slow, manually download the files and use
> `--skip-download` with files placed in `data/raw/`.

### 3. Train the model

```bash
bash scripts/run_train.sh
# or directly:
python -m src.training.train --config configs/train.yaml
```

Training uses **staged fine-tuning**:
- Epochs 1–3: backbone frozen, only classifier head trains
- Epochs 4+: full model fine-tuned with reduced learning rate

Best checkpoint is saved to `outputs/models/best_model.pth`.

### 4. Evaluate

```bash
bash scripts/run_eval.sh
# or directly:
python -m src.evaluation.evaluate --config configs/eval.yaml
```

Outputs:
- `outputs/metrics/test_metrics.json` — all numeric metrics
- `outputs/figures/confusion_matrix.png`
- `outputs/figures/roc_curves.png`
- `outputs/figures/per_class_f1.png`
- `outputs/figures/training_curves.png`

### 5. Video demo

```bash
# Place a traffic video at data/raw/sample_traffic.mp4 (or edit configs/demo.yaml)
bash scripts/run_demo.sh
```

Outputs:
- `outputs/predictions/annotated_output.mp4` — video with class labels
- `outputs/predictions/frame_predictions.csv` — per-frame predictions
- `outputs/predictions/class_counts.json` — aggregated counts

---

## Project Structure

```
├── configs/            # YAML configuration files
├── data/               # Dataset (raw, processed, splits)
├── scripts/            # CLI scripts and shell wrappers
├── src/
│   ├── datasets/       # Dataset classes and transforms
│   ├── models/         # MobileViT classifier wrapper
│   ├── training/       # Training loop, losses, engine
│   ├── evaluation/     # Metrics computation and plotting
│   ├── inference/      # Image/video prediction, aggregation
│   ├── utils/          # Seed, device, I/O, logging
│   └── app/            # Web dashboard (future)
├── outputs/            # Figures, metrics, models, predictions
├── tests/              # Unit tests
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Metrics (Interim #2)

The evaluation pipeline computes all metrics required by the course brief:

| Metric       | Description                              |
|-------------|------------------------------------------|
| Accuracy     | Overall correct predictions              |
| Sensitivity  | Per-class recall (true positive rate)    |
| Specificity  | True negative rate per class             |
| F-Score      | Per-class and macro F1                   |
| ROC / AUC    | One-vs-rest ROC curves + area under curve|

---

## Running Tests

```bash
python -m pytest tests/ -v
```

---

## Configuration

All settings are controlled via YAML files in `configs/`:

- `default.yaml` — shared defaults (model, image size, classes)
- `train.yaml` — training hyperparameters
- `eval.yaml` — evaluation settings and checkpoint path
- `demo.yaml` — video inference settings

---

## References

- MobileViT: [Mehta & Rastegari, 2022](https://arxiv.org/abs/2110.02178)
- UA-DETRAC: [Wen et al., 2020](https://detrac-db.rit.albany.edu/)
- timm: [Wightman, 2019](https://github.com/huggingface/pytorch-image-models)
