# TASKS.md — EPL445 Phase 2 Status

## Goal

Build the minimum working codebase for **Interim Presentation #2** of EPL445 (Digital Video Processing).

The milestone is **first runnable results**, not a full production system. The repo must support:

- traffic video input
- a lightweight **MobileViT-based** baseline
- evaluation on a **held-out validation/test set**
- exported metrics and plots for the presentation
- a simple video demo pipeline

This aligns with the course brief for Interim Presentation #2, which asks for **first runs** and performance metrics such as **accuracy, sensitivity, specificity, ROC curves, AUC, and F-score**, preferably on the **evaluation/test set**. The brief also allows using **pre-trained models** when full training is expensive.

## Completion Status

### ✅ Completed

1. **Project structure** — full repo layout with configs, scripts, src, tests
2. **Dataset preparation** — UA-DETRAC (car/bus/truck/background, ~475K crops, 4 classes)
3. **MobileViT model wrapper** — `timm`-based, with freeze/unfreeze for staged fine-tuning
4. **Training** — 15 epochs, best val accuracy **96.63%**, checkpoint saved
5. **Evaluation on test set** — accuracy **96.71%**, macro F1 **96.88%**, macro AUC **99.79%**
6. **All required plots** — confusion matrix, ROC curves, per-class F1, training curves
7. **Video demo pipeline** — sliding-window detection on independent traffic video
8. **Aggregated class counts** — per-frame detections, density estimate, JSON output
9. **README with exact commands** — full instructions for all pipeline steps
10. **Unit tests** — 10 tests, all passing

### Results Summary (Test Set)

| Metric           | Value   |
|-----------------|---------|
| Accuracy         | 96.71%  |
| Macro F1         | 96.88%  |
| Macro Precision  | 97.27%  |
| Macro Recall     | 96.82%  |
| Macro AUC        | 99.79%  |

| Class      | Precision | Recall | Specificity | F1    |
|-----------|-----------|--------|-------------|-------|
| car        | 0.940     | 0.983  | 0.972       | 0.961 |
| bus        | 0.997     | 0.987  | 0.999       | 0.992 |
| truck      | 0.958     | 0.942  | 0.978       | 0.950 |
| background | 0.995     | 0.960  | 0.999       | 0.977 |

### Generated Outputs

```
outputs/
├── figures/
│   ├── confusion_matrix.png
│   ├── per_class_f1.png
│   ├── roc_curves.png
│   └── training_curves.png
├── metrics/
│   ├── test_metrics.json
│   ├── training_history.csv
│   └── train.log
├── models/
│   └── best_model.pth
└── predictions/
    ├── annotated_output.mp4
    ├── class_counts.json
    └── frame_predictions.csv
```

## Non-goals for this phase

Do **not** spend this phase on:

- a complex production backend
- multi-user authentication
- large-scale deployment
- full digital twin integration
- advanced detection + tracking unless the baseline already works

## Phase 3 (Final — 25 May 2026)

Planned improvements for the final presentation:

- Web dashboard (Streamlit / FastAPI)
- Optional tracking (SORT / ByteTrack-lite)
- ROI-based counting per lane
- Real-time video stream support (RTSP)
- Time-series density plots
- Speed/accuracy trade-off analysis
