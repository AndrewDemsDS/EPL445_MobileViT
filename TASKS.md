# TASKS.md — EPL445 Final Status

## Goal

End-to-end traffic video analysis for the EPL445 final (25 May 2026): a MobileViT classifier paired with a YOLOv8-nano detector and a SORT tracker, served behind a FastAPI dashboard. The brief required a web-based application that handles video, performance metrics on a held-out test set, an IEEE-format four-page paper, and a 10–15 min live demo.

## Phase 2 (Interim Presentation, 23 Apr 2026) — done

1. Repo layout, configs, scripts, tests.
2. UA-DETRAC dataset prep (~475 K crops, 4 classes).
3. MobileViT-S wrapper from `timm`, freeze/unfreeze for two-phase fine-tuning.
4. 15-epoch training with AdamW + CosineAnnealingLR, best val macro F1 = 0.9807.
5. Sequence-disjoint evaluation on the test set — Acc 96.44%, Macro F1 95.90%, Macro AUC 99.59%.
6. All required plots (confusion matrix, ROC, per-class F1, training curves).
7. Sliding-window video inference + per-frame CSV + class counts.
8. 10 unit tests (`tests/`) passing.

## Phase 3 (Final Presentation, 25 May 2026) — done

1. **YOLOv8-nano + MobileViT hybrid** detector replaces the sliding window as the default in `configs/demo.yaml`; sliding stays available as a fallback (`detector: sliding`).
2. **SORT tracker** assigns persistent IDs across frames. `aggregate_counts` now reports `unique_vehicles_by_class` alongside the raw per-frame detection counts.
3. **FastAPI dashboard** (`src/app/`) with drag-and-drop upload, background inference job queue, density timeline (Chart.js), rectangle ROI filtering, and a polygon-based per-lane counter backed by `cv2.pointPolygonTest`.
4. **Browser-friendly playback** — OpenCV mp4v output is lazily re-encoded to H.264 (yuv420p, +faststart) the first time `/jobs/{id}/video` is hit; the worker pre-warms it after each job.
5. **Live stream** — `/stream/feed?source=...` serves MJPEG from a webcam (`source=0`), an RTSP URL, or a file path.
6. **Detector toggle in the upload form** sends `detector=yolo|sliding` as a form field; the worker plumbs it into the per-job config.
7. **Density plots** (`src/evaluation/density_plot.py`) regenerated from the YOLO-hybrid CSV.
8. **Speed/accuracy benchmark** at three input resolutions (`scripts/benchmark.py`).
9. **Side-by-side detector comparison figure** (Figure 6 in the paper).
10. **IEEE four-page paper** (`docs/paper/EPL445_Final_Paper.docx` + `.pdf`).
11. **17-slide presentation deck** (`EPL445_Final_MobileViT_Presentation.pptx`).
12. **Live demo script** for May 25 (`docs/demo_flow.md`).

### Results Summary (Test Set)

| Metric           | Value   |
|------------------|---------|
| Accuracy         | 96.44%  |
| Macro F1         | 95.90%  |
| Macro Precision  | 95.69%  |
| Macro Recall     | 96.15%  |
| Macro AUC        | 99.59%  |

| Class      | Precision | Recall | Specificity | F1    |
|------------|-----------|--------|-------------|-------|
| car        | 0.960     | 0.929  | 0.992       | 0.944 |
| bus        | 0.992     | 0.974  | 0.994       | 0.983 |
| truck      | 0.937     | 0.963  | 0.977       | 0.950 |
| background | 0.939     | 0.980  | 0.990       | 0.959 |

### Pipeline numbers (sample_traffic.mp4, 1 800 frames, frame_skip=3)

- 8 979 raw detections; 599 unique tracked vehicles (411 cars, 115 buses, 73 trucks).
- Left lane (operator-drawn polygon): 4 927 detections, 74% car / 21% bus / 5% truck.
- Right lane: 3 502 detections, 87% car / 5% bus / 8% truck.
- Throughput ~10 fps end-to-end with YOLO; ~3 fps with the sliding-window fallback.

### Repo layout (post-Phase 3)

```
outputs/
├── figures/         confusion_matrix, roc_curves, per_class_f1,
│                    training_curves, density_plot, density_plot_total,
│                    detector_comparison
├── metrics/         test_metrics.json, training_history.csv, train.log
├── models/          best_model.pth
└── predictions/     annotated_output.mp4 (YOLO), web_annotated_output.mp4 (H.264),
                     frame_predictions.csv, class_counts.json

docs/
├── paper/           EPL445_Final_Paper.{md,docx,pdf} + build_docx.js
├── screenshots/     dashboard_full.png, dashboard_with_lanes.png
├── demo_flow.md     live-demo script for 25 May
└── phase3_roadmap.md

src/app/             FastAPI dashboard (main, jobs, analytics, inference_worker, rtsp, static/)
src/inference/       predict_video, tracker (SORT), roi, stream, aggregate_counts, predict_image
EPL445_Final_MobileViT_Presentation.pptx
```

## Moodle submission bundle

- `docs/paper/EPL445_Final_Paper.pdf`
- `EPL445_Final_MobileViT_Presentation.pptx`
- GitHub URL with tagged release.
