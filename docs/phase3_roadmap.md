# Phase 3 Roadmap

Today: Wed 20 May 2026. Deadline: Mon 25 May 2026, 14:00. Five days.

## Triage

| Tier | Item | Notes |
|------|------|-------|
| **Must** | IEEE 4-page paper | 17% of the grade. Single biggest item. |
| **Must** | Final slide deck (10–15 min) | Required for the live presentation. |
| **Must** | Working demo on May 25 | The dashboard runs; polish the flow. |
| **Must** | GitHub repo tagged + Moodle upload | Submission requirement. |
| **Must** | Refresh metrics across docs | Stale numbers in the paper or slides will cost marks. |
| **Nice** | SORT tracker in the dashboard | Done as of today. Persistent IDs look good in the demo. |
| **Nice** | Polygon lane ROI in the dashboard | Module exists. Wiring takes about two hours. |
| **Drop** | RTSP/MJPEG in the browser | The standalone `cv2.imshow` version is enough. A web port costs four hours. |
| **Drop** | Recorded backup demo | The lab machine is stable. Live demo only. |

## Schedule

### Wed 20 May (today)
- Plan written.
- Run `bash scripts/run_demo.sh` with the new checkpoint and refresh `outputs/predictions/`.
- Lock the headline numbers from `outputs/metrics/test_metrics.json`.
- Wire `SORTTracker` into `predict_video.py`. ✓
- Show unique vehicle counts in the dashboard. ✓

### Thu 21 May — Integration
- Wire `ROILaneCounter` into FastAPI behind a new endpoint. Two hours.
- Refresh metrics in `README.md`, the demo notebook, and the Greek script. One hour.
- Add a Phase 3 section to the README. One hour.
- Smoke-test the dashboard end to end with Playwright. One hour.

### Fri 22 May — Paper day 1
- Pull the IEEE A4 conference template. 15 min.
- Abstract, ~150 words. 30 min.
- Introduction with five or six references. Two hours.
- Methodology section. Three to four hours.

### Sat 23 May — Paper day 2
- Results section: tables and existing PNGs. Two hours.
- Discussion and future work. 90 min.
- Bibliography in IEEE format. One hour.
- Read-through and fixes. One hour.

### Sun 24 May — Slides and polish
- Slide deck, 12–15 slides for a 10–15 min talk. Three to four hours.
- Speaker notes. One hour.
- One practice run with a timer. 30 min.
- Paper revision pass. One hour.
- Tag `v1.0`, push, bundle the Moodle upload. 30 min.

### Mon 25 May — Presentation day
- One more practice run in the morning.
- Verify laptop, slides, dashboard, cached `sample_traffic.mp4`.
- 14:00–17:00: present.

## Moodle deliverables

- [ ] `paper.pdf` (IEEE format, 4 pages, two columns)
- [ ] `presentation.pptx`
- [ ] GitHub URL with the `v1.0` tag
- [ ] Demo runs from a clean clone via one `uvicorn` command

## Fixed facts

| Item | Value |
|------|-------|
| Dataset | UA-DETRAC, four classes (car/bus/truck/background), ~475K crops, sequence-disjoint splits |
| Model | MobileViT-S (~5.6M params), timm, ImageNet-pretrained |
| Training | 15 epochs, AdamW lr=3e-4, weight decay=1e-4, CosineAnnealingLR, two-phase freeze/unfreeze, batch=4 |
| Hardware | AMD Radeon 780M (ROCm), HSA_OVERRIDE_GFX_VERSION=11.0.0 |
| Inference | Multi-scale sliding window {120, 180, 256}, NMS IoU 0.3, conf 0.70 |
| Test metrics (sequence-disjoint, current) | Acc 96.44%, Macro F1 95.90%, Macro AUC 99.59% |
| Best validation Macro F1 (retrain) | 0.9807 |

## Bibliography seed

1. Mehta & Rastegari, "MobileViT," ICLR 2022.
2. Wen et al., "UA-DETRAC benchmark," 2020.
3. Bewley et al., "Simple Online and Realtime Tracking (SORT)," ICIP 2016.
4. Dosovitskiy et al., "ViT," ICLR 2021.
5. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
6. Loshchilov & Hutter, "Decoupled Weight Decay Regularization (AdamW)," ICLR 2019.
7. Wightman, "PyTorch Image Models (timm)," 2019.
8. Redmon et al., "YOLOv3," 2018.
