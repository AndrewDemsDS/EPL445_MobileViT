# Phase 3 Roadmap — Final Presentation

> **Today: Wed 20 May 2026. Deadline: Mon 25 May 2026, 14:00. 5 days left.**

## Hard-cut triage (what makes it / what gets dropped)

| Tier | Item | Why |
|------|------|-----|
| **MUST** | IEEE 4-page paper | 17% of final grade — single biggest item. Cannot skip. |
| **MUST** | Final slide deck (10–15 min) | Required for live presentation. |
| **MUST** | Working demo on May 25 | Already functional; just needs polish. |
| **MUST** | GitHub repo tagged + Moodle upload | Submission requirement. |
| **MUST** | Refresh metrics across docs | Avoids embarrassing stale numbers in paper/slides. |
| **NICE-TO-HAVE** | SORT tracker wired into dashboard | Demo nicer if unique IDs shown, but not required. |
| **NICE-TO-HAVE** | Polygon lane ROI in dashboard | Standalone module already exists, can be demoed via CLI. |
| **DROP IF NO TIME** | RTSP/MJPEG in browser | Stream module works via `cv2.imshow` for demo. Refactor to MJPEG is 4+ hours, skip if Sat night. |
| **DROP IF NO TIME** | Backup recorded demo video | Live demo on stable machine is fine. |

---

## 5-day plan

### **Today — Wed 20 May**
- ✅ Plan written (this file), task list created
- Re-run `bash scripts/run_demo.sh` with new checkpoint → fresh `outputs/predictions/*`
- Read latest `outputs/metrics/test_metrics.json` to lock in final headline numbers
- Wire `SORTTracker` into `predict_video.py` (~2 h work, big demo win)

### **Thu 21 May — Integration day**
- Wire `ROILaneCounter` into FastAPI as additional endpoint (do NOT replace existing rectangle ROI — additive). 2 h.
- Refresh metrics in README, notebook, Greek script. 1 h.
- Update README with one Phase 3 features section (uvicorn run command, screenshots, feature list). 1 h.
- Smoke-test full dashboard end-to-end. 1 h.

### **Fri 22 May — Paper day 1**
- Download IEEE A4 conference paper template. 15 min.
- Abstract (~150 words). 30 min.
- Introduction with 5–6 references. 2 h.
- Methodology (longest section). 3–4 h.

### **Sat 23 May — Paper day 2**
- Results section: tables + figures (re-use existing PNGs). 2 h.
- Discussion + Future Work. 1.5 h.
- Bibliography in IEEE format. 1 h.
- First read-through, fix obvious issues. 1 h.

### **Sun 24 May — Slides + final polish**
- Build slide deck (12–15 slides for 10–15 min talk). 3–4 h.
- Speaker notes per slide. 1 h.
- Practice run once, time it. 30 min.
- Paper revision pass. 1 h.
- Tag `v1.0`, push, prep Moodle upload bundle (paper PDF + slides PPTX + GitHub URL). 30 min.

### **Mon 25 May — Presentation day**
- Morning: one more practice run.
- Final checks: laptop, slides, dashboard running, `sample_traffic.mp4` cached.
- 14:00–17:00: present.

---

## Deliverables checklist for Moodle

- [ ] `paper.pdf` (IEEE format, 4 pages, 2 columns)
- [ ] `presentation.pptx` (final slide deck)
- [ ] GitHub repository URL with `v1.0` tag
- [ ] Demo runnable from a clean clone (uvicorn one-liner)

---

## Key facts already nailed down

| Item | Value |
|------|-------|
| Dataset | UA-DETRAC, 4 classes (car/bus/truck/background), ~475K crops, sequence-disjoint splits |
| Model | MobileViT-S (~5.6M params), timm, ImageNet-pretrained |
| Training | 15 epochs, AdamW lr=3e-4, weight decay=1e-4, CosineAnnealingLR, two-phase freeze/unfreeze, batch=4 |
| Hardware | AMD Radeon 780M (ROCm), HSA_OVERRIDE_GFX_VERSION=11.0.0 |
| Inference | Multi-scale sliding window {120, 180, 256}, NMS IoU 0.3, conf 0.70 |
| Test metrics (Interim #2 checkpoint) | Acc 96.71%, Macro F1 96.88%, Macro AUC 99.79% |
| Retrained val F1 | 0.9807 |

## Bibliography seed

1. Mehta & Rastegari, "MobileViT," ICLR 2022.
2. Wen et al., "UA-DETRAC benchmark," 2020.
3. Bewley et al., "Simple Online and Realtime Tracking (SORT)," ICIP 2016.
4. Dosovitskiy et al., "ViT," ICLR 2021.
5. Vaswani et al., "Attention Is All You Need," NeurIPS 2017.
6. Loshchilov & Hutter, "Decoupled Weight Decay Regularization (AdamW)," ICLR 2019.
7. Wightman, "PyTorch Image Models (timm)," 2019.
8. Redmon et al., "YOLOv3," 2018. (comparison)
