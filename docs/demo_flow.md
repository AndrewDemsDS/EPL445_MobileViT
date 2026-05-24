# Final Demo Flow — Mon 25 May 2026, 14:00

A 5-minute live demo of the MobileViT traffic dashboard, scripted for the EPL445 final presentation. The aim is to show, in order:

1. The classifier runs end to end on a real video.
2. The pipeline produces per-class, per-frame, and per-lane analytics.
3. SORT turns raw detections into unique vehicle counts.
4. The dashboard is a single uvicorn command away.

## Before the slot

1. Power on the laptop, plug in the projector adapter.
2. Open three terminals:
   - **T1**: the project directory `/home/andrewdems/Documents/EPL445_MobileViT`, venv active.
   - **T2**: same directory, kept free for `curl` against the dashboard.
   - **T3**: kept free for `ffprobe` / `cat outputs/...` to show numbers if asked.
3. Open Firefox to `about:blank`.
4. Confirm `data/raw/sample_traffic.mp4` exists. Confirm `outputs/models/best_model.pth` exists.
5. Pre-warm caches: `python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"` — pulls weights to local cache if not already there.

## Demo script

### Step 1 — Start the dashboard (30 s)

T1:

```bash
uvicorn src.app.main:app --host 127.0.0.1 --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000`. Switch to Firefox, navigate to that URL.

**Say**: *"The dashboard is a FastAPI app with a single static HTML/JS frontend. No build step, no React, no extra services. One command starts it."*

### Step 2 — Seed a finished job (20 s)

T2:

```bash
curl -s -X POST http://localhost:8000/dev/seed | jq .job_id
```

Back in Firefox, click **Refresh** on the Past Jobs table, then **View** on the new row.

**Say**: *"This skips the 3-minute inference for the live demo — the underlying pipeline ran offline this morning and produced these outputs. Identical results to running it now."*

### Step 3 — Annotated video and per-class chart (40 s)

Show the annotated MP4 (point out the bounding boxes have track IDs `#1 car 0.96`).
Hover over the bar chart — explain the colour coding (car green, bus orange, truck blue).

**Say**: *"YOLOv8-nano proposes the boxes. MobileViT re-classifies each crop with our four-class taxonomy. The track ID is the same vehicle across frames — that's SORT."*

### Step 4 — Density timeline (20 s)

Scroll to the Vehicle Density chart. Point to two peaks.

**Say**: *"Each peak is a vehicle platoon passing through the intersection. The dashboard reads this from the per-frame CSV — no re-inference."*

### Step 5 — Polygon lane counting (60 s)

Scroll to **Per-lane Counting**. Click four corners around the left half of the frame, type `left`, click **Finish lane**. Repeat for the right half, name it `right`. Click **Apply lanes**.

Read the result aloud: left ~4 900 detections, mostly car; right ~3 500 detections, also car-dominant. Mention the unique-vehicle counts in parentheses.

**Say**: *"This is the per-lane breakdown a digital twin would consume. Operator-drawn polygons, no retraining, instant."*

### Step 6 — Optional fallback if Wi-Fi/projector hiccups (30 s)

If the video player stalls or the projector flickers, switch to the prepared screenshot:

```bash
xdg-open docs/screenshots/dashboard_full.png
```

**Say**: *"Same dashboard, captured 30 minutes ago — the numbers we see here are what the live one would show."*

### Step 7 — Wrap (10 s)

Switch back to the slide deck (Slide 18, Q&A).

**Say**: *"The MobileViT classifier is 5.6 million parameters. The full inference pipeline runs at ~10 fps on integrated graphics. Source is on GitHub at the v1.0 tag. Happy to take questions."*

## Failure recovery

| Symptom | Fix |
|---|---|
| `uvicorn` fails to bind port 8000 | `lsof -i :8000` to find the holder, `pkill -f 'uvicorn'`, retry. |
| `/jobs/{id}/video` returns 500 | Inspect `outputs/jobs/{id}/output.mp4` — if missing, the seed didn't run; rerun `curl -X POST .../dev/seed`. |
| Video plays but is silent grey | The mp4v output skipped re-encoding. Trigger: `curl http://localhost:8000/jobs/{id}/video > /dev/null` and wait 30 s; the re-encoded `web_*.mp4` will land. |
| Lane apply returns 400 | At least one polygon has fewer than three vertices. Click three more times before naming the lane. |
| Past Jobs table is empty | The in-memory store cleared on uvicorn restart. Re-seed. |

## Hard pre-flight checklist

- [ ] `uvicorn src.app.main:app --host 127.0.0.1 --port 8000` starts cleanly
- [ ] `curl -X POST http://localhost:8000/dev/seed` returns a job id
- [ ] `curl http://localhost:8000/jobs/<id>/counts` returns the expected JSON with `unique_vehicles_total`
- [ ] `curl -I http://localhost:8000/jobs/<id>/video` returns 200 within 30 s
- [ ] Firefox loads the dashboard, the video plays, the bar chart and density timeline render
- [ ] Polygon ROI accepts at least three vertices and returns per-lane counts

If every box above is ticked an hour before the slot, the demo is safe.

## What I will NOT do live

- Run a fresh `predict_video` inference. Three minutes is too much demo dead-time and ROCm has been flaky.
- Demo the RTSP/webcam stream live. The standalone `src/inference/stream.py` CLI uses `cv2.imshow` which fights the projector. Mention it as a feature, point to the file path.
- Edit code on stage.
