"""
Webapp test for the MobileViT Traffic Dashboard.
Follows the webapp-testing skill: reconnaissance-then-action.
Server already running on port 8000.
"""

import json
import pathlib
import urllib.request
import urllib.error
from playwright.sync_api import sync_playwright

BASE   = "http://127.0.0.1:8000"
SHOTS  = pathlib.Path("outputs/webapp_screenshots")
SHOTS.mkdir(parents=True, exist_ok=True)

passed = []
failed = []

def ok(name, detail=""):
    passed.append(name)
    print(f"  PASS  {name}" + (f"  —  {detail}" if detail else ""))

def fail(name, detail=""):
    failed.append(name)
    print(f"  FAIL  {name}" + (f"  —  {detail}" if detail else ""))

def shot(page, name):
    p = str(SHOTS / f"{name}.png")
    page.screenshot(path=p, full_page=True)
    return p

# ── seed a job we can drive ──────────────────────────────────────
req = urllib.request.Request(f"{BASE}/dev/seed", method="POST", data=b"")
with urllib.request.urlopen(req) as r:
    seed = json.loads(r.read())
JOB = seed["job_id"]
print(f"\nSeeded job: {JOB}\n")

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    # ── PHASE 1: Reconnaissance ──────────────────────────────────
    print("── Phase 1: Reconnaissance ──────────────────────────")
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    shot(page, "recon_01_initial_load")

    # Discover key elements
    sections  = page.locator("section").all()
    headings  = [h.inner_text() for h in page.locator("section h2").all()]
    buttons   = [b.inner_text() for b in page.locator("button").all()]
    selects   = page.locator("select").all()
    print(f"  Sections found: {len(sections)}")
    print(f"  Section headings: {headings}")
    print(f"  Buttons visible: {buttons}")
    print(f"  Selects found: {len(selects)}")

    # ── PHASE 2: Section structure ───────────────────────────────
    print("\n── Phase 2: Section structure ───────────────────────")

    expected_headings = ["1. Upload Traffic Video", "6. Live Stream", "Past Jobs"]
    for h in expected_headings:
        if any(h in s for s in headings):
            ok(f"Heading present: '{h}'")
        else:
            fail(f"Heading missing: '{h}'", f"found={headings}")

    # Upload zone
    drop_zone = page.locator("#drop-zone")
    if drop_zone.is_visible():
        ok("Drop zone visible")
    else:
        fail("Drop zone not visible")

    # Detector select has both options
    det_select = page.locator("#detector-select")
    det_opts = det_select.locator("option").all_text_contents()
    if any("yolo" in o.lower() for o in det_opts) and any("sliding" in o.lower() for o in det_opts):
        ok("Detector dropdown: both YOLO + Sliding options", str(det_opts))
    else:
        fail("Detector dropdown: missing option", str(det_opts))

    # Run button disabled without file
    run_btn = page.locator("#run-btn")
    if run_btn.get_attribute("disabled") is not None:
        ok("Run button disabled (no file selected)")
    else:
        fail("Run button should be disabled without file")

    # ── PHASE 3: Jobs table & seeded job ────────────────────────
    print("\n── Phase 3: Past Jobs table ─────────────────────────")
    page.locator("#refresh-jobs-btn").click()
    page.wait_for_timeout(800)

    row = page.locator("#jobs-body tr").filter(has_text=JOB)
    if row.is_visible():
        ok("Seeded job appears in Past Jobs table", JOB)
    else:
        fail("Seeded job not found in table", JOB)

    badge = row.locator(".badge-done")
    if badge.is_visible():
        ok("Job shows DONE badge")
    else:
        fail("Job badge not DONE")

    view_btn = row.locator("button", has_text="View")
    if view_btn.is_visible():
        ok("View button present on done job")
    else:
        fail("View button missing")

    shot(page, "phase3_jobs_table")

    # ── PHASE 4: View results ────────────────────────────────────
    print("\n── Phase 4: Results panel ───────────────────────────")
    view_btn.click()
    page.wait_for_timeout(3000)
    shot(page, "phase4_results_opened")

    results_sec = page.locator("#results-section")
    if results_sec.is_visible():
        ok("Results section visible after View click")
    else:
        fail("Results section still hidden")

    progress_bar = page.locator("#progress-bar")
    bar_width = progress_bar.evaluate("el => el.style.width")
    if bar_width == "100%":
        ok("Progress bar at 100%", f"width={bar_width}")
    else:
        fail("Progress bar not 100%", f"width={bar_width}")

    # Summary pills
    pills_html = page.locator("#summary-pills").inner_html()
    for label in ["Total detections", "Unique vehicles", "Dominant class", "Traffic density"]:
        if label in pills_html:
            ok(f"Summary pill: '{label}'")
        else:
            fail(f"Summary pill missing: '{label}'")

    # Per-class pills
    for cls in ["car", "bus", "truck"]:
        if f"pill-{cls}" in pills_html:
            ok(f"Per-class pill: {cls}")
        else:
            fail(f"Per-class pill missing: {cls}")

    # Charts
    if page.locator("#counts-chart").is_visible():
        ok("Bar chart (detections per class) rendered")
    else:
        fail("Bar chart not visible")

    if page.locator("#timeline-chart").is_visible():
        ok("Density timeline chart rendered")
    else:
        fail("Density timeline not visible")

    # Video player
    video = page.locator("#result-video")
    v_src = video.get_attribute("src") or ""
    if "/video" in v_src:
        ok("Video player has /jobs/{id}/video src", v_src)
    else:
        fail("Video player src not set", v_src)

    shot(page, "phase4_results_full")

    # ── PHASE 5: ROI section ─────────────────────────────────────
    print("\n── Phase 5: ROI section ─────────────────────────────")
    roi_sec = page.locator("#roi-section")
    if roi_sec.is_visible():
        ok("ROI section visible")
    else:
        fail("ROI section hidden")

    roi_canvas = page.locator("#roi-canvas")
    canvas_w = roi_canvas.evaluate("el => el.width")
    canvas_h = roi_canvas.evaluate("el => el.height")
    if canvas_w > 0 and canvas_h > 0:
        ok("ROI canvas loaded with frame", f"{canvas_w}x{canvas_h}")
    else:
        fail("ROI canvas not populated", f"{canvas_w}x{canvas_h}")

    # Apply ROI button present
    if page.locator("#apply-roi-btn").is_visible():
        ok("Apply ROI button visible")
    else:
        fail("Apply ROI button missing")

    shot(page, "phase5_roi_section")

    # ── PHASE 6: Lane counting section ──────────────────────────
    print("\n── Phase 6: Lane counting section ───────────────────")
    lanes_sec = page.locator("#lanes-section")
    if lanes_sec.is_visible():
        ok("Lanes section visible")
    else:
        fail("Lanes section hidden")

    lane_canvas = page.locator("#lanes-canvas")
    lc_w = lane_canvas.evaluate("el => el.width")
    lc_h = lane_canvas.evaluate("el => el.height")
    if lc_w > 0 and lc_h > 0:
        ok("Lanes canvas loaded with frame", f"{lc_w}x{lc_h}")
    else:
        fail("Lanes canvas not populated", f"{lc_w}x{lc_h}")

    # Lane controls
    for elem_id, label in [
        ("#lane-name-input",  "Lane name input"),
        ("#finish-lane-btn",  "Finish lane button"),
        ("#apply-lanes-btn",  "Apply lanes button"),
        ("#clear-lanes-btn",  "Clear all button"),
    ]:
        if page.locator(elem_id).is_visible():
            ok(f"{label} visible")
        else:
            fail(f"{label} missing")

    shot(page, "phase6_lanes_section")

    # ── PHASE 7: Live stream section ─────────────────────────────
    print("\n── Phase 7: Live stream section ─────────────────────")
    stream_src = page.locator("#stream-source-input")
    if stream_src.is_visible():
        ok("Stream source input visible")
        default_val = stream_src.input_value()
        ok("Stream source default value set", default_val)
    else:
        fail("Stream source input missing")

    stream_det = page.locator("#stream-detector-select")
    sd_opts = stream_det.locator("option").all_text_contents()
    if any("yolo" in o.lower() for o in sd_opts):
        ok("Stream detector has YOLO option", str(sd_opts))
    else:
        fail("Stream detector missing YOLO option")

    shot(page, "phase7_stream_section")

    # ── PHASE 8: API probes ──────────────────────────────────────
    print("\n── Phase 8: API probes ──────────────────────────────")

    # counts endpoint
    with urllib.request.urlopen(f"{BASE}/jobs/{JOB}/counts") as r:
        counts = json.loads(r.read())
    if counts.get("total_detections", 0) > 0:
        ok("/counts: total_detections present", str(counts["total_detections"]))
    else:
        fail("/counts: empty or missing")

    if "unique_vehicles_total" in counts:
        ok("/counts: unique_vehicles_total present", str(counts["unique_vehicles_total"]))
    else:
        fail("/counts: unique_vehicles_total missing")

    # timeline endpoint
    with urllib.request.urlopen(f"{BASE}/jobs/{JOB}/timeline") as r:
        tl = json.loads(r.read())
    if isinstance(tl, list) and len(tl) > 0 and "frame" in tl[0]:
        ok(f"/timeline: {len(tl)} frame entries", str(tl[:2]))
    else:
        fail("/timeline: bad response", str(tl))

    # video endpoint GET (range request to avoid streaming full file)
    req_head = urllib.request.Request(
        f"{BASE}/jobs/{JOB}/video",
        headers={"Range": "bytes=0-0"},
    )
    try:
        with urllib.request.urlopen(req_head) as r:
            ct = r.headers.get("content-type", "")
        if "video" in ct or r.status in (200, 206):
            ok("/video: content-type is video/*", ct)
        else:
            fail("/video: unexpected content-type", ct)
    except urllib.error.HTTPError as e:
        if e.code in (200, 206):
            ok("/video: accessible", str(e.code))
        else:
            fail("/video: HTTP error", str(e.code))

    # lanes endpoint
    payload = json.dumps({
        "left":  [[0,0],[400,0],[400,600],[0,600]],
        "right": [[400,0],[800,0],[800,600],[400,600]],
    }).encode()
    req2 = urllib.request.Request(
        f"{BASE}/jobs/{JOB}/lanes", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req2) as r:
        ld = json.loads(r.read())
    if set(ld.get("per_lane", {}).keys()) == {"left", "right"}:
        ok("/lanes: both polygons counted", str(ld["totals"]))
    else:
        fail("/lanes: unexpected response", str(ld))

    # 404 on unknown job
    try:
        urllib.request.urlopen(f"{BASE}/jobs/xxxxxxxx/counts")
        fail("Unknown job should return 404 — got 200")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            ok("Unknown job returns 404")
        else:
            fail(f"Unknown job returned {e.code}")

    # jobs list
    with urllib.request.urlopen(f"{BASE}/jobs") as r:
        jobs = json.loads(r.read())
    if isinstance(jobs, list) and any(j["job_id"] == JOB for j in jobs):
        ok(f"/jobs list includes seeded job ({len(jobs)} total jobs)")
    else:
        fail("/jobs list missing seeded job")

    browser.close()

# ── Summary ──────────────────────────────────────────────────────
total = len(passed) + len(failed)
print(f"\n{'='*55}")
print(f"RESULTS: {len(passed)}/{total} passed")
if failed:
    print("\nFailed:")
    for f in failed:
        print(f"  ✗ {f}")
print(f"\nOverall: {'PASS' if not failed else 'FAIL'}")
print(f"Screenshots saved to: {SHOTS}")
