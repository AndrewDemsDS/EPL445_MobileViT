import json, pathlib, urllib.request
from playwright.sync_api import sync_playwright

SHOTS = pathlib.Path("outputs/verify_screenshots")
SHOTS.mkdir(parents=True, exist_ok=True)
BASE = "http://127.0.0.1:8000"

results = []

def shot(page, name):
    p = str(SHOTS / f"{name}.png")
    page.screenshot(path=p, full_page=False)
    return p

with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1280, "height": 900})

    # Step 1: page load & branding
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    title        = page.title()
    wordmark_ok  = page.locator(".wordmark").is_visible()
    upload_ok    = page.locator("#upload-section").is_visible()
    jobs_ok      = page.locator("#jobs-section").is_visible()
    stream_ok    = page.locator("#stream-section").is_visible()
    s1 = shot(page, "01_page_load")
    results.append(("Page load & branding",
        all([wordmark_ok, upload_ok, jobs_ok, stream_ok]),
        f"title={title!r} wordmark={wordmark_ok} upload={upload_ok} jobs={jobs_ok} stream={stream_ok}",
        s1))

    # Step 2: detector dropdown
    opts = page.locator("#detector-select option").all_text_contents()
    yolo_opt    = any("yolo" in o.lower() or "YOLOv8" in o for o in opts)
    sliding_opt = any("sliding" in o.lower() or "Sliding" in o for o in opts)
    results.append(("Detector dropdown options",
        yolo_opt and sliding_opt,
        f"options={opts}",
        None))

    # Step 3: POST /dev/seed
    req = urllib.request.Request(f"{BASE}/dev/seed", method="POST", data=b"")
    with urllib.request.urlopen(req) as resp:
        seed = json.loads(resp.read())
    job_id  = seed["job_id"]
    seed_ok = seed["status"] == "done" and seed["progress"] == 100
    results.append(("POST /dev/seed",
        seed_ok,
        f"job_id={job_id} status={seed['status']} progress={seed['progress']}",
        None))

    # Step 4: jobs table
    page.reload()
    page.wait_for_load_state("networkidle")
    page.locator("#refresh-jobs-btn").click()
    page.wait_for_timeout(1000)
    row = page.locator("#jobs-body tr").filter(has_text=job_id)
    row_visible = row.is_visible()
    badge_done  = row.locator(".badge-done").is_visible() if row_visible else False
    s4 = shot(page, "02_jobs_table")
    results.append(("Jobs table – seeded job row visible",
        row_visible and badge_done,
        f"row_visible={row_visible} badge_done={badge_done}",
        s4))

    # Step 5: click View -> results
    if row_visible:
        row.locator("button", has_text="View").click()
        page.wait_for_timeout(3000)
    results_visible = page.locator("#results-section").is_visible()
    pills_html      = page.locator("#summary-pills").inner_html()
    has_total       = "Total detections" in pills_html
    has_unique      = "Unique vehicles"  in pills_html
    has_counts_cvs  = page.locator("#counts-chart").is_visible()
    has_timeline    = page.locator("#timeline-chart").is_visible()
    has_video       = page.locator("#result-video").is_visible()
    s5 = shot(page, "03_results_section")
    results.append(("Results section – pills + charts",
        all([results_visible, has_total, has_unique, has_counts_cvs, has_timeline, has_video]),
        f"visible={results_visible} total={has_total} unique={has_unique} counts={has_counts_cvs} timeline={has_timeline} video={has_video}",
        s5))

    # Step 6: ROI and lanes sections
    roi_visible   = page.locator("#roi-section").is_visible()
    lanes_visible = page.locator("#lanes-section").is_visible()
    s6 = shot(page, "04_roi_lanes")
    results.append(("ROI & lane counting sections visible",
        roi_visible and lanes_visible,
        f"roi={roi_visible} lanes={lanes_visible}",
        s6))

    # Probe A: counts API
    with urllib.request.urlopen(f"{BASE}/jobs/{job_id}/counts") as r:
        counts = json.loads(r.read())
    counts_ok = "class_counts" in counts and counts.get("total_detections", 0) > 0
    results.append(("Probe A: /jobs/{id}/counts JSON",
        counts_ok,
        f"keys={list(counts.keys())} total={counts.get('total_detections')} unique_total={counts.get('unique_vehicles_total')}",
        None))

    # Probe B: timeline API
    with urllib.request.urlopen(f"{BASE}/jobs/{job_id}/timeline") as r:
        timeline = json.loads(r.read())
    tl_ok = isinstance(timeline, list) and len(timeline) > 0
    results.append(("Probe B: /jobs/{id}/timeline JSON",
        tl_ok,
        f"entries={len(timeline)} sample={timeline[:2]}",
        None))

    # Probe C: run button disabled without file
    run_disabled = page.locator("#run-btn").get_attribute("disabled") is not None
    results.append(("Probe C: run button disabled without file",
        run_disabled,
        f"disabled={run_disabled}",
        None))

    # Probe D: lanes polygon API
    payload = json.dumps({
        "left":  [[0,0],[400,0],[400,600],[0,600]],
        "right": [[400,0],[800,0],[800,600],[400,600]],
    }).encode()
    req2 = urllib.request.Request(
        f"{BASE}/jobs/{job_id}/lanes",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req2) as r:
        lanes_data = json.loads(r.read())
    lanes_ok = "per_lane" in lanes_data and set(lanes_data["per_lane"]) == {"left", "right"}
    results.append(("Probe D: /jobs/{id}/lanes polygon counts",
        lanes_ok,
        f"per_lane={lanes_data.get('per_lane')} totals={lanes_data.get('totals')}",
        None))

    # Probe E: invalid job id returns 404
    try:
        urllib.request.urlopen(f"{BASE}/jobs/deadbeef/counts")
        e404_ok = False
    except urllib.error.HTTPError as e:
        e404_ok = e.code == 404
    results.append(("Probe E: unknown job_id returns 404",
        e404_ok,
        f"got 404={e404_ok}",
        None))

    browser.close()

print("\n=== VERIFY RESULTS ===")
all_pass = True
for name, ok, detail, screenshot in results:
    icon = "PASS" if ok else "FAIL"
    if not ok:
        all_pass = False
    print(f"[{icon}] {name}")
    print(f"       {detail}")
    if screenshot:
        print(f"       screenshot: {screenshot}")

print()
print("OVERALL:", "PASS" if all_pass else "FAIL")
