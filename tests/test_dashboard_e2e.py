"""End-to-end Playwright tests for the MobileViT Traffic Dashboard.

Run with:
    pytest tests/test_dashboard_e2e.py -v --headed   # open browser window
    pytest tests/test_dashboard_e2e.py -v            # headless (CI)

The session-scoped `dashboard_url` fixture (conftest.py) starts a live
uvicorn server on port 8001 so tests hit the real FastAPI app.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.parse

import pytest
from playwright.sync_api import Page, expect


# ── API smoke tests (no browser) ────────────────────────────────────────────

def test_api_root_returns_html(dashboard_url):
    """GET / should serve the index.html page."""
    with urllib.request.urlopen(f"{dashboard_url}/") as resp:
        assert resp.status == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/html" in content_type


def test_api_jobs_list(dashboard_url):
    """GET /jobs should return a JSON list."""
    with urllib.request.urlopen(f"{dashboard_url}/jobs") as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
        assert isinstance(data, list)


def test_api_seed_creates_done_job(dashboard_url):
    """POST /dev/seed should return a DONE job with the expected fields."""
    req = urllib.request.Request(
        f"{dashboard_url}/dev/seed", method="POST", data=b""
    )
    with urllib.request.urlopen(req) as resp:
        assert resp.status == 200
        job = json.loads(resp.read())
    assert job["status"] == "done"
    assert job["progress"] == 100
    assert "job_id" in job


def test_api_counts_after_seed(dashboard_url):
    """After seeding, /jobs/{id}/counts should return class_counts dict."""
    req = urllib.request.Request(
        f"{dashboard_url}/dev/seed", method="POST", data=b""
    )
    with urllib.request.urlopen(req) as resp:
        job = json.loads(resp.read())
    job_id = job["job_id"]

    with urllib.request.urlopen(f"{dashboard_url}/jobs/{job_id}/counts") as resp:
        assert resp.status == 200
        counts = json.loads(resp.read())
    assert "class_counts" in counts
    assert counts["total_detections"] > 0
    assert "unique_vehicles_total" in counts


def test_api_timeline_after_seed(dashboard_url):
    """After seeding, /jobs/{id}/timeline should return a list of frame dicts."""
    req = urllib.request.Request(
        f"{dashboard_url}/dev/seed", method="POST", data=b""
    )
    with urllib.request.urlopen(req) as resp:
        job = json.loads(resp.read())
    job_id = job["job_id"]

    with urllib.request.urlopen(f"{dashboard_url}/jobs/{job_id}/timeline") as resp:
        assert resp.status == 200
        timeline = json.loads(resp.read())
    assert isinstance(timeline, list)
    assert len(timeline) > 0
    assert "frame" in timeline[0] and "count" in timeline[0]


def test_api_lanes_after_seed(dashboard_url):
    """POST /jobs/{id}/lanes with a valid polygon should return per-lane counts."""
    req = urllib.request.Request(
        f"{dashboard_url}/dev/seed", method="POST", data=b""
    )
    with urllib.request.urlopen(req) as resp:
        job = json.loads(resp.read())
    job_id = job["job_id"]

    payload = json.dumps({
        "left": [[0, 0], [400, 0], [400, 600], [0, 600]],
        "right": [[400, 0], [800, 0], [800, 600], [400, 600]],
    }).encode()
    req2 = urllib.request.Request(
        f"{dashboard_url}/jobs/{job_id}/lanes",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req2) as resp:
        assert resp.status == 200
        data = json.loads(resp.read())
    assert "per_lane" in data
    assert "left" in data["per_lane"]
    assert "right" in data["per_lane"]
    assert "totals" in data


# ── Browser (Playwright) tests ───────────────────────────────────────────────

def test_page_title(dashboard_url, page: Page):
    page.goto(dashboard_url)
    expect(page).to_have_title("MobileViT Traffic Dashboard — EPL445")


def test_header_branding(dashboard_url, page: Page):
    page.goto(dashboard_url)
    header = page.locator("header")
    expect(header).to_be_visible()
    wordmark = page.locator(".wordmark")
    expect(wordmark).to_be_visible()
    expect(wordmark).to_contain_text("MobileViT")


def test_upload_section_present(dashboard_url, page: Page):
    page.goto(dashboard_url)
    expect(page.locator("#upload-section")).to_be_visible()
    expect(page.locator("#drop-zone")).to_be_visible()
    expect(page.locator("#detector-select")).to_be_visible()
    # Run button starts disabled (no file selected)
    run_btn = page.locator("#run-btn")
    expect(run_btn).to_be_visible()
    assert run_btn.get_attribute("disabled") is not None


def test_stream_section_present(dashboard_url, page: Page):
    page.goto(dashboard_url)
    expect(page.locator("#stream-section")).to_be_visible()
    expect(page.locator("#stream-source-input")).to_be_visible()
    expect(page.locator("#stream-start-btn")).to_be_visible()
    expect(page.locator("#stream-stop-btn")).to_be_visible()


def test_past_jobs_section_present(dashboard_url, page: Page):
    page.goto(dashboard_url)
    expect(page.locator("#jobs-section")).to_be_visible()
    expect(page.locator("#refresh-jobs-btn")).to_be_visible()
    expect(page.locator("#jobs-table")).to_be_visible()


def test_seed_job_appears_in_table(dashboard_url, page: Page):
    """Seed a job, refresh the Past Jobs table, verify the row appears."""
    # Seed via API first
    req = urllib.request.Request(
        f"{dashboard_url}/dev/seed", method="POST", data=b""
    )
    with urllib.request.urlopen(req) as resp:
        job = json.loads(resp.read())
    job_id = job["job_id"]

    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Click ↻ Refresh to force a table reload
    page.locator("#refresh-jobs-btn").click()
    page.wait_for_timeout(800)

    # The job row should contain the job_id prefix and a "done" badge
    row = page.locator("#jobs-body tr").filter(has_text=job_id)
    expect(row).to_be_visible(timeout=5000)
    expect(row.locator(".badge-done")).to_be_visible()


def test_view_results_renders_charts(dashboard_url, page: Page):
    """Seed a job, click View, and verify the results panel with charts renders."""
    req = urllib.request.Request(
        f"{dashboard_url}/dev/seed", method="POST", data=b""
    )
    with urllib.request.urlopen(req) as resp:
        job = json.loads(resp.read())
    job_id = job["job_id"]

    page.goto(dashboard_url)
    page.wait_for_load_state("networkidle")

    # Refresh and click View
    page.locator("#refresh-jobs-btn").click()
    page.wait_for_timeout(800)

    row = page.locator("#jobs-body tr").filter(has_text=job_id)
    expect(row).to_be_visible(timeout=5000)
    row.locator("button", has_text="View").click()

    # Results section should become visible
    expect(page.locator("#results-section")).not_to_have_class(
        "hidden", timeout=8000
    )

    # Summary pills (total detections, unique vehicles)
    pills = page.locator("#summary-pills")
    expect(pills).to_be_visible()
    expect(pills).to_contain_text("Total detections")
    expect(pills).to_contain_text("Unique vehicles")

    # Bar chart and density timeline canvases rendered
    expect(page.locator("#counts-chart")).to_be_visible()
    expect(page.locator("#timeline-chart")).to_be_visible()

    # Video player present (may have no src if output.mp4 missing, that's fine)
    expect(page.locator("#result-video")).to_be_visible()


def test_detector_select_options(dashboard_url, page: Page):
    """The detector dropdown must have the YOLO and sliding options."""
    page.goto(dashboard_url)
    select = page.locator("#detector-select")
    expect(select).to_be_visible()
    options = select.locator("option").all_text_contents()
    assert any("yolo" in o.lower() or "YOLOv8" in o for o in options)
    assert any("sliding" in o.lower() or "Sliding" in o for o in options)
