/* MobileViT Traffic Dashboard — frontend logic */

const CLASS_COLORS = {
  car:   'rgba(34, 197, 94, 0.8)',
  bus:   'rgba(249, 115, 22, 0.8)',
  truck: 'rgba(59, 130, 246, 0.8)',
};

// ── State ────────────────────────────────────────────────────────
let currentJobId = null;
let pollTimer = null;
let countsChart = null;
let timelineChart = null;

// ROI canvas state
let roiStart = null;
let roiEnd   = null;
let roiImage = null;

// ── DOM refs ─────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const fileNameEl    = document.getElementById('file-name');
const runBtn        = document.getElementById('run-btn');
const uploadStatus  = document.getElementById('upload-status');
const progressSec   = document.getElementById('progress-section');
const progressBar   = document.getElementById('progress-bar');
const progressLabel = document.getElementById('progress-label');
const resultsSec    = document.getElementById('results-section');
const roiSec        = document.getElementById('roi-section');
const resultVideo   = document.getElementById('result-video');
const summaryPills  = document.getElementById('summary-pills');
const countsCanvas  = document.getElementById('counts-chart');
const timelineCanvas= document.getElementById('timeline-chart');
const roiCanvas     = document.getElementById('roi-canvas');
const roiStatus     = document.getElementById('roi-status');
const roiResults    = document.getElementById('roi-results');
const jobsBody      = document.getElementById('jobs-body');

// ── File drop & select ───────────────────────────────────────────
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', ()  => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file);
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
  fileInput._file = file;
  fileNameEl.textContent = file.name;
  runBtn.disabled = false;
  uploadStatus.textContent = '';
}

// ── Run inference ────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  const file = fileInput._file;
  if (!file) return;

  runBtn.disabled = true;
  uploadStatus.textContent = 'Uploading…';

  const formData = new FormData();
  formData.append('file', file);

  try {
    const res = await fetch('/jobs', { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      uploadStatus.textContent = 'Error: ' + (err.detail || res.statusText);
      runBtn.disabled = false;
      return;
    }
    const data = await res.json();
    currentJobId = data.job_id;
    uploadStatus.textContent = `Job ${currentJobId} queued.`;
    show(progressSec);
    hide(resultsSec);
    hide(roiSec);
    startPolling();
  } catch (e) {
    uploadStatus.textContent = 'Upload failed: ' + e.message;
    runBtn.disabled = false;
  }
});

// ── Polling ──────────────────────────────────────────────────────
function startPolling() {
  clearInterval(pollTimer);
  pollTimer = setInterval(pollJob, 3000);
  pollJob();
}

async function pollJob() {
  if (!currentJobId) return;
  try {
    const res = await fetch(`/jobs/${currentJobId}`);
    const job = await res.json();

    const pct = job.progress || 0;
    progressBar.style.width = pct + '%';
    progressLabel.textContent = pct + '%  — ' + job.status;

    if (job.status === 'done') {
      clearInterval(pollTimer);
      progressBar.style.width = '100%';
      progressLabel.textContent = '100% — done';
      await loadResults(currentJobId);
    } else if (job.status === 'error') {
      clearInterval(pollTimer);
      progressLabel.textContent = 'Error: ' + job.error;
    }
  } catch (e) {
    console.error('Poll error:', e);
  }
}

// ── Results ──────────────────────────────────────────────────────
async function loadResults(jobId) {
  const [counts, timeline] = await Promise.all([
    fetch(`/jobs/${jobId}/counts`).then(r => r.json()),
    fetch(`/jobs/${jobId}/timeline`).then(r => r.json()),
  ]);

  // Video
  resultVideo.src = `/jobs/${jobId}/video`;
  resultVideo.load();

  // Summary pills
  renderPills(summaryPills, counts);

  // Bar chart
  renderCountsChart(counts.class_counts || {});

  // Timeline chart
  renderTimelineChart(timeline);

  show(resultsSec);

  // Load ROI frame
  await loadRoiFrame(jobId);
  show(roiSec);

  loadJobs();
}

function renderPills(container, counts) {
  const cc = counts.class_counts || {};
  const total = counts.total_detections || 0;
  const dominant = counts.dominant_class || '—';
  const density = (counts.density_estimate * 100).toFixed(1);

  container.innerHTML = `
    <div class="pill">Total detections <strong>${total}</strong></div>
    <div class="pill">Dominant class <strong>${dominant}</strong></div>
    <div class="pill">Traffic density <strong>${density}%</strong></div>
    ${Object.entries(cc).map(([cls, n]) =>
      `<div class="pill pill-${cls}">${cls} <strong>${n}</strong></div>`
    ).join('')}
  `;
}

function renderCountsChart(classCounts) {
  if (countsChart) countsChart.destroy();
  const labels = Object.keys(classCounts);
  const values = Object.values(classCounts);
  const colors = labels.map(l => CLASS_COLORS[l] || 'rgba(100,100,100,0.8)');

  countsChart = new Chart(countsCanvas, {
    type: 'bar',
    data: {
      labels,
      datasets: [{ label: 'Detections', data: values, backgroundColor: colors, borderRadius: 6 }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
    },
  });
}

function renderTimelineChart(timeline) {
  if (timelineChart) timelineChart.destroy();
  const frames = timeline.map(d => d.frame);
  const counts = timeline.map(d => d.count);

  timelineChart = new Chart(timelineCanvas, {
    type: 'line',
    data: {
      labels: frames,
      datasets: [{
        label: 'Vehicles detected',
        data: counts,
        borderColor: 'rgba(99, 102, 241, 0.9)',
        backgroundColor: 'rgba(99, 102, 241, 0.15)',
        fill: true,
        tension: 0.3,
        pointRadius: 0,
      }],
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: {
        x: { title: { display: true, text: 'Frame' } },
        y: { beginAtZero: true, title: { display: true, text: 'Vehicles' } },
      },
    },
  });
}

// ── ROI canvas ───────────────────────────────────────────────────
async function loadRoiFrame(jobId) {
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      roiImage = img;
      roiCanvas.width  = img.naturalWidth;
      roiCanvas.height = img.naturalHeight;
      roiCanvas.style.maxWidth = '100%';
      drawRoi();
      resolve();
    };
    img.onerror = resolve;
    img.src = `/jobs/${jobId}/frame?t=${Date.now()}`;
  });
}

function drawRoi() {
  if (!roiImage) return;
  const ctx = roiCanvas.getContext('2d');
  ctx.drawImage(roiImage, 0, 0);
  if (roiStart && roiEnd) {
    const x = Math.min(roiStart.x, roiEnd.x);
    const y = Math.min(roiStart.y, roiEnd.y);
    const w = Math.abs(roiEnd.x - roiStart.x);
    const h = Math.abs(roiEnd.y - roiStart.y);
    ctx.strokeStyle = '#f43f5e';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(244,63,94,0.1)';
    ctx.fillRect(x, y, w, h);
  }
}

function canvasPos(e) {
  const rect = roiCanvas.getBoundingClientRect();
  const scaleX = roiCanvas.width  / rect.width;
  const scaleY = roiCanvas.height / rect.height;
  const src = e.touches ? e.touches[0] : e;
  return {
    x: Math.round((src.clientX - rect.left) * scaleX),
    y: Math.round((src.clientY - rect.top)  * scaleY),
  };
}

roiCanvas.addEventListener('mousedown', e => { roiStart = canvasPos(e); roiEnd = null; });
roiCanvas.addEventListener('mousemove', e => {
  if (!roiStart) return;
  roiEnd = canvasPos(e);
  drawRoi();
});
roiCanvas.addEventListener('mouseup', () => { /* keep selection */ });

document.getElementById('apply-roi-btn').addEventListener('click', async () => {
  if (!roiStart || !roiEnd || !currentJobId) return;
  const x = Math.min(roiStart.x, roiEnd.x);
  const y = Math.min(roiStart.y, roiEnd.y);
  const w = Math.abs(roiEnd.x - roiStart.x);
  const h = Math.abs(roiEnd.y - roiStart.y);
  if (w < 10 || h < 10) { roiStatus.textContent = 'Draw a larger region.'; return; }

  roiStatus.textContent = 'Computing…';
  const res = await fetch(`/jobs/${currentJobId}/roi?x=${x}&y=${y}&w=${w}&h=${h}`, { method: 'POST' });
  const data = await res.json();
  roiStatus.textContent = `${data.total_detections} detection(s) in ROI`;
  renderRoiPills(data);
});

document.getElementById('clear-roi-btn').addEventListener('click', () => {
  roiStart = roiEnd = null;
  roiResults.innerHTML = '';
  roiStatus.textContent = '';
  drawRoi();
});

function renderRoiPills(data) {
  const cc = data.class_counts || {};
  roiResults.innerHTML = `
    <div class="pill">Total <strong>${data.total_detections}</strong></div>
    ${Object.entries(cc).map(([cls, n]) =>
      `<div class="pill pill-${cls}">${cls} <strong>${n}</strong></div>`
    ).join('')}
  `;
}

// ── Past jobs table ──────────────────────────────────────────────
document.getElementById('refresh-jobs-btn').addEventListener('click', loadJobs);

async function loadJobs() {
  try {
    const jobs = await fetch('/jobs').then(r => r.json());
    jobsBody.innerHTML = jobs.map(j => `
      <tr>
        <td><code>${j.job_id}</code></td>
        <td><span class="badge badge-${j.status}">${j.status}</span></td>
        <td>${j.progress}%</td>
        <td>${j.created_at.replace('T', ' ').slice(0, 19)}</td>
        <td>${j.status === 'done'
          ? `<button class="secondary small" onclick="loadJobResults('${j.job_id}')">View</button>`
          : ''}</td>
      </tr>
    `).join('');
  } catch (e) {
    console.error('Failed to load jobs:', e);
  }
}

async function loadJobResults(jobId) {
  currentJobId = jobId;
  show(progressSec);
  progressBar.style.width = '100%';
  progressLabel.textContent = '100% — done';
  await loadResults(jobId);
}

// ── Helpers ──────────────────────────────────────────────────────
function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

// Initial load
loadJobs();
