#!/usr/bin/env python3
"""Cross-platform launcher for the MobileViT traffic dashboard.

Detects the OS and available GPU backend, sets the environment
variables PyTorch needs on that platform, and execs uvicorn.

Works on:
  - Linux + AMD ROCm (Radeon 780M, gfx1103 spoofing via HSA_OVERRIDE_GFX_VERSION)
  - Linux + NVIDIA CUDA
  - Linux + CPU only
  - macOS Apple Silicon (MPS)
  - macOS Intel (CPU)
  - Windows + NVIDIA CUDA
  - Windows + AMD via DirectML (when torch-directml is installed)
  - Windows + CPU only

Usage::

    python scripts/run_dashboard.py
    python scripts/run_dashboard.py --port 9000
    python scripts/run_dashboard.py --stream-device cpu      # default
    python scripts/run_dashboard.py --stream-device cuda     # discrete GPU
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent


# ── Platform detection ──────────────────────────────────────────


def detect_backend() -> tuple[str, dict[str, str]]:
    """Return (label, env_vars_to_set) for the best backend on this machine."""
    sysname = platform.system()  # "Linux", "Darwin", "Windows"

    try:
        import torch
    except ImportError:
        return "cpu (torch not installed yet — `pip install -r requirements.txt`)", {}

    # NVIDIA CUDA — same on Linux and Windows
    if torch.cuda.is_available() and torch.version.cuda is not None:
        return f"CUDA ({torch.cuda.get_device_name(0)})", {}

    # AMD ROCm — Linux only
    if torch.cuda.is_available() and getattr(torch.version, "hip", None):
        env = {
            # gfx1103 (Radeon 780M iGPU) isn't in rocBLAS's TensileLibrary.
            # Spoof gfx1100 — works for most ops, occasionally aborts on
            # large allocations. CLAUDE: see also STREAM_DEVICE=cpu.
            "HSA_OVERRIDE_GFX_VERSION": "11.0.0",
            "PYTORCH_HIP_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:128",
            "HSA_XNACK": "1",
        }
        return f"ROCm ({torch.cuda.get_device_name(0)}, gfx1103 spoofed as gfx1100)", env

    # Apple Silicon (MPS) — macOS only
    if sysname == "Darwin" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return f"MPS ({platform.machine()})", {}

    # Windows + AMD via DirectML (separate package)
    if sysname == "Windows":
        try:
            import torch_directml  # type: ignore
            return f"DirectML ({torch_directml.device_name(0)})", {}
        except ImportError:
            pass

    return f"CPU ({sysname} {platform.machine()})", {}


def uvicorn_command() -> list[str]:
    """Locate the venv's uvicorn binary, falling back to system uvicorn."""
    if os.name == "nt":  # Windows
        candidates = [REPO_ROOT / "venv" / "Scripts" / "uvicorn.exe"]
    else:
        candidates = [REPO_ROOT / "venv" / "bin" / "uvicorn"]
    for c in candidates:
        if c.exists():
            return [str(c)]
    found = shutil.which("uvicorn")
    if found:
        return [found]
    return [sys.executable, "-m", "uvicorn"]


# ── Main ────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch the MobileViT dashboard.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--stream-device",
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device for the live stream pipeline. Defaults to cpu so an "
             "offline GPU job and a live stream don't share VRAM.",
    )
    args = parser.parse_args()

    backend, env = detect_backend()

    # Set platform env vars in our own environment so subprocess.run inherits them.
    for k, v in env.items():
        os.environ[k] = v
    os.environ["STREAM_DEVICE"] = args.stream_device

    print("═════════════════════════════════════════════════════════")
    print(f"  MobileViT Dashboard")
    print(f"  http://localhost:{args.port}")
    print(f"  OS:       {platform.system()} {platform.release()}")
    print(f"  Backend:  {backend}")
    print(f"  Stream:   {args.stream_device}   (offline jobs use the backend above)")
    if env:
        print(f"  Env:      {', '.join(f'{k}={v}' for k, v in env.items())}")
    print("═════════════════════════════════════════════════════════")

    cmd = uvicorn_command() + [
        "src.app.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    try:
        return subprocess.call(cmd, cwd=REPO_ROOT)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
