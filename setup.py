#!/usr/bin/env python3
"""One-click setup for Knowledge Catalyst (macOS and Windows)."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).parent.resolve()
VENV_DIR = REPO_ROOT / "env"
MODEL_DIR = REPO_ROOT / "models" / "e5-large-v2"
REQUIREMENTS = REPO_ROOT / "requirements.txt"

HF_BASE = "https://huggingface.co/intfloat/e5-large-v2/resolve/main"
MODEL_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "sentence_bert_config.json",
    "modules.json",
    "1_Pooling/config.json",
    "model.safetensors",  # largest — downloaded last
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def banner(text: str) -> None:
    width = 60
    print(f"\n{'─' * width}\n  {text}\n{'─' * width}")


def run(cmd: list[str], *, shell: bool = False, **kwargs) -> None:
    display = cmd if not shell else cmd
    if isinstance(display, list):
        print(f"  $ {' '.join(str(c) for c in display)}")
    subprocess.run(cmd, check=True, shell=shell, **kwargs)


def on_windows() -> bool:
    return platform.system() == "Windows"


def on_macos() -> bool:
    return platform.system() == "Darwin"


def venv_python() -> Path:
    if on_windows():
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


# ── Step 1: Python version ─────────────────────────────────────────────────────

def check_python() -> None:
    banner("Checking Python version")
    v = sys.version_info
    print(f"  Python {v.major}.{v.minor}.{v.micro}")
    if sys.version_info < (3, 10):
        sys.exit("  ERROR: Python 3.10 or later is required. "
                 "Download from https://python.org/downloads")
    print("  OK")


# ── Step 2: System dependencies ───────────────────────────────────────────────

def install_system_deps() -> None:
    banner("Installing system dependencies (Tesseract OCR + Poppler)")
    if on_macos():
        _install_macos_deps()
    elif on_windows():
        _install_windows_deps()
    else:
        sys.exit(
            "  Unsupported OS. Install Tesseract and Poppler manually, "
            "then re-run this script."
        )


def _install_macos_deps() -> None:
    # Ensure Homebrew is on PATH (Apple Silicon installs to /opt/homebrew/bin)
    os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")

    if not shutil.which("brew"):
        print("  Homebrew not found — installing (this may take a few minutes)...")
        run(
            '/bin/bash -c "$(curl -fsSL '
            'https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
            shell=True,
        )
        # Re-export PATH after Homebrew install
        os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")

    for pkg, check_bin in [("tesseract", "tesseract"), ("poppler", "pdfinfo")]:
        if shutil.which(check_bin):
            print(f"  {pkg}: already installed")
        else:
            run(["brew", "install", pkg])


def _install_windows_deps() -> None:
    if not shutil.which("choco"):
        print("  Chocolatey not found — installing...")
        ps = (
            "Set-ExecutionPolicy Bypass -Scope Process -Force; "
            "[System.Net.ServicePointManager]::SecurityProtocol = "
            "[System.Net.ServicePointManager]::SecurityProtocol -bor 3072; "
            "iex ((New-Object System.Net.WebClient).DownloadString("
            "'https://community.chocolatey.org/install.ps1'))"
        )
        run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps]
        )

    if not shutil.which("tesseract"):
        run(["choco", "install", "tesseract", "-y"])
    else:
        print("  tesseract: already installed")

    if not shutil.which("pdfinfo"):
        run(["choco", "install", "poppler", "-y"])
    else:
        print("  poppler: already installed")


# ── Step 3: Virtual environment ───────────────────────────────────────────────

def create_venv() -> None:
    banner("Creating virtual environment")
    if VENV_DIR.exists() and venv_python().exists():
        print(f"  {VENV_DIR.name}/ already exists, skipping")
        return
    run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print(f"  Created {VENV_DIR}")


# ── Step 4: Python dependencies ───────────────────────────────────────────────

def install_python_deps() -> None:
    banner("Installing Python dependencies (~1–1.5 GB)")
    py = str(venv_python())

    run([py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    if on_windows():
        # Install CPU-only PyTorch first to avoid the multi-GB CUDA wheel.
        # sentence-transformers will reuse this installation.
        print("  Installing CPU-only PyTorch (avoids CUDA download on Windows)...")
        run([
            py, "-m", "pip", "install", "torch",
            "--index-url", "https://download.pytorch.org/whl/cpu",
            "--quiet",
        ])

    run([py, "-m", "pip", "install", "-r", str(REQUIREMENTS), "--quiet"])


# ── Step 5: Embedding model ────────────────────────────────────────────────────

def download_model() -> None:
    banner("Downloading embedding model — safetensors only (~1.3 GB)")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "1_Pooling").mkdir(exist_ok=True)

    for filename in MODEL_FILES:
        dest = MODEL_DIR / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"  {filename}: already present, skipping")
            continue
        url = f"{HF_BASE}/{filename}"
        _download(url, dest)


def _download(url: str, dest: Path) -> None:
    label = dest.name

    def _progress(block_num: int, block_size: int, total_size: int) -> None:
        if total_size > 0:
            done = min(block_num * block_size, total_size)
            pct = done * 100 // total_size
            mb_done = done / 1_048_576
            mb_total = total_size / 1_048_576
            print(
                f"\r  {label}: {pct:3d}%  ({mb_done:.0f}/{mb_total:.0f} MB)",
                end="",
                flush=True,
            )

    try:
        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print(f"\r  {label}: done{' ' * 30}")
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


# ── Done ───────────────────────────────────────────────────────────────────────

def print_done() -> None:
    if on_windows():
        py = r"env\Scripts\python"
    else:
        py = "env/bin/python"

    banner("Setup complete!")
    print(f"""
  Ingest your documents:
    {py} ingest.py --paths ./data

  Search your knowledge base (requires LM Studio for summaries):
    {py} query.py --embedding-model models/e5-large-v2

  Generate embedding visualisation plots:
    {py} plot_embeddings.py --chroma-path ./chroma_db
""")


def main() -> None:
    check_python()
    install_system_deps()
    create_venv()
    install_python_deps()
    download_model()
    print_done()


if __name__ == "__main__":
    main()
