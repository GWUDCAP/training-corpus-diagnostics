#!/usr/bin/env python3
"""Build the formatted LaTeX preprint PDF."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LATEX_DIR = ROOT / "paper" / "latex"
LATEX_MAIN = LATEX_DIR / "manuscript.tex"
LATEX_PDF = LATEX_DIR / "manuscript.pdf"
DEFAULT_OUT = ROOT / "paper" / "lens_effects_preprint.pdf"


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the formatted LaTeX preprint PDF.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if shutil.which("tectonic") is None:
        raise SystemExit("tectonic is required to build the formatted preprint PDF")
    if not LATEX_MAIN.exists():
        raise SystemExit(f"missing LaTeX source: {LATEX_MAIN}")

    subprocess.run(["tectonic", LATEX_MAIN.name], cwd=LATEX_DIR, check=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(LATEX_PDF, args.out)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
