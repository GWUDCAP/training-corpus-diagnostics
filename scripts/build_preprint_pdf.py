#!/usr/bin/env python3
# ===== IMPORTS ===== #
## ===== STDLIB ===== ##
from __future__ import annotations

import argparse
import re
import textwrap
from pathlib import Path

## ===== 3RD-PARTY ===== ##
from PIL import Image, ImageDraw, ImageFont

# ===== GLOBALS ===== #

ROOT = Path(__file__).resolve().parents[1]
PAGE_W, PAGE_H = 1700, 2200
MARGIN_X, MARGIN_Y = 140, 120
LINE_PAD = 8
TEXT_W = PAGE_W - 2 * MARGIN_X
FIG_RE = re.compile(r"^!\[(?P<alt>.*?)\]\((?P<path>.*?)\)\s*$")

# ===== FUNCTIONS ===== #

def font(size: int, bold: bool = False, mono: bool = False) -> ImageFont.FreeTypeFont:
    candidates = []
    if mono:
        candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]
    elif bold:
        candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"]
    else:
        candidates = ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()

BODY = font(30)
BOLD = font(30, bold=True)
H1 = font(48, bold=True)
H2 = font(38, bold=True)
H3 = font(32, bold=True)
MONO = font(23, mono=True)
CAPTION = font(25)

def clean_inline(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = text.replace("\\_", "_")
    return text

def new_page() -> Image.Image:
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    return page

def draw_wrapped(draw: ImageDraw.ImageDraw, text: str, x: int, y: int, width: int, face: ImageFont.ImageFont, fill: str = "#111111") -> int:
    if not text:
        return y + face.size + LINE_PAD
    avg_char = max(7, int(draw.textlength("abcdefghijklmnopqrstuvwxyz", font=face) / 26))
    chars = max(28, width // avg_char)
    lines = []
    for raw in text.splitlines() or [""]:
        if not raw:
            lines.append("")
            continue
        lines.extend(textwrap.wrap(raw, width=chars, break_long_words=False, replace_whitespace=False) or [""])
    for line in lines:
        draw.text((x, y), line, font=face, fill=fill)
        y += face.size + LINE_PAD
    return y

def render_markdown(markdown: str, root: Path, out: Path) -> None:
    pages = []
    page = new_page()
    draw = ImageDraw.Draw(page)
    y = MARGIN_Y

    def ensure(space: int) -> None:
        nonlocal page, draw, y
        if y + space <= PAGE_H - MARGIN_Y:
            return
        pages.append(page)
        page = new_page()
        draw = ImageDraw.Draw(page)
        y = MARGIN_Y

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        match = FIG_RE.match(line)
        if match:
            rel = match.group("path").replace("../", "")
            fig_path = (root / rel).resolve()
            if fig_path.exists():
                img = Image.open(fig_path).convert("RGB")
                max_w, max_h = TEXT_W, 900
                scale = min(max_w / img.width, max_h / img.height, 1.0)
                size = (int(img.width * scale), int(img.height * scale))
                ensure(size[1] + 80)
                img = img.resize(size, Image.Resampling.LANCZOS)
                page.paste(img, (MARGIN_X + (TEXT_W - size[0]) // 2, y))
                y += size[1] + 36
            i += 1
            continue
        if not line:
            y += 22
            i += 1
            continue
        if line.startswith("# "):
            ensure(110)
            y = draw_wrapped(draw, clean_inline(line[2:]), MARGIN_X, y, TEXT_W, H1)
            y += 26
        elif line.startswith("## "):
            ensure(86)
            y += 14
            y = draw_wrapped(draw, clean_inline(line[3:]), MARGIN_X, y, TEXT_W, H2)
            y += 12
        elif line.startswith("### "):
            ensure(70)
            y += 8
            y = draw_wrapped(draw, clean_inline(line[4:]), MARGIN_X, y, TEXT_W, H3)
            y += 8
        elif line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i])
                i += 1
            for row in block:
                ensure(44)
                y = draw_wrapped(draw, row, MARGIN_X, y, TEXT_W, MONO, fill="#222222")
            y += 10
            continue
        elif line.startswith("**Figure"):
            ensure(80)
            y = draw_wrapped(draw, clean_inline(line), MARGIN_X, y, TEXT_W, CAPTION, fill="#333333")
            y += 16
        elif line.startswith("- ") or re.match(r"^\d+\. ", line):
            ensure(60)
            y = draw_wrapped(draw, clean_inline(line), MARGIN_X + 24, y, TEXT_W - 24, BODY)
        else:
            ensure(120)
            y = draw_wrapped(draw, clean_inline(line), MARGIN_X, y, TEXT_W, BODY)
            y += 10
        i += 1

    pages.append(page)
    out.parent.mkdir(parents=True, exist_ok=True)
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=150)
    print(f"[done] wrote {out} pages={len(pages)}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Build a self-contained PDF preview from manuscript Markdown and rendered figures.")
    ap.add_argument("--manuscript", type=Path, default=ROOT / "paper" / "manuscript.md")
    ap.add_argument("--out", type=Path, default=ROOT / "paper" / "lens_effects_preprint.pdf")
    args = ap.parse_args()
    render_markdown(args.manuscript.read_text(encoding="utf-8"), ROOT, args.out)

if __name__ == "__main__":
    main()
