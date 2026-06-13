#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


EXCLUDE = {
    "release_integrity_manifest.json",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix()
        if path.name in EXCLUDE or "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        rows.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    out = {
        "schema": "lens_effects.release_integrity_manifest.v1",
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "file_count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "files": rows,
    }
    (root / "release_integrity_manifest.json").write_text(json.dumps(out, indent=2, ensure_ascii=True) + "\n")
    print(f"[done] wrote {root / 'release_integrity_manifest.json'} files={out['file_count']} bytes={out['total_bytes']}")


if __name__ == "__main__":
    main()
