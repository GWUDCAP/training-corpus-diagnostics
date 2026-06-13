#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${1:-randomized_panel_2048_seed20260505}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_CONFIG="${2:-$ROOT/configs/paper_panel_sources_randomized_2048.json}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
DATA_DIR="$RUN_ROOT/data"
MATRIX_DIR="$RUN_ROOT/results/final_matrix"
FIG_DIR="$RUN_ROOT/figures"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$RUN_ROOT" "$DATA_DIR" "$MATRIX_DIR" "$FIG_DIR" "$LOG_DIR"
export MPLCONFIGDIR="$RUN_ROOT/matplotlib"
mkdir -p "$MPLCONFIGDIR"

echo "[battery] root=$ROOT"
echo "[battery] run_root=$RUN_ROOT"
echo "[battery] source_config=$SOURCE_CONFIG"
echo "[battery] python=$(command -v python)"
python - <<'PY'
import platform, sys
print("[battery] version=" + sys.version.replace("\n", " "))
print("[battery] platform=" + platform.platform())
PY

echo "[1/8] randomized chunks"
python "$ROOT/scripts/build_corpus_artifacts.py" build-chunks \
  --sources "$SOURCE_CONFIG" \
  --out-root "$DATA_DIR" \
  --overwrite 2>&1 | tee "$LOG_DIR/01_build_chunks.log"

echo "[2/8] embeddings"
python "$ROOT/scripts/build_corpus_artifacts.py" build-embeddings \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-root "$DATA_DIR" \
  --model-name Alibaba-NLP/gte-base-en-v1.5 \
  --batch-size 16 \
  --max-length 512 \
  --device auto \
  --overwrite 2>&1 | tee "$LOG_DIR/02_build_embeddings.log"

echo "[3/8] primary matrix"
python "$ROOT/scripts/run_corpus_matrix.py" \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-dir "$MATRIX_DIR/recomputed" 2>&1 | tee "$LOG_DIR/03_run_corpus_matrix.log"

echo "[4/8] directed targeted mixtures"
python "$ROOT/scripts/run_mixture_panel.py" \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-dir "$MATRIX_DIR/mixtures" 2>&1 | tee "$LOG_DIR/04_run_mixture_panel.log"

echo "[5/8] textual grid"
python "$ROOT/tools/rebuild_textual_grid_summary.py" \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-dir "$MATRIX_DIR/sweeps" 2>&1 | tee "$LOG_DIR/05_rebuild_textual_grid_summary.log"

echo "[6/8] semantic k stability"
python "$ROOT/scripts/run_semantic_k_stability.py" \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-dir "$MATRIX_DIR/stability" 2>&1 | tee "$LOG_DIR/06_run_semantic_k_stability.log"

echo "[7/8] bootstrap, leave-one-out, same-corpus split"
python "$ROOT/scripts/run_bootstrap_hardening.py" \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --scorecard-dir "$MATRIX_DIR/recomputed" \
  --out-dir "$MATRIX_DIR/stability" \
  --bootstrap-replicates 200 \
  --split-repeats 50 \
  --bootstrap-semantic-mode fixed 2>&1 | tee "$LOG_DIR/07_run_bootstrap_hardening.log"

echo "[8/8] figures"
python "$ROOT/scripts/make_figures.py" \
  --matrix-dir "$MATRIX_DIR" \
  --out-dir "$FIG_DIR" 2>&1 | tee "$LOG_DIR/08_make_figures.log"

python - <<PY
import json
from pathlib import Path
run_root = Path("$RUN_ROOT")
summary = {
    "schema": "lens_effects.randomized_release_battery.v1",
    "run_name": "$RUN_NAME",
    "run_root": str(run_root),
    "manifest": str(Path("$DATA_DIR") / "corpus_manifest.generated.json"),
    "matrix_dir": str(Path("$MATRIX_DIR")),
    "figures": str(Path("$FIG_DIR")),
}
(run_root / "RUN_DONE.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
print("[done] wrote", run_root / "RUN_DONE.json")
PY
