#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?usage: scripts/run_randomized_matrix_draw.sh <seed> [run_name] [source_config]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${2:-randomized_panel_2048_seed${SEED}_matrix}"
SOURCE_CONFIG="${3:-$ROOT/configs/paper_panel_sources_randomized_2048.json}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
DATA_DIR="$RUN_ROOT/data"
MATRIX_DIR="$RUN_ROOT/results/final_matrix"
LOG_DIR="$RUN_ROOT/logs"

mkdir -p "$RUN_ROOT" "$DATA_DIR" "$MATRIX_DIR" "$LOG_DIR"

echo "[draw] root=$ROOT"
echo "[draw] run_root=$RUN_ROOT"
echo "[draw] source_config=$SOURCE_CONFIG"
echo "[draw] sample_seed=$SEED"
echo "[draw] python=$(command -v python)"

echo "[1/3] randomized chunks"
python "$ROOT/scripts/build_corpus_artifacts.py" build-chunks \
  --sources "$SOURCE_CONFIG" \
  --out-root "$DATA_DIR" \
  --sample-seed "$SEED" \
  --overwrite 2>&1 | tee "$LOG_DIR/01_build_chunks.log"

echo "[2/3] embeddings"
python "$ROOT/scripts/build_corpus_artifacts.py" build-embeddings \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-root "$DATA_DIR" \
  --model-name Alibaba-NLP/gte-base-en-v1.5 \
  --batch-size 16 \
  --max-length 512 \
  --device auto \
  --overwrite 2>&1 | tee "$LOG_DIR/02_build_embeddings.log"

echo "[3/3] primary matrix"
python "$ROOT/scripts/run_corpus_matrix.py" \
  --manifest "$DATA_DIR/corpus_manifest.generated.json" \
  --out-dir "$MATRIX_DIR/recomputed" 2>&1 | tee "$LOG_DIR/03_run_corpus_matrix.log"

python - <<PY
import json
from pathlib import Path
run_root = Path("$RUN_ROOT")
summary = {
    "schema": "lens_effects.randomized_matrix_draw.v1",
    "run_name": "$RUN_NAME",
    "sample_seed": int("$SEED"),
    "run_root": str(run_root),
    "manifest": str(Path("$DATA_DIR") / "corpus_manifest.generated.json"),
    "matrix_dir": str(Path("$MATRIX_DIR")),
}
(run_root / "RUN_DONE.json").write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n")
print("[done] wrote", run_root / "RUN_DONE.json")
PY
