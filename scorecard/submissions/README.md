# Scorecard Submission PRs

Do not commit full submitted chunks or embeddings to this repository. Package the full submission bundle, host it somewhere reviewers can access, record its SHA-256 digest, and commit only the small scorecard entry JSON/Markdown in the pull request.

Recommended contributor flow:

```bash
python scripts/build_corpus_artifacts.py build-chunks \
  --sources my_sources.json \
  --out-root my_scorecard_run

python scripts/build_corpus_artifacts.py build-embeddings \
  --manifest my_scorecard_run/corpus_manifest.generated.json \
  --out-root my_scorecard_run \
  --model-name Alibaba-NLP/gte-base-en-v1.5

python scripts/run_corpus_matrix.py \
  --manifest my_scorecard_run/corpus_manifest.generated.json \
  --out-dir my_scorecard_run/matrix

python scripts/package_scorecard_submission.py \
  --manifest my_scorecard_run/corpus_manifest.generated.json \
  --matrix-dir my_scorecard_run/matrix \
  --labels my_new_corpus \
  --source-config my_sources.json \
  --artifact-url https://example.org/my_new_corpus_scorecard_submission.zip \
  --artifact-sha256 <sha256-of-zip> \
  --out-dir my_new_corpus_scorecard_submission \
  --zip
```

Then copy `my_new_corpus_scorecard_submission/scorecard_entry.json` into a PR under `scorecard/registry/entries/<label>.json`, add it to `scorecard/registry/index.json`, and run:

```bash
python scripts/validate_scorecard_submission.py --registry scorecard/registry
```

Maintainers or reviewing agents may promote entries through `artifact_validated`, `agent_reviewed`, and `reproduced` status tiers.

