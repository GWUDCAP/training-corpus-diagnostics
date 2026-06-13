## Type

- [ ] Release/paper correction
- [ ] Script or documentation improvement
- [ ] Scorecard submission

## Summary

Describe the change and the artifact or workflow it affects.

## Checks

- [ ] `python scripts/verify_release_artifacts.py`
- [ ] `python scripts/validate_scorecard_submission.py --registry scorecard/registry`
- [ ] No stale local paths, old figure names, or generated junk files

## Scorecard submissions only

- [ ] Adds or updates a small scorecard entry, not full chunks/embeddings
- [ ] Provides artifact URL and SHA-256 digest
- [ ] Declares source, extraction mode, chunk budget, embedding model, and provenance status
- [ ] Uses diagnostic-position language, not corpus quality ranking language

