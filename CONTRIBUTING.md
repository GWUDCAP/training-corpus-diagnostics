# Contributing

This repository has two contribution paths:

1. corrections to the release artifact, paper text, scripts, or documentation;
2. public scorecard submissions for additional corpora.

For paper or release corrections, open a normal pull request and describe the artifact or claim being changed. Any change that alters paper-facing numbers must include the regenerated JSON artifacts and a passing verifier run.

For scorecard additions, use the public pipeline and open a pull request with a small scorecard entry rather than committing full chunks and embeddings. Full artifacts should be hosted externally, with a SHA-256 digest recorded in the entry.

Scorecard entries are diagnostic positions under a fixed metric suite. They are not corpus quality rankings and do not claim downstream model behavior.

## Scorecard Inclusion Criteria

A public scorecard PR must include:

- a generated `scorecard_entry.json`;
- an artifact URL for the full submission bundle;
- a SHA-256 digest for that artifact bundle;
- corpus source, extraction, provenance, and license metadata;
- canonical settings unless explicitly marked noncanonical;
- `role: "candidate"` and `stage_role: "unlabeled"` unless the corpus role is externally established;
- passing `python scripts/validate_scorecard_submission.py --entry <path-to-entry>`.

Public registry status tiers:

- `submitted`: PR opened, not validated.
- `artifact_validated`: schema, hashes, metadata, and alignment checks pass.
- `agent_reviewed`: a maintainer or reviewing agent has checked provenance and interpretation language.
- `reproduced`: a maintainer or trusted runner reran the public metric suite and reproduced the submitted summaries within release tolerance.

Agent review should answer:

1. Does the submission use the declared metric-suite version?
2. Are source, extraction mode, chunk budget, and embedding model disclosed?
3. Is the artifact URL reachable to reviewers, and does its SHA-256 match?
4. Are chunks and embeddings row-aligned?
5. Is the scorecard language diagnostic rather than evaluative?
6. Should the entry be merged as `artifact_validated`, `agent_reviewed`, or held for reproduction?

