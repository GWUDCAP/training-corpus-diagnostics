# Publishing Notes

This repository is the canonical public artifact for **Lens Effects: Structured Geometry in Training Corpora**.

## Current Public Path

1. Verify the release locally:

   ```bash
   python scripts/verify_release_artifacts.py
   python scripts/validate_scorecard_submission.py --registry scorecard/registry
   ```

2. Regenerate figures and the review PDF if needed:

   ```bash
   python scripts/make_figures.py
   python scripts/build_preprint_pdf.py
   ```

3. Create a GitHub release from the final tag.
4. Archive that release on Zenodo and record the DOI in `CITATION.cff`, `README.md`, and the manuscript.
5. Submit the PDF/source package to arXiv after the DOI and repository links are stable.

## arXiv Positioning

Recommended primary category: `cs.CL`.

Recommended cross-list: `cs.LG`.

The paper should be described as an upstream corpus-measurement and reproducibility contribution. It does not claim downstream causality, a benchmark, or a universal taxonomy.

## Endorsement Packet

For an arXiv endorsement request, include:

- title and abstract;
- requested category;
- GitHub release URL;
- Zenodo DOI once available;
- one-sentence claim: the paper measures stage-linked corpus geometry before training;
- one-sentence boundary: it does not claim downstream causal effects;
- verification command and expected output;
- links to `README.md`, `paper/manuscript.md`, and `artifact_manifest.json`.

Ask for category endorsement, not a substantive review. Do not mass-email endorsers.

## AI-Use Disclosure

The manuscript was developed with substantial assistance from AI coding and writing tools. The human author is responsible for the paper's claims, analysis choices, artifact release, and final text. AI tools are not listed as authors.

## Scorecard Policy

The public scorecard is a registry of diagnostic positions, not a quality ranking. Submitted corpora require:

- generated scorecard entry JSON;
- hosted full artifact bundle URL;
- SHA-256 digest of that bundle;
- provenance and license metadata;
- clear status tier: `submitted`, `artifact_validated`, `agent_reviewed`, or `reproduced`.

Do not commit external chunks or embeddings for scorecard submissions.
