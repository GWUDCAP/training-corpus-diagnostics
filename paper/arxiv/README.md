# arXiv Source Bundle

This directory contains the self-contained LaTeX source for the paper.

- Entrypoint: `main.tex`
- Figures: `figures/*.pdf`
- Bibliography: references are included directly in `main.tex`; no BibTeX or Biber step is required.

Build locally with:

```bash
tectonic main.tex
```

For arXiv, upload the contents of this directory, with `main.tex` as the TeX source file.
