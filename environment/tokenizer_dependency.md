# Tokenizer Dependency

The retained textual summaries use two tokenizer modes:

- `regex`: local regular-expression tokenization implemented in `scripts/run_corpus_matrix.py`.
- `gpt2`: Hugging Face `gpt2` tokenizer pinned to revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`.

The release does not vendor the GPT-2 tokenizer files. Recomputing the GPT-2 half of the textual grid therefore requires either network access to Hugging Face or a pre-populated local Hugging Face cache containing that exact revision.

The retained paper-facing result artifacts already include the GPT-2-tokenizer computations. The cached GTE embeddings are bundled as `.npy` files, so metric recomputation from retained chunks and embeddings does not require downloading the embedding model unless embeddings are regenerated.

For offline recomputation, prepare the tokenizer cache before running:

```bash
python - <<'PY'
from transformers import AutoTokenizer
AutoTokenizer.from_pretrained(
    "gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    use_fast=True,
)
PY
```

Then run the release scripts with the same cache location available through the normal Hugging Face cache environment.
