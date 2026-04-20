# Reranker Annotation Pipeline

Three scripts that together build a gold-labelled ROR reranker dataset
from a list of raw affiliation strings.

```
                                (first time only, or when ROR updates)
                       ┌────────────────────────────────────────────┐
                       │  update_ror_registry.py                    │
                       │    - fetches latest dump from Zenodo       │
                       │    - normalizes via RegistryManager        │
                       └───────────────────┬────────────────────────┘
                                           ▼
                       data/registry/ror/ror_records.jsonl
                                           │
affils.txt ───────────────────────────────►│
(one raw affiliation                       │
 per line)                                 │
                                           ▼
                       ┌────────────────────────────────────────────┐
                       │  prepare_candidates.py                     │
                       │    - span identifier                       │
                       │      (affilgood-span-multilingual-v2)      │
                       │    - NER                                   │
                       │      (affilgood-ner-multilingual-v2)       │
                       │    - dense retriever                       │
                       │      (SIRIS-Lab/affilgood-dense-retriever) │
                       │    - direct match on (name, country)       │
                       └───────────────────┬────────────────────────┘
                                           ▼
                       candidates.csv   (1 row per ORG/SUBORG entity)
                                           │
                                           ▼
                       ┌────────────────────────────────────────────┐
                       │  annotate_with_llm.py                      │
                       │    - OpenAI chat completion in JSON mode   │
                       │    - gold_ror_id ∈ candidates ∪ {"NONE"}   │
                       └───────────────────┬────────────────────────┘
                                           ▼
                       candidates_annotated.csv   (reranker-ready)
```

---

## Quick start

### Environment setup (uv)

Python 3.10+ is required (the `src/run_affilel_experiments.py` file uses
PEP 604 union types like `str | Path`). Recommended setup with
[uv](https://docs.astral.sh/uv/):

```bash
# From the project root (where pyproject.toml lives)
uv venv --python 3.11
source .venv/bin/activate

uv sync                     # installs from pyproject.toml
# or, if you don't want to commit a lockfile:
uv pip install -e .
```

> **Why not just import `SpanIdentifier` / `NER` from `run_affilel_experiments.py`?**
> That file has ~900 lines of module-level code — it loads datasets,
> initializes models, and does file I/O at import time. Even on
> Python 3.10+ where its type syntax works, importing anything from it
> would trigger all that. `prepare_candidates.py` therefore keeps its
> own inline copies of the two classes it needs.

### Run the pipeline

```bash
# 0. one-time (or whenever you want the newest ROR release)
python update_ror_registry.py --data-dir data/registry --verbose

# 1. build candidate lists
python prepare_candidates.py \
    --input  affils.txt \
    --output candidates.csv \
    --ror-records data/registry/ror/ror_records.jsonl \
    --top-k 20 \
    --device cuda \
    --batch-size 16

# 2. annotate with an LLM
export OPENAI_API_KEY=sk-…
python annotate_with_llm.py \
    --input  candidates.csv \
    --output candidates_annotated.csv \
    --model  gpt-4o-mini \
    --max-candidates 20 \
    --skip-direct-matches        # optional: trust direct-match without LLM
```

Do a smoke test first with `--limit 20` on both scripts before committing
hours of GPU / API time.

---

## Script 0 — `update_ror_registry.py`

Fetches the **latest** ROR data dump from Zenodo and produces the
normalized `ror_records.jsonl` that everything downstream consumes.

Uses the Zenodo concept-DOI shortcut (`zenodo.org/api/records/6347574`)
which always resolves to the latest release, so it keeps working even
when Zenodo changes its API shape (as it has done twice since 2023).
Normalization goes through the project's own `RegistryManager` so the
output schema is identical to what the rest of the codebase expects.

```bash
# Fetch latest from Zenodo
python update_ror_registry.py --data-dir data/registry --verbose

# Force re-download even if a dump is already on disk
python update_ror_registry.py --data-dir data/registry --force --verbose

# Manual-download fallback (no network / corporate proxy issues):
# 1) grab v*-ror-data.zip from https://doi.org/10.5281/zenodo.6347574
# 2) extract the .json into data/registry/ror/
python update_ror_registry.py --skip-download --data-dir data/registry --verbose

# Higher rate limits if you automate this (CI job etc.)
python update_ror_registry.py --api-token "$ZENODO_API_TOKEN" --force
```

### What it produces

```
data/registry/ror/
├── v2.2-2026-01-29-ror-data.json   # raw ROR dump (~260 MB extracted)
└── ror_records.jsonl               # normalized — the file everything reads
```

### Heads-up: ROR v1 schema was deprecated on 8 Dec 2025

From release v2.0 (Dec 2025) onward, dumps are v2-only. `registry.py`
already handles both v1 and v2 paths, so normalization is fine; but if
you need a specific pre-Dec-2025 release for reproducibility you'll
have to fetch its Zenodo record ID manually.

---

## Script 1 — `prepare_candidates.py`

For each raw affiliation: segment into spans → run NER → for each
ORG/SUBORG entity build query variants, do a direct-match lookup, run
dense retrieval, and emit one output row per entity.

```bash
python prepare_candidates.py \
    --input  affils.txt \
    --output candidates.csv \
    --ror-records data/registry/ror/ror_records.jsonl \
    --top-k 20 \
    --device cuda \
    --batch-size 16
```

Key flags:

- `--top-k 20` — candidates to keep per entity after merging across
  query variants.
- `--retriever affilgood_dense` — uses the tagged-KB retriever; other
  retrievers from `RETRIEVER_CONFIGS` work too and will automatically
  fall back to the plain query format.
- `--limit 20` — process only first N input lines (for testing).
- `--faiss-dir` / `--faiss-auto-download` — **skip the 15-minute KB
  encoding step** by loading precomputed embeddings (see below).

### Fast path: precomputed FAISS index

Encoding the ~600k expanded KB entries with `affilgood-dense-retriever`
takes ~15 minutes on a decent GPU (much longer on CPU). If you're
running the pipeline more than once, download the precomputed FAISS
index bundled with the `SIRIS-Lab/affilgood-data` release:

```bash
# First time — downloads ~330 MB to ~/.cache/affilgood/v2.0.0/
python prepare_candidates.py \
    --input affils.txt \
    --output candidates.csv \
    --faiss-auto-download \
    --top-k 20 \
    --device cuda
```

After the first run, everything is cached — subsequent invocations
load the index in a few seconds. The `data_manager.py` file from the
affilgood project must be importable (either in `--src-path` or
installed via pip).

If you manage the cache yourself, point at the directory directly:

```bash
python prepare_candidates.py \
    --input affils.txt \
    --output candidates.csv \
    --faiss-dir ~/.cache/affilgood/v2.0.0/ror/dense \
    --top-k 20
```

**Critical constraint.** The FAISS row-IDs are keyed to whichever
`ror_records.jsonl` was used to build the index. When `--faiss-dir`
is set, the script auto-loads the paired `ror_records.jsonl` from
one directory up (`<faiss-dir>/../ror_records.jsonl`) so the
direct-match index uses the same ROR vocabulary as the FAISS index.
If you override `--ror-records` yourself, make sure it points at the
file that shipped with the bundle — otherwise you'll silently get
retrieval hits for ROR IDs that your direct-match / display lookup
doesn't know about. A warning is printed when the two diverge.

### `candidates.csv` output schema

| column | format | description |
|---|---|---|
| `row_id` | int | sequential id |
| `raw_line_id` | int | index of the source line in `affils.txt` |
| `raw_affiliation_string` | str | full original line (kept as context for the annotator) |
| `span_text` | str | sub-affiliation span that contains this entity |
| `entity` | str | ORG / SUBORG text returned by NER |
| **`affilgood_string`** | **str, plain comma-separated** | `"University of Maryland, Baltimore, USA"` — primary query variant in reranker-ready natural-text format |
| `all_variants` | str, `\|\|`-joined | all query variants in plain format |
| `retrieval_query` | str | what was actually sent to the retriever (**tagged** for `affilgood_dense`, plain otherwise); for reproducibility / debugging |
| `direct_match_ror_id` | str | exact (name + country) lookup result, empty if none |
| `direct_match_name` | str | canonical name of that record (readability) |
| `candidates_ror_ids` | JSON array | top-K ROR IDs, sorted by dense similarity desc |
| `candidates_names` | JSON array | canonical names (same order) |
| `candidates_cities` | JSON array | cities (same order) |
| `candidates_countries` | JSON array | countries (same order) |
| `candidates_scores` | JSON array | dense similarity scores (same order) |
| `n_spans` | int | spans the SpanIdentifier cut the line into |
| `n_entities_in_span` | int | ORG/SUBORG entities found in this span |
| `note` | str | `""`, `"no_spans_detected"`, or `"no_org_entity_in_span"` |

> **Format decision: `affilgood_string` is plain, not tagged.**
> The `[MENTION]/[CITY]/[COUNTRY]` tags are a retriever-specific trick
> (the `affilgood_dense` model was post-trained to understand them).
> Cross-encoder rerankers — `jina-reranker-v2/v3`, the `cometadata`
> affiliation fine-tunes on top of them, `ms-marco-MiniLM`,
> `Qwen3-Reranker` — were all trained on natural text and treat bracket
> tags as noise. Plain comma-separated concatenation matches the
> AffilGood paper's format and what `build_reranking_kb_lookup()`
> produces on the document side, so query and doc sides are consistent.
> The tagged form is still available in the `retrieval_query` column
> if you need it for reproducing retrieval runs.

---

## Script 2 — `annotate_with_llm.py`

Reads `candidates.csv` and, for each row, asks an OpenAI chat model to
pick the correct ROR ID from the candidate list (or `"NONE"` if no
candidate is correct). Writes a checkpoint after every row so long
jobs are resumable.

```bash
export OPENAI_API_KEY=sk-…
python annotate_with_llm.py \
    --input  candidates.csv \
    --output candidates_annotated.csv \
    --model  gpt-4o-mini \
    --max-candidates 20 \
    --skip-direct-matches
```

### `candidates_annotated.csv` output schema

Everything from `candidates.csv`, plus:

| column | description |
|---|---|
| `gold_ror_id` | ROR ID the LLM picked, or `"NONE"` |
| `gold_name` | canonical name of the gold pick |
| `gold_rank` | 1-based rank of gold inside `candidates_ror_ids` (0 = NONE, -1 = direct-match not in list) |
| `gold_in_topk` | `True` if the gold pick is in the candidate list |
| `llm_rationale` | one-line reason returned by the model |
| `llm_model` | model name used |
| `llm_status` | `ok`, `direct_match_short_circuit`, `skipped_no_candidates`, `api_error`, `parse_error` |

### Resuming

A checkpoint file is written next to the output CSV
(`candidates_annotated.csv.checkpoint.jsonl`). Rerun with `--resume`
to pick up where the previous run stopped. Annotated rows are replayed
from the checkpoint instead of hitting the API again.

### Hallucination guard

If the model returns a ROR ID that is not in the candidate list, the
script downgrades the row to `gold_ror_id = "NONE"` and
`llm_status = "parse_error"`, preserving the model's original reply in
`llm_rationale` for inspection. This prevents invented IDs from
polluting the gold set.

---

## Design notes

**Why one row per ORG/SUBORG, not per raw line?**
For multi-org affiliations like *"CNRS, Univ Montpellier, Montpellier,
France"* each org is a separate entity-linking decision. Splitting at
the entity level means the reranker is trained/evaluated on exactly
the unit it will score at inference time. The `raw_affiliation_string`
column preserves full context so the annotator (LLM or human) sees
disambiguating signals like *"School of Medicine"* or
*"Biomedical Research Centre"*.

**Why keep `direct_match_ror_id` in its own column instead of
collapsing to the final answer?**
The direct-match rule (exact `name_lower` + `country_lower` → unique
ROR) is a deterministic short-circuit used in production. Keeping its
output separate lets Script 2 either trust it (`--skip-direct-matches`)
or have the LLM verify it (default). Either way, the dataset records
what the rule would have done — useful for ablations.

**Why two format columns (`affilgood_string` vs `retrieval_query`)?**
Because the retriever and the reranker want different input formats.
The retriever needs whatever it was trained on (tagged for
`affilgood_dense`); the reranker wants natural text. Storing both
means you can reproduce retrieval exactly *and* hand the plain version
straight to a reranker without format surgery.

**Why OpenAI JSON mode instead of a strict schema?**
JSON mode is supported across `gpt-4o`, `gpt-4o-mini`, and compatible
providers (Azure, local proxies, OpenRouter, etc.). The schema is two
fields (`ror_id`, `rationale`); strict enforcement adds little. Swap
to `response_format={"type": "json_schema", …}` if you need it on
newer-generation models.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `TypeError: unsupported operand type(s) for \|: 'type' and 'type'` at import time | You're on Python 3.9. Upgrade to 3.10+ via uv: `uv venv --python 3.11`. |
| `update_ror_registry.py` fails with HTTP 403 or 429 | Pass `--api-token <zenodo-token>` or wait a minute and retry. |
| `update_ror_registry.py` times out on slow networks | Increase `--timeout 300` or use the manual-download fallback. |
| `prepare_candidates.py` produces 0 entity rows | Check `note` column: `no_spans_detected` means the span model produced nothing (possibly non-UTF-8 input); `no_org_entity_in_span` means NER found no ORG/SUBORG. |
| `--faiss-auto-download` fails with `ImportError: No module named 'data_manager'` | Copy `data_manager.py` from the affilgood project into your `--src-path`, or drop the flag and pass `--faiss-dir ~/.cache/affilgood/v2.0.0/ror/dense` after downloading the bundle manually. |
| Retrieval scores look nonsense after loading FAISS | Model mismatch: check `faiss_meta.json` — the query encoder in your `--retriever` config must be the same model that encoded the KB. |
| Retrieval is slow | Default batch-size 16 is conservative; try `--batch-size 64` on a decent GPU. |
| OpenAI cost is too high | Use `--model gpt-4o-mini` (default), enable `--skip-direct-matches`, or reduce `--max-candidates` to 10. |
| `annotate_with_llm.py` returns `parse_error` on many rows | The model is likely hitting a rate limit. Lower concurrency (this script is single-threaded by design) or switch model. |