"""
prepare_candidates.py
=====================
Script 1 of the reranker annotation pipeline.

Input
-----
A plain .txt file with one raw affiliation string per line.

Output
------
A CSV with **one row per ORG/SUBORG entity** detected in each affiliation:

    row_id              — sequential id (int)
    raw_line_id         — index of the source line in the input .txt
    raw_affiliation_string — full original line (context for the annotator)
    span_text           — the sub-affiliation span containing this entity
    entity              — the ORG/SUBORG text extracted by NER
    affilgood_string    — primary tagged query used for retrieval in the
                          caller's model-agnostic format. **Plain, comma-
                          separated**: "University of Maryland, Baltimore, USA".
                          This is the reranker-ready format (matches the
                          AffilGood paper's concatenation scheme and what every
                          cross-encoder reranker in this codebase was pretrained
                          on). To see what the *retriever* actually received
                          (which may be tagged, e.g. for affilgood_dense), look
                          at the `retrieval_query` column.
    all_variants        — "||"-joined list of every variant in plain format
    retrieval_query     — the primary query string that was actually sent to
                          the retriever (tagged for affilgood_dense, plain
                          otherwise) — kept for reproducibility/debugging
    direct_match_ror_id — exact (name+country) match; empty string if none
    direct_match_name   — canonical name of the direct-match record (readability)
    candidates_ror_ids  — JSON list of top-K ROR IDs sorted by retrieval score ↓
    candidates_names    — JSON list (same order) of canonical names
    candidates_cities   — JSON list (same order) of cities
    candidates_countries — JSON list (same order) of countries
    candidates_scores   — JSON list (same order) of dense similarity scores
    n_spans             — how many spans the SpanIdentifier cut the raw line into
    n_entities_in_span  — how many ORG/SUBORG entities in this span

Pipeline
--------
For each raw affiliation line:
    1.  SpanIdentifier (affilgood-span-multilingual-v2) → N sub-affiliation spans
    2.  NER (affilgood-ner-multilingual-v2) → {ORG, SUBORG, CITY, COUNTRY, …}
        per span
    3.  For each ORG/SUBORG entity in each span:
        a. Build geo-variants via build_queries_from_ner_dict()
        b. Direct match on (name_lower, country_lower) exact lookup
        c. Dense retrieval with SIRIS-Lab/affilgood-dense-retriever, merged
           over all variants by max-score → top-K
        d. Emit one output row

Usage
-----
    # Standard path (encodes KB from scratch, ~15 min on CPU):
    python prepare_candidates.py \
        --input affils.txt \
        --output candidates.csv \
        --ror-records data/registry/ror/ror_records.jsonl \
        --top-k 20 \
        --device cuda

    # Fast path: use the precomputed FAISS index from the AffilGood
    # data bundle (~330 MB compressed on first download, then cached):
    python prepare_candidates.py \
        --input affils.txt \
        --output candidates.csv \
        --faiss-auto-download \
        --top-k 20 \
        --device cuda

    # Or point at a manual path:
    python prepare_candidates.py \
        --input affils.txt \
        --output candidates.csv \
        --faiss-dir ~/.cache/affilgood/v2.0.0/ror/dense \
        --top-k 20

Run from the repo root so that `src/` is importable, OR adjust --src-path.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build candidate lists for reranker annotation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="Input .txt file (one raw affiliation per line).")
    p.add_argument("--output", "-o", required=True,
                   help="Output CSV path.")
    p.add_argument("--ror-records",
                   default="data/registry/ror/ror_records.jsonl",
                   help="Path to normalized ROR records (JSONL). If missing, "
                        "will fall back to RegistryManager to download.")
    p.add_argument("--src-path", default="src",
                   help="Path to the repo's src/ directory.")
    p.add_argument("--top-k", type=int, default=20,
                   help="Number of candidates to retrieve per entity.")
    p.add_argument("--device", default=None, choices=[None, "cpu", "cuda"],
                   help="Device for models (default: auto).")
    p.add_argument("--batch-size", type=int, default=16,
                   help="Batch size for span / NER / retriever.")
    p.add_argument("--retriever", default="affilgood_dense",
                   help="Retriever config name (see RETRIEVER_CONFIGS).")
    p.add_argument("--faiss-dir", default=None,
                   help="Use a precomputed FAISS index instead of encoding the "
                        "KB from scratch. Points to a directory containing "
                        "faiss.index, faiss_ids.json, faiss_texts.json, and "
                        "faiss_meta.json (the AffilGood data-manager layout). "
                        "Leave AFFILGOOD_DATA_DIR set or pass an explicit path "
                        "like ~/.cache/affilgood/v2.0.0/ror/dense/. When set, "
                        "the --ror-records argument MUST point to the bundled "
                        "ror_records.jsonl that ships with the same data "
                        "release (the FAISS row-ids are keyed to it).")
    p.add_argument("--faiss-auto-download", action="store_true",
                   help="If --faiss-dir is not provided, call the AffilGood "
                        "data_manager.ensure_data() to download and cache the "
                        "bundled v2.0.0 data (FAISS index + ror_records). "
                        "Requires a local copy of affilgood's data_manager.py "
                        "on --src-path (or anywhere on sys.path).")
    p.add_argument("--enrich-from", default=None,
                   help="Optional path to a richer ROR JSONL (or v2 JSON dump) "
                        "used to populate candidate names/cities/countries in "
                        "the output CSV, in case --ror-records is stripped "
                        "down (as the AffilGood FAISS bundle sometimes is). "
                        "If not given, enrichment comes from --ror-records.")
    p.add_argument("--ner-min-score", type=float, default=0.1,
                   help="Minimum score threshold for NER tokens.")
    p.add_argument("--include-suborg", action="store_true", default=True,
                   help="Also emit rows for SUBORG entities "
                        "(build_queries_from_ner_dict already does this).")
    p.add_argument("--limit", type=int, default=None,
                   help="Process only the first N input lines (for testing).")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Imports from the existing codebase (deferred so --help works)
# ------------------------------------------------------------------

def wire_imports(src_path: str):
    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"--src-path does not exist: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


# ------------------------------------------------------------------
# ROR loading
# ------------------------------------------------------------------

def _extract_display_fields(d: dict) -> Dict[str, str]:
    """
    Extract (name, city, country) from a ROR-ish dict, being lenient
    about schema. Handles:
      - The project's normalized RegistryRecord layout (flat keys)
      - The raw ROR v2 dump (names[] array, locations[] array)
      - Intermediate layouts where some fields are present and some aren't
    Empty strings are returned for anything we can't find.
    """
    name = d.get("name") or ""
    city = d.get("city") or ""
    country = d.get("country") or ""

    # ── ROR v2 fallback: names[] with types including 'ror_display' ──
    if not name and isinstance(d.get("names"), list):
        for n in d["names"]:
            if isinstance(n, dict) and "ror_display" in (n.get("types") or []):
                name = n.get("value", "") or name
                break
        # Final fallback: first name value
        if not name and d["names"]:
            first = d["names"][0]
            if isinstance(first, dict):
                name = first.get("value", "")

    # ── ROR v2 fallback: locations[].geonames_details ──
    if (not city or not country) and isinstance(d.get("locations"), list) and d["locations"]:
        loc0 = d["locations"][0]
        if isinstance(loc0, dict):
            # v2 location schema
            geo = loc0.get("geonames_details") or {}
            if isinstance(geo, dict):
                city    = city    or (geo.get("name") or "")
                country = country or (geo.get("country_name") or "")
            # Some bundles flatten this one level up
            city    = city    or (loc0.get("city") or loc0.get("name") or "")
            country = country or (loc0.get("country") or loc0.get("country_name") or "")

    # ── ROR v1 fallback: flat addresses[] array ──
    if not city and isinstance(d.get("addresses"), list) and d["addresses"]:
        addr0 = d["addresses"][0]
        if isinstance(addr0, dict):
            city = addr0.get("city", "") or city

    # ── Country v1 fallback: top-level "country" object ──
    if not country and isinstance(d.get("country"), dict):
        country = d["country"].get("country_name", "") or ""

    return {"name": name, "city": city, "country": country}


def build_id2record(enrichment_path: Path,
                    ror_records) -> Dict[str, Dict]:
    """
    Build `ror_id → {name, city, country}` for CSV enrichment.

    Strategy:
      1. Start from the normalized RegistryRecord list (fast, correct
         when fields are populated).
      2. For any record with an empty city/country, re-parse the raw
         JSON at `enrichment_path` to recover them.
      3. If the JSONL only contains ROR IDs (nothing else), we fall
         back to whatever we already have — candidate names/cities/
         countries will just be blank for those rows.

    This makes the pipeline tolerant of "stripped" bundles like the
    AffilGood FAISS package, which may ship a minimal JSONL.
    """
    id2record: Dict[str, Dict] = {}

    # Pass 1: normalized records
    n_missing = 0
    for rec in ror_records:
        rid = rec.id.replace("https://ror.org/", "").strip()
        entry = {
            "name":    rec.name or "",
            "city":    rec.city or "",
            "country": rec.country or "",
        }
        if not entry["city"] and not entry["country"]:
            n_missing += 1
        id2record[rid] = entry

    # Pass 2: re-parse the raw JSON for missing fields
    if n_missing and enrichment_path.exists():
        print(f"  → {n_missing:,} records lack city/country; re-parsing "
              f"{enrichment_path.name} for fallback fields")
        recovered = 0
        with enrichment_path.open("r", encoding="utf-8") as f:
            # Try JSONL first (one object per line)
            first_char = f.read(1)
            f.seek(0)
            if first_char == "[":
                # JSON array (raw ROR dump)
                data = json.load(f)
                iterator = iter(data)
            else:
                iterator = (json.loads(ln) for ln in f if ln.strip())

            for d in iterator:
                try:
                    raw_id = d.get("id", "")
                    rid = raw_id.replace("https://ror.org/", "").strip()
                    if not rid:
                        continue
                    entry = id2record.get(rid)
                    if entry is None or (entry["city"] and entry["country"]):
                        continue
                    fields = _extract_display_fields(d)
                    if not entry["name"]    and fields["name"]:
                        entry["name"] = fields["name"]
                    if not entry["city"]    and fields["city"]:
                        entry["city"] = fields["city"]; recovered += 1
                    if not entry["country"] and fields["country"]:
                        entry["country"] = fields["country"]
                except Exception:
                    continue
        if recovered:
            print(f"  ✅ Recovered city/country for {recovered:,} records")
        else:
            print(f"  ⚠ Could not recover any city/country from "
                  f"{enrichment_path.name} (schema might lack geographic data)")

    return id2record


def load_ror_records(path: str):
    """
    Load normalized ROR records from JSONL. Fall back to RegistryManager
    (which will download/normalize a fresh dump) if the file is missing.
    """
    from registry import RegistryManager, RegistryRecord  # noqa: F401
    p = Path(path)
    if p.exists():
        print(f"  → Loading ROR records from {p}")
        records = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                rec = RegistryRecord.from_dict(d)
                if rec.status == "active":
                    records.append(rec)
        print(f"  ✅ Loaded {len(records):,} active ROR records")
        return records

    print(f"  ⚠ {p} not found — falling back to RegistryManager download")
    data_dir = p.parent.parent  # .../data/registry/ror/x.jsonl → .../data/registry
    mgr = RegistryManager(data_dir=str(data_dir), verbose=True)
    return mgr.get_records(registry="ror", active_only=True)


# ------------------------------------------------------------------
# Span + NER  (inlined from run_affilel_experiments.py so we don't
# trigger that file's module-level side effects — it loads datasets
# and initializes models at import time, which we don't want here.)
# ------------------------------------------------------------------

DEFAULT_SPAN_MODEL = "nicolauduran45/affilgood-span-multilingual-v2"
DEFAULT_NER_MODEL  = "nicolauduran45/affilgood-ner-multilingual-v2"


class SpanIdentifier:
    """Splits a raw affiliation string into sub-affiliation spans."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 batch_size: int = 32, verbose: bool = False,
                 min_score: float = 0.0, fix_words: bool = True,
                 merge_spans: bool = True):
        self.model_path = model_path or DEFAULT_SPAN_MODEL
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.min_score = min_score
        self.fix_words_enabled = fix_words
        self.merge_spans_enabled = merge_spans
        self._pipeline = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline
            if self.verbose:
                print(f"[Span] Loading model: {self.model_path}")
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_path,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            self._available = True
        except Exception as e:
            if self.verbose:
                print(f"[Span] Model unavailable, using noop: {e}")
            self._pipeline = None
            self._available = False

    def identify_spans(self, items, batch_size=None):
        batch_size = batch_size or self.batch_size
        results, texts = [], []
        for item in items:
            out = dict(item)
            out["span_entities"] = []
            results.append(out)
            texts.append(item.get("raw_text", ""))

        if not self._available:
            for out in results:
                text = out.get("raw_text", "")
                out["span_entities"] = [text] if text else []
            return results

        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset
            dataset = Dataset.from_dict({"text": texts})
            outputs = list(self._pipeline(KeyDataset(dataset, "text"), batch_size=batch_size))
        except Exception as e:
            if self.verbose:
                print(f"[Span] Inference failed, using fallback: {e}")
            for out in results:
                text = out.get("raw_text", "")
                out["span_entities"] = [text] if text else []
            return results

        for out, raw_text, entities in zip(results, texts, outputs):
            spans = entities
            if self.fix_words_enabled:
                spans = self._fix_words(raw_text, spans)
            if self.merge_spans_enabled:
                spans = self._clean_and_merge_spans(spans, min_score=self.min_score)
            span_entities = [e.get("word", "") for e in spans if e.get("word")]
            if not span_entities and raw_text:
                span_entities = [raw_text]
            out["span_entities"] = span_entities
        return results

    @staticmethod
    def _fix_words(raw_text, entities):
        for e in entities:
            try:
                s, t = e.get("start"), e.get("end")
                if s is None or t is None:
                    continue
                e["word"] = raw_text[s:t]
            except Exception:
                continue
        return entities

    @staticmethod
    def _clean_and_merge_spans(entities, min_score=0.0):
        entities = [e for e in entities if e.get("score", 0) >= min_score]
        merged, i = [], 0
        while i < len(entities):
            cur = entities[i]
            if i + 1 < len(entities):
                nxt = entities[i + 1]
                try:
                    if cur.get("end") == nxt.get("start") and nxt.get("word") and nxt["word"][0].islower():
                        merged.append({
                            "entity_group": cur.get("entity_group"),
                            "score": min(cur.get("score", 0), nxt.get("score", 0)),
                            "word":  cur.get("word", "") + nxt.get("word", ""),
                            "start": cur.get("start"),
                            "end":   nxt.get("end"),
                        })
                        i += 2
                        continue
                except Exception:
                    pass
            merged.append(cur)
            i += 1
        return merged


class NER:
    """Extracts ORG / SUBORG / CITY / COUNTRY / … entities from each span."""

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu",
                 batch_size: int = 32, verbose: bool = False,
                 fix_words: bool = True, merge_entities: bool = True,
                 min_score: float = 0.0):
        self.model_path = model_path or DEFAULT_NER_MODEL
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.fix_words_enabled = fix_words
        self.merge_entities_enabled = merge_entities
        self.min_score = min_score
        self._pipeline = None
        self._available = False
        self._load_model()

    def _load_model(self):
        try:
            from transformers import pipeline
            if self.verbose:
                print(f"[NER] Loading model: {self.model_path}")
            self._pipeline = pipeline(
                "token-classification",
                model=self.model_path,
                aggregation_strategy="simple",
                device=0 if self.device == "cuda" else -1,
            )
            self._available = True
        except Exception as e:
            if self.verbose:
                print(f"[NER] Model unavailable, using noop: {e}")
            self._pipeline = None
            self._available = False

    def recognize_entities(self, items, batch_size=None):
        batch_size = batch_size or self.batch_size
        flat_spans, span_map = [], []
        for item_idx, item in enumerate(items):
            for span_idx, span in enumerate(item.get("span_entities", [])):
                flat_spans.append(span)
                span_map.append((item_idx, span_idx))

        results = []
        for item in items:
            out = dict(item)
            out["ner"]     = [{} for _ in item.get("span_entities", [])]
            out["ner_raw"] = [[] for _ in item.get("span_entities", [])]
            results.append(out)

        if not flat_spans or not self._available:
            return results

        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset
            dataset = Dataset.from_dict({"text": flat_spans})
            outputs = list(self._pipeline(KeyDataset(dataset, "text"), batch_size=batch_size))
        except Exception as e:
            if self.verbose:
                print(f"[NER] Inference failed: {e}")
            return results

        for output, (item_idx, span_idx), span_text in zip(outputs, span_map, flat_spans):
            raw = output
            if self.fix_words_enabled:
                raw = self._fix_words(span_text, raw)
            if self.merge_entities_enabled:
                raw = self._clean_and_merge_entities(raw, min_score=self.min_score)
            structured = self._group_entities(raw)
            results[item_idx]["ner"][span_idx] = structured
            results[item_idx]["ner_raw"][span_idx] = raw

        return results

    @staticmethod
    def _group_entities(raw_entities):
        grouped = {}
        for ent in raw_entities:
            label = ent.get("entity_group")
            text  = ent.get("word")
            if not label or not text:
                continue
            grouped.setdefault(label, []).append(text)
        return grouped

    @staticmethod
    def _fix_words(raw_text, entities):
        for e in entities:
            try:
                s, t = e.get("start"), e.get("end")
                if s is None or t is None:
                    continue
                txt = raw_text[s:t]
                # Heal trailing unclosed parentheses (e.g. "CNRS (UMR" → "CNRS (UMR 5297)")
                last_open  = txt.rfind("(")
                last_close = txt.rfind(")")
                if last_open > -1 and (last_close == -1 or last_open > last_close):
                    nxt_close = raw_text.find(")", t)
                    if nxt_close > -1:
                        between = raw_text[t:nxt_close]
                        if not any(d in between for d in [" ", ",", ";", ":", ".", "\n", "\t"]):
                            e["end"] = nxt_close + 1
                            txt = raw_text[s:nxt_close + 1]
                e["word"] = txt
            except Exception:
                continue
        return entities

    @staticmethod
    def _clean_and_merge_entities(entities, min_score=0.0):
        entities = [e for e in entities if e.get("score", 0) >= min_score]
        merged, i = [], 0
        while i < len(entities):
            cur = entities[i]
            if i + 1 < len(entities):
                nxt = entities[i + 1]
                try:
                    if (cur.get("end") == nxt.get("start")
                        and nxt.get("word")
                        and (nxt["word"][0].islower() or nxt["word"][0].isdigit())):
                        merged.append({
                            "entity_group": cur.get("entity_group"),
                            "score": min(cur.get("score", 0), nxt.get("score", 0)),
                            "word":  cur.get("word", "") + nxt.get("word", ""),
                            "start": cur.get("start"),
                            "end":   nxt.get("end"),
                        })
                        i += 2
                        continue
                except Exception:
                    pass
            merged.append(cur)
            i += 1
        return merged


# ------------------------------------------------------------------
# FAISS-backed retriever (skips KB encoding; uses precomputed embeddings)
# ------------------------------------------------------------------

class FaissDenseRetriever:
    """
    Drop-in replacement for `DenseRetriever` that loads a precomputed
    FAISS index instead of encoding the KB from scratch.

    Why this exists
    ---------------
    The SIRIS-Lab AffilGood data release bundles a FAISS index built
    from the affilgood-dense-retriever over the full ROR v2 KB. Using
    it avoids ~15 minutes of KB encoding at startup.

    Expected layout (the AffilGood data-manager default):
        <faiss_dir>/
            faiss.index        — faiss.Index (IndexFlatIP, normalized)
            faiss_ids.json     — list[str]   mapping row index → ROR ID
            faiss_texts.json   — list[str]   (optional; for debugging)
            faiss_meta.json    — dict        (optional; model_name, schema_v, …)

    Constraint
    ----------
    The FAISS row-IDs are keyed to whatever ROR dump was used to BUILD
    the index. To avoid silent divergence between retrieval hits and
    the direct-match index, `ror_records.jsonl` MUST come from the same
    release (i.e. use the one that shipped alongside the FAISS files).

    Interface
    ---------
    Matches BaseRetriever: has `.fit(...)` (which no-ops for KB-side
    work but stores `kb_ids`), `.retrieve(query, k=10)` returning
    `[(ror_id, score), ...]`, and `.free()`.
    """

    def __init__(self, faiss_dir: str | Path, model_name: str,
                 query_prefix: str = "", device: Optional[str] = None):
        import faiss  # type: ignore
        import torch
        from sentence_transformers import SentenceTransformer

        faiss_dir = Path(faiss_dir).expanduser().resolve()
        if not faiss_dir.is_dir():
            raise FileNotFoundError(f"--faiss-dir does not exist: {faiss_dir}")

        index_path = faiss_dir / "faiss.index"
        ids_path   = faiss_dir / "faiss_ids.json"
        meta_path  = faiss_dir / "faiss_meta.json"

        for p in (index_path, ids_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Missing file in --faiss-dir: {p.name}\n"
                    f"  expected layout: faiss.index, faiss_ids.json, "
                    f"faiss_texts.json (optional), faiss_meta.json (optional)"
                )

        # --- Load the FAISS index ---
        print(f"  → Loading FAISS index: {index_path}")
        self.index = faiss.read_index(str(index_path))
        self.dim = self.index.d
        self.n_vectors = self.index.ntotal
        print(f"  ✅ FAISS index: {self.n_vectors:,} vectors, dim={self.dim}")

        # --- Load row-index → ROR ID mapping ---
        with open(ids_path, "r", encoding="utf-8") as f:
            raw_kb_ids: List[str] = json.load(f)
        if len(raw_kb_ids) != self.n_vectors:
            raise ValueError(
                f"faiss_ids.json length ({len(raw_kb_ids):,}) != "
                f"index.ntotal ({self.n_vectors:,}). The files are out of sync."
            )

        # Normalize: the AffilGood bundle stores full URLs
        # ('https://ror.org/04jr1s231'), but the rest of this codebase —
        # including RegistryRecord.from_dict, build_direct_match_index, and
        # the id2record lookup built from ror_records.jsonl — uses the
        # stripped form ('04jr1s231'). Strip once at load time so downstream
        # code never has to care.
        self.kb_ids: List[str] = [
            rid.replace("https://ror.org/", "").strip() for rid in raw_kb_ids
        ]
        n_urls = sum(1 for r in raw_kb_ids if r.startswith("https://"))
        if n_urls:
            print(f"  → Stripped 'https://ror.org/' prefix from "
                  f"{n_urls:,}/{self.n_vectors:,} FAISS ids")

        # --- Inspect metadata (advisory only: we'll still load the model
        #     name the caller gave us, but warn if they disagree) ---
        if meta_path.exists():
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    self.meta = json.load(f)
                kb_model = self.meta.get("model_name") or self.meta.get("model")
                if kb_model and kb_model != model_name:
                    print(
                        f"  ⚠ Metadata reports the KB was encoded with "
                        f"'{kb_model}', but we're loading query encoder "
                        f"'{model_name}'. Retrieval scores will be "
                        f"meaningless if these don't match."
                    )
            except Exception as e:
                print(f"  ⚠ Could not read {meta_path.name}: {e}")
                self.meta = {}
        else:
            self.meta = {}

        # --- Load only the query encoder (KB side is already embedded) ---
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  → Loading query encoder: {model_name}  ({self.device})")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.qp = query_prefix
        self.model_name = model_name
        print(f"  ✅ Query encoder ready")

    def fit(self, kb_texts, kb_ids):
        """
        No-op for the KB side (already indexed). Kept for interface
        compatibility with BaseRetriever. If the caller passed different
        kb_ids, we warn but trust the FAISS-bundled ids.
        """
        if kb_ids and list(kb_ids) != self.kb_ids:
            # Not fatal — but a strong signal that the user's ror_records.jsonl
            # is out of sync with the FAISS artifacts.
            n_overlap = len(set(kb_ids) & set(self.kb_ids))
            print(
                f"  ⚠ kb_ids mismatch: caller provided {len(kb_ids):,} ids, "
                f"FAISS has {len(self.kb_ids):,} ids, {n_overlap:,} overlap. "
                f"Using FAISS ids. Check that --ror-records points to the "
                f"bundled release."
            )

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        import numpy as np
        q = f"{self.qp}{query}" if self.qp else query
        q_emb = self.model.encode([q], normalize_embeddings=True)
        q_emb = np.asarray(q_emb, dtype="float32")
        scores, idxs = self.index.search(q_emb, k)
        # idxs / scores are shape (1, k)
        out = []
        for rank in range(idxs.shape[1]):
            row = int(idxs[0, rank])
            if row < 0 or row >= len(self.kb_ids):
                continue  # -1 padding when index has < k vectors
            out.append((self.kb_ids[row], float(scores[0, rank])))
        return out

    def free(self):
        import gc, torch
        if getattr(self, "model", None) is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if getattr(self, "index", None) is not None:
            del self.index
            self.index = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resolve_faiss_dir(args) -> Optional[Path]:
    """
    Resolve where the FAISS artifacts live based on CLI flags.

    Priority:
      1. --faiss-dir (explicit path)
      2. --faiss-auto-download → call AffilGood's data_manager.ensure_data()
      3. None → caller falls back to encoding the KB from scratch
    """
    if args.faiss_dir:
        return Path(args.faiss_dir).expanduser().resolve()

    if args.faiss_auto_download:
        try:
            from data_manager import ensure_data  # type: ignore
        except ImportError as e:
            raise ImportError(
                "--faiss-auto-download set but data_manager.py is not on "
                "sys.path. Put affilgood's data_manager.py in --src-path "
                "(or install the affilgood package)."
            ) from e
        print("  → Calling data_manager.ensure_data() to fetch v2.0.0 bundle …")
        data_dir = ensure_data()  # returns Path, e.g. ~/.cache/affilgood/v2.0.0
        return Path(data_dir) / "ror" / "dense"

    return None




def run_span_and_ner(lines: List[str], device: str, batch_size: int,
                     ner_min_score: float) -> List[Dict]:
    """
    Return a list of dicts, one per input line:
      {
        "raw": <original line>,
        "spans": [
           {"text": <span_text>, "ner": {"ORG": [...], "CITY": [...], ...}},
           ...
        ]
      }
    """
    import gc
    import torch

    def _flush():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 1. Span identification (run on CPU; model is tiny) ---
    print("  → Loading span identifier …")
    span_model = SpanIdentifier(device="cpu", batch_size=batch_size, verbose=False)

    span_items = [{"row_id": i, "raw_text": t} for i, t in enumerate(lines)]
    span_results = span_model.identify_spans(span_items, batch_size=batch_size)

    # Map back: line_idx → list[str] of spans
    spans_per_line: List[List[str]] = [[] for _ in lines]
    for item, res in zip(span_items, span_results):
        spans_per_line[item["row_id"]] = res.get("span_entities", []) or []
        # Fallback: if no spans detected, use the whole raw text
        if not spans_per_line[item["row_id"]] and lines[item["row_id"]].strip():
            spans_per_line[item["row_id"]] = [lines[item["row_id"]]]

    del span_model
    _flush()

    # --- 2. NER on each span ---
    print("  → Loading NER model …")
    ner_model = NER(device=device or "cpu", batch_size=batch_size,
                    min_score=ner_min_score, verbose=False)

    ner_items = [
        {"row_id": i, "span_entities": spans_per_line[i]}
        for i in range(len(lines))
        if spans_per_line[i]
    ]
    ner_results = ner_model.recognize_entities(ner_items, batch_size=batch_size)

    # Assemble output
    ner_by_line: Dict[int, List[Dict]] = {}
    for item, res in zip(ner_items, ner_results):
        spans_with_ner = []
        span_texts = item["span_entities"]
        for span_txt, ner_dict in zip(span_texts, res["ner"]):
            spans_with_ner.append({"text": span_txt, "ner": ner_dict or {}})
        ner_by_line[item["row_id"]] = spans_with_ner

    del ner_model
    _flush()

    out = []
    for i, raw in enumerate(lines):
        out.append({
            "raw": raw,
            "spans": ner_by_line.get(i, []),
        })
    return out


# ------------------------------------------------------------------
# Main per-entity loop
# ------------------------------------------------------------------

def build_rows(ner_output: List[Dict],
               retriever, ret_cfg,
               dm_index: Dict,
               id2record: Dict[str, Dict],
               top_k: int) -> List[Dict]:
    """
    Iterate over every ORG/SUBORG entity in every span, run retrieval +
    direct match, and emit one output row per entity.
    """
    from ror_retrieval_experiments import (
        build_queries_from_ner_dict,
        direct_match_for_entity,
        format_query_tagged,
        format_query_plain,
    )

    # Two formatters, two jobs:
    #   - fmt_retrieval : what we *query the retriever with* (tagged if the
    #     retriever was trained on tagged KB text, e.g. affilgood_dense).
    #   - fmt_stored    : what we *store in the CSV* for downstream reranker
    #     consumption — always plain, comma-separated, natural text.
    #     Rerankers (jina-reranker-v2, ms-marco cross-encoders, the cometadata
    #     fine-tunes on top of them, Qwen3-Reranker) were all pre-trained on
    #     natural text; bracket tags like [MENTION]/[CITY] are tokenized as
    #     noise and hurt scoring.
    fmt_retrieval = format_query_tagged if ret_cfg["query_format"] == "tagged" else format_query_plain
    fmt_stored    = format_query_plain
    internal_k = top_k * 3  # over-fetch because expanded KB has dup ror_ids

    rows: List[Dict] = []
    row_id = 0

    for line_idx, item in enumerate(tqdm(ner_output, desc="  retrieval", leave=False)):
        raw = item["raw"]
        spans = item["spans"]
        n_spans = len(spans)

        if not spans:
            # No spans detected at all → emit a placeholder row so the user
            # sees which inputs produced nothing.
            rows.append({
                "row_id": row_id,
                "raw_line_id": line_idx,
                "raw_affiliation_string": raw,
                "span_text": "",
                "entity": "",
                "affilgood_string": "",
                "all_variants": "",
                "retrieval_query": "",
                "direct_match_ror_id": "",
                "direct_match_name": "",
                "candidates_ror_ids": "[]",
                "candidates_names": "[]",
                "candidates_cities": "[]",
                "candidates_countries": "[]",
                "candidates_scores": "[]",
                "n_spans": 0,
                "n_entities_in_span": 0,
                "note": "no_spans_detected",
            })
            row_id += 1
            continue

        for span in spans:
            span_text = span["text"]
            ner_dict = span["ner"]
            entity_queries = build_queries_from_ner_dict(ner_dict)
            n_ents = len(entity_queries)

            if not entity_queries:
                rows.append({
                    "row_id": row_id,
                    "raw_line_id": line_idx,
                    "raw_affiliation_string": raw,
                    "span_text": span_text,
                    "entity": "",
                    "affilgood_string": "",
                    "all_variants": "",
                    "retrieval_query": "",
                    "direct_match_ror_id": "",
                    "direct_match_name": "",
                    "candidates_ror_ids": "[]",
                    "candidates_names": "[]",
                    "candidates_cities": "[]",
                    "candidates_countries": "[]",
                    "candidates_scores": "[]",
                    "n_spans": n_spans,
                    "n_entities_in_span": 0,
                    "note": "no_org_entity_in_span",
                })
                row_id += 1
                continue

            for eq in entity_queries:
                entity = eq["entity"]
                variants = eq["variants"]

                # Two parallel renderings of the same variants list.
                variant_strs_retrieval = [fmt_retrieval(v) for v in variants]
                variant_strs_stored    = [fmt_stored(v)    for v in variants]

                # The "primary" affilgood_string is the most specific variant:
                # the construction order in build_queries_from_ner_dict
                # puts ORG+CITY+COUNTRY first when available.
                primary_stored    = variant_strs_stored[0]    if variant_strs_stored    else ""
                primary_retrieval = variant_strs_retrieval[0] if variant_strs_retrieval else ""

                # --- Direct match ---
                dm_rid = direct_match_for_entity(eq, dm_index) or ""
                dm_name = id2record.get(dm_rid, {}).get("name", "") if dm_rid else ""

                # --- Dense retrieval, merge over variants (tagged format) ---
                score_map: Dict[str, float] = {}
                for vstr in variant_strs_retrieval:
                    for rid, sc in retriever.retrieve(vstr, k=internal_k):
                        if rid not in score_map or sc > score_map[rid]:
                            score_map[rid] = sc
                ranked = sorted(score_map.items(), key=lambda x: -x[1])[:top_k]

                cand_ids    = [rid for rid, _ in ranked]
                cand_scores = [round(float(sc), 6) for _, sc in ranked]
                cand_names  = [id2record.get(rid, {}).get("name", "")    for rid in cand_ids]
                cand_cities = [id2record.get(rid, {}).get("city", "")    for rid in cand_ids]
                cand_countries = [id2record.get(rid, {}).get("country", "") for rid in cand_ids]

                rows.append({
                    "row_id": row_id,
                    "raw_line_id": line_idx,
                    "raw_affiliation_string": raw,
                    "span_text": span_text,
                    "entity": entity,
                    # Plain comma-separated: "University of Maryland, Baltimore, USA"
                    # — reranker-ready, matches the AffilGood paper's concatenation format.
                    "affilgood_string": primary_stored,
                    "all_variants": "||".join(variant_strs_stored),
                    # Kept for reproducibility: the exact query string that was
                    # sent to the retriever (tagged for affilgood_dense, plain
                    # for everything else).
                    "retrieval_query": primary_retrieval,
                    "direct_match_ror_id": dm_rid,
                    "direct_match_name": dm_name,
                    "candidates_ror_ids":   json.dumps(cand_ids,    ensure_ascii=False),
                    "candidates_names":     json.dumps(cand_names,  ensure_ascii=False),
                    "candidates_cities":    json.dumps(cand_cities, ensure_ascii=False),
                    "candidates_countries": json.dumps(cand_countries, ensure_ascii=False),
                    "candidates_scores":    json.dumps(cand_scores, ensure_ascii=False),
                    "n_spans": n_spans,
                    "n_entities_in_span": n_ents,
                    "note": "",
                })
                row_id += 1

    return rows


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()
    wire_imports(args.src_path)

    # Deferred imports (after sys.path is wired)
    from ror_retrieval_experiments import (
        prepare_kb, build_retriever, build_direct_match_index,
        RETRIEVER_CONFIGS,
    )

    # --- 1. Read input lines ---
    with open(args.input, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f if ln.strip()]
    if args.limit:
        lines = lines[: args.limit]
    print(f"\n📥 {len(lines):,} input lines from {args.input}")

    # --- 2. Resolve FAISS artifacts (if requested) + paired ROR jsonl ---
    faiss_dir = resolve_faiss_dir(args)
    ror_records_path = args.ror_records
    if faiss_dir is not None:
        # The AffilGood bundle layout places ror_records.jsonl at
        # <cache>/ror/ror_records.jsonl and dense/ one level deeper.
        bundled_jsonl = faiss_dir.parent / "ror_records.jsonl"
        user_default = args.ror_records == "data/registry/ror/ror_records.jsonl"
        if bundled_jsonl.exists() and user_default:
            ror_records_path = str(bundled_jsonl)
            print(f"  → Using ROR records bundled with FAISS: {ror_records_path}")
        elif not user_default:
            print(
                f"  ⚠ --ror-records was set explicitly to {args.ror_records}. "
                f"Make sure this is the file that shipped with your FAISS "
                f"bundle (the FAISS row-ids must match exactly)."
            )

    # --- 3. Load ROR ---
    ror_records = load_ror_records(ror_records_path)

    # --- 4. Build KB + direct-match index + id→record lookup ---
    print("\n🏗️  Preparing KB …")
    df_ror_kb = prepare_kb(ror_records)
    print("🏗️  Building direct-match index …")
    dm_index = build_direct_match_index(df_ror_kb, ror_records=ror_records)

    # One entry per canonical record, for pretty-printing candidates.
    id2record: Dict[str, Dict] = {}
    for rec in ror_records:
        rid = rec.id.replace("https://ror.org/", "").strip()
        id2record[rid] = {
            "name":    rec.name,
            "city":    rec.city or "",
            "country": rec.country or "",
        }

    # --- 5. Span + NER ---
    print("\n🧩 Running span identifier + NER on input lines …")
    ner_output = run_span_and_ner(
        lines,
        device=args.device,
        batch_size=args.batch_size,
        ner_min_score=args.ner_min_score,
    )

    # --- 6. Build retriever ---
    if faiss_dir is not None:
        # FAISS-backed path: skip KB encoding, load the precomputed index.
        print(f"\n🔎 Using FAISS index at {faiss_dir}")
        ret_cfg = RETRIEVER_CONFIGS[args.retriever]
        if ret_cfg["class"] != "DenseRetriever":
            raise ValueError(
                f"--faiss-dir only makes sense with a dense retriever "
                f"(config class 'DenseRetriever'), but '{args.retriever}' "
                f"is class '{ret_cfg['class']}'."
            )
        retriever = FaissDenseRetriever(
            faiss_dir=faiss_dir,
            model_name=ret_cfg["model_name"],
            query_prefix=ret_cfg.get("query_prefix", ""),
            device=args.device,
        )
        retriever.fit(
            kb_texts=None,
            kb_ids=df_ror_kb["ror_id"].tolist(),
        )
    else:
        # Standard path: encode the KB from scratch (~15 min on CPU).
        print(f"\n🔎 Building retriever '{args.retriever}' …")
        retriever, ret_cfg = build_retriever(
            args.retriever, device=args.device, batch_size=args.batch_size,
        )
        retriever.fit(
            df_ror_kb[ret_cfg["kb_field"]].tolist(),
            df_ror_kb["ror_id"].tolist(),
        )

    # --- 7. Generate rows ---
    print(f"\n🧮 Retrieving top-{args.top_k} candidates per entity …")
    rows = build_rows(
        ner_output=ner_output,
        retriever=retriever,
        ret_cfg=ret_cfg,
        dm_index=dm_index,
        id2record=id2record,
        top_k=args.top_k,
    )

    try:
        retriever.free()
    except Exception:
        pass

    # --- 8. Write output ---
    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    n_total = len(df)
    n_entity = int((df["entity"] != "").sum())
    n_dm = int((df["direct_match_ror_id"] != "").sum())
    print(f"\n💾 Wrote {n_total:,} rows → {out}")
    print(f"   • rows with a detected ORG/SUBORG entity : {n_entity:,}")
    print(f"   • rows with a direct-match ROR ID         : {n_dm:,}  "
          f"({n_dm / max(n_entity, 1):.1%} of entity rows)")


if __name__ == "__main__":
    main()