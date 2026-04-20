import json
from pathlib import Path
import pandas as pd
import os, gc, json, time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict

# display(df_affro.head())

import re

def parse_label(label: str):
    """
    Extract (org_name, ror_id) pairs from label field
    """
    if not isinstance(label, str):
        return [], []

    pattern = r"(.+?)\s*\{https://ror\.org/([^\}]+)\}"
    matches = re.findall(pattern, label)

    names = [m[0].strip() for m in matches]
    rors = [m[1].strip() for m in matches]

    return names, rors


# print(df_all.shape)
#df_all.head()

from typing import List, Dict, Any, Optional

import pandas as pd
"""
Lightweight span identification component.

Responsibilities:
- Identify meaningful spans in affiliation strings
- Optionally use a ML model
- Always return a stable structure

This component is defensive and batch-oriented.
"""

from typing import List, Dict, Any, Optional

DEFAULT_SPAN_MODEL = "nicolauduran45/affilgood-span-multilingual-v2"


class SpanIdentifier:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        verbose: bool = False,
        min_score: float = 0.0,
        fix_words: bool = True,
        merge_spans: bool = True,
    ):
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

    # -------------------------------------------------
    # Model loading (safe)
    # -------------------------------------------------

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
                print(f"[Span] Model unavailable, using noop span identifier: {e}")
            self._pipeline = None
            self._available = False

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def identify_spans(
        self,
        items: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Identify spans for each input item.

        Always returns:
        - span_entities: list[str]
        """
        batch_size = batch_size or self.batch_size

        results = []
        texts = []

        for item in items:
            out = dict(item)
            out["span_entities"] = []
            results.append(out)
            texts.append(item.get("raw_text", ""))

        # No model → fallback to full text
        if not self._available:
            for out in results:
                text = out.get("raw_text", "")
                out["span_entities"] = [text] if text else []
            return results

        # -------------------------------------------------
        # Batched inference
        # -------------------------------------------------
        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset

            dataset = Dataset.from_dict({"text": texts})

            outputs = list(
                self._pipeline(
                    KeyDataset(dataset, "text"),
                    batch_size=batch_size,
                )
            )

        except Exception as e:
            if self.verbose:
                print(f"[Span] Inference failed, using fallback: {e}")
            for out in results:
                text = out.get("raw_text", "")
                out["span_entities"] = [text] if text else []
            return results

        # -------------------------------------------------
        # Post-process
        # -------------------------------------------------
        for out, raw_text, entities in zip(results, texts, outputs):
            spans = entities

            if self.fix_words_enabled:
                spans = self._fix_words(raw_text, spans)

            if self.merge_spans_enabled:
                spans = self._clean_and_merge_spans(
                    spans,
                    min_score=self.min_score,
                )

            span_entities = [
                ent.get("word", "")
                for ent in spans
                if ent.get("word")
            ]

            # Defensive fallback
            if not span_entities and raw_text:
                span_entities = [raw_text]

            out["span_entities"] = span_entities

        return results

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _fix_words(self, raw_text: str, entities: List[Dict[str, Any]]):
        for entity in entities:
            try:
                start, end = entity.get("start"), entity.get("end")
                if start is None or end is None:
                    continue
                entity["word"] = raw_text[start:end]
            except Exception:
                continue
        return entities

    def _clean_and_merge_spans(self, entities, min_score=0.0):
        entities = [e for e in entities if e.get("score", 0) >= min_score]

        merged = []
        i = 0

        while i < len(entities):
            current = entities[i]

            if i + 1 < len(entities):
                nxt = entities[i + 1]
                try:
                    if (
                        current.get("end") == nxt.get("start")
                        and nxt.get("word")
                        and nxt["word"][0].islower()
                    ):
                        merged.append({
                            "entity_group": current.get("entity_group"),
                            "score": min(current.get("score", 0), nxt.get("score", 0)),
                            "word": current.get("word", "") + nxt.get("word", ""),
                            "start": current.get("start"),
                            "end": nxt.get("end"),
                        })
                        i += 2
                        continue
                except Exception:
                    pass

            merged.append(current)
            i += 1

        return merged


DEFAULT_NER_MODEL = "nicolauduran45/affilgood-ner-multilingual-v2"

def df_to_span_items_affro(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Prepare items for SpanIdentifier from AffRoDB rows only.
    """
    items = []

    for idx, row in df.iterrows():
        if row.get("source") != "affro":
            continue

        raw = row.get("raw_affiliation_string")
        if not isinstance(raw, str) or not raw.strip():
            continue

        items.append({
            "row_id": idx,
            "raw_text": raw,
        })

    return items

def apply_span_identifier_affro(
    df: pd.DataFrame,
    span_model: SpanIdentifier,
    batch_size: int = 32,
    span_column: str = "span_entities",
) -> pd.DataFrame:
    """
    Apply SPAN identification to AffRoDB rows only.
    """

    df = df.copy()
    df[span_column] = None

    items = df_to_span_items_affro(df)
    if not items:
        return df

    results = span_model.identify_spans(
        items,
        batch_size=batch_size,
    )

    for item, res in zip(items, results):
        idx = item["row_id"]
        df.at[idx, span_column] = res.get("span_entities", [])

    return df

class NER:
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "cpu",
        batch_size: int = 32,
        verbose: bool = False,
        fix_words: bool = True,
        merge_entities: bool = True,
        min_score: float = 0.0,
    ):
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

    # -------------------------------------------------
    # Model loading (safe)
    # -------------------------------------------------

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
                print(f"[NER] Model unavailable, using noop NER: {e}")
            self._pipeline = None
            self._available = False

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def recognize_entities(
        self,
        items: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run NER on span_entities.

        Always returns:
        - ner: list[dict] (per span)
        - ner_raw: list[list] (per span)
        """
        batch_size = batch_size or self.batch_size

        # Flatten spans
        flat_spans = []
        span_map = []  # (item_idx, span_idx)

        for item_idx, item in enumerate(items):
            spans = item.get("span_entities", [])
            for span_idx, span in enumerate(spans):
                flat_spans.append(span)
                span_map.append((item_idx, span_idx))

        # Prepare empty outputs
        results = []
        for item in items:
            out = dict(item)
            out["ner"] = [{} for _ in item.get("span_entities", [])]
            out["ner_raw"] = [[] for _ in item.get("span_entities", [])]
            results.append(out)

        # No spans or no model → noop
        if not flat_spans or not self._available:
            return results

        # -------------------------------------------------
        # Batched inference (KeyDataset)
        # -------------------------------------------------
        try:
            from datasets import Dataset
            from transformers.pipelines.pt_utils import KeyDataset

            dataset = Dataset.from_dict({"text": flat_spans})

            outputs = list(
                self._pipeline(
                    KeyDataset(dataset, "text"),
                    batch_size=batch_size,
                )
            )

        except Exception as e:
            if self.verbose:
                print(f"[NER] Inference failed: {e}")
            return results

        # -------------------------------------------------
        # Post-process and map back
        # -------------------------------------------------
        for output, (item_idx, span_idx), span_text in zip(
            outputs, span_map, flat_spans
        ):
            raw = output

            if self.fix_words_enabled:
                raw = self._fix_words(span_text, raw)

            if self.merge_entities_enabled:
                raw = self._clean_and_merge_entities(
                    raw, min_score=self.min_score
                )

            structured = self._group_entities(raw)

            results[item_idx]["ner"][span_idx] = structured
            results[item_idx]["ner_raw"][span_idx] = raw

        return results

    # -------------------------------------------------
    # Helpers
    # -------------------------------------------------

    def _group_entities(self, raw_entities: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        grouped: Dict[str, List[str]] = {}

        for ent in raw_entities:
            label = ent.get("entity_group")
            text = ent.get("word")

            if not label or not text:
                continue

            grouped.setdefault(label, []).append(text)

        return grouped

    def _fix_words(self, raw_text: str, entities: List[Dict[str, Any]]):
        for entity in entities:
            try:
                start, end = entity.get("start"), entity.get("end")
                if start is None or end is None:
                    continue

                entity_text = raw_text[start:end]

                last_open = entity_text.rfind("(")
                last_close = entity_text.rfind(")")

                if last_open > -1 and (last_close == -1 or last_open > last_close):
                    next_close = raw_text.find(")", end)
                    if next_close > -1:
                        between = raw_text[end:next_close]
                        if not any(d in between for d in [" ", ",", ";", ":", ".", "\n", "\t"]):
                            entity["end"] = next_close + 1
                            entity_text = raw_text[start : next_close + 1]

                entity["word"] = entity_text
            except Exception:
                continue

        return entities

    def _clean_and_merge_entities(self, entities, min_score=0.0):
        entities = [e for e in entities if e.get("score", 0) >= min_score]

        merged = []
        i = 0

        while i < len(entities):
            current = entities[i]

            if i + 1 < len(entities):
                nxt = entities[i + 1]
                try:
                    if (
                        current.get("end") == nxt.get("start")
                        and nxt.get("word")
                        and (nxt["word"][0].islower() or nxt["word"][0].isdigit())
                    ):
                        merged.append({
                            "entity_group": current.get("entity_group"),
                            "score": min(current.get("score", 0), nxt.get("score", 0)),
                            "word": current.get("word", "") + nxt.get("word", ""),
                            "start": current.get("start"),
                            "end": nxt.get("end"),
                        })
                        i += 2
                        continue
                except Exception:
                    pass

            merged.append(current)
            i += 1

        return merged


def df_to_ner_items(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """
    Convert dataframe rows to AffilGood NER input items.
    Only rows with raw_affiliation_string and no existing ner are included.
    """
    items = []

    for idx, row in df.iterrows():
        raw = row.get("raw_affiliation_string")

        if not isinstance(raw, str) or not raw.strip():
            continue

        # skip rows that already have oracle NER
        if row.get("ner") is not None:
            continue

        items.append({
            "row_id": idx,
            "span_entities": [raw],  # single-span strategy (for now)
        })

    return items

def apply_ner_to_spans_affro(
    df: pd.DataFrame,
    ner_model: NER,
    batch_size: int = 32,
    ner_column: str = "ner_pred",
    ner_raw_column: str = "ner_pred_raw",
) -> pd.DataFrame:
    """
    Apply NER to span_entities for AffRoDB rows.
    """

    df = df.copy()
    df[ner_column] = None
    df[ner_raw_column] = None

    items = []

    for idx, row in df.iterrows():
        if row.get("source") != "affro":
            continue

        spans = row.get("span_entities")
        if not isinstance(spans, list) or not spans:
            continue

        items.append({
            "row_id": idx,
            "span_entities": spans,
        })

    if not items:
        return df

    results = ner_model.recognize_entities(
        items,
        batch_size=batch_size,
    )

    for item, res in zip(items, results):
        idx = item["row_id"]
        df.at[idx, ner_column] = res["ner"]
        df.at[idx, ner_raw_column] = res["ner_raw"]

    return df

def apply_affilgood_ner_non_affro(
    df: pd.DataFrame,
    ner_model: NER,
    batch_size: int = 32,
    ner_column: str = "ner_pred",
    ner_raw_column: str = "ner_pred_raw",
) -> pd.DataFrame:
    """
    Apply AffilGood NER to NON-AffRoDB rows only (single-span).
    """

    df = df.copy()

    # initialize only if missing
    if ner_column not in df.columns:
        df[ner_column] = None
    if ner_raw_column not in df.columns:
        df[ner_raw_column] = None

    items = []

    for idx, row in df.iterrows():
        if row.get("source") == "affro":
            continue

        raw = row.get("raw_affiliation_string")
        if not isinstance(raw, str) or not raw.strip():
            continue

        if row.get(ner_column) is not None:
            continue

        items.append({
            "row_id": idx,
            "span_entities": [raw],  # single span
        })

    if not items:
        return df

    results = ner_model.recognize_entities(
        items,
        batch_size=batch_size,
    )

    for item, res in zip(items, results):
        idx = item["row_id"]
        df.at[idx, ner_column] = res["ner"][0]
        df.at[idx, ner_raw_column] = res["ner_raw"][0]

    return df

import gc
import torch

def free_cuda_memory():
    """
    Fully clear CUDA memory between model loads.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

# 1) Load + apply SPAN
span_model = SpanIdentifier(
    device="cpu",
    batch_size=16,
    verbose=True,
)

df_all = apply_span_identifier_affro(
    df_all,
    span_model=span_model,
)

# 2) Explicitly delete SPAN model
del span_model
free_cuda_memory()

# 3) Load + apply NER
ner_model = NER(
    device="cuda",
    batch_size=16,
    min_score=0.1,
    verbose=True,
)

df_all = apply_ner_to_spans_affro(
    df_all,
    ner_model=ner_model,
)

df_all = apply_affilgood_ner_non_affro(
    df_all,
    ner_model=ner_model,
)

df_all.drop(['ror_names'],axis=1,inplace=True)

##########
# PRepare registry

import pandas as pd
import sys
from pathlib import Path

# project root = parent of notebook/
PROJECT_ROOT = Path.cwd().parent

SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from registry import RegistryManager

registry = RegistryManager(
    data_dir="../data/registry",
    verbose=True,
)

ror_records = registry.get_records(
    registry="ror",
    active_only=True,
)

import os, gc, json, time
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from tqdm.auto import tqdm

# ================================================================
# DATA LOADING — fill this in
# ================================================================

# >>> Your code to load df_all and ror_records goes here <<<
# df_all: DataFrame with columns:
#   dataset, source, raw_affiliation_string, ner, ner_pred, ror_all
# ror_records: list[RegistryRecord] or DataFrame with ROR v41 data

# Example:
# import pickle
# df_all = pd.read_pickle("df_all.pkl")
# ror_records = pickle.load(open("ror_records.pkl", "rb"))


# ================================================================
# IMPORT MODULE
# ================================================================

from ror_retrieval_experiments import (
    prepare_kb,
    build_retriever, build_reranker,
    build_queries_for_row, build_reranking_kb_lookup,
    _merge_retrieval, _retrieve_per_entity, _rerank_per_entity,
    _aggregate_per_entity_top1, _merge_all_reranked,
    _flatten_entity_queries, format_query_plain,
    recall_at_k, split_train_test,
    find_best_threshold, compute_reranking_metrics,
    _flush_gpu, _run_reranking_on_split,
    RETRIEVER_CONFIGS, RERANKER_CONFIGS,
    # New: hybrid, direct match, cascade
    build_hybrid_retriever, HybridRetriever,
    build_direct_match_index, direct_match_for_row,
    evaluate_direct_match,
    run_cascade_on_split,
    # New: LLM listwise reranker + cascade v2
    build_llm_reranker, LLMListwiseReranker,
    run_cascade_llm_on_split,
)
import torch


# ================================================================
# CONFIGURATION
# ================================================================

RETRIEVER_NAMES = [
    # --- Sparse (no GPU) ---
    "tfidf",
    "bm25",
    "whoosh",
    # --- Dense (GPU) ---
    "e5",
    "jina",
    "affilgood_dense",
]

RERANKER_NAMES = [
    # --- CrossEncoder ---
    "cross_encoder",          # ms-marco-MiniLM (generic)
    "cross_encoder_comet",    # cometadata/ms-marco-ror-reranker (ROR fine-tuned)
    # --- Jina ---
    "jina_reranker",          # jinaai/jina-reranker-v3
    "jina_comet",             # cometadata affiliations v5
    "jina_comet_large",       # cometadata affiliations large
    # --- Qwen ---
    "qwen_reranker",          # Qwen3-Reranker-0.6B
]

K = 10
TEST_SIZE = 0.5
SEED = 42
DEVICE = None            # None = auto (cuda if available), or "cpu", "cuda"
BATCH_SIZE = 32

RESULTS_CSV = "ror_experiment_results.csv"
CHECKPOINT_FILE = "ror_experiment_checkpoint.json"


# ================================================================
# CHECKPOINTING
# ================================================================

def make_experiment_key(retriever, reranker):
    return f"{retriever}::{reranker}"


def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            return set(json.load(f))
    return set()


def save_checkpoint(completed):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(sorted(completed), f)


def append_result_to_csv(result_dict, csv_path):
    """Append one result row to CSV (creates header on first write)."""
    df = pd.DataFrame([result_dict])
    write_header = not os.path.exists(csv_path)
    df.to_csv(csv_path, mode="a", header=write_header, index=False)


# ================================================================
# GPU HELPERS
# ================================================================

def gpu_mem_info():
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"    [GPU: {alloc:.1f}/{total:.1f} GB]")


# ================================================================
# MAIN EXPERIMENT LOOP
# ================================================================


# ================================================================
# ERROR ANALYSIS EXPORT
# ================================================================

ERROR_DIR = "error_analysis"


def write_error_analysis_txt(
    filepath,
    df_test,
    test_rr_results,
    retriever, ret_cfg,
    reranker, ror_id_to_text, id2name,
    threshold, k=10,
    ret_name="", rr_name="",
):
    """
    Export all incorrectly matched test rows to a .txt file
    with preview-style formatting for error analysis.
    """
    from ror_retrieval_experiments import (
        build_queries_for_row, _retrieve_per_entity, _rerank_per_entity,
        _aggregate_per_entity_top1, _flatten_entity_queries,
        format_query_plain, format_query_tagged,
    )

    fmt_fn = format_query_tagged if ret_cfg["query_format"] == "tagged" else format_query_plain
    lines = []
    n_errors = 0
    n_with_gold = 0

    for (_, row), rr_res in zip(df_test.iterrows(), test_rr_results):
        golds = rr_res["golds"]
        if not golds:
            continue
        n_with_gold += 1

        gold_set = set(golds)
        top1_id = rr_res["top1_id"]
        top1_score = rr_res["top1_score"]

        # Determine correctness (with threshold)
        if threshold is not None and top1_score < threshold:
            predicted = None
            is_correct = not rr_res["gold_in_candidates"]
        else:
            predicted = top1_id
            is_correct = (top1_id in gold_set)

        if is_correct:
            continue

        n_errors += 1

        # ── Re-run per-entity pipeline for full detail ──
        entity_queries = build_queries_for_row(row)
        per_e_ret = _retrieve_per_entity(
            entity_queries, retriever, ret_cfg["query_format"], k,
        ) if entity_queries else []

        raw = row.get("raw_affiliation_string", "")
        if not raw or str(raw) in ("None", "nan", ""):
            flat = _flatten_entity_queries(entity_queries) if entity_queries else []
            raw = format_query_plain(flat[0]) if flat else ""

        per_e_rr = _rerank_per_entity(
            per_e_ret, raw, reranker, ror_id_to_text,
        ) if per_e_ret else []

        final_preds = _aggregate_per_entity_top1(per_e_rr) if per_e_rr else []

        # ── Format ──
        ds = row.get("dataset", "?")
        lines.append(f"\n  ┌─ [{ds}]  error #{n_errors}")
        if raw:
            lines.append(f"  │  raw: {str(raw)[:160]}")

        # NER source
        ner_src = "ner_pred"
        ner_used = row.get("ner_pred")
        if ner_used is None or (isinstance(ner_used, float) and pd.isna(ner_used)):
            ner_used = row.get("ner")
            ner_src = "ner (oracle)"
        lines.append(f"  │  {ner_src}: {ner_used}")

        # Gold
        gold_strs = [f"{g} → {id2name.get(g, '?')}" for g in golds]
        lines.append(f"  │  gold: {gold_strs}")

        if not entity_queries:
            lines.append(f"  │  ⚠️  no queries could be built")
            lines.append(f"  └─")
            continue

        # Per-entity detail
        for ei, ((eq, ret_cands), (_, rr_cands)) in enumerate(zip(per_e_ret, per_e_rr)):
            lines.append(f"  │")
            lines.append(f"  │  ── entity [{ei+1}/{len(per_e_ret)}]: {eq['entity']}")
            for vi, v in enumerate(eq["variants"]):
                lines.append(f"  │     variant {vi+1}: {fmt_fn(v)}")

            lines.append(f"  │     retrieval:")
            for rank, (rid, sc) in enumerate(ret_cands[:k], 1):
                hit = "✅" if rid in gold_set else "  "
                lines.append(
                    f"  │       {hit} {rank}. {rid}  "
                    f"{id2name.get(rid,'?'):<45s}  ret={sc:.4f}"
                )

            lines.append(f"  │     reranked:")
            for rank, (rid, sc) in enumerate(rr_cands[:k], 1):
                hit = "✅" if rid in gold_set else "  "
                lines.append(
                    f"  │       {hit} {rank}. {rid}  "
                    f"{id2name.get(rid,'?'):<45s}  rr={sc:.4f}"
                )

        # Final predictions
        lines.append(f"  │")
        lines.append(f"  │  ── PREDICTED (t*={threshold:.4f}):")
        if predicted is None:
            lines.append(
                f"  │     ⚠️  ABSTAINED (score {top1_score:.4f} < threshold)"
            )
        for rid, sc in final_preds:
            hit = "✅" if rid in gold_set else "❌"
            lines.append(
                f"  │     {hit} {rid}  "
                f"{id2name.get(rid,'?'):<45s}  score={sc:.4f}"
            )

        # Missed golds
        matched_golds = {rid for rid, _ in final_preds if rid in gold_set}
        missed = gold_set - matched_golds
        if missed:
            all_ret_ids = set()
            for eq, cands in per_e_ret:
                for rid, _ in cands:
                    all_ret_ids.add(rid)
            lines.append(f"  │  ── MISSED gold:")
            for g in missed:
                where = (
                    "not retrieved"
                    if g not in all_ret_ids
                    else "retrieved but not ranked #1"
                )
                lines.append(
                    f"  │     ⚠️  {g}  "
                    f"{id2name.get(g,'?'):<45s}  ({where})"
                )

        lines.append(f"  └─")

    # Write file
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    header = [
        f"ERROR ANALYSIS: {ret_name} → {rr_name}",
        f"Threshold: {threshold:.4f}",
        f"Test rows with gold: {n_with_gold}",
        f"Errors: {n_errors}",
        f"Error rate: {n_errors/n_with_gold:.2%}" if n_with_gold else "N/A",
        "=" * 80,
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")

    print(f"    📝 Error analysis: {n_errors}/{n_with_gold} errors → {filepath}")


def write_retrieval_error_analysis_txt(
    filepath,
    df_test,
    retriever, ret_cfg, id2name, k=10,
    ret_name="",
):
    """
    Export retrieval-only errors (gold not in top-1) to .txt.
    """
    from ror_retrieval_experiments import (
        build_queries_for_row, _merge_retrieval,
        _flatten_entity_queries, format_query_plain, format_query_tagged,
    )

    fmt_fn = format_query_tagged if ret_cfg["query_format"] == "tagged" else format_query_plain
    lines = []
    n_errors = 0
    n_with_gold = 0

    for _, row in df_test.iterrows():
        golds = row.get("ror_all", [])
        if not isinstance(golds, list) or not golds:
            continue
        n_with_gold += 1

        gold_set = set(golds)
        queries = build_queries_for_row(row)
        if not queries:
            n_errors += 1
            ds = row.get("dataset", "?")
            raw = row.get("raw_affiliation_string", "")
            lines.append(f"\n  ┌─ [{ds}]  error #{n_errors}")
            if raw:
                lines.append(f"  │  raw: {str(raw)[:160]}")
            gold_strs = [f"{g} → {id2name.get(g, '?')}" for g in golds]
            lines.append(f"  │  gold: {gold_strs}")
            lines.append(f"  │  ⚠️  no queries could be built")
            lines.append(f"  └─")
            continue

        ranked = _merge_retrieval(queries, retriever, ret_cfg["query_format"], k)
        top1 = ranked[0][0] if ranked else None

        if top1 in gold_set:
            continue

        n_errors += 1
        ds = row.get("dataset", "?")
        raw = row.get("raw_affiliation_string", "")

        lines.append(f"\n  ┌─ [{ds}]  error #{n_errors}")
        if raw and str(raw) not in ("None", "nan", ""):
            lines.append(f"  │  raw: {str(raw)[:160]}")

        ner_src = "ner_pred"
        ner_used = row.get("ner_pred")
        if ner_used is None or (isinstance(ner_used, float) and pd.isna(ner_used)):
            ner_used = row.get("ner")
            ner_src = "ner (oracle)"
        lines.append(f"  │  {ner_src}: {ner_used}")

        gold_strs = [f"{g} → {id2name.get(g, '?')}" for g in golds]
        lines.append(f"  │  gold: {gold_strs}")

        # Show query variants
        for ei, eq in enumerate(queries):
            lines.append(f"  │  entity[{ei}]: {eq['entity']}  →  "
                         f"{[fmt_fn(v) for v in eq['variants']]}")

        # Show candidates
        retrieved_ids = set()
        lines.append(f"  │  top-{k} candidates:")
        for rank, (rid, sc) in enumerate(ranked[:k], 1):
            hit = "✅" if rid in gold_set else "  "
            retrieved_ids.add(rid)
            lines.append(
                f"  │    {hit} {rank}. {rid}  "
                f"{id2name.get(rid,'?'):<45s}  score={sc:.4f}"
            )

        # Missed golds
        missed = gold_set - retrieved_ids
        if missed:
            lines.append(f"  │  MISSED gold:")
            for g in missed:
                in_topk = g in {rid for rid, _ in ranked}
                where = f"at rank >{k}" if in_topk else "not retrieved at all"
                lines.append(
                    f"  │    ⚠️  {g}  "
                    f"{id2name.get(g,'?'):<45s}  ({where})"
                )

        lines.append(f"  └─")

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    header = [
        f"RETRIEVAL ERROR ANALYSIS: {ret_name} (retrieval only)",
        f"Test rows with gold: {n_with_gold}",
        f"Errors (gold not in top-1): {n_errors}",
        f"Error rate: {n_errors/n_with_gold:.2%}" if n_with_gold else "N/A",
        "=" * 80,
    ]
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(header + lines) + "\n")

    print(f"  📝 Retrieval errors: {n_errors}/{n_with_gold} → {filepath}")

import pycountry

def _code_to_country(code: str) -> str:
    if not code or len(code) != 2:
        return code
    try:
        return pycountry.countries.get(alpha_2=code.upper()).name
    except (AttributeError, LookupError):
        return code

def run_experiments(df_all, ror_records):
    """
    Full experiment pipeline:
      1. Build KB
      2. Split train/test (50/50 per dataset, same split for everything)
      3. For each retriever:
         a. Fit on KB
         b. Compute retrieval R@1/5/10 on test (per dataset + ALL)
         c. For each reranker:
            - Rerank train → tune threshold
            - Rerank test  → Acc@1, NoC P/R
         d. Free retriever
    """
    start_time = time.time()

    # --- Build KB ---
    print("=" * 70)
    print("  BUILDING KNOWLEDGE BASE")
    print("=" * 70)
    df_ror_kb = prepare_kb(ror_records)
    ror_id_to_text = build_reranking_kb_lookup(df_ror_kb)
    id2name = dict(zip(df_ror_kb["ror_id"], df_ror_kb["name"]))

    # --- Split ---
    print("\n" + "=" * 70)
    print("  SPLITTING DATA")
    print("=" * 70)
    df_train, df_test = split_train_test(df_all, test_size=TEST_SIZE, seed=SEED)
    test_datasets = df_test["dataset"].tolist()
    dataset_names = sorted(df_test["dataset"].unique().tolist())

    # --- Load checkpoint ---
    completed_keys = load_checkpoint()
    print(f"\n  📋 Checkpoint: {len(completed_keys)} experiments already completed")

    all_results = []
    dm_index = None  # will be built when needed (direct match / cascade)

    # ================================================================
    # RETRIEVER LOOP
    # ================================================================

    for ret_idx, ret_name in enumerate(RETRIEVER_NAMES, 1):
        print(f"\n{'='*70}")
        print(f"  📦 RETRIEVER [{ret_idx}/{len(RETRIEVER_NAMES)}]: {ret_name}")
        print(f"{'='*70}")
        gpu_mem_info()

        # --- Build and fit retriever ---
        try:
            ret_cfg = RETRIEVER_CONFIGS[ret_name]
            retriever, ret_cfg = build_retriever(
                ret_name, device=DEVICE, batch_size=BATCH_SIZE,
            )
            retriever.fit(
                df_ror_kb[ret_cfg["kb_field"]].tolist(),
                df_ror_kb["ror_id"].tolist(),
            )
        except Exception as e:
            print(f"  ❌ Failed to build retriever {ret_name}: {e}")
            import traceback; traceback.print_exc()
            _flush_gpu()
            continue

        # ============================================================
        # RETRIEVAL-ONLY METRICS (R@1, R@5, R@10)
        # Always computed (needed for reranker result rows too)
        # ============================================================
        exp_key_ret = make_experiment_key(ret_name, "retrieval_only")
        already_saved = exp_key_ret in completed_keys

        print(f"\n  📊 Retrieval metrics on test set"
              f"{' (recomputing, already saved)' if already_saved else ''}:")
        retrieval_metrics = {}

        for ds in dataset_names + ["ALL"]:
            sub = df_test if ds == "ALL" else df_test[df_test["dataset"] == ds]
            all_golds, all_preds = [], []
            for _, row in tqdm(sub.iterrows(), total=len(sub),
                               desc=f"    ret {ds}", leave=False):
                golds = row.get("ror_all", [])
                if not isinstance(golds, list):
                    golds = []
                queries = build_queries_for_row(row)
                if not queries:
                    all_golds.append(golds)
                    all_preds.append([])
                    continue
                ranked = _merge_retrieval(
                    queries, retriever, ret_cfg["query_format"], K,
                )
                all_golds.append(golds)
                all_preds.append([rid for rid, _ in ranked])

            r1  = recall_at_k(all_golds, all_preds, 1)
            r5  = recall_at_k(all_golds, all_preds, 5)
            r10 = recall_at_k(all_golds, all_preds, 10)
            retrieval_metrics[ds] = {"R@1": r1, "R@5": r5, "R@10": r10}

            tag = "📊" if ds == "ALL" else "  "
            print(f"  {tag} {ds:25s}  "
                  f"R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

            if not already_saved:
                result = {
                    "retriever": ret_name, "reranker": "retrieval_only",
                    "dataset": ds, "n_samples": len(sub),
                    "R@1": round(r1, 5), "R@5": round(r5, 5), "R@10": round(r10, 5),
                    "ceiling": None, "Acc@1": None, "Acc@1_noT": None,
                    "NoC_P": None, "NoC_R": None, "threshold": None,
                }
                all_results.append(result)
                append_result_to_csv(result, RESULTS_CSV)

        if not already_saved:
            completed_keys.add(exp_key_ret)
            save_checkpoint(completed_keys)

        # ============================================================
        # RERANKER LOOP
        # ============================================================

        for rr_idx, rr_name in enumerate(RERANKER_NAMES, 1):
            exp_key = make_experiment_key(ret_name, rr_name)
            if exp_key in completed_keys:
                print(f"\n  ⏭️  Skipping reranker [{rr_idx}/{len(RERANKER_NAMES)}] "
                      f"{rr_name} (already done)")
                continue

            print(f"\n  🔄 Reranker [{rr_idx}/{len(RERANKER_NAMES)}]: {rr_name}")
            gpu_mem_info()

            reranker = None
            try:
                reranker = build_reranker(
                    rr_name, device=DEVICE, batch_size=BATCH_SIZE,
                )

                # --- Threshold tuning on train ---
                print(f"    ⏳ Threshold tuning (train) …")
                t0 = time.time()
                train_rr = _run_reranking_on_split(
                    df_train, retriever, ret_cfg, reranker,
                    ror_id_to_text, k=K,
                    desc=f"    train {rr_name}",
                )
                best_t = find_best_threshold(train_rr)
                print(f"    🎯 t*={best_t:.4f}  ({time.time()-t0:.0f}s)")

                # --- Evaluate on test ---
                print(f"    ⏳ Evaluating (test) …")
                t0 = time.time()
                test_rr = _run_reranking_on_split(
                    df_test, retriever, ret_cfg, reranker,
                    ror_id_to_text, k=K,
                    desc=f"    test {rr_name}",
                )
                print(f"    ✅ Done ({time.time()-t0:.0f}s)")

                # --- Per-dataset + ALL metrics ---
                for ds in dataset_names + ["ALL"]:
                    if ds == "ALL":
                        rr_sub = test_rr
                    else:
                        rr_sub = [
                            r for r, d in zip(test_rr, test_datasets) if d == ds
                        ]

                    m = compute_reranking_metrics(rr_sub, threshold=best_t)
                    ret_m = retrieval_metrics.get(
                        ds, {"R@1": 0, "R@5": 0, "R@10": 0},
                    )

                    tag = "📊" if ds == "ALL" else "    "
                    print(f"  {tag} {ds:25s}  "
                          f"ceil={m['ceiling']:.3f}  "
                          f"Acc@1={m['acc_at_1']:.3f}  "
                          f"noT={m['acc_at_1_no_threshold']:.3f}  "
                          f"NocP={m['noc_precision']:.3f}  "
                          f"NocR={m['noc_recall']:.3f}")

                    result = {
                        "retriever": ret_name, "reranker": rr_name,
                        "dataset": ds, "n_samples": m["n"],
                        "R@1": round(ret_m["R@1"], 5),
                        "R@5": round(ret_m["R@5"], 5),
                        "R@10": round(ret_m["R@10"], 5),
                        "ceiling": round(m["ceiling"], 5),
                        "Acc@1": round(m["acc_at_1"], 5),
                        "Acc@1_noT": round(m["acc_at_1_no_threshold"], 5),
                        "NoC_P": round(m["noc_precision"], 5),
                        "NoC_R": round(m["noc_recall"], 5),
                        "threshold": round(best_t, 5),
                    }
                    all_results.append(result)
                    append_result_to_csv(result, RESULTS_CSV)

                # Mark completed
                completed_keys.add(exp_key)
                save_checkpoint(completed_keys)

            except torch.cuda.OutOfMemoryError:
                print(f"    ⚠️  CUDA OOM on {rr_name} — skipping")
                _flush_gpu()

            except Exception as e:
                print(f"    ❌ {rr_name}: {e}")
                import traceback; traceback.print_exc()

            finally:
                # Aggressive cleanup after each reranker
                if reranker is not None:
                    print(f"    🧹 Cleaning up {rr_name}...")
                    reranker.free()
                    del reranker
                    reranker = None
                _flush_gpu()
                gpu_mem_info()

        # ============================================================
        # FREE RETRIEVER after all rerankers done
        # ============================================================
        print(f"\n  🧹 Cleaning up retriever {ret_name}...")
        retriever.free()
        del retriever
        retriever = None
        _flush_gpu()
        gpu_mem_info()

    # ================================================================
    # EXPERIMENT: DIRECT MATCH BASELINE
    # ================================================================
    exp_key_dm = make_experiment_key("direct_match", "exact_name_country")
    if exp_key_dm not in completed_keys:
        print(f"\n{'='*70}")
        print(f"  🎯 DIRECT MATCH BASELINE (exact name + country)")
        print(f"{'='*70}")

        dm_index = build_direct_match_index(df_ror_kb, ror_records=ror_records)
        dm_results = evaluate_direct_match(df_test, dm_index, id2name)

        for ds in dataset_names + ["ALL"]:
            m = dm_results.get(ds, {})
            if not m:
                continue
            tag = "📊" if ds == "ALL" else "  "
            print(f"  {tag} {ds:25s}  "
                  f"P={m['precision']:.4f}  R={m['recall']:.4f}  "
                  f"cov={m['coverage']:.4f}  "
                  f"({m['correct']}✓ {m['incorrect']}✗ {m['missed']}miss / {m['total_gold']}gold)")

            result = {
                "retriever": "direct_match", "reranker": "exact_name_country",
                "dataset": ds, "n_samples": m["n_rows"],
                "R@1": round(m["recall"], 5), "R@5": None, "R@10": None,
                "ceiling": round(m["coverage"], 5),
                "Acc@1": round(m["precision"], 5),
                "Acc@1_noT": round(m["precision"], 5),
                "NoC_P": None, "NoC_R": None, "threshold": None,
            }
            all_results.append(result)
            append_result_to_csv(result, RESULTS_CSV)

        completed_keys.add(exp_key_dm)
        save_checkpoint(completed_keys)
    else:
        print(f"\n  ⏭️  Skipping direct_match baseline (already done)")
        dm_index = build_direct_match_index(df_ror_kb, ror_records=ror_records)  # still need for cascade

    # ================================================================
    # EXPERIMENT: HYBRID RETRIEVER (tfidf + affilgood_dense)
    # ================================================================
    exp_key_hybrid_ret = make_experiment_key("hybrid_tfidf_affilgood", "retrieval_only")
    needs_hybrid = (
        exp_key_hybrid_ret not in completed_keys
        or any(
            make_experiment_key("hybrid_tfidf_affilgood", rr) not in completed_keys
            for rr in RERANKER_NAMES
        )
    )

    if needs_hybrid:
        print(f"\n{'='*70}")
        print(f"  📦 RETRIEVER: hybrid_tfidf_affilgood (RRF)")
        print(f"{'='*70}")
        gpu_mem_info()

        try:
            hybrid_retriever, hybrid_cfg = build_hybrid_retriever(
                df_ror_kb, device=DEVICE, batch_size=BATCH_SIZE,
            )

            # --- Retrieval-only metrics ---
            if exp_key_hybrid_ret not in completed_keys:
                print(f"\n  📊 Hybrid retrieval metrics on test set:")
                hybrid_ret_metrics = {}
                for ds in dataset_names + ["ALL"]:
                    sub = df_test if ds == "ALL" else df_test[df_test["dataset"] == ds]
                    all_golds, all_preds = [], []
                    for _, row in tqdm(sub.iterrows(), total=len(sub),
                                       desc=f"    ret {ds}", leave=False):
                        golds = row.get("ror_all", [])
                        if not isinstance(golds, list):
                            golds = []
                        queries = build_queries_for_row(row)
                        if not queries:
                            all_golds.append(golds); all_preds.append([])
                            continue
                        ranked = _merge_retrieval(
                            queries, hybrid_retriever, hybrid_cfg["query_format"], K,
                        )
                        all_golds.append(golds)
                        all_preds.append([rid for rid, _ in ranked])

                    r1 = recall_at_k(all_golds, all_preds, 1)
                    r5 = recall_at_k(all_golds, all_preds, 5)
                    r10 = recall_at_k(all_golds, all_preds, 10)
                    hybrid_ret_metrics[ds] = {"R@1": r1, "R@5": r5, "R@10": r10}

                    tag = "📊" if ds == "ALL" else "  "
                    print(f"  {tag} {ds:25s}  "
                          f"R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

                    result = {
                        "retriever": "hybrid_tfidf_affilgood",
                        "reranker": "retrieval_only",
                        "dataset": ds, "n_samples": len(sub),
                        "R@1": round(r1, 5), "R@5": round(r5, 5), "R@10": round(r10, 5),
                        "ceiling": None, "Acc@1": None, "Acc@1_noT": None,
                        "NoC_P": None, "NoC_R": None, "threshold": None,
                    }
                    all_results.append(result)
                    append_result_to_csv(result, RESULTS_CSV)

                completed_keys.add(exp_key_hybrid_ret)
                save_checkpoint(completed_keys)
            else:
                print(f"  ⏭️  Hybrid retrieval_only already done (recomputing metrics)")
                hybrid_ret_metrics = {}
                for ds in dataset_names + ["ALL"]:
                    hybrid_ret_metrics[ds] = {"R@1": 0, "R@5": 0, "R@10": 0}

            # --- Hybrid + each reranker ---
            for rr_idx, rr_name in enumerate(RERANKER_NAMES, 1):
                exp_key = make_experiment_key("hybrid_tfidf_affilgood", rr_name)
                if exp_key in completed_keys:
                    print(f"\n  ⏭️  Skipping hybrid + {rr_name} (already done)")
                    continue

                print(f"\n  🔄 Reranker [{rr_idx}/{len(RERANKER_NAMES)}]: {rr_name}")
                gpu_mem_info()

                reranker = None
                try:
                    reranker = build_reranker(
                        rr_name, device=DEVICE, batch_size=BATCH_SIZE,
                    )

                    print(f"    ⏳ Threshold tuning (train) …")
                    t0 = time.time()
                    train_rr = _run_reranking_on_split(
                        df_train, hybrid_retriever, hybrid_cfg, reranker,
                        ror_id_to_text, k=K,
                        desc=f"    train hybrid+{rr_name}",
                    )
                    best_t = find_best_threshold(train_rr)
                    print(f"    🎯 t*={best_t:.4f}  ({time.time()-t0:.0f}s)")

                    print(f"    ⏳ Evaluating (test) …")
                    t0 = time.time()
                    test_rr = _run_reranking_on_split(
                        df_test, hybrid_retriever, hybrid_cfg, reranker,
                        ror_id_to_text, k=K,
                        desc=f"    test hybrid+{rr_name}",
                    )
                    print(f"    ✅ Done ({time.time()-t0:.0f}s)")

                    for ds in dataset_names + ["ALL"]:
                        if ds == "ALL":
                            rr_sub = test_rr
                        else:
                            rr_sub = [
                                r for r, d in zip(test_rr, test_datasets) if d == ds
                            ]
                        m = compute_reranking_metrics(rr_sub, threshold=best_t)
                        ret_m = hybrid_ret_metrics.get(
                            ds, {"R@1": 0, "R@5": 0, "R@10": 0},
                        )
                        tag = "📊" if ds == "ALL" else "    "
                        print(f"  {tag} {ds:25s}  "
                              f"ceil={m['ceiling']:.3f}  "
                              f"Acc@1={m['acc_at_1']:.3f}  "
                              f"noT={m['acc_at_1_no_threshold']:.3f}  "
                              f"NocP={m['noc_precision']:.3f}  "
                              f"NocR={m['noc_recall']:.3f}")

                        result = {
                            "retriever": "hybrid_tfidf_affilgood",
                            "reranker": rr_name,
                            "dataset": ds, "n_samples": m["n"],
                            "R@1": round(ret_m["R@1"], 5),
                            "R@5": round(ret_m["R@5"], 5),
                            "R@10": round(ret_m["R@10"], 5),
                            "ceiling": round(m["ceiling"], 5),
                            "Acc@1": round(m["acc_at_1"], 5),
                            "Acc@1_noT": round(m["acc_at_1_no_threshold"], 5),
                            "NoC_P": round(m["noc_precision"], 5),
                            "NoC_R": round(m["noc_recall"], 5),
                            "threshold": round(best_t, 5),
                        }
                        all_results.append(result)
                        append_result_to_csv(result, RESULTS_CSV)

                    completed_keys.add(exp_key)
                    save_checkpoint(completed_keys)

                except torch.cuda.OutOfMemoryError:
                    print(f"    ⚠️  CUDA OOM on hybrid+{rr_name} — skipping")
                    _flush_gpu()
                except Exception as e:
                    print(f"    ❌ hybrid+{rr_name}: {e}")
                    import traceback; traceback.print_exc()
                finally:
                    if reranker is not None:
                        reranker.free(); del reranker; reranker = None
                    _flush_gpu()
                    gpu_mem_info()

            # Free hybrid retriever
            hybrid_retriever.free()
            del hybrid_retriever
            _flush_gpu()

        except Exception as e:
            print(f"  ❌ Failed to build hybrid retriever: {e}")
            import traceback; traceback.print_exc()
            _flush_gpu()

    else:
        print(f"\n  ⏭️  Skipping all hybrid experiments (already done)")

    # ================================================================
    # EXPERIMENT: CASCADE (direct match → hybrid + reranker)
    # ================================================================
    CASCADE_RERANKERS = ["jina_comet"]  # best reranker from results

    for rr_name in CASCADE_RERANKERS:
        exp_key_cascade = make_experiment_key("cascade_dm+hybrid", rr_name)
        if exp_key_cascade in completed_keys:
            print(f"\n  ⏭️  Skipping cascade+{rr_name} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"  🔗 CASCADE: direct_match → hybrid + {rr_name}")
        print(f"{'='*70}")
        gpu_mem_info()

        hybrid_retriever = None
        reranker = None
        try:
            # Build hybrid retriever
            hybrid_retriever, hybrid_cfg = build_hybrid_retriever(
                df_ror_kb, device=DEVICE, batch_size=BATCH_SIZE,
            )
            reranker = build_reranker(rr_name, device=DEVICE, batch_size=BATCH_SIZE)

            # Ensure dm_index exists
            if dm_index is None:
                dm_index = build_direct_match_index(df_ror_kb, ror_records=ror_records)

            # Train: cascade → tune threshold
            print(f"  ⏳ Cascade threshold tuning (train) …")
            t0 = time.time()
            train_cascade = run_cascade_on_split(
                df_train, dm_index, hybrid_retriever, hybrid_cfg, reranker,
                ror_id_to_text, k=K,
                desc=f"  train cascade+{rr_name}",
            )
            best_t = find_best_threshold(train_cascade)
            print(f"  🎯 t*={best_t:.4f}  ({time.time()-t0:.0f}s)")

            # Test: cascade → evaluate
            print(f"  ⏳ Cascade evaluation (test) …")
            t0 = time.time()
            test_cascade = run_cascade_on_split(
                df_test, dm_index, hybrid_retriever, hybrid_cfg, reranker,
                ror_id_to_text, k=K,
                desc=f"  test cascade+{rr_name}",
            )
            print(f"  ✅ Done ({time.time()-t0:.0f}s)")

            for ds in dataset_names + ["ALL"]:
                if ds == "ALL":
                    rr_sub = test_cascade
                else:
                    rr_sub = [
                        r for r, d in zip(test_cascade, test_datasets) if d == ds
                    ]

                m = compute_reranking_metrics(rr_sub, threshold=best_t)

                tag = "📊" if ds == "ALL" else "    "
                print(f"  {tag} {ds:25s}  "
                      f"ceil={m['ceiling']:.3f}  "
                      f"Acc@1={m['acc_at_1']:.3f}  "
                      f"noT={m['acc_at_1_no_threshold']:.3f}  "
                      f"NocP={m['noc_precision']:.3f}  "
                      f"NocR={m['noc_recall']:.3f}")

                result = {
                    "retriever": "cascade_dm+hybrid",
                    "reranker": rr_name,
                    "dataset": ds, "n_samples": m["n"],
                    "R@1": None, "R@5": None, "R@10": None,
                    "ceiling": round(m["ceiling"], 5),
                    "Acc@1": round(m["acc_at_1"], 5),
                    "Acc@1_noT": round(m["acc_at_1_no_threshold"], 5),
                    "NoC_P": round(m["noc_precision"], 5),
                    "NoC_R": round(m["noc_recall"], 5),
                    "threshold": round(best_t, 5),
                }
                all_results.append(result)
                append_result_to_csv(result, RESULTS_CSV)

            completed_keys.add(exp_key_cascade)
            save_checkpoint(completed_keys)

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️  CUDA OOM on cascade+{rr_name} — skipping")
            _flush_gpu()
        except Exception as e:
            print(f"  ❌ cascade+{rr_name}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if reranker is not None:
                reranker.free(); del reranker; reranker = None
            if hybrid_retriever is not None:
                hybrid_retriever.free(); del hybrid_retriever; hybrid_retriever = None
            _flush_gpu()
            gpu_mem_info()

    # ================================================================
    # EXPERIMENT: CASCADE v2  (DM+acronyms → affilgood_dense → LLM listwise)
    # ================================================================
    #
    # This experiment tests a fundamentally different reranking approach:
    # instead of pointwise cross-encoders, use a small instruction-following
    # LLM that sees ALL candidates simultaneously (listwise comparison).
    #
    # Architecture:
    #   1. Direct match with acronyms → handles easy cases (~35%)
    #   2. affilgood_dense retrieval → best single retriever (R@1=0.905)
    #   3. LLM listwise reranker → cross-candidate comparison
    #
    # No threshold tuning: the LLM directly picks the best candidate.
    # ================================================================

    LLM_RERANKER_MODELS = [
        "Qwen/Qwen2.5-3B-Instruct",
        # Can add more: "microsoft/Phi-3.5-mini-instruct", "Qwen/Qwen3-4B"
    ]

    for llm_model_name in LLM_RERANKER_MODELS:
        llm_short = llm_model_name.split("/")[-1].lower().replace("-", "_")
        exp_key_llm = make_experiment_key("cascade_dm+affilgood", f"llm_{llm_short}")

        if exp_key_llm in completed_keys:
            print(f"\n  ⏭️  Skipping cascade_v2 + {llm_short} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"  🧠 CASCADE v2: DM+acronyms → affilgood_dense → LLM ({llm_short})")
        print(f"{'='*70}")
        gpu_mem_info()

        afg_retriever = None
        llm_reranker = None
        try:
            # Build affilgood_dense retriever (best single retriever)
            afg_retriever, afg_cfg = build_retriever(
                "affilgood_dense", device=DEVICE, batch_size=BATCH_SIZE,
            )
            afg_retriever.fit(
                df_ror_kb[afg_cfg["kb_field"]].tolist(),
                df_ror_kb["ror_id"].tolist(),
            )

            # Build LLM listwise reranker
            llm_reranker = build_llm_reranker(
                model_name=llm_model_name, device=DEVICE,
            )

            # Ensure dm_index exists (with acronyms)
            if dm_index is None:
                dm_index = build_direct_match_index(df_ror_kb, ror_records=ror_records)

            # ── Test only (no threshold tuning for LLM) ──
            print(f"  ⏳ Cascade v2 evaluation (test) …")
            t0 = time.time()
            test_results_llm = run_cascade_llm_on_split(
                df_test, dm_index, afg_retriever, afg_cfg,
                llm_reranker, ror_id_to_text, k=K,
                desc=f"  test cascade_v2+{llm_short}",
            )
            print(f"  ✅ Done ({time.time()-t0:.0f}s)")

            # Use threshold=0 (always predict top candidate)
            eval_threshold = 0.0

            for ds in dataset_names + ["ALL"]:
                if ds == "ALL":
                    rr_sub = test_results_llm
                else:
                    rr_sub = [
                        r for r, d in zip(test_results_llm, test_datasets) if d == ds
                    ]

                m = compute_reranking_metrics(rr_sub, threshold=eval_threshold)

                tag = "📊" if ds == "ALL" else "    "
                print(f"  {tag} {ds:25s}  "
                      f"ceil={m['ceiling']:.3f}  "
                      f"Acc@1={m['acc_at_1']:.3f}  "
                      f"noT={m['acc_at_1_no_threshold']:.3f}  "
                      f"NocP={m['noc_precision']:.3f}  "
                      f"NocR={m['noc_recall']:.3f}")

                result = {
                    "retriever": "cascade_dm+affilgood",
                    "reranker": f"llm_{llm_short}",
                    "dataset": ds, "n_samples": m["n"],
                    "R@1": None, "R@5": None, "R@10": None,
                    "ceiling": round(m["ceiling"], 5),
                    "Acc@1": round(m["acc_at_1"], 5),
                    "Acc@1_noT": round(m["acc_at_1_no_threshold"], 5),
                    "NoC_P": round(m["noc_precision"], 5),
                    "NoC_R": round(m["noc_recall"], 5),
                    "threshold": eval_threshold,
                }
                all_results.append(result)
                append_result_to_csv(result, RESULTS_CSV)

            completed_keys.add(exp_key_llm)
            save_checkpoint(completed_keys)

        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️  CUDA OOM on cascade_v2+{llm_short} — skipping")
            _flush_gpu()
        except Exception as e:
            print(f"  ❌ cascade_v2+{llm_short}: {e}")
            import traceback; traceback.print_exc()
        finally:
            if llm_reranker is not None:
                llm_reranker.free(); del llm_reranker; llm_reranker = None
            if afg_retriever is not None:
                afg_retriever.free(); del afg_retriever; afg_retriever = None
            _flush_gpu()
            gpu_mem_info()

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  ✅ DONE — {len(all_results)} result rows in {elapsed/60:.1f} min")
    print(f"  📋 Checkpoint: {len(completed_keys)} experiments tracked")
    print(f"  💾 Results: {RESULTS_CSV}")
    print(f"{'='*70}")

    # Summary table: ALL dataset only
    df_all_res = pd.DataFrame(all_results)
    if len(df_all_res) > 0:
        summary = df_all_res[df_all_res["dataset"] == "ALL"][[
            "retriever", "reranker", "R@1", "R@10",
            "ceiling", "Acc@1", "Acc@1_noT", "NoC_P", "NoC_R", "threshold",
        ]].copy()
        # Sort: retriever, then retrieval_only first, then rerankers
        summary["_sort"] = summary["reranker"].apply(
            lambda x: "0" if x == "retrieval_only" else "1_" + x
        )
        summary = summary.sort_values(["retriever", "_sort"]).drop(columns="_sort")
        print(f"\n{'='*70}")
        print("  SUMMARY (ALL datasets)")
        print(f"{'='*70}")
        print(summary.to_string(index=False))
        print()

    return df_all_res

# ================================================================
# ENTRY POINT
# ================================================================

if __name__ == "__main__":
    # Make sure df_all and ror_records are defined above
    try:
        df_all
    except NameError:
        print("❌ df_all is not defined. Add your data loading code at the top.")
        print("   See the DATA LOADING section in the script.")
        exit(1)

    try:
        ror_records
    except NameError:
        print("❌ ror_records is not defined. Add your data loading code at the top.")
        exit(1)

    results_df = run_experiments(df_all, ror_records)