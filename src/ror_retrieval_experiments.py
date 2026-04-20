"""
ROR Retrieval Experiments  –  v2
================================
Evaluate R@1 / R@5 / R@10 for affiliation → ROR entity linking.

Retrievers
----------
- Sparse : Whoosh (BM25F index)
- Dense  : multilingual-e5-base, Jina-v3, SIRIS-Lab/affilgood-dense-retriever

KB formatting
-------------
- All retrievers except affilgood_dense  →  plain:  "name, alias, city, country"
- affilgood_dense                        →  tagged: "[MENTION] name [ACRONYM] acr [CITY] city [COUNTRY] country"

Query formatting mirrors KB formatting per retriever.

NER handling
------------
- `ner`      (oracle) : flat dict  {'ORG': 'string', 'CITY': 'string', 'COUNTRY': 'DK'}
- `ner_pred` (model)  : dict or list[dict] with list values  {'ORG': ['str', ...], ...}

Install
-------
pip install whoosh sentence-transformers torch scikit-learn tqdm
"""

import gc, os, shutil, warnings, tempfile
import numpy as np
import pandas as pd
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any
from tqdm.auto import tqdm

warnings.filterwarnings("ignore", message="flash_attn is not installed")

# ================================================================
# 1.  QUERY CONSTRUCTION
# ================================================================

def _to_list(val) -> List[str]:
    """Normalize any NER field value into a list of non-empty strings."""
    if val is None:
        return []
    if isinstance(val, str):
        val = val.strip()
        return [val] if val else []
    if isinstance(val, list):
        return [v.strip() for v in val if isinstance(v, str) and v.strip()]
    return []


def build_queries_from_ner_dict(ner: dict) -> List[Dict]:
    """
    Build entity-centric queries with multiple geo variants per entity.

    Returns list of EntityQuery dicts:
      {
        "entity": "CNRS",
        "variants": [
            {"org": "CNRS", "city": "Rouen", "country": "France"},
            {"org": "CNRS", "city": None,    "country": "France"},
        ]
      }

    Variant strategy per entity:
      - ORG + CITY + COUNTRY  (most specific, may miss HQ-based orgs)
      - ORG + COUNTRY         (catches orgs whose ROR city ≠ affiliation city)
      - ORG + REGION + COUNTRY (if region available, e.g. "Alberta")
      - ORG only              (fallback when no geo info at all)

    Deduplicates variants that would produce identical formatted queries.
    """
    if not ner or not isinstance(ner, dict):
        return []

    orgs      = _to_list(ner.get("ORG"))
    suborgs   = _to_list(ner.get("SUBORG"))
    cities    = _to_list(ner.get("CITY"))
    countries = _to_list(ner.get("COUNTRY"))
    regions   = _to_list(ner.get("REGION"))

    entities = orgs + suborgs
    if not entities:
        return []

    city    = cities[0] if len(cities) == 1 else None
    country = countries[0] if countries else None
    region  = regions[0] if regions else None

    result = []
    for e in entities:
        variants = []
        seen = set()  # dedup by (org, city, country) tuple

        def _add(org, c, co):
            key = (org, c, co)
            if key not in seen:
                seen.add(key)
                variants.append({"org": org, "city": c, "country": co})

        # Most specific: ORG + CITY + COUNTRY
        if city and country:
            _add(e, city, country)

        # ORG + COUNTRY only (critical for HQ orgs: CNRS→Paris, INSERM→Paris)
        if country:
            _add(e, None, country)

        # ORG + REGION + COUNTRY (e.g. "Alberta, Canada")
        if region and country:
            _add(e, region, country)

        # ORG + CITY only (when no country available)
        if city and not country:
            _add(e, city, None)

        # ORG only (fallback when no geo info at all)
        if not city and not country:
            _add(e, None, None)

        result.append({
            "entity": e,
            "variants": variants,
        })

    return result


def build_queries_for_row(row) -> List[Dict]:
    """
    Build entity queries for one df_all row.

    Priority: ner_pred  >  ner (oracle).
    Returns: list of EntityQuery dicts (one per ORG/SUBORG entity).
    """
    ner_data = row.get("ner_pred")
    if ner_data is None or (isinstance(ner_data, float) and pd.isna(ner_data)):
        ner_data = row.get("ner")
    if ner_data is None or (isinstance(ner_data, float) and pd.isna(ner_data)):
        return []

    # Normalize to list[dict]
    if isinstance(ner_data, dict):
        ner_dicts = [ner_data]
    elif isinstance(ner_data, list):
        ner_dicts = [d for d in ner_data if isinstance(d, dict)]
    else:
        return []

    entity_queries = []
    for nd in ner_dicts:
        entity_queries.extend(build_queries_from_ner_dict(nd))
    return entity_queries


def _flatten_entity_queries(entity_queries: List[Dict]) -> List[Dict[str, str]]:
    """Flatten EntityQuery list to flat list of variant dicts (for backward compat)."""
    flat = []
    for eq in entity_queries:
        flat.extend(eq["variants"])
    return flat


# --- Formatters -------------------------------------------------------

def format_query_plain(q: dict) -> str:
    """'Org Name, City, Country'"""
    parts = [q["org"]]
    if q.get("city"):
        parts.append(q["city"])
    if q.get("country"):
        parts.append(q["country"])
    return ", ".join(parts)


def format_query_tagged(q: dict) -> str:
    """'[MENTION] Org Name [CITY] City [COUNTRY] Country'"""
    parts = [f"[MENTION] {q['org']}"]
    if q.get("city"):
        parts.append(f"[CITY] {q['city']}")
    if q.get("country"):
        parts.append(f"[COUNTRY] {q['country']}")
    return " ".join(parts)


# ================================================================
# 2.  ROR KNOWLEDGE-BASE PREPARATION
# ================================================================

def normalize_ror_id(ror_id: str) -> str:
    return ror_id.replace("https://ror.org/", "").strip()


def build_ror_kb_from_records(ror_records) -> pd.DataFrame:
    """
    Build KB with expanded indexing strategy:

    Primary row (entry_type='name'):
      - canonical name + all acronyms in the same entry
      - plain:  "RMIT University, RMIT, Melbourne, Australia"
      - tagged: "[MENTION] RMIT University [ACRONYM] RMIT [CITY] Melbourne [COUNTRY] Australia"
      → acronyms alone are ambiguous (ASU = 5+ universities), keeping them
        with the full name + city/country lets retrieval disambiguate.

    Alias rows (entry_type='alias'):
      - one row per alias, with canonical name in parentheses for context
      - plain:  "Royal Melbourne Institute of Technology University (RMIT University), Melbourne, Australia"
      - tagged: "[MENTION] Royal Melbourne Institute of Technology University [CITY] Melbourne [COUNTRY] Australia"
      → direct match on alias text, but canonical name boosts relevance.

    Label rows (entry_type='label'):
      - same as aliases but for multilingual labels
      - plain:  "奈良女子大学 (Nara Women's University), Nara, Japan"

    All rows map to the same ror_id → retrieval deduplicates by max score.
    """
    rows = []
    for rec in ror_records:
        ror_id       = normalize_ror_id(rec.id)
        canonical    = rec.name
        aliases      = rec.aliases  or []
        acronyms     = rec.acronyms or []
        labels       = rec.labels   or []
        city         = rec.city or ""
        country      = rec.country or ""
        country_code = getattr(rec, "country_code", "")
        parent_id    = normalize_ror_id(rec.parent_id) if rec.parent_id else None

        # ── 1) Primary row: canonical name + acronyms ──
        acr_str = ", ".join(acronyms) if acronyms else ""

        # plain: "University of Wollongong, UOW, Wollongong, Australia"
        plain_parts = [canonical]
        if acr_str:  plain_parts.append(acr_str)
        if city:     plain_parts.append(city)
        if country:  plain_parts.append(country)
        plain_text = ", ".join(plain_parts)

        # tagged: "[MENTION] University of Wollongong [ACRONYM] UOW [CITY] Wollongong [COUNTRY] Australia"
        tag_parts = [f"[MENTION] {canonical}"]
        if acronyms: tag_parts.append(f"[ACRONYM] {acronyms[0]}")
        if city:     tag_parts.append(f"[CITY] {city}")
        if country:  tag_parts.append(f"[COUNTRY] {country}")
        tagged_text = " ".join(tag_parts)

        rows.append({
            "ror_id": ror_id, "name": canonical,
            "entry_name": canonical, "entry_type": "name",
            "city": city, "country": country,
            "country_code": country_code, "parent_ror_id": parent_id,
            "plain_text": plain_text, "tagged_text": tagged_text,
        })

        # ── 2) Alias rows: "alias (canonical name), city, country" ──
        for alias in aliases:
            alias = alias.strip()
            if not alias:
                continue

            a_plain_parts = [f"{alias} ({canonical})"]
            if city:    a_plain_parts.append(city)
            if country: a_plain_parts.append(country)

            a_tag_parts = [f"[MENTION] {alias}"]
            if city:    a_tag_parts.append(f"[CITY] {city}")
            if country: a_tag_parts.append(f"[COUNTRY] {country}")

            rows.append({
                "ror_id": ror_id, "name": canonical,
                "entry_name": alias, "entry_type": "alias",
                "city": city, "country": country,
                "country_code": country_code, "parent_ror_id": parent_id,
                "plain_text": ", ".join(a_plain_parts),
                "tagged_text": " ".join(a_tag_parts),
            })

        # ── 3) Label rows: "label (canonical name), city, country" ──
        for label in labels:
            label = label.strip()
            if not label:
                continue

            l_plain_parts = [f"{label} ({canonical})"]
            if city:    l_plain_parts.append(city)
            if country: l_plain_parts.append(country)

            l_tag_parts = [f"[MENTION] {label}"]
            if city:    l_tag_parts.append(f"[CITY] {city}")
            if country: l_tag_parts.append(f"[COUNTRY] {country}")

            rows.append({
                "ror_id": ror_id, "name": canonical,
                "entry_name": label, "entry_type": "label",
                "city": city, "country": country,
                "country_code": country_code, "parent_ror_id": parent_id,
                "plain_text": ", ".join(l_plain_parts),
                "tagged_text": " ".join(l_tag_parts),
            })

    df = pd.DataFrame(rows)
    n_ror = df["ror_id"].nunique()
    print(f"  KB expanded: {n_ror} ROR records → {len(df)} index entries")
    print(f"    {df['entry_type'].value_counts().to_dict()}")
    return df


def build_ror_kb_from_dataframe(df_ror: pd.DataFrame) -> pd.DataFrame:
    """
    Expand an existing df_ror DataFrame using the same strategy.
    Expects: ror_id, name, aliases, acronyms, labels, city, country.
    """
    rows = []
    for _, r in df_ror.iterrows():
        ror_id    = str(r["ror_id"])
        canonical = str(r.get("name", ""))
        aliases   = r.get("aliases", []) or []
        acronyms  = r.get("acronyms", []) or []
        labels    = r.get("labels", []) or []
        city      = str(r.get("city", "") or "")
        country   = str(r.get("country", "") or "")
        country_code = str(r.get("country_code", "") or "")
        parent_id = r.get("parent_ror_id")

        if isinstance(aliases, str):  aliases = [aliases]
        if isinstance(acronyms, str): acronyms = [acronyms]
        if isinstance(labels, str):   labels = [labels]

        # ── primary row ──
        acr_str = ", ".join(a for a in acronyms if a.strip())
        plain_parts = [canonical]
        if acr_str:  plain_parts.append(acr_str)
        if city:     plain_parts.append(city)
        if country:  plain_parts.append(country)

        tag_parts = [f"[MENTION] {canonical}"]
        if acronyms: tag_parts.append(f"[ACRONYM] {acronyms[0]}")
        if city:     tag_parts.append(f"[CITY] {city}")
        if country:  tag_parts.append(f"[COUNTRY] {country}")

        rows.append({
            "ror_id": ror_id, "name": canonical,
            "entry_name": canonical, "entry_type": "name",
            "city": city, "country": country,
            "country_code": country_code, "parent_ror_id": parent_id,
            "plain_text": ", ".join(plain_parts),
            "tagged_text": " ".join(tag_parts),
        })

        # ── alias rows ──
        for alias in aliases:
            if not isinstance(alias, str) or not alias.strip():
                continue
            alias = alias.strip()
            ap = [f"{alias} ({canonical})"]
            if city:    ap.append(city)
            if country: ap.append(country)
            at = [f"[MENTION] {alias}"]
            if city:    at.append(f"[CITY] {city}")
            if country: at.append(f"[COUNTRY] {country}")
            rows.append({
                "ror_id": ror_id, "name": canonical,
                "entry_name": alias, "entry_type": "alias",
                "city": city, "country": country,
                "country_code": country_code, "parent_ror_id": parent_id,
                "plain_text": ", ".join(ap), "tagged_text": " ".join(at),
            })

        # ── label rows ──
        for label in labels:
            if not isinstance(label, str) or not label.strip():
                continue
            label = label.strip()
            lp = [f"{label} ({canonical})"]
            if city:    lp.append(city)
            if country: lp.append(country)
            lt = [f"[MENTION] {label}"]
            if city:    lt.append(f"[CITY] {city}")
            if country: lt.append(f"[COUNTRY] {country}")
            rows.append({
                "ror_id": ror_id, "name": canonical,
                "entry_name": label, "entry_type": "label",
                "city": city, "country": country,
                "country_code": country_code, "parent_ror_id": parent_id,
                "plain_text": ", ".join(lp), "tagged_text": " ".join(lt),
            })

    df = pd.DataFrame(rows)
    n_ror = df["ror_id"].nunique()
    print(f"  KB expanded: {n_ror} rows → {len(df)} index entries")
    print(f"    {df['entry_type'].value_counts().to_dict()}")
    return df


def prepare_kb(ror_data) -> pd.DataFrame:
    """Accept list[RegistryRecord] or DataFrame → return KB with plain_text + tagged_text."""
    if isinstance(ror_data, pd.DataFrame):
        return build_ror_kb_from_dataframe(ror_data)
    elif isinstance(ror_data, list):
        return build_ror_kb_from_records(ror_data)
    raise ValueError(f"Expected DataFrame or list, got {type(ror_data)}")


# ================================================================
# 3.  GPU HELPERS
# ================================================================

def _flush_gpu():
    """Aggressively free GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        gc.collect()
        torch.cuda.empty_cache()


# ================================================================
# 4.  RETRIEVERS
# ================================================================

class BaseRetriever(ABC):
    @abstractmethod
    def fit(self, kb_texts: List[str], kb_ids: List[str]): ...
    @abstractmethod
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]: ...
    def free(self):
        pass


# ---------- TF-IDF ----------

class TfidfRetriever(BaseRetriever):
    """Sparse retriever using character n-gram TF-IDF + cosine similarity."""

    def __init__(self, analyzer="char_wb", ngram_range=(3, 6)):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            analyzer=analyzer, ngram_range=ngram_range,
            lowercase=True, sublinear_tf=True,
        )

    def fit(self, kb_texts, kb_ids):
        self.kb_ids = kb_ids
        self.kb_emb = self.vectorizer.fit_transform(kb_texts)
        print(f"  ✅ TF-IDF fitted: {self.kb_emb.shape}")

    def retrieve(self, query, k=10):
        from sklearn.metrics.pairwise import cosine_similarity
        q = self.vectorizer.transform([query])
        sims = cosine_similarity(q, self.kb_emb)[0]
        idxs = sims.argsort()[-k:][::-1]
        return [(self.kb_ids[i], float(sims[i])) for i in idxs]


# ---------- BM25 (rank_bm25) ----------

class BM25Retriever(BaseRetriever):
    """Sparse retriever using rank_bm25 (Okapi BM25)."""

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def fit(self, kb_texts, kb_ids):
        from rank_bm25 import BM25Okapi
        self.kb_ids = kb_ids
        tokenized = [text.lower().split() for text in kb_texts]
        self.bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        print(f"  ✅ BM25 fitted: {len(kb_ids)} docs")

    def retrieve(self, query, k=10):
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        idxs = scores.argsort()[-k:][::-1]
        return [(self.kb_ids[i], float(scores[i])) for i in idxs]


# ---------- Whoosh (BM25F) ----------

class WhooshRetriever(BaseRetriever):
    """
    Full-text sparse retriever backed by Whoosh with BM25F scoring.
    Creates a temporary on-disk index (cleaned up on .free()).
    """

    def __init__(self):
        self._index_dir = None
        self._ix = None

    def fit(self, kb_texts: List[str], kb_ids: List[str]):
        from whoosh import index as whoosh_index
        from whoosh.fields import Schema, TEXT, ID
        from whoosh.analysis import StandardAnalyzer

        self._index_dir = tempfile.mkdtemp(prefix="whoosh_ror_")

        schema = Schema(
            ror_id=ID(stored=True, unique=True),
            content=TEXT(analyzer=StandardAnalyzer(), stored=False),
        )
        self._ix = whoosh_index.create_in(self._index_dir, schema)

        writer = self._ix.writer(procs=1, limitmb=256)
        for rid, txt in zip(kb_ids, kb_texts):
            writer.add_document(ror_id=rid, content=txt)
        writer.commit()
        print(f"  ✅ Whoosh index built: {len(kb_ids)} docs  →  {self._index_dir}")

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        from whoosh.qparser import MultifieldParser, OrGroup
        from whoosh import scoring

        with self._ix.searcher(weighting=scoring.BM25F()) as searcher:
            parser = MultifieldParser(["content"], self._ix.schema, group=OrGroup)
            parsed = parser.parse(query)
            hits = searcher.search(parsed, limit=k)
            return [(h["ror_id"], float(h.score)) for h in hits]

    def free(self):
        self._ix = None
        if self._index_dir and os.path.isdir(self._index_dir):
            shutil.rmtree(self._index_dir, ignore_errors=True)
        self._index_dir = None


# ---------- Dense (SentenceTransformer) ----------

class DenseRetriever(BaseRetriever):
    """Generic dense retriever wrapping any SentenceTransformer model."""

    def __init__(self, model_name: str, query_prefix="", doc_prefix="",
                 batch_size=32, device: str = None):
        from sentence_transformers import SentenceTransformer
        _flush_gpu()  # free anything lingering before allocating
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.qp = query_prefix
        self.dp = doc_prefix
        self.bs = batch_size
        self.model_name = model_name
        print(f"  ✅ Dense model loaded: {model_name}  ({self.device})")

    def fit(self, kb_texts, kb_ids):
        self.kb_ids = kb_ids
        texts = [f"{self.dp}{t}" for t in kb_texts] if self.dp else kb_texts
        print(f"  ⏳ Encoding {len(kb_texts)} KB entries …")
        self.kb_emb = self.model.encode(
            texts, normalize_embeddings=True,
            show_progress_bar=True, batch_size=self.bs,
        )
        print(f"  ✅ KB encoded: {self.kb_emb.shape}")

    def retrieve(self, query, k=10):
        q = f"{self.qp}{query}" if self.qp else query
        q_emb = self.model.encode([q], normalize_embeddings=True)[0]
        sims = q_emb @ self.kb_emb.T
        idxs = sims.argsort()[-k:][::-1]
        return [(self.kb_ids[i], float(sims[i])) for i in idxs]

    def free(self):
        """Aggressively release GPU memory."""
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if hasattr(self, "kb_emb") and self.kb_emb is not None:
            del self.kb_emb
            self.kb_emb = None
        _flush_gpu()


# ---------- Jina ----------

class JinaRetriever(BaseRetriever):
    """Jina embeddings v3 retriever with task-specific encoding."""

    def __init__(self, model_name="jinaai/jina-embeddings-v3", device: str = None):
        from transformers import AutoModel
        _flush_gpu()  # free anything lingering before allocating
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"  ✅ Jina loaded: {model_name}  ({self.device})")

    def fit(self, kb_texts, kb_ids):
        self.kb_ids = kb_ids
        print(f"  ⏳ Jina encoding {len(kb_texts)} KB entries …")
        emb = self.model.encode(kb_texts, task="retrieval.passage",
                                normalize_embeddings=True)
        self.kb_emb = emb.cpu().numpy() if isinstance(emb, torch.Tensor) else emb
        print(f"  ✅ KB encoded: {self.kb_emb.shape}")

    def retrieve(self, query, k=10):
        emb = self.model.encode([query], task="retrieval.query",
                                normalize_embeddings=True)
        q_emb = emb[0].cpu().numpy() if isinstance(emb[0], torch.Tensor) else emb[0]
        sims = q_emb @ self.kb_emb.T
        idxs = sims.argsort()[-k:][::-1]
        return [(self.kb_ids[i], float(sims[i])) for i in idxs]

    def free(self):
        """Aggressively release GPU memory."""
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if hasattr(self, "kb_emb") and self.kb_emb is not None:
            del self.kb_emb
            self.kb_emb = None
        _flush_gpu()


# ================================================================
# 5.  RETRIEVER REGISTRY
# ================================================================

RETRIEVER_CONFIGS = {
    "tfidf": {
        "class": "TfidfRetriever",
        "kb_field": "plain_text",
        "query_format": "plain",
    },
    "bm25": {
        "class": "BM25Retriever",
        "kb_field": "plain_text",
        "query_format": "plain",
    },
    "whoosh": {
        "class": "WhooshRetriever",
        "kb_field": "plain_text",
        "query_format": "plain",
    },
    "e5": {
        "class": "DenseRetriever",
        "model_name": "intfloat/multilingual-e5-base",
        "query_prefix": "query: ",
        "doc_prefix": "passage: ",
        "kb_field": "plain_text",
        "query_format": "plain",
    },
    "jina": {
        "class": "JinaRetriever",
        "model_name": "jinaai/jina-embeddings-v3",
        "kb_field": "plain_text",
        "query_format": "plain",
    },
    "affilgood_dense": {
        "class": "DenseRetriever",
        "model_name": "SIRIS-Lab/affilgood-dense-retriever",
        "query_prefix": "",
        "doc_prefix": "",
        "kb_field": "tagged_text",
        "query_format": "tagged",
    },
}


def build_retriever(name: str, device: str = None, batch_size: int = 32) -> Tuple[BaseRetriever, dict]:
    """Instantiate a retriever by config name. device: 'cpu', 'cuda', or None (auto)."""
    cfg = RETRIEVER_CONFIGS[name]
    cls = cfg["class"]
    if cls == "TfidfRetriever":
        return TfidfRetriever(), cfg
    elif cls == "BM25Retriever":
        return BM25Retriever(), cfg
    elif cls == "WhooshRetriever":
        return WhooshRetriever(), cfg
    elif cls == "DenseRetriever":
        _flush_gpu()  # free anything lingering before loading model
        return DenseRetriever(
            model_name=cfg["model_name"],
            query_prefix=cfg.get("query_prefix", ""),
            doc_prefix=cfg.get("doc_prefix", ""),
            device=device,
            batch_size=batch_size,
        ), cfg
    elif cls == "JinaRetriever":
        _flush_gpu()
        return JinaRetriever(model_name=cfg["model_name"], device=device), cfg
    raise ValueError(f"Unknown class: {cls}")


# ================================================================
# 6.  EVALUATION HELPERS
# ================================================================

def recall_at_k(gold_ids_list, pred_ids_list, k):
    """Recall@k — a hit = any gold id in top-k predictions."""
    hits = total = 0
    for golds, preds in zip(gold_ids_list, pred_ids_list):
        if not golds:
            continue
        total += 1
        if any(g in set(preds[:k]) for g in golds):
            hits += 1
    return hits / total if total else 0.0


def _merge_retrieval(entity_queries, retriever, query_format, k):
    """
    Run all entity query variants, merge by max score, return ranked list.

    Accepts List[EntityQuery] (new format) or List[dict] (legacy flat).
    Over-fetches (k*3) to account for duplicate ror_ids from expanded KB.
    """
    fmt = format_query_tagged if query_format == "tagged" else format_query_plain
    internal_k = k * 3

    # Flatten to list of variant dicts
    if entity_queries and isinstance(entity_queries[0], dict) and "variants" in entity_queries[0]:
        flat_queries = _flatten_entity_queries(entity_queries)
    else:
        flat_queries = entity_queries  # legacy flat format

    score_map = {}
    for q in flat_queries:
        for rid, sc in retriever.retrieve(fmt(q), k=internal_k):
            if rid not in score_map or sc > score_map[rid]:
                score_map[rid] = sc
    return sorted(score_map.items(), key=lambda x: -x[1])[:k]


# ================================================================
# 7.  PREVIEW: ONE EXAMPLE PER DATASET
# ================================================================

def preview_examples(
    df_all: pd.DataFrame,
    df_ror_kb: pd.DataFrame,
    retriever_names: List[str] = None,
    k: int = 5,
    n_per_dataset: int = 1,
    seed: int = 42,
    device: str = None,
    batch_size: int = 32,
):
    """
    For each dataset, pick `n_per_dataset` rows, build queries,
    run every retriever and print retrieved candidates side-by-side.

    Loads/frees one retriever at a time to manage GPU memory.
    """
    if retriever_names is None:
        retriever_names = list(RETRIEVER_CONFIGS.keys())

    # --- pick example rows (same rows for all retrievers) ---
    rng = np.random.RandomState(seed)
    examples = []  # list of (original_idx, row_series)
    for ds in sorted(df_all["dataset"].unique()):
        sub = df_all[df_all["dataset"] == ds]
        # prefer rows that produce at least one query
        good = sub[sub.apply(lambda r: len(build_queries_for_row(r)) > 0, axis=1)]
        pool = good if len(good) >= n_per_dataset else sub
        chosen = pool.sample(n=min(n_per_dataset, len(pool)), random_state=rng)
        for idx, row in chosen.iterrows():
            examples.append((idx, row))

    # --- id→name lookup for display ---
    id2name = dict(zip(df_ror_kb["ror_id"], df_ror_kb["name"]))

    # --- run each retriever on the SAME set of examples ---
    for ret_name in retriever_names:
        cfg = RETRIEVER_CONFIGS[ret_name]
        print(f"\n{'━'*72}")
        print(f"  RETRIEVER: {ret_name}   (KB: {cfg['kb_field']},  query: {cfg['query_format']})")
        print(f"{'━'*72}")

        retriever, cfg = build_retriever(ret_name, device=device, batch_size=batch_size)
        retriever.fit(
            df_ror_kb[cfg["kb_field"]].tolist(),
            df_ror_kb["ror_id"].tolist(),
        )

        for idx, row in examples:
            ds    = row["dataset"]
            src   = row["source"]
            golds = row.get("ror_all", [])
            if not isinstance(golds, list):
                golds = []
            queries = build_queries_for_row(row)

            # ── header ──
            print(f"\n  ┌─ [{src}/{ds}]  row={idx}")
            raw = row.get("raw_affiliation_string")
            if raw and str(raw) not in ("None", "nan", ""):
                print(f"  │  raw: {str(raw)[:130]}")

            # ── show which NER was used ──
            ner_src = "ner_pred"
            ner_used = row.get("ner_pred")
            if ner_used is None or (isinstance(ner_used, float) and pd.isna(ner_used)):
                ner_used = row.get("ner")
                ner_src = "ner (oracle)"
            print(f"  │  {ner_src}: {ner_used}")

            # ── gold ──
            gold_strs = [f"{g} → {id2name.get(g, '?')}" for g in golds]
            print(f"  │  gold: {gold_strs}")

            if not queries:
                print(f"  │  ⚠ no queries could be built")
                print(f"  └─")
                continue

            # ── entity queries with variants ──
            fmt_fn = format_query_tagged if cfg["query_format"] == "tagged" else format_query_plain
            for ei, eq in enumerate(queries):
                entity_name = eq["entity"]
                variant_strs = [fmt_fn(v) for v in eq["variants"]]
                print(f"  │  entity[{ei}]: {entity_name}  →  variants: {variant_strs}")

            # ── retrieve (merges across all entities × variants) ──
            ranked = _merge_retrieval(queries, retriever, cfg["query_format"], k)

            # ── show candidates ──
            gold_set = set(golds)
            print(f"  │  top-{k} candidates:")
            for rank, (rid, sc) in enumerate(ranked, 1):
                name = id2name.get(rid, "?")
                hit  = "✅" if rid in gold_set else "  "
                print(f"  │    {hit} {rank}. {rid}  {name:<50s}  score={sc:.4f}")
            print(f"  └─")

        # cleanup
        retriever.free()
        del retriever
        _flush_gpu()

    print(f"\n{'━'*72}")
    print("  Done — refine query / KB formatting, then call run_all_experiments().")
    print(f"{'━'*72}")


# ================================================================
# 8.  FULL EVALUATION  (call once you're happy with the preview)
# ================================================================

@dataclass
class RetrievalResult:
    retriever: str
    dataset: str
    n_samples: int
    n_with_queries: int
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float


def run_retrieval_experiment(df_all, df_ror_kb, retriever_name, k=10, device=None, batch_size=32):
    """Evaluate one retriever on all datasets.  Returns list[RetrievalResult]."""
    print(f"\n{'='*60}")
    print(f"  RETRIEVER: {retriever_name}")
    print(f"{'='*60}")

    retriever, cfg = build_retriever(retriever_name, device=device, batch_size=batch_size)
    retriever.fit(
        df_ror_kb[cfg["kb_field"]].tolist(),
        df_ror_kb["ror_id"].tolist(),
    )

    datasets = sorted(df_all["dataset"].unique().tolist()) + ["ALL"]
    results = []

    for ds in datasets:
        df_sub = df_all if ds == "ALL" else df_all[df_all["dataset"] == ds]
        if len(df_sub) == 0:
            continue

        all_golds, all_preds = [], []
        n_q = 0

        for _, row in tqdm(df_sub.iterrows(), total=len(df_sub),
                           desc=f"  {ds}", leave=False):
            golds = row.get("ror_all", [])
            if not isinstance(golds, list):
                golds = []
            queries = build_queries_for_row(row)

            if not queries:
                all_golds.append(golds)
                all_preds.append([])
                continue

            n_q += 1
            ranked = _merge_retrieval(queries, retriever, cfg["query_format"], k)
            all_golds.append(golds)
            all_preds.append([rid for rid, _ in ranked])

        r1  = recall_at_k(all_golds, all_preds, 1)
        r5  = recall_at_k(all_golds, all_preds, 5)
        r10 = recall_at_k(all_golds, all_preds, 10)
        results.append(RetrievalResult(retriever_name, ds, len(df_sub), n_q, r1, r5, r10))

        tag = "📊" if ds == "ALL" else "  "
        print(f"  {tag} {ds:25s}  n={len(df_sub):5d}  "
              f"R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

    retriever.free()
    del retriever
    _flush_gpu()
    return results


def run_all_experiments(df_all, df_ror_kb, retriever_names=None, k=10,
                        output_csv="retrieval_results.csv", device=None, batch_size=32):
    """Run all retrievers sequentially, return combined DataFrame."""
    if retriever_names is None:
        retriever_names = list(RETRIEVER_CONFIGS.keys())

    all_res = []
    for rn in retriever_names:
        try:
            all_res.extend(run_retrieval_experiment(df_all, df_ror_kb, rn, k,
                                                    device=device, batch_size=batch_size))
        except Exception as e:
            print(f"  ❌ {rn}: {e}")
            import traceback
            traceback.print_exc()
        finally:
            _flush_gpu()

    df = pd.DataFrame([
        {"retriever": r.retriever, "dataset": r.dataset,
         "n": r.n_samples, "n_queries": r.n_with_queries,
         "R@1": r.recall_at_1, "R@5": r.recall_at_5, "R@10": r.recall_at_10}
        for r in all_res
    ])
    if output_csv:
        df.to_csv(output_csv, index=False)
        print(f"\n💾 Saved → {output_csv}")
    return df


# ================================================================
# 9.  RERANKERS
# ================================================================

class BaseReranker(ABC):
    """
    Rerankers score (query, document) pairs.
    Input: list of (query_text, doc_text) tuples.
    Output: list of float scores (higher = more relevant).
    """
    @abstractmethod
    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        pass

    def free(self):
        pass


# ---------- CrossEncoder (sentence-transformers) ----------

class CrossEncoderReranker(BaseReranker):
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-12-v2",
                 device=None, batch_size=32):
        from sentence_transformers import CrossEncoder
        _flush_gpu()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=self.device, trust_remote_code=True)
        self.bs = batch_size
        self.model_name = model_name
        print(f"  ✅ CrossEncoder reranker loaded: {model_name}  ({self.device})")

    def score_pairs(self, pairs):
        if not pairs:
            return []
        scores = self.model.predict(
            pairs, batch_size=self.bs, show_progress_bar=False,
        )
        return [float(s) for s in scores]

    def free(self):
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        _flush_gpu()


# ---------- Jina Reranker ----------
def _patch_xlm_roberta_position_ids():
    """
    Fix compatibility between cometadata/jina-reranker-v2-* and
    jinaai/jina-reranker-v3 custom code and transformers ≥4.45.
    """
    from transformers.models.xlm_roberta import modeling_xlm_roberta
    import torch

    # --- Fix 1: missing create_position_ids_from_input_ids ---
    if not hasattr(modeling_xlm_roberta, "create_position_ids_from_input_ids"):
        def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
            mask = input_ids.ne(padding_idx).int()
            incremental_indices = (
                torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length
            ) * mask
            return incremental_indices.long() + padding_idx

        modeling_xlm_roberta.create_position_ids_from_input_ids = create_position_ids_from_input_ids

    # --- Fix 2 & 3: patch _finalize_model_loading ---
        import transformers.modeling_utils as mu

        if getattr(mu, "_patched_finalize_done", False):
            return
        mu._patched_finalize_done = True

        orig = mu.PreTrainedModel._finalize_model_loading
        _orig_finalize = orig.__func__ if hasattr(orig, "__func__") else orig

        def _patched_finalize(*args, **kwargs):
            # args[0] is cls (from classmethod), args[1] is model
            model = args[1] if len(args) > 1 else args[0]

            if not hasattr(model, "all_tied_weights_keys"):
                model.all_tied_weights_keys = {}

            _orig_mark = model.mark_tied_weights_as_initialized

            def _safe_mark(*a, **kw):
                try:
                    return _orig_mark(*a, **kw)
                except AttributeError:
                    pass

            model.mark_tied_weights_as_initialized = _safe_mark

            return orig(*args[1:], **kwargs)

        mu.PreTrainedModel._finalize_model_loading = classmethod(_patched_finalize)

class JinaRerankerModel(BaseReranker):
    """
    Jina reranker via AutoModel with trust_remote_code.
    Works for: jinaai/jina-reranker-v3, cometadata/jina-reranker-v2-* models.
    Uses model.rerank(query, documents) API, grouping by query for efficiency.
    """

    def __init__(self, model_name="jinaai/jina-reranker-v3", device=None):
        from transformers import AutoModel
        _flush_gpu()
        _patch_xlm_roberta_position_ids()  # ← add this
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model_name = model_name
        print(f"  ✅ Jina reranker loaded: {model_name}  ({self.device})")

    def score_pairs(self, pairs):
        if not pairs:
            return []
        from collections import OrderedDict

        # Group by query (in our flow, usually 1 query per call)
        groups = OrderedDict()
        for i, (q, d) in enumerate(pairs):
            if q not in groups:
                groups[q] = []
            groups[q].append((i, d))

        scores = [0.0] * len(pairs)
        for query, idx_docs in groups.items():
            docs = [d for _, d in idx_docs]
            results = self.model.rerank(query, docs)
            # results: list of {'index': int, 'relevance_score': float, ...}
            for res in results:
                orig_idx = idx_docs[res["index"]][0]
                scores[orig_idx] = float(res["relevance_score"])
        return scores

    def free(self):
        if self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        _flush_gpu()


# ---------- Qwen3 Reranker ----------

class QwenReranker(BaseReranker):
    """
    Qwen3 reranker — causal LM that scores via yes/no logits.
    Uses the official Qwen3-Reranker prompt format.
    """

    def __init__(self, model_name="Qwen/Qwen3-Reranker-0.6B",
                 tokenizer_name=None, device=None, batch_size=32):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _flush_gpu()

        tok_name = tokenizer_name or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tok_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto", device_map="auto" if device is None else None,
        )
        if device is not None:
            self.model.to(device)
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)
        self.bs = batch_size
        self.model_name = model_name

        # yes/no token ids
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # prompt template
        self.prefix = (
            "<|im_start|>system\nJudge whether the Document meets the "
            "requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = (
            "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.max_length = 8192

        self.task_instruction = (
            "Given an affiliation string from a scientific publication, "
            "judge whether the Document describes the same research institution "
            "or organization mentioned in the affiliation."
        )
        print(f"  ✅ Qwen reranker loaded: {model_name}  ({self.device})")

    def _format_pair(self, query, document):
        return (
            f"<Instruct>: {self.task_instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def _process_inputs(self, formatted_pairs):
        inputs = self.tokenizer(
            formatted_pairs, padding=False, truncation="longest_first",
            return_attention_mask=False,
            max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )
        for i, ele in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=self.max_length,
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def _compute_scores(self, inputs):
        logits = self.model(**inputs).logits[:, -1, :]
        true_vec = logits[:, self.token_true_id]
        false_vec = logits[:, self.token_false_id]
        stacked = torch.stack([false_vec, true_vec], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return log_probs[:, 1].exp().tolist()

    def score_pairs(self, pairs):
        if not pairs:
            return []
        formatted = [self._format_pair(q, d) for q, d in pairs]
        all_scores = []
        for i in range(0, len(formatted), self.bs):
            batch = formatted[i : i + self.bs]
            inputs = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores)
        return all_scores

    def free(self):
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None
        _flush_gpu()


# ================================================================
# 10.  RERANKER REGISTRY
# ================================================================

RERANKER_CONFIGS = {
    "cross_encoder": {
        "class": "CrossEncoderReranker",
        "model_name": "cross-encoder/ms-marco-MiniLM-L-12-v2",
    },
    "cross_encoder_comet": {
        "class": "CrossEncoderReranker",
        "model_name": "cometadata/ms-marco-ror-reranker",
    },
    "jina_reranker": {
        "class": "JinaRerankerModel",          # ← v3 with .rerank() API
        "model_name": "jinaai/jina-reranker-v3",
    },
    "jina_comet": {
        "class": "CrossEncoderReranker",        # ← CHANGED: it's a CrossEncoder
        "model_name": "cometadata/jina-reranker-v2-multilingual-affiliations-v5",
    },
    "jina_comet_large": {
        "class": "CrossEncoderReranker",        # ← CHANGED: it's a CrossEncoder
        "model_name": "cometadata/jina-reranker-v2-multilingual-affiliations-large",
    },
    "qwen_reranker": {
        "class": "QwenReranker",
        "model_name": "Qwen/Qwen3-Reranker-0.6B",
    },
}


def build_reranker(name: str, device=None, batch_size=32) -> BaseReranker:
    """Instantiate a reranker by config name."""
    cfg = RERANKER_CONFIGS[name]
    cls = cfg["class"]
    _flush_gpu()
    if cls == "CrossEncoderReranker":
        return CrossEncoderReranker(
            model_name=cfg["model_name"], device=device, batch_size=batch_size,
        )
    elif cls == "JinaRerankerModel":
        return JinaRerankerModel(
            model_name=cfg["model_name"], device=device,
        )
    elif cls == "QwenReranker":
        return QwenReranker(
            model_name=cfg["model_name"], device=device, batch_size=batch_size,
        )
    raise ValueError(f"Unknown reranker class: {cls}")


# ================================================================
# 11.  RERANKING PIPELINE HELPERS
# ================================================================

def build_reranking_kb_lookup(df_ror_kb: pd.DataFrame) -> Dict[str, str]:
    """
    Build ror_id → rich text for reranker document side.

    Includes ALL name variants (canonical, aliases, acronyms, labels)
    so the reranker can match e.g. Japanese aliases or alternative names.

    Format: "Name | Alias1 | Alias2 | ACR, City, Country"
    """
    # Group all entry_names by ror_id
    grouped = df_ror_kb.groupby("ror_id").agg({
        "name": "first",           # canonical name
        "entry_name": list,        # all variants
        "entry_type": list,
        "city": "first",
        "country": "first",
    }).reset_index()

    lookup = {}
    for _, r in grouped.iterrows():
        # Deduplicate names while preserving order
        seen = set()
        all_names = []
        for n in r["entry_name"]:
            if n not in seen:
                seen.add(n)
                all_names.append(n)

        names_str = " | ".join(all_names)
        parts = [names_str]
        if r["city"]:
            parts.append(str(r["city"]))
        if r["country"]:
            parts.append(str(r["country"]))
        lookup[r["ror_id"]] = ", ".join(parts)

    return lookup


def retrieve_candidates_for_row(row, retriever, ret_cfg, k=10):
    """
    Build queries for a row, retrieve from all queries,
    merge by max score, return deduplicated [(ror_id, score), ...].
    """
    queries = build_queries_for_row(row)
    if not queries:
        return []
    return _merge_retrieval(queries, retriever, ret_cfg["query_format"], k)


def _retrieve_per_entity(entity_queries, retriever, query_format, k):
    """
    Retrieve candidates per entity, merging across all geo variants.

    Input:  list of EntityQuery dicts
    Output: list of (EntityQuery, [(ror_id, score), ...])

    For each entity, all variants (ORG+CITY+COUNTRY, ORG+COUNTRY, etc.)
    are queried and results merged by max score per ror_id.
    """
    fmt = format_query_tagged if query_format == "tagged" else format_query_plain
    internal_k = k * 3

    per_entity = []
    for eq in entity_queries:
        score_map = {}
        for variant in eq["variants"]:
            for rid, sc in retriever.retrieve(fmt(variant), k=internal_k):
                if rid not in score_map or sc > score_map[rid]:
                    score_map[rid] = sc
        ranked = sorted(score_map.items(), key=lambda x: -x[1])[:k]
        per_entity.append((eq, ranked))
    return per_entity


def _rerank_per_entity(per_entity_candidates, raw_affiliation, reranker, ror_id_to_text):
    """
    Rerank each entity's candidates using the full raw_affiliation as context.

    Input:  list of (EntityQuery, [(ror_id, ret_score), ...])
    Output: list of (EntityQuery, [(ror_id, rr_score), ...])
    """
    result = []
    for eq, candidates in per_entity_candidates:
        if not candidates or not raw_affiliation:
            result.append((eq, candidates))
            continue

        pairs = [(raw_affiliation, ror_id_to_text.get(rid, rid)) for rid, _ in candidates]
        scores = reranker.score_pairs(pairs)
        reranked = sorted(zip([rid for rid, _ in candidates], scores), key=lambda x: -x[1])
        result.append((eq, reranked))
    return result


def _aggregate_per_entity_top1(per_entity_reranked):
    """
    From per-entity reranked results, take top-1 per entity.
    Returns: list of (ror_id, score) — one per entity, deduplicated by max score.
    """
    score_map = {}
    for eq, ranked in per_entity_reranked:
        if ranked:
            rid, sc = ranked[0]
            if rid not in score_map or sc > score_map[rid]:
                score_map[rid] = sc
    return sorted(score_map.items(), key=lambda x: -x[1])


def _merge_all_reranked(per_entity_reranked, k=10):
    """
    Merge all candidates from all entities, keep max score per ror_id.
    Returns: [(ror_id, score), ...] sorted desc, up to k.
    """
    score_map = {}
    for eq, ranked in per_entity_reranked:
        for rid, sc in ranked:
            if rid not in score_map or sc > score_map[rid]:
                score_map[rid] = sc
    return sorted(score_map.items(), key=lambda x: -x[1])[:k]


def retrieve_and_rerank_for_row(row, retriever, ret_cfg, reranker, ror_id_to_text, k=10):
    """
    Full per-entity pipeline for one row:
      1. Build entity queries from NER (with geo variants)
      2. Retrieve per entity (merging across variants)
      3. Rerank per entity using raw affiliation as context
      4. Return per-entity results + aggregated top-1s

    Returns:
      - per_entity_retrieval:  list of (EntityQuery, [(ror_id, ret_score)])
      - per_entity_reranked:   list of (EntityQuery, [(ror_id, rr_score)])
      - final_predictions:     list of (ror_id, score) — top-1 per entity, deduplicated
    """
    entity_queries = build_queries_for_row(row)
    if not entity_queries:
        return [], [], []

    # Step 1: retrieve per entity
    per_entity_ret = _retrieve_per_entity(entity_queries, retriever, ret_cfg["query_format"], k)

    # Step 2: get raw affiliation for reranker context
    raw = row.get("raw_affiliation_string", "")
    if not raw or str(raw) in ("None", "nan", ""):
        flat = _flatten_entity_queries(entity_queries)
        raw = format_query_plain(flat[0]) if flat else ""

    # Step 3: rerank per entity
    per_entity_rr = _rerank_per_entity(per_entity_ret, raw, reranker, ror_id_to_text)

    # Step 4: aggregate top-1 per entity
    final_preds = _aggregate_per_entity_top1(per_entity_rr)

    return per_entity_ret, per_entity_rr, final_preds


# ================================================================
# 12.  TRAIN / TEST SPLIT
# ================================================================

def split_train_test(df_all, test_size=0.5, seed=42):
    """
    Split 50% per dataset, stratified by dataset.
    Returns df_train, df_test.
    """
    rng = np.random.RandomState(seed)
    train_parts, test_parts = [], []
    for ds in df_all["dataset"].unique():
        sub = df_all[df_all["dataset"] == ds].copy()
        idx = rng.permutation(len(sub))
        split_at = len(sub) // 2
        train_parts.append(sub.iloc[idx[:split_at]])
        test_parts.append(sub.iloc[idx[split_at:]])
    df_train = pd.concat(train_parts, ignore_index=True)
    df_test  = pd.concat(test_parts, ignore_index=True)
    print(f"  Split: train={len(df_train)}, test={len(df_test)}")
    for ds in sorted(df_all["dataset"].unique()):
        n_tr = (df_train["dataset"] == ds).sum()
        n_te = (df_test["dataset"] == ds).sum()
        print(f"    {ds}: train={n_tr}, test={n_te}")
    return df_train, df_test


# ================================================================
# 13.  THRESHOLD TUNING & METRICS
# ================================================================

def _run_reranking_on_split(
    df_split, retriever, ret_cfg, reranker, ror_id_to_text, k=10,
    desc="reranking",
):
    """
    Run per-entity retrieval + reranking on a DataFrame split.

    For each row:
      1. Build entity queries (with geo variants)
      2. Retrieve per entity (merging variants)
      3. Rerank per entity (using raw affiliation)
      4. Merge all reranked candidates → global top-1 + score

    Returns: list of dicts with keys:
      - golds, top1_id, top1_score, gold_in_candidates, per_entity_predictions
    """
    results = []
    for _, row in tqdm(df_split.iterrows(), total=len(df_split),
                       desc=desc, leave=False):
        golds = row.get("ror_all", [])
        if not isinstance(golds, list):
            golds = []

        per_e_ret, per_e_rr, final_preds = retrieve_and_rerank_for_row(
            row, retriever, ret_cfg, reranker, ror_id_to_text, k,
        )

        # Gold in candidates: check across ALL per-entity retrievals
        all_cand_ids = set()
        for eq, cands in per_e_ret:
            for rid, _ in cands:
                all_cand_ids.add(rid)
        gold_in_cands = any(g in all_cand_ids for g in golds) if golds else False

        # Global merged reranked → single top-1 for threshold tuning
        merged = _merge_all_reranked(per_e_rr, k=k)
        top1_id = merged[0][0] if merged else None
        top1_score = merged[0][1] if merged else 0.0

        results.append({
            "golds": golds,
            "top1_id": top1_id,
            "top1_score": top1_score,
            "gold_in_candidates": gold_in_cands,
            "per_entity_predictions": final_preds,
        })
    return results


def find_best_threshold(reranking_results):
    """
    Find the threshold that maximizes Acc@1 on the dev/train split.

    For each candidate threshold t:
      - If top1_score >= t → predict top1_id
      - If top1_score <  t → predict None
      - Correct if: (predict == gold) OR (predict None AND gold not in candidates)
    """
    rows_with_gold = [r for r in reranking_results if r["golds"]]
    if not rows_with_gold:
        return 0.0

    # Collect all unique scores as candidate thresholds
    all_scores = sorted(set(r["top1_score"] for r in rows_with_gold))
    # Add boundary values
    thresholds = [all_scores[0] - 0.01] + all_scores + [all_scores[-1] + 0.01]

    best_t, best_acc = 0.0, -1.0
    for t in thresholds:
        correct = 0
        for r in rows_with_gold:
            if r["top1_score"] >= t:
                # Predict top1_id
                if r["top1_id"] in set(r["golds"]):
                    correct += 1
            else:
                # Predict None → correct only if gold not in candidates
                if not r["gold_in_candidates"]:
                    correct += 1
        acc = correct / len(rows_with_gold)
        if acc > best_acc:
            best_acc = acc
            best_t = t

    return best_t


def compute_reranking_metrics(reranking_results, threshold=None):
    """
    Compute reranking metrics.

    Returns dict:
      - ceiling:    fraction of rows where gold is in retrieved candidates (retriever ceiling)
      - acc_at_1:   Acc@1 after reranking (with optional threshold)
      - noc_precision: P(gold not in candidates | predict None)
      - noc_recall:    P(predict None | gold not in candidates)
      - n, n_with_gold
    """
    rows = [r for r in reranking_results if r["golds"]]
    n = len(rows)
    if n == 0:
        return {"ceiling": 0, "acc_at_1": 0, "noc_precision": 0, "noc_recall": 0,
                "n": 0, "n_with_gold": 0}

    # Ceiling
    n_gold_in_cands = sum(1 for r in rows if r["gold_in_candidates"])
    ceiling = n_gold_in_cands / n

    # Counts
    tp_match = 0   # predict correctly
    tp_none = 0    # predict None AND gold not in candidates
    fp_none = 0    # predict None AND gold IS in candidates (missed it)
    fn_none = 0    # predict something AND gold NOT in candidates
    total = 0

    for r in rows:
        total += 1
        gold_set = set(r["golds"])
        use_threshold = threshold is not None

        if use_threshold and r["top1_score"] < threshold:
            # Predict None
            if not r["gold_in_candidates"]:
                tp_none += 1  # correctly rejected
            else:
                fp_none += 1  # should have predicted, but rejected
        else:
            # Predict top1_id
            if r["top1_id"] in gold_set:
                tp_match += 1
            elif not r["gold_in_candidates"]:
                fn_none += 1  # gold not retrievable, but we predicted something

    acc_at_1 = (tp_match + tp_none) / total if total else 0.0

    # NoC = None-of-Candidates
    noc_precision = tp_none / (tp_none + fp_none) if (tp_none + fp_none) > 0 else 0.0
    noc_recall = tp_none / (tp_none + fn_none) if (tp_none + fn_none) > 0 else 0.0

    return {
        "ceiling": ceiling,
        "acc_at_1": acc_at_1,
        "acc_at_1_no_threshold": tp_match / total if total else 0.0,
        "noc_precision": noc_precision,
        "noc_recall": noc_recall,
        "n": total,
        "n_with_gold": n_gold_in_cands,
        "threshold": threshold,
    }


# ================================================================
# 14.  FULL EXPERIMENT RUNNER  (retrieval + reranking)
# ================================================================

@dataclass
class FullExperimentResult:
    retriever: str
    reranker: str
    dataset: str
    n_samples: int
    # retrieval
    recall_at_1: float
    recall_at_5: float
    recall_at_10: float
    # reranking
    ceiling: float
    acc_at_1: float
    acc_at_1_no_threshold: float
    noc_precision: float
    noc_recall: float
    threshold: Optional[float]


def preview_reranking(
    df_all: pd.DataFrame,
    df_ror_kb: pd.DataFrame,
    retriever_name: str = "whoosh",
    reranker_name: str = "cross_encoder_comet",
    k: int = 5,
    n_per_dataset: int = 1,
    seed: int = 42,
    device: str = None,
    batch_size: int = 32,
):
    """
    Preview per-entity retrieval + reranking on a few examples per dataset.

    For each example shows:
      - Raw affiliation string
      - NER source (ner / ner_pred) and entities
      - Per entity: query variants, retrieval top-k (merged), reranked top-k
      - Final: aggregated matched ROR IDs for the whole affiliation
    """
    # Build retriever
    ret_cfg = RETRIEVER_CONFIGS[retriever_name]
    retriever, ret_cfg = build_retriever(retriever_name, device=device, batch_size=batch_size)
    retriever.fit(
        df_ror_kb[ret_cfg["kb_field"]].tolist(),
        df_ror_kb["ror_id"].tolist(),
    )

    # Build reranker
    reranker = build_reranker(reranker_name, device=device, batch_size=batch_size)
    ror_id_to_text = build_reranking_kb_lookup(df_ror_kb)
    id2name = dict(zip(df_ror_kb["ror_id"], df_ror_kb["name"]))

    # Query formatter
    fmt_fn = format_query_tagged if ret_cfg["query_format"] == "tagged" else format_query_plain

    # Pick examples
    rng = np.random.RandomState(seed)
    examples = []
    for ds in sorted(df_all["dataset"].unique()):
        sub = df_all[df_all["dataset"] == ds]
        good = sub[sub.apply(lambda r: len(build_queries_for_row(r)) > 0, axis=1)]
        pool = good if len(good) >= n_per_dataset else sub
        chosen = pool.sample(n=min(n_per_dataset, len(pool)), random_state=rng)
        for idx, row in chosen.iterrows():
            examples.append((idx, row))

    print(f"\n{'━'*80}")
    print(f"  RETRIEVER: {retriever_name}  →  RERANKER: {reranker_name}")
    print(f"{'━'*80}")

    for idx, row in examples:
        ds    = row["dataset"]
        golds = row.get("ror_all", [])
        if not isinstance(golds, list):
            golds = []
        gold_set = set(golds)

        # ── Header ──
        print(f"\n  ┌─ [{ds}]  row={idx}")
        raw = row.get("raw_affiliation_string", "")
        if raw and str(raw) not in ("None", "nan", ""):
            print(f"  │  raw: {str(raw)[:140]}")

        # ── NER source ──
        ner_src = "ner_pred"
        ner_used = row.get("ner_pred")
        if ner_used is None or (isinstance(ner_used, float) and pd.isna(ner_used)):
            ner_used = row.get("ner")
            ner_src = "ner (oracle)"
        print(f"  │  {ner_src}: {ner_used}")

        # ── Gold ──
        gold_strs = [f"{g} → {id2name.get(g, '?')}" for g in golds]
        print(f"  │  gold: {gold_strs}")

        # ── Build entity queries ──
        entity_queries = build_queries_for_row(row)
        if not entity_queries:
            print(f"  │  ⚠ no queries could be built")
            print(f"  └─")
            continue

        # ── Per-entity retrieval (merge across variants) ──
        per_e_ret = _retrieve_per_entity(entity_queries, retriever, ret_cfg["query_format"], k)

        # ── Per-entity reranking ──
        raw_for_rr = raw if (raw and str(raw) not in ("None", "nan", "")) else ""
        if not raw_for_rr:
            flat = _flatten_entity_queries(entity_queries)
            raw_for_rr = format_query_plain(flat[0]) if flat else ""
        per_e_rr = _rerank_per_entity(per_e_ret, raw_for_rr, reranker, ror_id_to_text)

        # ── Display per entity ──
        n_entities = len(entity_queries)
        for ei, ((eq, ret_cands), (_, rr_cands)) in enumerate(zip(per_e_ret, per_e_rr)):
            print(f"  │")
            print(f"  │  ── entity [{ei+1}/{n_entities}]: {eq['entity']}")

            # Show query variants
            for vi, v in enumerate(eq["variants"]):
                print(f"  │     variant {vi+1}: {fmt_fn(v)}")

            # Retrieval candidates (merged across variants)
            print(f"  │     retrieval (merged):")
            for rank, (rid, sc) in enumerate(ret_cands[:k], 1):
                hit = "✅" if rid in gold_set else "  "
                print(f"  │       {hit} {rank}. {id2name.get(rid,'?'):<42s}  ret={sc:.4f}")

            # Reranked candidates
            print(f"  │     reranked:")
            for rank, (rid, sc) in enumerate(rr_cands[:k], 1):
                hit = "✅" if rid in gold_set else "  "
                print(f"  │       {hit} {rank}. {id2name.get(rid,'?'):<42s}  rr={sc:.4f}")
            # Show what reranker sees for top-1
            if rr_cands:
                top_rid = rr_cands[0][0]
                print(f"  │     (reranker doc: {ror_id_to_text.get(top_rid, '?')[:80]})")

        # ── Final aggregated predictions ──
        final_preds = _aggregate_per_entity_top1(per_e_rr)
        print(f"  │")
        print(f"  │  ── FINAL MATCHED ({len(final_preds)} org{'s' if len(final_preds)!=1 else ''}):")
        matched_golds = set()
        for rid, sc in final_preds:
            hit = "✅" if rid in gold_set else "❌"
            if rid in gold_set:
                matched_golds.add(rid)
            print(f"  │     {hit} {rid}  {id2name.get(rid,'?'):<42s}  score={sc:.4f}")

        # Show unmatched golds
        missed = gold_set - matched_golds
        if missed:
            print(f"  │  ── MISSED gold ROR IDs:")
            for g in missed:
                all_ret_ids = set()
                for eq, cands in per_e_ret:
                    for rid, _ in cands:
                        all_ret_ids.add(rid)
                where = "not retrieved" if g not in all_ret_ids else "retrieved but not ranked #1"
                print(f"  │     ⚠ {g}  {id2name.get(g,'?'):<42s}  ({where})")

        print(f"  └─")

    # Cleanup
    reranker.free(); del reranker
    retriever.free(); del retriever
    _flush_gpu()

    print(f"\n{'━'*80}")
    print("  Done preview.")
    print(f"{'━'*80}")


def run_reranking_experiment(
    df_all: pd.DataFrame,
    df_ror_kb: pd.DataFrame,
    retriever_name: str,
    reranker_name: str,
    k: int = 10,
    test_size: float = 0.5,
    seed: int = 42,
    device: str = None,
    batch_size: int = 32,
    output_csv: str = None,
):
    """
    Full pipeline: retrieval → reranking → threshold tuning → evaluation.

    1. Split data 50/50 per dataset
    2. Build retriever, fit on KB
    3. Compute pure retrieval metrics on test (R@1, R@5, R@10)
    4. Build reranker
    5. Run reranking on train → tune threshold
    6. Run reranking on test → compute Acc@1, NoC P/R
    7. Report per-dataset + ALL
    """
    print(f"\n{'='*70}")
    print(f"  RETRIEVER: {retriever_name}  →  RERANKER: {reranker_name}")
    print(f"{'='*70}")

    # --- Split ---
    df_train, df_test = split_train_test(df_all, test_size=test_size, seed=seed)

    # --- Build retriever ---
    ret_cfg = RETRIEVER_CONFIGS[retriever_name]
    retriever, ret_cfg = build_retriever(retriever_name, device=device, batch_size=batch_size)
    retriever.fit(
        df_ror_kb[ret_cfg["kb_field"]].tolist(),
        df_ror_kb["ror_id"].tolist(),
    )

    # --- Pure retrieval on test (for R@1/5/10 reporting) ---
    print(f"\n  📊 Pure retrieval metrics on test set:")
    retrieval_metrics = {}
    for ds in sorted(df_test["dataset"].unique().tolist()) + ["ALL"]:
        sub = df_test if ds == "ALL" else df_test[df_test["dataset"] == ds]
        all_golds, all_preds = [], []
        for _, row in sub.iterrows():
            golds = row.get("ror_all", [])
            if not isinstance(golds, list): golds = []
            queries = build_queries_for_row(row)
            if not queries:
                all_golds.append(golds); all_preds.append([])
                continue
            ranked = _merge_retrieval(queries, retriever, ret_cfg["query_format"], k)
            all_golds.append(golds)
            all_preds.append([rid for rid, _ in ranked])
        r1 = recall_at_k(all_golds, all_preds, 1)
        r5 = recall_at_k(all_golds, all_preds, 5)
        r10 = recall_at_k(all_golds, all_preds, 10)
        retrieval_metrics[ds] = {"R@1": r1, "R@5": r5, "R@10": r10}
        tag = "📊" if ds == "ALL" else "  "
        print(f"  {tag} {ds:25s}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

    # --- Build reranker ---
    print(f"\n  🔄 Loading reranker: {reranker_name}")
    reranker = build_reranker(reranker_name, device=device, batch_size=batch_size)
    ror_id_to_text = build_reranking_kb_lookup(df_ror_kb)

    # --- Rerank on train → tune threshold ---
    print(f"  ⏳ Reranking train set for threshold tuning …")
    train_rr = _run_reranking_on_split(df_train, retriever, ret_cfg, reranker, ror_id_to_text, k,
                                        desc=f"  train {reranker_name}")
    best_threshold = find_best_threshold(train_rr)
    print(f"  🎯 Best threshold (tuned on train): {best_threshold:.4f}")

    # --- Rerank on test → evaluate ---
    print(f"  ⏳ Reranking test set …")
    test_rr_all = _run_reranking_on_split(df_test, retriever, ret_cfg, reranker, ror_id_to_text, k,
                                          desc=f"  test {reranker_name}")

    # Attach dataset labels for per-dataset eval
    test_datasets = df_test["dataset"].tolist()

    # --- Report per-dataset + ALL ---
    results = []
    print(f"\n  📊 Reranking results (threshold={best_threshold:.4f}):")
    for ds in sorted(df_test["dataset"].unique().tolist()) + ["ALL"]:
        if ds == "ALL":
            rr_subset = test_rr_all
        else:
            rr_subset = [r for r, d in zip(test_rr_all, test_datasets) if d == ds]

        metrics = compute_reranking_metrics(rr_subset, threshold=best_threshold)
        ret_m = retrieval_metrics.get(ds, {"R@1": 0, "R@5": 0, "R@10": 0})

        tag = "📊" if ds == "ALL" else "  "
        print(f"  {tag} {ds:25s}  "
              f"ceil={metrics['ceiling']:.3f}  "
              f"Acc@1={metrics['acc_at_1']:.3f}  "
              f"(noT={metrics['acc_at_1_no_threshold']:.3f})  "
              f"NoC_P={metrics['noc_precision']:.3f}  "
              f"NoC_R={metrics['noc_recall']:.3f}")

        results.append(FullExperimentResult(
            retriever=retriever_name, reranker=reranker_name,
            dataset=ds, n_samples=metrics["n"],
            recall_at_1=ret_m["R@1"], recall_at_5=ret_m["R@5"], recall_at_10=ret_m["R@10"],
            ceiling=metrics["ceiling"],
            acc_at_1=metrics["acc_at_1"],
            acc_at_1_no_threshold=metrics["acc_at_1_no_threshold"],
            noc_precision=metrics["noc_precision"],
            noc_recall=metrics["noc_recall"],
            threshold=best_threshold,
        ))

    # --- Cleanup ---
    reranker.free(); del reranker
    retriever.free(); del retriever
    _flush_gpu()

    # --- Save ---
    df_results = pd.DataFrame([vars(r) for r in results])
    if output_csv:
        df_results.to_csv(output_csv, index=False)
        print(f"\n  💾 Saved → {output_csv}")

    return df_results


def run_all_reranking_experiments(
    df_all: pd.DataFrame,
    df_ror_kb: pd.DataFrame,
    retriever_names: List[str] = None,
    reranker_names: List[str] = None,
    k: int = 10,
    test_size: float = 0.5,
    seed: int = 42,
    device: str = None,
    batch_size: int = 32,
    output_csv: str = "reranking_results.csv",
):
    """
    Run all retriever × reranker combinations.

    Retriever is loaded once, all rerankers run against it, then freed.
    This minimizes model loading overhead.
    """
    if retriever_names is None:
        retriever_names = list(RETRIEVER_CONFIGS.keys())
    if reranker_names is None:
        reranker_names = list(RERANKER_CONFIGS.keys())

    # --- Split once (same split for all experiments) ---
    df_train, df_test = split_train_test(df_all, test_size=test_size, seed=seed)

    ror_id_to_text = build_reranking_kb_lookup(df_ror_kb)
    id2name = dict(zip(df_ror_kb["ror_id"], df_ror_kb["name"]))
    test_datasets = df_test["dataset"].tolist()

    all_results = []

    for ret_name in retriever_names:
        print(f"\n{'='*70}")
        print(f"  📦 RETRIEVER: {ret_name}")
        print(f"{'='*70}")

        # Build and fit retriever
        try:
            ret_cfg = RETRIEVER_CONFIGS[ret_name]
            retriever, ret_cfg = build_retriever(ret_name, device=device, batch_size=batch_size)
            retriever.fit(
                df_ror_kb[ret_cfg["kb_field"]].tolist(),
                df_ror_kb["ror_id"].tolist(),
            )
        except Exception as e:
            print(f"  ❌ Failed to build retriever {ret_name}: {e}")
            _flush_gpu()
            continue

        # --- Pure retrieval metrics on test ---
        print(f"\n  📊 Retrieval metrics:")
        retrieval_metrics = {}
        for ds in sorted(df_test["dataset"].unique().tolist()) + ["ALL"]:
            sub = df_test if ds == "ALL" else df_test[df_test["dataset"] == ds]
            all_golds, all_preds = [], []
            for _, row in sub.iterrows():
                golds = row.get("ror_all", [])
                if not isinstance(golds, list): golds = []
                queries = build_queries_for_row(row)
                if not queries:
                    all_golds.append(golds); all_preds.append([])
                    continue
                ranked = _merge_retrieval(queries, retriever, ret_cfg["query_format"], k)
                all_golds.append(golds)
                all_preds.append([rid for rid, _ in ranked])
            r1 = recall_at_k(all_golds, all_preds, 1)
            r5 = recall_at_k(all_golds, all_preds, 5)
            r10 = recall_at_k(all_golds, all_preds, 10)
            retrieval_metrics[ds] = {"R@1": r1, "R@5": r5, "R@10": r10}
            tag = "📊" if ds == "ALL" else "  "
            print(f"  {tag} {ds:25s}  R@1={r1:.4f}  R@5={r5:.4f}  R@10={r10:.4f}")

        # --- For each reranker ---
        for rr_name in reranker_names:
            print(f"\n  🔄 Reranker: {rr_name}")
            reranker = None

            try:
                reranker = build_reranker(rr_name, device=device, batch_size=batch_size)

                # Train: rerank + tune threshold
                print(f"    ⏳ Threshold tuning (train) …")
                train_rr = _run_reranking_on_split(
                    df_train, retriever, ret_cfg, reranker, ror_id_to_text, k,
                    desc=f"  train {rr_name}",
                )
                best_t = find_best_threshold(train_rr)
                print(f"    🎯 t*={best_t:.4f}")

                # Test: rerank + evaluate
                print(f"    ⏳ Evaluating (test) …")
                test_rr = _run_reranking_on_split(
                    df_test, retriever, ret_cfg, reranker, ror_id_to_text, k,
                    desc=f"  test {rr_name}",
                )

                # Per-dataset + ALL
                for ds in sorted(df_test["dataset"].unique().tolist()) + ["ALL"]:
                    if ds == "ALL":
                        rr_sub = test_rr
                    else:
                        rr_sub = [r for r, d in zip(test_rr, test_datasets) if d == ds]

                    m = compute_reranking_metrics(rr_sub, threshold=best_t)
                    ret_m = retrieval_metrics.get(ds, {"R@1": 0, "R@5": 0, "R@10": 0})

                    tag = "📊" if ds == "ALL" else "    "
                    print(f"  {tag} {ds:25s}  "
                          f"ceil={m['ceiling']:.3f}  "
                          f"Acc@1={m['acc_at_1']:.3f}  "
                          f"noT={m['acc_at_1_no_threshold']:.3f}  "
                          f"NocP={m['noc_precision']:.3f}  NocR={m['noc_recall']:.3f}")

                    all_results.append(FullExperimentResult(
                        retriever=ret_name, reranker=rr_name,
                        dataset=ds, n_samples=m["n"],
                        recall_at_1=ret_m["R@1"], recall_at_5=ret_m["R@5"],
                        recall_at_10=ret_m["R@10"],
                        ceiling=m["ceiling"],
                        acc_at_1=m["acc_at_1"],
                        acc_at_1_no_threshold=m["acc_at_1_no_threshold"],
                        noc_precision=m["noc_precision"],
                        noc_recall=m["noc_recall"],
                        threshold=best_t,
                    ))

            except torch.cuda.OutOfMemoryError:
                print(f"    ⚠️  CUDA OOM on {rr_name} — skipping")
                _flush_gpu()
            except Exception as e:
                print(f"    ❌ {rr_name}: {e}")
                import traceback
                traceback.print_exc()
            finally:
                if reranker is not None:
                    reranker.free()
                    del reranker
                    reranker = None
                _flush_gpu()

        # Free retriever after all rerankers
        retriever.free()
        del retriever
        _flush_gpu()

    # --- Save combined results ---
    df_out = pd.DataFrame([vars(r) for r in all_results])
    if output_csv and len(df_out) > 0:
        df_out.to_csv(output_csv, index=False)
        print(f"\n💾 Saved → {output_csv}")

    # --- Summary table ---
    if len(df_out) > 0:
        summary = df_out[df_out["dataset"] == "ALL"][[
            "retriever", "reranker", "recall_at_1", "recall_at_10",
            "ceiling", "acc_at_1", "noc_precision", "noc_recall", "threshold",
        ]].round(4)
        print(f"\n{'='*70}")
        print("  SUMMARY (ALL datasets)")
        print(f"{'='*70}")
        print(summary.to_string(index=False))

    return df_out


# ================================================================
# 15.  HYBRID RETRIEVER (sparse + dense, RRF)
# ================================================================

class HybridRetriever(BaseRetriever):
    """
    Reciprocal Rank Fusion (RRF) combining two retrievers.

    RRF score(d) = sum_r 1 / (rrf_k + rank_r(d))
    where rrf_k=60 is the standard smoothing constant.
    """

    def __init__(self, retriever_a: BaseRetriever, retriever_b: BaseRetriever,
                 rrf_k: int = 60):
        self.ret_a = retriever_a
        self.ret_b = retriever_b
        self.rrf_k = rrf_k

    def fit(self, kb_texts, kb_ids):
        # Both retrievers should already be fit externally
        pass

    def retrieve(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        internal_k = k * 3  # over-fetch for fusion

        results_a = self.ret_a.retrieve(query, k=internal_k)
        results_b = self.ret_b.retrieve(query, k=internal_k)

        rrf_scores = {}
        for rank, (rid, _) in enumerate(results_a, 1):
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (self.rrf_k + rank)
        for rank, (rid, _) in enumerate(results_b, 1):
            rrf_scores[rid] = rrf_scores.get(rid, 0.0) + 1.0 / (self.rrf_k + rank)

        ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:k]
        return ranked

    def free(self):
        self.ret_a.free()
        self.ret_b.free()
        _flush_gpu()


def build_hybrid_retriever(df_ror_kb, device=None, batch_size=32):
    """
    Build a hybrid TF-IDF + AffilGood dense retriever.
    Returns (hybrid_retriever, ret_cfg) where ret_cfg has the KB/query format info.
    """
    # Build sparse
    ret_sparse, cfg_sparse = build_retriever("tfidf")
    ret_sparse.fit(
        df_ror_kb[cfg_sparse["kb_field"]].tolist(),
        df_ror_kb["ror_id"].tolist(),
    )

    # Build dense
    ret_dense, cfg_dense = build_retriever("affilgood_dense", device=device, batch_size=batch_size)
    ret_dense.fit(
        df_ror_kb[cfg_dense["kb_field"]].tolist(),
        df_ror_kb["ror_id"].tolist(),
    )

    hybrid = HybridRetriever(ret_sparse, ret_dense)
    # Use plain format (both sub-retrievers use plain)
    ret_cfg = {
        "class": "HybridRetriever",
        "kb_field": "plain_text",
        "query_format": "plain",
    }
    return hybrid, ret_cfg


# ================================================================
# 16.  DIRECT MATCH (exact name + country)
# ================================================================

def _normalize_country(c: str) -> str:
    """Normalize country to lowercase, handle common abbreviations."""
    if not c:
        return ""
    return c.strip().lower()


def build_direct_match_index(df_ror_kb: pd.DataFrame,
                             ror_records=None) -> Dict:
    """
    Build a lookup for exact string matching.

    Returns dict with:
      - name_country_to_rids: dict[(name_lower, country_lower)] → set[ror_id]
      - name_to_rids:         dict[name_lower] → set[ror_id]
      - ror_id_to_country:    dict[ror_id] → country_lower
      - ror_id_to_cc:         dict[ror_id] → country_code_lower

    Indexes: canonical name, aliases, labels — all lowercased.
    If ror_records is provided, also indexes acronyms as separate entries.
    """
    name_country_to_rids = {}  # (name_lower, country_lower) → {ror_ids}
    name_to_rids = {}          # name_lower → {ror_ids}
    ror_id_to_country = {}
    ror_id_to_cc = {}

    def _add_name(name_lower, rid, country, cc):
        """Helper: index a single name string."""
        if not name_lower:
            return
        if name_lower not in name_to_rids:
            name_to_rids[name_lower] = set()
        name_to_rids[name_lower].add(rid)
        if country:
            key = (name_lower, country)
            if key not in name_country_to_rids:
                name_country_to_rids[key] = set()
            name_country_to_rids[key].add(rid)
        if cc:
            key_cc = (name_lower, cc)
            if key_cc not in name_country_to_rids:
                name_country_to_rids[key_cc] = set()
            name_country_to_rids[key_cc].add(rid)

    # ── 1) Index all entry_names from the expanded KB ──
    for _, r in df_ror_kb.iterrows():
        rid = r["ror_id"]
        entry_name = str(r["entry_name"]).strip().lower()
        country = _normalize_country(str(r.get("country", "")))
        cc = str(r.get("country_code", "")).strip().lower()

        ror_id_to_country[rid] = country
        ror_id_to_cc[rid] = cc
        _add_name(entry_name, rid, country, cc)

    n_before = len(name_to_rids)

    # ── 2) Index acronyms from original ror_records ──
    n_acronyms_added = 0
    if ror_records is not None:
        for rec in ror_records:
            rid = normalize_ror_id(rec.id)
            acronyms = rec.acronyms or []
            country = ror_id_to_country.get(rid, "")
            cc = ror_id_to_cc.get(rid, "")

            for acr in acronyms:
                acr_lower = acr.strip().lower()
                if not acr_lower:
                    continue
                _add_name(acr_lower, rid, country, cc)
                n_acronyms_added += 1

    n_names = len(name_to_rids)
    n_name_country = len(name_country_to_rids)
    n_rids = len(ror_id_to_country)
    acr_msg = (f", +{n_acronyms_added} acronym entries "
               f"({n_names - n_before} new unique)" if ror_records else "")
    print(f"  ✅ Direct match index: {n_names} unique names, "
          f"{n_name_country} (name, country) pairs, {n_rids} ROR IDs{acr_msg}")

    return {
        "name_country_to_rids": name_country_to_rids,
        "name_to_rids": name_to_rids,
        "ror_id_to_country": ror_id_to_country,
        "ror_id_to_cc": ror_id_to_cc,
    }


def direct_match_for_entity(entity_query: Dict, dm_index: Dict) -> Optional[str]:
    """
    Try to directly match an entity to exactly one ROR ID.

    Strategy:
      1. Try (org_lower, country_lower) → if exactly 1 match → return it
      2. If 0 or 2+ matches with country → return None (ambiguous or not found)

    Only fires when country is available (from any variant).
    Returns: ror_id or None.
    """
    org = entity_query["entity"].strip().lower()
    nc = dm_index["name_country_to_rids"]

    # Extract country from variants (take first available)
    country = None
    for v in entity_query["variants"]:
        if v.get("country"):
            country = _normalize_country(v["country"])
            break

    if not country:
        return None  # can't do direct match without country

    # Try (name, country)
    matches = nc.get((org, country), set())
    if len(matches) == 1:
        return next(iter(matches))

    return None  # 0 or 2+ matches → ambiguous


def direct_match_for_row(row, dm_index: Dict) -> Dict:
    """
    Run direct match on all entities in a row.

    Returns:
      {
        "matched": [(entity_name, ror_id), ...],  # entities with exactly 1 match
        "unmatched": [EntityQuery, ...],           # entities needing retrieval
      }
    """
    entity_queries = build_queries_for_row(row)
    if not entity_queries:
        return {"matched": [], "unmatched": []}

    matched = []
    unmatched = []

    for eq in entity_queries:
        rid = direct_match_for_entity(eq, dm_index)
        if rid:
            matched.append((eq["entity"], rid))
        else:
            unmatched.append(eq)

    return {"matched": matched, "unmatched": unmatched}


def evaluate_direct_match(df_test, dm_index, id2name=None):
    """
    Evaluate direct match as a standalone baseline.

    For each row:
      - Each entity either matches exactly 1 ROR ID or is unmatched.
      - We measure: what fraction of gold ROR IDs are correctly matched?

    Returns dict with per-dataset and ALL metrics.
    """
    from collections import defaultdict
    stats = defaultdict(lambda: {"correct": 0, "incorrect": 0,
                                  "missed": 0, "total_gold": 0, "n_rows": 0})

    for _, row in tqdm(df_test.iterrows(), total=len(df_test),
                       desc="  direct match", leave=False):
        ds = row["dataset"]
        golds = row.get("ror_all", [])
        if not isinstance(golds, list):
            golds = []
        gold_set = set(golds)

        result = direct_match_for_row(row, dm_index)

        matched_ids = set()
        for _, rid in result["matched"]:
            matched_ids.add(rid)

        for bucket in [ds, "ALL"]:
            stats[bucket]["n_rows"] += 1
            stats[bucket]["total_gold"] += len(gold_set)
            for rid in matched_ids:
                if rid in gold_set:
                    stats[bucket]["correct"] += 1
                else:
                    stats[bucket]["incorrect"] += 1
            for g in gold_set:
                if g not in matched_ids:
                    stats[bucket]["missed"] += 1

    results = {}
    for ds, s in stats.items():
        precision = s["correct"] / (s["correct"] + s["incorrect"]) if (s["correct"] + s["incorrect"]) > 0 else 0
        recall = s["correct"] / s["total_gold"] if s["total_gold"] > 0 else 0
        coverage = (s["correct"] + s["incorrect"]) / s["total_gold"] if s["total_gold"] > 0 else 0
        results[ds] = {
            "precision": precision, "recall": recall, "coverage": coverage,
            "correct": s["correct"], "incorrect": s["incorrect"],
            "missed": s["missed"], "total_gold": s["total_gold"],
            "n_rows": s["n_rows"],
        }
    return results


# ================================================================
# 17.  CASCADE PIPELINE (direct match → hybrid + reranker)
# ================================================================

def cascade_for_row(
    row, dm_index, retriever, ret_cfg, reranker, ror_id_to_text, k=10,
):
    """
    Cascade pipeline for one row:
      1. Try direct match for each entity
      2. For unmatched entities → retrieve + rerank
      3. Combine: direct matches (score=1.0) + reranked results

    Returns:
      - final_predictions: list of (ror_id, score)
      - n_direct: how many entities resolved by direct match
      - n_retrieval: how many entities needed retrieval+reranking
      - all_candidate_ids: set of all ROR IDs ever considered (for gold_in_candidates)
    """
    entity_queries = build_queries_for_row(row)
    if not entity_queries:
        return [], 0, 0, set()

    # Step 1: direct match per entity
    direct_preds = {}    # ror_id → score (1.0)
    unmatched_eqs = []
    all_candidate_ids = set()

    for eq in entity_queries:
        rid = direct_match_for_entity(eq, dm_index)
        if rid:
            direct_preds[rid] = 1.0  # high confidence
            all_candidate_ids.add(rid)
        else:
            unmatched_eqs.append(eq)

    n_direct = len(entity_queries) - len(unmatched_eqs)
    n_retrieval = len(unmatched_eqs)

    # Step 2: retrieval + reranking for unmatched entities only
    rr_preds = {}
    if unmatched_eqs and retriever is not None and reranker is not None:
        # Retrieve per entity
        per_e_ret = _retrieve_per_entity(
            unmatched_eqs, retriever, ret_cfg["query_format"], k,
        )

        # Collect all retrieval candidates for gold_in_candidates check
        for eq, cands in per_e_ret:
            for rid, _ in cands:
                all_candidate_ids.add(rid)

        # Raw affiliation for reranker
        raw = row.get("raw_affiliation_string", "")
        if not raw or str(raw) in ("None", "nan", ""):
            flat = _flatten_entity_queries(entity_queries)
            raw = format_query_plain(flat[0]) if flat else ""

        # Rerank per entity
        per_e_rr = _rerank_per_entity(per_e_ret, raw, reranker, ror_id_to_text)

        # Take top-1 per entity
        for eq, ranked in per_e_rr:
            if ranked:
                rid, sc = ranked[0]
                if rid not in rr_preds or sc > rr_preds[rid]:
                    rr_preds[rid] = sc

    # Step 3: combine (direct matches take priority)
    all_preds = {}
    all_preds.update(rr_preds)
    all_preds.update(direct_preds)  # overwrite with higher-confidence direct

    final = sorted(all_preds.items(), key=lambda x: -x[1])
    return final, n_direct, n_retrieval, all_candidate_ids


def run_cascade_on_split(
    df_split, dm_index, retriever, ret_cfg, reranker, ror_id_to_text,
    k=10, desc="cascade",
):
    """
    Run cascade pipeline on a DataFrame split.
    Returns list of result dicts compatible with find_best_threshold / compute_reranking_metrics.
    """
    results = []
    total_direct = total_retrieval = 0

    for _, row in tqdm(df_split.iterrows(), total=len(df_split),
                       desc=desc, leave=False):
        golds = row.get("ror_all", [])
        if not isinstance(golds, list):
            golds = []

        final_preds, n_dm, n_rr, all_cand_ids = cascade_for_row(
            row, dm_index, retriever, ret_cfg, reranker, ror_id_to_text, k,
        )
        total_direct += n_dm
        total_retrieval += n_rr

        top1_id = final_preds[0][0] if final_preds else None
        top1_score = final_preds[0][1] if final_preds else 0.0

        gold_in_cands = any(g in all_cand_ids for g in golds) if golds else False

        results.append({
            "golds": golds,
            "top1_id": top1_id,
            "top1_score": top1_score,
            "gold_in_candidates": gold_in_cands,
            "per_entity_predictions": final_preds,
        })

    print(f"    entities resolved: {total_direct} direct, {total_retrieval} retrieval")
    return results


# ================================================================
# 18.  LLM LISTWISE RERANKER
# ================================================================
#
# Instead of scoring (query, candidate) pairs independently (pointwise),
# present ALL candidates to a small instruction-following LLM and let it
# choose the best match.  This addresses three blind spots of pointwise
# cross-encoders identified in the error analysis:
#   1. No cross-candidate comparison  (same-name disambiguation)
#   2. Retrieval signal discarded      (retriever-was-right cases)
#   3. Structured metadata as soft text (city/country as tokens)
#
# Inspired by CMC (Song et al., EMNLP 2024) and LELA (2025).
# Zero-shot — no training or threshold tuning required.
# ================================================================

LETTERS = "ABCDEFGHIJKLMNOPQRST"   # supports up to 20 candidates


class LLMListwiseReranker:
    """
    Listwise LLM reranker for entity linking.

    Given a raw affiliation string and K candidate ROR entries, the LLM
    sees ALL candidates at once and selects the best match.  Scoring is
    done via first-token logit probabilities over candidate letters (A–J),
    following the FIRST approach (Gangi Reddy et al., EMNLP 2024).

    Parameters
    ----------
    model_name : str
        HuggingFace model id.  Recommended:
        - "Qwen/Qwen2.5-3B-Instruct"  (~6 GB, multilingual)
        - "Qwen/Qwen3-4B"             (~8 GB, stronger reasoning)
        - "microsoft/Phi-3.5-mini-instruct"  (~8 GB)
    device : str | None
        "cuda", "cpu", or None (auto).
    max_tokens : int
        Max input tokens (prompt + candidates).
    """

    SYSTEM_PROMPT = (
        "You are an expert at linking research institution mentions in "
        "scientific publication affiliations to entries in the ROR "
        "(Research Organization Registry) database.\n"
        "You will be given an affiliation string, an entity mention to "
        "link, and a list of candidate ROR entries with their names, "
        "alternative names, city, and country.\n"
        "Answer with ONLY a single letter corresponding to the best "
        "matching entry, or N if none of the candidates match."
    )

    def __init__(self, model_name="Qwen/Qwen2.5-3B-Instruct",
                 device=None, max_tokens=2048):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        _flush_gpu()

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, padding_side="left",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype="auto",
            device_map="auto" if device is None else None,
        )
        if device is not None:
            self.model.to(device)
        self.model.eval()
        self.device = str(next(self.model.parameters()).device)

        # Pre-compute token IDs for candidate letters + "N" (none)
        self._letter_token_ids = {}
        for ch in LETTERS + "N":
            ids = self.tokenizer.encode(ch, add_special_tokens=False)
            self._letter_token_ids[ch] = ids[-1]   # last sub-token

        print(f"  ✅ LLM listwise reranker loaded: {model_name}  ({self.device})")

    # ---- prompt construction ----

    def _build_prompt(self, raw_affiliation, entity_name,
                      candidates: List[Tuple[str, str]]) -> str:
        """
        Build the user prompt with numbered candidates.

        candidates: [(ror_id, description_text), ...]
        """
        n = len(candidates)
        cand_lines = []
        for i, (_, desc) in enumerate(candidates):
            cand_lines.append(f"{LETTERS[i]}) {desc}")

        letter_range = f"A-{LETTERS[n-1]}" if n > 1 else "A"

        return (
            f'Given the affiliation string:\n'
            f'"{raw_affiliation}"\n\n'
            f'Which of the following ROR entries best matches '
            f'the entity "{entity_name}"?\n\n'
            + "\n".join(cand_lines) + "\n\n"
            f"Answer with ONLY a single letter ({letter_range}), "
            f"or N if none match."
        )

    # ---- scoring via first-token logits ----

    @torch.no_grad()
    def _score_candidates(self, prompt_text, n_candidates):
        """
        Forward-pass → extract logit distribution over candidate letters.

        Returns: list of float scores (one per candidate), plus none_score.
        """
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text},
        ]

        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True,
            max_length=self.max_tokens,
        ).to(self.device)

        logits = self.model(**inputs).logits[0, -1, :]   # (vocab,)

        # Gather logits for candidate letters + N
        letter_logits = []
        for i in range(n_candidates):
            tid = self._letter_token_ids[LETTERS[i]]
            letter_logits.append(logits[tid].item())

        none_logit = logits[self._letter_token_ids["N"]].item()

        # Softmax over candidates + none
        all_logits = torch.tensor(letter_logits + [none_logit])
        probs = torch.softmax(all_logits, dim=0).tolist()

        cand_probs = probs[:n_candidates]
        none_prob  = probs[n_candidates]
        return cand_probs, none_prob

    # ---- public API ----

    def rerank_entity(
        self,
        raw_affiliation: str,
        entity_name: str,
        candidates: List[Tuple[str, str]],   # (ror_id, description_text)
    ) -> List[Tuple[str, float]]:
        """
        Rerank candidates for a single entity mention.

        Returns: sorted list of (ror_id, score), highest first.
                 Score = softmax probability from the LLM's first-token logits.
        """
        if not candidates:
            return []

        # Limit to 20 candidates (letter limit)
        candidates = candidates[:min(len(candidates), len(LETTERS))]

        prompt = self._build_prompt(raw_affiliation, entity_name, candidates)
        cand_probs, none_prob = self._score_candidates(prompt, len(candidates))

        results = [(rid, prob) for (rid, _), prob in zip(candidates, cand_probs)]
        results.sort(key=lambda x: -x[1])
        return results

    def free(self):
        if hasattr(self, "model") and self.model is not None:
            try:
                self.model.cpu()
            except Exception:
                pass
            del self.model
            self.model = None
        if hasattr(self, "tokenizer"):
            del self.tokenizer
            self.tokenizer = None
        _flush_gpu()


def build_llm_reranker(model_name="Qwen/Qwen2.5-3B-Instruct",
                       device=None) -> LLMListwiseReranker:
    """Convenience factory."""
    return LLMListwiseReranker(model_name=model_name, device=device)


# ================================================================
# 19.  PER-ENTITY LLM RERANKING HELPER
# ================================================================

def _rerank_per_entity_llm(
    per_entity_candidates,
    raw_affiliation: str,
    llm_reranker: LLMListwiseReranker,
    ror_id_to_text: Dict[str, str],
):
    """
    Listwise LLM reranking — each entity's candidate list is presented
    as a single prompt so the model can compare candidates against each
    other (unlike the pointwise _rerank_per_entity).

    Input:  list of (EntityQuery, [(ror_id, ret_score), ...])
    Output: list of (EntityQuery, [(ror_id, llm_score), ...])
    """
    result = []
    for eq, candidates in per_entity_candidates:
        if not candidates or not raw_affiliation:
            result.append((eq, candidates))
            continue

        # Build (ror_id, text) pairs for the LLM
        cand_with_text = [
            (rid, ror_id_to_text.get(rid, rid)) for rid, _ in candidates
        ]
        entity_name = eq["entity"]

        reranked = llm_reranker.rerank_entity(
            raw_affiliation, entity_name, cand_with_text,
        )
        result.append((eq, reranked))
    return result


# ================================================================
# 20.  CASCADE v2  —  DM (+acronyms) → retriever → LLM listwise
# ================================================================

def cascade_for_row_llm(
    row, dm_index, retriever, ret_cfg,
    llm_reranker, ror_id_to_text, k=10,
):
    """
    Cascade pipeline v2 for one row:
      1. Try direct match for each entity (now includes acronyms)
      2. For unmatched entities → retrieve with affilgood_dense
      3. LLM listwise reranking (sees all candidates simultaneously)
      4. Combine: direct matches (score=1.0) + LLM picks

    Returns same tuple as cascade_for_row for compatibility:
      (final_predictions, n_direct, n_retrieval, all_candidate_ids)
    """
    entity_queries = build_queries_for_row(row)
    if not entity_queries:
        return [], 0, 0, set()

    # Step 1: direct match per entity
    direct_preds = {}
    unmatched_eqs = []
    all_candidate_ids = set()

    for eq in entity_queries:
        rid = direct_match_for_entity(eq, dm_index)
        if rid:
            direct_preds[rid] = 1.0
            all_candidate_ids.add(rid)
        else:
            unmatched_eqs.append(eq)

    n_direct = len(entity_queries) - len(unmatched_eqs)
    n_retrieval = len(unmatched_eqs)

    # Step 2: retrieval + LLM listwise reranking for unmatched
    rr_preds = {}
    if unmatched_eqs and retriever is not None and llm_reranker is not None:
        per_e_ret = _retrieve_per_entity(
            unmatched_eqs, retriever, ret_cfg["query_format"], k,
        )

        for eq, cands in per_e_ret:
            for rid, _ in cands:
                all_candidate_ids.add(rid)

        raw = row.get("raw_affiliation_string", "")
        if not raw or str(raw) in ("None", "nan", ""):
            flat = _flatten_entity_queries(entity_queries)
            raw = format_query_plain(flat[0]) if flat else ""

        # LLM listwise reranking (cross-candidate comparison)
        per_e_rr = _rerank_per_entity_llm(
            per_e_ret, raw, llm_reranker, ror_id_to_text,
        )

        for eq, ranked in per_e_rr:
            if ranked:
                rid, sc = ranked[0]
                if rid not in rr_preds or sc > rr_preds[rid]:
                    rr_preds[rid] = sc

    # Step 3: combine (direct matches take priority)
    all_preds = {}
    all_preds.update(rr_preds)
    all_preds.update(direct_preds)

    final = sorted(all_preds.items(), key=lambda x: -x[1])
    return final, n_direct, n_retrieval, all_candidate_ids


def run_cascade_llm_on_split(
    df_split, dm_index, retriever, ret_cfg,
    llm_reranker, ror_id_to_text,
    k=10, desc="cascade_llm",
):
    """
    Run cascade v2 (DM + retriever + LLM) on a DataFrame split.

    Returns list of result dicts compatible with compute_reranking_metrics.
    """
    results = []
    total_direct = total_retrieval = 0

    for _, row in tqdm(df_split.iterrows(), total=len(df_split),
                       desc=desc, leave=False):
        golds = row.get("ror_all", [])
        if not isinstance(golds, list):
            golds = []

        final_preds, n_dm, n_rr, all_cand_ids = cascade_for_row_llm(
            row, dm_index, retriever, ret_cfg,
            llm_reranker, ror_id_to_text, k,
        )
        total_direct += n_dm
        total_retrieval += n_rr

        top1_id = final_preds[0][0] if final_preds else None
        top1_score = final_preds[0][1] if final_preds else 0.0

        gold_in_cands = any(g in all_cand_ids for g in golds) if golds else False

        results.append({
            "golds": golds,
            "top1_id": top1_id,
            "top1_score": top1_score,
            "gold_in_candidates": gold_in_cands,
            "per_entity_predictions": final_preds,
        })

    print(f"    entities resolved: {total_direct} direct, {total_retrieval} retrieval")
    return results