"""
annotate_with_llm.py
====================
Script 2 of the reranker annotation pipeline.

Reads the CSV produced by prepare_candidates.py and, for each row,
asks an OpenAI model to pick which of the retrieved candidates is the
correct ROR match for the affiliation — or NONE if none of them is.

The result is a "gold"-annotated CSV usable as a reranker training/eval
set (the reranker's job is exactly this pick).

Input CSV (from Script 1) must have at least these columns:
    raw_affiliation_string, span_text, entity, affilgood_string,
    direct_match_ror_id, candidates_ror_ids, candidates_names,
    candidates_cities, candidates_countries, candidates_scores

Output CSV = input CSV + these columns:
    gold_ror_id        — ROR ID picked by the LLM, or "NONE"
    gold_name          — canonical name of gold pick (readability)
    gold_rank          — 1-based position of gold_ror_id in candidates list,
                         or 0 if "NONE", or -1 if direct-match short-circuit
    gold_in_topk       — True if gold_ror_id is somewhere in the candidates
    llm_rationale      — one-line reason returned by the model
    llm_model          — model name used
    llm_status         — "ok", "direct_match_short_circuit", "skipped_no_candidates",
                         "api_error", or "parse_error"

Prompt shape
------------
    Full affiliation context: <raw_affiliation_string>
    Affiliation to match:     <span_text>
      variant 1:              <affilgood_string>
      retrieval (top-K):
          1. 04rq5mt64  University of Maryland, Baltimore   (Baltimore, USA)  sim=0.8561
          2. 02qskvh78  University of Maryland, Baltimore County  ...
          ...
    Task: return the ROR ID of the correct match, or "NONE".

Usage
-----
    export OPENAI_API_KEY=sk-…
    python annotate_with_llm.py \
        --input candidates.csv \
        --output candidates_annotated.csv \
        --model gpt-4o-mini \
        --skip-direct-matches \
        --max-candidates 20

A checkpoint file (<output>.checkpoint.jsonl) is written as we go so the
run can be resumed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Annotate the candidate list with an LLM to produce a "
                    "gold-labelled reranker dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input", "-i", required=True,
                   help="CSV produced by prepare_candidates.py")
    p.add_argument("--output", "-o", required=True,
                   help="Annotated CSV path")
    p.add_argument("--model", default="gpt-4o-mini",
                   help="OpenAI chat model to use (gpt-4o-mini is cheap; "
                        "gpt-4o gives somewhat better disambiguation).")
    p.add_argument("--api-key", default=None,
                   help="OpenAI API key (defaults to $OPENAI_API_KEY).")
    p.add_argument("--max-candidates", type=int, default=20,
                   help="Show the top-N candidates to the LLM.")
    p.add_argument("--skip-direct-matches", action="store_true",
                   help="Don't call the LLM for rows with a direct_match_ror_id; "
                        "adopt that ID as gold directly. Cheaper, but skips "
                        "LLM verification of the direct-match rule.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--limit", type=int, default=None,
                   help="Annotate only first N rows (for testing).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the checkpoint file if it exists.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert annotator for ROR (Research Organization Registry) "
    "entity linking. Given an affiliation string and a ranked list of ROR "
    "candidates, your job is to decide which candidate — if any — is the "
    "correct institution that the affiliation refers to.\n\n"
    "Decision rules:\n"
    "  • Pick the ROR record that best matches the specific institution "
    "    named in the affiliation. Honour geographic signals (city, country) "
    "    and disambiguate system / parent / branch orgs correctly.\n"
    "  • Prefer the most specific correct match. A medical school affiliated "
    "    with a university maps to the university unless the medical school "
    "    itself has a ROR record in the candidate list.\n"
    "  • Do NOT invent ROR IDs. Only choose from the candidate list.\n"
    "  • If no candidate is correct, return ror_id = \"NONE\".\n"
    "Respond with JSON only, matching the schema given in the user message."
)


def render_candidates_block(cand_ids: List[str],
                            cand_names: List[str],
                            cand_cities: List[str],
                            cand_countries: List[str],
                            cand_scores: List[float],
                            max_n: int) -> str:
    lines = []
    for rank, (rid, nm, city, ctry, sc) in enumerate(
        zip(cand_ids, cand_names, cand_cities, cand_countries, cand_scores),
        start=1,
    ):
        if rank > max_n:
            break
        loc = ", ".join([x for x in (city, ctry) if x])
        loc_str = f"({loc})" if loc else ""
        lines.append(
            f"  {rank:>2}. {rid}  {nm}  {loc_str}  sim={sc:.4f}"
        )
    return "\n".join(lines)


def build_user_prompt(row: pd.Series, max_n: int) -> str:
    cand_ids   = json.loads(row["candidates_ror_ids"] or "[]")
    cand_names = json.loads(row["candidates_names"]   or "[]")
    cand_cit   = json.loads(row["candidates_cities"]  or "[]")
    cand_ctry  = json.loads(row["candidates_countries"] or "[]")
    cand_sc    = json.loads(row["candidates_scores"]  or "[]")

    cand_block = render_candidates_block(
        cand_ids, cand_names, cand_cit, cand_ctry, cand_sc, max_n,
    )

    parts = [
        f'Full affiliation context:\n  "{row["raw_affiliation_string"]}"',
        "",
        f'Affiliation to match (NER span):\n  "{row["span_text"]}"',
        "",
        f"ORG entity detected: {row['entity']}",
        f"Normalized affiliation: {row['affilgood_string']}",
        "",
        f"Top candidates (ranked by dense similarity, descending):",
        cand_block if cand_block else "  (none)",
        "",
        "Return JSON with this exact schema:",
        "  {",
        '    "ror_id": "<one of the candidate ROR IDs, or the string NONE>",',
        '    "rationale": "<one short sentence explaining the choice>"',
        "  }",
    ]
    return "\n".join(parts)


# ------------------------------------------------------------------
# OpenAI call
# ------------------------------------------------------------------

def call_openai(client, model: str, user_prompt: str,
                temperature: float, max_retries: int) -> Tuple[Dict, str]:
    """
    Returns (parsed_json_or_empty, status_string).
    status_string ∈ {"ok", "api_error", "parse_error"}.
    """
    last_err = ""
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt},
                ],
            )
            content = resp.choices[0].message.content or ""
            try:
                parsed = json.loads(content)
                return parsed, "ok"
            except json.JSONDecodeError as e:
                last_err = f"json parse: {e}; content[:200]={content[:200]!r}"
                # One more attempt, in case the model glitched
        except Exception as e:  # API / network / auth error
            last_err = f"api: {type(e).__name__}: {e}"
            time.sleep(min(2 ** attempt, 10))

    return {"_error": last_err}, ("parse_error" if "json parse" in last_err else "api_error")


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def main():
    args = parse_args()

    try:
        from openai import OpenAI
    except ImportError:
        print("❌ 'openai' package not installed.  pip install openai")
        sys.exit(1)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        # Try to load from a .env file in the current working directory
        # or in the parent of the input CSV. python-dotenv is a soft
        # dependency — if it's not installed, we fall through to the
        # existing error message suggesting shell sourcing.
        try:
            from dotenv import load_dotenv
            for p in [Path.cwd() / ".env",
                      Path(args.input).resolve().parent / ".env",
                      Path(args.input).resolve().parent.parent / ".env"]:
                if p.exists():
                    load_dotenv(p, override=False)
                    if args.verbose:
                        print(f"  → Loaded environment from {p}")
                    break
            api_key = os.environ.get("OPENAI_API_KEY")
        except ImportError:
            pass

    if not api_key:
        print("❌ No OpenAI API key found.")
        print("   Options:")
        print("     1. pass --api-key sk-...")
        print("     2. export OPENAI_API_KEY=sk-...")
        print("     3. put OPENAI_API_KEY=sk-... in a .env file "
              "(and: uv add python-dotenv)")
        print("     4. bash: set -a; source .env; set +a")
        sys.exit(1)
    client = OpenAI(api_key=api_key)

    # --- Read input ---
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    if args.limit:
        df = df.head(args.limit).copy()
    print(f"📥 {len(df):,} rows from {args.input}")

    # --- Checkpoint handling ---
    ckpt_path = Path(str(args.output) + ".checkpoint.jsonl")
    already: Dict[str, Dict] = {}  # row_id → annotation dict
    if args.resume and ckpt_path.exists():
        with ckpt_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    already[str(d["row_id"])] = d
                except Exception:
                    continue
        print(f"  ⤴ Resuming — {len(already):,} rows already annotated in {ckpt_path}")

    ckpt_f = ckpt_path.open("a", encoding="utf-8")

    # --- Annotate ---
    annotations: List[Dict] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="LLM annotation"):
        row_id_str = str(row["row_id"])

        # Resume hit?
        if row_id_str in already:
            annotations.append(already[row_id_str])
            continue

        out = {
            "row_id":        row["row_id"],
            "gold_ror_id":   "",
            "gold_name":     "",
            "gold_rank":     0,
            "gold_in_topk":  False,
            "llm_rationale": "",
            "llm_model":     args.model,
            "llm_status":    "ok",
        }

        cand_ids = json.loads(row["candidates_ror_ids"] or "[]")
        cand_nms = json.loads(row["candidates_names"]   or "[]")
        id2name_row = dict(zip(cand_ids, cand_nms))

        # 1. No candidates at all → nothing to annotate
        if not cand_ids and not row.get("direct_match_ror_id"):
            out["llm_status"] = "skipped_no_candidates"
            out["gold_ror_id"] = "NONE"
            annotations.append(out)
            ckpt_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            continue

        # 2. Direct match shortcut (optional)
        dm = (row.get("direct_match_ror_id") or "").strip()
        if args.skip_direct_matches and dm:
            out["gold_ror_id"]    = dm
            out["gold_name"]      = row.get("direct_match_name", "")
            out["gold_rank"]      = (cand_ids.index(dm) + 1) if dm in cand_ids else -1
            out["gold_in_topk"]   = dm in cand_ids
            out["llm_rationale"]  = "adopted from direct_match_ror_id (skip-direct-matches)"
            out["llm_status"]     = "direct_match_short_circuit"
            annotations.append(out)
            ckpt_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            continue

        # 3. Otherwise, call the LLM
        prompt = build_user_prompt(row, args.max_candidates)
        if args.verbose:
            print("\n" + "─" * 70)
            print(prompt)

        parsed, status = call_openai(
            client, args.model, prompt, args.temperature, args.max_retries,
        )

        if status != "ok":
            out["llm_status"]    = status
            out["llm_rationale"] = parsed.get("_error", "")
            out["gold_ror_id"]   = ""
            annotations.append(out)
            ckpt_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            ckpt_f.flush()
            continue

        gold = str(parsed.get("ror_id", "")).strip()
        rat  = str(parsed.get("rationale", "")).strip()

        # Validate: either "NONE" or one of the candidate ids
        if gold.upper() == "NONE" or gold == "":
            out["gold_ror_id"]   = "NONE"
            out["gold_rank"]     = 0
            out["gold_in_topk"]  = False
        elif gold in cand_ids:
            out["gold_ror_id"]   = gold
            out["gold_name"]     = id2name_row.get(gold, "")
            out["gold_rank"]     = cand_ids.index(gold) + 1
            out["gold_in_topk"]  = True
        else:
            # LLM hallucinated a ROR ID not in the candidate list
            out["gold_ror_id"]   = "NONE"
            out["gold_rank"]     = 0
            out["gold_in_topk"]  = False
            out["llm_status"]    = "parse_error"
            rat = f"model returned non-candidate id {gold!r}: " + rat

        out["llm_rationale"] = rat
        annotations.append(out)

        ckpt_f.write(json.dumps(out, ensure_ascii=False) + "\n")
        ckpt_f.flush()

    ckpt_f.close()

    # --- Merge annotations onto df and save ---
    ann_df = pd.DataFrame(annotations)
    # Ensure row_id is compatible for merge
    ann_df["row_id"] = ann_df["row_id"].astype(str)
    df["row_id"]     = df["row_id"].astype(str)

    out_df = df.merge(
        ann_df, on="row_id", how="left", suffixes=("", "_ann"),
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)

    # --- Summary ---
    n_total = len(out_df)
    n_gold  = int((out_df["gold_ror_id"].fillna("") != "").sum())
    n_none  = int((out_df["gold_ror_id"] == "NONE").sum())
    n_err   = int(out_df["llm_status"].isin(["api_error", "parse_error"]).sum())
    n_dm    = int((out_df["llm_status"] == "direct_match_short_circuit").sum())

    print(f"\n💾 Wrote {n_total:,} annotated rows → {args.output}")
    print(f"   • annotated with a gold ROR ID : {n_gold - n_none:,}")
    print(f"   • annotated as NONE            : {n_none:,}")
    print(f"   • direct-match short-circuit   : {n_dm:,}")
    print(f"   • errors                       : {n_err:,}")
    print(f"   (checkpoint preserved at {ckpt_path})")


if __name__ == "__main__":
    main()