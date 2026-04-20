"""
update_ror_registry.py
======================
Fetch the latest ROR data dump from Zenodo, extract it, and produce the
normalized `ror_records.jsonl` that the rest of this codebase consumes
(via RegistryManager / prepare_candidates.py).

Why not just call `RegistryManager.download_ror()`?
---------------------------------------------------
The Zenodo API call hard-coded in src/registry.py uses:
    https://zenodo.org/api/records?communities=ror-data&sort=mostrecent
but in Oct 2023 Zenodo upgraded and ROR documented the new path:
    https://zenodo.org/api/communities/ror-data/records?sort=newest
Plus, since v2.0 (Dec 2025), ROR dumps are **v2-schema only** (no more
parallel v1 files in the zip). This script uses the modern path and the
concept-DOI shortcut, which always resolves to the latest release.

Usage
-----
    python update_ror_registry.py \
        --data-dir data/registry \
        --verbose

    # Force re-download even if a dump is already present:
    python update_ror_registry.py --force

    # Already have a JSON dump on disk? Just normalize it:
    python update_ror_registry.py --skip-download \
        --dump path/to/v2.2-2026-01-29-ror-data.json

Output layout (matches the rest of the repo)
--------------------------------------------
    <data-dir>/ror/
        v2.2-2026-01-29-ror-data.zip        # downloaded archive (auto-deleted)
        v2.2-2026-01-29-ror-data.json       # extracted dump
        ror_records.jsonl                   # normalized, what everything reads
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from typing import Optional


# Concept DOI — always resolves to the latest ROR release.
ZENODO_CONCEPT_RECORD = "6347574"
ZENODO_API_BASE = "https://zenodo.org/api/records"

# A real user-agent is recommended; Zenodo sometimes rejects the python-requests
# default. This one is polite and identifies the project.
USER_AGENT = "ror-registry-updater/1.0 (+research use)"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch the latest ROR dump and normalize it.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir", default="data/registry",
                   help="Base registry directory. A 'ror/' subdirectory will "
                        "be created/used inside it.")
    p.add_argument("--src-path", default="src",
                   help="Path to the repo's src/ directory (for imports).")
    p.add_argument("--force", action="store_true",
                   help="Re-download even if a dump is already on disk.")
    p.add_argument("--skip-download", action="store_true",
                   help="Skip the Zenodo fetch; normalize an existing dump.")
    p.add_argument("--dump", default=None,
                   help="When --skip-download is set, path to the JSON dump "
                        "to normalize. Otherwise auto-detected in the ror/ dir.")
    p.add_argument("--api-token", default=None,
                   help="Optional Zenodo API token (raises rate limit).")
    p.add_argument("--timeout", type=int, default=120,
                   help="HTTP timeout in seconds for API + download.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


# ------------------------------------------------------------------
# Zenodo fetch
# ------------------------------------------------------------------

def _http_get_json(url: str, timeout: int, api_token: Optional[str]) -> dict:
    headers = {"Accept": "application/json", "User-Agent": USER_AGENT}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_download(url: str, out_path: Path, timeout: int,
                   api_token: Optional[str], verbose: bool):
    headers = {"User-Agent": USER_AGENT}
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        chunk = 1024 * 256
        downloaded = 0
        with open(out_path, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if verbose and total:
                    pct = downloaded / total * 100
                    print(f"\r  downloading… {downloaded/1e6:6.1f} MB / "
                          f"{total/1e6:6.1f} MB ({pct:5.1f}%)", end="", flush=True)
        if verbose and total:
            print()  # newline after progress


def fetch_latest_zenodo_metadata(timeout: int, api_token: Optional[str],
                                 verbose: bool) -> dict:
    """
    Use the Zenodo concept record (6347574) to find the latest version
    of the ROR data dump.
    """
    url = f"{ZENODO_API_BASE}/{ZENODO_CONCEPT_RECORD}"
    if verbose:
        print(f"  → Querying Zenodo concept record: {url}")
    # Small retry loop for transient 5xx / rate-limit hiccups.
    last_err = None
    for attempt in range(1, 4):
        try:
            return _http_get_json(url, timeout, api_token)
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 500, 502, 503, 504):
                wait = 2 ** attempt
                if verbose:
                    print(f"  ⚠ Zenodo returned HTTP {e.code}, retrying in {wait}s…")
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            last_err = e
            if verbose:
                print(f"  ⚠ Network error: {e}, retrying…")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to reach Zenodo after 3 attempts: {last_err}")


def download_latest_ror_dump(ror_dir: Path, timeout: int,
                             api_token: Optional[str], verbose: bool) -> Path:
    """Download the latest ROR dump .zip and extract the JSON. Returns the JSON path."""
    meta = fetch_latest_zenodo_metadata(timeout, api_token, verbose)

    version = (meta.get("metadata") or {}).get("version", "?")
    pub_date = (meta.get("metadata") or {}).get("publication_date", "?")
    resolved_id = meta.get("id", "?")
    if verbose:
        print(f"  ✅ Latest ROR release: {version}  "
              f"(published {pub_date}, Zenodo record {resolved_id})")

    files = meta.get("files", [])
    zip_entry = next(
        (f for f in files
         if (f.get("key", "") or f.get("filename", "")).endswith(".zip")),
        None,
    )
    if zip_entry is None:
        raise RuntimeError(f"No .zip file found in Zenodo record {resolved_id}. "
                           f"Files listed: {[f.get('key') for f in files]}")

    zip_name = zip_entry.get("key") or zip_entry.get("filename")
    zip_url = (zip_entry.get("links", {}) or {}).get("self")
    if not zip_url:
        # Construct the canonical URL if 'self' isn't present.
        zip_url = f"{ZENODO_API_BASE}/{resolved_id}/files/{zip_name}/content"

    zip_path = ror_dir / zip_name
    if verbose:
        print(f"  → Downloading {zip_url}")
    _http_download(zip_url, zip_path, timeout, api_token, verbose)

    # Extract the largest .json (the dump itself). In v2.0+ there's typically
    # only one v2 JSON; earlier releases had both v1 + v2 files.
    if verbose:
        print(f"  → Extracting {zip_path.name}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        json_entries = [n for n in zf.namelist() if n.endswith(".json")]
        if not json_entries:
            raise RuntimeError(f"No JSON inside {zip_path}")
        # Prefer v2 if present (files ending in .json that contain 'schema_v2'
        # or — in v2.0+ — just the version-prefixed name).
        json_entries.sort(key=lambda n: zf.getinfo(n).file_size, reverse=True)
        largest = json_entries[0]
        zf.extract(largest, str(ror_dir))
        extracted = ror_dir / largest

    # Tidy up the zip — the JSON is what we need.
    try:
        zip_path.unlink()
    except OSError:
        pass

    if verbose:
        size_mb = extracted.stat().st_size / 1e6
        print(f"  ✅ Extracted {extracted.name}  ({size_mb:.1f} MB)")

    return extracted


# ------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------

def normalize_dump(dump_path: Path, ror_dir: Path, src_path: str,
                   verbose: bool) -> Path:
    """
    Use the project's own RegistryManager to normalize the dump into
    ror_records.jsonl. This guarantees the output is compatible with
    everything else (build_direct_match_index, prepare_kb, etc.).
    """
    src = Path(src_path).resolve()
    if not src.exists():
        raise FileNotFoundError(f"--src-path does not exist: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    from registry import RegistryManager

    if verbose:
        print(f"  → Normalizing {dump_path.name} via RegistryManager")

    # RegistryManager wants its data_dir to *contain* a 'ror/' subdir.
    # Our dump is already in <data_dir>/ror/, so pass the parent.
    data_dir = ror_dir.parent
    mgr = RegistryManager(data_dir=str(data_dir), verbose=verbose)

    # ._normalize_ror_dump returns records; ._save_jsonl writes them.
    records = mgr._normalize_ror_dump(dump_path)
    jsonl_path = ror_dir / "ror_records.jsonl"
    mgr._save_jsonl(records, jsonl_path)

    n_active = sum(1 for r in records if r.status == "active")
    if verbose:
        print(f"  ✅ Wrote {len(records):,} records "
              f"({n_active:,} active) → {jsonl_path}")
    return jsonl_path


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    ror_dir = data_dir / "ror"
    ror_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Registry directory: {ror_dir.resolve()}")

    # --- Step 1: obtain the JSON dump ---
    if args.skip_download:
        if args.dump:
            dump_path = Path(args.dump).resolve()
        else:
            # Auto-detect in ror_dir
            candidates = sorted(
                [p for p in ror_dir.glob("*.json")
                 if p.name != "ror_records.jsonl"],
                key=lambda p: p.stat().st_mtime, reverse=True,
            )
            if not candidates:
                print("❌ --skip-download set, but no *.json dump found in "
                      f"{ror_dir}. Pass --dump <path>.")
                sys.exit(2)
            dump_path = candidates[0]
        if not dump_path.exists():
            print(f"❌ Dump file not found: {dump_path}")
            sys.exit(2)
        print(f"📄 Using existing dump: {dump_path.name}")

    else:
        # Skip download if we already have one and --force is not set
        existing = sorted(
            [p for p in ror_dir.glob("*.json")
             if p.name != "ror_records.jsonl"],
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        if existing and not args.force:
            dump_path = existing[0]
            print(f"📄 Found existing dump, skipping download: {dump_path.name}")
            print("   (use --force to re-download the latest release)")
        else:
            print("🌐 Fetching latest release from Zenodo…")
            try:
                dump_path = download_latest_ror_dump(
                    ror_dir=ror_dir,
                    timeout=args.timeout,
                    api_token=args.api_token,
                    verbose=args.verbose,
                )
            except Exception as e:
                print(f"\n❌ Download failed: {e}")
                print(
                    "\n   Manual fallback:"
                    "\n   1. Open https://doi.org/10.5281/zenodo.6347574 in a browser"
                    "\n   2. Download the latest v*-ror-data.zip"
                    f"\n   3. Extract the .json file into {ror_dir}"
                    "\n   4. Re-run this script with --skip-download"
                )
                sys.exit(1)

    # --- Step 2: normalize ---
    print("\n🧬 Normalizing dump into ror_records.jsonl…")
    jsonl_path = normalize_dump(
        dump_path=dump_path,
        ror_dir=ror_dir,
        src_path=args.src_path,
        verbose=args.verbose,
    )

    print(f"\n✅ Registry ready.")
    print(f"   Dump:        {dump_path}")
    print(f"   Normalized:  {jsonl_path}")
    print(
        f"\n→ Now run prepare_candidates.py with "
        f"--ror-records {jsonl_path.as_posix()}"
    )


if __name__ == "__main__":
    main()