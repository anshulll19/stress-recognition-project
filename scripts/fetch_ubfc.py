#!/usr/bin/env python3
"""UBFC-Phys bulk downloader (IEEE DataPort node 49099, storage node 3658).

The dataset page advertises 56 files `s1.zip`..`s56.zip`, but that is stale
description text. The files are actually served as an *extracted* S3 folder
tree, browsed through two authenticated JSON endpoints:

    GET /dataport/load-directory/{node}?prefix={prefix}
        -> {"html": "<li>...</li>"}  one level of the tree

    GET /dataport/s3-download-url/{node}?key={base64(s3_key)}
        -> {"status": "success", "download_url": "<presigned S3 URL>"}

Presigned URLs are short-lived, so one is minted immediately before each
transfer (and again before each resume attempt).

Per subject the payload splits sharply:
    ~220 KB  physiological  (bvp/eda CSV, self-reported anxiety, info)
    ~14 GB   video          (vid_sN_T1..T3.avi)

Usage:
    python scripts/fetch_ubfc.py --manifest-only     # build file index
    python scripts/fetch_ubfc.py --signals           # skip .avi (~12 MB total)
    python scripts/fetch_ubfc.py                     # everything (~790 GB)
    python scripts/fetch_ubfc.py --subjects 1-5      # subset

Re-running is safe: complete files are skipped, partial files resume.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

NODE = "3658"
DATASET_NODE = "49099"
BASE = "https://ieee-dataport.org"
ROOT = Path(os.environ.get("UBFC_DEST", r"H:/stress_datasets/data/ubfc-phys"))
COOKIE_FILE = Path(
    os.environ.get("UBFC_COOKIE_FILE", r"H:/stress_datasets/secrets/ieee_cookie.txt")
)
MANIFEST = ROOT / "manifest.json"
COMPLETED = ROOT / "completed.json"
LOG = ROOT / "download.log"

FOLDER_RE = re.compile(r'data-prefix="([^"]+)"')
FILE_RE = re.compile(r'data-download-url="([^"]+)"[^>]*>([^<]+)</a>')


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}  {msg}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def session() -> requests.Session:
    if not COOKIE_FILE.is_file() or not COOKIE_FILE.read_text().strip():
        sys.exit(f"ERROR: no IEEE session cookie at {COOKIE_FILE}")
    raw = COOKIE_FILE.read_text().strip()
    s = requests.Session()
    s.headers.update(
        {
            "Cookie": raw,
            "User-Agent": "Mozilla/5.0",
            "X-Requested-With": "XMLHttpRequest",
        }
    )
    return s


def list_dir(s: requests.Session, prefix: str) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Return (subfolder prefixes, [(download_url, filename, size_text)])."""
    r = s.get(
        f"{BASE}/dataport/load-directory/{NODE}",
        params={"prefix": prefix},
        timeout=90,
    )
    r.raise_for_status()
    markup = r.json().get("html", "")

    folders = [html.unescape(p) for p in FOLDER_RE.findall(markup)]

    files: list[tuple[str, str, str]] = []
    for raw_url, raw_label in FILE_RE.findall(markup):
        url = html.unescape(raw_url)
        label = html.unescape(raw_label).strip()
        # Labels look like "vid_s1_T1.avi (4.61 GB)"; split name from size.
        nm = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", label)
        files.append((url, nm.group(1) if nm else label, nm.group(2) if nm else ""))
    return folders, files


def walk(s: requests.Session, prefix: str, depth: int = 0) -> list[dict]:
    """Depth-first walk of one S3 prefix, returning flat file records."""
    out: list[dict] = []
    folders, files = list_dir(s, prefix)
    for url, name, size in files:
        key_b64 = url.split("key=")[-1]
        s3_key = base64.b64decode(requests.utils.unquote(key_b64) + "==").decode(
            "utf-8", "replace"
        )
        out.append(
            {
                "name": name,
                "size_text": size,
                "download_url": url,
                "s3_key": s3_key,
                "rel_path": s3_key.split(f"/{NODE}/", 1)[-1],
            }
        )
    for sub in folders:
        out.extend(walk(s, sub, depth + 1))
    return out


def build_manifest(s: requests.Session, subjects: list[int]) -> list[dict]:
    records: list[dict] = []
    for i in subjects:
        prefix = f"open/{DATASET_NODE}/{NODE}/s{i}_zip/"
        try:
            recs = walk(s, prefix)
        except Exception as exc:  # noqa: BLE001
            log(f"s{i}: listing FAILED ({exc})")
            continue
        records.extend(recs)
        log(f"s{i}: indexed {len(recs)} files")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(records, indent=1), encoding="utf-8")
    return records


def load_completed() -> dict[str, int]:
    """Verified sizes of finished files, so resumes cost no network calls."""
    if COMPLETED.is_file():
        try:
            return json.loads(COMPLETED.read_text(encoding="utf-8"))
        except ValueError:
            return {}
    return {}


def mark_completed(state: dict[str, int], rel_path: str, size: int) -> None:
    state[rel_path] = size
    tmp = COMPLETED.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
    tmp.replace(COMPLETED)


def signed_url(s: requests.Session, rec: dict) -> str | None:
    r = s.get(BASE + rec["download_url"], timeout=90)
    if r.status_code != 200:
        return None
    try:
        d = r.json()
    except ValueError:
        return None
    return d.get("download_url") if d.get("status") == "success" else None


def fetch(s: requests.Session, rec: dict, state: dict[str, int]) -> bool:
    dest = ROOT / rec["rel_path"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Fast path: previously verified and still the right size on disk.
    known = state.get(rec["rel_path"])
    if known is not None and dest.exists() and dest.stat().st_size == known:
        return True

    for attempt in (1, 2, 3):
        url = signed_url(s, rec)
        if not url:
            log(f"{rec['rel_path']}: could not mint signed URL (attempt {attempt})")
            time.sleep(10)
            continue

        have = dest.stat().st_size if dest.exists() else 0

        # The presigned URL is signed for GET only -- HEAD returns 403 -- so the
        # authoritative object size comes from Content-Range on a ranged GET
        # ("bytes 0-1023/4954673952"). Always range from `have` so the same
        # request both resumes and reports the total.
        headers = {"Range": f"bytes={have}-"}
        mode = "ab" if have else "wb"

        try:
            with requests.get(
                url, headers=headers, stream=True, timeout=(60, 300)
            ) as resp:
                if resp.status_code == 416:
                    # Range beyond EOF: already have the whole object.
                    mark_completed(state, rec["rel_path"], have)
                    log(f"{rec['rel_path']}: complete ({have} bytes) - skip")
                    return True
                if resp.status_code not in (200, 206):
                    log(f"{rec['rel_path']}: HTTP {resp.status_code} (attempt {attempt})")
                    time.sleep(10)
                    continue

                cr = resp.headers.get("Content-Range", "")
                if "/" in cr:
                    total = int(cr.rsplit("/", 1)[1])
                elif resp.status_code == 200:
                    total = int(resp.headers.get("Content-Length", 0))
                    mode = "wb"  # server ignored Range; restart the file
                else:
                    total = 0

                if total and have == total:
                    mark_completed(state, rec["rel_path"], have)
                    log(f"{rec['rel_path']}: complete ({have} bytes) - skip")
                    return True

                with dest.open(mode) as fh:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
        except Exception as exc:  # noqa: BLE001
            log(f"{rec['rel_path']}: transfer error {exc} (attempt {attempt})")
            time.sleep(15)
            continue

        got = dest.stat().st_size
        if total and got == total:
            mark_completed(state, rec["rel_path"], got)
            log(f"{rec['rel_path']}: OK ({got} bytes)")
            return True
        if not total:
            log(f"{rec['rel_path']}: OK ({got} bytes, size unverified)")
            return True
        log(f"{rec['rel_path']}: size mismatch {got}/{total} (attempt {attempt})")
        time.sleep(10)

    log(f"{rec['rel_path']}: FAILED after 3 attempts")
    return False


def parse_subjects(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1-56")
    ap.add_argument("--signals", action="store_true", help="skip .avi video")
    ap.add_argument("--video-only", action="store_true", help="only .avi video")
    ap.add_argument("--manifest-only", action="store_true")
    ap.add_argument("--refresh-manifest", action="store_true")
    args = ap.parse_args()

    subjects = parse_subjects(args.subjects)
    s = session()

    if MANIFEST.is_file() and not args.refresh_manifest and not args.manifest_only:
        records = json.loads(MANIFEST.read_text(encoding="utf-8"))
        log(f"loaded manifest: {len(records)} files")
    else:
        log(f"indexing subjects {subjects[0]}..{subjects[-1]} ...")
        records = build_manifest(s, subjects)
        log(f"manifest written: {len(records)} files -> {MANIFEST}")
        if args.manifest_only:
            return 0

    wanted = [r for r in records if f"/s{r['rel_path'].split('/')[1]}/" or True]
    keep = []
    for r in records:
        subj = re.match(r"s(\d+)_zip/", r["rel_path"])
        if subj and int(subj.group(1)) not in subjects:
            continue
        is_video = r["name"].lower().endswith(".avi")
        if args.signals and is_video:
            continue
        if args.video_only and not is_video:
            continue
        keep.append(r)

    state = load_completed()
    todo = [r for r in keep if r["rel_path"] not in state]
    log(f"{len(keep)} files in scope; {len(keep) - len(todo)} already done; {len(todo)} to fetch")

    failed = 0
    for i, rec in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {rec['rel_path']} ({rec['size_text']})")
        if not fetch(s, rec, state):
            failed += 1
        else:
            # Gentle pacing: rapid signed-URL minting gets throttled.
            time.sleep(0.5)
    log(f"done: {len(todo) - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
