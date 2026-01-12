
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUM Bundler v30 (Standalone, AoR-grade)
======================================

Builds a single evidence bundle for ALL demos under:
  demos/<domain>/<demo-folder>/demo.py

Modes:
  - RUN (default): runs demos (prefers --cert if supported) and captures logs + hashes.
  - INGEST (--ingest-only): does not run; scans existing artifacts + hashes.
  - LEDGER (--ledger PATH): ingests an existing run ledger instead of running.

Outputs:
  gum/GUM_Bundles/GUM_BUNDLE_v30_<UTC>/
    bundle.json
    bundle_sha256.txt
    repo_inventory.json
    runs.json
    artifacts_index.json
    values.jsonl
    tables/*.csv + tables/*.json
    manifest.json

Principles:
  - No guessing: every value has provenance.
  - Hash everything (code, artifacts, logs, bundle).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

ART_EXTS = {".json", ".png", ".pdf", ".csv", ".txt"}
UTC_STAMP = "%Y%m%dT%H%M%SZ"


# --------------------------
# Hash helpers
# --------------------------
def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


# --------------------------
# Git helpers
# --------------------------
def git_head() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return out.decode("utf-8").strip()
    except Exception:
        return None

def git_is_clean() -> Optional[bool]:
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return (out.decode("utf-8").strip() == "")
    except Exception:
        return None

def git_ls_files() -> List[str]:
    try:
        out = subprocess.check_output(["git", "ls-files"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL)
        return [x for x in out.decode("utf-8").splitlines() if x.strip()]
    except Exception:
        return []


# --------------------------
# Demo discovery
# --------------------------
@dataclass
class Demo:
    demo_id: str
    domain: str
    folder: str
    demo_dir: Path
    demo_py: Path

def parse_demo_id(folder: str) -> str:
    # expected folder like: demo-33-first-principles-...
    m = re.match(r"^demo-([0-9]{2})(?:-|$)", folder)
    if m:
        return m.group(1)
    return folder

def discover_demos(demos_root: Path) -> List[Demo]:
    demos: List[Demo] = []
    for demo_py in sorted(demos_root.glob("*/*/demo.py")):
        demo_dir = demo_py.parent
        domain = demo_dir.parent.name
        folder = demo_dir.name
        demo_id = parse_demo_id(folder)
        demos.append(Demo(demo_id=demo_id, domain=domain, folder=folder, demo_dir=demo_dir, demo_py=demo_py))
    return demos


# --------------------------
# Artifact scanning
# --------------------------
def list_artifacts(demo_dir: Path) -> List[Path]:
    out = []
    for p in demo_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ART_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: x.name.lower())


# --------------------------
# JSON flattening → values.jsonl
# --------------------------
def flatten_json_obj(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    out: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k in sorted(obj.keys()):
            key = f"{prefix}.{k}" if prefix else str(k)
            out.extend(flatten_json_obj(obj[k], key))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            key = f"{prefix}[{i}]"
            out.extend(flatten_json_obj(v, key))
    else:
        out.append((prefix, obj))
    return out

def extract_values_from_json_file(
    demo_id: str,
    domain: str,
    artifact_path: Path,
    artifact_sha: str,
    max_items: int = 20000
) -> List[Dict[str, Any]]:
    try:
        obj = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    flat = flatten_json_obj(obj)
    values = []
    for k, v in flat[:max_items]:
        if isinstance(v, (int, float, str, bool)) or v is None:
            values.append({
                "demo_id": demo_id,
                "domain": domain,
                "value_name": k,
                "value": v,
                "units": None,
                "source_type": "json_path",
                "source_path": str(artifact_path.relative_to(REPO_ROOT)),
                "source_sha256": artifact_sha,
                "source_locator": k,
            })
    return values


# --------------------------
# Running demos (optional)
# --------------------------
@dataclass
class RunRecord:
    demo_id: str
    domain: str
    folder: str
    demo_path: str
    mode: str
    status: str
    returncode: Optional[int]
    seconds: float
    code_sha256: str
    stdout_path: str
    stderr_path: str
    stdout_sha256: str
    stderr_sha256: str
    notes: str

def supports_cert(demo_py: Path) -> bool:
    try:
        txt = demo_py.read_text(encoding="utf-8", errors="replace")
        return "--cert" in txt
    except Exception:
        return False

def run_demo(python_exe: str, demo: Demo, out_logs_dir: Path, timeout_s: int) -> RunRecord:
    demo_rel_dir = demo.demo_dir.relative_to(REPO_ROOT)
    stdout_path = out_logs_dir / f"{demo.domain}__{demo.folder}.out.txt"
    stderr_path = out_logs_dir / f"{demo.domain}__{demo.folder}.err.txt"

    code_sha = sha256_file(demo.demo_py)
    mode = "cert" if supports_cert(demo.demo_py) else "run"
    cmd = [python_exe, "demo.py"] + (["--cert"] if mode == "cert" else [])

    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(demo.demo_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        dt = time.time() - t0
        stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
        stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")

        stdout_sha = sha256_file(stdout_path)
        stderr_sha = sha256_file(stderr_path)
        status = "PASS" if proc.returncode == 0 else "FAIL"

        return RunRecord(
            demo_id=demo.demo_id,
            domain=demo.domain,
            folder=demo.folder,
            demo_path=str(demo_rel_dir),
            mode=mode,
            status=status,
            returncode=proc.returncode,
            seconds=round(dt, 6),
            code_sha256=code_sha,
            stdout_path=str(stdout_path.relative_to(REPO_ROOT)),
            stderr_path=str(stderr_path.relative_to(REPO_ROOT)),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            notes="",
        )

    except subprocess.TimeoutExpired as e:
        dt = time.time() - t0
        stdout_path.write_text((e.stdout or ""), encoding="utf-8", errors="replace")
        stderr_path.write_text((e.stderr or "") + f"\n\n[TIMEOUT after {timeout_s}s]\n", encoding="utf-8", errors="replace")
        stdout_sha = sha256_file(stdout_path)
        stderr_sha = sha256_file(stderr_path)

        return RunRecord(
            demo_id=demo.demo_id,
            domain=demo.domain,
            folder=demo.folder,
            demo_path=str(demo_rel_dir),
            mode=mode,
            status="TIMEOUT",
            returncode=None,
            seconds=round(dt, 6),
            code_sha256=code_sha,
            stdout_path=str(stdout_path.relative_to(REPO_ROOT)),
            stderr_path=str(stderr_path.relative_to(REPO_ROOT)),
            stdout_sha256=stdout_sha,
            stderr_sha256=stderr_sha,
            notes=f"timeout>{timeout_s}s",
        )


# --------------------------
# Ledger ingestion (optional)
# --------------------------
def ingest_external_ledger(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def normalize_ledger_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "demo_id": str(row.get("demo_id", "")),
        "domain": str(row.get("category", "")),
        "folder": str(row.get("folder", "")),
        "demo_path": str(row.get("demo_path", "")),
        "mode": str(row.get("mode", "")),
        "status": str(row.get("status", "")),
        "returncode": row.get("returncode"),
        "seconds": row.get("seconds"),
        "code_sha256": row.get("code_sha256"),
        "stdout_path": row.get("stdout_path"),
        "stderr_path": row.get("stderr_path"),
        "artifacts": row.get("artifacts") or [],
        "notes": row.get("notes") or "",
    }


# --------------------------
# Bundle builder
# --------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")

def write_text(p: Path, text: str) -> None:
    p.write_text(text, encoding="utf-8")

def build_repo_inventory() -> Dict[str, Any]:
    files = git_ls_files()
    recs = []
    for rel in files:
        fp = REPO_ROOT / rel
        if fp.exists() and fp.is_file():
            recs.append({
                "path": rel,
                "sha256": sha256_file(fp),
                "size_bytes": fp.stat().st_size,
            })
    return {
        "git_head": git_head(),
        "git_is_clean": git_is_clean(),
        "tracked_count": len(recs),
        "tracked_files": recs,
    }

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def main() -> int:
    ap = argparse.ArgumentParser(description="GUM Bundler v30 (AoR-grade).")
    ap.add_argument("--demos-root", default="demos", help="Root demos directory (default: demos)")
    ap.add_argument("--outdir", default="gum/GUM_Bundles", help="Output root (default: gum/GUM_Bundles)")
    ap.add_argument("--ledger", default="", help="Optional path to external ledger JSON (smoketest summary.json)")
    ap.add_argument("--ingest-only", dest="ingest_only", action="store_true",
                    help="Do not run demos; only ingest existing artifacts")
    ap.add_argument("--timeout", type=int, default=600, help="Per-demo timeout seconds (default: 600)")
    ap.add_argument("--python", default=sys.executable, help="Python executable (default: current)")
    ap.add_argument("--vendor-artifacts", dest="vendor_artifacts", action="store_true",
                    help="Copy artifacts into bundle directory (self-contained but larger). Default is index-by-path.")
    args = ap.parse_args()

    demos_root = (REPO_ROOT / args.demos_root).resolve()
    if not demos_root.exists():
        raise SystemExit(f"Missing demos root: {demos_root}")

    ts = datetime.now(timezone.utc).strftime(UTC_STAMP)
    bundle_dir = (REPO_ROOT / args.outdir / f"GUM_BUNDLE_v30_{ts}").resolve()
    ensure_dir(bundle_dir)
    ensure_dir(bundle_dir / "tables")

    logs_dir = REPO_ROOT / "artifacts" / "bundle_logs_v30"
    ensure_dir(logs_dir)

    demos = discover_demos(demos_root)

    runs_norm: Dict[str, Any] = {}
    external_ledger_obj: Optional[Dict[str, Any]] = None
    ledger_sha = None

    if args.ledger:
        ledger_path = (REPO_ROOT / args.ledger).resolve() if not os.path.isabs(args.ledger) else Path(args.ledger)
        if ledger_path.exists():
            ledger_sha = sha256_file(ledger_path)
            external_ledger_obj = ingest_external_ledger(ledger_path)
            for row in external_ledger_obj.get("results", []):
                nr = normalize_ledger_record(row)
                did = str(nr.get("demo_id") or "")
                if did:
                    runs_norm[did] = nr

    if (not args.ingest_only) and (not runs_norm):
        for d in demos:
            rr = run_demo(args.python, d, logs_dir, args.timeout)
            runs_norm[d.demo_id] = {
                "demo_id": rr.demo_id,
                "domain": rr.domain,
                "folder": rr.folder,
                "demo_path": rr.demo_path,
                "mode": rr.mode,
                "status": rr.status,
                "returncode": rr.returncode,
                "seconds": rr.seconds,
                "code_sha256": rr.code_sha256,
                "stdout_path": rr.stdout_path,
                "stderr_path": rr.stderr_path,
                "stdout_sha256": rr.stdout_sha256,
                "stderr_sha256": rr.stderr_sha256,
                "notes": rr.notes,
            }

    artifacts_index: List[Dict[str, Any]] = []
    values_jsonl_path = bundle_dir / "values.jsonl"
    values_f = values_jsonl_path.open("w", encoding="utf-8")

    vendored_dir = bundle_dir / "vendored_artifacts"
    if args.vendor_artifacts:
        ensure_dir(vendored_dir)

    for d in demos:
        for art in list_artifacts(d.demo_dir):
            art_sha = sha256_file(art)
            rel_art = str(art.relative_to(REPO_ROOT))
            rec = {
                "demo_id": d.demo_id,
                "domain": d.domain,
                "demo_dir": str(d.demo_dir.relative_to(REPO_ROOT)),
                "artifact": rel_art,
                "sha256": art_sha,
                "size_bytes": art.stat().st_size,
                "ext": art.suffix.lower(),
                "role": "unknown",
            }
            n = art.name.lower()
            if n.endswith(".json"):
                rec["role"] = "primary_results_json" if ("outputs" in n or "results" in n or "report" in n or "manifest" in n) else "json"
            elif n.endswith(".png"):
                rec["role"] = "figure_png"
            elif n.endswith(".pdf"):
                rec["role"] = "figure_pdf"
            elif n.endswith(".csv"):
                rec["role"] = "table_csv"
            elif n.endswith(".txt"):
                rec["role"] = "text_txt"

            artifacts_index.append(rec)

            if args.vendor_artifacts:
                dest = vendored_dir / f"{d.domain}__{d.folder}__{art.name}"
                if not dest.exists():
                    dest.write_bytes(art.read_bytes())

            if art.suffix.lower() == ".json":
                for row in extract_values_from_json_file(d.demo_id, d.domain, art, art_sha):
                    values_f.write(json.dumps(row, sort_keys=True) + "\n")

    values_f.close()

    repo_inventory = build_repo_inventory()

    demo_index_rows = []
    for d in demos:
        rr = runs_norm.get(d.demo_id)
        demo_index_rows.append({
            "demo_id": d.demo_id,
            "domain": d.domain,
            "folder": d.folder,
            "demo_dir": str(d.demo_dir.relative_to(REPO_ROOT)),
            "status": rr.get("status") if rr else "NOT_IN_RUNS",
            "mode": rr.get("mode") if rr else "",
        })

    fals_rows = []
    for d in demos:
        rr = runs_norm.get(d.demo_id, {})
        one_liner = f'(cd "{d.demo_dir.relative_to(REPO_ROOT)}" && python demo.py' + (' --cert' if supports_cert(d.demo_py) else '') + ')'
        fals_rows.append({
            "demo_id": d.demo_id,
            "domain": d.domain,
            "folder": d.folder,
            "one_liner": one_liner,
            "status": rr.get("status", "NOT_IN_RUNS"),
            "returncode": rr.get("returncode", ""),
            "seconds": rr.get("seconds", ""),
            "code_sha256": rr.get("code_sha256", sha256_file(d.demo_py)),
        })

    repro_rows = []
    for demo_id, rr in sorted(runs_norm.items(), key=lambda kv: kv[0]):
        repro_rows.append({
            "demo_id": demo_id,
            "domain": rr.get("domain",""),
            "status": rr.get("status",""),
            "mode": rr.get("mode",""),
            "returncode": rr.get("returncode",""),
            "seconds": rr.get("seconds",""),
            "code_sha256": rr.get("code_sha256",""),
            "stdout_path": rr.get("stdout_path",""),
            "stderr_path": rr.get("stderr_path",""),
            "stdout_sha256": rr.get("stdout_sha256",""),
            "stderr_sha256": rr.get("stderr_sha256",""),
            "notes": rr.get("notes",""),
        })

    constants_rows = []
    with values_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            v = row.get("value")
            if isinstance(v, (int, float)):
                constants_rows.append({
                    "demo_id": row.get("demo_id"),
                    "domain": row.get("domain"),
                    "name": row.get("value_name"),
                    "value": v,
                    "units": row.get("units"),
                    "source_path": row.get("source_path"),
                    "source_sha256": row.get("source_sha256"),
                    "locator": row.get("source_locator"),
                })

    write_csv(bundle_dir / "tables" / "demo_index.csv", demo_index_rows)
    write_csv(bundle_dir / "tables" / "falsification_matrix.csv", fals_rows)
    write_csv(bundle_dir / "tables" / "run_reproducibility.csv", repro_rows)
    write_csv(bundle_dir / "tables" / "constants_master.csv", constants_rows)

    write_json(bundle_dir / "repo_inventory.json", repo_inventory)
    write_json(bundle_dir / "runs.json", {
        "source_ledger": args.ledger if args.ledger else None,
        "source_ledger_sha256": ledger_sha,
        "generated_by": "RUN" if ((not args.ingest_only) and (not args.ledger)) else ("LEDGER" if args.ledger else "INGEST_ONLY"),
        "runs": runs_norm,
    })
    write_json(bundle_dir / "artifacts_index.json", {"artifacts": artifacts_index})
    write_json(bundle_dir / "tables" / "demo_index.json", demo_index_rows)
    write_json(bundle_dir / "tables" / "falsification_matrix.json", fals_rows)
    write_json(bundle_dir / "tables" / "run_reproducibility.json", repro_rows)
    write_json(bundle_dir / "tables" / "constants_master.json", constants_rows)

    bundle = {
        "bundle_meta": {
            "version": "v30",
            "timestamp_utc": ts,
            "git_head": repo_inventory.get("git_head"),
            "git_is_clean": repo_inventory.get("git_is_clean"),
            "python": {
                "version": sys.version.replace("\n", " "),
                "executable": sys.executable,
                "platform": platform.platform(),
            },
            "mode": "INGEST_ONLY" if args.ingest_only else ("LEDGER" if args.ledger else "RUN"),
            "ledger_path": args.ledger or None,
            "ledger_sha256": ledger_sha,
            "vendor_artifacts": bool(args.vendor_artifacts),
        },
        "repo_inventory": repo_inventory,
        "demo_count": len(demos),
        "runs": {"count": len(runs_norm), "runs_json": "runs.json"},
        "artifacts": {"count": len(artifacts_index), "artifacts_index_json": "artifacts_index.json", "values_jsonl": "values.jsonl"},
        "tables": {
            "demo_index": "tables/demo_index.csv",
            "falsification_matrix": "tables/falsification_matrix.csv",
            "run_reproducibility": "tables/run_reproducibility.csv",
            "constants_master": "tables/constants_master.csv",
        },
        "notes": [
            "All demos discovered under demos/*/*/demo.py are included.",
            "All JSON artifacts are flattened into values.jsonl with provenance.",
            "In INGEST_ONLY mode, missing artifacts indicate demos have not produced outputs yet in this checkout.",
        ],
    }

    bundle_path = bundle_dir / "bundle.json"
    write_json(bundle_path, bundle)
    bundle_sha = sha256_file(bundle_path)
    write_text(bundle_dir / "bundle_sha256.txt", bundle_sha + "\n")

    missing_runs = [d.folder for d in demos if d.demo_id not in runs_norm]
    missing_artifacts = [d.folder for d in demos if len(list_artifacts(d.demo_dir)) == 0]
    manifest = {
        "bundle_dir": str(bundle_dir.relative_to(REPO_ROOT)),
        "bundle_sha256": bundle_sha,
        "demo_count": len(demos),
        "runs_count": len(runs_norm),
        "artifact_records": len(artifacts_index),
        "missing_runs": missing_runs,
        "missing_artifacts": missing_artifacts,
    }
    write_json(bundle_dir / "manifest.json", manifest)

    print("BUNDLE WRITTEN:")
    print("  ", bundle_dir)
    print("  bundle_sha256:", bundle_sha)
    print("  demos:", len(demos), "runs:", len(runs_norm), "artifact_records:", len(artifacts_index))
    if missing_runs:
        print("  missing runs for demos:", len(missing_runs))
    if missing_artifacts:
        print("  demos with zero local artifacts:", len(missing_artifacts))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

