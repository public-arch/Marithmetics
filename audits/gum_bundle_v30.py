#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GUM Bundler v30.1 (Standalone, AoR-grade)
========================================

Fixes vs v30:
  - Treat stdout/stderr logs as first-class artifacts (counted and hashed).
  - Extract structured values from stdout into values.jsonl (paper-citable).
  - Capture artifacts written outside demo folders during run windows.
  - Normalize demo ids including 66A/66B.

Modes:
  - RUN (default): runs demos and captures logs + hashes + artifacts.
  - INGEST (--ingest-only): no run; scan existing artifacts + hash them.
  - LEDGER (--ledger PATH): ingest an external ledger (optional).

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

def norm_demo_id(s: str) -> str:
    s = str(s).strip()
    # normalize 66a/66b
    m = re.match(r"^([0-9]{2})([aAbB])$", s)
    if m:
        return m.group(1) + m.group(2).upper()
    return s

def parse_demo_id(folder: str) -> str:
    """
    Supports:
      demo-33-...
      demo-66a-...
      demo-66b-...
    """
    m = re.match(r"^demo-([0-9]{2})([aAbB])(?:-|$)", folder)
    if m:
        return norm_demo_id(m.group(1) + m.group(2))
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
def list_artifacts_in_dir(d: Path) -> List[Path]:
    out = []
    if not d.exists():
        return out
    for p in d.iterdir():
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
    max_items: int = 50000
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
# Stdout parsing → values.jsonl
# --------------------------
KV_LINE = re.compile(r"^\s*(.{1,80}?)\s{2,}(.+?)\s*$")  # key <2+ spaces> value
SHA_LINE = re.compile(r"^\s*(.+?)\s+sha256\s+([0-9a-f]{64})\s*$", re.IGNORECASE)

def extract_values_from_stdout(
    demo_id: str,
    domain: str,
    stdout_path: Path,
    stdout_sha: str,
    max_lines: int = 6000
) -> List[Dict[str, Any]]:
    txt = stdout_path.read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()[:max_lines]
    out: List[Dict[str, Any]] = []

    # 1) sha256 lines (high-signal)
    for i, line in enumerate(lines):
        m = SHA_LINE.match(line)
        if m:
            name = m.group(1).strip()
            h = m.group(2).strip()
            out.append({
                "demo_id": demo_id,
                "domain": domain,
                "value_name": f"stdout.sha256.{name}",
                "value": h,
                "units": None,
                "source_type": "stdout_regex",
                "source_path": str(stdout_path.relative_to(REPO_ROOT)),
                "source_sha256": stdout_sha,
                "source_locator": f"SHA_LINE@L{i+1}",
            })

    # 2) aligned kv lines (common in your demos)
    for i, line in enumerate(lines):
        m = KV_LINE.match(line)
        if not m:
            continue
        k = m.group(1).strip()
        v = m.group(2).strip()
        # Filter out noise lines
        if k.startswith("—") or k.startswith("═") or k.lower() in {"run", "scan"}:
            continue
        # Try to coerce numerics when trivial (keep string if not)
        vv: Any = v
        try:
            # take first token for numeric-ish lines like "0.123 GeV"
            t0 = v.split()[0]
            if re.fullmatch(r"[+-]?[0-9]*\.?[0-9]+([eE][+-]?[0-9]+)?", t0):
                vv = float(t0)
        except Exception:
            vv = v

        out.append({
            "demo_id": demo_id,
            "domain": domain,
            "value_name": f"stdout.kv.{k}",
            "value": vv,
            "units": None,
            "source_type": "stdout_kv",
            "source_path": str(stdout_path.relative_to(REPO_ROOT)),
            "source_sha256": stdout_sha,
            "source_locator": f"KV_LINE@L{i+1}",
        })

    return out


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
    run_start_epoch: float
    run_end_epoch: float

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

    run_start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(demo.demo_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        run_end = time.time()

        dt = run_end - run_start
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
            run_start_epoch=run_start,
            run_end_epoch=run_end,
        )

    except subprocess.TimeoutExpired as e:
        run_end = time.time()
        dt = run_end - run_start
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
            run_start_epoch=run_start,
            run_end_epoch=run_end,
        )


# --------------------------
# External ledger ingestion (optional)
# --------------------------
def ingest_external_ledger(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

def normalize_ledger_record(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "demo_id": norm_demo_id(str(row.get("demo_id", ""))),
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
            recs.append({"path": rel, "sha256": sha256_file(fp), "size_bytes": fp.stat().st_size})
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

def classify_role(name: str) -> str:
    n = name.lower()
    if n.endswith(".json"):
        return "primary_results_json" if any(x in n for x in ["outputs", "results", "report", "manifest"]) else "json"
    if n.endswith(".png"):
        return "figure_png"
    if n.endswith(".pdf"):
        return "figure_pdf"
    if n.endswith(".csv"):
        return "table_csv"
    if n.endswith(".txt"):
        return "text_txt"
    return "unknown"

def scan_external_artifacts_since(run_start_epoch: float) -> List[Path]:
    """
    Capture artifacts written outside the demo folder during the run window.
    We scan a few controlled locations for files with mtime >= run_start_epoch - 1.
    """
    roots = [
        REPO_ROOT,                              # sometimes demos write to repo root
        REPO_ROOT / "gum" / "GUM_Report",       # report plates
        REPO_ROOT / "artifacts",                # logs and generated outputs
    ]
    found: List[Path] = []
    cutoff = run_start_epoch - 1.0
    for r in roots:
        if not r.exists():
            continue
        for p in r.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in ART_EXTS:
                continue
            try:
                if p.stat().st_mtime >= cutoff:
                    found.append(p)
            except Exception:
                continue
    # de-dup
    uniq = {}
    for p in found:
        uniq[str(p)] = p
    return sorted(uniq.values(), key=lambda x: str(x).lower())

def main() -> int:
    ap = argparse.ArgumentParser(description="GUM Bundler v30.1 (AoR-grade).")
    ap.add_argument("--demos-root", default="demos")
    ap.add_argument("--outdir", default="gum/GUM_Bundles")
    ap.add_argument("--ledger", default="")
    ap.add_argument("--ingest-only", dest="ingest_only", action="store_true")
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--vendor-artifacts", dest="vendor_artifacts", action="store_true")
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
    ledger_sha = None

    if args.ledger:
        ledger_path = (REPO_ROOT / args.ledger).resolve() if not os.path.isabs(args.ledger) else Path(args.ledger)
        if ledger_path.exists():
            ledger_sha = sha256_file(ledger_path)
            led = ingest_external_ledger(ledger_path)
            for row in led.get("results", []):
                nr = normalize_ledger_record(row)
                did = nr.get("demo_id") or ""
                if did:
                    runs_norm[did] = nr

    # Run all demos if needed
    run_windows: Dict[str, Tuple[float, float]] = {}
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
            run_windows[d.demo_id] = (rr.run_start_epoch, rr.run_end_epoch)

    # Prepare vendoring
    vendored_dir = bundle_dir / "vendored_artifacts"
    if args.vendor_artifacts:
        ensure_dir(vendored_dir)

    artifacts_index: List[Dict[str, Any]] = []
    values_jsonl_path = bundle_dir / "values.jsonl"
    vf = values_jsonl_path.open("w", encoding="utf-8")

    # Helper to index an artifact
    def index_artifact(demo_id: str, domain: str, demo_dir_rel: str, p: Path, role_override: Optional[str] = None) -> None:
        try:
            sha = sha256_file(p)
            rel = str(p.relative_to(REPO_ROOT))
            rec = {
                "demo_id": demo_id,
                "domain": domain,
                "demo_dir": demo_dir_rel,
                "artifact": rel,
                "sha256": sha,
                "size_bytes": p.stat().st_size,
                "ext": p.suffix.lower(),
                "role": role_override or classify_role(p.name),
            }
            artifacts_index.append(rec)

            if args.vendor_artifacts:
                dest = vendored_dir / f"{domain}__{demo_id}__{p.name}"
                if not dest.exists():
                    dest.write_bytes(p.read_bytes())

            if p.suffix.lower() == ".json":
                for row in extract_values_from_json_file(demo_id, domain, p, sha):
                    vf.write(json.dumps(row, sort_keys=True) + "\n")
        except Exception:
            return

    # Index per-demo artifacts + logs + stdout values
    for d in demos:
        did = d.demo_id
        demo_dir_rel = str(d.demo_dir.relative_to(REPO_ROOT))

        # 1) artifacts in demo dir
        for art in list_artifacts_in_dir(d.demo_dir):
            index_artifact(did, d.domain, demo_dir_rel, art)

        # 2) logs as artifacts (if run record present)
        rr = runs_norm.get(did, {})
        sp = rr.get("stdout_path")
        ep = rr.get("stderr_path")
        if sp:
            p = REPO_ROOT / sp
            if p.exists():
                index_artifact(did, d.domain, demo_dir_rel, p, role_override="stdout_log")
                # extract stdout values
                sha = sha256_file(p)
                for row in extract_values_from_stdout(did, d.domain, p, sha):
                    vf.write(json.dumps(row, sort_keys=True) + "\n")
        if ep:
            p = REPO_ROOT / ep
            if p.exists():
                index_artifact(did, d.domain, demo_dir_rel, p, role_override="stderr_log")

        # 3) external artifacts created during run window (RUN mode only)
        if did in run_windows:
            run_start, _ = run_windows[did]
            for p in scan_external_artifacts_since(run_start):
                # avoid double counting if already under demo dir
                try:
                    if str(p.resolve()).startswith(str(d.demo_dir.resolve())):
                        continue
                except Exception:
                    pass
                index_artifact(did, d.domain, demo_dir_rel, p, role_override="external_artifact")

    vf.close()

    repo_inventory = build_repo_inventory()

    # Tables
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
            "version": "v30.1",
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
            "Stdout logs are treated as evidence artifacts and parsed into values.jsonl.",
            "External artifacts created during run windows are captured under role=external_artifact.",
        ],
    }

    bundle_path = bundle_dir / "bundle.json"
    write_json(bundle_path, bundle)
    bundle_sha = sha256_file(bundle_path)
    write_text(bundle_dir / "bundle_sha256.txt", bundle_sha + "\n")

    demos_with_zero_artifacts = []
    for d in demos:
        did = d.demo_id
        if not any(a.get("demo_id") == did for a in artifacts_index):
            demos_with_zero_artifacts.append(d.folder)

    manifest = {
        "bundle_dir": str(bundle_dir.relative_to(REPO_ROOT)),
        "bundle_sha256": bundle_sha,
        "demo_count": len(demos),
        "runs_count": len(runs_norm),
        "artifact_records": len(artifacts_index),
        "demos_with_zero_indexed_artifacts": demos_with_zero_artifacts,
    }
    write_json(bundle_dir / "manifest.json", manifest)

    print("BUNDLE WRITTEN:")
    print("  ", bundle_dir)
    print("  bundle_sha256:", bundle_sha)
    print("  demos:", len(demos), "runs:", len(runs_norm), "artifact_records:", len(artifacts_index))
    print("  demos_with_zero_indexed_artifacts:", len(demos_with_zero_artifacts))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
