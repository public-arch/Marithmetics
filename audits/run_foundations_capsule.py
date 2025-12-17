#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Foundations Capsule Runner (stdlib-only)

What it does
- Runs a canonical allowlist of scripts (Tier-A/A2) and captures stdout/stderr
- Writes a frozen results bundle under audits/results/foundations_a2_<UTCSTAMP>/
- Produces:
    run_metadata.json
    runs.json
    code_sha256.json
    constants_table.json
    RUN_COMMAND.txt
    output_sha256.json   (hashes of all artifacts EXCEPT output_sha256.json and BUNDLE_SHA256.txt)
    BUNDLE_SHA256.txt    (hash of key evidence files)

Verify mode
- --verify <folder> recomputes output_sha256.json (with the same exclusions) and the bundle hash.
- This is designed to be stable and not fail due to self-hashing.

Usage
  python audits/run_foundations_capsule.py
  python audits/run_foundations_capsule.py --include-optional
  python audits/run_foundations_capsule.py --out audits/results/foundations_a2_custom
  python audits/run_foundations_capsule.py --verify audits/results/foundations_a2_YYYY_MM_DDTHHMMSSZ
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Set


# -----------------------------
# Canonical allowlist
# -----------------------------

CORE_SCRIPTS = [
    # Referee harnesses / authority capsules
    "audits/tier_a_master_referee_demo_v6.py",
    "audits/a2_archive_master.py",
    "audits/marithmetics_unified_theory.py",

    # Engines run directly (as requested)
    "sm/sm_standard_model_demo_v1.py",
    "cosmo/bb_grand_emergence_masterpiece_runner_v1.py",
    "omega/omega_observer_commutant_fejer_v1.py",
    "substrate/scfp_integer_selector_v1.py",

    # DOC / Rosetta
    "doc/doc_rosetta_base_gauge_v1.py",
    "doc/doc_referee_closeout_audit_v1.py",
]

OPTIONAL_SCRIPTS = [
    # Explicitly downstream / optional diagnostic
    "cosmo/gum_camb_check.py",
]

# Output hash exclusions:
# - output_sha256.json would otherwise self-hash (unstable)
# - BUNDLE_SHA256.txt is computed after and depends on key files
OUTPUT_HASH_EXCLUDE_DEFAULT: Set[str] = {"output_sha256.json", "BUNDLE_SHA256.txt"}

# Conservative extraction keys (best-effort, not required for verification)
EXTRACT_KEYS = [
    "alpha", "alpha_s", "sin2", "sin^2", "sin2theta", "sin2w",
    "h0", "H0",
    "omega_b", "omega_c", "omega_l", "omega_lambda", "omega_r", "omega_tot", "omega_m",
    "as", "a_s", "ns", "n_s", "tau",
    "delta21", "d21", "dm21", "delta_21",
    "delta31", "d31", "dm31", "delta_31",
    "sumv", "sum_mnu", "sigmamnu",
    "etab", "etaB", "yhe", "YHe",
    "ell1", "l1",
]


# -----------------------------
# Utilities
# -----------------------------

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_stamp() -> str:
    # timezone-aware UTC stamp (no utcnow() warnings)
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y_%m_%dT%H%M%SZ")


def find_repo_root(start: Path) -> Path:
    cur = start.resolve()
    for _ in range(12):
        if (cur / ".git").exists() or (cur / "MANIFEST.md").exists() or (cur / "Readme.md").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def git_commit(repo_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def git_status_porcelain(repo_root: Path) -> Optional[str]:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=True,
        )
        return r.stdout.strip()
    except Exception:
        return None


def safe_slug(path_str: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path_str.strip().replace("/", "_").replace("\\", "_"))


def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def extract_constants_from_text(text: str) -> Dict[str, Any]:
    """
    Best-effort extractor. Non-critical. Used for convenience in paper table prep.
    """
    out: Dict[str, Any] = {}
    lines = text.splitlines()
    key_set = set(k.lower() for k in EXTRACT_KEYS)
    num_re = r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"
    pat = re.compile(r"^\s*([A-Za-z0-9_^]+)\s*[:=]\s*" + num_re + r"\s*$")

    for line in lines:
        m = pat.match(line)
        if not m:
            continue
        k_raw = m.group(1).strip()
        if k_raw.lower() not in key_set:
            continue
        v_str = m.group(2)
        try:
            v = float(v_str)
        except Exception:
            continue
        out[k_raw] = {"value": v, "value_str": v_str, "line": line.strip()}

    embed_pat = re.compile(r"\b([A-Za-z0-9_^]+)\b\s*=\s*" + num_re)
    for line in lines:
        for m in embed_pat.finditer(line):
            k_raw = m.group(1).strip()
            if k_raw.lower() not in key_set:
                continue
            v_str = m.group(2)
            try:
                v = float(v_str)
            except Exception:
                continue
            out[k_raw] = {"value": v, "value_str": v_str, "line": line.strip()}

    return out


def compute_output_hashes(out_dir: Path, exclude: Optional[Set[str]] = None) -> Dict[str, str]:
    if exclude is None:
        exclude = set(OUTPUT_HASH_EXCLUDE_DEFAULT)
    m: Dict[str, str] = {}
    for p in sorted(out_dir.glob("*")):
        if p.is_file() and p.name not in exclude:
            m[p.name] = sha256_file(p)
    return m


def compute_code_hashes(repo_root: Path, scripts: List[str]) -> Dict[str, Optional[str]]:
    m: Dict[str, Optional[str]] = {}
    for rel in scripts:
        p = (repo_root / rel).resolve()
        m[rel] = sha256_file(p) if p.exists() and p.is_file() else None
    return m


def bundle_sha256(out_dir: Path, key_files: List[str]) -> str:
    """
    Bundle fingerprint: hash the bytes of key evidence files (in order).
    """
    h = hashlib.sha256()
    for name in key_files:
        p = out_dir / name
        if not p.exists():
            h.update(f"MISSING:{name}\n".encode("utf-8"))
            continue
        h.update(p.read_bytes())
        h.update(b"\n")
    return h.hexdigest()


def run_one(repo_root: Path, script_rel: str, out_dir: Path) -> Dict[str, Any]:
    script_path = (repo_root / script_rel).resolve()
    slug = safe_slug(script_rel)

    rec: Dict[str, Any] = {
        "script_rel": script_rel,
        "script_abs": str(script_path),
        "exists": script_path.exists(),
        "returncode": None,
        "seconds": None,
        "stdout_file": None,
        "stderr_file": None,
        "constants_extracted": {},
    }

    if not script_path.exists():
        return rec

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONHASHSEED"] = "0"

    t0 = time.time()
    p = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        env=env,
    )
    t1 = time.time()

    stdout_path = out_dir / f"stdout_{slug}.txt"
    stderr_path = out_dir / f"stderr_{slug}.txt"

    stdout_path.write_text(p.stdout, encoding="utf-8", errors="replace")
    stderr_path.write_text(p.stderr, encoding="utf-8", errors="replace")

    rec["returncode"] = p.returncode
    rec["seconds"] = round(t1 - t0, 6)
    rec["stdout_file"] = stdout_path.name
    rec["stderr_file"] = stderr_path.name
    rec["constants_extracted"] = extract_constants_from_text(p.stdout)

    return rec


# -----------------------------
# Commands
# -----------------------------

def cmd_run(args: argparse.Namespace) -> int:
    repo_root = find_repo_root(Path(__file__).parent)
    stamp = utc_stamp()

    out_dir = Path(args.out).resolve() if args.out else (repo_root / "audits" / "results" / f"foundations_a2_{stamp}").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    scripts = list(CORE_SCRIPTS)
    if args.include_optional:
        scripts += OPTIONAL_SCRIPTS

    # Write command early so it is included in output hashes
    (out_dir / "RUN_COMMAND.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    meta = {
        "utc_stamp": stamp,
        "repo_root": str(repo_root),
        "git_commit": git_commit(repo_root),
        "git_status_porcelain": git_status_porcelain(repo_root),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "argv": sys.argv,
        "scripts_core": CORE_SCRIPTS,
        "scripts_optional": OPTIONAL_SCRIPTS,
        "scripts_ran": scripts,
    }
    write_json(out_dir / "run_metadata.json", meta)

    runs: List[Dict[str, Any]] = []
    for rel in scripts:
        runs.append(run_one(repo_root, rel, out_dir))
    write_json(out_dir / "runs.json", runs)

    code_hashes = compute_code_hashes(repo_root, scripts)
    write_json(out_dir / "code_sha256.json", code_hashes)

    merged_constants: Dict[str, Any] = {}
    for r in runs:
        for k, v in (r.get("constants_extracted") or {}).items():
            merged_constants[k] = {
                "value": v.get("value"),
                "value_str": v.get("value_str"),
                "line": v.get("line"),
                "source_script": r.get("script_rel"),
            }
    write_json(out_dir / "constants_table.json", merged_constants)

    # Output hashes exclude output_sha256.json and BUNDLE_SHA256.txt
    out_hashes = compute_output_hashes(out_dir)
    write_json(out_dir / "output_sha256.json", out_hashes)

    key_files = [
        "run_metadata.json",
        "runs.json",
        "code_sha256.json",
        "constants_table.json",
        "RUN_COMMAND.txt",
        "output_sha256.json",
    ]
    b = bundle_sha256(out_dir, key_files)
    (out_dir / "BUNDLE_SHA256.txt").write_text(b + "\n", encoding="utf-8")

    print(f"[OK] wrote results to: {out_dir}")
    print(f"[OK] BUNDLE_SHA256: {b}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    out_dir = Path(args.verify).resolve()
    if not out_dir.exists():
        print(f"[ERR] verify path not found: {out_dir}")
        return 2

    required = [
        "run_metadata.json",
        "runs.json",
        "code_sha256.json",
        "constants_table.json",
        "RUN_COMMAND.txt",
        "output_sha256.json",
        "BUNDLE_SHA256.txt",
    ]
    missing = [x for x in required if not (out_dir / x).exists()]
    if missing:
        print("[ERR] missing required files:")
        for x in missing:
            print("  - " + x)
        return 3

    stored_outputs = json.loads((out_dir / "output_sha256.json").read_text(encoding="utf-8"))
    recomputed_outputs = compute_output_hashes(out_dir)

    if recomputed_outputs != stored_outputs:
        print("[FAIL] output_sha256 mismatch")
        return 4

    key_files = [
        "run_metadata.json",
        "runs.json",
        "code_sha256.json",
        "constants_table.json",
        "RUN_COMMAND.txt",
        "output_sha256.json",
    ]
    recomputed_bundle = bundle_sha256(out_dir, key_files)
    stored_bundle = (out_dir / "BUNDLE_SHA256.txt").read_text(encoding="utf-8").strip()

    if recomputed_bundle != stored_bundle:
        print("[FAIL] BUNDLE_SHA256 mismatch")
        print("  stored    :", stored_bundle)
        print("  recomputed:", recomputed_bundle)
        return 5

    print("[OK] verify passed")
    print("[OK] BUNDLE_SHA256:", stored_bundle)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", help="output directory path (default audits/results/foundations_a2_<UTC>)")
    ap.add_argument("--include-optional", action="store_true", help="also run optional scripts (CAMB check)")
    ap.add_argument("--verify", help="verify an existing results folder (re-hash)")
    args = ap.parse_args()

    if args.verify:
        return cmd_verify(args)
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())

