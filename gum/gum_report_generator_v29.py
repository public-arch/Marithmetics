cat > gum/gum_report_generator_v29.py << 'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUM Report Generator v29
========================

Goal
----
Produce a single PDF ("system audit") that:
  * preserves the v27/v28 visual flow (DRPT geometry, Fejér smoothing, dashboards, CMB overlay),
  * upgrades the report with an explicit falsification-first architecture:
        - Falsification Quickstart (copy/paste one-liners)
        - Bridge hypothesis (DRPT -> physics transition made explicit)
        - Φ-channel lane table (pulled from Demo 33 stdout when available)
        - Unified Dashboard (headline values from demo artifacts/stdout logs)
        - Falsification Matrix (demo -> claim -> command -> status -> hashes)
        - Flagship Certificates (modular proof units)

Inputs
------
- Optional smoketest ledger JSON (recommended):
    artifacts/smoketest_all/summary.json

  The v29 report uses this ledger as the source of truth for:
    * pass/fail status
    * runtime
    * code sha256
    * artifact sha256
    * stdout/stderr log paths (hashed)

- Optional per-demo metadata file (future-proof, not required):
    <demo_dir>/demo_meta.json

  If present, it can provide:
    * title, claim, falsifiers, expected outputs, controls, etc.
  (This generator will merge it over the embedded default metadata.)

Output
------
Writes into:
  gum/GUM_Report/

  * GUM_Report_v29_<UTC timestamp>.pdf
  * GUM_manifest_v29_<UTC timestamp>.json
  * GUM_manifest_SHA256_v29_<UTC timestamp>.txt
  * (optional) GUM_environment_v29_<UTC timestamp>.txt (pip freeze)

Notes
-----
This script intentionally stays "audit toned":
- It does not claim new measurements.
- It exposes commands, hashes, and artifacts so the reader can falsify or reproduce.

"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---- ReportLab (PDF) ----
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires 'reportlab'. Install with:\n"
        "  python -m pip install reportlab\n\n"
        f"Original import error: {e}"
    )


# --------------------------------------------------------------------------------------
# Repo bootstrap (ensure imports resolve even when executed from tools/, etc.)
# --------------------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# --------------------------------------------------------------------------------------
# Constants / Defaults
# --------------------------------------------------------------------------------------
UTC_TS_FMT = "%Y-%m-%d_%H-%M-%S"

DEFAULT_OUTDIR = REPO_ROOT / "gum" / "GUM_Report"
DEFAULT_LEDGER = REPO_ROOT / "artifacts" / "smoketest_all" / "summary.json"

# The flagship shortlist can evolve; keep it centralized.
DEFAULT_FLAGSHIPS: List[str] = [
    "40",   # substrate
    "71",   # dynamics / one action
    "33",   # standard model closure (new)
    "55",   # proton charge radius
    "66B",  # quantum gravity v2
    "36",   # big bang / cosmology
    "67",   # navier-stokes
    "68",   # general relativity
    "70",   # higgs
]

# Minimal, embedded metadata (used when demo_meta.json isn't available).
# Keep these as tight, falsifiable, technical "certificate claims"—no manifesto language.
EMBEDDED_FLAGSHIP_META: Dict[str, Dict[str, Any]] = {
    "40": {
        "title": "Universe from Zero (Substrate Closure)",
        "domain": "Substrate",
        "claim": (
            "Starting from a single integer input, the pipeline deterministically constructs a "
            "discrete substrate and emits a constrained parameter bundle. The certificate is "
            "a reproducibility test: the output bundle and its sha256 must match exactly."
        ),
    },
    "71": {
        "title": "One Action / Noether Closure",
        "domain": "Foundations",
        "claim": (
            "Derives a compact one-action representation and verifies invariance witnesses. "
            "The certificate is structural: conserved quantities / invariants must satisfy "
            "internal consistency checks to numerical tolerance."
        ),
    },
    "33": {
        "title": "First‑Principles Standard Model (SM‑28 closure)",
        "domain": "Standard Model",
        "claim": (
            "Selects the (U(1), SU(2), SU(3)) survivor triple via SCFP++ and propagates it through "
            "a closed pipeline (SCFP++ → Φ → SM) to produce a structural SM manifest. "
            "PDG comparisons are evaluation-only (optional overlay)."
        ),
    },
    "55": {
        "title": "Proton Charge Radius (high-precision falsifier)",
        "domain": "Standard Model / Precision",
        "claim": (
            "Computes the proton charge radius from the closed structural pipeline and evaluates "
            "the discrepancy against CODATA / muonic-hydrogen determinations. The certificate is "
            "a falsification target: the reported radius value and discrepancy must reproduce."
        ),
    },
    "66B": {
        "title": "Quantum Gravity (Master Flagship v2)",
        "domain": "Quantum Gravity",
        "claim": (
            "Runs the v2 quantum-gravity pipeline and emits a deterministic results bundle. "
            "The certificate is the sha256 of the produced results artifact(s) and pass/fail "
            "integrity gates."
        ),
    },
    "36": {
        "title": "Big Bang (Master Flagship) / Structural Cosmology",
        "domain": "Cosmology",
        "claim": (
            "Constructs a structural cosmology parameter set (e.g., H0, ΩΛ, Ωm, Ωb, ns, σ8) "
            "from the integer substrate and validates internal coherence gates. "
            "Optional: produce/overlay CMB TT spectrum via CAMB if assets/deps are present."
        ),
    },
    "67": {
        "title": "Navier–Stokes (Universality / Turbulence)",
        "domain": "Continuum / Universality",
        "claim": (
            "Runs the Navier–Stokes universality demonstration and emits a results artifact "
            "with deterministic hashes. The certificate checks numerical stability and "
            "artifact reproducibility."
        ),
    },
    "68": {
        "title": "General Relativity (Master Flagship)",
        "domain": "General Relativity",
        "claim": (
            "Computes GR observables (e.g., weak-field regimes / inspiral phasing) from the "
            "structural pipeline and verifies consistency gates. The certificate is reproducible "
            "outputs + hashes."
        ),
    },
    "70": {
        "title": "Higgs (Master Flagship)",
        "domain": "Standard Model / Higgs",
        "claim": (
            "Derives a Higgs-sector surrogate from the same structural channel and emits a "
            "self-contained JSON report + hashes. The certificate verifies the report hash "
            "and internal gate checks."
        ),
    },
}


# --------------------------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------------------------
def norm_demo_id(demo_id: str) -> str:
    """Normalize demo id strings (e.g., '66b' -> '66B')."""
    s = str(demo_id).strip()
    if s.lower().endswith("b") and s[:-1].isdigit():
        return s[:-1] + "B"
    return s


def sha256_bytes(data: bytes) -> str:
    h = hashlib.sha256()
    h.update(data)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_read_text(path: Path, limit: int = 200_000) -> str:
    try:
        data = path.read_text(encoding="utf-8", errors="replace")
        return data[:limit]
    except Exception:
        try:
            data = path.read_bytes()
            return data[:limit].decode("utf-8", errors="replace")
        except Exception:
            return ""


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def find_first_existing(paths: List[Path]) -> Optional[Path]:
    for p in paths:
        if p and p.exists():
            return p
    return None


def find_asset(name: str) -> Optional[Path]:
    """
    Locate an asset by filename across common locations.
    We avoid hardcoding too rigidly because the repo is being refactored.
    """
    candidates = [
        REPO_ROOT / "gum" / "assets" / name,
        REPO_ROOT / "gum" / "GUM_Report" / name,
        REPO_ROOT / "gum" / "GUM_Report" / "CAMB" / name,
        REPO_ROOT / "zz_archive" / "gum" / "assets" / name,
        REPO_ROOT / "zz_archive" / "gum" / "GUM_Report" / name,
    ]
    return find_first_existing(candidates)


def git_head() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return None


# --------------------------------------------------------------------------------------
# Ledger model
# --------------------------------------------------------------------------------------
@dataclass
class DemoRun:
    demo_id: str
    category: str
    folder: str
    demo_path: str
    mode: str
    status: str
    returncode: int
    seconds: float
    code_sha256: str
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    artifacts: Optional[List[Dict[str, Any]]] = None
    notes: Optional[str] = None

    @property
    def demo_dir(self) -> Path:
        return REPO_ROOT / self.demo_path

    def bash_one_liner(self) -> str:
        # Safer than invoking by path because many demos assume cwd==demo_dir.
        args = " --cert" if self.mode == "cert" else ""
        return f'(cd "{self.demo_path}" && python demo.py{args})'

    def ps_one_liner(self) -> str:
        args = " --cert" if self.mode == "cert" else ""
        # PowerShell: push-location/pop-location
        return f'Push-Location "{self.demo_path}"; python demo.py{args}; Pop-Location'


def load_smoketest_ledger(path: Path) -> Dict[str, DemoRun]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs: Dict[str, DemoRun] = {}
    for row in data.get("results", []):
        did = norm_demo_id(row.get("demo_id", ""))
        if not did:
            continue
        runs[did] = DemoRun(
            demo_id=did,
            category=str(row.get("category", "")),
            folder=str(row.get("folder", "")),
            demo_path=str(row.get("demo_path", "")),
            mode=str(row.get("mode", "")),
            status=str(row.get("status", "")),
            returncode=int(row.get("returncode", 0)),
            seconds=float(row.get("seconds", 0.0)),
            code_sha256=str(row.get("code_sha256", "")),
            stdout_path=row.get("stdout_path"),
            stderr_path=row.get("stderr_path"),
            artifacts=row.get("artifacts") or [],
            notes=row.get("notes"),
        )
    return runs


# --------------------------------------------------------------------------------------
# Metric extraction (best-effort)
# --------------------------------------------------------------------------------------
def _parse_demo55_from_stdout(stdout_text: str) -> Dict[str, Any]:
    """
    Demo-55 (proton charge radius) historically prints key lines like:
      r_p(2-loop Lambda_5) = 0.840... fm
      relative error vs CODATA = ...
      discrepancy = ... sigma
    We extract a small set of values for the certificate.
    """
    out: Dict[str, Any] = {}
    m = re.search(r"r_p\(2-loop\s+Lambda_5\)\s*=\s*([0-9.]+)\s*fm", stdout_text)
    if m:
        out["rp_2loop_L5_fm"] = float(m.group(1))
    m = re.search(r"r_p\(1-loop\s+Lambda_5\)\s*=\s*([0-9.]+)\s*fm", stdout_text)
    if m:
        out["rp_1loop_L5_fm"] = float(m.group(1))
    m = re.search(r"relative error vs CODATA\s*=\s*([0-9.eE+-]+)", stdout_text)
    if m:
        try:
            out["relerr_vs_CODATA"] = float(m.group(1))
        except Exception:
            out["relerr_vs_CODATA"] = m.group(1)
    m = re.search(r"discrepancy\s*=\s*([0-9.]+)\s*sigma", stdout_text)
    if m:
        out["sigma_discrepancy"] = float(m.group(1))
    return out


def _parse_demo33_from_stdout(stdout_text: str) -> Dict[str, Any]:
    """
    Demo-33 (First‑Principles Standard Model) prints a compact SCFP++ witness summary.
    We extract:
      * survivor triple (U(1),SU(2),SU(3))
      * survivor pool sizes |U(1)|,|SU(2)|,|SU(3)|
    This enables the report to reproduce the Φ-channel lane table without importing legacy modules.
    """
    out: Dict[str, Any] = {}

    m = re.search(
        r"Survivors\s*\(U\(1\),SU\(2\),SU\(3\)\)\s*([0-9]+)\s*,\s*([0-9]+)\s*,\s*([0-9]+)",
        stdout_text,
    )
    if m:
        out["survivor_triple"] = [int(m.group(1)), int(m.group(2)), int(m.group(3))]

    m = re.search(
        r"\|U\(1\)\|\s*=\s*([0-9]+)\s*,\s*\|SU\(2\)\|\s*=\s*([0-9]+)\s*,\s*\|SU\(3\)\|\s*=\s*([0-9]+)",
        stdout_text,
    )
    if m:
        out["pool_U1"] = int(m.group(1))
        out["pool_SU2"] = int(m.group(2))
        out["pool_SU3"] = int(m.group(3))

    return out


def extract_demo_key_metrics(run: DemoRun) -> Dict[str, Any]:
    """
    Best-effort extraction of a few key metrics for flagship demos.
    This is designed to NEVER break the report build; it returns {} if unknown.
    """
    did = run.demo_id
    demo_dir = run.demo_dir
    metrics: Dict[str, Any] = {}

    # Demo 33 (if present): SM manifest JSONs + stdout witness parse.
    if did == "33":
        pure = demo_dir / "sm_outputs_pure.json"
        dressed = demo_dir / "sm_outputs.json"
        if pure.exists():
            j = safe_read_json(pure)
            if isinstance(j, dict):
                for k in ["snapshot_hash", "alpha0_inv", "sin2W", "alpha_s", "v", "MW", "MZ", "G_F"]:
                    if k in j:
                        metrics[k] = j[k]
                if "manifest" in j and isinstance(j["manifest"], dict):
                    mf = j["manifest"]
                    for k in ["alpha0_inv", "sin2W", "alpha_s", "v", "MW", "MZ", "Lambda_QCD"]:
                        if k in mf:
                            metrics[f"manifest.{k}"] = mf[k]
        if dressed.exists():
            j = safe_read_json(dressed)
            if isinstance(j, dict):
                if "alpha_MZ_inv" in j:
                    metrics["alpha_MZ_inv"] = j["alpha_MZ_inv"]
        # Also parse stdout witness lines (pool sizes / survivor triple), if available
        if run.stdout_path:
            stdout_file = REPO_ROOT / run.stdout_path
            if stdout_file.exists():
                metrics.update(_parse_demo33_from_stdout(safe_read_text(stdout_file)))

    # Demo 36: bb36_master_results.json
    if did == "36":
        j = safe_read_json(demo_dir / "bb36_master_results.json")
        if isinstance(j, dict):
            cs = j.get("cosmo_structural") or {}
            if isinstance(cs, dict):
                for k in ["H0", "Omega_b", "Omega_c", "Omega_m", "Omega_L", "n_s", "sigma8"]:
                    if k in cs:
                        metrics[k] = cs[k]
            if "determinism_sha256" in j:
                metrics["determinism_sha256"] = j["determinism_sha256"]

    # Demo 40: demo40_master_upgrade_results.json
    if did == "40":
        j = safe_read_json(demo_dir / "demo40_master_upgrade_results.json")
        if isinstance(j, dict):
            core = j.get("core_bundle") or {}
            if isinstance(core, dict):
                pp = core.get("params_primary") or {}
                if isinstance(pp, dict):
                    for k in ["H0", "Omega_L", "Omega_m", "Omega_b", "alpha0_inv", "sin2W", "alpha_s"]:
                        if k in pp:
                            metrics[k] = pp[k]
            if "bundle_sha256" in j:
                metrics["bundle_sha256"] = j["bundle_sha256"]

    # Demo 66B: demo66_master_results.json
    if did == "66B":
        j = safe_read_json(demo_dir / "demo66_master_results.json")
        if isinstance(j, dict):
            for k in ["status", "snapshot_hash", "bundle_sha256", "notes"]:
                if k in j:
                    metrics[k] = j[k]

    # Demo 67: demo67_master_results_*.json
    if did == "67":
        candidates = sorted(demo_dir.glob("demo67_master_results_*.json"))
        if candidates:
            j = safe_read_json(candidates[0])
            if isinstance(j, dict):
                for k in ["status", "snapshot_hash", "dt", "Re", "grid", "notes"]:
                    if k in j:
                        metrics[k] = j[k]

    # Demo 68: DEMO68_GR_results.json
    if did == "68":
        j = safe_read_json(demo_dir / "DEMO68_GR_results.json")
        if isinstance(j, dict):
            for k in ["status", "snapshot_hash", "notes"]:
                if k in j:
                    metrics[k] = j[k]

    # Demo 70: prework70A_higgs_report.json
    if did == "70":
        j = safe_read_json(demo_dir / "prework70A_higgs_report.json")
        if isinstance(j, dict):
            ew = j.get("ew_lawful") or {}
            if isinstance(ew, dict):
                for k in ["v", "MW", "MZ", "alpha_MZ_inv"]:
                    if k in ew:
                        metrics[k] = ew[k]
            if "primary" in j and isinstance(j["primary"], dict):
                p = j["primary"]
                if "triple" in p:
                    metrics["triple"] = p["triple"]

    # Demo 71: demo71_master_results.json
    if did == "71":
        j = safe_read_json(demo_dir / "demo71_master_results.json")
        if isinstance(j, dict):
            for k in ["status", "snapshot_hash", "notes"]:
                if k in j:
                    metrics[k] = j[k]

    # Demo 55: parse stdout log (if available)
    if did == "55":
        if run.stdout_path:
            stdout_file = REPO_ROOT / run.stdout_path
            if stdout_file.exists():
                metrics.update(_parse_demo55_from_stdout(safe_read_text(stdout_file)))

    return metrics


# --------------------------------------------------------------------------------------
# Manifest / Report assembly
# --------------------------------------------------------------------------------------
def build_manifest(
    *,
    version: str,
    timestamp_utc: str,
    ledger_path: Optional[str],
    ledger_sha256: Optional[str],
    runs: Dict[str, DemoRun],
    flagships: List[str],
    assets: Dict[str, Optional[str]],
    notes: List[str],
) -> Dict[str, Any]:
    """
    Create a machine-readable manifest for citation and auditing.
    """
    statuses = [r.status for r in runs.values()]
    summary = {
        "total": len(statuses),
        "pass": sum(1 for s in statuses if s.upper() == "PASS"),
        "fail": sum(1 for s in statuses if s.upper() == "FAIL"),
        "error": sum(1 for s in statuses if s.upper() == "ERROR"),
        "unknown": sum(1 for s in statuses if s.upper() not in {"PASS", "FAIL", "ERROR"}),
    }

    demos: Dict[str, Any] = {}
    for did, r in sorted(runs.items(), key=lambda kv: kv[0]):
        rec = {
            "demo_id": did,
            "category": r.category,
            "folder": r.folder,
            "demo_path": r.demo_path,
            "mode": r.mode,
            "status": r.status,
            "returncode": r.returncode,
            "seconds": r.seconds,
            "code_sha256": r.code_sha256,
            "stdout_path": r.stdout_path,
            "stderr_path": r.stderr_path,
            "artifacts": r.artifacts or [],
        }
        if did in set(flagships):
            rec["key_metrics"] = extract_demo_key_metrics(r)
        demos[did] = rec

    manifest: Dict[str, Any] = {
        "gum_report_version": version,
        "timestamp_utc": timestamp_utc,
        "git_head": git_head(),
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "ledger": {
            "path": ledger_path,
            "sha256": ledger_sha256,
        },
        "summary": summary,
        "flagships": [norm_demo_id(x) for x in flagships],
        "assets": assets,
        "demos": demos,
        "notes": notes,
    }

    canonical = json.dumps(manifest, sort_keys=True, indent=2)
    manifest["manifest_sha256"] = sha256_text(canonical)

    numeric_tokens: List[str] = []
    for did in manifest["flagships"]:
        rec = demos.get(did, {})
        km = rec.get("key_metrics", {}) if isinstance(rec, dict) else {}
        if isinstance(km, dict):
            for k in sorted(km.keys()):
                v = km[k]
                if isinstance(v, (int, float)):
                    numeric_tokens.append(f"{did}:{k}={v:.16g}")
                elif isinstance(v, str) and re.fullmatch(r"[0-9.eE+-]+", v):
                    numeric_tokens.append(f"{did}:{k}={v}")
    manifest["numeric_sha256"] = sha256_text("\n".join(numeric_tokens))

    return manifest


def _monospace(text: str) -> str:
    esc = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f'<font face="Courier">{esc}</font>'


def _table(data: List[List[Any]], col_widths: Optional[List[float]] = None) -> Table:
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return t


def _img(path: Path, width: float) -> Image:
    im = Image(str(path))
    im.drawWidth = width
    if im.imageHeight:
        im.drawHeight = width * (im.imageHeight / im.imageWidth)
    return im


def build_pdf_report(
    *,
    pdf_path: Path,
    manifest: Dict[str, Any],
    runs: Dict[str, DemoRun],
    flagships: List[str],
    meta: Dict[str, Dict[str, Any]],
    assets: Dict[str, Optional[Path]],
) -> None:
    styles = getSampleStyleSheet()
    # Custom styles (avoid name collisions across reportlab versions)
    styles.add(ParagraphStyle(name="GUM_h1", parent=styles["Heading1"], fontSize=16, spaceAfter=10))
    styles.add(ParagraphStyle(name="GUM_h2", parent=styles["Heading2"], fontSize=12, spaceAfter=8))
    styles.add(ParagraphStyle(name="GUM_body", parent=styles["BodyText"], fontSize=9, leading=12))
    styles.add(ParagraphStyle(name="GUM_mono", parent=styles["BodyText"], fontSize=8, leading=10, fontName="Courier"))
    styles.add(ParagraphStyle(name="GUM_tiny", parent=styles["BodyText"], fontSize=7, leading=9))

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="GUM Report v29 (System Audit)",
        author="Marithmetics",
    )

    story: List[Any] = []

    # ----------------------------------------------------------------------------------
    # Title / Audit Contract / Quickstart
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("GUM Report v29 — System Audit / Falsification Ledger", styles["Title"]))
    story.append(Spacer(1, 0.2 * inch))

    quote = "“Within everything accepted lies everything overlooked.”"
    story.append(Paragraph(quote, styles["GUM_body"]))
    story.append(Spacer(1, 0.15 * inch))

    audit_contract = (
        "This document is an <b>audit artifact</b>. It does not propose new empirical measurements. "
        "The central claim is structural and computational: a deterministic integer-driven pipeline "
        "emits parameter bundles and derived observables. The reader is invited to audit, reproduce, "
        "or falsify the pipeline by running the one-liners below and verifying the recorded hashes."
    )
    story.append(Paragraph(audit_contract, styles["GUM_body"]))
    story.append(Spacer(1, 0.2 * inch))

    # Quickstart table (flagships first)
    qs_rows: List[List[Any]] = [
        ["Falsification Quickstart (copy/paste)", "What it exercises"],
    ]
    for did in [norm_demo_id(x) for x in flagships]:
        r = runs.get(did)
        m = meta.get(did, {})
        # Optional per-demo metadata override
        if r:
            dm = safe_read_json(r.demo_dir / "demo_meta.json")
            if isinstance(dm, dict):
                m = {**m, **dm}

        title = m.get("title", f"Demo {did}")
        if r:
            cmd = r.bash_one_liner()
        else:
            cmd = f'(cd "demos/<category>/demo-{did}-<slug>" && python demo.py)'
        qs_rows.append(
            [Paragraph(_monospace(cmd), styles["GUM_mono"]), Paragraph(title, styles["GUM_body"])]
        )

    story.append(_table(qs_rows, col_widths=[4.5 * inch, 2.5 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    story.append(
        Paragraph(
            f"<b>Manifest SHA256</b>: {manifest.get('manifest_sha256','')}"
            f"<br/><b>Numeric SHA256</b>: {manifest.get('numeric_sha256','')}",
            styles["GUM_tiny"],
        )
    )
    story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Section 0: Visual proof — DRPT Geometry (keep v28 flow)
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Section 0 — Visual Proof: DRPT Geometry", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "Digital Root Power Tables (DRPT) visualize stable modular patterns that recur across bases. "
            "In v29 we treat these as <i>structural witnesses</i>: they motivate why specific integers "
            "appear as persistent “identity pillars” and “echo” motifs in the substrate scan.",
            styles["GUM_body"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    img_names = ["DRPTCity137_v2.png", "DRPTSurvivor137_v2.png", "Identity9.png", "Echo10.png"]
    imgs: List[Optional[Image]] = []
    for name in img_names:
        p = assets.get(name)
        imgs.append(_img(p, width=2.8 * inch) if p else None)

    grid = [
        [imgs[0] or Paragraph(f"(missing {img_names[0]})", styles["GUM_tiny"]),
         imgs[1] or Paragraph(f"(missing {img_names[1]})", styles["GUM_tiny"])],
        [imgs[2] or Paragraph(f"(missing {img_names[2]})", styles["GUM_tiny"]),
         imgs[3] or Paragraph(f"(missing {img_names[3]})", styles["GUM_tiny"])],
    ]
    story.append(Table(grid, colWidths=[3.2 * inch, 3.2 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    bridge = (
        "<b>Bridge hypothesis (explicit)</b>: We posit that physical constants are the eigenvalues of "
        "discrete integer structures. Just as vibrational modes of a boundary-value problem determine "
        "its resonant spectrum, the stable “identity pillars” of the integer substrate constrain the "
        "resonant masses and couplings observed in effective field theories."
    )
    story.append(Paragraph(bridge, styles["GUM_body"]))

    story.append(Spacer(1, 0.12 * inch))
    story.append(Paragraph("Φ-channel lane rules (witness space)", styles["GUM_h2"]))
    # Default lane rules (legacy v28 witness thresholds). Pool sizes are pulled from Demo 33 stdout when available.
    pool_u1 = pool_su2 = pool_su3 = "—"
    r33 = runs.get("33")
    if r33:
        km33 = extract_demo_key_metrics(r33)
        pool_u1 = str(km33.get("pool_U1", pool_u1))
        pool_su2 = str(km33.get("pool_SU2", pool_su2))
        pool_su3 = str(km33.get("pool_SU3", pool_su3))

    lane_rows = [
        ["Lane", "Modulus", "Θ threshold", "Legendre", "v2(w−1)", "Pool size"],
        ["U(1)", "137", "Θ ≥ 0.30", "+1", "0", pool_u1],
        ["SU(2)", "107", "Θ ≥ 0.29", "−1", "1", pool_su2],
        ["SU(3)", "103", "Θ ≥ 0.30", "+1", "1", pool_su3],
    ]
    story.append(_table([[Paragraph(str(c), styles["GUM_tiny"]) for c in row] for row in lane_rows],
                        col_widths=[0.8*inch, 0.8*inch, 1.2*inch, 0.8*inch, 0.9*inch, 0.9*inch]))
    story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Section 0B: Fejér smoothing (keep)
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Section 0B — Analytic Filter: Fejér Smoothing", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "We retain the Fejér smoothing plates as the canonical visualization of the analytic filter "
            "used in the legacy pipeline. These images are treated as fixed reference plates for audit.",
            styles["GUM_body"],
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    fejer_left = assets.get("FejerSmoothignLeft.png")
    fejer_right = assets.get("FejerSmoothingRight.png")
    if fejer_left and fejer_right:
        row = [
            _img(fejer_left, width=3.0 * inch),
            _img(fejer_right, width=3.0 * inch),
        ]
        story.append(Table([row], colWidths=[3.2 * inch, 3.2 * inch]))
    else:
        story.append(Paragraph("(Fejér plates missing — see manifest assets list)", styles["GUM_tiny"]))
    story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Section 1: Executive summary + what changed in v29
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Section 1 — Executive Summary (v29)", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "v29 keeps the v28 narrative spine (DRPT → filter → outputs → validation) but changes the "
            "reader contract: the report is organized as an <b>audit</b>. The primary deliverables are "
            "copy/paste falsification commands, reproducible artifacts, and hashes.",
            styles["GUM_body"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    # Ledger summary (if available)
    summ = manifest.get("summary", {}) if isinstance(manifest, dict) else {}
    if summ:
        story.append(
            Paragraph(
                f"<b>Smoketest ledger</b>: total={summ.get('total')}  "
                f"PASS={summ.get('pass')}  FAIL={summ.get('fail')}  ERROR={summ.get('error')}",
                styles["GUM_body"],
            )
        )
        story.append(Spacer(1, 0.08 * inch))

    story.append(Paragraph("Flagship coverage (shortlist)", styles["GUM_h2"]))
    bullets = []
    for did in [norm_demo_id(x) for x in flagships]:
        r = runs.get(did)
        m = meta.get(did, {})
        # Optional per-demo metadata override
        if r:
            dm = safe_read_json(r.demo_dir / "demo_meta.json")
            if isinstance(dm, dict):
                m = {**m, **dm}

        title = m.get("title", f"Demo {did}")
        dom = m.get("domain", "—")
        status = runs.get(did).status if runs.get(did) else "MISSING"
        bullets.append(f"• <b>Demo {did}</b> [{dom}] — {title} — status: <b>{status}</b>")
    story.append(Paragraph("<br/>".join(bullets), styles["GUM_body"]))

    story.append(Spacer(1, 0.14 * inch))
    story.append(Paragraph("Unified Dashboard (headline values)", styles["GUM_h2"]))

    def _pick_metric(candidates: List[Tuple[str, str]]) -> Tuple[str, Any]:
        """
        candidates: [(demo_id, key), ...]
        returns: (source_label, value) or ("—","—")
        """
        for demo_id, key in candidates:
            rr = runs.get(demo_id)
            if not rr:
                continue
            km = extract_demo_key_metrics(rr)
            if key in km and km[key] is not None:
                return (f"Demo {demo_id}", km[key])
        return ("—", "—")

    dash_spec: List[Tuple[str, List[Tuple[str, str]]]] = [
        ("Survivor triple (U(1),SU(2),SU(3))", [("33", "survivor_triple"), ("70", "triple")]),
        ("α0⁻¹", [("33", "alpha0_inv"), ("40", "alpha0_inv")]),
        ("sin²θW", [("33", "sin2W"), ("40", "sin2W")]),
        ("αs", [("33", "alpha_s"), ("40", "alpha_s")]),
        ("v  [GeV]", [("70", "v"), ("33", "v")]),
        ("MW [GeV]", [("70", "MW"), ("33", "MW")]),
        ("MZ [GeV]", [("70", "MZ"), ("33", "MZ")]),
        ("H0", [("36", "H0"), ("40", "H0")]),
        ("Ωb", [("36", "Omega_b")]),
        ("Ωm", [("36", "Omega_m")]),
        ("ΩΛ", [("36", "Omega_L")]),
        ("ns", [("36", "n_s")]),
        ("σ8", [("36", "sigma8")]),
        ("Proton charge radius rp [fm]", [("55", "rp_2loop_L5_fm")]),
    ]

    dash_rows = [["Quantity", "Value", "Source"]]
    for label, cands in dash_spec:
        src, val = _pick_metric(cands)
        dash_rows.append([label, str(val), src])

    story.append(_table([[Paragraph(str(c), styles["GUM_tiny"]) for c in row] for row in dash_rows],
                        col_widths=[2.8*inch, 2.2*inch, 1.2*inch]))

    story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Section 2: Falsification Matrix (flagships) + ledger pointer
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Section 2 — Falsification Matrix", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "This matrix is the front-door for skeptics: it lists what each flagship claims, how to "
            "run it, and what hashes must reproduce. Full-demo coverage is provided by the smoketest "
            "ledger referenced in the manifest.",
            styles["GUM_body"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    fm_rows: List[List[Any]] = [
        ["Demo", "Claim (1‑paragraph certificate)", "One‑liner", "Status", "Code SHA256"],
    ]
    for did in [norm_demo_id(x) for x in flagships]:
        r = runs.get(did)
        m = meta.get(did, {})
        # Optional per-demo metadata override
        if r:
            dm = safe_read_json(r.demo_dir / "demo_meta.json")
            if isinstance(dm, dict):
                m = {**m, **dm}

        claim = m.get("claim", "")
        title = m.get("title", f"Demo {did}")
        if r:
            cmd = r.bash_one_liner()
            status = r.status
            code_sha = r.code_sha256[:16] + "…" if r.code_sha256 else ""
        else:
            cmd = ""
            status = "MISSING"
            code_sha = ""
        fm_rows.append(
            [
                Paragraph(f"<b>{did}</b><br/>{title}", styles["GUM_tiny"]),
                Paragraph(claim, styles["GUM_tiny"]),
                Paragraph(_monospace(cmd), styles["GUM_mono"]),
                Paragraph(status, styles["GUM_tiny"]),
                Paragraph(_monospace(code_sha), styles["GUM_mono"]),
            ]
        )

    story.append(_table(fm_rows, col_widths=[0.8 * inch, 2.5 * inch, 2.3 * inch, 0.7 * inch, 1.2 * inch]))
    story.append(Spacer(1, 0.15 * inch))

    ledger_info = manifest.get("ledger", {})
    story.append(
        Paragraph(
            f"<b>Full smoketest ledger</b>: {ledger_info.get('path')}<br/>"
            f"<b>Ledger SHA256</b>: {ledger_info.get('sha256')}",
            styles["GUM_tiny"],
        )
    )
    story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Section 3: Flagship Certificates (one per demo)
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Section 3 — Flagship Certificates", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "Each certificate is a modular unit: premise → claim → command → outputs → hashes. "
            "If any listed hash does not reproduce under the stated command, the certificate fails.",
            styles["GUM_body"],
        )
    )
    story.append(PageBreak())

    for did in [norm_demo_id(x) for x in flagships]:
        r = runs.get(did)
        m = meta.get(did, {})
        # Optional per-demo metadata override
        if r:
            dm = safe_read_json(r.demo_dir / "demo_meta.json")
            if isinstance(dm, dict):
                m = {**m, **dm}

        title = m.get("title", f"Demo {did}")
        dom = m.get("domain", "—")
        claim = m.get("claim", "")

        story.append(Paragraph(f"Demo {did} — {title}", styles["GUM_h1"]))
        story.append(Paragraph(f"<b>Domain</b>: {dom}", styles["GUM_body"]))
        story.append(Spacer(1, 0.06 * inch))
        story.append(Paragraph(claim, styles["GUM_body"]))
        story.append(Spacer(1, 0.10 * inch))

        if r:
            cmd = r.bash_one_liner()
            story.append(Paragraph(f"<b>Falsify / Reproduce</b>:<br/>{_monospace(cmd)}", styles["GUM_mono"]))
            story.append(Spacer(1, 0.08 * inch))

            km = extract_demo_key_metrics(r)
            if km:
                km_rows = [["Key metric", "Value"]]
                for k in sorted(km.keys()):
                    km_rows.append([Paragraph(k, styles["GUM_tiny"]), Paragraph(str(km[k]), styles["GUM_tiny"])])
                story.append(_table(km_rows, col_widths=[2.2 * inch, 4.8 * inch]))
                story.append(Spacer(1, 0.10 * inch))

            art_rows = [["Artifact", "SHA256"]]
            for a in (r.artifacts or [])[:12]:  # cap for PDF; full list remains in manifest
                art_rows.append([Paragraph(str(a.get("name","")), styles["GUM_tiny"]),
                                 Paragraph(_monospace(str(a.get("sha256",""))), styles["GUM_mono"])])
            if r.stdout_path:
                sp = REPO_ROOT / r.stdout_path
                if sp.exists():
                    art_rows.append([Paragraph("stdout.log (smoketest)", styles["GUM_tiny"]),
                                     Paragraph(_monospace(sha256_file(sp)), styles["GUM_mono"])])
            if r.stderr_path:
                ep = REPO_ROOT / r.stderr_path
                if ep.exists():
                    art_rows.append([Paragraph("stderr.log (smoketest)", styles["GUM_tiny"]),
                                     Paragraph(_monospace(sha256_file(ep)), styles["GUM_mono"])])

            story.append(_table(art_rows, col_widths=[2.2 * inch, 4.8 * inch]))
            story.append(Spacer(1, 0.10 * inch))

            story.append(
                Paragraph(
                    f"<b>Status</b>: {r.status} (exit={r.returncode}, {r.seconds:.2f}s)"
                    f"<br/><b>Code SHA256</b>: {_monospace(r.code_sha256)}",
                    styles["GUM_tiny"],
                )
            )
        else:
            story.append(Paragraph("Status: MISSING from ledger (run smoketest to populate).", styles["GUM_body"]))

        story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Section 4: Legacy Visual Validation (Big Bang + CAMB overlay)
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Section 4 — Validation Plates (Cosmology)", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "We retain the legacy validation plates. If a plate is missing in your environment, "
            "the report does not silently invent it; it records the absence and points to the data bundle.",
            styles["GUM_body"],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    bb = assets.get("BB36_big_bang.png")
    if bb:
        story.append(Paragraph("<b>BB36: Structural Big Bang plate</b>", styles["GUM_h2"]))
        story.append(_img(bb, width=6.2 * inch))
        story.append(Spacer(1, 0.12 * inch))
    else:
        story.append(Paragraph("(missing BB36_big_bang.png)", styles["GUM_tiny"]))

    cmb = assets.get("TT_spectrum_Planck2018_vs_GUM.png") or assets.get("TT_spectrum_Planck_vs_GUM.png")
    if cmb:
        story.append(Paragraph("<b>CMB TT Power Spectrum (Planck vs GUM)</b>", styles["GUM_h2"]))
        story.append(_img(cmb, width=6.2 * inch))
        story.append(Spacer(1, 0.12 * inch))
    else:
        story.append(
            Paragraph(
                "(CMB overlay image missing. If you want this plate, ensure the Planck/CAMB asset bundle "
                "is present or run the CAMB check script in gum/.)",
                styles["GUM_tiny"],
            )
        )

    story.append(PageBreak())

    # ----------------------------------------------------------------------------------
    # Appendix: Reproducibility pointers
    # ----------------------------------------------------------------------------------
    story.append(Paragraph("Appendix — Reproducibility", styles["GUM_h1"]))
    story.append(
        Paragraph(
            "Primary reproduction path is: (1) run the repo smoketest to generate the ledger, "
            "then (2) regenerate this report using that ledger. The manifest JSON is the citation unit.",
            styles["GUM_body"],
        )
    )
    story.append(Spacer(1, 0.15 * inch))

    # Failures / errors (if any)
    if runs:
        bad = [did for did, rr in sorted(runs.items()) if rr.status.upper() in {"FAIL", "ERROR"}]
        if bad:
            story.append(Paragraph("Smoketest failures (must resolve before publication)", styles["GUM_h2"]))
            lines = []
            for did in bad:
                rr = runs[did]
                lines.append(f"• Demo {did} ({rr.category}) — {rr.status} (exit={rr.returncode})")
            story.append(Paragraph("<br/>".join(lines), styles["GUM_body"]))
            story.append(Spacer(1, 0.12 * inch))
        else:
            story.append(Paragraph("Smoketest status: <b>all demos PASS</b> in the provided ledger.", styles["GUM_body"]))
            story.append(Spacer(1, 0.12 * inch))
    else:
        story.append(Paragraph("Smoketest status: ledger not provided (no run data).", styles["GUM_body"]))
        story.append(Spacer(1, 0.12 * inch))

    excerpt_rows = [
        ["Field", "Value"],
        ["git_head", manifest.get("git_head") or ""],
        ["manifest_sha256", manifest.get("manifest_sha256","")],
        ["numeric_sha256", manifest.get("numeric_sha256","")],
        ["ledger_path", (manifest.get("ledger", {}) or {}).get("path","")],
        ["ledger_sha256", (manifest.get("ledger", {}) or {}).get("sha256","")],
    ]
    story.append(_table([[Paragraph(a, styles["GUM_tiny"]), Paragraph(_monospace(str(b)), styles["GUM_mono"])] for a,b in excerpt_rows],
                        col_widths=[1.4*inch, 5.6*inch]))

    doc.build(story)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="Generate the GUM v29 PDF report (system audit).")
    ap.add_argument("--ledger", type=str, default=str(DEFAULT_LEDGER),
                    help="Path to smoketest ledger JSON (default: artifacts/smoketest_all/summary.json).")
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR),
                    help="Output directory (default: gum/GUM_Report).")
    ap.add_argument("--flagships", type=str, default=",".join(DEFAULT_FLAGSHIPS),
                    help="Comma-separated demo ids to treat as flagships.")
    ap.add_argument("--no-pip-freeze", action="store_true",
                    help="Skip pip freeze capture (faster; less reproducible).")
    args = ap.parse_args()

    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc).strftime(UTC_TS_FMT)

    ledger_path = Path(args.ledger).resolve()
    runs: Dict[str, DemoRun] = {}
    ledger_sha = None
    notes: List[str] = []

    if ledger_path.exists():
        ledger_sha = sha256_file(ledger_path)
        runs = load_smoketest_ledger(ledger_path)
    else:
        notes.append(f"ledger_missing: {str(ledger_path)}")
        runs = {}

    flagships = [norm_demo_id(x) for x in (args.flagships.split(",") if args.flagships else []) if x.strip()]

    asset_names = [
        "DRPTCity137_v2.png",
        "DRPTSurvivor137_v2.png",
        "Identity9.png",
        "Identity10.png",
        "Echo6.png",
        "Echo10.png",
        "FejerSmoothignLeft.png",
        "FejerSmoothingRight.png",
        "BB36_big_bang.png",
        "TT_spectrum_Planck2018_vs_GUM.png",
        "TT_spectrum_Planck_vs_GUM.png",
    ]
    assets: Dict[str, Optional[Path]] = {name: find_asset(name) for name in asset_names}

    meta: Dict[str, Dict[str, Any]] = {norm_demo_id(k): v for k, v in EMBEDDED_FLAGSHIP_META.items()}

    assets_manifest = {k: (str(v) if v else None) for k, v in assets.items()}
    manifest = build_manifest(
        version="v29",
        timestamp_utc=ts,
        ledger_path=str(ledger_path) if ledger_path.exists() else None,
        ledger_sha256=ledger_sha,
        runs=runs,
        flagships=flagships,
        assets=assets_manifest,
        notes=notes,
    )

    canonical = json.dumps(manifest, sort_keys=True, indent=2)
    manifest_sha = sha256_text(canonical)
    manifest["manifest_sha256"] = manifest_sha

    manifest_path = outdir / f"GUM_manifest_v29_{ts}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    sha_path = outdir / f"GUM_manifest_SHA256_v29_{ts}.txt"
    sha_path.write_text(manifest.get("manifest_sha256", "") + "\n", encoding="utf-8")

    if not args.no_pip_freeze:
        try:
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], cwd=str(REPO_ROOT))
            env_txt = freeze.decode("utf-8", errors="replace")
        except Exception as e:
            env_txt = f"(pip freeze failed: {e})\n"
        env_path = outdir / f"GUM_environment_v29_{ts}.txt"
        env_path.write_text(env_txt, encoding="utf-8")

    pdf_path = outdir / f"GUM_Report_v29_{ts}.pdf"
    build_pdf_report(
        pdf_path=pdf_path,
        manifest=manifest,
        runs=runs,
        flagships=flagships,
        meta=meta,
        assets=assets,
    )

    print("Wrote:")
    print(f"  {pdf_path}")
    print(f"  {manifest_path}")
    print(f"  {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
PY

chmod +x gum/gum_report_generator_v29.py
