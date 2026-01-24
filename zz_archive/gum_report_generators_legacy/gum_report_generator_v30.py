#!/usr/bin/env python3
"""
GUM Report v29 - System Audit (Masterpiece spec)

Goals vs v28:
- Keep v28's visual proof plates (DRPT + Fejer + BB36).
- Move falsification forward: a quickstart, a full matrix, and certificate pages.
- Use smoketest ledger hashes wherever possible (paper-citable).
- Never print unusable commands (no "cd into a file path"; no blind "--cert").

Run (from repo root):
    python gum/gum_report_generator_v29.py --ledger artifacts/smoketest_all/summary.json

Output:
    gum/GUM_Report/GUM_Report_v29_<timestamp>.pdf
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# ReportLab (PDF)
# -----------------------------
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Image,
        Table,
        TableStyle,
        PageBreak,
        Preformatted,
    )
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "This script requires reportlab. Install with:\n"
        "    python -m pip install reportlab\n"
        f"Original error: {e}"
    )


VERSION = "v29"
TITLE = "GUM Report v29"
SUBTITLE = "System Audit of a Discrete-Integer Substrate (Marithmetics)"
QUOTE = '"Within everything accepted lies everything overlooked."'

# -----------------------------
# Utilities
# -----------------------------
def now_utc_stamp() -> str:
    return _dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_json(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256_bytes(s.encode("utf-8"))


def safe_read_text(path: Path, max_bytes: int = 2_000_000) -> str:
    try:
        b = path.read_bytes()
        if len(b) > max_bytes:
            b = b[:max_bytes]
        return b.decode("utf-8", errors="replace")
    except Exception:
        return ""


def try_git_rev(repo_root: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def shorten_sha(sha: str, n: int = 12) -> str:
    sha = (sha or "").strip()
    return sha[:n] if len(sha) >= n else sha


def human_seconds(sec: Optional[float]) -> str:
    if sec is None:
        return "n/a"
    try:
        sec_f = float(sec)
    except Exception:
        return "n/a"
    if sec_f < 1.0:
        return f"{sec_f:.3f}s"
    if sec_f < 60.0:
        return f"{sec_f:.2f}s"
    m = int(sec_f // 60)
    s = sec_f - 60 * m
    return f"{m}m{s:0.1f}s"


def escape_xml(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# -----------------------------
# Demo model
# -----------------------------
@dataclass
class DemoMeta:
    demo_id: str
    category: str = ""
    slug: str = ""
    demo_dir: Optional[Path] = None
    script_path: Optional[Path] = None
    readme_path: Optional[Path] = None
    title: str = ""
    summary: str = ""  # one paragraph
    falsify: str = ""  # short falsifier
    run_cmd: str = ""        # preferred (quickstart/certs)
    run_cmd_smoke: str = ""  # safe baseline (matrix)
    run_cmd_cert: str = ""   # optional (only if provably supported)


@dataclass
class DemoLedger:
    status: str = "UNKNOWN"
    exit_code: Optional[int] = None
    duration_s: Optional[float] = None
    code_sha256: str = ""
    stdout_sha256: str = ""
    stdout_path: Optional[Path] = None
    command: str = ""  # optional
    artifacts: Dict[str, str] = field(default_factory=dict)


@dataclass
class DemoRecord:
    meta: DemoMeta
    ledger: DemoLedger


# -----------------------------
# demo_map.yaml parsing (no PyYAML dependency)
# -----------------------------
_YAML_LINE_RE = re.compile(
    r'^\s*"(?P<id>[0-9A-Za-z]+)"\s*:\s*\{category:\s*"(?P<cat>[^"]+)"\s*,\s*slug:\s*"(?P<slug>[^"]+)"\s*\}\s*$'
)


def parse_demo_map_yaml(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    txt = safe_read_text(path)
    out: Dict[str, Dict[str, str]] = {}
    in_demos = False
    for raw in txt.splitlines():
        line = raw.rstrip("\n")
        if not in_demos:
            if line.strip() == "demos:":
                in_demos = True
            continue
        if not line.strip():
            continue
        m = _YAML_LINE_RE.match(line)
        if not m:
            continue
        demo_id = m.group("id").strip()
        out[demo_id] = {"category": m.group("cat").strip(), "slug": m.group("slug").strip()}
    return out


def normalize_demo_id_for_dir(demo_id: str) -> str:
    return demo_id.strip().lower()


def canonical_category(cat: str) -> str:
    c = (cat or "").strip()
    aliases = {"sm": "standard_model", "gr": "general_relativity", "cosmology": "cosmo"}
    return aliases.get(c, c)


def title_from_slug(slug: str) -> str:
    s = (slug or "").strip()
    return s.replace("-", " ").replace("_", " ").title() if s else ""


def resolve_demo_dir(repo_root: Path, demo_id: str, category: str, slug: str) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    demos_root = repo_root / "demos"
    if not demos_root.exists():
        return None, None, None

    cid = normalize_demo_id_for_dir(demo_id)
    dir_name = f"demo-{cid}-{slug}".lower()

    cat_candidates = [canonical_category(category)]
    if category and category != canonical_category(category):
        cat_candidates.append(category)

    # 1) direct
    for cat in cat_candidates:
        cand = demos_root / cat / dir_name
        if cand.exists() and (cand / "demo.py").exists():
            readme = cand / "README.md"
            return cand, cand / "demo.py", readme if readme.exists() else None

    # 2) search exact directory name
    for cand in demos_root.glob(f"*/{dir_name}"):
        if cand.is_dir() and (cand / "demo.py").exists():
            readme = cand / "README.md"
            return cand, cand / "demo.py", readme if readme.exists() else None

    # 3) prefix search
    prefix = f"demo-{cid}-"
    for cand in demos_root.rglob(f"{prefix}*"):
        if cand.is_dir() and (cand / "demo.py").exists():
            readme = cand / "README.md"
            return cand, cand / "demo.py", readme if readme.exists() else None

    return None, None, None


def script_supports_flag(script_path: Optional[Path], flag: str = "--cert") -> bool:
    """Conservative: only advertise optional flags if we can prove support via source scan."""
    if not script_path or not script_path.exists():
        return False
    try:
        txt = script_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return False
    return flag in txt


# -----------------------------
# README extraction helpers
# -----------------------------
def extract_one_paragraph_summary(readme_text: str) -> str:
    if not readme_text:
        return ""
    lines = [ln.rstrip() for ln in readme_text.splitlines()]

    # Pattern: "## SUMMARY ..."
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("## summary"):
            buf: List[str] = []
            for j in range(i + 1, len(lines)):
                s = lines[j].strip()
                if not s:
                    if buf:
                        break
                    continue
                if s.startswith("#"):
                    break
                if s.startswith("```"):
                    continue
                buf.append(s)
            return " ".join(buf).strip()

    # Fallback: first paragraph after header
    buf2: List[str] = []
    started = False
    for ln in lines:
        s = ln.strip()
        if not started:
            if not s or s.startswith("#"):
                continue
            started = True
            buf2.append(s)
            continue
        if not s:
            break
        if s.startswith("#") or s.startswith("```"):
            break
        buf2.append(s)
    return " ".join(buf2).strip()


def extract_falsifier_hint(readme_text: str) -> str:
    if not readme_text:
        return ""
    lines = [ln.rstrip() for ln in readme_text.splitlines()]
    for i, ln in enumerate(lines):
        low = ln.strip().lower()
        if low.startswith("## falsif") or low.startswith("## teeth") or low.startswith("## pass/fail"):
            buf: List[str] = []
            for j in range(i + 1, min(i + 16, len(lines))):
                s = lines[j].strip()
                if not s:
                    if buf:
                        break
                    continue
                if s.startswith("#") or s.startswith("```"):
                    break
                buf.append(s.lstrip("*- ").strip())
            out = " ".join(buf).strip()
            return out[:240]
    return ""


# -----------------------------
# Smoketest ledger loading / normalization
# -----------------------------
def _as_path(repo_root: Path, p: Any) -> Optional[Path]:
    if not p:
        return None
    s = str(p).strip()
    if not s:
        return None
    path = Path(s)
    if not path.is_absolute():
        path = repo_root / path
    return path


def load_smoketest_ledger(repo_root: Path, ledger_path: Path) -> Dict[str, Any]:
    if not ledger_path.exists():
        return {}
    try:
        return json.loads(ledger_path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(ledger_path.read_bytes().decode("utf-8", errors="replace"))
        except Exception:
            return {}


def normalize_ledger(repo_root: Path, raw: Dict[str, Any]) -> Dict[str, DemoLedger]:
    out: Dict[str, DemoLedger] = {}
    if not raw:
        return out

    demos = raw.get("demos")
    if isinstance(demos, dict):
        for did, info in demos.items():
            if isinstance(info, dict):
                out[str(did)] = _normalize_one_ledger(repo_root, str(did), info)
        return out

    runs = raw.get("runs")
    if isinstance(runs, list):
        for item in runs:
            if not isinstance(item, dict):
                continue
            did = str(item.get("demo_id") or item.get("id") or item.get("demo") or "").strip()
            if not did:
                continue
            out[did] = _normalize_one_ledger(repo_root, did, item)
        return out

    # fallback: treat top-level keys as ids
    for k, v in raw.items():
        if isinstance(v, dict) and re.fullmatch(r"[0-9A-Za-z]+", str(k)):
            out[str(k)] = _normalize_one_ledger(repo_root, str(k), v)
    return out


def _normalize_one_ledger(repo_root: Path, demo_id: str, info: Dict[str, Any]) -> DemoLedger:
    led = DemoLedger()

    # exit code / status
    exit_code = info.get("exit_code", info.get("returncode", info.get("rc")))
    try:
        if exit_code is not None:
            led.exit_code = int(exit_code)
    except Exception:
        led.exit_code = None

    status = str(info.get("status") or info.get("result") or "").strip().upper()
    if not status:
        if led.exit_code is None:
            status = "UNKNOWN"
        else:
            status = "PASS" if led.exit_code == 0 else "FAIL"
    led.status = status

    # duration
    dur = info.get("duration_s", info.get("duration", info.get("runtime_s")))
    try:
        if dur is not None:
            led.duration_s = float(dur)
    except Exception:
        led.duration_s = None

    # hashes
    led.code_sha256 = str(info.get("code_sha256") or info.get("code_hash") or info.get("sha256_code") or "").strip()
    led.stdout_sha256 = str(info.get("stdout_sha256") or info.get("output_sha256") or info.get("stdout_hash") or "").strip()

    # command (optional)
    cmd = info.get("command") or info.get("cmd") or info.get("argv")
    if isinstance(cmd, str):
        led.command = cmd.strip()
    elif isinstance(cmd, list):
        led.command = " ".join(str(x) for x in cmd)

    # stdout path (optional)
    led.stdout_path = _as_path(repo_root, info.get("stdout_path") or info.get("stdout_file") or info.get("log_path"))

    # artifacts
    artifacts = info.get("artifacts") or info.get("outputs") or {}
    if isinstance(artifacts, dict):
        for k, v in artifacts.items():
            sha = ""
            if isinstance(v, dict):
                sha = str(v.get("sha256") or v.get("hash") or "").strip()
            else:
                sha = str(v).strip()
            if sha:
                led.artifacts[str(k).strip()] = sha

    return led


def load_stdout_for_demo(repo_root: Path, ledger: DemoLedger, demo_id: str) -> str:
    if ledger.stdout_path and ledger.stdout_path.exists():
        return safe_read_text(ledger.stdout_path)

    base = repo_root / "artifacts" / "smoketest_all"
    if base.exists():
        did = demo_id.lower()
        patterns = [f"*{did}*stdout*.txt", f"*{did}*output*.txt", f"*{did}*log*.txt"]
        for pat in patterns:
            hits = [p for p in base.rglob(pat) if p.is_file()]
            if hits:
                hits.sort(key=lambda p: len(str(p)))
                return safe_read_text(hits[0])
    return ""


# -----------------------------
# Metric extraction from stdout (best-effort)
# -----------------------------
def _extract_first_float(text: str, patterns: Iterable[str]) -> Optional[float]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if not m:
            continue
        try:
            return float(m.group(1))
        except Exception:
            continue
    return None


def extract_metrics_from_stdout(demo_id: str, stdout_text: str) -> Dict[str, float]:
    did = demo_id.strip().upper()
    t = stdout_text or ""
    out: Dict[str, float] = {}

    if did == "55":
        v = _extract_first_float(t, [r"rp_final_predicted_fm\s+([0-9]+\.[0-9]+)"])
        if v is None:
            v = _extract_first_float(t, [r"r_p\(2-loop.*?\)\s*=\s*([0-9]+\.[0-9]+)\s*fm"])
        if v is None:
            v = _extract_first_float(t, [r"r_p.*?=\s*([0-9]+\.[0-9]+)\s*fm"])
        if v is not None:
            out["rp_pred_fm"] = v

    if did == "40":
        h0 = _extract_first_float(t, [
            r"^H0\s*:\s*([0-9]+\.[0-9]+)",
            r"H0\s*\(.*?\)\s*:\s*([0-9]+\.[0-9]+)",
            r"H0\s+([0-9]+\.[0-9]+)\s*km/s/Mpc",
        ])
        if h0 is not None:
            out["H0"] = h0

        ol = _extract_first_float(t, [
            r"^Omega_L(?:ambda)?\s*:\s*([0-9]+\.[0-9]+)",
            r"Omega_Lambda\s*=\s*([0-9]+\.[0-9]+)",
            r"Omega_L\s*=\s*([0-9]+\.[0-9]+)",
        ])
        if ol is not None:
            out["Omega_Lambda"] = ol

        om = _extract_first_float(t, [
            r"^Omega_m\s*:\s*([0-9]+\.[0-9]+)",
            r"Omega_m\s*=\s*([0-9]+\.[0-9]+)",
        ])
        if om is not None:
            out["Omega_m"] = om

    if did == "33":
        mw = _extract_first_float(t, [r"MW_dressed\s+([0-9]+\.[0-9]+)"])
        mz = _extract_first_float(t, [r"MZ_dressed\s+([0-9]+\.[0-9]+)"])
        gz = _extract_first_float(t, [r"(?:GammaZ_dressed|ΓZ_dressed).*?\s+([0-9]+\.[0-9]+)"])
        a_mz = _extract_first_float(t, [
            r"alpha\(MZ\)\^\-1\s+\(derived\)\s+([0-9]+\.[0-9]+)",
            r"alpha\(MZ\).*?([0-9]+\.[0-9]+)",
        ])
        if mw is not None:
            out["MW_dressed_GeV"] = mw
        if mz is not None:
            out["MZ_dressed_GeV"] = mz
        if gz is not None:
            out["GammaZ_dressed_GeV"] = gz
        if a_mz is not None:
            out["alpha_MZ_inv"] = a_mz

    if did == "70":
        mh = _extract_first_float(t, [r"mH_dressed_best\s+([0-9]+\.[0-9]+)"])
        if mh is None:
            mh = _extract_first_float(t, [r"Best mode:.*?mH[^0-9]*([0-9]+\.[0-9]+)"])
        if mh is not None:
            out["mH_dressed_best_GeV"] = mh

        v = _extract_first_float(t, [r"v_dressed\s*(?:=|:)?\s*([0-9]+\.[0-9]+)\s*GeV"])
        if v is not None:
            out["v_dressed_GeV"] = v

    if did == "36":
        h0 = _extract_first_float(t, [
            r"H0\s*\(km/s/Mpc\)\s*=\s*([0-9]+\.[0-9]+)",
            r"^H0\s*:\s*([0-9]+\.[0-9]+)",
        ])
        if h0 is not None:
            out["H0"] = h0

    return out


def compute_headline_numbers(repo_root: Path, records: List[DemoRecord]) -> Dict[str, Any]:
    by_id = {r.meta.demo_id: r for r in records}
    metrics: Dict[str, Any] = {}

    def pull(demo_id: str) -> Dict[str, float]:
        rec = by_id.get(demo_id)
        if not rec:
            return {}
        txt = load_stdout_for_demo(repo_root, rec.ledger, demo_id)
        if not txt:
            return {}
        return extract_metrics_from_stdout(demo_id, txt)

    metrics.update(pull("55"))
    metrics.update(pull("40"))
    metrics.update(pull("33"))
    metrics.update(pull("70"))
    if "H0" not in metrics:
        metrics.update(pull("36"))

    # Structural constants (stable defaults)
    metrics.setdefault("alpha0_inv", 137.0)
    metrics.setdefault("sin2W_structural", 7.0 / 30.0)
    metrics.setdefault("alpha_s_structural", 2.0 / 17.0)
    return metrics


# -----------------------------
# ReportLab styles and helpers
# -----------------------------
def make_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: Dict[str, ParagraphStyle] = {}

    styles["Title"] = ParagraphStyle(
        "Title", parent=base["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, spaceAfter=12
    )
    styles["Subtitle"] = ParagraphStyle(
        "Subtitle", parent=base["Normal"], fontName="Helvetica", fontSize=12, leading=14, textColor=colors.grey, spaceAfter=18
    )
    styles["Quote"] = ParagraphStyle(
        "Quote", parent=base["Normal"], fontName="Helvetica-Oblique", fontSize=11, leading=14, leftIndent=18, rightIndent=18, spaceAfter=18
    )
    styles["H1"] = ParagraphStyle(
        "H1", parent=base["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=19, spaceBefore=14, spaceAfter=8
    )
    styles["H2"] = ParagraphStyle(
        "H2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=13, leading=16, spaceBefore=10, spaceAfter=6
    )
    styles["Body"] = ParagraphStyle(
        "Body", parent=base["BodyText"], fontName="Helvetica", fontSize=10, leading=13, spaceAfter=8
    )
    styles["BodySmall"] = ParagraphStyle(
        "BodySmall", parent=base["BodyText"], fontName="Helvetica", fontSize=9, leading=12, spaceAfter=6
    )
    styles["Code"] = ParagraphStyle(
        "Code",
        parent=base["BodyText"],
        fontName="Courier",
        fontSize=9,
        leading=11,
        backColor=colors.whitesmoke,
        borderPadding=6,
        leftIndent=6,
        rightIndent=6,
        spaceAfter=10,
    )
    styles["MonoSmall"] = ParagraphStyle(
        "MonoSmall", parent=base["BodyText"], fontName="Courier", fontSize=8, leading=10, spaceAfter=6
    )
    return styles


def img_scaled(path: Path, max_w: float, max_h: float) -> Optional[Image]:
    if not path.exists():
        return None
    try:
        img = Image(str(path))
        iw, ih = img.imageWidth, img.imageHeight
        if iw <= 0 or ih <= 0:
            return None
        scale = min(max_w / iw, max_h / ih)
        img.drawWidth = iw * scale
        img.drawHeight = ih * scale
        return img
    except Exception:
        return None


def header_footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 9)
    canvas.setFillColor(colors.grey)
    canvas.drawString(0.75 * inch, 0.65 * inch, f"{TITLE}  |  {VERSION}")
    canvas.drawRightString(7.75 * inch, 0.65 * inch, f"Page {doc.page}")
    canvas.restoreState()


def spacer(h_in: float = 0.15) -> Spacer:
    return Spacer(1, h_in * inch)


# -----------------------------
# Curated claims (flagships)
# -----------------------------
FLAGSHIP_ORDER: List[str] = ["55", "33", "40", "70", "36", "66B", "67", "71"]

DEMO_CLAIMS: Dict[str, Dict[str, str]] = {
    "33": {
        "title": "First-Principles Standard Model (SM-28 closure)",
        "claim": "From a discrete survivor triple (137,107,103) and a closed-form one-action scale, derive a self-consistent SM parameter set and a dressed (Authority v1) prediction layer; PDG is not used upstream.",
        "falsifier": "If upstream selection depends on PDG constants (without an explicit overlay flag), or if invariance checks fail, the claim fails.",
    },
    "55": {
        "title": "Proton Charge Radius (High-Precision Tooth)",
        "claim": "A 1D structural route produces a proton charge radius prediction in the 0.84 fm band without importing the measurement upstream.",
        "falsifier": "If any step uses experimental rp as an input (not merely overlay), or if the prediction is unstable under the declared ablations, the claim fails.",
    },
    "40": {
        "title": "Universe From Zero (Substrate Root)",
        "claim": "Starting from integer geometry, recover a coupled bundle of cosmological numbers (including H0 and Omega_Lambda) with no fitted continuous parameters.",
        "falsifier": "If outputs vary arbitrarily under small rule perturbations or require hidden calibration, the claim fails.",
    },
    "70": {
        "title": "Higgs Master Flagship",
        "claim": "From an integer mode + closed-form dressing, predict a Higgs-scale proxy (with explicit assumptions) and generate reproducible artifacts.",
        "falsifier": "If the selected mode is post-chosen to match PDG, or the pipeline depends on hidden hand-tuning, the claim fails.",
    },
    "36": {
        "title": "Big Bang / H0 Master Flagship",
        "claim": "A closed-form integer substrate pipeline generates an LCDM-like parameter bundle and a structural H0 without regression to Planck.",
        "falsifier": "If parameters require tuning to match Planck, or the pipeline changes under ablation, the claim fails.",
    },
    "66B": {
        "title": "Quantum Gravity v2 (Weak/Strong Field Master)",
        "claim": "A discrete screening law yields a QG testbed spanning weak/strong field regimes with explicit invariance certificates.",
        "falsifier": "If the invariants collapse when constraints are removed, or if results depend on unstated constants, the claim fails.",
    },
    "67": {
        "title": "Navier-Stokes Master Flagship",
        "claim": "Demonstrates a deterministic operator-calculus construction for NS-like behavior; outputs are audited by hashes and certificates.",
        "falsifier": "If the demo cannot be reproduced from the provided scripts and environment, or if constraints are silently tuned, the claim fails.",
    },
    "71": {
        "title": "One Action / Noether Master Flagship",
        "claim": "Derives conserved-quantity structure from a single action-like invariant in the integer substrate framework; formulated as an audit trail.",
        "falsifier": "If conservation statements depend on external calibration or cannot be reproduced from the code path, the claim fails.",
    },
}

CERT_METRICS: Dict[str, List[Tuple[str, str, str]]] = {
    "55": [("rp_pred_fm", "Proton radius", "fm")],
    "40": [("H0", "H0", "km/s/Mpc"), ("Omega_Lambda", "Omega_Lambda", ""), ("Omega_m", "Omega_m", "")],
    "70": [("mH_dressed_best_GeV", "mH proxy", "GeV"), ("v_dressed_GeV", "v_dressed", "GeV")],
    "33": [("alpha_MZ_inv", "alpha(MZ)^-1", ""), ("MW_dressed_GeV", "MW_dressed", "GeV"), ("MZ_dressed_GeV", "MZ_dressed", "GeV"), ("GammaZ_dressed_GeV", "GammaZ_dressed", "GeV")],
    "36": [("H0", "H0", "km/s/Mpc")],
}


# -----------------------------
# Build demo records (map + ledger)
# -----------------------------
def build_demo_records(repo_root: Path, demo_map: Dict[str, Dict[str, str]], ledger_by_id: Dict[str, DemoLedger]) -> List[DemoRecord]:
    records: Dict[str, DemoRecord] = {}

    # from demo_map
    for demo_id, m in demo_map.items():
        cat = canonical_category(m.get("category", ""))
        slug = m.get("slug", "")
        demo_dir, script, readme = resolve_demo_dir(repo_root, demo_id, cat, slug)

        title = DEMO_CLAIMS.get(demo_id, {}).get("title") or title_from_slug(slug) or f"Demo {demo_id}"

        summary = ""
        falsify = ""
        if readme and readme.exists():
            txt = safe_read_text(readme)
            summary = extract_one_paragraph_summary(txt)
            falsify = extract_falsifier_hint(txt)

        if demo_id in DEMO_CLAIMS:
            falsify = DEMO_CLAIMS[demo_id].get("falsifier", falsify)

        # baseline command (safe)
        run_cmd_smoke = ""
        run_cmd_cert = ""
        if script and script.exists():
            rel = script.relative_to(repo_root)
            run_cmd_smoke = f'python "{rel}"'
            if script_supports_flag(script, "--cert"):
                run_cmd_cert = f'python "{rel}" --cert'
        else:
            rel = f"demos/{cat}/demo-{normalize_demo_id_for_dir(demo_id)}-{slug}/demo.py"
            run_cmd_smoke = f'python "{rel}"'

        run_cmd = run_cmd_cert or run_cmd_smoke

        led = (
            ledger_by_id.get(demo_id)
            or ledger_by_id.get(demo_id.upper())
            or ledger_by_id.get(demo_id.lower())
            or DemoLedger()
        )

        # If ledger says how it ran, trust that for the smoke command
        if led.command:
            run_cmd_smoke = led.command
            run_cmd = run_cmd_cert or run_cmd_smoke

        meta = DemoMeta(
            demo_id=demo_id,
            category=cat,
            slug=slug,
            demo_dir=demo_dir,
            script_path=script,
            readme_path=readme,
            title=title,
            summary=summary,
            falsify=falsify,
            run_cmd=run_cmd,
            run_cmd_smoke=run_cmd_smoke,
            run_cmd_cert=run_cmd_cert,
        )
        records[demo_id] = DemoRecord(meta=meta, ledger=led)

    # ledger-only demos
    for demo_id, led in ledger_by_id.items():
        if demo_id in records:
            continue
        meta = DemoMeta(
            demo_id=demo_id,
            title=DEMO_CLAIMS.get(demo_id, {}).get("title", f"Demo {demo_id}"),
        )
        meta.run_cmd_smoke = led.command or f'python demos/**/demo-{normalize_demo_id_for_dir(demo_id)}-*/demo.py'
        meta.run_cmd = meta.run_cmd_smoke
        records[demo_id] = DemoRecord(meta=meta, ledger=led)

    def sort_key(did: str) -> Tuple[int, str]:
        m = re.match(r"^(\d+)", did)
        n = int(m.group(1)) if m else 9999
        return (n, did)

    return [records[k] for k in sorted(records.keys(), key=sort_key)]


# -----------------------------
# Tables / sections
# -----------------------------
def build_executive_table(styles: Dict[str, ParagraphStyle], headline: Dict[str, Any]) -> Table:
    # Declared reference values (overlay only)
    ref = {
        "alpha_inv": 137.035999084,
        "sin2W": 0.2312,
        "alpha_s": 0.1179,
        "H0": 67.36,
        "Omega_Lambda": 0.6889,
        "rp_fm": 0.84075,
        "MW": 80.379,
        "MZ": 91.1876,
        "GammaZ": 2.4952,
        "mH": 125.25,
    }

    def rel_err(pred: Optional[float], refv: float) -> str:
        if pred is None:
            return "n/a"
        try:
            return f"{(pred - refv) / refv:+.3e}"
        except Exception:
            return "n/a"

    pred_alpha_inv = float(headline.get("alpha0_inv") or 137.0)
    pred_sin2w = float(headline.get("sin2W_structural") or (7.0 / 30.0))
    pred_alpha_s = float(headline.get("alpha_s_structural") or (2.0 / 17.0))

    def fget(k: str) -> Optional[float]:
        v = headline.get(k)
        return float(v) if isinstance(v, (int, float)) else None

    pred_h0 = fget("H0")
    pred_ol = fget("Omega_Lambda")
    pred_rp = fget("rp_pred_fm")
    pred_mw = fget("MW_dressed_GeV")
    pred_mz = fget("MZ_dressed_GeV")
    pred_gz = fget("GammaZ_dressed_GeV")
    pred_mh = fget("mH_dressed_best_GeV")

    rows = [
        ["Quantity", "Predicted (GUM)", "Reference (overlay)", "Rel. error", "Primary demo"],
        ["alpha^-1 (structural)", f"{pred_alpha_inv:.12f}", f"{ref['alpha_inv']:.12f}", rel_err(pred_alpha_inv, ref["alpha_inv"]), "33"],
        ["sin^2(theta_W) (structural)", f"{pred_sin2w:.12f}", f"{ref['sin2W']:.4f}", rel_err(pred_sin2w, ref["sin2W"]), "33"],
        ["alpha_s (structural)", f"{pred_alpha_s:.12f}", f"{ref['alpha_s']:.4f}", rel_err(pred_alpha_s, ref["alpha_s"]), "33"],
        ["H0 (km/s/Mpc)", f"{pred_h0:.6f}" if pred_h0 is not None else "n/a", f"{ref['H0']:.2f}", rel_err(pred_h0, ref["H0"]), "40/36"],
        ["Omega_Lambda", f"{pred_ol:.6f}" if pred_ol is not None else "n/a", f"{ref['Omega_Lambda']:.4f}", rel_err(pred_ol, ref["Omega_Lambda"]), "40"],
        ["rp (fm)", f"{pred_rp:.10f}" if pred_rp is not None else "n/a", f"{ref['rp_fm']:.5f}", rel_err(pred_rp, ref["rp_fm"]), "55"],
        ["MW (Authority v1)", f"{pred_mw:.6f}" if pred_mw is not None else "n/a", f"{ref['MW']:.3f}", rel_err(pred_mw, ref["MW"]), "33"],
        ["MZ (Authority v1)", f"{pred_mz:.6f}" if pred_mz is not None else "n/a", f"{ref['MZ']:.4f}", rel_err(pred_mz, ref["MZ"]), "33"],
        ["GammaZ (Authority v1)", f"{pred_gz:.6f}" if pred_gz is not None else "n/a", f"{ref['GammaZ']:.4f}", rel_err(pred_gz, ref["GammaZ"]), "33"],
        ["mH proxy (Authority v1)", f"{pred_mh:.6f}" if pred_mh is not None else "n/a", f"{ref['mH']:.2f}", rel_err(pred_mh, ref["mH"]), "70"],
    ]

    tbl = Table(rows, colWidths=[1.55 * inch, 1.7 * inch, 1.65 * inch, 1.05 * inch, 1.3 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-2, -1), "RIGHT"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    return tbl


def build_dashboard_table(styles: Dict[str, ParagraphStyle], headline: Dict[str, Any]) -> Table:
    """
    Compact "raw vs dressed" dashboard (v28 flavor, v29 audit tone).
    """
    def fmt(v: Any, nd: int = 6) -> str:
        if v is None:
            return "n/a"
        if isinstance(v, (int, float)):
            return f"{float(v):.{nd}f}"
        return "n/a"

    rows = [
        ["Layer", "Quantity", "Value"],
        ["Structural", "alpha0^-1", fmt(headline.get("alpha0_inv"), 12)],
        ["Structural", "sin^2(theta_W)", fmt(headline.get("sin2W_structural"), 12)],
        ["Structural", "alpha_s", fmt(headline.get("alpha_s_structural"), 12)],
        ["Structural", "H0 (km/s/Mpc)", fmt(headline.get("H0"), 6)],
        ["Structural", "Omega_Lambda", fmt(headline.get("Omega_Lambda"), 6)],
        ["Structural", "rp (fm)", fmt(headline.get("rp_pred_fm"), 10)],
        ["Dressed / proxy", "MW (GeV)", fmt(headline.get("MW_dressed_GeV"), 6)],
        ["Dressed / proxy", "MZ (GeV)", fmt(headline.get("MZ_dressed_GeV"), 6)],
        ["Dressed / proxy", "GammaZ (GeV)", fmt(headline.get("GammaZ_dressed_GeV"), 6)],
        ["Dressed / proxy", "mH proxy (GeV)", fmt(headline.get("mH_dressed_best_GeV"), 6)],
        ["Dressed / proxy", "v_dressed (GeV)", fmt(headline.get("v_dressed_GeV"), 6)],
    ]
    tbl = Table(rows, colWidths=[1.2 * inch, 2.7 * inch, 3.3 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("ALIGN", (2, 1), (2, -1), "RIGHT"),
            ]
        )
    )
    return tbl


def build_falsification_matrix(styles: Dict[str, ParagraphStyle], records: List[DemoRecord]) -> Table:
    rows: List[List[str]] = [["Demo", "Domain", "Status", "Run", "Code SHA", "Stdout SHA", "Runtime"]]
    statuses: List[str] = []
    for r in records:
        statuses.append((r.ledger.status or "UNKNOWN").upper())
        rows.append(
            [
                r.meta.demo_id,
                r.meta.category or "-",
                r.ledger.status or "UNKNOWN",
                r.meta.run_cmd_smoke or r.meta.run_cmd or "-",
                shorten_sha(r.ledger.code_sha256, 10) or "-",
                shorten_sha(r.ledger.stdout_sha256, 10) or "-",
                human_seconds(r.ledger.duration_s),
            ]
        )

    tbl = Table(rows, colWidths=[0.55 * inch, 0.9 * inch, 0.65 * inch, 3.05 * inch, 0.8 * inch, 0.8 * inch, 0.65 * inch])

    green = colors.Color(0.88, 0.97, 0.88)
    red = colors.Color(0.98, 0.88, 0.88)

    style_cmds: List[Tuple] = [
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
        ("FONT", (3, 1), (3, -1), "Courier", 7),
    ]

    # Status coloring
    for i, st in enumerate(statuses, start=1):
        if st == "PASS":
            style_cmds.append(("BACKGROUND", (2, i), (2, i), green))
        elif st == "FAIL":
            style_cmds.append(("BACKGROUND", (2, i), (2, i), red))

    tbl.setStyle(TableStyle(style_cmds))
    return tbl


def add_validation_plates(story: List[Any], styles: Dict[str, ParagraphStyle], repo_root: Path) -> None:
    assets = repo_root / "gum" / "assets"

    story.append(Paragraph("Validation Plates (Visual Proof)", styles["H1"]))
    story.append(
        Paragraph(
            "These plates are preserved from v28 and carried into v29. They are visual witnesses of the integer substrate: "
            "identity pillars, echo patterns, and the declared smoothing transform.",
            styles["Body"],
        )
    )

    # Identity/Echo grid
    imgs = [assets / "Identity9.png", assets / "Identity10.png", assets / "Echo6.png", assets / "Echo10.png"]
    grid: List[List[Any]] = []
    row: List[Any] = []
    for p in imgs:
        im = img_scaled(p, max_w=3.35 * inch, max_h=2.5 * inch)
        row.append(im if im else Paragraph(f"(missing: {p.name})", styles["BodySmall"]))
        if len(row) == 2:
            grid.append(row)
            row = []
    if row:
        grid.append(row)

    t = Table(grid, colWidths=[3.5 * inch, 3.5 * inch])
    t.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(Paragraph("DRPT identity pillars and echo patterns", styles["H2"]))
    story.append(t)
    story.append(spacer(0.15))

    for p, cap in [
        (assets / "DRPTCity137_v2.png", "DRPT City (137)"),
        (assets / "DRPTSurvivor137_v2.png", "DRPT Survivor (137)"),
    ]:
        im = img_scaled(p, max_w=7.2 * inch, max_h=3.25 * inch)
        story.append(Paragraph(cap, styles["H2"]))
        story.append(im if im else Paragraph(f"(missing: {p.name})", styles["BodySmall"]))
        story.append(spacer(0.12))

    # Fejer pair
    story.append(Paragraph("Fejer Smoothing (Spectral Witness)", styles["H2"]))
    fejer_left = assets / "FejerSmoothignLeft.png"  # legacy filename
    fejer_right = assets / "FejerSmoothingRight.png"
    pair = []
    for p in [fejer_left, fejer_right]:
        im = img_scaled(p, max_w=3.35 * inch, max_h=2.8 * inch)
        pair.append(im if im else Paragraph(f"(missing: {p.name})", styles["BodySmall"]))
    t2 = Table([pair], colWidths=[3.5 * inch, 3.5 * inch])
    t2.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER")]))
    story.append(t2)
    story.append(PageBreak())

    # BB36
    story.append(Paragraph("BB36: Structural Big Bang Plate", styles["H1"]))
    bb = assets / "BB36_big_bang.png"
    imbb = img_scaled(bb, max_w=7.2 * inch, max_h=5.0 * inch)
    story.append(imbb if imbb else Paragraph("(missing: BB36_big_bang.png)", styles["BodySmall"]))
    story.append(spacer(0.15))

    # CMB overlay (conditional)
    story.append(Paragraph("CMB TT Overlay (Conditional)", styles["H2"]))
    candidates = [
        assets / "TT_spectrum_Planck2018_vs_GUM.png",
        assets / "TT_spectrum_Planck_vs_GUM.png",
        repo_root / "TT_spectrum_Planck2018_vs_GUM.png",
        repo_root / "TT_spectrum_Planck_vs_GUM.png",
    ]
    tt = next((p for p in candidates if p.exists()), None)
    if tt:
        imtt = img_scaled(tt, max_w=7.2 * inch, max_h=4.5 * inch)
        story.append(imtt if imtt else Paragraph(f"(could not load: {tt.name})", styles["BodySmall"]))
    else:
        story.append(
            Paragraph(
                "CMB overlay image not found (expected TT_spectrum_Planck2018_vs_GUM.png). "
                "This is a conditional plate: reproducible when the Planck/CAMB bundle is present.",
                styles["BodySmall"],
            )
        )
    story.append(PageBreak())


def build_key_metrics_table(styles: Dict[str, ParagraphStyle], rows: List[Tuple[str, str, str]]) -> Table:
    data = [["Metric", "Value", "Unit"]]
    data.extend([[a, b, c] for a, b, c in rows])
    tbl = Table(data, colWidths=[2.2 * inch, 2.6 * inch, 1.0 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 9),
                ("ALIGN", (1, 1), (1, -1), "RIGHT"),
            ]
        )
    )
    return tbl


def add_flagship_certificates(story: List[Any], styles: Dict[str, ParagraphStyle], repo_root: Path, records: List[DemoRecord]) -> None:
    by_id = {r.meta.demo_id: r for r in records}

    story.append(Paragraph("Flagship Certificates", styles["H1"]))
    story.append(
        Paragraph(
            "Each flagship is presented as an audit certificate: Claim -> Falsifier -> Run command -> "
            "Key metrics (when provable from artifacts) -> Reproducibility hashes. These pages are intended to be cited directly.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    for did in FLAGSHIP_ORDER:
        rec = by_id.get(did)
        if not rec:
            continue

        title = DEMO_CLAIMS.get(did, {}).get("title") or rec.meta.title or f"Demo {did}"
        claim = DEMO_CLAIMS.get(did, {}).get("claim") or rec.meta.summary or ""
        falsifier = DEMO_CLAIMS.get(did, {}).get("falsifier") or rec.meta.falsify or ""

        story.append(Paragraph(f"Certificate: Demo {did} — {escape_xml(title)}", styles["H1"]))

        if claim:
            story.append(Paragraph(f"<b>Claim:</b> {escape_xml(claim)}", styles["Body"]))
        if falsifier:
            story.append(Paragraph(f"<b>Falsifier:</b> {escape_xml(falsifier)}", styles["Body"]))

        story.append(Paragraph("<b>Run (baseline):</b>", styles["Body"]))
        story.append(Preformatted(rec.meta.run_cmd_smoke or rec.meta.run_cmd, styles["Code"]))
        if rec.meta.run_cmd_cert:
            story.append(Paragraph("<b>Run (certificate mode):</b>", styles["BodySmall"]))
            story.append(Preformatted(rec.meta.run_cmd_cert, styles["Code"]))

        # Key metrics (from stdout, best-effort)
        stdout_txt = load_stdout_for_demo(repo_root, rec.ledger, did)
        metrics = extract_metrics_from_stdout(did, stdout_txt) if stdout_txt else {}
        metric_rows: List[Tuple[str, str, str]] = []
        for key, label, unit in CERT_METRICS.get(did, []):
            if key not in metrics:
                continue
            v = metrics[key]
            if unit in {"GeV", "km/s/Mpc"}:
                metric_rows.append((label, f"{v:.6f}", unit))
            elif unit == "fm":
                metric_rows.append((label, f"{v:.10f}", unit))
            else:
                metric_rows.append((label, f"{v:.6f}", unit))
        if metric_rows:
            story.append(Paragraph("Key metrics (provable from artifacts)", styles["H2"]))
            story.append(build_key_metrics_table(styles, metric_rows))

        # Ledger info
        led = rec.ledger
        rows = [
            ["status", led.status],
            ["exit_code", str(led.exit_code) if led.exit_code is not None else "n/a"],
            ["runtime", human_seconds(led.duration_s)],
            ["code_sha256", led.code_sha256 or "n/a"],
            ["stdout_sha256", led.stdout_sha256 or "n/a"],
        ]
        tbl = Table(rows, colWidths=[1.3 * inch, 5.9 * inch])
        tbl.setStyle(
            TableStyle(
                [
                    ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                    ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ]
            )
        )
        story.append(Paragraph("Reproducibility (from smoketest ledger)", styles["H2"]))
        story.append(tbl)

        # Artifact hashes
        if led.artifacts:
            art_rows = [["artifact", "sha256"]]
            for k, v in sorted(led.artifacts.items()):
                art_rows.append([k, v])
            at = Table(art_rows, colWidths=[3.2 * inch, 4.0 * inch])
            at.setStyle(
                TableStyle(
                    [
                        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("FONT", (0, 1), (-1, -1), "Courier", 7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                )
            )
            story.append(Paragraph("Artifacts", styles["H2"]))
            story.append(at)

        story.append(PageBreak())


# -----------------------------
# PDF build
# -----------------------------
def build_report_pdf(
    repo_root: Path,
    out_pdf: Path,
    records: List[DemoRecord],
    ledger_path: Optional[Path],
    headline_metrics: Dict[str, Any],
    manifest_sha: str,
    numeric_sha: str,
) -> None:
    styles = make_styles()
    story: List[Any] = []

    # Title page
    story.append(Paragraph(TITLE, styles["Title"]))
    story.append(Paragraph(SUBTITLE, styles["Subtitle"]))
    story.append(Paragraph(QUOTE, styles["Quote"]))

    git_rev = try_git_rev(repo_root)
    env_lines = [
        f"Generated (UTC): {now_utc_stamp()}",
        f"Repo: {git_rev or 'n/a'}",
        f"Python: {platform.python_version()}",
        f"Platform: {platform.platform()}",
        f"Ledger: {ledger_path.relative_to(repo_root) if ledger_path and ledger_path.exists() else 'n/a'}",
        f"Manifest SHA256: {manifest_sha}",
        f"Numeric SHA256: {numeric_sha}",
    ]
    story.append(Preformatted("\n".join(env_lines), styles["MonoSmall"]))
    story.append(spacer(0.25))
    story.append(
        Paragraph(
            "<b>Framing:</b> This document is an audit report. It does not claim new empirical measurement. "
            "It presents deterministic computations whose outputs can be verified, falsified, and hashed.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # Quickstart
    story.append(Paragraph("Falsification Quickstart", styles["H1"]))
    story.append(
        Paragraph(
            "All commands below are copy-paste from the repo root. If a demo supports <font face='Courier'>--cert</font>, "
            "we show it as a second (optional) line.",
            styles["Body"],
        )
    )
    by_id = {r.meta.demo_id: r for r in records}
    quick_lines: List[str] = []
    for did in FLAGSHIP_ORDER:
        rec = by_id.get(did)
        if not rec:
            continue
        title = DEMO_CLAIMS.get(did, {}).get("title") or rec.meta.title or f"Demo {did}"
        quick_lines.append(f"# Demo {did}: {title}")
        quick_lines.append(rec.meta.run_cmd_smoke or rec.meta.run_cmd)
        if rec.meta.run_cmd_cert:
            quick_lines.append(rec.meta.run_cmd_cert)
        quick_lines.append("")
    story.append(Preformatted("\n".join(quick_lines).rstrip(), styles["Code"]))
    story.append(PageBreak())

    # Executive summary
    story.append(Paragraph("Executive Technical Summary (Headline Numbers)", styles["H1"]))
    story.append(
        Paragraph(
            "This table shows only values we can extract from the current artifact/ledger environment. "
            "Missing values are shown as n/a (not guessed).",
            styles["Body"],
        )
    )
    story.append(build_executive_table(styles, headline_metrics))
    story.append(
        Paragraph(
            "Note: 'Reference' values are overlay-only and may be updated; the Numeric SHA256 commits to the predicted values in this build.",
            styles["BodySmall"],
        )
    )
    story.append(spacer(0.15))

    story.append(Paragraph("Unified Dashboard (Structural vs Dressed)", styles["H2"]))
    story.append(
        Paragraph(
            "This dashboard mirrors the v28 'raw vs dressed' reporting style, but in v29 the emphasis is auditability: "
            "every displayed value must be provable from artifacts or declared as a stable structural constant.",
            styles["BodySmall"],
        )
    )
    story.append(build_dashboard_table(styles, headline_metrics))
    story.append(PageBreak())

    # Bridge narrative
    story.append(Paragraph("Bridge: From Integer Geometry to Physical Constants", styles["H1"]))
    bridge = (
        "The report begins with DRPT geometry because it is the lowest-level visual witness: the integer substrate "
        "exhibits stable identity pillars and echo patterns across bases. The working hypothesis is that physical "
        "constants arise as eigenvalues of these discrete structures, analogously to how boundary conditions determine "
        "resonant modes in a physical system. A declared sequence of transforms (selection rules, smoothing, and dressing layers) "
        "maps these substrate eigenvalues into comparison space (Authority v1) where PDG/Planck overlays can be evaluated."
    )
    story.append(Paragraph(bridge, styles["Body"]))
    story.append(
        Paragraph(
            "Audit instruction: the most important question is not 'do the numbers look close' but "
            "'is every transform declared, reproducible, and falsifiable'.",
            styles["Body"],
        )
    )
    story.append(PageBreak())

    # Plates
    add_validation_plates(story, styles, repo_root)

    # Matrix
    story.append(Paragraph("Falsification Matrix (All Demos)", styles["H1"]))
    story.append(
        Paragraph(
            "Each row is a runnable test plus hashes. PASS/FAIL is ledger-derived; hashes are paper-citable.",
            styles["Body"],
        )
    )
    # Summary counts
    counts: Dict[str, int] = {}
    for r in records:
        st = (r.ledger.status or "UNKNOWN").upper()
        counts[st] = counts.get(st, 0) + 1
    count_line = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    story.append(Paragraph(f"<b>Status counts:</b> {escape_xml(count_line)}", styles["BodySmall"]))
    story.append(build_falsification_matrix(styles, records))
    story.append(PageBreak())

    # Certificates
    add_flagship_certificates(story, styles, repo_root, records)

    # Appendix index
    story.append(Paragraph("Appendix: Demo Index (Why It Matters)", styles["H1"]))
    story.append(Paragraph("A compact map of the demo suite. Flagships are expanded in the certificate section.", styles["Body"]))

    index_rows = [["Demo", "Domain", "Why it matters (1 line)", "Falsifier (1 line)"]]
    for r in records:
        did = r.meta.demo_id
        dom = r.meta.category or "-"
        claim_line = DEMO_CLAIMS.get(did, {}).get("claim") or r.meta.summary or ""
        fals_line = DEMO_CLAIMS.get(did, {}).get("falsifier") or r.meta.falsify or ""
        claim_line = (claim_line or "").strip()
        fals_line = (fals_line or "").strip()
        if len(claim_line) > 140:
            claim_line = claim_line[:137] + "..."
        if len(fals_line) > 120:
            fals_line = fals_line[:117] + "..."
        index_rows.append([did, dom, claim_line or "-", fals_line or "-"])

    idx = Table(index_rows, colWidths=[0.55 * inch, 1.05 * inch, 3.2 * inch, 2.8 * inch])
    idx.setStyle(
        TableStyle(
            [
                ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 9),
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    story.append(idx)
    story.append(spacer(0.2))
    story.append(Paragraph("End of Report", styles["H2"]))
    story.append(
        Paragraph(
            "Audit tip: start with Demo 55 or Demo 33, run the baseline command, verify hashes, then expand outward through the matrix.",
            styles["Body"],
        )
    )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(out_pdf),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.85 * inch,
        bottomMargin=0.85 * inch,
        title=TITLE,
        author="Marithmetics",
    )
    doc.build(story, onFirstPage=header_footer, onLaterPages=header_footer)


# -----------------------------
# Main
# -----------------------------
def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GUM Report v29 (System Audit PDF).")
    parser.add_argument("--ledger", type=str, default="artifacts/smoketest_all/summary.json", help="Path to smoketest ledger JSON.")
    parser.add_argument("--outdir", type=str, default="gum/GUM_Report", help="Output directory for the PDF.")
    parser.add_argument("--outfile", type=str, default="", help="Optional explicit output PDF filename.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ledger_path = Path(args.ledger)
    if not ledger_path.is_absolute():
        ledger_path = repo_root / ledger_path

    raw = load_smoketest_ledger(repo_root, ledger_path)
    ledger_by_id = normalize_ledger(repo_root, raw)

    demo_map = parse_demo_map_yaml(repo_root / "tools" / "demo_map.yaml")
    records = build_demo_records(repo_root, demo_map, ledger_by_id)

    headline = compute_headline_numbers(repo_root, records)

    manifest_obj = {
        "version": VERSION,
        "title": TITLE,
        "git_rev": try_git_rev(repo_root),
        "ledger": str(ledger_path.relative_to(repo_root)) if ledger_path.exists() else "",
        "demos": [
            {"id": r.meta.demo_id, "category": r.meta.category, "slug": r.meta.slug, "run": r.meta.run_cmd_smoke}
            for r in records
        ],
    }
    manifest_sha = sha256_json(manifest_obj)
    numeric_sha = sha256_json(headline)

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = repo_root / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    if args.outfile:
        out_pdf = Path(args.outfile)
        if not out_pdf.is_absolute():
            out_pdf = outdir / out_pdf
    else:
        out_pdf = outdir / f"GUM_Report_v29_{now_utc_stamp()}.pdf"

    build_report_pdf(
        repo_root=repo_root,
        out_pdf=out_pdf,
        records=records,
        ledger_path=ledger_path if ledger_path.exists() else None,
        headline_metrics=headline,
        manifest_sha=manifest_sha,
        numeric_sha=numeric_sha,
    )

    print(f"[v29] Wrote: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
