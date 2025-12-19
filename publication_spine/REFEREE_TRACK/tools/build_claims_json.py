cat > publication_spine/REFEREE_TRACK/tools/build_claims_json.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Referee Track: CLAIMS.json builder (stdlib-only)

Goal:
- Stable Claim IDs with HARD/EVIDENCE tiers.
- Each HARD claim includes non-null expected values.
- Expected values are extracted from AoR stdout where possible.
- If extraction fails for any expected field, we fall back to canonical A2 values
  (and print warnings listing what fell back).

Usage:
  python publication_spine/REFEREE_TRACK/tools/build_claims_json.py \
    --commit <SHA> \
    --aor-dir audits/results/<AOR_FOLDER> \
    --bundle-sha256 <BUNDLE_SHA>

Writes:
  publication_spine/REFEREE_TRACK/CLAIMS.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------
# Canonical A2 fallback values
# ----------------------------
CANON = {
    "wU": 137, "s2": 107, "s3": 103,
    "q2": 30, "q3": 17, "v2_wU_minus_1": 3,
    "alpha_frac": "1/137", "alpha": float(Fraction(1, 137)),
    "alpha_s_frac": "2/17", "alpha_s": float(Fraction(2, 17)),
    "sin2_frac": "7/30", "sin2": float(Fraction(7, 30)),
    "du_frac": "8/3", "du": float(Fraction(8, 3)),
    "lu_frac": "13/8", "lu": float(Fraction(13, 8)),
    "delta_iso_frac": "1/24", "delta_iso": float(Fraction(1, 24)),
    "palette_best": ["0","4/3","7/4","8/3","4","11/3","13/8","21/8","9/2"],
    "q_tuple": [17, 13, 17],
    "RU": ["1", "5"], "R2": ["3"], "R3": ["1"],
    "Omega_b": 0.04498189308849704,
    "Omega_c": 0.24297831265548442,
    "Omega_L": 0.7118999678358255,
    "Omega_r": 8.42249728384458e-05,
    "Omega_tot": 0.9999443985526454,
    "H0": 70.44939596437767,
    "As": 2.099094113200873e-09,
    "ns": 0.9647460711549101,
    "tau": 0.05397948497016616,
    "d21": 7.499740433804768e-05,
    "d31": 0.002499999351253991,
    "sumv": 0.05999487172609798,
    "hierarchy": 33.3358,  # printed approx; exact may vary by rounding in stdout
    "tt_rms_rel_diff": 3.0445e-02,
    "tt_max_rel_diff": 7.3055e-02,
}


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_blob_url(repo_url: str, commit: str, rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"{repo_url}/blob/{commit}/{rel_path}"


def _repo_tree_url(repo_url: str, commit: str, rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"{repo_url}/tree/{commit}/{rel_path}"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _first_match(text: str, patterns: List[str]) -> Optional[re.Match]:
    for pat in patterns:
        m = re.search(pat, text, flags=re.MULTILINE)
        if m:
            return m
    return None


def _find_int(text: str, patterns: List[str]) -> Optional[int]:
    m = _first_match(text, patterns)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_float(text: str, patterns: List[str]) -> Optional[float]:
    m = _first_match(text, patterns)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _find_fraction(text: str, patterns: List[str]) -> Optional[str]:
    m = _first_match(text, patterns)
    if not m:
        return None
    return m.group(1).strip()


def _find_list(text: str, patterns: List[str]) -> Optional[List[str]]:
    m = _first_match(text, patterns)
    if not m:
        return None
    raw = m.group(1).strip()
    toks = re.split(r"\s*,\s*", raw)
    out: List[str] = []
    for t in toks:
        t = t.strip().strip("[](){}").strip().strip("'").strip('"')
        if t:
            out.append(t)
    return out if out else None


def _v2(n: int) -> int:
    c = 0
    while n > 0 and n % 2 == 0:
        n //= 2
        c += 1
    return c


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/public-arch/Marithmetics")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--aor-dir", required=True)
    ap.add_argument("--bundle-sha256", required=True)
    ap.add_argument("--out", default="publication_spine/REFEREE_TRACK/CLAIMS.json")
    args = ap.parse_args()

    repo_url = args.repo_url.rstrip("/")
    commit = args.commit.strip()
    aor_dir = Path(args.aor_dir)
    out_path = Path(args.out)

    # AoR stdout files (expected paths)
    scfp_txt = _read_text(aor_dir / "stdout_substrate_scfp_integer_selector_v1.py.txt")
    a2_txt = _read_text(aor_dir / "stdout_audits_a2_archive_master.py.txt")
    cosmo_txt = _read_text(aor_dir / "stdout_cosmo_bb_grand_emergence_masterpiece_runner_v1.py.txt")
    camb_txt = _read_text(aor_dir / "stdout_cosmo_gum_camb_check.py.txt")

    warnings: List[str] = []

    def get_or_fallback(key: str, val: Any) -> Any:
        if val is None:
            warnings.append(key)
            return CANON[key]
        return val

    # --- Extract triple ---
    wU = _find_int(scfp_txt, [
        r"\(wU,s2,s3\)\s*=\s*\((\d+),\s*\d+,\s*\d+\)",
        r"Recovered triple:\s*\(wU,s2,s3\)=\((\d+),\s*\d+,\s*\d+\)",
    ]) or _find_int(a2_txt, [r"Unique triple:\s*\(wU,s2,s3\)\s*=\s*\((\d+),\s*\d+,\s*\d+\)"])

    s2 = _find_int(scfp_txt, [
        r"\(wU,s2,s3\)\s*=\s*\(\d+,\s*(\d+),\s*\d+\)",
        r"Recovered triple:\s*\(wU,s2,s3\)=\(\d+,\s*(\d+),\s*\d+\)",
    ]) or _find_int(a2_txt, [r"Unique triple:\s*\(wU,s2,s3\)\s*=\s*\(\d+,\s*(\d+),\s*\d+\)"])

    s3 = _find_int(scfp_txt, [
        r"\(wU,s2,s3\)\s*=\s*\(\d+,\s*\d+,\s*(\d+)\)",
        r"Recovered triple:\s*\(wU,s2,s3\)=\(\d+,\s*\d+,\s*(\d+)\)",
    ]) or _find_int(a2_txt, [r"Unique triple:\s*\(wU,s2,s3\)\s*=\s*\(\d+,\s*\d+,\s*(\d+)\)"])

    wU = get_or_fallback("wU", wU)
    s2 = get_or_fallback("s2", s2)
    s3 = get_or_fallback("s3", s3)

    q2 = wU - s2
    v = _v2(wU - 1)
    q3 = (wU - 1) // (2 ** v)

    # exact rationals
    alpha_frac = f"1/{wU}"
    alpha = float(Fraction(1, wU))
    alpha_s_frac = f"2/{q3}"
    alpha_s = float(Fraction(2, q3))
    sin2_frac = f"7/{q2}"
    sin2 = float(Fraction(7, q2))

    # --- Gauge lawbook parse ---
    q_tuple = None
    m = _first_match(a2_txt, [r"\bq=\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)"])
    if m:
        q_tuple = [int(m.group(1)), int(m.group(2)), int(m.group(3))]
    q_tuple = get_or_fallback("q_tuple", q_tuple)

    RU = _find_list(a2_txt, [r"\bRU=\[([^\]]+)\]"])
    R2 = _find_list(a2_txt, [r"\bR2=\[([^\]]+)\]"])
    R3 = _find_list(a2_txt, [r"\bR3=\[([^\]]+)\]"])
    RU = get_or_fallback("RU", RU)
    R2 = get_or_fallback("R2", R2)
    R3 = get_or_fallback("R3", R3)

    # --- Yukawa parse ---
    du_frac = _find_fraction(a2_txt, [r"\bdu\s*=\s*([0-9]+/[0-9]+)"])
    lu_frac = _find_fraction(a2_txt, [r"\blu\s*=\s*([0-9]+/[0-9]+)"])
    # delta_iso often prints as decimal; allow either fraction or decimal
    delta_iso_frac = _find_fraction(a2_txt, [r"\bdelta[_ ]iso\s*=\s*([0-9]+/[0-9]+)"])
    if delta_iso_frac is None:
        # if decimal equals 0.041666..., treat as 1/24
        di = _find_float(a2_txt, [r"\bdelta[_ ]iso[^0-9\-+]*([0-9]+\.[0-9]+)"])
        if di is not None and abs(di - float(Fraction(1, 24))) < 1e-9:
            delta_iso_frac = "1/24"

    palette_best = _find_list(a2_txt, [r"\bD1 best:\s*\[([^\]]+)\]"])

    du_frac = get_or_fallback("du_frac", du_frac)
    lu_frac = get_or_fallback("lu_frac", lu_frac)
    delta_iso_frac = get_or_fallback("delta_iso_frac", delta_iso_frac)
    palette_best = get_or_fallback("palette_best", palette_best)

    du = float(Fraction(*map(int, du_frac.split("/"))))
    lu = float(Fraction(*map(int, lu_frac.split("/"))))
    delta_iso = float(Fraction(*map(int, delta_iso_frac.split("/"))))

    # --- Omega values ---
    def omega_val(name: str) -> Optional[float]:
        pats = [
            rf"\bOmega[_ ]{name}\b[^0-9\-+]*([0-9.eE+\-]+)",
            rf"\bΩ{name}\b[^0-9\-+]*([0-9.eE+\-]+)",
        ]
        return _find_float(a2_txt, pats) or _find_float(cosmo_txt, pats)

    Omega_b = omega_val("b")
    Omega_c = omega_val("c")
    Omega_L = omega_val("L") or omega_val("Λ")
    Omega_r = omega_val("r")
    Omega_tot = _find_float(a2_txt, [r"\bOmega[_ ]tot\b[^0-9\-+]*([0-9.eE+\-]+)", r"\bΩtot\b[^0-9\-+]*([0-9.eE+\-]+)"]) \
        or _find_float(cosmo_txt, [r"\bOmega[_ ]tot\b[^0-9\-+]*([0-9.eE+\-]+)", r"\bΩtot\b[^0-9\-+]*([0-9.eE+\-]+)"])

    Omega_b = get_or_fallback("Omega_b", Omega_b)
    Omega_c = get_or_fallback("Omega_c", Omega_c)
    Omega_L = get_or_fallback("Omega_L", Omega_L)
    Omega_r = get_or_fallback("Omega_r", Omega_r)
    Omega_tot = get_or_fallback("Omega_tot", Omega_tot)

    # --- H0 / primordial / neutrinos from A2 stdout ---
    H0 = _find_float(a2_txt, [r"\bH0\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    As = _find_float(a2_txt, [r"\bAs\b[^0-9\-+]*([0-9]+\.[0-9]+e[+\-]?[0-9]+)"])
    ns = _find_float(a2_txt, [r"\bns\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    tau = _find_float(a2_txt, [r"\btau\b[^0-9\-+]*([0-9]+\.[0-9]+)"])

    H0 = get_or_fallback("H0", H0)
    As = get_or_fallback("As", As)
    ns = get_or_fallback("ns", ns)
    tau = get_or_fallback("tau", tau)

    d21 = _find_float(a2_txt, [r"\bd21\b[^0-9\-+]*([0-9]+\.[0-9]+e[+\-]?[0-9]+)"])
    d31 = _find_float(a2_txt, [r"\bd31\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    sumv = _find_float(a2_txt, [r"\bsumv\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    hierarchy = _find_float(a2_txt, [r"\bhierarchy[^0-9\-+]*([0-9]+\.[0-9]+)"])

    d21 = get_or_fallback("d21", d21)
    d31 = get_or_fallback("d31", d31)
    sumv = get_or_fallback("sumv", sumv)
    # hierarchy is optional numeric; if not found, use canonical approx
    hierarchy = get_or_fallback("hierarchy", hierarchy)

    # CAMB metrics
    tt_rms = _find_float(camb_txt, [r"RMS rel diff\s*:\s*([0-9.eE+\-]+)"])
    tt_max = _find_float(camb_txt, [r"Max rel diff\s*:\s*([0-9.eE+\-]+)"])
    tt_rms = get_or_fallback("tt_rms_rel_diff", tt_rms)
    tt_max = get_or_fallback("tt_max_rel_diff", tt_max)

    # claim builder
    def add_claim(cid: str, tier: str, title: str, statement: str,
                  scripts: List[str], evidence: List[str], expected: Dict[str, Any],
                  falsifies_if: List[str], notes: str = "") -> Dict[str, Any]:
        return {
            "id": cid,
            "tier": tier,
            "title": title,
            "statement": statement,
            "scripts": [{"path": p, "url": _repo_blob_url(repo_url, commit, p)} for p in scripts],
            "evidence": [{"path": p, "url": _repo_blob_url(repo_url, commit, p)} for p in evidence],
            "expected": expected,
            "falsifies_if": falsifies_if,
            "notes": notes,
        }

    def aor(p: str) -> str:
        return str((aor_dir / p).as_posix())

    claims: List[Dict[str, Any]] = []

    claims.append(add_claim(
        "FND-A2-001", "HARD", "SCFP++ survivor triple",
        "SCFP++ selector yields (wU, s2, s3) and derived invariants.",
        ["substrate/scfp_integer_selector_v1.py"],
        [aor("stdout_substrate_scfp_integer_selector_v1.py.txt")],
        {"wU": wU, "s2": s2, "s3": s3, "q2": q2, "q3": q3, "v2_wU_minus_1": v},
        ["Selector stdout differs for the canonical triple or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-002", "HARD", "Phi-channel rationals",
        "Exact rationals alpha=1/wU, alpha_s=2/q3, sin2=7/q2 are printed and consistent.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"alpha_frac": alpha_frac, "alpha": alpha, "alpha_s_frac": alpha_s_frac, "alpha_s": alpha_s, "sin2_frac": sin2_frac, "sin2": sin2},
        ["A2 archive output does not print the exact fractions or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-003", "HARD", "Gauge lawbook closure",
        "Gauge lawbook is uniquely derived under declared contracts and yields the canonical triple.",
        ["audits/a2_archive_master.py", "audits/tier_a_master_referee_demo_v6.py"],
        [aor("stdout_audits_a2_archive_master.py.txt"), aor("stdout_audits_tier_a_master_referee_demo_v6.py.txt")],
        {"q_tuple": q_tuple, "RU": RU, "R2": R2, "R3": R3},
        ["Gauge lawbook is not unique or neighborhood scan reports other/multi or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-004", "HARD", "Yukawa closure",
        "Canonical offsets and Palette-B selection close under the A2 selector.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"du_frac": du_frac, "du": du, "lu_frac": lu_frac, "lu": lu, "delta_iso_frac": delta_iso_frac, "delta_iso": delta_iso, "palette_best": palette_best},
        ["Offsets/palette differ or rank/isolation witness changes or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-005", "HARD", "Cosmology Omega-lawbook and near-flatness",
        "BB-36 Omega templates produce Omega_tot close to 1 with printed residual.",
        ["cosmo/bb_grand_emergence_masterpiece_runner_v1.py", "audits/a2_archive_master.py"],
        [aor("stdout_cosmo_bb_grand_emergence_masterpiece_runner_v1.py.txt"), aor("stdout_audits_a2_archive_master.py.txt")],
        {"Omega_b": Omega_b, "Omega_c": Omega_c, "Omega_L": Omega_L, "Omega_r": Omega_r, "Omega_tot": Omega_tot},
        ["Omega templates/values differ or flatness gate fails or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-006", "HARD", "H0 BB-36 closure",
        "BB-36 template yields H0 with rank witness in A2 archive.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"H0": H0},
        ["H0 template/value differs or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-007", "HARD", "Primordial trio closure",
        "BB-36 templates produce As, ns, tau with printed witnesses.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"As": As, "ns": ns, "tau": tau},
        ["Primordial values differ or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-008", "HARD", "Neutrino bundle closure",
        "BB-36 templates produce d21, d31, sumv with hierarchy check.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"d21": d21, "d31": d31, "sumv": sumv, "hierarchy": hierarchy},
        ["Neutrino values differ or hierarchy check fails or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-009", "HARD", "Representation independence",
        "Cross-base roundtrip, CRT injectivity/collision, digit injection designed-fail.",
        ["doc/doc_rosetta_base_gauge_v1.py", "doc/doc_referee_closeout_audit_v1.py"],
        [aor("stdout_doc_doc_rosetta_base_gauge_v1.py.txt"), aor("stdout_doc_doc_referee_closeout_audit_v1.py.txt")],
        {},
        ["Any representation-independence test fails or AoR verify fails."]
    ))

    claims.append(add_claim(
        "FND-A2-010", "HARD", "DOC/DAO Fejer legality and commutant",
        "Finite operator legality certificates for cyclic Fejer kernel and commutation.",
        ["omega/omega_observer_commutant_fejer_v1.py", "audits/tier_a_master_referee_demo_v6.py"],
        [aor("stdout_omega_omega_observer_commutant_fejer_v1.py.txt"), aor("stdout_audits_tier_a_master_referee_demo_v6.py.txt")],
        {},
        ["Fejer legality/commutation fails or AoR verify fails."]
    ))

    claims.append(add_claim(
        "EVID-001", "EVIDENCE", "CAMB TT structural check",
        "CAMB compares Planck vs GUM TT spectra and reports RMS/max diffs under thresholds.",
        ["cosmo/gum_camb_check.py"],
        [aor("stdout_cosmo_gum_camb_check.py.txt"), aor("gum_camb_output/TT_spectrum_Planck_vs_GUM.png")],
        {"tt_rms_rel_diff": tt_rms, "tt_max_rel_diff": tt_max},
        ["CAMB script fails to run when enabled or AoR verify fails."],
        notes="Evidence-only. Does not participate in any selector."
    ))

    doc = {
        "meta": {
            "generated_utc": _utc_now(),
            "repo_url": repo_url,
            "commit": commit,
            "aor_dir": aor_dir.as_posix(),
            "bundle_sha256": args.bundle_sha256,
            "aor_url": _repo_tree_url(repo_url, commit, aor_dir.as_posix()),
            "warnings_used_fallback": warnings,
        },
        "claims": claims,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {out_path} with {len(claims)} claims")
    if warnings:
        print("[WARN] used fallback for:", ", ".join(sorted(set(warnings))))


if __name__ == "__main__":
    main()
PY
chmod +x publication_spine/REFEREE_TRACK/tools/build_claims_json.py
