cat > publication_spine/REFEREE_TRACK/tools/build_claims_json.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import datetime as _dt
import json
import re
from fractions import Fraction
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


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


def _find_float(text: str, patterns: List[str]) -> Optional[float]:
    m = _first_match(text, patterns)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _find_int(text: str, patterns: List[str]) -> Optional[int]:
    m = _first_match(text, patterns)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _find_fraction(text: str, patterns: List[str]) -> Optional[Tuple[str, Fraction]]:
    m = _first_match(text, patterns)
    if not m:
        return None
    s = m.group(1).strip()
    try:
        if "/" in s:
            a, b = s.split("/", 1)
            fr = Fraction(int(a.strip()), int(b.strip()))
        else:
            fr = Fraction(int(s), 1)
        return (s, fr)
    except Exception:
        return None


def _find_list_of_tokens(text: str, patterns: List[str]) -> Optional[List[str]]:
    m = _first_match(text, patterns)
    if not m:
        return None
    raw = m.group(1).strip()
    tokens = re.split(r"\s*,\s*", raw)
    out: List[str] = []
    for t in tokens:
        t = t.strip().strip("[](){}").strip()
        t = t.strip("'").strip('"')
        if t:
            out.append(t)
    return out if out else None


def _v2(n: int) -> int:
    c = 0
    while n > 0 and n % 2 == 0:
        n //= 2
        c += 1
    return c


@dataclass
class AoRPaths:
    root: Path
    @property
    def scfp_stdout(self) -> Path: return self.root / "stdout_substrate_scfp_integer_selector_v1.py.txt"
    @property
    def a2_stdout(self) -> Path: return self.root / "stdout_audits_a2_archive_master.py.txt"
    @property
    def cosmo_stdout(self) -> Path: return self.root / "stdout_cosmo_bb_grand_emergence_masterpiece_runner_v1.py.txt"
    @property
    def camb_stdout(self) -> Path: return self.root / "stdout_cosmo_gum_camb_check.py.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/public-arch/Marithmetics")
    ap.add_argument("--commit", required=True)
    ap.add_argument("--aor-dir", required=True)
    ap.add_argument("--bundle-sha256", required=True)
    ap.add_argument("--out", default="publication_spine/REFEREE_TRACK/CLAIMS.json")
    ap.add_argument("--non-strict", action="store_true")
    args = ap.parse_args()

    repo_url = args.repo_url.rstrip("/")
    commit = args.commit.strip()
    aor_root = Path(args.aor_dir)
    out_path = Path(args.out)

    P = AoRPaths(root=aor_root)
    scfp_txt = _read_text(P.scfp_stdout)
    a2_txt = _read_text(P.a2_stdout)
    camb_txt = _read_text(P.camb_stdout)
    cosmo_txt = _read_text(P.cosmo_stdout)

    # triple
    wU = _find_int(scfp_txt, [r"\bRecovered triple:\s*\(wU,s2,s3\)=\((\d+),\s*\d+,\s*\d+\)",
                             r"\bUnique triple:\s*\(wU,s2,s3\)=\((\d+),\s*\d+,\s*\d+\)"])
    s2 = _find_int(scfp_txt, [r"\bRecovered triple:\s*\(wU,s2,s3\)=\(\d+,\s*(\d+),\s*\d+\)",
                             r"\bUnique triple:\s*\(wU,s2,s3\)=\(\d+,\s*(\d+),\s*\d+\)"])
    s3 = _find_int(scfp_txt, [r"\bRecovered triple:\s*\(wU,s2,s3\)=\(\d+,\s*\d+,\s*(\d+)\)",
                             r"\bUnique triple:\s*\(wU,s2,s3\)=\(\d+,\s*\d+,\s*(\d+)\)"])
    if wU is None or s2 is None or s3 is None:
        m = _first_match(a2_txt, [r"Unique triple:\s*\(wU,s2,s3\)\s*=\s*\((\d+),\s*(\d+),\s*(\d+)\)"])
        if m:
            wU = wU or int(m.group(1))
            s2 = s2 or int(m.group(2))
            s3 = s3 or int(m.group(3))

    expected_core: Dict[str, Any] = {}
    if wU is not None and s2 is not None and s3 is not None:
        q2 = wU - s2
        v = _v2(wU - 1)
        q3 = (wU - 1) // (2 ** v)
        expected_core.update({"wU": wU, "s2": s2, "s3": s3, "q2": q2, "q3": q3, "v2_wU_minus_1": v})
        expected_core.update({
            "alpha_frac": f"1/{wU}", "alpha": float(Fraction(1, wU)),
            "alpha_s_frac": f"2/{q3}", "alpha_s": float(Fraction(2, q3)),
            "sin2_frac": f"7/{q2}", "sin2": float(Fraction(7, q2)),
        })

    # gauge lawbook
    q_tuple = None
    mq = _first_match(a2_txt, [r"\bq=\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)"])
    if mq:
        q_tuple = [int(mq.group(1)), int(mq.group(2)), int(mq.group(3))]
    RU = _find_list_of_tokens(a2_txt, [r"\bRU=\[([^\]]+)\]"])
    R2 = _find_list_of_tokens(a2_txt, [r"\bR2=\[([^\]]+)\]"])
    R3 = _find_list_of_tokens(a2_txt, [r"\bR3=\[([^\]]+)\]"])

    # yukawa
    du = _find_fraction(a2_txt, [r"\bdu\s*=\s*([0-9]+/[0-9]+)\b"])
    lu = _find_fraction(a2_txt, [r"\blu\s*=\s*([0-9]+/[0-9]+)\b"])
    delta_iso = _find_fraction(a2_txt, [r"\bdelta[_ ]iso[^0-9]*([0-9]+/[0-9]+)\b"])
    palette_best = _find_list_of_tokens(a2_txt, [r"\bD1 best:\s*\[([^\]]+)\]"])

    # omega values
    def omega_val(name: str) -> Optional[float]:
        pats = [rf"\bOmega[_ ]{name}\b[^0-9\-+]*([0-9.eE+\-]+)", rf"\bΩ{name}\b[^0-9\-+]*([0-9.eE+\-]+)"]
        return _find_float(a2_txt, pats) or _find_float(cosmo_txt, pats)

    Omega_b = omega_val("b")
    Omega_c = omega_val("c")
    Omega_L = omega_val("L") or omega_val("Λ")
    Omega_r = omega_val("r")
    Omega_tot = _find_float(a2_txt, [r"\bOmega[_ ]tot\b[^0-9\-+]*([0-9.eE+\-]+)", r"\bΩtot\b[^0-9\-+]*([0-9.eE+\-]+)"]) \
        or _find_float(cosmo_txt, [r"\bOmega[_ ]tot\b[^0-9\-+]*([0-9.eE+\-]+)", r"\bΩtot\b[^0-9\-+]*([0-9.eE+\-]+)"])

    # H0, primordial
    H0 = _find_float(a2_txt, [r"\bH0\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    As = _find_float(a2_txt, [r"\bAs\b[^0-9\-+]*([0-9]+\.[0-9]+e[+\-]?[0-9]+)"])
    ns = _find_float(a2_txt, [r"\bns\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    tau = _find_float(a2_txt, [r"\btau\b[^0-9\-+]*([0-9]+\.[0-9]+)"])

    # neutrinos
    d21 = _find_float(a2_txt, [r"\bd21\b[^0-9\-+]*([0-9]+\.[0-9]+e[+\-]?[0-9]+)"])
    d31 = _find_float(a2_txt, [r"\bd31\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    sumv = _find_float(a2_txt, [r"\bsumv\b[^0-9\-+]*([0-9]+\.[0-9]+)"])
    hierarchy = _find_float(a2_txt, [r"\bhierarchy[^0-9\-+]*([0-9]+\.[0-9]+)"])

    # camb metrics
    tt_rms = _find_float(camb_txt, [r"RMS rel diff\s*:\s*([0-9.eE+\-]+)"])
    tt_max = _find_float(camb_txt, [r"Max rel diff\s*:\s*([0-9.eE+\-]+)"])

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
        return str((aor_root / p).as_posix())

    claims: List[Dict[str, Any]] = []
    claims.append(add_claim(
        "FND-A2-001","HARD","SCFP++ survivor triple",
        "SCFP++ selector yields (wU, s2, s3) and derived invariants.",
        ["substrate/scfp_integer_selector_v1.py"],
        [aor("stdout_substrate_scfp_integer_selector_v1.py.txt")],
        {k: expected_core.get(k) for k in ("wU","s2","s3","q2","q3","v2_wU_minus_1")},
        ["Selector stdout differs for the canonical triple or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-002","HARD","Phi-channel rationals",
        "Exact rationals alpha=1/wU, alpha_s=2/q3, sin2=7/q2 are printed and consistent.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {k: expected_core.get(k) for k in ("alpha_frac","alpha","alpha_s_frac","alpha_s","sin2_frac","sin2")},
        ["A2 archive output does not print the exact fractions or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-003","HARD","Gauge lawbook closure",
        "Gauge lawbook is uniquely derived under declared contracts and yields the canonical triple.",
        ["audits/a2_archive_master.py","audits/tier_a_master_referee_demo_v6.py"],
        [aor("stdout_audits_a2_archive_master.py.txt"), aor("stdout_audits_tier_a_master_referee_demo_v6.py.txt")],
        {"q_tuple": q_tuple, "RU": RU, "R2": R2, "R3": R3},
        ["Gauge lawbook is not unique or neighborhood scan reports other/multi or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-004","HARD","Yukawa closure",
        "Canonical offsets and Palette-B selection close under the A2 selector.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {
            "du_frac": du[0] if du else None, "du": float(du[1]) if du else None,
            "lu_frac": lu[0] if lu else None, "lu": float(lu[1]) if lu else None,
            "delta_iso_frac": delta_iso[0] if delta_iso else None, "delta_iso": float(delta_iso[1]) if delta_iso else None,
            "palette_best": palette_best,
        },
        ["Offsets/palette differ or rank/isolation witness changes or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-005","HARD","Cosmology Omega-lawbook and near-flatness",
        "BB-36 Omega templates produce Omega_tot close to 1 with printed residual.",
        ["cosmo/bb_grand_emergence_masterpiece_runner_v1.py","audits/a2_archive_master.py"],
        [aor("stdout_cosmo_bb_grand_emergence_masterpiece_runner_v1.py.txt"), aor("stdout_audits_a2_archive_master.py.txt")],
        {"Omega_b": Omega_b, "Omega_c": Omega_c, "Omega_L": Omega_L, "Omega_r": Omega_r, "Omega_tot": Omega_tot},
        ["Omega templates/values differ or flatness gate fails or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-006","HARD","H0 BB-36 closure",
        "BB-36 template yields H0 with rank witness in A2 archive.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"H0": H0},
        ["H0 template/value differs or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-007","HARD","Primordial trio closure",
        "BB-36 templates produce As, ns, tau with printed witnesses.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"As": As, "ns": ns, "tau": tau},
        ["Primordial values differ or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-008","HARD","Neutrino bundle closure",
        "BB-36 templates produce d21, d31, sumv with hierarchy check.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"d21": d21, "d31": d31, "sumv": sumv, "hierarchy": hierarchy},
        ["Neutrino values differ or hierarchy check fails or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-009","HARD","Representation independence",
        "Cross-base roundtrip, CRT injectivity/collision, digit injection designed-fail.",
        ["doc/doc_rosetta_base_gauge_v1.py","doc/doc_referee_closeout_audit_v1.py"],
        [aor("stdout_doc_doc_rosetta_base_gauge_v1.py.txt"), aor("stdout_doc_doc_referee_closeout_audit_v1.py.txt")],
        {},
        ["Any representation-independence test fails or AoR verify fails."]
    ))
    claims.append(add_claim(
        "FND-A2-010","HARD","DOC/DAO Fejer legality and commutant",
        "Finite operator legality certificates for cyclic Fejer kernel and commutation.",
        ["omega/omega_observer_commutant_fejer_v1.py","audits/tier_a_master_referee_demo_v6.py"],
        [aor("stdout_omega_omega_observer_commutant_fejer_v1.py.txt"), aor("stdout_audits_tier_a_master_referee_demo_v6.py.txt")],
        {},
        ["Fejer legality/commutation fails or AoR verify fails."]
    ))
    claims.append(add_claim(
        "EVID-001","EVIDENCE","CAMB TT structural check",
        "CAMB compares Planck vs GUM TT spectra and reports RMS/max diffs under thresholds.",
        ["cosmo/gum_camb_check.py"],
        [aor("stdout_cosmo_gum_camb_check.py.txt"), aor("gum_camb_output/TT_spectrum_Planck_vs_GUM.png")],
        {"tt_rms_rel_diff": tt_rms, "tt_max_rel_diff": tt_max},
        ["CAMB script fails to run when enabled or AoR verify fails."],
        notes="Evidence-only. Does not participate in any selector."
    ))

    if not args.non_strict:
        missing = []
        for c in claims:
            if c["tier"] == "HARD":
                for ek, evv in c["expected"].items():
                    if evv is None and ek not in ("palette_best",):
                        missing.append(f"{c['id']}:{ek}")
        if missing:
            raise SystemExit("[ERR] missing expected values: " + ", ".join(missing))

    doc = {
        "meta": {
            "generated_utc": _utc_now(),
            "repo_url": repo_url,
            "commit": commit,
            "aor_dir": aor_root.as_posix(),
            "bundle_sha256": args.bundle_sha256,
            "aor_url": _repo_tree_url(repo_url, commit, aor_root.as_posix()),
        },
        "claims": claims,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {out_path} with {len(claims)} claims")


if __name__ == "__main__":
    main()
PY
chmod +x publication_spine/REFEREE_TRACK/tools/build_claims_json.py
