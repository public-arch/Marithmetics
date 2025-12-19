#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build CLAIMS.json for the Referee Track.

Stdlib-only. Reads:
  - AoR runs.json (optional, for sanity)
  - AoR constants_table.json (for expected values)
Writes:
  - publication_spine/REFEREE_TRACK/CLAIMS.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _repo_blob_url(repo_url: str, commit: str, rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"{repo_url}/blob/{commit}/{rel_path}"


def _repo_tree_url(repo_url: str, commit: str, rel_path: str) -> str:
    rel_path = rel_path.lstrip("/")
    return f"{repo_url}/tree/{commit}/{rel_path}"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/public-arch/Marithmetics")
    ap.add_argument("--commit", required=True, help="Git commit SHA for pinning URLs")
    ap.add_argument("--aor-dir", required=True, help="AoR directory path, e.g. audits/results/foundations_a2_...Z")
    ap.add_argument("--bundle-sha256", required=True, help="BUNDLE_SHA256 value")
    ap.add_argument("--out", default="publication_spine/REFEREE_TRACK/CLAIMS.json")
    args = ap.parse_args()

    repo_url = args.repo_url.rstrip("/")
    commit = args.commit.strip()
    aor_dir = Path(args.aor_dir)
    out_path = Path(args.out)

    constants = _load_json(aor_dir / "constants_table.json")
    # Some bundles use nested structure. Normalize to a flat dict when possible.
    if isinstance(constants.get("constants"), dict):
        flat = dict(constants.get("constants"))
        # also copy top-level common keys if present
        for k, v in constants.items():
            if k not in ("constants",) and not isinstance(v, (dict, list)):
                flat[k] = v
        constants = flat

    # Helper to safely fetch expected values by key
    def ev(key: str) -> Optional[Any]:
        return constants.get(key)

    # Claim templates: keep stable IDs forever.
    claims: List[Dict[str, Any]] = []

    def add_claim(
        cid: str,
        tier: str,
        title: str,
        statement: str,
        scripts: List[str],
        evidence: List[str],
        expected: Dict[str, Any],
        falsifies_if: List[str],
        notes: Optional[str] = None,
    ) -> None:
        claims.append({
            "id": cid,
            "tier": tier,  # HARD | EVIDENCE | OPTIONAL
            "title": title,
            "statement": statement,
            "scripts": [{"path": p, "url": _repo_blob_url(repo_url, commit, p)} for p in scripts],
            "evidence": [{"path": p, "url": _repo_blob_url(repo_url, commit, p)} for p in evidence],
            "expected": expected,
            "falsifies_if": falsifies_if,
            "notes": notes or "",
        })

    # AoR evidence paths
    def aor(p: str) -> str:
        return str((aor_dir / p).as_posix())

    # --- Core claims (adjust titles/statements as needed, keep IDs stable) ---
    add_claim(
        "FND-A2-001", "HARD",
        "SCFP++ survivor triple",
        "SCFP++ selector yields (wU, s2, s3) and derived invariants.",
        ["substrate/scfp_integer_selector_v1.py"],
        [aor("stdout_substrate_scfp_integer_selector_v1.py.txt")],
        {"wU": ev("wU"), "s2": ev("s2"), "s3": ev("s3"), "q2": ev("q2"), "q3": ev("q3")},
        ["Selector stdout differs for the canonical triple or AoR verify fails."]
    )

    add_claim(
        "FND-A2-002", "HARD",
        "Phi-channel rationals",
        "Exact rationals alpha=1/wU, alpha_s=2/q3, sin2=7/q2 are printed and consistent.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"alpha": ev("alpha"), "alpha_s": ev("alpha_s"), "sin2": ev("sin2")},
        ["A2 archive output does not print the exact fractions or AoR verify fails."]
    )

    add_claim(
        "FND-A2-003", "HARD",
        "Gauge lawbook closure",
        "Gauge lawbook is uniquely derived under declared contracts and yields the canonical triple.",
        ["audits/a2_archive_master.py", "audits/tier_a_master_referee_demo_v6.py"],
        [aor("stdout_audits_a2_archive_master.py.txt"), aor("stdout_audits_tier_a_master_referee_demo_v6.py.txt")],
        {"qU": ev("qU"), "q2_mod": ev("q2_mod"), "q3_mod": ev("q3_mod")},
        ["Gauge lawbook is not unique or neighborhood scan reports other/multi or AoR verify fails."]
    )

    add_claim(
        "FND-A2-004", "HARD",
        "Yukawa closure",
        "Canonical offsets and Palette-B selection close under the A2 selector.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"du": ev("du"), "lu": ev("lu"), "palette_best": ev("palette_best"), "delta_iso": ev("delta_iso")},
        ["Offsets/palette differ or rank/isolation witness changes or AoR verify fails."]
    )

    add_claim(
        "FND-A2-005", "HARD",
        "Cosmology Omega-lawbook and near-flatness",
        "BB-36 Omega templates produce Omega_tot close to 1 with printed residual.",
        ["cosmo/bb_grand_emergence_masterpiece_runner_v1.py", "audits/a2_archive_master.py"],
        [aor("stdout_cosmo_bb_grand_emergence_masterpiece_runner_v1.py.txt"), aor("stdout_audits_a2_archive_master.py.txt")],
        {"Omega_b": ev("Omega_b"), "Omega_c": ev("Omega_c"), "Omega_L": ev("Omega_L"), "Omega_r": ev("Omega_r"), "Omega_tot": ev("Omega_tot")},
        ["Omega templates/values differ or flatness gate fails or AoR verify fails."]
    )

    add_claim(
        "FND-A2-006", "HARD",
        "H0 BB-36 closure",
        "BB-36 template yields H0 with rank witness in A2 archive.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"H0": ev("H0")},
        ["H0 template/value differs or AoR verify fails."]
    )

    add_claim(
        "FND-A2-007", "HARD",
        "Primordial trio closure",
        "BB-36 templates produce As, ns, tau with printed witnesses.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"As": ev("As"), "ns": ev("ns"), "tau": ev("tau")},
        ["Primordial values differ or AoR verify fails."]
    )

    add_claim(
        "FND-A2-008", "HARD",
        "Neutrino bundle closure",
        "BB-36 templates produce d21, d31, sumv with hierarchy check.",
        ["audits/a2_archive_master.py"],
        [aor("stdout_audits_a2_archive_master.py.txt")],
        {"d21": ev("d21"), "d31": ev("d31"), "sumv": ev("sumv"), "hierarchy": ev("hierarchy")},
        ["Neutrino values differ or hierarchy check fails or AoR verify fails."]
    )

    add_claim(
        "FND-A2-009", "HARD",
        "Representation independence",
        "Cross-base roundtrip, CRT injectivity/collision, digit injection designed-fail.",
        ["doc/doc_rosetta_base_gauge_v1.py", "doc/doc_referee_closeout_audit_v1.py"],
        [aor("stdout_doc_doc_rosetta_base_gauge_v1.py.txt"), aor("stdout_doc_doc_referee_closeout_audit_v1.py.txt")],
        {},
        ["Any representation-independence test fails or AoR verify fails."]
    )

    add_claim(
        "FND-A2-010", "HARD",
        "DOC/DAO Fejer legality and commutant",
        "Finite operator legality certificates for cyclic Fejer kernel and commutation.",
        ["omega/omega_observer_commutant_fejer_v1.py", "audits/tier_a_master_referee_demo_v6.py"],
        [aor("stdout_omega_omega_observer_commutant_fejer_v1.py.txt"), aor("stdout_audits_tier_a_master_referee_demo_v6.py.txt")],
        {},
        ["Fejer legality/commutation fails or AoR verify fails."]
    )

    add_claim(
        "EVID-001", "EVIDENCE",
        "CAMB TT structural check",
        "CAMB compares Planck vs GUM TT spectra and reports RMS/max diffs under thresholds.",
        ["cosmo/gum_camb_check.py"],
        [aor("stdout_cosmo_gum_camb_check.py.txt"), aor("gum_camb_output/TT_spectrum_Planck_vs_GUM.png")],
        {"tt_rms_rel_diff": ev("tt_rms_rel_diff"), "tt_max_rel_diff": ev("tt_max_rel_diff")},
        ["CAMB script fails to run when enabled or AoR verify fails."],
        notes="Evidence-only. Does not participate in any selector."
    )

    doc = {
        "meta": {
            "generated_utc": _utc_now(),
            "repo_url": repo_url,
            "commit": commit,
            "aor_dir": aor_dir.as_posix(),
            "bundle_sha256": args.bundle_sha256,
            "aor_url": _repo_tree_url(repo_url, commit, aor_dir.as_posix()),
        },
        "claims": claims,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[OK] wrote {out_path} with {len(claims)} claims")


if __name__ == "__main__":
    main()
