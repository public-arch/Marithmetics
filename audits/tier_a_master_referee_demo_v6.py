# -*- coding: utf-8 -*-
"""
Tier‑A Master Referee Demo — v6 (Fix + Upgrade)
SCFP++ / Φ‑channel / Yukawa / Cosmology / DOC

Fix:
  • Correct Yukawa negative control: swapping order cannot fail E1 because E1 checks sorted sectors.
    New NC3 duplicates a value inside a sector -> strict inequality fails -> correct "gate has teeth" test.

Upgrade:
  • Adds Stage 1d: Residue τ‑pressure rigidity scan (23,040 residue lawbooks × 27 τ-patterns)
    to quantify robustness/rigidity vs residue tuning.

Run:
  python tier_a_master_referee_demo_v6.py
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
import platform
import random
import sys
import time
from fractions import Fraction
from typing import Dict, List, Optional, Tuple, Any


# -------------------------
# Pretty CLI
# -------------------------

USE_COLOR = sys.stdout.isatty()

class C:
    reset = "\033[0m" if USE_COLOR else ""
    bold  = "\033[1m" if USE_COLOR else ""
    dim   = "\033[2m" if USE_COLOR else ""
    grn   = "\033[32m" if USE_COLOR else ""
    red   = "\033[31m" if USE_COLOR else ""
    ylw   = "\033[33m" if USE_COLOR else ""
    cyn   = "\033[36m" if USE_COLOR else ""

def hr(title: str = "", w: int = 88):
    if title:
        pad = max(0, w - 2 - len(title))
        left = pad // 2
        right = pad - left
        print("═" * left + f" {title} " + "═" * right)
    else:
        print("═" * w)

def section(title: str):
    print()
    print(C.bold + title + C.reset)
    print("─" * 88)

def badge(ok: bool) -> str:
    return (C.grn + "✔" + C.reset) if ok else (C.red + "✖" + C.reset)

def now_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def sha256_head(path: str, nhex: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:nhex]


# -------------------------
# Module loader (by filename in CWD)
# -------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # Marithmetics/ repo root

def load_module(module_name: str, relpath: str):
    path = (REPO_ROOT / relpath).resolve()
    if not path.exists():
        raise FileNotFoundError(str(path))
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore
    return mod, str(path)


# -------------------------
# Core math (self-contained SCFP lane runner)
# -------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True

def v2(n: int) -> int:
    if n == 0:
        return 0
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k

def euler_phi(n: int) -> int:
    if n <= 0:
        return 0
    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if x > 1:
        result -= result // x
    return result

@dataclasses.dataclass(frozen=True)
class LaneLaw:
    label: str
    q: int
    residues: Tuple[int, ...]
    tau: float
    span_lo: int
    span_hi: int  # inclusive

_LANE_CACHE: Dict[Tuple[int, Tuple[int,...], float, int, int], List[int]] = {}

def lane_survivors(cfg: LaneLaw) -> List[int]:
    key = (cfg.q, tuple(sorted(cfg.residues)), float(cfg.tau), cfg.span_lo, cfg.span_hi)
    if key in _LANE_CACHE:
        return _LANE_CACHE[key][:]

    out = []
    q = cfg.q
    res = set(cfg.residues)
    for w in range(cfg.span_lo, cfg.span_hi + 1):
        if not is_prime(w):
            continue
        # C2
        if (w % q) not in res:
            continue
        # C3
        if not (q > math.isqrt(w)):
            continue
        # C4
        theta = euler_phi(w - 1) / (w - 1)
        if theta < cfg.tau:
            continue
        out.append(w)

    _LANE_CACHE[key] = out[:]
    return out

def admissible_triples(U: List[int], S2: List[int], S3: List[int]) -> List[Tuple[int,int,int]]:
    triples = []
    for wU in U:
        for s2 in S2:
            for s3 in S3:
                if len({wU, s2, s3}) != 3:   # T1 distinct
                    continue
                if (wU - s2) <= 0:           # T2 positive q2
                    continue
                triples.append((wU, s2, s3))
    return sorted(triples)

def classify_triple_from_laws(laws: Dict[str, LaneLaw]):
    U = lane_survivors(laws["U(1)"])
    S2 = lane_survivors(laws["SU(2)"])
    S3 = lane_survivors(laws["SU(3)"])
    triples = admissible_triples(U, S2, S3)
    if len(triples) == 0:
        return "none", None, (U, S2, S3)
    if len(triples) == 1:
        return "unique", triples[0], (U, S2, S3)
    return "multi", triples, (U, S2, S3)


# -------------------------
# Yukawa lawbook (E1–E6)
# -------------------------

PALETTE_B = [
    Fraction(0,1),  Fraction(4,3),  Fraction(7,4),
    Fraction(8,3),  Fraction(4,1),  Fraction(11,3),
    Fraction(13,8), Fraction(21,8), Fraction(9,2),
]

def check_E1_E5(p: List[Fraction]) -> bool:
    # Sector-wise sorted copies
    u = sorted(p[0:3])
    d = sorted(p[3:6])
    l = sorted(p[6:9])

    # E1: monotone ladders (strict in value, NOT coordinate order)
    if not (u[0] < u[1] < u[2] and d[0] < d[1] < d[2] and l[0] < l[1] < l[2]):
        return False

    # E2: duality offsets between sector minima
    if (d[0] - u[0]) != Fraction(8, 3):
        return False
    if (l[0] - u[0]) != Fraction(13, 8):
        return False

    # E3: allowed denominators for palette entries
    base_denoms = {2, 3, 4, 6, 8}
    if not all(fr.denominator in (base_denoms | {1}) for fr in p):
        return False

    # E4: spacing denominators across the ordered 9-tuple
    allowed = base_denoms | {1, 12, 16, 24}
    for i in range(8):
        if (p[i + 1] - p[i]).denominator not in allowed:
            return False

    # E5: sum denominator divides 24
    sden = sum(p, start=Fraction(0, 1)).denominator
    if sden not in {1, 2, 3, 4, 6, 8, 12, 16, 24}:
        return False

    return True

def isolation_gap(p: List[Fraction]) -> float:
    deltas = [Fraction(0, 1), Fraction(1, 8), -Fraction(1, 8)]
    lattice_denoms = {1, 2, 3, 4, 6, 8, 12, 16, 24}

    def in_lattice(fr: Fraction) -> bool:
        return (0 <= fr <= 5) and (fr.denominator in lattice_denoms)

    def L1(A: List[Fraction], B: List[Fraction]) -> float:
        return sum(abs(float(A[i] - B[i])) for i in range(9))

    neighbors = []
    for i in range(9):
        for dlt in deltas:
            v = list(p)
            v[i] = v[i] + dlt
            if v == p:
                continue
            if all(in_lattice(fr) for fr in v):
                neighbors.append(v)

    if not neighbors:
        return float("inf")
    return min(L1(p, c) for c in neighbors)

def yukawa_D1_stats(p0: List[Fraction]) -> Tuple[List[int], int, int]:
    deltas = [Fraction(0,1), Fraction(1,8), -Fraction(1,8)]
    lattice_denoms = {1, 2, 3, 4, 6, 8, 12, 16, 24}

    def in_lattice(fr: Fraction) -> bool:
        return (0 <= fr <= 5) and (fr.denominator in lattice_denoms)

    options_per_idx: List[List[Fraction]] = []
    for i in range(9):
        opts = []
        for d in deltas:
            v = p0[i] + d
            if in_lattice(v):
                opts.append(v)
        opts = sorted(set(opts))
        options_per_idx.append(opts)

    sizes = [len(o) for o in options_per_idx]
    n15 = 0
    n16 = 0

    for tup in itertools.product(*options_per_idx):
        p = list(tup)
        if check_E1_E5(p):
            n15 += 1
            if isolation_gap(p) > 0.05:
                n16 += 1

    return sizes, n15, n16


# -------------------------
# Canonical cyclic Fejér kernel PSD check (self-contained)
# -------------------------

def canonical_fejer_first_row(N: int, L: int) -> List[float]:
    row = [0.0] * N
    denom = (L + 1)
    for k in range(N):
        d = min(k, N - k)
        row[k] = (1.0 - (d / denom)) if d <= L else 0.0
    s = sum(row)
    return [x / s for x in row] if s > 0 else row

def dft_real(x: List[float]) -> List[complex]:
    N = len(x)
    out = []
    for m in range(N):
        acc = 0j
        for n in range(N):
            ang = -2 * math.pi * m * n / N
            acc += x[n] * complex(math.cos(ang), math.sin(ang))
        out.append(acc)
    return out

def fejer_doc_dao_audit(N: int = 64, L: int = 8, eps_psd: float = 1e-12) -> Dict[str, Any]:
    row = canonical_fejer_first_row(N, L)
    sym_circ = all(abs(row[k] - row[-k % N]) < 1e-15 for k in range(N))
    unit_mass = abs(sum(row) - 1.0) < 1e-12
    nonneg = all(x >= -1e-15 for x in row)
    hat = dft_real(row)
    min_hat = min(h.real for h in hat)
    max_im = max(abs(h.imag) for h in hat)
    psd = (min_hat >= -eps_psd)
    comm_norm = 0.0  # circulant => commutes with shift
    return {
        "symmetric_circular": sym_circ,
        "unit_mass": unit_mass,
        "nonneg": nonneg,
        "PSD": psd,
        "min_hatF": min_hat,
        "max|Im|": max_im,
        "comm_norm": comm_norm,
    }


# -------------------------
# Scoreboard
# -------------------------

@dataclasses.dataclass
class TestResult:
    name: str
    ok: bool
    detail: str = ""
    tier: str = "HARD"  # HARD or EVIDENCE

def add_result(results: List[TestResult], name: str, ok: bool, detail: str = "", tier: str = "HARD"):
    results.append(TestResult(name=name, ok=ok, detail=detail, tier=tier))


# -------------------------
# Stage 1d helper: fast τ-pressure evaluation for residue lawbooks
# -------------------------

def build_candidate_tables(span_lo: int, span_hi: int, qs: List[int]) -> Dict[int, Dict[int, List[Tuple[int, float]]]]:
    primes = [w for w in range(span_lo, span_hi + 1) if is_prime(w)]
    out: Dict[int, Dict[int, List[Tuple[int, float]]]] = {}
    for q in qs:
        out[q] = {r: [] for r in range(1, q)}
        for w in primes:
            if not (q > math.isqrt(w)):  # C3
                continue
            r = w % q
            if r == 0:
                continue
            theta = euler_phi(w - 1) / (w - 1)
            out[q][r].append((w, theta))
    return out

def filter_by_tau(cands: List[Tuple[int, float]], tau: float) -> List[int]:
    # candidates are tiny; list comprehension is fastest enough
    return [w for (w, th) in cands if th >= tau]

def classify_from_candidates(
    U_cands: List[Tuple[int,float]], tauU: float,
    S2_cands: List[Tuple[int,float]], tau2: float,
    S3_cands: List[Tuple[int,float]], tau3: float,
) -> Tuple[str, Optional[Tuple[int,int,int]]]:
    U = filter_by_tau(U_cands, tauU)
    S2 = filter_by_tau(S2_cands, tau2)
    S3 = filter_by_tau(S3_cands, tau3)
    triples = admissible_triples(U, S2, S3)
    if len(triples) == 0:
        return "none", None
    if len(triples) == 1:
        return "unique", triples[0]
    return "multi", None


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Tier‑A Master Referee Demo (v6) — fixed Yukawa NC + residue τ-pressure rigidity")
    ap.add_argument("--fast", action="store_true", help="Reduce Monte Carlo draws (still deterministic)")
    ap.add_argument("--no-camb", action="store_true", help="Skip CAMB even if installed")
    args = ap.parse_args()

    draws_mc = 5000 if args.fast else 20000

    hr("Tier‑A Master Referee Demo — v6 (Fix + Upgrade)", 88)
    print()

    results: List[TestResult] = []

    # Stage 0
    section("Stage 0 — Environment & Module Discovery")
    print(f"• UTC time: {now_utc_iso()}")
    print(f"• Python : {sys.version.split()[0]} ({platform.platform()})")
    print(f"• CWD    : {os.getcwd()}")
    print(f"• Script : {os.path.dirname(os.path.abspath(__file__))}")
    print()

    mods = {}
    for key, fname in [
        ("sm",   "sm/sm_standard_model_demo_v1.py"),
        ("bb",   "cosmo/bb_grand_emergence_masterpiece_runner_v1.py"),
        ("om",   "omega/omega_observer_commutant_fejer_v1.py"),
        ("scfp", "substrate/scfp_integer_selector_v1.py"),
    ]:

        try:
            mod, path = load_module(key, fname)
            mods[key] = (mod, path)
            head = sha256_head(path)
            print(f"{badge(True)} {fname[:-3]} loaded  ({os.path.basename(path)}, sha256[:16]={head})")
            add_result(results, f"Module present: {fname}", True, path, tier="HARD")
        except Exception as e:
            print(f"{badge(False)} Could not load {fname}: {repr(e)}")
            add_result(results, f"Module present: {fname}", False, repr(e), tier="HARD")

    # Require SM + BB + Ω
    req = [
        "Module present: sm_standard_model_demo_v1.py",
        "Module present: bb_grand_emergence_masterpiece_runner_v1.py",
        "Module present: omega_observer_commutant_fejer_v1.py",
    ]
    if not all(r.ok for r in results if r.name in req):
        hr("ABORT: Missing required modules", 88)
        sys.exit(2)

    sm = mods["sm"][0]
    bb = mods["bb"][0]

    # Stage 1
    section("Stage 1 — SCFP++ Gauge Integer Selection (C1–C4 + T1–T2)")
    expected = (137, 107, 103)

    canonical = {
        "U(1)" : LaneLaw("U(1)",  q=17, residues=(1,5), tau=0.31, span_lo=97, span_hi=180),
        "SU(2)": LaneLaw("SU(2)", q=13, residues=(3,),  tau=0.30, span_lo=97, span_hi=180),
        "SU(3)": LaneLaw("SU(3)", q=17, residues=(1,),  tau=0.30, span_lo=97, span_hi=180),
    }

    print("Canonical lane lawbook:")
    for k in ["U(1)", "SU(2)", "SU(3)"]:
        law = canonical[k]
        print(f"  {k:5s}: q={law.q}, residues={list(law.residues)}, τ≥{law.tau}, span={law.span_lo}..{law.span_hi}")
    print(f"  expected triple: {expected}\n")

    kind, tri, lanes = classify_triple_from_laws(canonical)
    U, S2, S3 = lanes
    print("Lane survivors (self-contained):")
    print(f"  U(1) : {U}")
    print(f"  SU(2): {S2}")
    print(f"  SU(3): {S3}\n")
    print(f"Triple status: {kind}")
    print(f"Unique triple: {tri}\n")

    add_result(results, "Gauge triple (self-contained) equals canonical", (kind == "unique" and tri == expected),
               f"kind={kind}, triple={tri}", tier="HARD")

    # SM triple
    try:
        survivors_sm, _ = sm.scfp_survivors_rerun()
        tri_sm = (survivors_sm["U(1)"][0], survivors_sm["SU(2)"][0], survivors_sm["SU(3)"][0])
        print(f"{badge(True)} SM triple: {tri_sm}")
        add_result(results, "Gauge triple (SM) equals canonical", (tri_sm == expected), f"SM={tri_sm}", tier="HARD")
    except Exception as e:
        print(f"{badge(False)} SM triple extraction failed: {repr(e)}")
        add_result(results, "Gauge triple (SM) equals canonical", False, repr(e), tier="HARD")

    # BB triple
    try:
        scfp_dict = bb.select_scfp_integers()
        tri_bb = (int(scfp_dict["wU"]), int(scfp_dict["s2"]), int(scfp_dict["s3"]))
        print(f"{badge(True)} BB‑36 triple: {tri_bb}")
        add_result(results, "Gauge triple (BB‑36) equals canonical", (tri_bb == expected), f"BB‑36={tri_bb}", tier="HARD")
    except Exception as e:
        print(f"{badge(False)} BB‑36 triple extraction failed: {repr(e)}")
        add_result(results, "Gauge triple (BB‑36) equals canonical", False, repr(e), tier="HARD")

    # Stage 1b robustness
    section("Stage 1b — Gauge Lawbook Robustness (τ + span neighborhood)")
    tau_grid = {"U(1)": [0.29,0.31,0.33], "SU(2)": [0.28,0.30,0.32], "SU(3)": [0.28,0.30,0.32]}
    span_grid = [(87,170),(87,180),(87,190),(97,170),(97,180),(97,190),(107,170),(107,180),(107,190)]

    tested = canonical_count = other_count = multi_count = none_count = 0
    for tU in tau_grid["U(1)"]:
        for t2 in tau_grid["SU(2)"]:
            for t3 in tau_grid["SU(3)"]:
                for (loU, hiU) in span_grid:
                    for (lo2, hi2) in span_grid:
                        for (lo3, hi3) in span_grid:
                            tested += 1
                            laws = {
                                "U(1)" : LaneLaw("U(1)",  17, (1,5), tU, loU, hiU),
                                "SU(2)": LaneLaw("SU(2)", 13, (3,),  t2, lo2, hi2),
                                "SU(3)": LaneLaw("SU(3)", 17, (1,),  t3, lo3, hi3),
                            }
                            k, t, _ = classify_triple_from_laws(laws)
                            if k == "none":
                                none_count += 1
                            elif k == "multi":
                                multi_count += 1
                            else:
                                if t == expected:
                                    canonical_count += 1
                                else:
                                    other_count += 1

    print("Results:")
    print(f"  tested:    {tested}")
    print(f"  canonical: {canonical_count}")
    print(f"  other:     {other_count}")
    print(f"  multi:     {multi_count}")
    print(f"  none:      {none_count}\n")

    add_result(results, "Gauge robustness: no other/multi triples in neighborhood",
               (other_count == 0 and multi_count == 0),
               f"tested={tested}, canonical={canonical_count}, other={other_count}, multi={multi_count}, none={none_count}",
               tier="HARD")

    # Negative control: raise SU(3) tau slightly
    laws_bad = dict(canonical)
    laws_bad["SU(3)"] = LaneLaw("SU(3)", 17, (1,), 0.32, 97, 180)
    kbad, tb, _ = classify_triple_from_laws(laws_bad)
    print(f"Negative control (raise SU(3) τ to 0.32): {kbad}, tri={tb}")
    add_result(results, "Negative control: raising τ removes triple", (kbad == "none"), f"kind={kbad}, tri={tb}", tier="HARD")

    # Stage 1c residue baseline scan (as before, but no overclaim)
    section("Stage 1c — Residue Scan (baseline τ/span)")
    print("Scan class:")
    print("  • U(1):  q=17, choose 2 residues from {1..16}")
    print("  • SU(2): q=13, choose 1 residue from {1..12}")
    print("  • SU(3): q=17, choose 1 residue from {1..16}")
    print("  • τ and span fixed to canonical.\n")

    U_q = 17
    SU2_q = 13
    SU3_q = 17
    U_pairs = list(itertools.combinations(range(1, U_q), 2))
    SU2_res = list(range(1, SU2_q))
    SU3_res = list(range(1, SU3_q))

    total_res_cfg = 0
    unique_any = 0
    unique_canon = 0
    unique_other = 0
    other_triples: Dict[Tuple[int,int,int], int] = {}

    for (r1, r2) in U_pairs:
        for r2s in SU2_res:
            for r3s in SU3_res:
                total_res_cfg += 1
                lawsR = {
                    "U(1)" : LaneLaw("U(1)",  U_q,  (r1, r2), 0.31, 97, 180),
                    "SU(2)": LaneLaw("SU(2)", SU2_q,(r2s,),  0.30, 97, 180),
                    "SU(3)": LaneLaw("SU(3)", SU3_q,(r3s,),  0.30, 97, 180),
                }
                k, t, _ = classify_triple_from_laws(lawsR)
                if k == "unique":
                    unique_any += 1
                    if t == expected:
                        unique_canon += 1
                    else:
                        unique_other += 1
                        other_triples[t] = other_triples.get(t, 0) + 1

    print(f"Total residue-lawbooks tested: {total_res_cfg}")
    print(f"Unique triple found:          {unique_any}")
    print(f"  → canonical triple:         {unique_canon}")
    print(f"  → other triples:            {unique_other}")

    if unique_other > 0:
        top = sorted(other_triples.items(), key=lambda kv: kv[1], reverse=True)[:8]
        print("\nOther unique triples observed (top):")
        for t, cnt in top:
            print(f"  {t}  count={cnt}")

    add_result(results, "Residue scan: canonical triple appears (baseline τ/span) [evidence]",
               (unique_canon > 0),
               f"tested={total_res_cfg}, unique_any={unique_any}, canon={unique_canon}, other={unique_other}",
               tier="EVIDENCE")

    # Stage 1d: τ-pressure rigidity scan across ALL residue lawbooks (upgrade)
    section("Stage 1d — Residue τ‑Pressure Rigidity Scan (27 τ‑patterns, spans fixed)")
    print("Goal: quantify whether residue tuning yields stable outputs under τ-pressure (3³ grid).")
    print("Categories:")
    print("  • all_27_same           : all 27 patterns produce a unique triple, always identical")
    print("  • some_none_one_triple  : some patterns yield NONE, but whenever unique appears it’s the same triple")
    print("  • multiple_triples      : unique outcomes change across τ-patterns")
    print("  • multi                 : any τ-pattern yields multiple admissible triples")
    print("  • all_none              : no τ-pattern yields a triple\n")

    # Precompute candidate tables (fast)
    cand = build_candidate_tables(97, 180, qs=[13,17])

    tauU_list = [0.29, 0.31, 0.33]
    tau2_list = [0.28, 0.30, 0.32]
    tau3_list = [0.28, 0.30, 0.32]
    tau_patterns = list(itertools.product(tauU_list, tau2_list, tau3_list))  # 27

    counts = {"all_27_same": 0, "some_none_one_triple": 0, "multiple_triples": 0, "multi": 0, "all_none": 0}
    stable_triple_freq: Dict[Tuple[int,int,int], int] = {}

    # Track canonical residue lawbook explicitly
    canonical_residues = ((1,5), 3, 1)
    canonical_cat = None
    canonical_tau_unique = 0
    canonical_tau_none = 0

    for (r1, r2) in U_pairs:
        U_cands = cand[17][r1] + cand[17][r2]
        for r2s in SU2_res:
            S2_cands = cand[13][r2s]
            for r3s in SU3_res:
                S3_cands = cand[17][r3s]

                uniq_triples = []
                saw_multi = False
                saw_none = False

                for (tU, t2, t3) in tau_patterns:
                    k, t = classify_from_candidates(U_cands, tU, S2_cands, t2, S3_cands, t3)
                    if k == "multi":
                        saw_multi = True
                        break
                    if k == "none":
                        saw_none = True
                    else:
                        uniq_triples.append(t)

                if saw_multi:
                    counts["multi"] += 1
                    cat = "multi"
                else:
                    if len(uniq_triples) == 0:
                        counts["all_none"] += 1
                        cat = "all_none"
                    else:
                        uniq_set = set(uniq_triples)
                        if len(uniq_set) == 1:
                            t0 = next(iter(uniq_set))
                            stable_triple_freq[t0] = stable_triple_freq.get(t0, 0) + 1
                            if saw_none:
                                counts["some_none_one_triple"] += 1
                                cat = "some_none_one_triple"
                            else:
                                counts["all_27_same"] += 1
                                cat = "all_27_same"
                        else:
                            counts["multiple_triples"] += 1
                            cat = "multiple_triples"

                if (r1, r2) == canonical_residues[0] and r2s == canonical_residues[1] and r3s == canonical_residues[2]:
                    canonical_cat = cat
                    canonical_tau_unique = len(uniq_triples)
                    canonical_tau_none = (27 - canonical_tau_unique) if not saw_multi else 0

    print("Rigidity distribution across residue lawbooks:")
    for k in ["all_27_same","some_none_one_triple","multiple_triples","multi","all_none"]:
        print(f"  {k:>22s}: {counts[k]}")
    print()

    # Report canonical residue lawbook category
    print("Canonical residue lawbook (U={1,5}, SU2={3}, SU3={1}) rigidity:")
    print(f"  category: {canonical_cat}")
    if canonical_cat in ("all_27_same", "some_none_one_triple"):
        print(f"  unique patterns: {canonical_tau_unique}/27, none patterns: {canonical_tau_none}/27")
    print()

    # Show top stable triples by how many residue-lawbooks make them stable (informational)
    if stable_triple_freq:
        top = sorted(stable_triple_freq.items(), key=lambda kv: kv[1], reverse=True)[:10]
        print("Top stable triples (by # of residue-lawbooks yielding stable output):")
        for t, cnt in top:
            print(f"  {t} : {cnt}")
        print()

    add_result(
        results,
        "Residue τ‑pressure rigidity: canonical residue lawbook is stable (not multi/variable)",
        (canonical_cat in ("all_27_same", "some_none_one_triple")),
        f"category={canonical_cat}, unique_patterns={canonical_tau_unique}/27",
        tier="EVIDENCE",
    )

    # Stage 2: Φ-channel
    section("Stage 2 — Φ‑channel Gauge Couplings (exact rationals)")
    wU, s2, s3 = expected
    q2 = wU - s2
    v2U = v2(wU - 1)
    q3 = (wU - 1) // (2**v2U)

    alpha_em = Fraction(1, wU)
    Theta_q2 = Fraction(euler_phi(q2), q2)
    sin2W = Theta_q2 * Fraction(2**v2U - 1, 2**v2U)
    alpha_s = Fraction(2, q3)

    print(f"α_em   = 1/{wU} = {float(alpha_em):.9f}")
    print(f"sin²θW = {sin2W.numerator}/{sin2W.denominator} = {float(sin2W):.9f}")
    print(f"α_s    = 2/{q3} = {float(alpha_s):.9f}\n")

    try:
        survivors_sm, _ = sm.scfp_survivors_rerun()
        a_sm, s2_sm, as_sm, _meta = sm.structural_constants_from_survivors(survivors_sm)
        ok_phi = (abs(a_sm - float(alpha_em)) < 1e-15 and abs(s2_sm - float(sin2W)) < 1e-15 and abs(as_sm - float(alpha_s)) < 1e-15)
        print(f"{badge(ok_phi)} SM Φ constants match self-contained.")
        add_result(results, "Φ‑channel: SM constants match self-contained", ok_phi,
                   f"|Δα|={abs(a_sm-float(alpha_em)):.3e}, |Δsin2|={abs(s2_sm-float(sin2W)):.3e}, |Δαs|={abs(as_sm-float(alpha_s)):.3e}",
                   tier="HARD")
    except Exception as e:
        print(f"{badge(False)} Could not validate SM Φ constants: {repr(e)}")
        add_result(results, "Φ‑channel: SM constants match self-contained", False, repr(e), tier="HARD")

    # Stage 3: Yukawa
    section("Stage 3 — Yukawa Exponent Lawbook (E1–E6 + negative controls)")
    ok_E15 = check_E1_E5(PALETTE_B)
    gap = isolation_gap(PALETTE_B)
    print(f"Palette‑B: E1–E5={ok_E15}, δ_iso={gap:.6f}")
    add_result(results, "Yukawa: Palette‑B satisfies E1–E5", ok_E15, tier="HARD")
    add_result(results, "Yukawa: Palette‑B passes E6 (δ_iso>0.05)", (gap > 0.05), f"δ_iso={gap:.6f}", tier="HARD")

    sizes, n15, n16 = yukawa_D1_stats(PALETTE_B)
    total_D1 = 1
    for s in sizes:
        total_D1 *= s
    print(f"\nLocal D¹ options per index: {sizes}")
    print(f"Total D¹ candidates: {total_D1}")
    print(f"E1–E5 survivors:    {n15}")
    print(f"E1–E6 survivors:    {n16}\n")
    add_result(results, "Yukawa D¹ survivor counts match expected (81,81)", (n15 == 81 and n16 == 81),
               f"E1–E5={n15}, E1–E6={n16}", tier="HARD")

    # Global MC scarcity
    random.seed(1337)
    allowed_denoms = [1,2,3,4,6,8,12,16,24]
    L_Yuk = sorted({Fraction(n, d) for d in allowed_denoms for n in range(0, 5*d + 1)})
    hits15 = hits16 = 0
    for _ in range(draws_mc):
        p = [random.choice(L_Yuk) for _ in range(9)]
        if check_E1_E5(p):
            hits15 += 1
            if isolation_gap(p) > 0.05:
                hits16 += 1
    print(f"Global MC: draws={draws_mc}, E1–E5 hits={hits15}, E1–E6 hits={hits16}\n")
    add_result(results, "Yukawa global MC yields 0 hits (deterministic)", (hits15 == 0 and hits16 == 0),
               f"draws={draws_mc}, hits15={hits15}, hits16={hits16}", tier="HARD")

    # Negative controls (correct and guaranteed)
    # NC1: break E2 offsets by perturbing U-minimum
    p_bad1 = list(PALETTE_B)
    p_bad1[0] = p_bad1[0] + Fraction(1,8)
    bad1_ok = check_E1_E5(p_bad1)
    add_result(results, "Negative control (Yukawa): perturb u-min breaks E2 (fails E1–E5)", (bad1_ok is False),
               f"E1–E5(p_bad1)={bad1_ok}", tier="HARD")

    # NC2: break E3 by inserting denom 5 (not allowed)
    p_bad2 = list(PALETTE_B)
    p_bad2[1] = Fraction(1,5)
    bad2_ok = check_E1_E5(p_bad2)
    add_result(results, "Negative control (Yukawa): denom=5 breaks E3 (fails E1–E5)", (bad2_ok is False),
               f"E1–E5(p_bad2)={bad2_ok}", tier="HARD")

    # NC3 (FIXED): violate E1 strict inequality by duplicating a u-sector value
    # Because E1 checks sorted sector values, *ordering changes do nothing*; duplicates are a real E1 violation.
    p_bad3 = list(PALETTE_B)
    p_bad3[2] = p_bad3[1]  # duplicate -> u-sector not strictly increasing
    bad3_ok = check_E1_E5(p_bad3)
    add_result(results, "Negative control (Yukawa): duplicate in u-sector violates E1 (fails E1–E5)", (bad3_ok is False),
               f"E1–E5(p_bad3)={bad3_ok}", tier="HARD")

    # Stage 4: Cosmology (keep simple; you already have exponent scans elsewhere)
    section("Stage 4 — Structural Cosmology (engine output)")
    engine = bb.build_structural_cosmo({"wU": float(wU), "s2": float(s2), "s3": float(s3), "q3": float(q3)})
    H0 = float(engine["H0_SCFP"])
    Om_t = float(engine["Omega_tot_SCFP"])
    print(f"H0={H0:.6f}, Ωtot={Om_t:.9f}\n")
    add_result(results, "Flatness (Ω_tot≈1)", (abs(Om_t - 1.0) < 1e-3), f"Ω_tot={Om_t:.9f}", tier="HARD")

    # Stage 5: CAMB closure
    section("Stage 5 — CAMB Closure (optional)")
    if args.no_camb:
        print(f"{badge(True)} CAMB skipped by flag.")
        add_result(results, "CAMB closure executed", True, "skipped by --no-camb", tier="EVIDENCE")
    else:
        try:
            bb.run_camb_eb(engine)
            add_result(results, "CAMB closure executed", True, "bb.run_camb_eb ran", tier="EVIDENCE")
        except Exception as e:
            print(f"{badge(False)} CAMB closure failed: {repr(e)}")
            add_result(results, "CAMB closure executed", False, repr(e), tier="EVIDENCE")

    # Stage 6: DOC/DAO Fejér audit
    section("Stage 6 — DOC / Fejér Kernel / DAO Audit (self-contained)")
    doc = fejer_doc_dao_audit(N=64, L=8, eps_psd=1e-12)
    print(f"symmetric_circular={doc['symmetric_circular']}, unit_mass={doc['unit_mass']}, nonneg={doc['nonneg']}, "
          f"PSD={doc['PSD']}, min_hatF={doc['min_hatF']:.3e}, max|Im|={doc['max|Im|']:.3e}\n")

    add_result(results, "DOC/DAO: cyclic Fejér kernel passes",
               (doc["symmetric_circular"] and doc["unit_mass"] and doc["nonneg"] and doc["PSD"] and doc["comm_norm"] == 0.0),
               f"{doc}", tier="HARD")

    # Summary
    hr("Tier‑A Master Demo — Summary Scoreboard", 88)

    hard_fail = False
    for r in results:
        if r.tier == "HARD" and not r.ok:
            hard_fail = True
        tier_tag = "[HARD]" if r.tier == "HARD" else "[EVID]"
        print(f"{badge(r.ok)} {tier_tag} {r.name}")
        if r.detail:
            print(C.dim + f"    {r.detail}" + C.reset)

    hr("", 88)
    if hard_fail:
        print(C.red + "❌ RESULT: FAIL — One or more HARD checks failed." + C.reset)
        sys.exit(1)
    else:
        print(C.grn + "✅ RESULT: PASS — All HARD checks passed. (Evidence tests reported above.)" + C.reset)
        sys.exit(0)


if __name__ == "__main__":
    main()


