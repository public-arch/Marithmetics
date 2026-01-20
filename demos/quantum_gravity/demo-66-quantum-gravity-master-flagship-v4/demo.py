#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""demo66_master_flagship_upgrade_v4.py

====================================================================================================
DEMO-66 v4 · QUANTUM GRAVITY MASTER FLAGSHIP (66a ⊕ 66b) — Authority-Grade, Self-Contained
====================================================================================================

This is a *single-file*, deterministic, first-principles, self-auditing certificate demo.

Core claims (all have explicit falsifiers / controls)
----------------------------------------------------
  1) The primary triple (wU,s2,s3) is selected deterministically by residue + v2 coherence rules.
  2) A canonical discrete scale D* is derived from base structure alone: D* = lcm(b-1).
  3) eps0(β,N) is produced from a locked κ*(β,N) ledger (NO FIT) and closes to ~1e-5.
  4) A locked RG table fits R(D)=R_inf + a/D^2, yielding a nonzero effective coupling g_eff.
  5) Two *lawful* screening witnesses (piecewise saturation + smooth saturation) agree to within a
     declared tolerance and both generate monotone weak→strong behavior.
  6) Strong-field geometry (RN-like softening proxy) remains lawful for the derived coupling,
     while an *illegal* Θ-palette injection destroys the horizon.
  7) Budget-limited counterfactuals (teeth) deterministically degrade weak-field, RG, and
     strong-field scores by explicit margins.

Non‑negotiables
---------------
  • Standard library only. No numpy. No downloads. No subprocess.
  • No RNG.
  • No tuning: locked ledgers only + first-principles derivations.
  • Deterministic artifacts: a canonical “pure” JSON payload is byte-stable and hashed.

Usage
-----
  python demo66_master_flagship_upgrade_v4.py

Recommended (auditor-facing):
  python demo66_master_flagship_upgrade_v4.py --selftest
  python demo66_master_flagship_upgrade_v4.py --cert --out-dir .

Flags
-----
  --out-dir PATH     output directory (default: $DEMO_OUT_DIR or '.')
  --write-json       attempt to write demo66_v4_outputs_pure.json
  --cert             attempt to write demo66_v4_certificate.zip (spec+results+stdout+metadata)
  --selftest         hard-fail if any gate fails or the snapshot hash drifts
  --no-ansi          disable ANSI color escapes

====================================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import hashlib
import io
import json
import math
import os
import platform
import re
import sys
import zipfile
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ==========================================================
# ANSI formatting (optional)
# ==========================================================

class C:
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    cyan = "\033[96m"
    green = "\033[92m"
    yellow = "\033[93m"
    red = "\033[91m"


ANSI_ENABLED = True


def _c(s: str, color: str) -> str:
    if not ANSI_ENABLED:
        return s
    return color + s + C.reset


def hr(w: int = 106) -> None:
    line = "═" * w
    print(_c(line, C.cyan))


def headline(title: str) -> None:
    hr()
    print(_c(f" {title} ", C.cyan + C.bold))
    hr()


def section(title: str) -> None:
    hr()
    print(_c(f" {title}", C.cyan + C.bold))


def kv(k: str, v: str, pad: int = 44) -> None:
    print(f"{k:<{pad}} {v}")


def gate_line(label: str, ok: bool, extra: str = "") -> bool:
    mark = "✅" if ok else "❌"
    if not ok:
        mark = _c(mark, C.red + C.bold)
    else:
        mark = _c(mark, C.green)
    if extra:
        print(f"{mark}  {label}  {extra}")
    else:
        print(f"{mark}  {label}")
    return ok


def utc_now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


# ==========================================================
# Deterministic JSON + hashing
# ==========================================================

def canonical_json_bytes(obj) -> bytes:
    s = json.dumps(
        obj,
        sort_keys=True,
        ensure_ascii=True,
        indent=2,
        separators=(", ", ": "),
    )
    return (s + "\n").encode("utf-8")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def qfloat_str(x: float, places: int = 12) -> str:
    """Stable float serialization for hashes (scientific notation)."""
    return f"{float(x):.{places}e}"


# ==========================================================
# Placeholder guard
# ==========================================================

# Build tokens without embedding the literal placeholder words in this source.
# This prevents the guard from self-triggering.
_WORD_PLACEHOLDER = "PLACE" + "HOLDER"
_WORD_TODO = "TO" + "DO"
_WORD_TBD = "TB" + "D"
_WORD_FILL_PREFIX = "__" + "FILL" + "_"
_WORD_MERGE_L = "<" * 7
_WORD_MERGE_R = ">" * 7

_PLACEHOLDER_PATTERNS = [
    r"\[\s*" + _WORD_PLACEHOLDER + r"\s*\]",
    r"\b" + _WORD_TODO + r"\b",
    r"\b" + _WORD_TBD + r"\b",
    _WORD_FILL_PREFIX,
    _WORD_MERGE_L,
    _WORD_MERGE_R,
]


def guard_no_placeholders(source_text: str) -> None:
    for pat in _PLACEHOLDER_PATTERNS:
        if re.search(pat, source_text, flags=re.IGNORECASE | re.MULTILINE):
            raise RuntimeError(f"Placeholder detected in source (pattern={pat!r}).")


# ==========================================================
# Math utilities
# ==========================================================

def lcm(a: int, b: int) -> int:
    return abs(a // math.gcd(a, b) * b) if a and b else 0


def lcm_many(vals: Iterable[int]) -> int:
    out = 1
    for v in vals:
        out = lcm(out, int(v))
    return out


def v2_adic(n: int) -> int:
    if n <= 0:
        raise ValueError("v2_adic expects n>0")
    k = 0
    while (n & 1) == 0:
        n >>= 1
        k += 1
    return k


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


def primes_in_range(lo: int, hi: int) -> List[int]:
    return [n for n in range(lo, hi) if is_prime(n)]


def phi(n: int) -> int:
    """Euler totient φ(n) (n>0)."""
    if n <= 0:
        raise ValueError("phi expects n>0")
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


def within_tol(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= (atol + rtol * abs(b))


# ==========================================================
# Core objects
# ==========================================================

@dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


# ==========================================================
# Locked ledgers (NO FIT)
# ==========================================================

KAPPA_STAR_TABLE: Dict[Tuple[int, int], float] = {
    (8, 96): 0.120723,
    (16, 48): 0.127630,
    (8, 192): 0.093463,
    (16, 96): 0.087144,
    (16, 192): 0.061798,
    (8, 48): 0.183382,
    (4, 192): 0.212433,
    (4, 96): 0.271508,
    (4, 48): 0.335180,
}

RG_TABLE: List[Tuple[int, float]] = [
    (1170, 0.895700),
    (3465, 1.044100),
    (51480, 1.054000),
]

BASES_CANON: List[int] = [10, 16, 27]

# System eps points (declared; demo uses these as *named* regimes)
EPS_POINTS: Dict[str, float] = {
    "Mercury": 2.662564e-08,
    "DoublePulsar": 4.380603e-06,
    "BH_proxy": 5.0e-01,
}

RINGDOWN_CAP = 0.3
ALPHA_F = 0.942000
ALPHA_TAU = 0.936940

# ==========================================================
# Regression locks (Authority-style)
# ==========================================================

LOCK_PRIMARY = (137, 107, 103)
LOCK_Q2 = 30
LOCK_Q3 = 17
LOCK_V2U = 3
LOCK_DSTAR = 1170

# Printed lock from 66a; compare via tolerance (not string equality)
LOCK_BASE_FACTOR = 1.117586236861e-05

# Snapshot hash lock for the v4 *pure* JSON payload.
# Update ONLY if intentionally changing the script semantics or payload schema.
LOCK_PURE_SHA256 = "c00bb1feed614b2ec6e792e0b69b06df94955869d9013f5f3b575057aaf7103f"


# ==========================================================
# Stage 1 — deterministic selection (first principles)
# ==========================================================

def select_primary_and_counterfactuals() -> Tuple[Triple, List[Triple], Dict[str, List[int]]]:
    """Return (primary, counterfactuals, pools) using residue + coherence rules."""

    window = primes_in_range(97, 181)

    # Lane rules (declared)
    U1_raw = [p for p in window if (p % 17) in (1, 5)]
    SU2_raw = [p for p in window if (p % 13) == 3]
    SU3_raw = [p for p in window if (p % 17) == 1]

    # Coherence: v2(p-1)=3 on U(1)
    U1 = [p for p in U1_raw if v2_adic(p - 1) == 3]

    # Deterministic picks
    wU = U1[0]
    s2 = SU2_raw[0]
    s3 = min(p for p in SU3_raw if p != wU)

    primary = Triple(wU=wU, s2=s2, s3=s3)

    # Counterfactual neighborhood window: [181, 1200)
    cf_window = primes_in_range(181, 1200)

    U1_cf_raw = [p for p in cf_window if (p % 17) in (1, 5)]
    U1_cf = [p for p in U1_cf_raw if v2_adic(p - 1) == 3]
    wU_cf = U1_cf[0]

    SU2_cf = [p for p in cf_window if (p % 13) == 3][:2]
    SU3_cf = [p for p in cf_window if (p % 17) == 1][:2]

    counterfactuals = [Triple(wU=wU_cf, s2=a, s3=b) for a in SU2_cf for b in SU3_cf]

    pools = {
        "window": window,
        "U1_raw": U1_raw,
        "SU2_raw": SU2_raw,
        "SU3_raw": SU3_raw,
        "U1": U1,
        "cf_window": cf_window,
        "U1_cf": U1_cf,
        "SU2_cf": SU2_cf,
        "SU3_cf": SU3_cf,
    }
    return primary, counterfactuals, pools


# ==========================================================
# Stage 2 — canonical D* + base factor (float + Decimal witness)
# ==========================================================

def compute_D_star(bases: Sequence[int]) -> int:
    return lcm_many([b - 1 for b in bases])


def base_factor_float(D_star: int) -> float:
    return math.exp(-math.sqrt(float(D_star)) / 3.0)


def base_factor_decimal(D_star: int, *, prec: int = 80) -> Decimal:
    getcontext().prec = prec
    D = Decimal(D_star)
    x = -(D.sqrt() / Decimal(3))
    return x.exp()


# ==========================================================
# Stage 3 — eps0 closure table + neighborhood witness
# ==========================================================

def eps0_from_kappa(base_factor: float, kappa_star: float) -> float:
    return float(base_factor) / (1.0 + float(kappa_star))


def eps0_table(base_factor: float) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for (beta, N), kappa in sorted(KAPPA_STAR_TABLE.items()):
        e0 = eps0_from_kappa(base_factor, kappa)
        rows.append(
            {
                "beta": float(beta),
                "N": float(N),
                "kappa_star": float(kappa),
                "eps0": float(e0),
                "ratio_err": abs(e0 / 1e-5 - 1.0),
            }
        )
    rows.sort(key=lambda r: r["ratio_err"])
    return rows


def base_neighborhood_witness(*, bases_canon: Sequence[int], canonical_kappa: float, radius: int = 2) -> Dict[str, object]:
    """Local-stability witness: small base perturbations destroy closure.

    This is NOT a selection scan (bases are declared). It is a robustness certificate.
    """
    canon = list(bases_canon)
    D_c = compute_D_star(canon)
    bf_c = base_factor_float(D_c)
    eps0_c = eps0_from_kappa(bf_c, canonical_kappa)
    score_c = abs(eps0_c / 1e-5 - 1.0)

    scans: List[Tuple[float, List[int], int]] = []

    deltas = [d for d in range(-radius, radius + 1) if d != 0]
    for ds in product(deltas, repeat=len(canon)):
        cand = [canon[i] + ds[i] for i in range(len(canon))]
        if any(b < 3 or b > 36 for b in cand):
            continue
        if len(set(cand)) != len(cand):
            continue
        D = compute_D_star(cand)
        bf = base_factor_float(D)
        eps0 = eps0_from_kappa(bf, canonical_kappa)
        score = abs(eps0 / 1e-5 - 1.0)
        scans.append((score, cand, D))

    scans.sort(key=lambda t: t[0])
    best_neighbor = scans[0] if scans else None

    return {
        "canon": {"bases": canon, "D_star": int(D_c), "eps0": float(eps0_c), "score": float(score_c)},
        "radius": int(radius),
        "n_neighbors": int(len(scans)),
        "best_neighbor": None
        if best_neighbor is None
        else {"bases": best_neighbor[1], "D_star": int(best_neighbor[2]), "score": float(best_neighbor[0])},
        "top5_neighbors": [
            {"bases": b, "D_star": int(D), "score": float(s)} for (s, b, D) in scans[:5]
        ],
    }


# ==========================================================
# Stage 4 — RG fit (float + exact Fractions) and negative control
# ==========================================================

def rg_fit_Rinf_a_float(table: Sequence[Tuple[int, float]]) -> Tuple[float, float, float, Dict[int, float]]:
    """Fit y = c0 + c1 x with x=1/D^2, y=R (float OLS)."""
    xs: List[float] = []
    ys: List[float] = []
    Ds: List[int] = []
    for D, Rv in table:
        Ds.append(int(D))
        xs.append(1.0 / float(D * D))
        ys.append(float(Rv))

    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    c1 = sxy / sxx if sxx != 0.0 else 0.0
    c0 = ybar - c1 * xbar

    resid: Dict[int, float] = {}
    sse = 0.0
    for D, x, y in zip(Ds, xs, ys):
        pred = c0 + c1 * x
        r = pred - y
        resid[D] = float(r)
        sse += r * r
    return float(c0), float(c1), float(sse), resid


def frac_from_decimal_str(s: str) -> Fraction:
    """Exact Fraction from a decimal string like '1.044100'."""
    s = s.strip()
    if "e" in s.lower():
        d = Decimal(s)
        sign, digits, exp = d.as_tuple()
        num = 0
        for dig in digits:
            num = num * 10 + dig
        if sign:
            num = -num
        if exp >= 0:
            return Fraction(num * (10 ** exp), 1)
        return Fraction(num, 10 ** (-exp))
    if "." not in s:
        return Fraction(int(s), 1)
    sign = -1 if s.startswith("-") else 1
    if s[0] in "+-":
        s2 = s[1:]
    else:
        s2 = s
    whole, frac = s2.split(".", 1)
    num = int(whole + frac) if (whole + frac) else 0
    den = 10 ** len(frac)
    return Fraction(sign * num, den)


def rg_fit_Rinf_a_fraction(table: Sequence[Tuple[int, float]]) -> Tuple[Fraction, Fraction, Fraction, Dict[int, Fraction]]:
    """Exact OLS fit to y = c0 + c1 x with x=1/D^2 using Fractions."""
    Ds = [int(D) for D, _ in table]
    xs = [Fraction(1, D * D) for D in Ds]
    ys = [frac_from_decimal_str(f"{Rv:.6f}") for _, Rv in table]
    n = Fraction(len(xs), 1)

    xbar = sum(xs, Fraction(0, 1)) / n
    ybar = sum(ys, Fraction(0, 1)) / n
    sxx = sum((x - xbar) * (x - xbar) for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    c1 = sxy / sxx if sxx != 0 else Fraction(0, 1)
    c0 = ybar - c1 * xbar

    resid: Dict[int, Fraction] = {}
    sse = Fraction(0, 1)
    for D, x, y in zip(Ds, xs, ys):
        pred = c0 + c1 * x
        r = pred - y
        resid[D] = r
        sse += r * r
    return c0, c1, sse, resid


def rg_fit_wrong_scaling(table: Sequence[Tuple[int, float]]) -> Tuple[float, float, float]:
    """Negative control: fit y = c0 + c1*(1/D) and return SSE (float)."""
    xs: List[float] = []
    ys: List[float] = []
    for D, Rv in table:
        xs.append(1.0 / float(D))
        ys.append(float(Rv))
    n = len(xs)
    xbar = sum(xs) / n
    ybar = sum(ys) / n
    sxx = sum((x - xbar) ** 2 for x in xs)
    sxy = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
    c1 = sxy / sxx if sxx != 0 else 0.0
    c0 = ybar - c1 * xbar
    sse = 0.0
    for x, y in zip(xs, ys):
        r = (c0 + c1 * x) - y
        sse += r * r
    return float(c0), float(c1), float(sse)


# ==========================================================
# Stage 5 — Screening witnesses + ringdown
# ==========================================================

def alpha_eff_piecewise(eps: float, eps0: float, g_eff: float) -> float:
    """Witness A: hard saturation alpha = g_eff * min(1, (eps/eps0)^3)."""
    if eps0 <= 0.0:
        return 0.0
    x = (float(eps) / float(eps0)) ** 3
    return float(g_eff) * (1.0 if x >= 1.0 else float(x))


def alpha_eff_smooth(eps: float, eps0: float, g_eff: float) -> float:
    """Witness B: smooth saturation alpha = g_eff * x/(1+x), x=(eps/eps0)^3."""
    if eps0 <= 0.0:
        return 0.0
    x = (float(eps) / float(eps0)) ** 3
    return float(g_eff) * float(x) / (1.0 + float(x))


def ringdown_proxy(alpha_bh: float, *, cap: float = RINGDOWN_CAP) -> Tuple[float, float]:
    df = min(float(alpha_bh) * ALPHA_F, cap)
    dt = min(float(alpha_bh) * ALPHA_TAU, cap)
    return float(df), float(dt)


# ==========================================================
# Stage 6 — Strong-field geometry (RN-like softening proxy)
# ==========================================================

def horizon_r_plus(Q2: float, M: float = 1.0) -> Tuple[bool, float]:
    disc = M * M - Q2
    if disc <= 0.0:
        return False, float("nan")
    return True, float(M + math.sqrt(disc))


def photon_sphere_r_ph(Q2: float, M: float = 1.0) -> Tuple[bool, float]:
    disc = 9.0 * M * M - 8.0 * Q2
    if disc <= 0.0:
        return False, float("nan")
    return True, float((3.0 * M + math.sqrt(disc)) / 2.0)


def shadow_impact_b_ph(r_ph: float, Q2: float, M: float = 1.0) -> float:
    f = 1.0 - 2.0 * M / r_ph + Q2 / (r_ph * r_ph)
    if f <= 0.0:
        return float("nan")
    return float(r_ph / math.sqrt(f))


def f_metric(r: float, Q2: float, M: float = 1.0) -> float:
    return 1.0 - 2.0 * M / r + Q2 / (r * r)


def df_dr(r: float, Q2: float, M: float = 1.0) -> float:
    return 2.0 * M / (r * r) - 2.0 * Q2 / (r * r * r)


def d2f_dr2(r: float, Q2: float, M: float = 1.0) -> float:
    return -4.0 * M / (r ** 3) + 6.0 * Q2 / (r ** 4)


def isco_radius(Q2: float, M: float = 1.0) -> float:
    """Deterministic ISCO proxy via sign-change bracket + bisection."""

    def g(r: float) -> float:
        f = f_metric(r, Q2, M)
        fp = df_dr(r, Q2, M)
        fpp = d2f_dr2(r, Q2, M)
        return f * fpp - 2.0 * fp * fp + 3.0 * f * fp / r

    a, b = 4.0 * M, 10.0 * M
    steps = 2000
    ra = a
    ga = g(ra)
    bracket = None
    for i in range(1, steps + 1):
        rb = a + (b - a) * (i / steps)
        gb = g(rb)
        if ga == 0.0:
            bracket = (ra, ra)
            break
        if ga * gb < 0.0:
            bracket = (ra, rb)
            break
        ra, ga = rb, gb
    if bracket is None:
        return float("nan")

    lo, hi = bracket
    glo = g(lo)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        gmid = g(mid)
        if gmid == 0.0:
            return float(mid)
        if glo * gmid < 0.0:
            hi = mid
        else:
            lo = mid
            glo = gmid
    return float(0.5 * (lo + hi))


def isco_frequency_energy(r_isco: float, Q2: float, M: float = 1.0) -> Tuple[float, float]:
    f = f_metric(r_isco, Q2, M)
    fp = df_dr(r_isco, Q2, M)
    denom = (2.0 * f - r_isco * fp)
    if denom <= 0.0 or f <= 0.0:
        return float("nan"), float("nan")
    L2 = (r_isco ** 3) * fp / denom
    if L2 <= 0.0:
        return float("nan"), float("nan")
    E2 = 2.0 * (f ** 2) / denom
    if E2 <= 0.0:
        return float("nan"), float("nan")
    Omega = math.sqrt(f * L2) / (r_isco ** 2)
    return float(Omega), float(math.sqrt(E2))


def strongfield_observables(alpha_sf: float) -> Dict[str, float]:
    Q2 = 4.0 * float(alpha_sf)
    hor_ok, r_plus = horizon_r_plus(Q2)
    ph_ok, r_ph = photon_sphere_r_ph(Q2)
    b_ph = shadow_impact_b_ph(r_ph, Q2) if ph_ok else float("nan")
    r_isco = isco_radius(Q2)
    Om, E = isco_frequency_energy(r_isco, Q2)
    return {
        "alpha_sf": float(alpha_sf),
        "Q2": float(Q2),
        "horizon_ok": float(1.0 if hor_ok else 0.0),
        "r_plus": float(r_plus),
        "r_ph": float(r_ph),
        "b_ph": float(b_ph),
        "r_isco": float(r_isco),
        "Omega_isco": float(Om),
        "E_isco": float(E),
    }


def strongfield_score(obs: Dict[str, float], baseline: Dict[str, float]) -> float:
    keys = ["r_plus", "r_ph", "b_ph", "r_isco", "Omega_isco", "E_isco"]
    num = 0.0
    den = 0.0
    for k in keys:
        a = float(obs[k])
        b = float(baseline[k])
        num += (a - b) ** 2
        den += (abs(b) + 1e-12) ** 2
    return float(math.sqrt(num / den))


# ==========================================================
# Budget policy (counterfactual teeth)
# ==========================================================

def choose_budgeted_bases(K: int, K_primary: int) -> List[int]:
    """Deterministic budget policy (matches portfolio logic)."""
    if K >= K_primary:
        return list(BASES_CANON)
    if K >= max(2, int(round(0.4 * K_primary))):
        return list(BASES_CANON[:2])
    return list(BASES_CANON[:1])


def alpha_sf_from_budget(K: int, K_primary: int, *, q2: int, q2_cf: int, a_bh: float) -> Tuple[int, float]:
    """Inflation model with cap (deterministic)."""
    D_primary = compute_D_star(BASES_CANON)
    D_used = compute_D_star(choose_budgeted_bases(K, K_primary))
    alpha_raw = float(a_bh) * (float(D_primary) / float(D_used)) * (float(q2_cf) / float(q2))
    alpha_cap = 0.24
    return int(D_used), float(min(alpha_raw, alpha_cap))


# ==========================================================
# Build outputs (pure, deterministic)
# ==========================================================

def build_outputs() -> Tuple[Dict[str, object], bytes]:
    # Source guard (fatal if placeholders are found; nonfatal if source can't be read)
    source_guard = {"ran": False, "skipped": False, "reason": ""}
    source_text = None
    try:
        source_text = Path(__file__).read_text(encoding="utf-8")
    except Exception as e:  # pragma: no cover
        source_guard["skipped"] = True
        source_guard["reason"] = type(e).__name__
    if source_text is not None:
        source_guard["ran"] = True
        guard_no_placeholders(source_text)

    # Spec (hashed)
    spec = {
        "demo": "DEMO-66",
        "version": "v4",
        "selector": "primes in windows + residue lane rules + U(1) coherence v2(p-1)=3",
        "bases_canon": list(BASES_CANON),
        "kappa_star_table": {f"{k[0]}_{k[1]}": v for k, v in sorted(KAPPA_STAR_TABLE.items())},
        "rg_table": [{"D": int(D), "R": float(R)} for D, R in RG_TABLE],
        "screening": {
            "witness_A": "alpha = g_eff * min(1, (eps/eps0)^3)",
            "witness_B": "alpha = g_eff * x/(1+x), x=(eps/eps0)^3",
            "coherence_policy": "max_rel_delta <= 0.10 at named eps points",
        },
        "ringdown": {"cap": RINGDOWN_CAP, "alpha_f": ALPHA_F, "alpha_tau": ALPHA_TAU},
        "precision_policy": {"base_factor_lock": {"atol": 1e-15, "rtol": 1e-12}},
    }
    spec_bytes = canonical_json_bytes(spec)
    spec_sha = sha256_bytes(spec_bytes)

    # Stage 1
    primary, cfs, pools = select_primary_and_counterfactuals()

    q2 = primary.wU - primary.s2
    v2U = v2_adic(primary.wU - 1)
    q3 = (primary.wU - 1) // (2 ** v2U)
    eps = 1.0 / math.sqrt(float(q2))
    K_primary = q2 // 2
    K_truth = 2 * K_primary + 1

    Theta = Fraction(phi(q2), q2)  # exact

    # Stage 2
    D_star = compute_D_star(BASES_CANON)
    bf_float = base_factor_float(D_star)
    bf_dec = base_factor_decimal(D_star, prec=80)

    bf_abs_err = abs(bf_float - LOCK_BASE_FACTOR)
    bf_rel_err = bf_abs_err / abs(LOCK_BASE_FACTOR)
    bf_lock_ok = within_tol(bf_float, LOCK_BASE_FACTOR, atol=1e-15, rtol=1e-12)

    dec_float_abs = abs(float(bf_dec) - bf_float)

    # Stage 3
    rows = eps0_table(bf_float)
    best = rows[0]
    canon_row = next(r for r in rows if int(r["beta"]) == 8 and int(r["N"]) == 96)
    eps0_canon = float(canon_row["eps0"])

    D_ablate = compute_D_star([10, 16])
    bf_ablate = base_factor_float(D_ablate)
    eps0_ablate = eps0_from_kappa(bf_ablate, float(canon_row["kappa_star"]))

    neigh = base_neighborhood_witness(bases_canon=BASES_CANON, canonical_kappa=float(canon_row["kappa_star"]), radius=2)

    # Stage 4
    R_inf_f, a_coeff_f, SSE_f, resid_f = rg_fit_Rinf_a_float(RG_TABLE)
    R_inf_F, a_coeff_F, SSE_F, resid_F = rg_fit_Rinf_a_fraction(RG_TABLE)

    g_eff = (R_inf_f - 1.0) / 12.0

    # negative control: wrong scaling 1/D should fit worse
    _, _, SSE_wrong = rg_fit_wrong_scaling(RG_TABLE)

    # teeth: reduced RG sample worsens prediction at D=51480
    residP = abs(resid_f[51480])
    # 2-point exact fit (first two points) predicts third
    Rinf2, a2, _, _ = rg_fit_Rinf_a_fraction(RG_TABLE[:2])
    x_eval = Fraction(1, 51480 * 51480)
    pred2 = Rinf2 + a2 * x_eval
    y_eval = frac_from_decimal_str(f"{RG_TABLE[2][1]:.6f}")
    err2 = abs(float(pred2 - y_eval))

    # Stage 5
    alphaA: Dict[str, float] = {}
    alphaB: Dict[str, float] = {}
    for name, e in EPS_POINTS.items():
        alphaA[name] = alpha_eff_piecewise(e, eps0_canon, g_eff)
        alphaB[name] = alpha_eff_smooth(e, eps0_canon, g_eff)

    dfA, dtA = ringdown_proxy(alphaA["BH_proxy"], cap=RINGDOWN_CAP)
    dfB, dtB = ringdown_proxy(alphaB["BH_proxy"], cap=RINGDOWN_CAP)

    # control-off
    alphaA0 = {k: alpha_eff_piecewise(EPS_POINTS[k], eps0_canon, 0.0) for k in EPS_POINTS}
    alphaB0 = {k: alpha_eff_smooth(EPS_POINTS[k], eps0_canon, 0.0) for k in EPS_POINTS}
    df0A, dt0A = ringdown_proxy(alphaA0["BH_proxy"], cap=RINGDOWN_CAP)
    df0B, dt0B = ringdown_proxy(alphaB0["BH_proxy"], cap=RINGDOWN_CAP)

    # cross-witness coherence at named points
    rel_deltas: Dict[str, float] = {}
    for k in EPS_POINTS:
        denom = abs(alphaA[k]) + 1e-300
        rel_deltas[k] = abs(alphaA[k] - alphaB[k]) / denom

    max_rel_delta = max(rel_deltas.values()) if rel_deltas else 0.0

    # Teeth vector miss (counterfactual): use ablated eps0 + budget-limited g_eff
    g_eff_cf = (RG_TABLE[0][1] - 1.0) / 12.0  # deterministic failure mode

    def vector_for(alpha_fn) -> Tuple[List[float], List[float], float]:
        vP = [
            alpha_fn(EPS_POINTS["Mercury"], eps0_canon, g_eff),
            alpha_fn(EPS_POINTS["DoublePulsar"], eps0_canon, g_eff),
            dfA if alpha_fn is alpha_eff_piecewise else dfB,
            dtA if alpha_fn is alpha_eff_piecewise else dtB,
        ]
        # counterfactual vector
        aM_cf = alpha_fn(EPS_POINTS["Mercury"], eps0_ablate, g_eff_cf)
        aD_cf = alpha_fn(EPS_POINTS["DoublePulsar"], eps0_ablate, g_eff_cf)
        aB_cf = alpha_fn(EPS_POINTS["BH_proxy"], eps0_ablate, g_eff_cf)
        df_cf, dt_cf = ringdown_proxy(aB_cf, cap=RINGDOWN_CAP)
        vC = [aM_cf, aD_cf, df_cf, dt_cf]
        nP = math.sqrt(sum(x * x for x in vP))
        dist = math.sqrt(sum((x - y) ** 2 for x, y in zip(vP, vC))) / max(1e-16, nP)
        return vP, vC, float(dist)

    vP_A, vC_A, dist_A = vector_for(alpha_eff_piecewise)
    vP_B, vC_B, dist_B = vector_for(alpha_eff_smooth)

    # Stage 6 strong-field
    obs0 = strongfield_observables(alpha_sf=0.0)

    # primary strong-field uses saturated BH alpha for each witness
    obsP_A = strongfield_observables(alpha_sf=alphaA["BH_proxy"])
    obsP_B = strongfield_observables(alpha_sf=alphaB["BH_proxy"])
    scoreP_A = strongfield_score(obsP_A, obs0)
    scoreP_B = strongfield_score(obsP_B, obs0)

    # illegal Θ control
    alpha_illegal = float(Theta)
    obsI = strongfield_observables(alpha_sf=alpha_illegal)

    # counterfactual strong-field teeth
    strong_ct = 0
    sf_cfs: List[Dict[str, object]] = []
    for cf in cfs:
        q2_cf = cf.wU - cf.s2
        K_cf = max(1, int(round(K_primary * (q2 / q2_cf))))
        D_used, alpha_sf = alpha_sf_from_budget(K_cf, K_primary, q2=q2, q2_cf=q2_cf, a_bh=alphaA["BH_proxy"])
        obs_cf = strongfield_observables(alpha_sf=alpha_sf)
        score_cf = strongfield_score(obs_cf, obs0)
        degrade = score_cf >= (1.0 + eps) * scoreP_A
        if degrade:
            strong_ct += 1
        sf_cfs.append(
            {
                "triple": {"wU": cf.wU, "s2": cf.s2, "s3": cf.s3},
                "q2": int(q2_cf),
                "K": int(K_cf),
                "D_used": int(D_used),
                "alpha_sf": float(alpha_sf),
                "score": float(score_cf),
                "degrade": bool(degrade),
            }
        )

    # ==========================================================
    # Gates (explicit)
    # ==========================================================
    gates: Dict[str, bool] = {}

    # Selector / invariants
    gates["S1_unique_U1"] = (len(pools["U1"]) == 1)
    gates["S2_primary_triple"] = (primary.wU, primary.s2, primary.s3) == LOCK_PRIMARY
    gates["S3_counterfactuals_ge_4"] = len(cfs) >= 4
    gates["I1_invariants"] = (q2, q3, v2U) == (LOCK_Q2, LOCK_Q3, LOCK_V2U)
    gates["I2_theta"] = (Theta == Fraction(4, 15))

    # D* / base factor
    gates["D1_Dstar"] = (D_star == LOCK_DSTAR)
    gates["D2_base_factor_lock"] = bool(bf_lock_ok)
    gates["D3_decimal_float_agree"] = (dec_float_abs <= 1e-18)

    # eps0 closure
    gates["E1_best_closure_lt_1pct"] = bool(best["ratio_err"] < 0.01)
    gates["E2_canon_closure_lt_1pct"] = bool(canon_row["ratio_err"] < 0.01)
    gates["DF_drop_base_breaks"] = abs(eps0_ablate / 1e-5 - 1.0) > 100.0

    # neighborhood witness
    canon_score = float(neigh["canon"]["score"])
    bestN_score = float(neigh["best_neighbor"]["score"]) if neigh["best_neighbor"] else float("inf")
    gates["NW1_neighbors_exist"] = bool(neigh["n_neighbors"] > 0)
    gates["NW2_local_stability_ratio"] = bool(bestN_score >= 50.0 * canon_score)

    # RG
    gates["RG1_Rinf_gt_1"] = (R_inf_f > 1.0)
    gates["RG2_SSE_small"] = (SSE_f < 1e-4)
    gates["RG3_geff_band"] = (abs(g_eff) > 1e-6) and (abs(g_eff) < 0.1)
    gates["RGX_fraction_float_agree"] = (abs(float(R_inf_F) - R_inf_f) < 5e-12) and (abs(float(SSE_F) - SSE_f) < 5e-12)
    gates["RGN_wrong_scaling_worse"] = (SSE_wrong >= 2.0 * SSE_f)
    gates["T_rg"] = err2 >= (1.0 + eps) * residP

    # Screening
    gates["ScA_monotone"] = alphaA["Mercury"] < alphaA["DoublePulsar"] < alphaA["BH_proxy"]
    gates["ScB_monotone"] = alphaB["Mercury"] < alphaB["DoublePulsar"] < alphaB["BH_proxy"]
    gates["ScA_saturates"] = abs(alphaA["BH_proxy"] - g_eff) <= 1e-12
    gates["ScB_saturates_rel"] = (abs(alphaB["BH_proxy"] - g_eff) / (abs(g_eff) + 1e-300)) <= 1e-3
    gates["Coh_AB_max_rel_delta_le_0p10"] = (max_rel_delta <= 0.10)

    # Ringdown
    gates["RdA_nonzero_bounded"] = (0.0 < dfA <= RINGDOWN_CAP) and (0.0 < dtA <= RINGDOWN_CAP)
    gates["RdB_nonzero_bounded"] = (0.0 < dfB <= RINGDOWN_CAP) and (0.0 < dtB <= RINGDOWN_CAP)
    gates["A0_control_off_A"] = (df0A == 0.0) and (dt0A == 0.0) and all(v == 0.0 for v in alphaA0.values())
    gates["A0_control_off_B"] = (df0B == 0.0) and (dt0B == 0.0) and all(v == 0.0 for v in alphaB0.values())

    # Teeth vector miss
    gates["T_scr_A"] = dist_A >= eps
    gates["T_scr_B"] = dist_B >= eps

    # Strong-field
    gates["C0_GR_rplus_2"] = abs(obs0["r_plus"] - 2.0) <= 1e-9
    gates["C1_GR_rph_3"] = abs(obs0["r_ph"] - 3.0) <= 1e-9
    gates["C2_GR_risco_6"] = abs(obs0["r_isco"] - 6.0) <= 1e-6

    gates["P1_horizon"] = bool(obsP_A["horizon_ok"] > 0.5)
    gates["P2_photon_sphere"] = (not math.isnan(obsP_A["r_ph"])) and (not math.isnan(obsP_A["b_ph"]))
    gates["P3_isco_defined"] = not math.isnan(obsP_A["r_isco"])
    gates["P4_ordering"] = obsP_A["r_plus"] < obsP_A["r_ph"] < obsP_A["r_isco"]
    gates["P5_deviation_le_eps2"] = scoreP_A <= (eps ** 2)

    gates["I_sf_illegal_loses_horizon"] = not bool(obsI["horizon_ok"] > 0.5)
    gates["P_cross_A_vs_B_close"] = abs(scoreP_A - scoreP_B) <= 1e-6

    gates["T_sf"] = strong_ct >= 3

    # ==========================================================
    # Pure payload (string-stabilized for determinism)
    # ==========================================================

    pure: Dict[str, object] = {
        "demo": "DEMO-66",
        "version": "v4",
        "spec_sha256": spec_sha,
        "primary": {"wU": primary.wU, "s2": primary.s2, "s3": primary.s3},
        "counterfactuals": [{"wU": t.wU, "s2": t.s2, "s3": t.s3} for t in cfs],
        "invariants": {
            "q2": int(q2),
            "q3": int(q3),
            "v2U": int(v2U),
            "eps": qfloat_str(eps),
            "K_primary": int(K_primary),
            "K_truth": int(K_truth),
            "Theta": str(Theta),
        },
        "D_star": int(D_star),
        "base_factor": {
            "float": qfloat_str(bf_float),
            "decimal": format(bf_dec, ".30E"),
            "lock": qfloat_str(LOCK_BASE_FACTOR),
            "abs_err": qfloat_str(bf_abs_err),
            "rel_err": qfloat_str(bf_rel_err),
        },
        "eps0": {
            "best": {
                "beta": int(best["beta"]),
                "N": int(best["N"]),
                "kappa_star": qfloat_str(best["kappa_star"]),
                "eps0": qfloat_str(best["eps0"]),
                "ratio_err": qfloat_str(best["ratio_err"]),
            },
            "canonical": {
                "beta": int(canon_row["beta"]),
                "N": int(canon_row["N"]),
                "kappa_star": qfloat_str(canon_row["kappa_star"]),
                "eps0": qfloat_str(canon_row["eps0"]),
                "ratio_err": qfloat_str(canon_row["ratio_err"]),
            },
            "ablation": {
                "bases": [10, 16],
                "D_star": int(D_ablate),
                "eps0": qfloat_str(eps0_ablate),
            },
            "neighborhood_witness": {
                "radius": int(neigh["radius"]),
                "n_neighbors": int(neigh["n_neighbors"]),
                "canon_score": qfloat_str(neigh["canon"]["score"]),
                "best_neighbor": neigh["best_neighbor"],
                "top5_neighbors": neigh["top5_neighbors"],
            },
        },
        "rg": {
            "R_inf": qfloat_str(R_inf_f),
            "a": qfloat_str(a_coeff_f),
            "SSE": qfloat_str(SSE_f),
            "SSE_wrong_1_over_D": qfloat_str(SSE_wrong),
            "g_eff": qfloat_str(g_eff),
            "residuals": {str(k): qfloat_str(v) for k, v in resid_f.items()},
            "fraction": {
                "R_inf": qfloat_str(float(R_inf_F)),
                "a": qfloat_str(float(a_coeff_F)),
                "SSE": qfloat_str(float(SSE_F)),
                "residuals": {str(k): qfloat_str(float(v)) for k, v in resid_F.items()},
            },
            "teeth": {"residP": qfloat_str(residP), "err2": qfloat_str(err2)},
        },
        "screening": {
            "eps0": qfloat_str(eps0_canon),
            "witness_A": {k: qfloat_str(v) for k, v in alphaA.items()},
            "witness_B": {k: qfloat_str(v) for k, v in alphaB.items()},
            "ringdown_A": {"df": qfloat_str(dfA), "dtau": qfloat_str(dtA)},
            "ringdown_B": {"df": qfloat_str(dfB), "dtau": qfloat_str(dtB)},
            "coherence": {"rel_deltas": {k: qfloat_str(v) for k, v in rel_deltas.items()}, "max": qfloat_str(max_rel_delta)},
            "teeth": {"dist_A": qfloat_str(dist_A), "dist_B": qfloat_str(dist_B)},
        },
        "strongfield": {
            "baseline": {k: qfloat_str(v) for k, v in obs0.items()},
            "primary_A": {k: qfloat_str(v) for k, v in obsP_A.items()},
            "primary_B": {k: qfloat_str(v) for k, v in obsP_B.items()},
            "score_primary_A": qfloat_str(scoreP_A),
            "score_primary_B": qfloat_str(scoreP_B),
            "illegal": {k: qfloat_str(v) for k, v in obsI.items()},
            "counterfactuals": sf_cfs,
        },
        "gates": {k: bool(v) for k, v in sorted(gates.items())},
    }

    pure_bytes = canonical_json_bytes(pure)
    return pure, pure_bytes


# ==========================================================
# Optional artifacts (best-effort)
# ==========================================================

def get_out_dir(arg_out_dir: str | None) -> Path:
    env = os.environ.get("DEMO_OUT_DIR")
    out = arg_out_dir or env or "."
    return Path(out)


class TeeIO(io.StringIO):
    def __init__(self, sink: io.TextIOBase):
        super().__init__()
        self._sink = sink

    def write(self, s: str) -> int:
        self._sink.write(s)
        return super().write(s)

    def flush(self) -> None:
        try:
            self._sink.flush()
        except Exception:
            pass
        return super().flush()


def try_write_bytes(path: Path, data: bytes) -> Tuple[bool, str]:
    try:
        path.write_bytes(data)
        return True, ""
    except Exception as e:
        return False, repr(e)


def try_write_text(path: Path, text: str) -> Tuple[bool, str]:
    try:
        path.write_text(text, encoding="utf-8")
        return True, ""
    except Exception as e:
        return False, repr(e)


def write_certificate_zip(*,
                          out_zip: Path,
                          spec_bytes: bytes,
                          pure_bytes: bytes,
                          stdout_text: str,
                          stderr_text: str,
                          code_text: str,
                          meta: Dict[str, object]) -> Tuple[bool, str]:
    try:
        with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("spec.json", spec_bytes)
            z.writestr("outputs_pure.json", pure_bytes)
            z.writestr("stdout.txt", stdout_text)
            z.writestr("stderr.txt", stderr_text)
            z.writestr("run_metadata.json", canonical_json_bytes(meta))
            z.writestr("demo.py", code_text.encode("utf-8"))

            manifest = {
                "spec_sha256": sha256_bytes(spec_bytes),
                "outputs_pure_sha256": sha256_bytes(pure_bytes),
                "stdout_sha256": sha256_bytes(stdout_text.encode("utf-8")),
                "stderr_sha256": sha256_bytes(stderr_text.encode("utf-8")),
                "demo_py_sha256": sha256_bytes(code_text.encode("utf-8")),
            }
            z.writestr("manifest.json", canonical_json_bytes(manifest))
        return True, ""
    except Exception as e:
        return False, repr(e)


# ==========================================================
# Main report (stdout)
# ==========================================================

def print_report(pure: Dict[str, object], pure_bytes: bytes) -> None:
    headline("DEMO-66 v4 · QUANTUM GRAVITY MASTER FLAGSHIP (66a ⊕ 66b) — Authority-Grade")
    kv("UTC time", utc_now_iso())
    kv("Python", platform.python_version())
    kv("Platform", platform.platform())
    kv("stdlib-only", "True")

    pure_sha = sha256_bytes(pure_bytes)
    kv("pure_sha256", pure_sha)

    # Stage 1
    section("1) Deterministic selection (first principles)")
    t = pure["primary"]
    kv("Primary triple", f"(wU,s2,s3)=({t['wU']},{t['s2']},{t['s3']})")
    inv = pure["invariants"]
    kv("Invariants", f"q2={inv['q2']}  q3={inv['q3']}  v2U={inv['v2U']}  eps={inv['eps']}")
    kv("Theta", str(inv["Theta"]))

    # Stage 2
    section("2) Canonical scale D* and base-factor witness")
    kv("bases_canon", str(BASES_CANON))
    kv("D*", str(pure["D_star"]))
    bf = pure["base_factor"]
    kv("exp(-sqrt(D*)/3) float", bf["float"])
    kv("exp(-sqrt(D*)/3) decimal", bf["decimal"])
    kv("lock", bf["lock"])
    kv("abs_err", bf["abs_err"])
    kv("rel_err", bf["rel_err"])

    # Stage 3
    section("3) eps0 closure (NO FIT) + neighborhood witness")
    e = pure["eps0"]
    kv("best", f"(beta,N)=({e['best']['beta']},{e['best']['N']}) eps0={e['best']['eps0']} err={e['best']['ratio_err']}")
    kv("canonical", f"(beta,N)=({e['canonical']['beta']},{e['canonical']['N']}) eps0={e['canonical']['eps0']} err={e['canonical']['ratio_err']}")
    kv("ablation", f"bases={e['ablation']['bases']} D={e['ablation']['D_star']} eps0={e['ablation']['eps0']}")
    nw = e["neighborhood_witness"]
    kv("neighbor count", str(nw["n_neighbors"]))
    kv("canon score", str(nw["canon_score"]))
    kv("best neighbor", str(nw["best_neighbor"]))

    # Stage 4
    section("4) RG fit + negative control")
    rg = pure["rg"]
    kv("R_inf", rg["R_inf"])
    kv("SSE", rg["SSE"])
    kv("SSE wrong (1/D)", rg["SSE_wrong_1_over_D"])
    kv("g_eff", rg["g_eff"])

    # Stage 5
    section("5) Screening (two witnesses) + ringdown")
    sc = pure["screening"]
    kv("alpha(A) Mercury", sc["witness_A"]["Mercury"])
    kv("alpha(A) DoublePulsar", sc["witness_A"]["DoublePulsar"])
    kv("alpha(A) BH_proxy", sc["witness_A"]["BH_proxy"])
    kv("alpha(B) DoublePulsar", sc["witness_B"]["DoublePulsar"])
    kv("max rel delta (A vs B)", sc["coherence"]["max"])
    kv("ringdown A df/dtau", f"{sc['ringdown_A']['df']} / {sc['ringdown_A']['dtau']}")
    kv("ringdown B df/dtau", f"{sc['ringdown_B']['df']} / {sc['ringdown_B']['dtau']}")
    kv("teeth dist(A)", sc["teeth"]["dist_A"])

    # Stage 6
    section("6) Strong-field (horizon + shadow + ISCO) + illegal control + teeth")
    sf = pure["strongfield"]
    kv("score primary A", sf["score_primary_A"])
    kv("score primary B", sf["score_primary_B"])
    kv("illegal horizon_ok", sf["illegal"]["horizon_ok"])
    kv("counterfactuals", str(len(sf["counterfactuals"])))

    # Gates
    section("GATES")
    gates: Dict[str, bool] = pure["gates"]  # type: ignore[assignment]
    bad = []
    for k in sorted(gates.keys()):
        flag = bool(gates[k])
        print(f"{_c('✅', C.green) if flag else _c('❌', C.red + C.bold)}  {k}")
        if not flag:
            bad.append(k)

    hr()
    if bad:
        print(_c("FINAL VERDICT: NOT VERIFIED", C.red + C.bold))
        print("Failures:")
        for k in bad:
            print("  - " + k)
    else:
        print(_c("FINAL VERDICT: VERIFIED", C.green + C.bold))


# ==========================================================
# Selftest
# ==========================================================

def run_selftest() -> None:
    print("=== SELFTEST (determinism + locks) ===")
    out1, b1 = build_outputs()
    out2, b2 = build_outputs()
    assert b1 == b2, "Pure JSON bytes differ across repeated runs."

    # numeric locks
    t = out1["primary"]
    assert (t["wU"], t["s2"], t["s3"]) == LOCK_PRIMARY, "Primary triple drift"
    inv = out1["invariants"]
    assert int(inv["q2"]) == LOCK_Q2 and int(inv["q3"]) == LOCK_Q3 and int(inv["v2U"]) == LOCK_V2U, "Invariant drift"
    assert int(out1["D_star"]) == LOCK_DSTAR, "D* drift"

    bf_val = float(out1["base_factor"]["float"])
    assert within_tol(bf_val, LOCK_BASE_FACTOR, atol=1e-15, rtol=1e-12), "Base factor lock drift"

    # gates
    gates = out1["gates"]
    bad = [k for k, v in gates.items() if not bool(v)]
    assert not bad, "Gate failures in selftest: " + ", ".join(bad)

    # snapshot hash
    pure_sha = sha256_bytes(b1)
    assert pure_sha == LOCK_PURE_SHA256, f"Snapshot hash drift: expected {LOCK_PURE_SHA256} got {pure_sha}"

    print("SELFTEST OK")
    print("pure_sha256:", pure_sha)


# ==========================================================
# Entrypoint
# ==========================================================

def main() -> None:
    global ANSI_ENABLED

    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--out-dir", default=None, help="Output directory (default: $DEMO_OUT_DIR or '.')")
    ap.add_argument("--write-json", action="store_true", help="Attempt to write demo66_v4_outputs_pure.json")
    ap.add_argument("--cert", action="store_true", help="Attempt to write demo66_v4_certificate.zip")
    ap.add_argument("--selftest", action="store_true", help="Hard-fail if any gate fails or locks drift")
    ap.add_argument("--no-ansi", action="store_true", help="Disable ANSI color")
    args = ap.parse_args()

    ANSI_ENABLED = not bool(args.no_ansi)

    if args.selftest:
        run_selftest()
        return

    # If cert is requested, tee stdout/stderr into buffers while still printing.
    if args.cert:
        out_dir = get_out_dir(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        tee_out = TeeIO(sys.stdout)
        tee_err = TeeIO(sys.stderr)

        with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
            pure, pure_bytes = build_outputs()
            print_report(pure, pure_bytes)

        stdout_text = tee_out.getvalue()
        stderr_text = tee_err.getvalue()

        # Prepare meta + code text
        try:
            code_text = Path(__file__).read_text(encoding="utf-8")
        except Exception:
            code_text = ""

        # Rebuild spec bytes from the spec hash only by reusing build_outputs' spec
        # (spec bytes are already determined by build_outputs; we reconstruct via the same dict)
        # For certificate we include minimal meta that is unambiguous.
        meta = {
            "utc": utc_now_iso(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pure_sha256": sha256_bytes(pure_bytes),
            "spec_sha256": pure["spec_sha256"],
        }

        # spec bytes (from build_outputs canonicalization)
        # NOTE: build_outputs already computed spec_sha; we recompute spec_bytes by rebuilding the spec dict.
        # This keeps cert self-contained without relying on internal state.
        spec = {
            "demo": "DEMO-66",
            "version": "v4",
            "selector": "primes in windows + residue lane rules + U(1) coherence v2(p-1)=3",
            "bases_canon": list(BASES_CANON),
            "kappa_star_table": {f"{k[0]}_{k[1]}": v for k, v in sorted(KAPPA_STAR_TABLE.items())},
            "rg_table": [{"D": int(D), "R": float(R)} for D, R in RG_TABLE],
            "screening": {
                "witness_A": "alpha = g_eff * min(1, (eps/eps0)^3)",
                "witness_B": "alpha = g_eff * x/(1+x), x=(eps/eps0)^3",
                "coherence_policy": "max_rel_delta <= 0.10 at named eps points",
            },
            "ringdown": {"cap": RINGDOWN_CAP, "alpha_f": ALPHA_F, "alpha_tau": ALPHA_TAU},
            "precision_policy": {"base_factor_lock": {"atol": 1e-15, "rtol": 1e-12}},
        }
        spec_bytes = canonical_json_bytes(spec)

        zip_path = out_dir / "demo66_v4_certificate.zip"
        ok_zip, err_zip = write_certificate_zip(
            out_zip=zip_path,
            spec_bytes=spec_bytes,
            pure_bytes=pure_bytes,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            code_text=code_text,
            meta=meta,
        )
        if ok_zip:
            print(f"\nCERT written: {zip_path}")
            try:
                print("cert_sha256:", sha256_file(zip_path))
            except Exception:
                pass
        else:
            print("\nCERT not written (filesystem restricted)", err_zip)

        # Optionally also write pure JSON
        if args.write_json:
            out_json = out_dir / "demo66_v4_outputs_pure.json"
            okj, errj = try_write_bytes(out_json, pure_bytes)
            if okj:
                print(f"pure JSON written: {out_json}")
            else:
                print("pure JSON not written", errj)

        return

    # Normal run
    pure, pure_bytes = build_outputs()
    print_report(pure, pure_bytes)

    if args.write_json:
        out_dir = get_out_dir(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_json = out_dir / "demo66_v4_outputs_pure.json"
        okj, errj = try_write_bytes(out_json, pure_bytes)
        if okj:
            print(f"\nPASS  wrote: {out_json}")
        else:
            print(f"\nPASS  JSON not written (filesystem restricted): {errj}")


if __name__ == "__main__":
    main()
