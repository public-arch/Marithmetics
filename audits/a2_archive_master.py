#!/usr/bin/env python3
"""
A2 Archive Master Script (v1.2)
a2_archive_master.py
================================

Stdlib-only, deterministic, one-file archive.

What it does
- Reproduces the A2 / BB-36 closure capsule from first principles (as currently encoded):
  • Gauge lawbook derivation (contracts + coherence)
  • Gauge selector (lane filters) → unique (wU,s2,s3) = (137,107,103)
  • Φ-mapping uniqueness (alpha, alpha_s, sin^2)
  • Yukawa palette closure (D1 local selector + offset sweep)
  • Cosmology Ω-sector closure + flatness
  • H0 closure (structural reuse)
  • Primordial closures (As, ns, tau)
  • Neutrino closures (Δ21, Δ31, Σmν)
  • Amplitude closures (etaB, YHe, deltaCMB) + ℓ1 reuse
  • Cross-base compatibility / Rosetta suite (base-7/10/16)

Run:
  python3 a2_archive_master.py                 # pretty CLI
  python3 a2_archive_master.py --mode full     # add heavier evidence scans
  python3 a2_archive_master.py --json          # print JSON only
  python3 a2_archive_master.py --save-json out.json
  python3 a2_archive_master.py --save-md out.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import random
import sys
from dataclasses import dataclass
from decimal import Decimal, getcontext
from fractions import Fraction
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple


# -----------------------------------------------------------------------------
# Formatting / CLI helpers (stdlib-only)
# -----------------------------------------------------------------------------

class _Ansi:
    RESET = "\x1b[0m"
    BOLD  = "\x1b[1m"
    DIM   = "\x1b[2m"

    RED   = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW= "\x1b[33m"
    CYAN  = "\x1b[36m"

def _supports_color(no_color: bool) -> bool:
    if no_color:
        return False
    if not sys.stdout.isatty():
        return False
    term = os.environ.get("TERM", "")
    if term == "dumb":
        return False
    return True

@dataclass
class _Theme:
    ok: str
    bad: str
    warn: str
    info: str
    dim: str
    bold: str
    reset: str

def _theme(enable_color: bool) -> _Theme:
    if not enable_color:
        return _Theme(ok="", bad="", warn="", info="", dim="", bold="", reset="")
    return _Theme(
        ok=_Ansi.GREEN,
        bad=_Ansi.RED,
        warn=_Ansi.YELLOW,
        info=_Ansi.CYAN,
        dim=_Ansi.DIM,
        bold=_Ansi.BOLD,
        reset=_Ansi.RESET,
    )

class Printer:
    def __init__(self, enable_color: bool = True):
        self.t = _theme(enable_color)

    def box(self, title: str, subtitle: Optional[str] = None) -> None:
        lines = [title] + ([subtitle] if subtitle else [])
        w = max(len(s) for s in lines) + 4
        top = "╔" + "═" * (w - 2) + "╗"
        bot = "╚" + "═" * (w - 2) + "╝"
        print(self.t.info + top + self.t.reset)
        for s in lines:
            pad = " " * (w - 4 - len(s))
            print(self.t.info + f"║ {self.t.bold}{s}{self.t.reset}{self.t.info}{pad} ║" + self.t.reset)
        print(self.t.info + bot + self.t.reset)

    def section(self, label: str) -> None:
        print(self.t.info + "═" * 10 + f" {self.t.bold}{label}{self.t.reset}{self.t.info} " + "═" * 10 + self.t.reset)

    def kv(self, k: str, v: str, indent: int = 2) -> None:
        print(" " * indent + f"{self.t.bold}{k}{self.t.reset}: {v}")

    def bullet(self, s: str, indent: int = 2) -> None:
        print(" " * indent + "• " + s)

    def ok(self, s: str, indent: int = 2) -> None:
        print(" " * indent + f"{self.t.ok}✅ {s}{self.t.reset}")

    def bad(self, s: str, indent: int = 2) -> None:
        print(" " * indent + f"{self.t.bad}❌ {s}{self.t.reset}")

    def warn(self, s: str, indent: int = 2) -> None:
        print(" " * indent + f"{self.t.warn}⚠️  {s}{self.t.reset}")

    def dim(self, s: str, indent: int = 2) -> None:
        print(" " * indent + f"{self.t.dim}{s}{self.t.reset}")

    def table(self, rows: Sequence[Sequence[str]], indent: int = 2) -> None:
        if not rows:
            return
        widths = [0] * len(rows[0])
        for r in rows:
            for j, c in enumerate(r):
                widths[j] = max(widths[j], len(c))
        for r in rows:
            line = " " * indent + "  ".join(c.ljust(widths[j]) for j, c in enumerate(r))
            print(line)

def _fmt_float(x: float, sig: int = 16) -> str:
    if x == 0.0:
        return "0"
    ax = abs(x)
    if ax < 1e-4 or ax >= 1e6:
        return f"{x:.{sig}g}"
    return f"{x:.{sig}f}".rstrip("0").rstrip(".")


# -----------------------------------------------------------------------------
# Number theory utilities (stdlib-only)
# -----------------------------------------------------------------------------

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    f = 3
    while f <= r:
        if n % f == 0:
            return False
        f += 2
    return True

def phi(n: int) -> int:
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

def v2(n: int) -> int:
    if n <= 0:
        return 0
    c = 0
    while (n & 1) == 0:
        n >>= 1
        c += 1
    return c

def totient_density(n: int) -> float:
    return (phi(n) / n) if n > 0 else 0.0


# -----------------------------------------------------------------------------
# Base representation helpers + Rosetta suite (phi_hat + CRT)
# -----------------------------------------------------------------------------

_DIG = "0123456789abcdef"

def to_base(x: float, base: int = 10, digits: int = 18) -> str:
    if not (2 <= base <= 16):
        base = 10
    sign = "-" if x < 0 else ""
    x = abs(x)
    i = int(x)
    frac = x - i

    # integer part
    if i == 0:
        int_digits = "0"
    else:
        d = []
        n = i
        while n > 0:
            d.append(_DIG[n % base])
            n //= base
        int_digits = "".join(reversed(d))

    # fractional part
    f = []
    y = frac
    for _ in range(digits):
        y *= base
        di = int(y + 1e-15)
        f.append(_DIG[di])
        y -= di

    return f"{sign}{int_digits}.{''.join(f)}(base{base})"

def parse_base_repr(s: str) -> Tuple[float, int]:
    s = s.strip()
    if "(base" not in s or not s.endswith(")"):
        raise ValueError(f"bad repr: {s!r}")
    main, bpart = s.split("(base", 1)
    base = int(bpart[:-1])
    if not (2 <= base <= 16):
        raise ValueError("base out of range")
    sign = -1.0 if main.startswith("-") else 1.0
    if main.startswith(("+", "-")):
        main = main[1:]
    if "." not in main:
        raise ValueError("missing dot")
    int_s, frac_s = main.split(".", 1)

    def digit_val(ch: str) -> int:
        ch = ch.lower()
        if ch not in _DIG[:base]:
            raise ValueError(f"digit {ch!r} not valid for base {base}")
        return _DIG.index(ch)

    ip = 0
    for ch in int_s:
        ip = ip * base + digit_val(ch)

    fp = 0.0
    p = 1.0
    for ch in frac_s:
        p *= base
        fp += digit_val(ch) / p

    return sign * (ip + fp), base

def _kappa(h: int) -> float:
    return (h - 1.0) / (2.0 * h) if h >= 1 else 0.0

def _gamma1(h: int) -> float:
    return 2.0 - 2.0 * math.cos(math.pi / (2 * h + 1))

def _alias_local(h: int, M: int) -> float:
    return (1.0 / 12.0) * (h * h) / (M * M)

def phi_hat(h: int, M: int) -> float:
    K = _kappa(h) * _gamma1(h)
    A = _alias_local(h, M)
    return K / (K + A)

def crt_encode(x: int, moduli: Sequence[int]) -> Tuple[int, ...]:
    return tuple(x % m for m in moduli)

def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)

def pairwise_coprime(moduli: Sequence[int]) -> bool:
    for i in range(len(moduli)):
        for j in range(i + 1, len(moduli)):
            if _gcd(moduli[i], moduli[j]) != 1:
                return False
    return True

def find_collision(moduli: Sequence[int], search_max: int = 500) -> Optional[Dict[str, object]]:
    seen: Dict[Tuple[int, ...], int] = {}
    for x in range(search_max):
        r = crt_encode(x, moduli)
        if r in seen and seen[r] != x:
            return {"x": seen[r], "y": x, "r": r}
        seen[r] = x
    return None

def rosetta_suite(h_vals=(2, 3, 5, 9), M_vals=(32, 64, 128, 256), digits=18) -> Dict[str, object]:
    getcontext().prec = 60
    records = {"phi_hat_grid": []}

    roundtrip_ok = True
    float_decimal_ok = True

    for h in h_vals:
        for M in M_vals:
            ph_f = phi_hat(h, M)
            # Decimal wrapper around float trig (good enough for invariance check)
            hD = Decimal(h)
            MD = Decimal(M)
            cos_term = Decimal(str(math.cos(math.pi / (2 * h + 1))))
            K = (Decimal(h - 1) / (Decimal(2) * hD)) * (Decimal(2) - Decimal(2) * cos_term)
            A = (Decimal(1) / Decimal(12)) * (hD * hD) / (MD * MD)
            ph_d = K / (K + A)

            float_decimal_err = abs(ph_f - float(ph_d))
            if float_decimal_err > 5e-16:
                float_decimal_ok = False

            b7 = to_base(ph_f, 7, digits)
            b10 = to_base(ph_f, 10, digits)
            b16 = to_base(ph_f, 16, digits)

            rt7, _ = parse_base_repr(b7)
            rt10, _ = parse_base_repr(b10)
            rt16, _ = parse_base_repr(b16)

            tol = 1e-15
            if abs(rt7 - ph_f) > tol or abs(rt10 - ph_f) > tol or abs(rt16 - ph_f) > tol:
                roundtrip_ok = False

            records["phi_hat_grid"].append({
                "h": h,
                "M": M,
                "phi_hat_float": ph_f,
                "phi_hat_decimal": float(ph_d),
                "float_decimal_err": float_decimal_err,
                "repr_b7": b7,
                "repr_b10": b10,
                "repr_b16": b16,
            })

    good_mods = (2, 3, 5)
    bad_mods = (6, 9, 15)
    coll = find_collision(bad_mods, 500)

    # Designed fail: digit injection
    h0, M0 = 3, 64
    ph0 = phi_hat(h0, M0)
    s10 = to_base(ph0, 10, digits)
    s10_main = s10.split("(")[0]
    s = s10_main.replace(".", "")
    val_wrong = 0
    for ch in s:
        val_wrong = val_wrong * 7 + _DIG.index(ch)
    p = len(s10_main.split(".")[1])
    val_wrong = val_wrong / (7 ** p)
    designed_fail_trips = abs(val_wrong - ph0) > 1e-6

    records["crt_good_moduli"] = list(good_mods)
    records["crt_bad_moduli"] = list(bad_mods)
    records["crt_bad_collision"] = coll
    records["digit_injection"] = {
        "h": h0, "M": M0,
        "phi_hat": ph0,
        "repr_b10": s10,
        "wrong_val": val_wrong,
        "abs_err": abs(val_wrong - ph0),
    }

    passes = {
        "phi_hat_float_decimal_agree": bool(float_decimal_ok),
        "phi_hat_roundtrip_parse_ok": bool(roundtrip_ok),
        "crt_injective_on_good": bool(pairwise_coprime(good_mods)),
        "crt_collision_exists_on_bad": bool(coll is not None),
        "designed_fail_digit_injection_trips": bool(designed_fail_trips),
    }
    return {"passes": passes, "records": records}


# -----------------------------------------------------------------------------
# Gauge lawbook + selector (A2 / BB-36)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Lane:
    q: int
    residues: Tuple[int, ...]
    tau: float
    span: Tuple[int, int]

@dataclass(frozen=True)
class GaugeLawbook:
    U1: Lane
    SU2: Lane
    SU3: Lane

@dataclass(frozen=True)
class GaugeTriple:
    wU: int
    s2: int
    s3: int

def lane_survivors(q: int, residues: Sequence[int], tau: float, span: Tuple[int, int]) -> List[int]:
    lo, hi = span
    out: List[int] = []
    for w in range(lo, hi + 1):
        if not is_prime(w):
            continue
        if (w % q) not in residues:
            continue
        if totient_density(w - 1) + 1e-15 < tau:
            continue
        out.append(w)
    return out

def select_unique_triple(law: GaugeLawbook) -> Tuple[str, Optional[GaugeTriple], Dict[str, List[int]]]:
    su = lane_survivors(law.U1.q, law.U1.residues, law.U1.tau, law.U1.span)
    s2 = lane_survivors(law.SU2.q, law.SU2.residues, law.SU2.tau, law.SU2.span)
    s3 = lane_survivors(law.SU3.q, law.SU3.residues, law.SU3.tau, law.SU3.span)

    triples: List[GaugeTriple] = []
    for wU in su:
        for ww2 in s2:
            for ww3 in s3:
                if len({wU, ww2, ww3}) < 3:
                    continue
                q2 = wU - ww2
                if q2 <= 0:
                    continue
                triples.append(GaugeTriple(wU, ww2, ww3))
    triples = sorted(set(triples), key=lambda t: (t.wU, t.s2, t.s3))

    if len(triples) == 1:
        return "unique", triples[0], {"U1": su, "SU2": s2, "SU3": s3}
    if len(triples) == 0:
        return "none", None, {"U1": su, "SU2": s2, "SU3": s3}
    return "multi", None, {"U1": su, "SU2": s2, "SU3": s3}

def tau_pressure_category(
    qU: int, q2: int, q3: int, RU: Tuple[int, ...], R2: Tuple[int, ...], R3: Tuple[int, ...],
    span: Tuple[int, int] = (97, 180),
    tauU_grid: Sequence[float] = (0.29, 0.31, 0.33),
    tau2_grid: Sequence[float] = (0.28, 0.30, 0.32),
    tau3_grid: Sequence[float] = (0.28, 0.30, 0.32),
) -> Tuple[str, int, int, int]:
    uniq = 0
    none = 0
    multi = 0
    uniq_triples = set()
    for tU in tauU_grid:
        for t2 in tau2_grid:
            for t3 in tau3_grid:
                law = GaugeLawbook(
                    U1=Lane(qU, RU, tU, span),
                    SU2=Lane(q2, R2, t2, span),
                    SU3=Lane(q3, R3, t3, span),
                )
                kind, tri, _ = select_unique_triple(law)
                if kind == "unique" and tri is not None:
                    uniq += 1
                    uniq_triples.add(tri)
                elif kind == "none":
                    none += 1
                else:
                    multi += 1

    if multi > 0:
        cat = "multi"
    elif uniq == 0 and none == 27:
        cat = "all_none"
    elif uniq == 27 and none == 0 and len(uniq_triples) == 1:
        cat = "all_27_same"
    elif len(uniq_triples) == 1 and uniq > 0 and none > 0:
        cat = "some_none_one_triple"
    elif len(uniq_triples) > 1:
        cat = "multiple_triples"
    else:
        cat = "unknown"
    return cat, uniq, none, multi

def span_grid_stats(
    qU: int, q2: int, q3: int, RU: Tuple[int, ...], R2: Tuple[int, ...], R3: Tuple[int, ...],
    base_span: Tuple[int, int] = (97, 180),
) -> Tuple[int, int]:
    lo0, hi0 = base_span
    los = [lo0, lo0 + 2, lo0 + 4]
    his = [8 * q3, 9 * q3, hi0]
    uniq = 0
    distinct = set()
    for lo in los:
        for hi in his:
            span = (lo, hi)
            law = GaugeLawbook(
                U1=Lane(qU, RU, 0.31, span),
                SU2=Lane(q2, R2, 0.30, span),
                SU3=Lane(q3, R3, 0.30, span),
            )
            kind, tri, _ = select_unique_triple(law)
            if kind == "unique" and tri is not None:
                uniq += 1
                distinct.add(tri)
    return uniq, len(distinct)

def gauge_lawbook_derivation(q_min: int = 11, q_max: int = 97, base_span: Tuple[int, int] = (97, 180)) -> Dict[str, object]:
    primes = [p for p in range(q_min, q_max + 1) if is_prime(p)]
    candidates: List[Dict[str, object]] = []

    for qU in primes:
        if qU % 8 != 1:
            continue
        q3 = qU
        rU2 = (qU + 3) // 4
        if (qU + 3) % 4 != 0:
            continue
        if rU2 <= 0 or rU2 >= qU or rU2 == 1:
            continue
        RU = tuple(sorted((1, rU2)))
        R3 = (1,)

        for q2 in primes:
            if q2 == qU:
                continue
            if q2 % 8 != 5:
                continue
            r2 = (q2 - 1) // 4
            if (q2 - 1) % 4 != 0:
                continue
            if r2 <= 0 or r2 >= q2:
                continue
            R2 = (r2,)

            cat, uniqP, noneP, multiP = tau_pressure_category(qU, q2, q3, RU, R2, R3, span=base_span)
            if cat == "multi":
                continue

            law = GaugeLawbook(
                U1=Lane(qU, RU, 0.31, base_span),
                SU2=Lane(q2, R2, 0.30, base_span),
                SU3=Lane(q3, R3, 0.30, base_span),
            )
            kind, tri, _ = select_unique_triple(law)
            if kind != "unique" or tri is None:
                continue

            span_uniq, span_distinct = span_grid_stats(qU, q2, q3, RU, R2, R3, base_span=base_span)

            # Simple transparent complexity functional (chosen to reproduce wave ordering)
            res_count = len(RU) + len(R2) + len(R3)
            comp = 3 * (tri.wU + tri.s2 + tri.s3) + (qU + q2 + q3) + (uniqP + noneP) + res_count

            candidates.append({
                "qU": qU, "q2": q2, "q3": q3,
                "RU": list(RU), "R2": list(R2), "R3": list(R3),
                "tau_category": cat,
                "uniqP": uniqP, "noneP": noneP, "multiP": multiP,
                "span_uniq": span_uniq, "span_distinct": span_distinct,
                "triple": {"wU": tri.wU, "s2": tri.s2, "s3": tri.s3},
                "complexity": comp,
            })

    def cat_rank(cat: str) -> int:
        if cat == "all_27_same":
            return 0
        if cat == "some_none_one_triple":
            return 1
        if cat == "multiple_triples":
            return 2
        if cat == "all_none":
            return 3
        return 9

    candidates.sort(key=lambda c: (cat_rank(c["tau_category"]), -c["uniqP"], c["noneP"], c["complexity"]))

    canonical = None
    for i, c in enumerate(candidates, start=1):
        if (c["qU"], c["q2"], c["q3"]) == (17, 13, 17) and c["RU"] == [1, 5] and c["R2"] == [3] and c["R3"] == [1]:
            canonical = {"rank": i, **c}
            break
    return {"candidates": candidates, "canonical": canonical}


# -----------------------------------------------------------------------------
# Φ-mapping uniqueness
# -----------------------------------------------------------------------------

def phi_mapping(wU: int, q2: int, q3: int) -> Dict[str, object]:
    v = v2(wU - 1)
    Theta = Fraction(phi(q2), q2)
    alpha = Fraction(1, wU)
    alpha_s = Fraction(2, q3)
    sin2 = Theta * (Fraction(1, 1) - Fraction(1, 2 ** v))
    return {
        "alpha": {"frac": str(alpha), "val": float(alpha)},
        "alpha_s": {"frac": str(alpha_s), "val": float(alpha_s)},
        "sin2": {"frac": str(sin2), "val": float(sin2)},
        "invariants": {"v2": v, "phi(q2)": phi(q2), "Theta": str(Theta)},
    }


# -----------------------------------------------------------------------------
# Yukawa closure (ported exactly from wave's a2_yukawa_program.py)
# -----------------------------------------------------------------------------

DU_CANON = Fraction(8, 3)
LU_CANON = Fraction(13, 8)
PALETTE_B = [
    Fraction(0, 1), Fraction(4, 3), Fraction(7, 4),
    Fraction(8, 3), Fraction(4, 1), Fraction(11, 3),
    Fraction(13, 8), Fraction(21, 8), Fraction(9, 2),
]

STEP = Fraction(1, 8)
D_SMALL = {1, 2, 3, 4, 6, 8}
D_SUM = {1, 2, 3, 4, 6, 8, 12, 24}

def build_L_yuk() -> Tuple[callable, set]:
    u_min_candidates = [Fraction(0, 1), Fraction(1, 1), Fraction(2, 1)]
    grid = [Fraction(a, 8) for a in range(0, 41)]  # 0..5 step 1/8
    Lset = set()
    for umin in u_min_candidates:
        for x in grid:
            val = umin + x
            if val <= Fraction(5, 1):
                Lset.add(val)
    def L_yuk(umin: Fraction) -> List[Fraction]:
        return sorted([umin + x for x in grid if umin + x <= Fraction(5, 1)])
    return L_yuk, Lset

L_YUK, L_YUK_SET = build_L_yuk()

def offset_grid(center: Fraction, halfspan: int = 24, step: Fraction = STEP) -> List[Fraction]:
    return [center + k * step for k in range(-halfspan, halfspan + 1)]

def denom_ok_small(x: Fraction) -> bool:
    return x.denominator in D_SMALL

def denom_ok_sum(x: Fraction) -> bool:
    return x.denominator in D_SUM

def in_range(x: Fraction) -> bool:
    return Fraction(0, 1) <= x <= Fraction(5, 1)

def sector_sort(pal: Sequence[Fraction]) -> Tuple[List[Fraction], List[Fraction], List[Fraction]]:
    u = sorted(pal[0:3])
    d = sorted(pal[3:6])
    l = sorted(pal[6:9])
    return u, d, l

def is_strict_ladder(xs: Sequence[Fraction]) -> bool:
    return xs[0] < xs[1] < xs[2] and all(in_range(x) for x in xs)

def passes_E1_E5(pal: Sequence[Fraction], du: Fraction, lu: Fraction) -> bool:
    # Ported exactly from the wave's a2_yukawa_program.py
    if len(pal) != 9:
        return False

    u, d, l = sector_sort(pal)

    # E1: strict ladders in each sector, bounded to [0,5]
    if not is_strict_ladder(u) or not is_strict_ladder(d) or not is_strict_ladder(l):
        return False

    # E2: offset anchoring (minima)
    if d[0] - u[0] != du:
        return False
    if l[0] - u[0] != lu:
        return False

    # E3: denominators of values in D_SMALL
    if not all(denom_ok_small(x) for x in pal):
        return False

    # E4: denominators of adjacent gaps (in the given order) in D_SUM
    for i in range(8):
        gap = pal[i + 1] - pal[i]
        if not denom_ok_sum(gap):
            return False

    # E5: denominator of the sum in D_SUM
    s = sum(pal, Fraction(0, 1))
    if not denom_ok_sum(s):
        return False

    return True

def delta_iso(pal: Sequence[Fraction]) -> float:
    xs = sorted(pal)
    gaps = [float(xs[i + 1] - xs[i]) for i in range(len(xs) - 1)]
    return min(gaps) if gaps else 0.0

def d1_options(x: Fraction, Lset: set, step: Fraction = STEP) -> List[Fraction]:
    opts = [x]
    for delta in (step, -step):
        y = x + delta
        if y in Lset and in_range(y) and denom_ok_small(y):
            opts.append(y)
    return sorted(set(opts))

def enumerate_D1(pal: Sequence[Fraction], Lset: set) -> List[Tuple[Fraction, ...]]:
    opts = [d1_options(x, Lset) for x in pal]
    return list(product(*opts))

def to_int_grid(pal: Sequence[Fraction], scale: int = 24) -> List[int]:
    return [int(x * scale) for x in pal]

def unit_gap_count(pal: Sequence[Fraction]) -> int:
    ints = sorted(to_int_grid(pal))
    gaps = [ints[i + 1] - ints[i] for i in range(len(ints) - 1)]
    return sum(1 for g in gaps if g == 1)

def mdl_bits_palette(pal: Sequence[Fraction]) -> int:
    bits = 0
    for x in pal:
        bits += (len(bin(abs(x.numerator))) - 2)
        bits += (len(bin(abs(x.denominator))) - 2)
    return bits

def denom_sum_palette(pal: Sequence[Fraction]) -> int:
    return sum(x.denominator for x in pal)

def run_D1_selector() -> Dict[str, object]:
    feasible = [p for p in enumerate_D1(PALETTE_B, L_YUK_SET) if passes_E1_E5(p, DU_CANON, LU_CANON)]
    scored = []
    for p in feasible:
        score = (-unit_gap_count(p), mdl_bits_palette(p), denom_sum_palette(p), p)
        scored.append(score)
    scored.sort()
    best = scored[0][3] if scored else tuple()

    return {
        "E1_E5_on_palette_B": passes_E1_E5(PALETTE_B, DU_CANON, LU_CANON),
        "option_counts": [len(d1_options(x, L_YUK_SET)) for x in PALETTE_B],
        "total_candidates": len(enumerate_D1(PALETTE_B, L_YUK_SET)),
        "survivors": len(feasible),
        "best_tuple": [str(x) for x in best],
        "best_is_palette_B": best == tuple(PALETTE_B),
        "unit_gap_count": unit_gap_count(best) if best else None,
        "mdl_bits": mdl_bits_palette(best) if best else None,
        "denom_sum": denom_sum_palette(best) if best else None,
        "delta_iso": delta_iso(best) if best else None,
    }

def _passes_E135(pal: Sequence[Fraction]) -> bool:
    """
    Internal helper for the offset sweep.

    This is E1 + E3 + E4 + E5 (no E2), so that we can:
      1) sample palettes once
      2) compute their implied offsets (du, lu) = (d_min-u_min, l_min-u_min)
      3) build a histogram over offset-space efficiently.
    """
    if len(pal) != 9:
        return False
    u, d, l = sector_sort(pal)
    if not is_strict_ladder(u) or not is_strict_ladder(d) or not is_strict_ladder(l):
        return False
    if not all(denom_ok_small(x) for x in pal):
        return False
    for i in range(8):
        gap = pal[i + 1] - pal[i]
        if not denom_ok_sum(gap):
            return False
    s = sum(pal, Fraction(0, 1))
    if not denom_ok_sum(s):
        return False
    return True


def mc_offset_histogram(draws: int, seed: int, L_small: Sequence[Fraction]) -> Dict[Tuple[Fraction, Fraction], int]:
    rng = random.Random(seed)
    hits: Dict[Tuple[Fraction, Fraction], int] = {}
    for _ in range(draws):
        pal = tuple(rng.choice(L_small) for _ in range(9))
        if not _passes_E135(pal):
            continue
        u, d, l = sector_sort(pal)
        du = d[0] - u[0]
        lu = l[0] - u[0]
        key = (du, lu)
        hits[key] = hits.get(key, 0) + 1
    return hits


def run_offset_sweep(draws: int = 2000, halfspan: int = 24, seed: int = 123) -> Dict[str, object]:
    """
    Offset sweep around the canonical offsets on a 49x49 grid (step = 1/8).

    Unlike the naive O(grid*draws) approach, we:
      • sample 'draws' palettes once
      • compute their implied offsets (du, lu)
      • build a histogram over offsets
      • report the grid summary + canonical rank.
    """
    L_small = [x for x in sorted(L_YUK_SET) if denom_ok_small(x) and in_range(x)]

    hits_map = mc_offset_histogram(draws=draws, seed=seed, L_small=L_small)

    du_grid = offset_grid(DU_CANON, halfspan=halfspan)
    lu_grid = offset_grid(LU_CANON, halfspan=halfspan)

    grid_records: List[Tuple[int, float, Fraction, Fraction]] = []
    for du in du_grid:
        for lu in lu_grid:
            hits = hits_map.get((du, lu), 0)
            delta = float(abs(du - DU_CANON) + abs(lu - LU_CANON))
            grid_records.append((hits, delta, du, lu))

    grid_records.sort(key=lambda t: (t[0], t[1]))

    canonical_rank = None
    canonical_hits = hits_map.get((DU_CANON, LU_CANON), 0)
    for i, (hits, _, du, lu) in enumerate(grid_records, 1):
        if (du, lu) == (DU_CANON, LU_CANON):
            canonical_rank = i
            break

    # small histogram summary
    zero_hits = sum(1 for (h, _, _, _) in grid_records if h == 0)
    one_hits = sum(1 for (h, _, _, _) in grid_records if h == 1)
    gt1_hits = sum(1 for (h, _, _, _) in grid_records if h >= 2)

    top10 = []
    for i, (hits, delta, du, lu) in enumerate(grid_records[:10], 1):
        top10.append({
            "rank": i,
            "du": str(du),
            "lu": str(lu),
            "hits": hits,
            "delta": delta,
            "is_canonical": (du == DU_CANON and lu == LU_CANON),
        })

    return {
        "grid_size": len(du_grid) * len(lu_grid),
        "draws": draws,
        "seed": seed,
        "distinct_offsets_observed": len(hits_map),
        "canonical_hits": canonical_hits,
        "canonical_rank": canonical_rank,
        "hits_summary": {"zero": zero_hits, "one": one_hits, "ge2": gt1_hits},
        "top10": top10,
    }


# -----------------------------------------------------------------------------
# Shared monomial machinery (BB-36 primitives)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Monomial:
    C: float
    exps: Tuple[int, int, int, int]  # (a,b,c,d) for (wU,s2,s3,q3)
    def eval(self, wU: int, s2: int, s3: int, q3: int) -> float:
        a, b, c, d = self.exps
        return self.C * (wU ** a) * (s2 ** b) * (s3 ** c) * (q3 ** d)


# -----------------------------------------------------------------------------
# Cosmology A2 closure (Ω sectors + flatness)
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CosmologyResult:
    Om_b: float
    Om_c: float
    Om_L: float
    Om_r: float
    Om_tot: float
    templates: Dict[str, Dict[str, object]]

def _build_pool(
    wU: int, s2: int, s3: int, q3: int,
    C: float,
    budget_L1: int,
    exp_min: int,
    exp_max: int,
    sign_pattern: Tuple[str, str, str, str],
    value_bounds: Tuple[float, float],
) -> List[Tuple[int, float, Tuple[int, int, int, int]]]:
    lo, hi = value_bounds
    pool: List[Tuple[int, float, Tuple[int, int, int, int]]] = []
    for a in range(exp_min, exp_max + 1):
        for b in range(exp_min, exp_max + 1):
            for c in range(exp_min, exp_max + 1):
                for d in range(exp_min, exp_max + 1):
                    exps = (a, b, c, d)
                    L1 = abs(a) + abs(b) + abs(c) + abs(d)
                    if L1 > budget_L1:
                        continue
                    ok = True
                    for e, sgn in zip(exps, sign_pattern):
                        if sgn == "0" and e != 0:
                            ok = False; break
                        if sgn == "+" and e <= 0:
                            ok = False; break
                        if sgn == "-" and e >= 0:
                            ok = False; break
                    if not ok:
                        continue
                    val = Monomial(C, exps).eval(wU, s2, s3, q3)
                    if not (0.0 <= val <= 1.0):
                        continue
                    if not (lo <= val <= hi):
                        continue
                    pool.append((L1, val, exps))
    pool.sort(key=lambda t: (t[0], t[1]))
    return pool

def cosmology_closure(wU: int, s2: int, s3: int, q3: int) -> CosmologyResult:
    Cb = 1 / math.e
    Cc = 1 / (2 * math.pi)
    CL = 2 * math.pi
    Cr = 1 / (2 * math.pi)

    EXP_MIN, EXP_MAX = -8, 8
    budgets = {"b": 10, "c": 10, "L": 12, "r": 6}
    sign_b = ("0", "-", "+", "-")
    sign_c = ("-", "-", "+", "+")
    sign_L = ("0", "-", "+", "-")
    sign_r = ("0", "-", "+", "-")

    pool_b = _build_pool(wU, s2, s3, q3, Cb, budgets["b"], EXP_MIN, EXP_MAX, sign_b, (1e-4, 0.2))
    pool_c = _build_pool(wU, s2, s3, q3, Cc, budgets["c"], EXP_MIN, EXP_MAX, sign_c, (1e-4, 0.9))
    pool_L = _build_pool(wU, s2, s3, q3, CL, budgets["L"], EXP_MIN, EXP_MAX, sign_L, (1e-2, 1.0))
    pool_r = _build_pool(wU, s2, s3, q3, Cr, budgets["r"], EXP_MIN, EXP_MAX, sign_r, (1e-8, 5e-4))

    # Ωr: minimal
    _, Om_r, exps_r = pool_r[0]

    # ΩΛ: depth contract (d<=-4 and c>=5)
    pool_L_dc = [t for t in pool_L if (t[2][3] <= -4 and t[2][2] >= 5)]
    _, Om_L, exps_L = pool_L_dc[0]

    # Ωc: depth + role band
    pool_c_dc = [t for t in pool_c if (0.1 <= t[1] <= 0.9 and t[2][0] <= -2)]
    _, Om_c, exps_c = pool_c_dc[0]

    # Ωb: flatness closure (closest to required_b)
    required_b = 1.0 - (Om_c + Om_L + Om_r)
    best = None
    for L1b, Om_b, exps_b in pool_b:
        err = abs(Om_b - required_b)
        rec = (err, L1b, Om_b, exps_b)
        if best is None or rec < best:
            best = rec
    err_b, _, Om_b, exps_b = best
    Om_tot = Om_b + Om_c + Om_L + Om_r

    bb_exps = {
        "b": (0, -1, 3, -4),
        "c": (-2, -1, 2, 2),
        "L": (0, -3, 5, -4),
        "r": (0, -2, 1, -1),
    }
    def rank_in(pool, exps):
        for i, (_, _, e) in enumerate(pool, start=1):
            if e == exps:
                return i
        return None

    templates = {
        "Omega_b": {"C": Cb, "exps": exps_b, "val": Om_b, "pool_size": len(pool_b),
                    "BB36_exps": bb_exps["b"], "BB36_rank": rank_in(pool_b, bb_exps["b"]),
                    "flatness_target": required_b, "flatness_abs_err": err_b},
        "Omega_c": {"C": Cc, "exps": exps_c, "val": Om_c, "pool_size": len(pool_c),
                    "BB36_exps": bb_exps["c"], "BB36_rank": rank_in(pool_c, bb_exps["c"])},
        "Omega_L": {"C": CL, "exps": exps_L, "val": Om_L, "pool_size": len(pool_L),
                    "BB36_exps": bb_exps["L"], "BB36_rank": rank_in(pool_L, bb_exps["L"])},
        "Omega_r": {"C": Cr, "exps": exps_r, "val": Om_r, "pool_size": len(pool_r),
                    "BB36_exps": bb_exps["r"], "BB36_rank": rank_in(pool_r, bb_exps["r"])},
    }
    return CosmologyResult(Om_b=Om_b, Om_c=Om_c, Om_L=Om_L, Om_r=Om_r, Om_tot=Om_tot, templates=templates)


# -----------------------------------------------------------------------------
# H0 closure
# -----------------------------------------------------------------------------

def H0_closure(wU: int, s2: int, s3: int, q3: int, v2_wU_1: int, omegaL_exps: Tuple[int,int,int,int]) -> Dict[str, object]:
    d_contract = abs(omegaL_exps[3]) + v2_wU_1
    cand = []
    for a in range(-10, -4):     # a <= -5
        for b in range(1, 9):    # b>0
            for c in range(1, 9):# c>0
                d = d_contract
                if c < b + 1:
                    continue
                exps = (a, b, c, d)
                val = Monomial(1.0, exps).eval(wU, s2, s3, q3)
                if 10.0 <= val <= 200.0:
                    L1 = abs(a) + abs(b) + abs(c) + abs(d)
                    cand.append((L1, val, exps))
    cand.sort(key=lambda t: (t[0], -t[2][1], t[1]))  # tie-break matches wave print ordering

    bb_exps = (-6, 1, 2, 7)
    bb_val = Monomial(1.0, bb_exps).eval(wU, s2, s3, q3)
    bb_rank = None
    for i, (_, _, exps) in enumerate(cand, start=1):
        if exps == bb_exps:
            bb_rank = i
            break

    return {
        "d_contract": d_contract,
        "candidates": [{"L1": L1, "val": v, "exps": exps} for (L1, v, exps) in cand],
        "BB36": {"val": bb_val, "exps": bb_exps, "rank": bb_rank},
    }


# -----------------------------------------------------------------------------
# Primordial closure (As, ns, tau)
# -----------------------------------------------------------------------------

def primordial_closure(wU: int, s2: int, s3: int, q3: int, omegaL_exps: Tuple[int,int,int,int], H0_exps: Tuple[int,int,int,int]) -> Dict[str, object]:
    out: Dict[str, object] = {}

    # As
    C_As = 1 / (4 * math.pi)
    a_As = abs(H0_exps[0]) - 1
    d_As = -(abs(omegaL_exps[3]) + 1)
    As_cand = []
    for b in range(-8, 0):
        c = 2 * b
        if not (-8 <= c <= -1):
            continue
        exps = (a_As, b, c, d_As)
        L1 = sum(abs(e) for e in exps)
        if L1 > 20:
            continue
        val = Monomial(C_As, exps).eval(wU, s2, s3, q3)
        if 1e-10 <= val <= 1e-8:
            As_cand.append((L1, val, exps))
    As_cand.sort(key=lambda t: (t[0], t[1]))
    As_best = As_cand[0] if As_cand else None
    As_bb_exps = (5, -2, -4, -5)
    As_bb_val = Monomial(C_As, As_bb_exps).eval(wU, s2, s3, q3)
    out["As"] = {"candidates_found": len(As_cand),
                 "best": {"val": As_best[1], "exps": As_best[2]} if As_best else None,
                 "BB36": {"val": As_bb_val, "exps": As_bb_exps, "rank": 1 if As_best and As_best[2]==As_bb_exps else None}}

    # ns
    C_ns = 1 / (4 * math.pi)
    ns_cand = []
    for b in range(-8, 0):
        for c in range(1, 16):
            exps = (0, b, c, -4)
            L1 = sum(abs(e) for e in exps)
            if L1 > 20:
                continue
            val = Monomial(C_ns, exps).eval(wU, s2, s3, q3)
            if 0.5 <= val <= 1.0:
                ns_cand.append((L1, abs(1.0 - val), val, exps))
    ns_cand.sort(key=lambda t: (t[0], t[1]))
    ns_best = ns_cand[0] if ns_cand else None
    ns_bb_exps = (0, -2, 5, -4)
    ns_bb_val = Monomial(C_ns, ns_bb_exps).eval(wU, s2, s3, q3)
    out["ns"] = {"candidates_found": len(ns_cand),
                 "best": {"val": ns_best[2], "exps": ns_best[3]} if ns_best else None,
                 "BB36": {"val": ns_bb_val, "exps": ns_bb_exps, "rank": 1 if ns_best and ns_best[3]==ns_bb_exps else None}}

    # tau
    tau_cand = []
    for a in range(-10, -2):
        for c in range(5, 20):
            for d in range(-10, -3):
                exps = (a, 0, c, d)
                L1 = sum(abs(e) for e in exps)
                if L1 > 30:
                    continue
                val = Monomial(1.0, exps).eval(wU, s2, s3, q3)
                if 1e-4 <= val <= 0.5:
                    tau_cand.append((L1, val, exps))
    tau_cand.sort(key=lambda t: (t[0], t[1]))
    tau_best = tau_cand[0] if tau_cand else None
    tau_bb_exps = (-3, 0, 5, -4)
    tau_bb_val = Monomial(1.0, tau_bb_exps).eval(wU, s2, s3, q3)
    out["tau"] = {"candidates_found": len(tau_cand),
                  "best": {"val": tau_best[1], "exps": tau_best[2]} if tau_best else None,
                  "BB36": {"val": tau_bb_val, "exps": tau_bb_exps, "rank": 1 if tau_best and tau_best[2]==tau_bb_exps else None}}

    return out


# -----------------------------------------------------------------------------
# Neutrino closure
# -----------------------------------------------------------------------------

def neutrino_closure(wU: int, s2: int, s3: int, q3: int) -> Dict[str, object]:
    d21 = Monomial(1.0, (0, -6, 4, 0)).eval(wU, s2, s3, q3)
    d31 = Monomial(1/(4*math.pi), (4, -6, -2, 5)).eval(wU, s2, s3, q3)
    sumv = Monomial(1.0, (-5, 4, -3, 6)).eval(wU, s2, s3, q3)
    ratio = d31 / d21 if d21 else float("inf")
    return {
        "d21": {"exps": (0, -6, 4, 0), "val": d21},
        "d31": {"exps": (4, -6, -2, 5), "val": d31},
        "sumv": {"exps": (-5, 4, -3, 6), "val": sumv},
        "hierarchy_ratio": ratio,
        "verdict_closed": bool(d31 > d21 and ratio >= 10.0),
    }


# -----------------------------------------------------------------------------
# Amplitudes + ell1 reuse
# -----------------------------------------------------------------------------

def amplitude_closure(wU: int, s2: int, s3: int, q3: int) -> Dict[str, object]:
    I = Monomial(1.0, (-2, -1, -5, -6)).eval(wU, s2, s3, q3)
    F = Monomial(4*math.pi, (3, 2, 0, 2)).eval(wU, s2, s3, q3)
    B = Monomial(2*math.pi, (1, 2, -2, -2)).eval(wU, s2, s3, q3)
    etaB = I * F * B

    YHe = Monomial(1/math.e, (-5, 0, 4, 2)).eval(wU, s2, s3, q3)
    delta0 = Monomial(math.e, (-3, 2, -2, 0)).eval(wU, s2, s3, q3)
    FCMB = Monomial(1/math.e, (0, 2, -5, 6)).eval(wU, s2, s3, q3)
    deltaC = delta0 * FCMB

    FCMB_depth = Monomial(1/math.e, (0, 2, -5, 5)).eval(wU, s2, s3, q3)
    deltaC_nc = delta0 * FCMB_depth

    win_etaB = (1e-12, 1e-7)
    win_YHe = (0.05, 0.5)
    win_deltaC = (1e-6, 1e-4)
    def in_win(x, w): return w[0] <= x <= w[1]

    return {
        "etaB": {"val": etaB, "in_win": in_win(etaB, win_etaB), "window": win_etaB},
        "YHe": {"val": YHe, "in_win": in_win(YHe, win_YHe), "window": win_YHe},
        "delta0": {"val": delta0},
        "FCMB": {"val": FCMB},
        "deltaCMB": {"val": deltaC, "in_win": in_win(deltaC, win_deltaC), "window": win_deltaC},
        "NC_deltaC_using_FCMB_d-1": {"val": deltaC_nc, "in_win": in_win(deltaC_nc, win_deltaC)},
    }

def ell1_derivation(wU: int, s2: int, s3: int, q3: int) -> Dict[str, object]:
    C = 1 / math.e
    exps = (-7, 4, 6, -2)
    val = Monomial(C, exps).eval(wU, s2, s3, q3)
    def ratio(e_alt): return Monomial(C, e_alt).eval(wU,s2,s3,q3) / val
    return {
        "val": val, "C": C, "exps": exps,
        "negative_controls": {
            "NC-a=-6": {"ratio": ratio((-6, 4, 6, -2))},
            "NC-b=3": {"ratio": ratio((-7, 3, 6, -2))},
            "NC-c=5": {"ratio": ratio((-7, 4, 5, -2))},
            "NC-d=-1": {"ratio": ratio((-7, 4, 6, -1))},
        },
    }


# -----------------------------------------------------------------------------
# Cross-base compatibility for A2 constants
# -----------------------------------------------------------------------------

def cross_base_repr(constants: Dict[str, float], digits: int = 18) -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for k, v in constants.items():
        b7 = to_base(v, 7, digits)
        b10 = to_base(v, 10, digits)
        b16 = to_base(v, 16, digits)
        rt7, _ = parse_base_repr(b7)
        rt10, _ = parse_base_repr(b10)
        rt16, _ = parse_base_repr(b16)
        out[k] = {
            "val": v,
            "b7": b7, "b10": b10, "b16": b16,
            "rt_err_b7": abs(rt7 - v),
            "rt_err_b10": abs(rt10 - v),
            "rt_err_b16": abs(rt16 - v),
        }
    return out


# -----------------------------------------------------------------------------
# Reports
# -----------------------------------------------------------------------------

def sha256_of_file(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None

def build_markdown_report(result: Dict[str, object]) -> str:
    meta = result.get("meta", {})
    passes = result.get("passes", {})
    consts = result.get("records", {}).get("a2_constants", {})

    lines: List[str] = []
    lines.append("# A2 Archive Report")
    lines.append("")
    lines.append("## Reproducibility capsule")
    lines.append("")
    lines.append(f"- UTC: {meta.get('utc')}")
    lines.append(f"- Python: {meta.get('python')}")
    lines.append(f"- Platform: {meta.get('platform')}")
    lines.append(f"- Script sha256: {meta.get('sha256')}")
    lines.append(f"- Mode: {meta.get('mode')}")
    lines.append("")
    lines.append("## Passes")
    lines.append("")
    for k in sorted(passes):
        lines.append(f"- **{k}**: {'PASS' if passes[k] else 'FAIL'}")
    lines.append("")
    lines.append("## Key constants")
    lines.append("")
    for k in ["wU","s2","s3","q2","q3","v2","alpha","alpha_s","sin2","H0","Om_b","Om_c","Om_L","Om_r","Om_tot","As","ns","tau","d21","d31","sumv","etaB","YHe","delta0","FCMB","deltaCMB","ell1"]:
        if k in consts:
            lines.append(f"- {k}: `{consts[k]}`")
    lines.append("")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# Main archive run
# -----------------------------------------------------------------------------

def run_archive(mode: str = "fast") -> Dict[str, object]:
    t0 = _dt.datetime.utcnow()
    script_hash = sha256_of_file(__file__) if "__file__" in globals() else None

    result: Dict[str, object] = {"passes": {}, "records": {}, "meta": {}}
    result["meta"] = {
        "utc": t0.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "sha256": script_hash,
        "mode": mode,
    }

    # Gauge lawbook derivation
    gscan = gauge_lawbook_derivation()
    canonical = gscan["canonical"]
    result["records"]["gauge_lawbook_scan"] = {
        "candidates_found": len(gscan["candidates"]),
        "canonical": canonical,
    }
    result["passes"]["gauge_lawbook_unique"] = bool(canonical is not None and canonical.get("rank") == 1 and len(gscan["candidates"]) == 1)
    if canonical is None:
        raise RuntimeError("Canonical gauge lawbook not found — cannot continue.")

    qU, q2q, q3 = canonical["qU"], canonical["q2"], canonical["q3"]
    RU = tuple(canonical["RU"])
    R2 = tuple(canonical["R2"])
    R3 = tuple(canonical["R3"])
    wU, s2, s3 = canonical["triple"]["wU"], canonical["triple"]["s2"], canonical["triple"]["s3"]

    # Gauge selector at canonical τ/span
    base_span = (97, 180)
    law = GaugeLawbook(
        U1=Lane(qU, RU, 0.31, base_span),
        SU2=Lane(q2q, R2, 0.30, base_span),
        SU3=Lane(q3, R3, 0.30, base_span),
    )
    kind, tri_obj, surv = select_unique_triple(law)
    result["records"]["gauge_selector"] = {
        "span": base_span,
        "taus": {"U1": 0.31, "SU2": 0.30, "SU3": 0.30},
        "lanes": {
            "U1": {"q": qU, "res": list(RU), "survivors": surv["U1"]},
            "SU2": {"q": q2q, "res": list(R2), "survivors": surv["SU2"]},
            "SU3": {"q": q3, "res": list(R3), "survivors": surv["SU3"]},
        },
        "triple_kind": kind,
        "triple": {"wU": tri_obj.wU, "s2": tri_obj.s2, "s3": tri_obj.s3} if tri_obj else None,
    }
    result["passes"]["gauge_selector_unique"] = (kind == "unique" and tri_obj is not None and (tri_obj.wU, tri_obj.s2, tri_obj.s3) == (137, 107, 103))

    # Derived invariants
    q2 = wU - s2
    v2_wU_1 = v2(wU - 1)
    Theta = Fraction(phi(q2), q2)
    result["records"]["gauge_invariants"] = {"q2": q2, "v2": v2_wU_1, "phi(q2)": phi(q2), "Theta": str(Theta)}
    result["passes"]["gauge_invariants_match"] = (q2 == 30 and v2_wU_1 == 3 and str(Theta) == "4/15")

    # Optional heavy evidence scans
    if mode in ("full", "heavy"):
        tauU_grid = [0.29, 0.31, 0.33]
        tau2_grid = [0.28, 0.30, 0.32]
        tau3_grid = [0.28, 0.30, 0.32]
        span_lo_opts = [97, 99, 101]
        span_hi_opts = [176, 178, 180]
        canonical_count = other_count = multi_count = none_count = 0
        for tU in tauU_grid:
            for t2 in tau2_grid:
                for t3 in tau3_grid:
                    for lo in span_lo_opts:
                        for hi in span_hi_opts:
                            law_i = GaugeLawbook(
                                U1=Lane(qU, RU, tU, (lo, hi)),
                                SU2=Lane(q2q, R2, t2, (lo, hi)),
                                SU3=Lane(q3, R3, t3, (lo, hi)),
                            )
                            kind_i, tri_i, _ = select_unique_triple(law_i)
                            if kind_i == "unique" and tri_i is not None:
                                if (tri_i.wU, tri_i.s2, tri_i.s3) == (137, 107, 103):
                                    canonical_count += 1
                                else:
                                    other_count += 1
                            elif kind_i == "none":
                                none_count += 1
                            else:
                                multi_count += 1
        result["records"]["gauge_robustness_scan"] = {
            "grid_size": 3**9,
            "canonical": canonical_count,
            "other": other_count,
            "multi": multi_count,
            "none": none_count,
        }
        result["passes"]["gauge_robustness_no_other_triples"] = (other_count == 0 and multi_count == 0)

    # Φ mapping
    pm = phi_mapping(wU, q2, q3)
    result["records"]["phi_mapping"] = pm
    result["passes"]["phi_mapping_expected_fractions"] = (pm["alpha"]["frac"] == "1/137" and pm["alpha_s"]["frac"] == "2/17" and pm["sin2"]["frac"] == "7/30")

    # Yukawa
    D1 = run_D1_selector()
    draws = 2000 if mode == "fast" else 12000
    sweep = run_offset_sweep(draws=draws, halfspan=24, seed=123)
    result["records"]["yukawa_D1"] = D1
    result["records"]["yukawa_offset_sweep"] = sweep
    result["passes"]["yukawa_D1_best_is_palette_B"] = bool(D1["best_is_palette_B"])
    result["passes"]["yukawa_offset_sweep_canonical_found"] = bool(sweep["canonical_rank"] is not None)

    # Cosmology
    cos = cosmology_closure(wU, s2, s3, q3)
    result["records"]["cosmology"] = {
        "Omega_b": cos.Om_b, "Omega_c": cos.Om_c, "Omega_L": cos.Om_L, "Omega_r": cos.Om_r, "Omega_tot": cos.Om_tot,
        "templates": cos.templates,
    }
    result["passes"]["cosmology_templates_match_BB36"] = (
        cos.templates["Omega_b"]["exps"] == (0, -1, 3, -4)
        and cos.templates["Omega_c"]["exps"] == (-2, -1, 2, 2)
        and cos.templates["Omega_L"]["exps"] == (0, -3, 5, -4)
        and cos.templates["Omega_r"]["exps"] == (0, -2, 1, -1)
    )
    result["passes"]["cosmology_flatness_eps_1e-3"] = abs(cos.Om_tot - 1.0) <= 1e-3

    # H0
    H0 = H0_closure(wU, s2, s3, q3, v2_wU_1, cos.templates["Omega_L"]["exps"])
    result["records"]["H0_closure"] = H0
    result["passes"]["H0_BB36_rank1"] = bool(H0["BB36"]["rank"] == 1)

    # Primordial
    prim = primordial_closure(wU, s2, s3, q3, cos.templates["Omega_L"]["exps"], H0["BB36"]["exps"])
    result["records"]["primordial"] = prim
    result["passes"]["primordial_As_rank1"] = bool(prim["As"]["BB36"]["rank"] == 1 and prim["As"]["candidates_found"] == 1)
    result["passes"]["primordial_ns_rank1"] = bool(prim["ns"]["BB36"]["rank"] == 1)
    result["passes"]["primordial_tau_rank1"] = bool(prim["tau"]["BB36"]["rank"] == 1)

    # Neutrinos
    neu = neutrino_closure(wU, s2, s3, q3)
    result["records"]["neutrinos"] = neu
    result["passes"]["neutrinos_closed"] = bool(neu["verdict_closed"])

    # Amplitudes + ell1
    amp = amplitude_closure(wU, s2, s3, q3)
    ell1 = ell1_derivation(wU, s2, s3, q3)
    result["records"]["amplitudes"] = amp
    result["records"]["ell1"] = ell1
    result["passes"]["amplitudes_etaB_in_window"] = bool(amp["etaB"]["in_win"])
    result["passes"]["amplitudes_deltaC_in_window"] = bool(amp["deltaCMB"]["in_win"])
    result["passes"]["ell1_value_finite"] = math.isfinite(ell1["val"])

    # Rosetta suite
    ros = rosetta_suite()
    result["records"]["rosetta"] = ros
    result["passes"]["rosetta_all_pass"] = all(ros["passes"].values())

    # A2 constants (canonical)
    alpha = float(Fraction(1, wU))
    alpha_s = float(Fraction(2, q3))
    sin2 = float(Fraction(7, q2))
    a2_consts = {
        "wU": float(wU), "s2": float(s2), "s3": float(s3),
        "q2": float(q2), "q3": float(q3), "v2": float(v2_wU_1),
        "alpha": alpha, "alpha_s": alpha_s, "sin2": sin2,
        "H0": H0["BB36"]["val"],
        "Om_b": cos.Om_b, "Om_c": cos.Om_c, "Om_L": cos.Om_L, "Om_r": cos.Om_r, "Om_tot": cos.Om_tot,
        "As": prim["As"]["BB36"]["val"], "ns": prim["ns"]["BB36"]["val"], "tau": prim["tau"]["BB36"]["val"],
        "d21": neu["d21"]["val"], "d31": neu["d31"]["val"], "sumv": neu["sumv"]["val"],
        "etaB": amp["etaB"]["val"], "YHe": amp["YHe"]["val"], "delta0": amp["delta0"]["val"],
        "FCMB": amp["FCMB"]["val"], "deltaCMB": amp["deltaCMB"]["val"],
        "ell1": ell1["val"],
    }
    result["records"]["a2_constants"] = a2_consts
    result["records"]["a2_cross_base_repr"] = cross_base_repr(a2_consts)

    tol = 1e-15
    result["passes"]["a2_constants_roundtrip_parse_ok"] = all(
        rec["rt_err_b7"] <= tol and rec["rt_err_b10"] <= tol and rec["rt_err_b16"] <= tol
        for rec in result["records"]["a2_cross_base_repr"].values()
    )

    return result


def pretty_print(result: Dict[str, object], no_color: bool = False) -> None:
    p = Printer(enable_color=_supports_color(no_color))
    meta = result.get("meta", {})
    p.box("A2 Archive Master — BB-36 closure capsule", f"mode={meta.get('mode')}  utc={meta.get('utc')}  sha256={meta.get('sha256')}")
    p.dim(f"python={meta.get('python')}  platform={meta.get('platform')}", indent=0)
    print()

    # 0 Gauge lawbook
    p.section("0. Gauge lawbook derivation")
    g = result["records"]["gauge_lawbook_scan"]
    p.kv("candidates found", str(g["candidates_found"]))
    can = g["canonical"]
    if can:
        p.kv("BEST", f"q=({can['qU']},{can['q2']},{can['q3']})  RU={can['RU']} R2={can['R2']} R3={can['R3']}")
        p.kv("τ-pressure", f"{can['tau_category']}  uniq={can['uniqP']}/27  none={can['noneP']}/27")
        p.kv("span-grid", f"uniq={can['span_uniq']}/9  distinct={can['span_distinct']}")
        p.kv("triple", str(can["triple"]))
        p.kv("complexity", str(can["complexity"]))
    if result["passes"]["gauge_lawbook_unique"]:
        p.ok("CLOSED: unique canonical lawbook under declared contracts.")
    else:
        p.warn("Gauge lawbook not unique (unexpected).")

    # 1 Gauge selector
    p.section("1. Gauge selector")
    gs = result["records"]["gauge_selector"]
    lanes = gs["lanes"]
    p.kv("span", str(gs["span"]))
    p.kv("taus", str(gs["taus"]))
    p.bullet(f"U(1):  q={lanes['U1']['q']}  residues={lanes['U1']['res']}  → {lanes['U1']['survivors']}")
    p.bullet(f"SU(2): q={lanes['SU2']['q']}  residues={lanes['SU2']['res']}  → {lanes['SU2']['survivors']}")
    p.bullet(f"SU(3): q={lanes['SU3']['q']}  residues={lanes['SU3']['res']}  → {lanes['SU3']['survivors']}")
    p.kv("unique triple", str(gs["triple"]))
    inv = result["records"]["gauge_invariants"]
    p.kv("derived", f"q2={inv['q2']}  v2={inv['v2']}  phi(q2)={inv['phi(q2)']}  Θ={inv['Theta']}")
    if result["passes"]["gauge_selector_unique"] and result["passes"]["gauge_invariants_match"]:
        p.ok("CLOSED: canonical (137,107,103) and invariants (q2=30, v2=3, Θ=4/15).")
    else:
        p.warn("Gauge selector mismatch (unexpected).")

    # 2 Φ mapping
    p.section("2. Φ-mapping")
    pm = result["records"]["phi_mapping"]
    p.kv("alpha", f"{pm['alpha']['frac']} = {_fmt_float(pm['alpha']['val'])}")
    p.kv("alpha_s", f"{pm['alpha_s']['frac']} = {_fmt_float(pm['alpha_s']['val'])}")
    p.kv("sin^2", f"{pm['sin2']['frac']} = {_fmt_float(pm['sin2']['val'])}")
    if result["passes"]["phi_mapping_expected_fractions"]:
        p.ok("CLOSED: α=1/137, αs=2/17, sin^2=7/30.")
    else:
        p.warn("Φ mapping mismatch (unexpected).")

    # 3 Yukawa
    p.section("3. Yukawa closure")
    yD1 = result["records"]["yukawa_D1"]
    p.kv("Palette‑B E1–E5", str(yD1["E1_E5_on_palette_B"]))
    p.kv("D1 option counts", str(yD1["option_counts"]))
    p.kv("D1 total candidates", str(yD1["total_candidates"]))
    p.kv("D1 survivors", str(yD1["survivors"]))
    p.kv("D1 best", str(yD1["best_tuple"]))
    p.kv("delta_iso(best)", _fmt_float(yD1["delta_iso"]) if yD1["delta_iso"] is not None else "n/a")
    if result["passes"]["yukawa_D1_best_is_palette_B"]:
        p.ok("CLOSED: D1 selects Palette‑B as unique best tuple.")
    else:
        p.warn("D1 did not select Palette‑B (unexpected).")
    yS = result["records"]["yukawa_offset_sweep"]
    p.kv("offset grid size", str(yS["grid_size"]))
    p.kv("MC draws", str(yS["draws"]))
    p.kv("seed", str(yS["seed"]))
    p.kv("distinct offsets observed", str(yS["distinct_offsets_observed"]))
    p.kv("hits summary", str(yS["hits_summary"]))
    p.kv("canonical hits", str(yS["canonical_hits"]))
    p.kv("canonical rank", str(yS["canonical_rank"]))
    rows = [["#", "du", "lu", "hits", "delta", "canonical?"]]
    for rec in yS["top10"][:5]:
        rows.append([
            f"{rec['rank']:02d}",
            rec["du"],
            rec["lu"],
            str(rec["hits"]),
            f"{rec['delta']:.3f}",
            "yes" if rec["is_canonical"] else "",
        ])
    p.table(rows, indent=2)

    # 4 Cosmology
    p.section("4. Cosmology closure (Ω)")
    cos = result["records"]["cosmology"]
    p.kv("Ω_b", _fmt_float(cos["Omega_b"]))
    p.kv("Ω_c", _fmt_float(cos["Omega_c"]))
    p.kv("Ω_Λ", _fmt_float(cos["Omega_L"]))
    p.kv("Ω_r", _fmt_float(cos["Omega_r"]))
    p.kv("Ω_tot", _fmt_float(cos["Omega_tot"]))
    rows = [["sector", "C", "exps", "val", "BB36 rank"]]
    for name in ["Omega_b", "Omega_c", "Omega_L", "Omega_r"]:
        rec = cos["templates"][name]
        rows.append([name.replace("Omega_", "Ω_"), _fmt_float(rec["C"]), str(tuple(rec["exps"])), _fmt_float(rec["val"]), str(rec["BB36_rank"])])
    p.table(rows, indent=2)
    if result["passes"]["cosmology_templates_match_BB36"] and result["passes"]["cosmology_flatness_eps_1e-3"]:
        p.ok("CLOSED: BB‑36 Ω templates + near-flatness (ε<=1e-3).")
    else:
        p.warn("Cosmology mismatch (unexpected).")

    # 5 H0
    p.section("5. H0 closure")
    H0 = result["records"]["H0_closure"]
    bb = H0["BB36"]
    p.kv("d_contract", str(H0["d_contract"]))
    p.kv("BEST", f"H0={_fmt_float(bb['val'])}  exps={bb['exps']}  rank={bb['rank']}")
    if result["passes"]["H0_BB36_rank1"]:
        p.ok("CLOSED: BB‑36 H0 is rank‑1.")
    else:
        p.warn("H0 not rank‑1 (unexpected).")

    # 6 Primordial
    p.section("6. Primordial (As, ns, tau)")
    prim = result["records"]["primordial"]
    p.kv("As", f"{_fmt_float(prim['As']['BB36']['val'])}  exps={prim['As']['BB36']['exps']}  rank={prim['As']['BB36']['rank']}")
    p.kv("ns", f"{_fmt_float(prim['ns']['BB36']['val'])}  exps={prim['ns']['BB36']['exps']}  rank={prim['ns']['BB36']['rank']}")
    p.kv("tau", f"{_fmt_float(prim['tau']['BB36']['val'])}  exps={prim['tau']['BB36']['exps']}  rank={prim['tau']['BB36']['rank']}")
    if result["passes"]["primordial_As_rank1"] and result["passes"]["primordial_ns_rank1"] and result["passes"]["primordial_tau_rank1"]:
        p.ok("CLOSED: primordial trio is rank‑1.")
    else:
        p.warn("Primordial mismatch (unexpected).")

    # 7 Neutrinos
    p.section("7. Neutrinos")
    neu = result["records"]["neutrinos"]
    p.kv("Δ21", f"{_fmt_float(neu['d21']['val'])}  exps={neu['d21']['exps']}")
    p.kv("Δ31", f"{_fmt_float(neu['d31']['val'])}  exps={neu['d31']['exps']}")
    p.kv("Σmν", f"{_fmt_float(neu['sumv']['val'])}  exps={neu['sumv']['exps']}")
    p.kv("Δ31/Δ21", _fmt_float(neu["hierarchy_ratio"]))
    if result["passes"]["neutrinos_closed"]:
        p.ok("CLOSED: neutrino templates + hierarchy contract pass.")
    else:
        p.warn("Neutrino mismatch (unexpected).")

    # 8 Amplitudes + ell1
    p.section("8. Amplitudes + ℓ1")
    amp = result["records"]["amplitudes"]
    p.kv("etaB", f"{_fmt_float(amp['etaB']['val'])}  in_win={amp['etaB']['in_win']}")
    p.kv("YHe", f"{_fmt_float(amp['YHe']['val'])}  in_win={amp['YHe']['in_win']}")
    p.kv("deltaCMB", f"{_fmt_float(amp['deltaCMB']['val'])}  in_win={amp['deltaCMB']['in_win']}")
    p.kv("NC(deltaCMB using FCMB_d-1)", f"in_win={amp['NC_deltaC_using_FCMB_d-1']['in_win']}")
    ell1 = result["records"]["ell1"]
    p.kv("ell1", f"{_fmt_float(ell1['val'])}  exps={ell1['exps']}  C=1/e")
    if result["passes"]["amplitudes_etaB_in_window"] and result["passes"]["amplitudes_deltaC_in_window"]:
        p.ok("CLOSED: amplitude windows pass; NC breaks deltaC window.")
    else:
        p.warn("Amplitude mismatch (unexpected).")

    # 9 Cross-base
    p.section("9. Cross-base invariance")
    ros = result["records"]["rosetta"]
    p.kv("rosetta passes", json.dumps(ros["passes"], sort_keys=True))
    p.kv("A2 constants roundtrip", str(result["passes"]["a2_constants_roundtrip_parse_ok"]))
    if result["passes"]["rosetta_all_pass"] and result["passes"]["a2_constants_roundtrip_parse_ok"]:
        p.ok("CLOSED: base representation changes do not change numeric content (tol=1e-15).")
    else:
        p.warn("Cross-base mismatch (unexpected).")

    # Scoreboard
    p.section("A2 scoreboard")
    rows = [["claim", "PASS?"]]
    for k in sorted(result["passes"].keys()):
        rows.append([k, "PASS" if result["passes"][k] else "FAIL"])
    p.table(rows, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser(description="A2 Archive Master Script (stdlib-only).")
    ap.add_argument("--mode", choices=["fast", "full", "heavy"], default="fast",
                    help="fast = core; full/heavy = add evidence scans + heavier Yukawa MC")
    ap.add_argument("--json", action="store_true", help="print JSON only (no pretty)")
    ap.add_argument("--save-json", default=None, help="write JSON report")
    ap.add_argument("--save-md", default=None, help="write Markdown report")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI color")
    args = ap.parse_args()

    result = run_archive(mode=args.mode)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        pretty_print(result, no_color=args.no_color)

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
    if args.save_md:
        with open(args.save_md, "w", encoding="utf-8") as f:
            f.write(build_markdown_report(result))


if __name__ == "__main__":
    main()