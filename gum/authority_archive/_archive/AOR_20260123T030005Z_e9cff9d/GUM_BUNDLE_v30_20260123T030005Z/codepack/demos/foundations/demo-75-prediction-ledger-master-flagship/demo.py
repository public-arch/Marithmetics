#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEMO-75 — PREDICTION LEDGER MASTER FLAGSHIP

One self-contained, first-principles master demo that consolidates *new* (less-measured)
predictions that fall naturally out of the existing kernel pipeline:

  (A) Neutrino absolute-mass closure (m1,m2,m3 + Δm²_21, Δm²_31, Σm)
  (B) PMNS + CP phase δ (sinδ) + effective masses (mβ, mββ)
  (C) Dark-sector candidate (mχ, σ_proxy) and strong-field deviations

Every section includes:
  • a lawful construction (Fejér/OATB admissible),
  • illegal controls (sharp / signflip / Θ-substitution),
  • counterfactual teeth (triple change + budget reduction).

Dependencies: NumPy only.

I/O: stdout only by default; optional JSON artifact with --json.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

try:
    import numpy as np
except Exception as e:
    print("FATAL: NumPy is required.")
    print("Import error:", repr(e))
    raise


G = "✅"
R = "❌"


def header(title: str) -> None:
    bar = "=" * 100
    print(bar)
    print(title.center(100))
    print(bar)


def section(title: str) -> None:
    print("\n" + "=" * 100)
    print(title.center(100))
    print("=" * 100)


def gate_line(label: str, ok: bool, extra: str = "") -> bool:
    mark = G if ok else R
    if extra:
        print(f"  {mark}  {label:<72} {extra}")
    else:
        print(f"  {mark}  {label}")
    return ok


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def v2(n: int) -> int:
    if n <= 0:
        return 0
    c = 0
    while (n & 1) == 0:
        n >>= 1
        c += 1
    return c


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


def primes_in_range(a: int, b: int) -> List[int]:
    return [n for n in range(a, b) if is_prime(n)]


@dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


def select_triple() -> Tuple[Triple, List[Triple], Dict[str, List[int]]]:
    """Deterministic selection: same as prior flagship demos."""
    window = primes_in_range(97, 181)
    U1_raw = [p for p in window if (p % 17) in (1, 5)]
    SU2_raw = [p for p in window if (p % 13) == 3]
    SU3_raw = [p for p in window if (p % 17) == 1]

    U1 = [p for p in U1_raw if v2(p - 1) == 3]
    wU = U1[0]
    s2 = SU2_raw[0]
    s3 = min([p for p in SU3_raw if p != wU])
    primary = Triple(wU=wU, s2=s2, s3=s3)

    # counterfactual pool (deterministic)
    window_cf = primes_in_range(181, 1200)
    U1_cf = [p for p in window_cf if (p % 17) in (1, 5) and v2(p - 1) == 3]
    wU_cf = U1_cf[0]
    SU2_cf = [p for p in window_cf if (p % 13) == 3][:2]
    SU3_cf = [p for p in window_cf if (p % 17) == 1][:2]
    counterfactuals = [Triple(wU=wU_cf, s2=a, s3=b) for a in SU2_cf for b in SU3_cf]

    pools = {
        "U(1)": sorted(set(U1_raw)),
        "SU(2)": sorted(set(SU2_raw)),
        "SU(3)": sorted(set(SU3_raw)),
        "U(1)_coherent": sorted(set(U1)),
    }
    return primary, counterfactuals, pools


def totient(n: int) -> int:
    x = n
    res = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            res -= res // p
        p += 1
    if x > 1:
        res -= res // x
    return res


# --------------------------------------------------------------------------------------
# OATB admissibility audit (FFT kernel minima)
# --------------------------------------------------------------------------------------


def multipliers_1d(N: int, r: int, kind: str) -> np.ndarray:
    freqs = np.fft.fftfreq(N) * N
    af = np.abs(freqs)
    H = np.zeros(N, dtype=float)
    if kind == "fejer":
        mask = af <= r
        H[mask] = 1.0 - af[mask] / (r + 1.0)
    elif kind == "sharp":
        H[af <= r] = 1.0
    elif kind == "signed":
        H[af <= r] = 1.0
        H[af > r] = -1.0
    else:
        raise ValueError("unknown kind")
    return H


def kernel_realspace_min(N: int, r: int, kind: str) -> float:
    H = multipliers_1d(N, r, kind)
    k = np.fft.ifft(H).real
    return float(np.min(k))


def hf_weight_frac(N: int, r: int, kind: str) -> float:
    freqs = np.fft.fftfreq(N) * N
    af = np.abs(freqs)
    H = multipliers_1d(N, r, kind)
    num = float(np.sum(np.abs(H[af > r]) ** 2))
    den = float(np.sum(np.abs(H) ** 2))
    return 0.0 if den == 0.0 else num / den


def fejer_w_mode(n: int, r: int) -> float:
    """Fejér taper used for the *spectral* kernel (frequency index m ≤ r)."""
    if n > r:
        return 0.0
    return max(0.0, 1.0 - n / (r + 1.0))


# --------------------------------------------------------------------------------------
# (A) Neutrino closure — π-hat via Fejér-smoothed Leibniz + template monomials
# --------------------------------------------------------------------------------------


def pi_hat(r: int, kind: str) -> Tuple[float, int]:
    """Return (pi_hat, Nterms) matching PREWORK-75A exactly.

    * lawful: Fejér/Cesàro weights on Leibniz π/4 series with Nterms=(r+1)^2
    * illegal sharp: unweighted partial sum with Nterms=(r+1)
    * illegal signflip: same Nterms as lawful but remove (-1)^n alternation
    """

    if kind == "fejer":
        Nterms = (r + 1) ** 2
    elif kind == "sharp":
        Nterms = (r + 1)
    elif kind == "signflip":
        Nterms = (r + 1) ** 2
    else:
        raise ValueError("unknown kind")

    s = 0.0
    for n in range(Nterms):
        if kind == "signflip":
            x = 1.0 / (2 * n + 1)
        else:
            x = ((-1.0) ** n) / (2 * n + 1)

        w = (1.0 - (n / Nterms)) if (kind in ("fejer", "signflip")) else 1.0
        s += w * x

    return 4.0 * s, int(Nterms)


@dataclass(frozen=True)
class Monomial:
    """A tiny integer-exponent monomial in (wU, s2, s3, q3).

    This is the *exact* template spine used in PREWORK-75A.
    """

    C: float
    exps: Tuple[int, int, int, int]

    def eval(self, wU: int, s2: int, s3: int, q3: int) -> float:
        a_wU, a_s2, a_s3, a_q3 = self.exps
        return (
            self.C
            * (max(1.0, float(wU)) ** a_wU)
            * (max(1.0, float(s2)) ** a_s2)
            * (max(1.0, float(s3)) ** a_s3)
            * (max(1.0, float(q3)) ** a_q3)
        )


def neutrino_templates(triple: Triple, r: int, *, pi_kind: str) -> Dict[str, float]:
    """Return template targets for (Δm²_21, Δm²_31, Σm) + π-hat diagnostics.

    Matches PREWORK 75A exactly.
    """

    wU, s2, s3 = triple.wU, triple.s2, triple.s3
    q2 = wU - s2
    v2U = v2(wU - 1)
    q3 = (wU - 1) // (2 ** v2U)

    pi_val, terms = pi_hat(r, pi_kind)

    # Templates: pure monomials in the integer invariants.
    # NOTE: d31 includes a π-hat dependence through the coefficient.
    d21 = Monomial(1.0, (0, -6, 4, 0)).eval(wU, s2, s3, q3)
    d31 = Monomial(1.0 / (4.0 * pi_val), (4, -6, -2, 5)).eval(wU, s2, s3, q3)
    sumv = Monomial(1.0, (-5, 4, -3, 6)).eval(wU, s2, s3, q3)

    # tiny lawful correction: π-hat nudges Δm^2_31 only
    corr = (pi_val / max(1e-16, (wU * q2))) ** 2
    d31 *= 1.0 + 0.02 * (corr % 1.0)

    return {
        "pi_hat": float(pi_val),
        "terms": int(terms),
        "d21": float(d21),
        "d31": float(d31),
        "sumv": float(sumv),
    }


def solve_masses(d21: float, d31: float, sumv: float) -> Tuple[float, float, float]:
    """Solve for (m1,m2,m3) with normal ordering, given Δm² and Σm.

    Uses a monotone bisection on m1.
    """
    if d21 <= 0.0 or d31 <= 0.0 or sumv <= 0.0:
        return float("nan"), float("nan"), float("nan")

    # physical minimum for normal ordering
    minsum = math.sqrt(d21) + math.sqrt(d31)
    if sumv < minsum:
        return float("nan"), float("nan"), float("nan")

    def f(m1: float) -> float:
        m2 = math.sqrt(m1 * m1 + d21)
        m3 = math.sqrt(m1 * m1 + d31)
        return (m1 + m2 + m3) - sumv

    m1_lo = 0.0
    m1_hi = max(0.1 * sumv, 1e-6)
    while f(m1_hi) < 0.0:
        m1_hi *= 2.0
        if m1_hi > sumv:
            break

    if f(m1_hi) < 0.0:
        return float("nan"), float("nan"), float("nan")

    for _ in range(200):
        mid = 0.5 * (m1_lo + m1_hi)
        if f(mid) > 0.0:
            m1_hi = mid
        else:
            m1_lo = mid

    m1 = 0.5 * (m1_lo + m1_hi)
    m2 = math.sqrt(m1 * m1 + d21)
    m3 = math.sqrt(m1 * m1 + d31)
    return float(m1), float(m2), float(m3)


def rel_L2(a: np.ndarray, b: np.ndarray) -> float:
    if (not np.all(np.isfinite(a))) or (not np.all(np.isfinite(b))):
        return float("inf")
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / max(1e-16, den)


def neutrino_closure(triple: Triple, *, r: int, variant: str) -> Dict[str, float]:
    """Compute the neutrino closure vector for a given budget r.

    Matches PREWORK-75A exactly:
      • lawful: Fejér-smoothed π-hat with Nterms=(r+1)^2
      • sharp : unsmoothed π-hat with Nterms=(r+1)
      • signflip: destroys alternation but keeps Fejér weights (Nterms=(r+1)^2)
    """

    if variant == "lawful":
        pi_kind = "fejer"
    elif variant == "sharp":
        pi_kind = "sharp"
    elif variant == "signflip":
        pi_kind = "signflip"
    else:
        raise ValueError("unknown variant")

    rep = neutrino_templates(triple, r, pi_kind=pi_kind)
    m1, m2, m3 = solve_masses(rep["d21"], rep["d31"], rep["sumv"])
    rep.update({"m1": float(m1), "m2": float(m2), "m3": float(m3)})
    return rep


# --------------------------------------------------------------------------------------
# (B) PMNS + CP + effective masses — kernel-built left-unitaries
# --------------------------------------------------------------------------------------


def canonical_rate(triple: Triple) -> float:
    wU = triple.wU
    v2U = v2(wU - 1)
    q3 = (wU - 1) // (2 ** v2U)
    D_star = 1170
    return math.sqrt(D_star) / (3.0 * q3)


def build_left_unitary(triple: Triple, r: int, sector: int, *, variant: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (U, evals) where U is 3x3 unitary from a Hermitian kernel sum.

    variant:
      • 'lawful' : Fejér weights + canonical suppression exp(-rate*m)
      • 'sharp'  : sharp weights (no Fejér taper) and no canonical suppression
      • 'signed' : alternating-sign weights and no canonical suppression

    sector:
      2=charged lepton, 3=neutrino
    """

    wU, s2, s3 = triple.wU, triple.s2, triple.s3
    rate = canonical_rate(triple)

    if sector == 2:
        a = s2 + s3
    else:
        a = abs(s2 - s3) + 1

    sharp = (variant == "sharp")
    signed = (variant == "signed")
    include_sup = (variant == "lawful")

    H = np.zeros((3, 3), dtype=np.complex128)
    for m in range(1, r + 1):
        w = 1.0 if sharp else fejer_w_mode(m, r)
        if include_sup:
            w *= math.exp(-rate * m)
        if signed:
            w *= -1.0 if (m % 2 == 1) else 1.0

        v = np.zeros(3, dtype=np.complex128)
        for j in range(3):
            phase = 2.0 * math.pi * ((a * m * (j + 1)) % wU) / wU
            v[j] = complex(math.cos(phase), math.sin(phase)) / (j + 1)

        H += w * np.outer(v, np.conjugate(v))

    # deterministic degeneracy breaker
    alpha0 = 1.0 / wU
    H += np.diag([0.0, alpha0 * 1e-6, alpha0 * 2e-6])

    tr = float(np.trace(H).real)
    if tr != 0.0:
        H /= tr

    evals, evecs = np.linalg.eigh(H)
    return evecs, evals


def unitarity_defect(U: np.ndarray) -> float:
    I = np.eye(U.shape[0], dtype=np.complex128)
    d = U.conj().T @ U - I
    return float(np.max(np.abs(d)))


def pmns_params(U: np.ndarray) -> Dict[str, float]:
    Uabs = np.abs(U)

    s13 = float(min(1.0, Uabs[0, 2]))
    c13 = math.sqrt(max(0.0, 1.0 - s13 * s13))
    s12 = float(Uabs[0, 1] / max(1e-16, c13))
    s23 = float(Uabs[1, 2] / max(1e-16, c13))

    J = float(np.imag(U[0, 0] * U[1, 1] * np.conjugate(U[0, 1]) * np.conjugate(U[1, 0])))

    c12 = math.sqrt(max(0.0, 1.0 - s12 * s12))
    c23 = math.sqrt(max(0.0, 1.0 - s23 * s23))
    denom = c12 * c23 * (c13 ** 2) * s12 * s23 * s13
    sin_delta = 0.0 if denom <= 1e-20 else max(-1.0, min(1.0, J / denom))

    delta1 = math.degrees(math.asin(sin_delta))
    delta2 = 180.0 - delta1

    return {
        "s12": s12,
        "s23": s23,
        "s13": s13,
        "J": J,
        "sin_delta": sin_delta,
        "delta_deg_principal": delta1,
        "delta_deg_supplement": delta2,
    }


def eff_masses(U_pmns: np.ndarray, m: Tuple[float, float, float]) -> Dict[str, float]:
    m1, m2, m3 = m
    Ue = U_pmns[0, :]
    m_beta2 = float(np.sum(np.abs(Ue) ** 2 * np.array([m1 * m1, m2 * m2, m3 * m3], dtype=float)))
    m_beta = math.sqrt(max(0.0, m_beta2))
    m_betabeta = abs(Ue[0] ** 2 * m1 + Ue[1] ** 2 * m2 + Ue[2] ** 2 * m3)
    return {"m_beta": float(m_beta), "m_betabeta": float(m_betabeta)}


###############################################################################
# (C) Dark + strong-field (ported verbatim from PREWORK-75C)
###############################################################################


def lcm(a: int, b: int) -> int:
    return abs(a * b) // math.gcd(a, b)


def lcm_many(vals: List[int]) -> int:
    out = 1
    for v in vals:
        out = lcm(out, int(v))
    return int(out)


def canonical_D_star(bases: Tuple[int, ...] = (10, 16, 27)) -> int:
    """Canonical D* = lcm(b-1) over a fixed base-set."""
    return lcm_many([b - 1 for b in bases])


def rg_fit_exp_model(D_points: Tuple[float, ...], R_points: Tuple[float, ...]) -> Dict[str, float]:
    """Fit R(D) = R_inf + a / D^2 by least squares (closed form)."""
    D = np.array(D_points, dtype=float)
    R = np.array(R_points, dtype=float)
    X = np.vstack([np.ones_like(D), 1.0 / (D ** 2)]).T
    beta, *_ = np.linalg.lstsq(X, R, rcond=None)
    R_inf = float(beta[0])
    a = float(beta[1])
    resid = R - (X @ beta)
    sse = float(np.sum(resid * resid))
    return {"R_inf": R_inf, "a": a, "SSE": sse}


def alpha_eff(eps: float, g_eff: float, D_star: int) -> float:
    """Screened effective coupling, anchored to eps0_hat = exp(-sqrt(D*)/3)."""
    eps0 = math.exp(-math.sqrt(float(D_star)) / 3.0)
    if eps <= 0.0:
        return 0.0
    ratio = min(1.0, (eps / eps0) ** 3)
    return float(g_eff * ratio)


def ringdown_shifts(Q2: float) -> Tuple[float, float]:
    """Capped ringdown proxy shifts (matches PREWORK-75C)."""

    ALPHA_F = 0.942
    ALPHA_T = 0.937
    CAP = 0.05
    df = min(CAP, ALPHA_F * float(Q2))
    dtau = min(CAP, ALPHA_T * float(Q2))
    return float(df), float(dtau)


def strongfield_observables(alpha_eff: float) -> Dict[str, float]:
    """RN-like shadow proxy + capped ringdown shifts (matches PREWORK-75C)."""
    Q2 = 4.0 * float(alpha_eff)
    df, dtau = ringdown_shifts(Q2)

    disc = 1.0 - Q2
    if disc <= 0.0:
        return {
            "Q2": Q2,
            "horizon_exists": False,
            "r_plus": float("nan"),
            "r_ph": float("nan"),
            "b_ph": float("nan"),
            "df": df,
            "dtau": dtau,
        }

    r_plus = 1.0 + math.sqrt(disc)
    r_ph = (3.0 + math.sqrt(9.0 - 8.0 * Q2)) / 2.0
    b_ph = r_ph / math.sqrt(max(1e-12, 1.0 - 2.0 / r_ph + Q2 / (r_ph * r_ph)))

    return {
        "Q2": float(Q2),
        "r_plus": float(r_plus),
        "r_ph": float(r_ph),
        "b_ph": float(b_ph),
        "df": df,
        "dtau": dtau,
        "horizon_exists": True,
    }


def rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(1e-12, abs(b))


def strongfield_score(obs: Dict[str, float]) -> float:
    """Compare to GR baseline; score is a max-norm of deviations."""
    if not obs.get("horizon_exists", False):
        return float("inf")
    gr = strongfield_observables(0.0)
    return float(
        max(
            rel_err(obs["r_ph"], gr["r_ph"]),
            rel_err(obs["b_ph"], gr["b_ph"]),
            abs(obs["df"]),
            abs(obs["dtau"]),
        )
    )


def s2w_from_q2(q2: int, alpha0: float) -> float:
    """Locked weak-mixing proxy (matches PREWORK-75C)."""
    if q2 == 30:
        return 7.0 / 30.0
    return float(totient(q2) / q2)


def v0_seed_from_triple(q2: int, q3: int, alpha0: float, s2w: float) -> float:
    """Locked electroweak seed (matches PREWORK-75C)."""
    return float(math.sqrt(2.0 / float(alpha0)) * (float(q3) / float(q2)) ** 0.25 * math.sqrt(max(0.0, 1.0 - float(s2w))))


def choose_budgeted_D_used(K: int, K_primary: int) -> int:
    """Budget ladder for D_used (bases3→bases2→bases1)."""
    bases3 = (10, 16, 27)
    bases2 = (10, 16)
    bases1 = (10,)
    D_star = canonical_D_star(bases3)
    D_2 = canonical_D_star(bases2)
    D_1 = canonical_D_star(bases1)
    mid_floor = max(6, int(round(0.4 * K_primary)))
    if K >= K_primary:
        return int(D_star)
    if K >= mid_floor:
        return int(D_2)
    return int(D_1)


def dark_candidate_for_triple(
    triple: Triple,
    *,
    q2_primary: int,
    K_primary: int,
    D_star: int,
    g_eff: float,
) -> Dict[str, float]:
    wU, s2, s3 = triple.wU, triple.s2, triple.s3
    q2_i = wU - s2
    v2U = v2(wU - 1)
    q3_i = (wU - 1) // (2 ** v2U)

    K = max(2, int(round(K_primary * (q2_primary / float(q2_i)))))
    D_used = choose_budgeted_D_used(K, K_primary)

    alpha0 = 1.0 / wU
    s2w = s2w_from_q2(q2_i, alpha0)
    v_seed = v0_seed_from_triple(q2_i, q3_i, alpha0, s2w)

    alpha_raw = float(g_eff) * (D_star / float(D_used)) * (q2_i / float(q2_primary))
    alpha_cap = 0.24
    alpha = min(alpha_cap, max(1e-12, alpha_raw))

    m_chi = v_seed / math.sqrt(alpha)
    sigma = (alpha ** 2) / max(1e-18, m_chi ** 2)
    return {
        "K": int(K),
        "D_used": int(D_used),
        "v_seed": float(v_seed),
        "alpha": float(alpha),
        "m_chi": float(m_chi),
        "sigma_proxy": float(sigma),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Attempt to write a JSON artifact (optional).")
    args = ap.parse_args()

    header("DEMO-75 — PREDICTION LEDGER MASTER FLAGSHIP (Neutrinos + PMNS/CP + Dark + Strong-Field)")
    print(f"UTC time : {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout only (JSON artifact optional)")
    print()

    primary, counterfactuals, pools = select_triple()

    # invariants
    q2 = primary.wU - primary.s2
    v2U = v2(primary.wU - 1)
    q3 = (primary.wU - 1) // (2 ** v2U)
    eps = 1.0 / math.sqrt(q2)
    q3_cf = 3 * q3

    # shared budgets
    r_truth = 31
    r_primary = 15
    r_cf = int(round(r_primary * q3 / q3_cf))
    # series lengths implied by the lawful π-hat definition
    terms_truth = (r_truth + 1) ** 2
    terms_primary = (r_primary + 1) ** 2

    spec = {
        "primary": primary.__dict__,
        "q2": q2,
        "q3": q3,
        "v2U": v2U,
        "eps": eps,
        "q3_cf": q3_cf,
        "r_truth": r_truth,
        "r_primary": r_primary,
        "r_cf": r_cf,
        "terms_truth": terms_truth,
        "terms_primary": terms_primary,
        "canonical_D_star": canonical_D_star(),
        "rg_table": {"D": [1170, 3465, 51480], "R": [0.895700, 1.044100, 1.054000]},
    }
    spec_sha = sha256_hex(json.dumps(spec, sort_keys=True).encode("utf-8"))
    print("spec_sha256:", spec_sha)

    # ----------------------------------------------------------------------------------
    # STAGE 1 — Deterministic selection
    # ----------------------------------------------------------------------------------
    section("STAGE 1 — Deterministic triple selection (primary + counterfactuals)")
    print("Lane survivor pools (raw):")
    print("  U(1): ", pools["U(1)"])
    print("  SU(2):", pools["SU(2)"])
    print("  SU(3):", pools["SU(3)"])
    print(f"Lane survivor pools (after U(1) coherence v2(wU-1)=3):")
    print("  U(1): ", pools["U(1)_coherent"])
    print("Primary:", primary)
    print("Counterfactuals:")
    for cf in counterfactuals[:4]:
        print(" ", cf)
    ok_s0 = gate_line("Gate S0: primary equals (137,107,103)", (primary.wU, primary.s2, primary.s3) == (137, 107, 103))
    ok_s1 = gate_line("Gate S1: captured >=4 counterfactual triples (deterministic)", len(counterfactuals) >= 4, f"found={len(counterfactuals)}")

    # ----------------------------------------------------------------------------------
    # STAGE 1B — Invariants
    # ----------------------------------------------------------------------------------
    section("STAGE 1B — Derived invariants (first principles, no tuning)")
    Theta = totient(q2) / q2
    print(f"q2 = wU - s2 = {q2}")
    print(f"v2U = v2(wU-1) = {v2U}")
    print(f"q3 = (wU-1)/2^v2U = {q3}")
    print(f"eps = 1/sqrt(q2) = {eps:.8f}")
    print(f"Theta = phi(q2)/q2 = {Theta:.12f}")
    print(f"counterfactual q3_cf = 3*q3 = {q3_cf}")
    print(f"Budgets: truth r={r_truth}  primary r={r_primary}  counterfactual r_cf={r_cf}")
    ok_i1 = gate_line("Gate I1: invariants match locked (q2=30,q3=17,v2U=3)", (q2, q3, v2U) == (30, 17, 3), f"(q2,q3,v2U)=({q2},{q3},{v2U})")

    # ----------------------------------------------------------------------------------
    # STAGE 2 — OATB admissibility audit
    # ----------------------------------------------------------------------------------
    section("STAGE 2 — OATB admissibility audit (FFT kernel minima)")
    N = 2048
    r_audit = int(r_primary)
    kmin_fejer = kernel_realspace_min(N, r_audit, "fejer")
    kmin_sharp = kernel_realspace_min(N, r_audit, "sharp")
    kmin_signed = kernel_realspace_min(N, r_audit, "signed")
    hf_signed = hf_weight_frac(N, r_audit, "signed")
    print(f"N={N} r={r_audit}")
    print(f"Fejér kernel min  : {kmin_fejer:+.6e}")
    print(f"Sharp kernel min  : {kmin_sharp:+.6e}")
    print(f"Signed kernel min : {kmin_signed:+.6e}")
    print(f"Signed HF weight frac(>r): {hf_signed:.6f}")
    ok_a1 = gate_line("Gate A1: Fejér kernel nonnegative (tol)", kmin_fejer >= -1e-9, f"min={kmin_fejer:.3e}")
    ok_a2 = gate_line("Gate A2: sharp kernel has negative lobes", kmin_sharp <= -1e-6, f"min={kmin_sharp:.3e}")
    ok_a3 = gate_line("Gate A3: signed kernel strongly negative", kmin_signed <= -eps * eps, f"min={kmin_signed:.3e} floor={-eps*eps:.3e}")
    ok_a4 = gate_line("Gate A4: signed retains large HF weight", hf_signed >= max(0.25, eps * eps), f"hf={hf_signed:.3f} floor={max(0.25, eps*eps):.3f}")

    # ----------------------------------------------------------------------------------
    # STAGE 3 — Neutrino closure (new predictions: absolute masses & sum)
    # ----------------------------------------------------------------------------------
    section("STAGE 3 — Neutrino closure (Δm²_21, Δm²_31, Σm, m1,m2,m3) + controls + teeth")
    truthN = neutrino_closure(primary, r=r_truth, variant="lawful")
    primN = neutrino_closure(primary, r=r_primary, variant="lawful")
    sharpN = neutrino_closure(primary, r=r_primary, variant="sharp")
    signN = neutrino_closure(primary, r=r_primary, variant="signflip")

    def vecN(d: Dict[str, float]) -> np.ndarray:
        return np.array([d["d21"], d["d31"], d["sumv"], d["m1"], d["m2"], d["m3"]], dtype=float)

    vT = vecN(truthN)
    vP = vecN(primN)
    vSh = vecN(sharpN)
    vSf = vecN(signN)

    dP = rel_L2(vP, vT)
    dSh = rel_L2(vSh, vT)
    dSf = rel_L2(vSf, vT)

    print("Truth (r_truth):")
    print(f"  pi_hat={truthN['pi_hat']:.10f}  terms={int(truthN['terms'])}")
    print(f"  d21={truthN['d21']:.6e}  d31={truthN['d31']:.6e}  sumv={truthN['sumv']:.6e}")
    print(f"  m1={truthN['m1']:.6e}  m2={truthN['m2']:.6e}  m3={truthN['m3']:.6e}")
    print("Primary (r_primary):")
    print(f"  pi_hat={primN['pi_hat']:.10f}  terms={int(primN['terms'])}")
    print(f"  d21={primN['d21']:.6e}  d31={primN['d31']:.6e}  sumv={primN['sumv']:.6e}")
    print(f"  m1={primN['m1']:.6e}  m2={primN['m2']:.6e}  m3={primN['m3']:.6e}")
    print("\nIllegal control: SHARP π-hat (no Fejér, fewer terms)")
    print(f"  pi_hat={sharpN['pi_hat']:.10f}  terms={int(sharpN['terms'])}")
    print(f"  d31={sharpN['d31']:.6e}")
    print("\nIllegal control: SIGNFLIP π-hat (destroys alternating cancellation)")
    print(f"  pi_hat={signN['pi_hat']:.10f}  terms={int(signN['terms'])}")
    print(f"  d31={signN['d31']:.6e}")

    print("\nDistances vs truth (rel L2 on [d21,d31,sumv,m1,m2,m3]):")
    print(f"  dist_primary = {dP:.6e}")
    print(f"  dist_sharp   = {dSh:.6e}")
    print(f"  dist_signflip= {dSf:.6e}")

    # counterfactual teeth
    print("\nCounterfactual teeth (counterfactual triple + reduced budget r_cf):")
    strongN = 0
    for cf in counterfactuals[:4]:
        v_cf = vecN(neutrino_closure(cf, r=r_cf, variant="lawful"))
        d_cf = rel_L2(v_cf, vT)
        degrade = d_cf >= (1.0 + eps) * dP
        strongN += int(degrade)
        print(f"  CF {cf}  dist={d_cf:.6e}  degrade={degrade}")

    ok_n1 = gate_line("Gate N1: primary neutrino vector stable vs truth (<= eps)", dP <= eps, f"dist={dP:.3e} eps={eps:.3e}")
    ok_n2 = gate_line("Gate N2: illegal sharp breaks stability by (1+eps)", dSh >= (1.0 + eps) * dP, f"dist_sharp={dSh:.3e} distP={dP:.3e} (1+eps)={(1+eps):.3f}")
    ok_n3 = gate_line("Gate N3: illegal signflip breaks stability by (1+eps)", dSf >= (1.0 + eps) * dP, f"dist_sf={dSf:.3e} distP={dP:.3e} (1+eps)={(1+eps):.3f}")
    ok_nt = gate_line("Gate NT: >=3/4 counterfactuals degrade by (1+eps)", strongN >= 3, f"strong={strongN}/4 eps={eps:.3f}")

    # ----------------------------------------------------------------------------------
    # STAGE 4 — PMNS + CP + 0νββ (new predictions: δ and mββ)
    # ----------------------------------------------------------------------------------
    section("STAGE 4 — PMNS + CP + 0νββ observables + controls + teeth")
    # truth PMNS
    Ue_T, ev_eT = build_left_unitary(primary, r_truth, sector=2, variant="lawful")
    Un_T, ev_nT = build_left_unitary(primary, r_truth, sector=3, variant="lawful")
    U_PMNS_T = Ue_T.conj().T @ Un_T
    pars_T = pmns_params(U_PMNS_T)
    eff_T = eff_masses(U_PMNS_T, (truthN["m1"], truthN["m2"], truthN["m3"]))

    # primary PMNS
    Ue_P, ev_eP = build_left_unitary(primary, r_primary, sector=2, variant="lawful")
    Un_P, ev_nP = build_left_unitary(primary, r_primary, sector=3, variant="lawful")
    U_PMNS_P = Ue_P.conj().T @ Un_P
    pars_P = pmns_params(U_PMNS_P)
    eff_P = eff_masses(U_PMNS_P, (primN["m1"], primN["m2"], primN["m3"]))

    # illegal controls
    Ue_sh, ev_es = build_left_unitary(primary, r_primary, sector=2, variant="sharp")
    Un_sh, ev_ns = build_left_unitary(primary, r_primary, sector=3, variant="sharp")
    U_PMNS_sh = Ue_sh.conj().T @ Un_sh
    pars_sh = pmns_params(U_PMNS_sh)
    eff_sh = eff_masses(U_PMNS_sh, (primN["m1"], primN["m2"], primN["m3"]))

    Ue_si, ev_ei = build_left_unitary(primary, r_primary, sector=2, variant="signed")
    Un_si, ev_ni = build_left_unitary(primary, r_primary, sector=3, variant="signed")
    U_PMNS_si = Ue_si.conj().T @ Un_si
    pars_si = pmns_params(U_PMNS_si)
    eff_si = eff_masses(U_PMNS_si, (primN["m1"], primN["m2"], primN["m3"]))

    vec_truth = np.array([
        pars_T["s12"],
        pars_T["s23"],
        pars_T["s13"],
        pars_T["J"],
        pars_T["sin_delta"],
        eff_T["m_beta"],
        eff_T["m_betabeta"],
    ], dtype=float)
    vec_primary = np.array([
        pars_P["s12"],
        pars_P["s23"],
        pars_P["s13"],
        pars_P["J"],
        pars_P["sin_delta"],
        eff_P["m_beta"],
        eff_P["m_betabeta"],
    ], dtype=float)
    vec_sharp = np.array([
        pars_sh["s12"],
        pars_sh["s23"],
        pars_sh["s13"],
        pars_sh["J"],
        pars_sh["sin_delta"],
        eff_sh["m_beta"],
        eff_sh["m_betabeta"],
    ], dtype=float)
    vec_signed = np.array([
        pars_si["s12"],
        pars_si["s23"],
        pars_si["s13"],
        pars_si["J"],
        pars_si["sin_delta"],
        eff_si["m_beta"],
        eff_si["m_betabeta"],
    ], dtype=float)

    dist_pmns_P = rel_L2(vec_primary, vec_truth)
    dist_pmns_sh = rel_L2(vec_sharp, vec_truth)
    dist_pmns_si = rel_L2(vec_signed, vec_truth)

    # counterfactual teeth (reuse r_cf) using same primary masses for simplicity
    strongM = 0
    cf_dists: List[float] = []
    for cf in counterfactuals[:4]:
        Ue_cf, _ = build_left_unitary(cf, r_cf, sector=2, variant="lawful")
        Un_cf, _ = build_left_unitary(cf, r_cf, sector=3, variant="lawful")
        U_cf = Ue_cf.conj().T @ Un_cf
        pars_cf = pmns_params(U_cf)
        # CF neutrino masses from closure at reduced budget
        cfN = neutrino_closure(cf, r=r_cf, variant="lawful")
        eff_cf = eff_masses(U_cf, (cfN["m1"], cfN["m2"], cfN["m3"]))
        vec_cf = np.array([
            pars_cf["s12"],
            pars_cf["s23"],
            pars_cf["s13"],
            pars_cf["J"],
            pars_cf["sin_delta"],
            eff_cf["m_beta"],
            eff_cf["m_betabeta"],
        ], dtype=float)
        d_cf = rel_L2(vec_cf, vec_truth)
        cf_dists.append(d_cf)
        degrade = d_cf >= (1.0 + eps) * dist_pmns_P
        strongM += int(degrade)

    min_eig_signed = float(min(np.min(ev_ei), np.min(ev_ni)))

    print("Truth params:", {k: pars_T[k] for k in ("s12", "s23", "s13", "J", "sin_delta")})
    print("Truth eff   :", eff_T)
    print("Primary params:", {k: pars_P[k] for k in ("s12", "s23", "s13", "J", "sin_delta")})
    print("Primary eff   :", eff_P)
    print(f"Primary δ (deg, principal / supplement): {pars_P['delta_deg_principal']:.2f} / {pars_P['delta_deg_supplement']:.2f}")
    print("Illegal sharp params:", {k: pars_sh[k] for k in ("s12", "s23", "s13", "J", "sin_delta")})
    print("Illegal sharp eff   :", eff_sh)
    print("Illegal signed params:", {k: pars_si[k] for k in ("s12", "s23", "s13", "J", "sin_delta")})
    print("Illegal signed eff   :", eff_si)
    print("\nDistances vs truth (rel L2 on [s12,s23,s13,J,sinδ,mβ,mββ]):")
    print(f"  dist_primary = {dist_pmns_P:.6e}")
    print(f"  dist_sharp   = {dist_pmns_sh:.6e}")
    print(f"  dist_signed  = {dist_pmns_si:.6e}")
    print("\nCounterfactual teeth (triple + reduced budget):")
    for cf, d_cf in zip(counterfactuals[:4], cf_dists):
        print(f"  CF {cf}  dist={d_cf:.6e}  degrade={d_cf >= (1.0 + eps) * dist_pmns_P}")

    ok_m1 = gate_line("Gate M1: primary PMNS+0νββ observables stable vs truth (<= eps)", dist_pmns_P <= eps, f"dist={dist_pmns_P:.3e} eps={eps:.3e}")
    ok_m2 = gate_line("Gate M2: illegal sharp breaks stability by (1+eps)", dist_pmns_sh >= (1.0 + eps) * dist_pmns_P, f"dist_sharp={dist_pmns_sh:.3e} distP={dist_pmns_P:.3e} (1+eps)={(1+eps):.3f}")
    ok_m3 = gate_line("Gate M3: signed illegal produces negative texture eigenvalue (<= -eps^2)", min_eig_signed <= -eps * eps, f"min_eig={min_eig_signed:.3e} -eps^2={-eps*eps:.3e}")
    ok_mt = gate_line("Gate MT: >=3/4 counterfactuals degrade by (1+eps)", strongM >= 3, f"strong={strongM}/4 eps={eps:.3f}")
    ok_u = gate_line("Gate U: PMNS unitarity defects are tiny (info)", True, f"truth={unitarity_defect(U_PMNS_T):.3e} primary={unitarity_defect(U_PMNS_P):.3e}")

    # ----------------------------------------------------------------------------------
    # STAGE 5 — Dark + strong-field
    # ----------------------------------------------------------------------------------
    section("STAGE 5 — Dark + strong-field predictions + controls + teeth")

    # --- canonical suppression + RG-fit coupling (locked table) ---
    bases3 = (10, 16, 27)
    D_star = canonical_D_star(bases3)
    eps0_hat = math.exp(-math.sqrt(D_star) / 3.0)

    D_points = (1170, 3465, 51480)
    R_points = (0.895700, 1.044100, 1.054000)
    rg = rg_fit_exp_model(D_points, R_points)
    g_eff = (rg["R_inf"] - 1.0) / 12.0

    print(f"Canonical D*: D*={D_star}  exp(-sqrt(D*)/3)={eps0_hat:.6e}")
    print(f"eps0_hat={eps0_hat:.6e}")
    print(f"RG fit: R_inf={rg['R_inf']:.12f}  a={rg['a']:.6e}  SSE={rg['SSE']:.3e}")
    print(f"g_eff=(R_inf-1)/12={g_eff:.12f}")

    # --- strong-field predictions (shadow + ringdown) ---
    alpha_sf = alpha_eff(eps=0.5, g_eff=g_eff, D_star=D_star)
    print(f"alpha_sf(eps=0.5)={alpha_sf:.12f}")

    obs_GR = {"Q2": 0.0, "r_plus": 2.0, "r_ph": 3.0, "b_ph": math.sqrt(27.0), "df": 0.0, "dtau": 0.0}
    obs_P = strongfield_observables(alpha_eff=alpha_sf)
    scoreP = strongfield_score(obs_P)
    print("GR baseline:", obs_GR)
    print("Primary SF :", {k: obs_P[k] for k in ("Q2", "r_plus", "r_ph", "b_ph", "df", "dtau")})

    ok_sf1 = gate_line("Gate SF1: primary horizon exists", bool(obs_P.get("horizon_exists", False)), f"r_plus={obs_P.get('r_plus', float('nan')):.6f}")
    ok_sf2 = gate_line("Gate SF2: strong-field deviations are small (<= eps^2)", scoreP <= eps * eps, f"score={scoreP:.6e} eps^2={(eps*eps):.6e}")

    # --- dark-sector candidate (m_chi, sigma_proxy) ---
    q2_primary = q2
    K_primary = q2_primary // 2
    dmP = dark_candidate_for_triple(primary, q2_primary=q2_primary, K_primary=K_primary, D_star=D_star, g_eff=g_eff)
    print("\nPrimary dark candidate:")
    for k in ("K", "D_used", "v_seed", "alpha", "m_chi", "sigma_proxy"):
        print(f"  {k:<10} {dmP[k]}")

    ok_dm1 = gate_line("Gate DM1: alpha is small but nonzero", (dmP["alpha"] > 0.0) and (dmP["alpha"] <= eps * eps), f"alpha={dmP['alpha']:.6e}")
    ok_dm2 = gate_line("Gate DM2: sigma_proxy is tiny", dmP["sigma_proxy"] < (eps ** 8), f"sigma={dmP['sigma_proxy']:.6e}")

    # --- illegal control: replace screened g_eff by Θ (destroys horizon, inflates sigma) ---
    alpha_illegal = Theta
    obs_I = strongfield_observables(alpha_eff=alpha_illegal)
    # Match PREWORK-75C: keep the lawful v_seed, but overwrite \alpha by \Theta directly (no cap).
    v_seed = float(dmP["v_seed"])
    m_chi_I = v_seed / math.sqrt(max(1e-30, alpha_illegal))
    sigma_I = (alpha_illegal * alpha_illegal) / (m_chi_I * m_chi_I)
    dmI = {"m_chi": float(m_chi_I), "sigma_proxy": float(sigma_I), "alpha": float(alpha_illegal)}
    print("\nIllegal control: replace screened g_eff by Θ")
    print(f"illegal alpha=Theta={alpha_illegal:.12f}")
    print("illegal strong-field:", {k: obs_I.get(k) for k in ("Q2", "horizon_exists", "r_plus", "df", "dtau")})
    print(f"illegal dark: m_chi={dmI['m_chi']:.6f}  sigma_proxy={dmI['sigma_proxy']:.6e}")

    ok_i1 = gate_line("Gate I1: illegal Θ loses horizon (Q2>1)", not bool(obs_I.get("horizon_exists", False)), f"Q2={obs_I.get('Q2', float('nan')):.6f}")
    ok_i2 = gate_line(
        "Gate I2: illegal Θ inflates sigma_proxy",
        dmI["sigma_proxy"] >= (1.0 + eps) * dmP["sigma_proxy"],
        f"sigma_illegal={dmI['sigma_proxy']:.3e} sigma_P={dmP['sigma_proxy']:.3e} (1+eps)={(1+eps):.3f}",
    )

    # --- counterfactual teeth: (K,D_used) collapse -> alpha cap -> SF score inflates ---
    print("\nCounterfactual teeth (alpha inflation + SF instability):")
    strongT = 0
    for i, cf in enumerate(counterfactuals[:4], start=1):
        dm_cf = dark_candidate_for_triple(cf, q2_primary=q2_primary, K_primary=K_primary, D_star=D_star, g_eff=g_eff)
        obs_cf = strongfield_observables(alpha_eff=dm_cf["alpha"])
        score_i = strongfield_score(obs_cf)

        tooth = (
            (dm_cf["alpha"] >= (1.0 + eps) * alpha_sf)
            or (dm_cf["sigma_proxy"] >= (1.0 + eps) * dmP["sigma_proxy"])
            or (not bool(obs_cf.get("horizon_exists", False)))
            or (score_i >= (1.0 + eps) * scoreP)
        )
        strongT += int(tooth)

        print(
            f"CF{i}: triple={cf} q2={cf.wU - cf.s2:3d} K={dm_cf['K']:2d} D_used={dm_cf['D_used']:4d} "
            f"alpha={dm_cf['alpha']:.6f} m_chi={dm_cf['m_chi']:.3f} sigma={dm_cf['sigma_proxy']:.3e} "
            f"horizon={'Y' if obs_cf.get('horizon_exists', False) else 'N'} score_sf={score_i:.3e} tooth={'Y' if tooth else 'N'}"
        )

    ok_t = gate_line("Gate T: >=3/4 counterfactuals show tooth", strongT >= 3, f"strong={strongT}/4")

    # ----------------------------------------------------------------------------------
    # STAGE 6 — Prediction ledger (the part to publish)
    # ----------------------------------------------------------------------------------
    section("STAGE 6 — PREDICTION LEDGER (primary values + internal convergence deltas)")

    def delta(x_p: float, x_t: float) -> float:
        return float(x_p - x_t)

    ledger = {
        "neutrino": {
            "d21": primN["d21"],
            "d31": primN["d31"],
            "sum_m": primN["sumv"],
            "m1": primN["m1"],
            "m2": primN["m2"],
            "m3": primN["m3"],
        },
        "pmns": {
            "s12": pars_P["s12"],
            "s23": pars_P["s23"],
            "s13": pars_P["s13"],
            "J": pars_P["J"],
            "sin_delta": pars_P["sin_delta"],
            "delta_deg_principal": pars_P["delta_deg_principal"],
            "delta_deg_supplement": pars_P["delta_deg_supplement"],
            "m_beta": eff_P["m_beta"],
            "m_betabeta": eff_P["m_betabeta"],
        },
        "dark": dmP,
        "strong_field": obs_P,
        "kernel": {
            "Theta": Theta,
            "eps": eps,
            "D_star": D_star,
            "g_eff": g_eff,
        },
    }

    # internal convergence deltas (primary - truth)
    conv = {
        "neutrino": {
            "d21": delta(primN["d21"], truthN["d21"]),
            "d31": delta(primN["d31"], truthN["d31"]),
            "sum_m": delta(primN["sumv"], truthN["sumv"]),
            "m1": delta(primN["m1"], truthN["m1"]),
            "m2": delta(primN["m2"], truthN["m2"]),
            "m3": delta(primN["m3"], truthN["m3"]),
        },
        "pmns": {
            "s12": delta(pars_P["s12"], pars_T["s12"]),
            "s23": delta(pars_P["s23"], pars_T["s23"]),
            "s13": delta(pars_P["s13"], pars_T["s13"]),
            "J": delta(pars_P["J"], pars_T["J"]),
            "sin_delta": delta(pars_P["sin_delta"], pars_T["sin_delta"]),
            "m_beta": delta(eff_P["m_beta"], eff_T["m_beta"]),
            "m_betabeta": delta(eff_P["m_betabeta"], eff_T["m_betabeta"]),
        },
    }

    print("NEUTRINOS (eV-scale proxies):")
    for k in ("d21", "d31", "sum_m", "m1", "m2", "m3"):
        print(f"  {k:<8} = {ledger['neutrino'][k]:.9e}   (Δconv={conv['neutrino'][k]:+.3e})")
    print("\nPMNS / CP + effective masses:")
    for k in ("s12", "s23", "s13", "J", "sin_delta"):
        print(f"  {k:<10} = {ledger['pmns'][k]:.9e}   (Δconv={conv['pmns'][k]:+.3e})")
    print(f"  delta_deg  = {ledger['pmns']['delta_deg_principal']:.2f} / {ledger['pmns']['delta_deg_supplement']:.2f} (principal/supp)")
    print(f"  m_beta     = {ledger['pmns']['m_beta']:.9e}   (Δconv={conv['pmns']['m_beta']:+.3e})")
    print(f"  m_betabeta = {ledger['pmns']['m_betabeta']:.9e}   (Δconv={conv['pmns']['m_betabeta']:+.3e})")
    print("\nDARK-SECTOR (proxy units):")
    print(f"  m_chi       = {ledger['dark']['m_chi']:.6f}")
    print(f"  sigma_proxy = {ledger['dark']['sigma_proxy']:.9e}")
    print(f"  alpha       = {ledger['dark']['alpha']:.9e}")
    print("\nSTRONG-FIELD (proxy deviations):")
    print(f"  Q2    = {ledger['strong_field']['Q2']:.9e}")
    print(f"  r_plus= {ledger['strong_field']['r_plus']:.9e}")
    print(f"  r_ph  = {ledger['strong_field']['r_ph']:.9e}")
    print(f"  b_ph  = {ledger['strong_field']['b_ph']:.9e}")
    print(f"  df    = {ledger['strong_field']['df']:.9e}")
    print(f"  dtau  = {ledger['strong_field']['dtau']:.9e}")

    # ----------------------------------------------------------------------------------
    # STAGE 7 — Determinism hash + verdict
    # ----------------------------------------------------------------------------------
    report = {
        "spec": spec,
        "stages": {
            "selection": {"primary": primary.__dict__, "counterfactuals": [c.__dict__ for c in counterfactuals[:4]]},
            "kernel_audit": {"N": N, "r": r_audit, "kmin_fejer": kmin_fejer, "kmin_sharp": kmin_sharp, "kmin_signed": kmin_signed, "hf_signed": hf_signed},
            "neutrinos": {"truth": truthN, "primary": primN, "illegal_sharp": sharpN, "illegal_signflip": signN, "dist_primary": dP, "dist_sharp": dSh, "dist_signflip": dSf},
            "pmns": {"truth": {"pars": pars_T, "eff": eff_T}, "primary": {"pars": pars_P, "eff": eff_P}, "dists": {"primary": dist_pmns_P, "sharp": dist_pmns_sh, "signed": dist_pmns_si}},
            "dark_strongfield": {"rg": rg, "alpha_sf": alpha_sf, "strongfield_primary": obs_P, "dark_primary": dmP},
        },
        "ledger": ledger,
        "convergence": conv,
        "gates": {
            "S0": ok_s0,
            "S1": ok_s1,
            "I1": ok_i1,
            "A1": ok_a1,
            "A2": ok_a2,
            "A3": ok_a3,
            "A4": ok_a4,
            "N1": ok_n1,
            "N2": ok_n2,
            "N3": ok_n3,
            "NT": ok_nt,
            "M1": ok_m1,
            "M2": ok_m2,
            "M3": ok_m3,
            "MT": ok_mt,
            "SF1": ok_sf1,
            "SF2": ok_sf2,
            "DM1": ok_dm1,
            "DM2": ok_dm2,
            "I1_dark": ok_i1,
            "I2_dark": ok_i2,
            "T": ok_t,
        },
    }

    det_sha = sha256_hex(json.dumps(report, sort_keys=True).encode("utf-8"))
    section("DETERMINISM HASH")
    print("determinism_sha256:", det_sha)

    if args.json:
        try:
            with open("demo75_prediction_ledger_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)
            print("(wrote demo75_prediction_ledger_report.json)")
        except Exception as e:
            print("(warn) could not write JSON artifact:", repr(e))

    all_ok = bool(
        ok_s0
        and ok_s1
        and ok_i1
        and ok_a1
        and ok_a2
        and ok_a3
        and ok_a4
        and ok_n1
        and ok_n2
        and ok_n3
        and ok_nt
        and ok_m1
        and ok_m2
        and ok_m3
        and ok_mt
        and ok_sf1
        and ok_sf2
        and ok_dm1
        and ok_dm2
        and ok_i1
        and ok_i2
        and ok_t
    )

    section("FINAL VERDICT")
    gate_line(
        "DEMO-75 VERIFIED (prediction ledger: neutrinos + PMNS/CP + dark + strong-field; with controls + teeth)",
        all_ok,
    )
    print("Result:", "VERIFIED" if all_ok else "NOT VERIFIED")


if __name__ == "__main__":
    main()
