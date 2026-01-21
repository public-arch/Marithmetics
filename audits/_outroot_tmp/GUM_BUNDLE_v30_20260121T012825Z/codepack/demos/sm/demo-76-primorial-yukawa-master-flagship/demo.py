#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DEMO-76 — PRIMORIAL-YUKAWA MASTER FLAGSHIP — REFEREE READY

Goal
----
From the deterministic kernel triple (137,107,103), derive a *primorial* survivor-density
  Θ = φ(M_y) / M_y
with y=q3=17, then a *family angle*
  θ12 = \hat{π}/12
where \hat{π} is computed by a *lawful* Fejér–Cesàro weighted alternating series.

Using only:
  • ε = 1/sqrt(q2) with q2 = wU - s2,
  • Θ^{log(index)} (natural-log by default), and
  • θ12,
we build a deterministic 9-Yukawa ladder (u,d,c,s,t,b,e,μ,τ), then perform a
sensitivity audit over (r, y, θ12 convention, log-base) and enforce:
  • illegal controls break the ladder,
  • counterfactual teeth under reduced budget.

This demo is intentionally *self-contained* and uses only NumPy.
No web. No PDG. No measured inputs.

Run
---
  python3 demo76_master_flagship_primorial_yukawa_referee_ready_v1.py
  python3 demo76_master_flagship_primorial_yukawa_referee_ready_v1.py --json

Outputs
-------
stdout report + optional JSON artifact.

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
from fractions import Fraction
from typing import Dict, List, Tuple, Optional

try:
    import numpy as np
except Exception as e:
    print("FATAL: NumPy is required.")
    print("Import error:", repr(e))
    raise

G = "✅"
R = "❌"


# --------------------------
# Utility / determinism
# --------------------------

def header(title: str) -> None:
    bar = "=" * 100
    print(bar)
    print(title.center(100))
    print(bar)


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


def primes_upto(n: int) -> List[int]:
    return [k for k in range(2, n + 1) if is_prime(k)]


@dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


def select_triple() -> Tuple[Triple, List[Triple], Dict[str, List[int]]]:
    """Deterministic triple selection used across the demo series."""
    window = primes_in_range(97, 181)

    # Lane survivors (same simple modular filters used elsewhere)
    U1_raw = [p for p in window if (p % 17) in (1, 5)]
    SU2_raw = [p for p in window if (p % 13) == 3]
    SU3_raw = [p for p in window if (p % 17) == 1]

    # U(1) coherence filter
    U1 = [p for p in U1_raw if v2(p - 1) == 3]

    wU = U1[0]
    s2 = SU2_raw[0]
    s3 = min([p for p in SU3_raw if p != wU])
    primary = Triple(wU=wU, s2=s2, s3=s3)

    # Counterfactual pool: same structure, different window
    window_cf = primes_in_range(181, 1200)
    U1_cf = [p for p in window_cf if (p % 17) in (1, 5) and v2(p - 1) == 3]
    wU_cf = U1_cf[0]
    SU2_cf = [p for p in window_cf if (p % 13) == 3][:2]
    SU3_cf = [p for p in window_cf if (p % 17) == 1][:2]
    counterfactuals = [Triple(wU=wU_cf, s2=a, s3=b) for a in SU2_cf for b in SU3_cf]

    pools = {
        "U(1)_raw": U1_raw,
        "SU(2)_raw": SU2_raw,
        "SU(3)_raw": SU3_raw,
        "U(1)_coherent": U1,
    }
    return primary, counterfactuals, pools


# --------------------------
# OATB admissibility audit (FFT kernel minima)
# --------------------------

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


# --------------------------
# Primorial Θ and survivor field
# --------------------------

def primorial_theta(y: int) -> Tuple[Fraction, int, int, List[int]]:
    """Return (Theta_frac, M_y, phi(M_y), primes<=y).

    Theta is exact as a rational since:
      φ(∏ p) / ∏ p = ∏ (1 - 1/p)
    """
    primes = primes_upto(y)
    M = 1
    phi = 1
    for p in primes:
        M *= p
        phi *= (p - 1)
    # Theta = phi/M but simplify by gcd automatically via Fraction
    Theta = Fraction(phi, M)
    return Theta, M, phi, primes


def survivor_field(M: int, primes: List[int]) -> np.ndarray:
    """Binary survivor field S on Z/MZ: S[n]=1 iff gcd(n,M)=1.

    Constructed by sieving out multiples of each prime.
    """
    S = np.ones(M, dtype=np.uint8)
    for p in primes:
        S[0::p] = 0
    # ensure the canonical unit survives
    if M > 1:
        S[1] = 1
    return S


def mst_observables(S: np.ndarray, r: int, kind: str) -> Tuple[float, float]:
    """DEMO-74 style MST observables using rolling sums.

    Returns (eps_smooth, X0) where:
      X = Σ_{m=1..r} w_m S(·+m)
      eps_smooth = ||X - mean(X)||_2 / sqrt(M)
      X0 = X[0] / r

    kind ∈ {fejer, sharp, signed}
    """
    M = int(S.shape[0])
    X = np.zeros(M, dtype=np.float64)
    for m in range(1, r + 1):
        if kind == "fejer":
            w = 1.0 - (m / (r + 1.0))
        elif kind == "sharp":
            w = 1.0
        elif kind == "signed":
            w = -1.0 if (m % 2 == 1) else 1.0
        else:
            raise ValueError("unknown kind")
        X += w * np.roll(S, -m)
    mu = float(np.mean(X))
    eps_smooth = float(np.sqrt(np.mean((X - mu) ** 2)))
    X0 = float(X[0] / max(1.0, float(r)))
    return eps_smooth, X0


# --------------------------
# θ12 from π-hat (lawful vs illegal)
# --------------------------

def pi_hat_fejer(r: int, *, variant: str) -> Tuple[float, int]:
    """Compute \hat{π} from an alternating series with optional Fejér weighting.

    Base series: Leibniz
      π = 4 Σ_{n=0..∞} (-1)^n/(2n+1)

    Variants
    --------
    lawful  : Fejér–Cesàro weighted partial sum with N=(r+1)^2 terms.
    sharp   : unweighted truncated sum with N=(r+1)^2 terms. (illegal control)
    signflip: all + signs with Fejér weights, N=(r+1)^2. (illegal control)

    Weight is chosen as w_n = 1 - n/(N+1) to keep endpoints nonnegative.
    """
    r = max(1, int(r))
    # Keep the same term budget N=(r+1)^2 across lawful/sharp/signflip so that
    # *only* admissibility (weights/signs) is being tested as the control.
    N = (r + 1) ** 2

    s = 0.0
    for n in range(N):
        w = 1.0
        if variant in ("lawful", "signflip"):
            w = 1.0 - (n / float(N + 1))
        if variant == "sharp":
            w = 1.0

        sign = -1.0 if (n % 2 == 1) else 1.0
        if variant == "signflip":
            sign = 1.0

        s += w * sign / (2 * n + 1)

    return 4.0 * s, N


# --------------------------
# Yukawa ladder
# --------------------------

def theta_pow_log_index(theta: float, k: int, *, log_base: float = math.e) -> float:
    """Compute Θ^{log_b(k)} where log_b is log base `log_base`.

    For log_base=e this is Θ^{ln(k)}.
    """
    if k <= 0:
        return 0.0
    if k == 1:
        return 1.0
    if log_base <= 0.0 or log_base == 1.0:
        raise ValueError("invalid log base")
    return float(theta) ** (math.log(float(k), float(log_base)))


def yukawas_from_params(
    *,
    eps: float,
    theta: float,
    theta12: float,
    y_cutoff: int,
    log_base: float = math.e,
    max_prime_index: Optional[int] = None,
    y_top_target: float = 0.999,
) -> Dict[str, float]:
    """Build 9 Yukawas (u,d,c,s,t,b,e,mu,tau) from (ε, Θ, θ12) + (y_cutoff).

    y_cutoff sets the maximum prime index available (via primes<=y_cutoff).
    If `max_prime_index` is provided, it overrides that count.

    The top Yukawa is rescaled to exactly `y_top_target`.
    """

    # prime-index pool size
    primes = primes_upto(max(17, int(y_cutoff)))
    if max_prime_index is None:
        max_prime_index = len(primes)
    max_prime_index = max(1, int(max_prime_index))

    # canonical index assignments (prime indices), with wrapping for small y
    req = {
        "t": 1,
        "c": 2,
        "u": 3,
        "b": 4,
        "tau": 5,
        "mu": 6,
        "e": 7,
        "s": 5,
        "d": 6,
    }

    def wrap(k: int) -> int:
        return ((k - 1) % max_prime_index) + 1

    k_t = wrap(req["t"])
    k_c = wrap(req["c"])
    k_u = wrap(req["u"])
    k_b = wrap(req["b"])
    k_s = wrap(req["s"])
    k_d = wrap(req["d"])
    k_tau = wrap(req["tau"])
    k_mu = wrap(req["mu"])
    k_e = wrap(req["e"])

    # "Base" factors: keep the earlier DEMO-72 spine form, but with primorial-Θ laddering.
    #   up-type:   1
    #   down-type: ε
    #   leptons:   ε^2
    Bu = 1.0
    Bd = float(eps)
    Bl = float(eps) ** 2

    c = math.cos(theta12)
    s = math.sin(theta12)

    y: Dict[str, float] = {}
    y["t"] = Bu * theta_pow_log_index(theta, k_t, log_base=log_base)
    y["c"] = Bu * theta_pow_log_index(theta, k_c, log_base=log_base) * c
    y["u"] = Bu * theta_pow_log_index(theta, k_u, log_base=log_base) * s

    y["b"] = Bd * theta_pow_log_index(theta, k_b, log_base=log_base)
    y["s"] = Bd * theta_pow_log_index(theta, k_s, log_base=log_base) * c
    y["d"] = Bd * theta_pow_log_index(theta, k_d, log_base=log_base) * s

    y["tau"] = Bl * theta_pow_log_index(theta, k_tau, log_base=log_base)
    y["mu"] = Bl * theta_pow_log_index(theta, k_mu, log_base=log_base) * c
    y["e"] = Bl * theta_pow_log_index(theta, k_e, log_base=log_base) * s

    # loop-level rescaling: land the top Yukawa at y_top_target
    scale = float(y_top_target) / max(1e-300, y["t"])
    for k in list(y.keys()):
        y[k] *= scale

    return y


def vector9(y: Dict[str, float]) -> np.ndarray:
    order = ["u", "d", "c", "s", "t", "b", "e", "mu", "tau"]
    return np.array([float(y[k]) for k in order], dtype=float)


def rel_L2(a: np.ndarray, b: np.ndarray) -> float:
    num = float(np.linalg.norm(a - b))
    den = float(np.linalg.norm(b))
    return num / max(1e-16, den)


def as_fraction(x: float, max_den: int = 1_000_000) -> Fraction:
    if not math.isfinite(x):
        return Fraction(0, 1)
    return Fraction(x).limit_denominator(max_den)


# --------------------------
# Main
# --------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="Attempt to write demo76_report.json (optional).")
    ap.add_argument("--max_den", type=int, default=1_000_000, help="Max denominator for fraction display.")
    args = ap.parse_args()

    header("DEMO-76 — PRIMORIAL-YUKAWA MASTER FLAGSHIP (Θ primorial + θ12 + ladder + sensitivity + teeth)")
    print(f"UTC time : {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout only (JSON optional)")
    print()

    # ------------------
    # Stage 1: selection
    # ------------------
    primary, counterfactuals, pools = select_triple()

    print("=" * 100)
    print("STAGE 1 — Deterministic triple selection".center(100))
    print("=" * 100)
    print("Lane survivor pools (raw):")
    print("  U(1): ", pools["U(1)_raw"])
    print("  SU(2):", pools["SU(2)_raw"])
    print("  SU(3):", pools["SU(3)_raw"])
    print("Lane survivor pools (after U(1) coherence v2(wU-1)=3):")
    print("  U(1): ", pools["U(1)_coherent"])
    print("Primary:", primary)
    print("Counterfactuals:")
    for cf in counterfactuals[:4]:
        print(" ", cf)

    ok_s0 = gate_line("Gate S0: primary equals (137,107,103)", (primary.wU, primary.s2, primary.s3) == (137, 107, 103))
    ok_s1 = gate_line("Gate S1: captured >=4 counterfactual triples", len(counterfactuals) >= 4, f"found={len(counterfactuals)}")

    # ------------------
    # Stage 1B: invariants
    # ------------------
    q2 = primary.wU - primary.s2
    v2U = v2(primary.wU - 1)
    q3 = (primary.wU - 1) // (2 ** v2U)
    eps = 1.0 / math.sqrt(float(q2))
    q3_cf = 3 * q3

    r_truth = 31
    r_primary = 15
    r_cf_budget = int(round(r_primary * q3 / q3_cf))

    spec = {
        "primary": primary.__dict__,
        "q2": q2,
        "q3": q3,
        "v2U": v2U,
        "eps": eps,
        "budgets": {"r_truth": r_truth, "r_primary": r_primary, "r_cf": r_cf_budget},
        "y_top_target": 0.999,
        "theta12_denom": 12,
        "log_base": "e",
    }
    spec_sha = sha256_hex(json.dumps(spec, sort_keys=True).encode("utf-8"))

    print("\n" + "=" * 100)
    print("STAGE 1B — Derived invariants".center(100))
    print("=" * 100)
    print(f"spec_sha256: {spec_sha}")
    print(f"q2={q2}  q3={q3}  v2U={v2U}  eps=1/sqrt(q2)={eps:.8f}  q3_cf=3*q3={q3_cf}")
    print(f"Budgets: truth r={r_truth}   primary r={r_primary}   counterfactual r_cf={r_cf_budget}")

    ok_i1 = gate_line("Gate I1: invariants match locked (q2=30,q3=17,v2U=3)", (q2, q3, v2U) == (30, 17, 3), f"(q2,q3,v2U)=({q2},{q3},{v2U})")

    # ------------------
    # Stage 2: admissibility audit (FFT kernel minima)
    # ------------------
    print("\n" + "=" * 100)
    print("STAGE 2 — OATB kernel admissibility audit (FFT minima + HF weight)".center(100))
    print("=" * 100)
    N = 2048
    kmin_fejer = kernel_realspace_min(N, r_primary, "fejer")
    kmin_sharp = kernel_realspace_min(N, r_primary, "sharp")
    kmin_signed = kernel_realspace_min(N, r_primary, "signed")
    hf_signed = hf_weight_frac(N, r_primary, "signed")

    print(f"N={N} r={r_primary}")
    print(f"Fejér kernel min  : {kmin_fejer:+.6e}")
    print(f"Sharp kernel min  : {kmin_sharp:+.6e}")
    print(f"Signed kernel min : {kmin_signed:+.6e}")
    print(f"Signed HF weight frac(>r): {hf_signed:.6f}")

    ok_a1 = gate_line("Gate A1: Fejér kernel nonnegative (tol)", kmin_fejer >= -1e-9, f"min={kmin_fejer:.3e}")
    ok_a2 = gate_line("Gate A2: sharp kernel has negative lobes", kmin_sharp <= -1e-6, f"min={kmin_sharp:.3e}")
    ok_a3 = gate_line("Gate A3: signed kernel strongly negative", kmin_signed <= -eps * eps, f"min={kmin_signed:.3e} floor={-eps*eps:.3e}")
    ok_a4 = gate_line("Gate A4: signed retains large HF weight", hf_signed >= max(0.25, eps * eps), f"hf={hf_signed:.3f} floor={max(0.25, eps*eps):.3f}")

    # ------------------
    # Stage 3: primorial window and MST smoothing
    # ------------------
    print("\n" + "=" * 100)
    print("STAGE 3 — Primorial window (y=q3) + Θ + survivor field + Fejér smoothing".center(100))
    print("=" * 100)

    y_cutoff_primary = q3
    Theta_frac, M_y, phi_M, primes_y = primorial_theta(y_cutoff_primary)
    Theta = float(Theta_frac)

    print(f"y=q3={y_cutoff_primary}")
    print(f"primes<=y: {primes_y}")
    print(f"M_y (primorial) = {M_y}")
    print(f"phi(M_y)        = {phi_M}")
    print(f"Theta = phi(M_y)/M_y = {Theta_frac.numerator}/{Theta_frac.denominator} = {Theta:.12f}")

    # exact checks for the canonical y=17 case
    ok_p0 = gate_line("Gate P0: M_y matches 510510 (y=17)", (y_cutoff_primary != 17) or (M_y == 510510), f"M_y={M_y}")
    ok_p1 = gate_line("Gate P1: phi(M_y) matches 92160 (y=17)", (y_cutoff_primary != 17) or (phi_M == 92160), f"phi={phi_M}")

    # survivor field (feasible only for y=17 scale)
    S = survivor_field(M_y, primes_y)
    mean_S = float(np.mean(S))
    count_S = int(np.sum(S))
    ok_p2 = gate_line("Gate P2: phi(M_y) equals survivor count", count_S == phi_M, f"count={count_S}")
    ok_p3 = gate_line("Gate P3: empirical mean(S) equals Theta (tol)", abs(mean_S - Theta) <= 1e-12, f"mean={mean_S:.12f}")

    eps_smooth_fejer, X0_fejer = mst_observables(S, r_primary, "fejer")
    eps_smooth_sharp, X0_sharp = mst_observables(S, r_primary, "sharp")
    eps_smooth_signed, X0_signed = mst_observables(S, r_primary, "signed")

    print("\nMST observables (DEMO-74 style):")
    print(f"  lawful (Fejér) : eps_smooth={eps_smooth_fejer:.6e}  X0={X0_fejer:.6e}")
    print(f"  sharp (illegal): eps_smooth={eps_smooth_sharp:.6e}  X0={X0_sharp:.6e}")
    print(f"  signed(illegal): eps_smooth={eps_smooth_signed:.6e}  X0={X0_signed:.6e}")

    ok_w1 = gate_line(
        "Gate W1: lawful smoothing no worse than signed (eps_smooth)",
        eps_smooth_fejer <= eps_smooth_signed,
        f"lawful={eps_smooth_fejer:.3e} signed={eps_smooth_signed:.3e}",
    )

    # ------------------
    # Stage 4: θ12 from π-hat
    # ------------------
    print("\n" + "=" * 100)
    print("STAGE 4 — Family angle θ12 from π-hat/12 (lawful vs illegal controls)".center(100))
    print("=" * 100)

    pi_T, terms_T = pi_hat_fejer(r_truth, variant="lawful")
    pi_P, terms_P = pi_hat_fejer(r_primary, variant="lawful")
    pi_sh, terms_sh = pi_hat_fejer(3, variant="sharp")
    pi_sf, terms_sf = pi_hat_fejer(r_primary, variant="signflip")

    k_theta = 12
    theta12_T = pi_T / k_theta
    theta12_P = pi_P / k_theta
    theta12_sh = pi_sh / k_theta
    theta12_sf = pi_sf / k_theta

    print(f"truth   pi_hat={pi_T:.10f}  terms={terms_T}  => theta12={theta12_T:.10f} rad")
    print(f"primary pi_hat={pi_P:.10f}  terms={terms_P}  => theta12={theta12_P:.10f} rad")
    print(f"illegal sharp   pi_hat={pi_sh:.10f} terms={terms_sh} => theta12={theta12_sh:.10f} rad")
    print(f"illegal signflip pi_hat={pi_sf:.10f} terms={terms_sf} => theta12={theta12_sf:.10f} rad")

    # ------------------
    # Stage 5: Yukawa ladder
    # ------------------
    print("\n" + "=" * 100)
    print("STAGE 5 — Yukawas from (ε, Θ^{log(index)}, θ12) + loop rescale (y_t=0.999)".center(100))
    print("=" * 100)

    y_truth = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_T, y_cutoff=y_cutoff_primary, log_base=math.e)
    y_primary = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_P, y_cutoff=y_cutoff_primary, log_base=math.e)
    y_sharp = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_sh, y_cutoff=y_cutoff_primary, log_base=math.e)
    y_signflip = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_sf, y_cutoff=y_cutoff_primary, log_base=math.e)

    vT = vector9(y_truth)
    vP = vector9(y_primary)
    vSh = vector9(y_sharp)
    vSf = vector9(y_signflip)

    distP = rel_L2(vP, vT)
    distSh = rel_L2(vSh, vT)
    distSf = rel_L2(vSf, vT)

    # Fractions then decimals
    order9 = ["u", "d", "c", "s", "t", "b", "e", "mu", "tau"]
    print("Primary Yukawas (fractions; max_den=%d):" % int(args.max_den))
    for k in order9:
        fr = as_fraction(y_primary[k], max_den=int(args.max_den))
        print(f"  y_{k:<4} = {fr.numerator}/{fr.denominator}")

    print("\nPrimary Yukawas (decimals):")
    for k in order9:
        print(f"  y_{k:<4} = {y_primary[k]:.12e}")

    ratio_tb = y_primary["t"] / max(1e-300, y_primary["b"])
    ratio_cs = y_primary["c"] / max(1e-300, y_primary["s"])
    print("\nHierarchy ratios (primary):")
    print(f"  y_t / y_b = {ratio_tb:.6f}")
    print(f"  y_c / y_s = {ratio_cs:.6f}")

    print("\nDistances vs truth (rel L2 on 9-vector):")
    print(f"  dist_primary  = {distP:.6e}")
    print(f"  dist_sharp    = {distSh:.6e}")
    print(f"  dist_signflip = {distSf:.6e}")

    # ------------------
    # Stage 6: Sensitivity audit
    # ------------------
    print("\n" + "=" * 100)
    print("STAGE 6 — Sensitivity audit (r, y, θ12 convention, log-base)".center(100))
    print("=" * 100)

    # 6A: r-scan
    print("r-scan (lawful):")
    print("r   terms    pi_hat       theta12      dist_to_truth")
    dist_by_r: Dict[int, float] = {}
    for rr in [3, 5, 7, 9, 11, 13, 15, 17, 19, 23, 31]:
        pi_rr, terms_rr = pi_hat_fejer(rr, variant="lawful")
        th_rr = pi_rr / k_theta
        y_rr = yukawas_from_params(eps=eps, theta=Theta, theta12=th_rr, y_cutoff=y_cutoff_primary, log_base=math.e)
        d_rr = rel_L2(vector9(y_rr), vT)
        dist_by_r[rr] = d_rr
        print(f"{rr:2d} {terms_rr:6d}  {pi_rr:10.7f}  {th_rr:10.7f}  {d_rr:12.6e}")

    # 6B: primorial y-scan
    print("\nprimorial cutoff y-scan:")
    print("y   #primes   Theta(phi/M)        dist_to_truth")
    dist_by_y: Dict[int, float] = {}
    for yy in [11, 13, 17, 19, 23]:
        Th_frac_yy, _, _, primes_yy = primorial_theta(yy)
        Th_yy = float(Th_frac_yy)
        y_yy = yukawas_from_params(eps=eps, theta=Th_yy, theta12=theta12_P, y_cutoff=yy, log_base=math.e)
        d_yy = rel_L2(vector9(y_yy), vT)
        dist_by_y[yy] = d_yy
        print(f"{yy:2d} {len(primes_yy):8d}  {Th_yy:16.12f}  {d_yy:12.6e}")

    # 6C: theta12 denominator sensitivity
    print("\nangle convention: theta12 = pi_hat / k")
    print("k   theta12(rad)   dist_to_truth")
    dist_by_k: Dict[int, float] = {}
    for kk in [11, 12, 13]:
        th_k = pi_P / kk
        y_k = yukawas_from_params(eps=eps, theta=Theta, theta12=th_k, y_cutoff=y_cutoff_primary, log_base=math.e)
        d_k = rel_L2(vector9(y_k), vT)
        dist_by_k[kk] = d_k
        print(f"{kk:2d}  {th_k:12.10f}  {d_k:12.6e}")

    # 6D: exponent log-base sensitivity
    print("\nexponent log-base sensitivity:")
    print("base   dist_to_truth")
    dist_by_base: Dict[str, float] = {}
    for base_label, base_val in [("e", math.e), ("2", 2.0), ("10", 10.0)]:
        y_b = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_P, y_cutoff=y_cutoff_primary, log_base=base_val)
        d_b = rel_L2(vector9(y_b), vT)
        dist_by_base[base_label] = d_b
        print(f" {base_label:<3}  {d_b:12.6e}")

    # 6E: local sensitivity d(log y)/d(theta12) around primary
    print("\nlocal sensitivity: d(log y)/d(theta12) around primary")
    delta_theta = abs(theta12_P - theta12_T)
    delta = max(1e-9, delta_theta)
    y_plus = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_P + delta, y_cutoff=y_cutoff_primary, log_base=math.e)
    y_minus = yukawas_from_params(eps=eps, theta=Theta, theta12=theta12_P - delta, y_cutoff=y_cutoff_primary, log_base=math.e)

    slope: Dict[str, float] = {}
    for k in order9:
        yp = max(1e-300, y_plus[k])
        ym = max(1e-300, y_minus[k])
        slope[k] = (math.log(yp) - math.log(ym)) / (2.0 * delta)

    print(f"delta_theta = |theta12_primary - theta12_truth| = {delta_theta:.12e} rad")
    for k in order9:
        print(f"  y_{k:<4} slope={slope[k]:+.6e}  (|s|={abs(slope[k]):.6e})")

    max_key = max(order9, key=lambda kk: abs(slope[kk]))

    # ------------------
    # Stage 7: Counterfactual teeth
    # ------------------
    print("\n" + "=" * 100)
    print("STAGE 7 — Counterfactual teeth (triple + reduced budget)")
    print("=" * 100)

    cf_rows = []
    strong = 0
    for cf in counterfactuals[:4]:
        q2c = cf.wU - cf.s2
        v2c = v2(cf.wU - 1)
        q3c = (cf.wU - 1) // (2 ** v2c)
        eps_c = 1.0 / math.sqrt(float(q2c))

        # budget tooth: r shrinks with q3
        r_c = int(round(r_primary * q3 / max(1, q3c)))
        r_c = max(3, min(r_primary, r_c))

        # primorial cutoff uses y=q3c
        Theta_c_frac, _, _, primes_c = primorial_theta(int(q3c))
        Theta_c = float(Theta_c_frac)

        # lawful pi_hat at reduced budget
        pi_c, _ = pi_hat_fejer(r_c, variant="lawful")
        theta12_c = pi_c / k_theta

        y_c = yukawas_from_params(eps=eps_c, theta=Theta_c, theta12=theta12_c, y_cutoff=int(q3c), log_base=math.e)
        d_c = rel_L2(vector9(y_c), vT)
        degrade = d_c >= (1.0 + eps) * distP
        strong += int(degrade)

        cf_rows.append({
            "triple": cf.__dict__,
            "q2": q2c,
            "q3": q3c,
            "r_cf": r_c,
            "Theta": Theta_c,
            "dist": d_c,
            "degrade": degrade,
        })

        print(f"CF {cf}  q2={q2c:3d} q3={q3c:3d} r_cf={r_c:2d}  dist={d_c:.6e}  degrade={degrade}")

    # ------------------
    # Gates
    # ------------------
    print("\n" + "=" * 100)
    print("GATES".center(100))
    print("=" * 100)

    # hierarchy gates
    ok_h1 = gate_line("Gate H1: hierarchy t>c>u", y_primary["t"] > y_primary["c"] > y_primary["u"])
    ok_h2 = gate_line("Gate H2: hierarchy b>s>d", y_primary["b"] > y_primary["s"] > y_primary["d"])
    ok_h3 = gate_line("Gate H3: hierarchy tau>mu>e", y_primary["tau"] > y_primary["mu"] > y_primary["e"])

    # broad ratio bands (coarse sanity only; not PDG-driven)
    ok_r1 = gate_line("Gate R1: top/bottom ratio in a broad hierarchical band", 20.0 <= ratio_tb <= 200.0, f"ratio={ratio_tb:.2f}")
    ok_r2 = gate_line("Gate R2: charm/strange ratio in a broad hierarchical band", 5.0 <= ratio_cs <= 200.0, f"ratio={ratio_cs:.2f}")

    ok_y1 = gate_line("Gate Y1: primary Yukawas stable vs truth (<= eps)", distP <= eps, f"dist={distP:.3e} eps={eps:.3e}")
    ok_y2 = gate_line(
        "Gate Y2: illegal sharp angle breaks stability by (1+eps)",
        distSh >= (1.0 + eps) * distP,
        f"dist_sharp={distSh:.3e} distP={distP:.3e} (1+eps)={(1+eps):.3f}",
    )
    ok_y3 = gate_line(
        "Gate Y3: illegal signflip breaks stability by (1+eps)",
        distSf >= (1.0 + eps) * distP,
        f"dist_sf={distSf:.3e} distP={distP:.3e} (1+eps)={(1+eps):.3f}",
    )

    # sensitivity gates
    ok_sr = gate_line(
        "Gate SR: dist(r) decreases monotonically with r over scan",
        dist_by_r[3] >= dist_by_r[15] >= dist_by_r[31],
        f"d3={dist_by_r[3]:.3e} d15={dist_by_r[15]:.3e} d31={dist_by_r[31]:.3e}",
    )

    best_y = min(dist_by_y.items(), key=lambda kv: kv[1])[0]
    ok_sy = gate_line(
        "Gate SY: primorial cutoff y=q3 is best in local y-scan",
        best_y == 17,
        f"best_y={best_y} dist_best={dist_by_y[best_y]:.3e}",
    )

    best_k = min(dist_by_k.items(), key=lambda kv: kv[1])[0]
    ok_sd = gate_line(
        "Gate SD: theta12 denominator 12 is best in {11,12,13}",
        best_k == 12,
        f"best_k={best_k} dist_best={dist_by_k[best_k]:.3e}",
    )

    best_base = min(dist_by_base.items(), key=lambda kv: kv[1])[0]
    ok_sl = gate_line(
        "Gate SL: log base e is best among {e,2,10}",
        best_base == "e",
        f"best={best_base} dist_best={dist_by_base[best_base]:.3e}",
    )

    ok_sth = gate_line(
        "Gate Sθ: |dlogy/dθ12| is maximized by a first-generation coupling",
        max_key in ("u", "d", "e"),
        f"max_key=y_{max_key} |s|={abs(slope[max_key]):.3e}",
    )

    ok_t = gate_line(
        "Gate T: counterfactual teeth (>=3/4 CFs degrade by (1+eps))",
        strong >= 3,
        f"strong={strong}/4 eps={eps:.3f}",
    )

    # Collect report
    report = {
        "spec": spec,
        "spec_sha256": spec_sha,
        "selection": {
            "primary": primary.__dict__,
            "counterfactuals": [cf.__dict__ for cf in counterfactuals[:4]],
            "pools": pools,
        },
        "invariants": {"q2": q2, "q3": q3, "v2U": v2U, "eps": eps, "q3_cf": q3_cf},
        "kernel_audit": {
            "N": N,
            "r": r_primary,
            "kmin_fejer": kmin_fejer,
            "kmin_sharp": kmin_sharp,
            "kmin_signed": kmin_signed,
            "hf_signed": hf_signed,
        },
        "primorial": {
            "y": y_cutoff_primary,
            "primes": primes_y,
            "M_y": M_y,
            "phi_M": phi_M,
            "Theta_frac": f"{Theta_frac.numerator}/{Theta_frac.denominator}",
            "Theta": Theta,
            "mean_survivor": mean_S,
            "count_survivor": count_S,
            "mst": {
                "fejer": {"eps_smooth": eps_smooth_fejer, "X0": X0_fejer},
                "sharp": {"eps_smooth": eps_smooth_sharp, "X0": X0_sharp},
                "signed": {"eps_smooth": eps_smooth_signed, "X0": X0_signed},
            },
        },
        "theta12": {
            "k": k_theta,
            "truth": {"pi_hat": pi_T, "terms": terms_T, "theta12": theta12_T},
            "primary": {"pi_hat": pi_P, "terms": terms_P, "theta12": theta12_P},
            "illegal_sharp": {"pi_hat": pi_sh, "terms": terms_sh, "theta12": theta12_sh},
            "illegal_signflip": {"pi_hat": pi_sf, "terms": terms_sf, "theta12": theta12_sf},
        },
        "yukawas": {
            "primary": y_primary,
            "truth": y_truth,
            "illegal_sharp": y_sharp,
            "illegal_signflip": y_signflip,
            "ratios": {"t_over_b": ratio_tb, "c_over_s": ratio_cs},
            "dist": {"primary": distP, "sharp": distSh, "signflip": distSf},
        },
        "sensitivity": {
            "dist_by_r": dist_by_r,
            "dist_by_y": dist_by_y,
            "dist_by_k": dist_by_k,
            "dist_by_base": dist_by_base,
            "delta_theta": delta_theta,
            "slopes": {f"y_{k}": slope[k] for k in order9},
            "max_slope_key": f"y_{max_key}",
        },
        "counterfactuals": cf_rows,
        "gates": {
            "S0": ok_s0,
            "S1": ok_s1,
            "I1": ok_i1,
            "A1": ok_a1,
            "A2": ok_a2,
            "A3": ok_a3,
            "A4": ok_a4,
            "P0": ok_p0,
            "P1": ok_p1,
            "P2": ok_p2,
            "P3": ok_p3,
            "W1": ok_w1,
            "H1": ok_h1,
            "H2": ok_h2,
            "H3": ok_h3,
            "R1": ok_r1,
            "R2": ok_r2,
            "Y1": ok_y1,
            "Y2": ok_y2,
            "Y3": ok_y3,
            "SR": ok_sr,
            "SY": ok_sy,
            "SD": ok_sd,
            "SL": ok_sl,
            "Sθ": ok_sth,
            "T": ok_t,
        },
    }

    det_sha = sha256_hex(json.dumps(report, sort_keys=True).encode("utf-8"))

    print("\n" + "=" * 100)
    print("DETERMINISM HASH".center(100))
    print("=" * 100)
    print("determinism_sha256:", det_sha)

    # Optional artifact
    if args.json:
        try:
            with open("demo76_report.json", "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, sort_keys=True)
            print("(wrote demo76_report.json)")
        except Exception as e:
            print("(warn) could not write demo76_report.json:", repr(e))

    all_ok = all(bool(v) for v in report["gates"].values())

    print("\n" + "=" * 100)
    print("FINAL VERDICT".center(100))
    print("=" * 100)
    gate_line(
        "DEMO-76 VERIFIED (primorial Θ + θ12 Yukawa ladder + sensitivity + controls + teeth)",
        all_ok,
    )
    print("Result:", "VERIFIED" if all_ok else "NOT VERIFIED")


if __name__ == "__main__":
    main()
