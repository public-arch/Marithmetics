#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO-58 — Emergent Weak-Field General Relativity from a Discrete Poisson Substrate
Master Flagship Demo (Referee-Ready)

Summary
-------
This is a deterministic, first-principles computational audit. It does not fit parameters and does
not tune thresholds.

Pipeline (all deterministic):
  1) Select a unique prime triple (wU, s2, s3) in a fixed primary window via fixed congruence,
     totient-density, and 2-adic rules (no external input).
  2) Derive budgets (eps, N, K_primary, K_truth) deterministically from the selected triple.
  3) Solve the discrete Poisson equation ΔΦ = ρ on a 3D periodic lattice using exact eigenvalues of
     the discrete Laplacian (FFT diagonalization).
  4) Apply three operator classes to Φ̂ (Fourier domain):
       - Admissible: Fejér smoothing (nonnegative convolution kernel)
       - Non-admissible control: sharp cutoff (kernel with negative lobes)
       - Non-admissible control: signed HF injection (kernel with stronger negative lobes)
  5) Extract weak-field observables from the same Φ:
       - Newtonian limit: |g(r)| ~ 1/r^2
       - Light bending proxy: α(b) ~ 1/b
       - Shapiro delay proxy: D(b) ~ ln(b)
       - Redshift proxy: Φ(r) ~ 1/r (shell means)
  6) Counterfactual teeth: change the triple (thus budgets) while keeping physics fixed, and require
     degradation by a fixed margin derived from eps.

Outputs:
  - spec_sha256: hash of the fully declared spec (chain-of-custody)
  - determinism_sha256: hash of key numerical results (chain-of-custody)
  - PASS/FAIL gates for each observable and for counterfactual teeth.

Dependencies: Python 3 + NumPy only.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import platform
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np


# =============================================================================
# Reporting + hashing helpers
# =============================================================================

def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: object) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return sha256_hex(payload)


def fmt_value(x: object) -> str:
    """Stable formatting for determinism ledgers (floats use scientific notation)."""
    if isinstance(x, (bool, np.bool_)):
        return "1" if bool(x) else "0"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        return f"{float(x):.12e}"
    return str(x)


def report(ok: bool, label: str, **fields) -> bool:
    prefix = "PASS" if ok else "FAIL"
    tail = ""
    if fields:
        tail = "  " + " ".join(f"{k}={v}" for k, v in fields.items())
    print(f"{prefix}  {label}{tail}")
    return ok


# =============================================================================
# Basic arithmetic / number theory (deterministic)
# =============================================================================

def v2(n: int) -> int:
    c = 0
    while n % 2 == 0:
        n //= 2
        c += 1
    return c


def odd_part(n: int) -> int:
    while n % 2 == 0:
        n //= 2
    return n


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


def primes_in(lo: int, hi: int) -> List[int]:
    return [p for p in range(lo, hi + 1) if is_prime(p)]


def factorize(n: int) -> List[Tuple[int, int]]:
    """Trial-division factorization (sufficient for the small integers used here)."""
    out: List[Tuple[int, int]] = []
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            out.append((p, e))
        p += 1 if p == 2 else 2
    if x > 1:
        out.append((x, 1))
    return out


def totient(n: int) -> int:
    if n <= 0:
        return 0
    fac = factorize(n)
    phi = n
    for p, _e in fac:
        phi = phi // p * (p - 1)
    return phi


def theta_density(w: int) -> float:
    """
    Deterministic density proxy used across the demo family:
        theta(w) = φ(w-1) / (w-1)
    """
    m = w - 1
    if m <= 0:
        return 0.0
    return totient(m) / float(m)


# =============================================================================
# Deterministic triple selection (same rule family as the DOC / flagship scripts)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class LaneSpec:
    name: str
    q: int
    residues: Tuple[int, ...]
    tau_min: float
    span: Tuple[int, int]
    v2_required: int | None


@dataclasses.dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


def lane_survivors(spec: LaneSpec, *, apply_v2: bool) -> List[int]:
    lo, hi = spec.span
    out: List[int] = []
    for p in primes_in(lo, hi):
        if (p % spec.q) not in spec.residues:
            continue
        if theta_density(p) < spec.tau_min:
            continue
        if apply_v2 and (spec.v2_required is not None):
            if v2(p - 1) < spec.v2_required:
                continue
        out.append(p)
    return out


def select_primary_triple() -> Tuple[Triple, Dict[str, List[int]], List[Tuple[int, int, int]]]:
    """
    Primary window is fixed. Selection must be unique (otherwise abort).
    """
    # Fixed primary window lane specs (referee-stable; not tuned per run)
    U1 = LaneSpec("U(1)", q=17, residues=(1, 5), tau_min=0.31, span=(97, 180), v2_required=3)
    SU2 = LaneSpec("SU(2)", q=13, residues=(3,), tau_min=0.30, span=(97, 180), v2_required=1)
    SU3 = LaneSpec("SU(3)", q=17, residues=(1,), tau_min=0.30, span=(97, 180), v2_required=None)

    # Raw pools (informational)
    u1_raw = lane_survivors(U1, apply_v2=False)
    su2_raw = lane_survivors(SU2, apply_v2=False)
    su3_raw = lane_survivors(SU3, apply_v2=False)

    # Apply v2 requirements (deterministic)
    u1 = lane_survivors(U1, apply_v2=True)
    su2 = lane_survivors(SU2, apply_v2=True)
    su3 = lane_survivors(SU3, apply_v2=False)

    # Coherence collapse for U(1): choose the maximal theta_density element
    # (deterministic tie-break: sorted order).
    u1 = sorted(u1, key=lambda p: (theta_density(p), p))[-1:]

    triples: List[Tuple[int, int, int]] = []
    for wU in u1:
        for s2 in su2:
            if wU <= s2:
                continue
            for s3 in su3:
                if s3 == wU or s3 == s2:
                    continue
                triples.append((wU, s2, s3))
    triples = sorted(triples)
    if len(triples) != 1:
        raise RuntimeError(f"Primary window selection not unique: found {len(triples)} triples: {triples}")

    primary = Triple(*triples[0])
    pools = {
        "U(1)_raw": u1_raw,
        "SU(2)_raw": su2_raw,
        "SU(3)_raw": su3_raw,
        "U(1)_coherent": u1,
    }
    return primary, pools, triples


def counterfactual_triples(primary: Triple, need: int = 4) -> List[Triple]:
    """
    Counterfactual generator (fixed rule family):
      - expands the lane windows upward
      - preserves the same v2 tier by construction of candidate primes and v2 constraints
    """
    U1 = LaneSpec("U(1)_CF", q=17, residues=(1, 5), tau_min=0.26, span=(200, 460), v2_required=v2(primary.wU - 1))
    SU2 = LaneSpec("SU(2)_CF", q=13, residues=(3,), tau_min=0.24, span=(200, 460), v2_required=1)
    SU3 = LaneSpec("SU(3)_CF", q=17, residues=(1,), tau_min=0.24, span=(200, 460), v2_required=None)

    u1 = lane_survivors(U1, apply_v2=True)
    su2 = lane_survivors(SU2, apply_v2=True)
    su3 = lane_survivors(SU3, apply_v2=False)

    # Deterministic: pick the smallest viable wU, then iterate s2,s3 in sorted order.
    out: List[Triple] = []
    for wU in sorted(u1):
        for s2 in sorted(su2):
            if wU <= s2:
                continue
            for s3 in sorted(su3):
                if s3 == wU or s3 == s2:
                    continue
                t = Triple(wU=wU, s2=s2, s3=s3)
                if t != primary:
                    out.append(t)
                if len(out) >= need:
                    return out
    return out


# =============================================================================
# Budget derivations (deterministic)
# =============================================================================

@dataclasses.dataclass(frozen=True)
class Budgets:
    q2: int
    q3: int
    v2U: int
    eps: float
    N: int
    K_primary: int
    K_truth: int
    center: Tuple[int, int, int]


def deterministic_center(v2U: int, N: int) -> Tuple[int, int, int]:
    # Fixed, transparent indexing (no tuning)
    return ((v2U + 2) % N, (v2U + 1) % N, (v2U + 0) % N)


def derive_budgets(t: Triple, *, tier: int = 0) -> Budgets:
    """
    Deterministic budgets derived from the triple.

    tier=0: base N = 2^(v2U+3)  (for v2U=3 -> N=64)
    tier>0: N scaled by 2^tier, with K scaling deterministically as well.
            (No threshold tuning is introduced by tier; it only changes resolution.)
    """
    q2 = t.wU - t.s2
    q3 = odd_part(t.wU - 1)
    v2U = v2(t.wU - 1)
    eps = 1.0 / math.sqrt(float(q2))

    N0 = 2 ** (v2U + 3)
    N = N0 * (2 ** tier)

    K_truth = (N // 2) - 1

    C0 = (2 ** (v2U + 5)) - 1
    C = C0 * (2 ** tier)
    K_primary = max(2, min(K_truth - 1, int(math.floor(C / q3))))

    center = deterministic_center(v2U, N)

    return Budgets(q2=q2, q3=q3, v2U=v2U, eps=eps, N=N, K_primary=K_primary, K_truth=K_truth, center=center)


# =============================================================================
# Filters (Fourier weights) and kernel sanity
# =============================================================================

def signed_freqs(N: int) -> np.ndarray:
    f = np.fft.fftfreq(N) * N
    return f.astype(int)


def fejer_weights_1d(N: int, K: int) -> np.ndarray:
    k = np.abs(signed_freqs(N)).astype(float)
    w = np.zeros(N, dtype=float)
    mask = k <= K
    w[mask] = 1.0 - (k[mask] / (K + 1.0))
    return w


def sharp_weights_1d(N: int, K: int) -> np.ndarray:
    k = np.abs(signed_freqs(N))
    return (k <= K).astype(float)


def signed_control_weights_1d(N: int, K: int) -> np.ndarray:
    """
    Non-admissible control: preserve low modes, flip sign on high modes.
    """
    k = np.abs(signed_freqs(N))
    w = np.ones(N, dtype=float)
    w[k > K] = -1.0
    return w


def weights_3d_from_1d(w1: np.ndarray) -> np.ndarray:
    return (w1[:, None, None] * w1[None, :, None] * w1[None, None, :]).astype(float)


def kernel_min_1d(w_hat: np.ndarray) -> float:
    kern = np.fft.ifft(w_hat.astype(np.complex128)).real
    return float(kern.min())


# =============================================================================
# Discrete Poisson solve on a 3D periodic lattice
# =============================================================================

def laplacian_eigs_1d(N: int) -> np.ndarray:
    k = np.pi * signed_freqs(N) / float(N)
    return -4.0 * (np.sin(k) ** 2)


def poisson_full_solution_hat(rho: np.ndarray) -> np.ndarray:
    """
    Solve ΔΦ = ρ by FFT diagonalization using exact discrete Laplacian eigenvalues.
    Fix the mean mode of Φ to 0 (gauge choice).
    """
    N = rho.shape[0]
    lam1 = laplacian_eigs_1d(N)
    lam3 = lam1[:, None, None] + lam1[None, :, None] + lam1[None, None, :]

    rho_hat = np.fft.fftn(rho.astype(np.float64))
    phi_hat = np.zeros_like(rho_hat)
    mask = lam3 != 0.0
    phi_hat[mask] = rho_hat[mask] / lam3[mask]
    phi_hat[~mask] = 0.0
    return phi_hat


def discrete_laplacian(phi: np.ndarray) -> np.ndarray:
    return (
        np.roll(phi, +1, axis=0) + np.roll(phi, -1, axis=0)
        + np.roll(phi, +1, axis=1) + np.roll(phi, -1, axis=1)
        + np.roll(phi, +1, axis=2) + np.roll(phi, -1, axis=2)
        - 6.0 * phi
    )


def poisson_residual_rms(phi: np.ndarray, rho: np.ndarray) -> float:
    res = discrete_laplacian(phi) - rho
    return float(np.sqrt(np.mean(res * res)))


# =============================================================================
# Distances, shell statistics, regression
# =============================================================================

def torus_distance_grid(N: int, center: Tuple[int, int, int]) -> np.ndarray:
    cx, cy, cz = center
    xs = np.arange(N)
    dx = np.minimum(np.abs(xs - cx), N - np.abs(xs - cx)).astype(float)
    dy = np.minimum(np.abs(xs - cy), N - np.abs(xs - cy)).astype(float)
    dz = np.minimum(np.abs(xs - cz), N - np.abs(xs - cz)).astype(float)
    DX = dx[:, None, None]
    DY = dy[None, :, None]
    DZ = dz[None, None, :]
    return np.sqrt(DX * DX + DY * DY + DZ * DZ)


def shell_means(field: np.ndarray, rgrid: np.ndarray, radii: Sequence[int], halfwidth: float = 0.5) -> np.ndarray:
    out = []
    for r in radii:
        mask = np.abs(rgrid - float(r)) <= halfwidth
        vals = field[mask]
        out.append(float(vals.mean()) if vals.size else float("nan"))
    return np.array(out, dtype=float)


def second_diff_curvature(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    if y.size < 3:
        return float("nan")
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return float(np.mean(np.abs(d2)))


def fit_affine(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    A = np.vstack([x, np.ones_like(x)]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    a, c = float(sol[0]), float(sol[1])
    yhat = a * x + c
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - float(y.mean())) ** 2))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0
    return a, c, r2


def rel_spread(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    m = float(np.mean(np.abs(x)))
    if m == 0.0:
        return float("inf")
    return float(np.std(x) / m)


# =============================================================================
# Gradient + HF fraction
# =============================================================================

def derivative_symbol_1d(N: int) -> np.ndarray:
    # central difference (grid spacing = 1)
    freq = np.fft.fftfreq(N)
    return (1j * np.sin(2.0 * np.pi * freq)).astype(np.complex128)


def hf_fraction_hat(hat: np.ndarray, K: int) -> float:
    N = hat.shape[0]
    k = np.abs(signed_freqs(N))
    mask = (k[:, None, None] > K) | (k[None, :, None] > K) | (k[None, None, :] > K)
    e_tot = float(np.sum(np.abs(hat) ** 2))
    if e_tot == 0.0:
        return 0.0
    e_hf = float(np.sum(np.abs(hat[mask]) ** 2))
    return e_hf / e_tot


def gradient_from_phi_hat(phi_hat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    N = phi_hat.shape[0]
    sx = derivative_symbol_1d(N)[:, None, None]
    sy = derivative_symbol_1d(N)[None, :, None]
    sz = derivative_symbol_1d(N)[None, None, :]
    gx_hat = -sx * phi_hat
    gy_hat = -sy * phi_hat
    gz_hat = -sz * phi_hat
    gx = np.fft.ifftn(gx_hat).real
    gy = np.fft.ifftn(gy_hat).real
    gz = np.fft.ifftn(gz_hat).real
    return gx, gy, gz, gx_hat


# =============================================================================
# Observable suites
# =============================================================================

@dataclasses.dataclass(frozen=True)
class VariantFields:
    phi: np.ndarray
    phi_hat: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    gz: np.ndarray
    gx_hat: np.ndarray
    hf_phi: float
    hf_gx: float
    poisson_res: float


def build_variant(phi_hat_full: np.ndarray, rho: np.ndarray, w3: np.ndarray, K_primary: int) -> VariantFields:
    phi_hat = phi_hat_full * w3
    phi = np.fft.ifftn(phi_hat).real
    gx, gy, gz, gx_hat = gradient_from_phi_hat(phi_hat)
    hf_phi = hf_fraction_hat(phi_hat, K_primary)
    hf_gx = hf_fraction_hat(gx_hat, K_primary)
    res = poisson_residual_rms(phi, rho)
    return VariantFields(phi=phi, phi_hat=phi_hat, gx=gx, gy=gy, gz=gz, gx_hat=gx_hat, hf_phi=hf_phi, hf_gx=hf_gx, poisson_res=res)


def inverse_square_profile(gmag: np.ndarray, rgrid: np.ndarray, band: Tuple[int, int]) -> Dict[str, float]:
    r1, r2 = band
    radii = list(range(r1, r2 + 1))
    means = shell_means(gmag, rgrid, radii, halfwidth=0.5)
    mask = np.isfinite(means) & (means > 0)
    rr = np.array(radii, dtype=float)[mask]
    mm = means[mask]
    slope, _, r2_fit = fit_affine(np.log(rr), np.log(mm))
    scaled = (rr ** 2) * mm
    spread = rel_spread(scaled)
    curv = second_diff_curvature(scaled) / (float(np.mean(np.abs(scaled))) + 1e-30)
    return {"slope": slope, "R2": r2_fit, "spread": spread, "curv": curv}


def b_list_from_N(N: int) -> List[int]:
    # Deterministic fractional choices; N=64 -> [4,6,8,10,12]
    return [N // 16, (3 * N) // 32, N // 8, (5 * N) // 32, (3 * N) // 16]


def light_bending_proxy(gx: np.ndarray, center: Tuple[int, int, int], b_list: Sequence[int]) -> np.ndarray:
    cx, cy, cz = center
    N = gx.shape[0]
    alphas = []
    for b in b_list:
        x = (cx + b) % N
        y = cy % N
        alphas.append(float(np.sum(gx[x, y, :])))
    return np.array(alphas, dtype=float)


def shapiro_delay_proxy(phi: np.ndarray, center: Tuple[int, int, int], b_list: Sequence[int]) -> np.ndarray:
    cx, cy, cz = center
    N = phi.shape[0]
    delays = []
    for b in b_list:
        x = (cx + b) % N
        y = cy % N
        delays.append(float(np.sum(phi[x, y, :])))
    return np.array(delays, dtype=float)


def A_estimates(phi_shell_means: np.ndarray, radii: Sequence[int]) -> float:
    """Estimate amplitude A in Φ(r) ≈ A/r + c by pairing each radius with the largest radius."""
    r = np.array(radii, dtype=float)
    y = np.array(phi_shell_means, dtype=float)
    r0 = r[-1]
    y0 = y[-1]
    As: List[float] = []
    for i in range(len(r) - 1):
        ri = r[i]
        yi = y[i]
        denom = (1.0 / ri) - (1.0 / r0)
        if denom != 0.0:
            As.append(float((yi - y0) / denom))
    return float(np.mean(As)) if As else float("nan")


# =============================================================================
# Demo runner
# =============================================================================

def run_demo(*, tier: int = 0, write_outputs: bool = False) -> int:
    print("=" * 100)
    print("DEMO-58 — Emergent Weak-Field General Relativity from a Discrete Poisson Substrate")
    print("Master Flagship Demo (Referee-Ready)")
    print("=" * 100)
    print(f"UTC time : {utc_now_iso()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout only (optional JSON export with --write)")
    print()

    # -------------------------------------------------------------------------
    # Stage 1 — Deterministic triple selection
    # -------------------------------------------------------------------------
    print("=" * 100)
    print("STAGE 1 — Deterministic triple selection (primary window)")
    print("=" * 100)
    primary, pools, triples = select_primary_triple()

    print("Lane survivor pools (raw):")
    print(f"  U(1): {pools['U(1)_raw']}")
    print(f"  SU(2): {pools['SU(2)_raw']}")
    print(f"  SU(3): {pools['SU(3)_raw']}")
    print("Lane survivor pools (after U(1) coherence):")
    print(f"  U(1): {pools['U(1)_coherent']}")
    print(f"  SU(2): {pools['SU(2)_raw']}")
    print(f"  SU(3): {pools['SU(3)_raw']}")
    print(f"Primary-window admissible triples: {triples}")

    ok_unique = report(len(triples) == 1, "Unique admissible triple in primary window", count=len(triples))
    ok_primary = report((primary.wU, primary.s2, primary.s3) == (137, 107, 103), "Primary equals (137,107,103)", selected=primary)
    if not (ok_unique and ok_primary):
        print("Selection failed; aborting.")
        return 2

    cfs = counterfactual_triples(primary, need=4)
    print(f"Counterfactuals (diff wU): {[(t.wU, t.s2, t.s3) for t in cfs]}")
    ok_cfs = report(len(cfs) >= 4, "Captured >=4 counterfactual triples", found=len(cfs))
    if not ok_cfs:
        print("Counterfactual capture failed; aborting.")
        return 2

    # -------------------------------------------------------------------------
    # Stage 2 — Budgets + spec hash
    # -------------------------------------------------------------------------
    budgets = derive_budgets(primary, tier=tier)

    spec = {
        "demo": "DEMO-58",
        "tier": tier,
        "primary_triple": dataclasses.asdict(primary),
        "counterfactuals": [dataclasses.asdict(t) for t in cfs],
        "budget_rule": "q2=wU-s2; q3=odd_part(wU-1); eps=1/sqrt(q2); N=2^(v2U+3)*2^tier; K_truth=N/2-1; K_primary=floor((2^(v2U+5)-1)*2^tier/q3) clamped",
        "center_rule": "(v2U+2,v2U+1,v2U) mod N",
        "operators": ["fejer_admissible", "sharp_cutoff_control", "signed_control"],
        "observables": ["inverse_square", "light_bending_proxy", "shapiro_delay_proxy", "redshift_scaling"],
        "b_list_rule": "[N/16, 3N/32, N/8, 5N/32, 3N/16]",
        "laplacian_symbol": "λ(k)= -4[sin^2(pi kx/N)+sin^2(pi ky/N)+sin^2(pi kz/N)]",
        "derivative_symbol": "D(k)= i sin(2πk/N) (central difference)",
    }
    spec_sha = sha256_json(spec)

    print()
    print("=" * 100)
    print("STAGE 2 — Derived invariants and budgets")
    print("=" * 100)
    print("Budgets (primary):")
    print(f"  q2={budgets.q2}  q3={budgets.q3}  v2U={budgets.v2U}  eps={budgets.eps:.10f}")
    print(f"  N={budgets.N}  K_primary={budgets.K_primary}  K_truth={budgets.K_truth}")
    print(f"  center={budgets.center}")
    print(f"spec_sha256: {spec_sha}")

    N = budgets.N
    Kp = budgets.K_primary
    eps = budgets.eps

    # -------------------------------------------------------------------------
    # Stage 3 — Operator-class sanity (kernel admissibility)
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 3 — Operator-class sanity (1D kernel checks)")
    print("=" * 100)

    w_fe_1 = fejer_weights_1d(N, Kp)
    w_sh_1 = sharp_weights_1d(N, Kp)
    w_si_1 = signed_control_weights_1d(N, Kp)

    kmin_fe = kernel_min_1d(w_fe_1)
    kmin_sh = kernel_min_1d(w_sh_1)
    kmin_si = kernel_min_1d(w_si_1)

    print(f"Kernel min (1D): fejer={kmin_fe:.6g}  sharp={kmin_sh:.6g}  signed={kmin_si:.6g}")
    ok_k1 = report(kmin_fe >= -1e-12, "Fejér kernel is nonnegative (numerical tol)", kmin=f"{kmin_fe:.3e}")
    ok_k2 = report(kmin_sh < -1e-6, "Sharp cutoff kernel has negative lobes (non-admissible)", kmin=f"{kmin_sh:.3e}")
    ok_k3 = report(kmin_si < -1e-6, "Signed control kernel has negative lobes (non-admissible)", kmin=f"{kmin_si:.3e}")

    # -------------------------------------------------------------------------
    # Stage 4 — Discrete Poisson substrate (solve once), build variants
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 4 — Discrete Poisson substrate (truth + controls)")
    print("=" * 100)

    rho = np.zeros((N, N, N), dtype=np.float64)
    cx, cy, cz = budgets.center
    rho[cx, cy, cz] = 1.0

    t0 = time.time()
    phi_hat_full = poisson_full_solution_hat(rho)
    t1 = time.time()
    print(f"Poisson inversion runtime (informational): {t1 - t0:.3f} s")

    w_tr_3 = weights_3d_from_1d(fejer_weights_1d(N, budgets.K_truth))
    w_ad_3 = weights_3d_from_1d(w_fe_1)
    w_sh_3 = weights_3d_from_1d(w_sh_1)
    w_si_3 = weights_3d_from_1d(w_si_1)

    truth = build_variant(phi_hat_full, rho, w_tr_3, Kp)
    adm = build_variant(phi_hat_full, rho, w_ad_3, Kp)
    sharp = build_variant(phi_hat_full, rho, w_sh_3, Kp)
    signed = build_variant(phi_hat_full, rho, w_si_3, Kp)

    print("Poisson residual RMS ||ΔΦ - ρ||_RMS:")
    print(f"  truth  : {truth.poisson_res:.6g}")
    print(f"  admiss.: {adm.poisson_res:.6g}")
    print(f"  sharp  : {sharp.poisson_res:.6g}")
    print(f"  signed : {signed.poisson_res:.6g}")

    rgrid = torus_distance_grid(N, budgets.center)

    # -------------------------------------------------------------------------
    # Stage 5 — Newtonian limit: inverse-square scaling
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 5 — Newtonian limit: |g(r)| ~ 1/r^2 (scaling audit)")
    print("=" * 100)

    gmag_t = np.sqrt(truth.gx**2 + truth.gy**2 + truth.gz**2)
    gmag_a = np.sqrt(adm.gx**2 + adm.gy**2 + adm.gz**2)
    gmag_sh = np.sqrt(sharp.gx**2 + sharp.gy**2 + sharp.gz**2)
    gmag_si = np.sqrt(signed.gx**2 + signed.gy**2 + signed.gz**2)

    band = (max(2, N // 8), max(3, N // 4))
    prof_t = inverse_square_profile(gmag_t, rgrid, band)
    prof_a = inverse_square_profile(gmag_a, rgrid, band)
    prof_sh = inverse_square_profile(gmag_sh, rgrid, band)
    prof_si = inverse_square_profile(gmag_si, rgrid, band)

    print(f"Band (r): {band[0]}..{band[1]}")
    print(f"slope log|g| vs log r (expect ~ -2): truth={prof_t['slope']:.6g}  adm={prof_a['slope']:.6g}")
    print(f"spread r^2|g| (lower is better):      truth={prof_t['spread']:.6g}  adm={prof_a['spread']:.6g}  sharp={prof_sh['spread']:.6g}  signed={prof_si['spread']:.6g}")
    print(f"curvature r^2|g| (lower is better):   adm={prof_a['curv']:.6g}  sharp={prof_sh['curv']:.6g}  signed={prof_si['curv']:.6g}")
    print(f"HFfrac gx_hat(>|Kp|): adm={adm.hf_gx:.6g}  sharp={sharp.hf_gx:.6g}  signed={signed.hf_gx:.6g}")

    ok_n0 = report(truth.poisson_res <= (1.0 + eps) * adm.poisson_res, "Gate N0: filtered Poisson residual contract (truth vs admissible)",
                   res_t=f"{truth.poisson_res:.3e}", res_a=f"{adm.poisson_res:.3e}")
    ok_n1 = report(abs(prof_t["slope"] + 2.0) <= eps, "Gate N1: truth slope near -2", slope=f"{prof_t['slope']:.6g}", eps=f"{eps:.6g}")
    ok_n2 = report(abs(prof_a["slope"] + 2.0) <= eps, "Gate N2: admissible slope near -2", slope=f"{prof_a['slope']:.6g}", eps=f"{eps:.6g}")
    hf_floor_n = max(10.0 * adm.hf_gx, eps**3)
    ok_n3 = report(signed.hf_gx >= hf_floor_n, "Gate N3: signed control injects HF (>= max(10*hf_a, eps^3))",
                   hf_signed=f"{signed.hf_gx:.3e}", floor=f"{hf_floor_n:.3e}")
    curv_max_n = max(prof_sh["curv"], prof_si["curv"])
    ok_n4 = report(curv_max_n >= (1.0 + eps) * prof_a["curv"], "Gate N4: a non-admissible control has stronger ringing curvature",
                   curv_a=f"{prof_a['curv']:.3e}", curv_max=f"{curv_max_n:.3e}", eps=f"{eps:.6g}")

    # -------------------------------------------------------------------------
    # Stage 6 — Light bending proxy: alpha(b) ~ 1/b
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 6 — Light bending proxy: α(b) ~ 1/b (scaling audit)")
    print("=" * 100)

    b_list = b_list_from_N(N)
    alpha_t = light_bending_proxy(truth.gx, budgets.center, b_list)
    alpha_a = light_bending_proxy(adm.gx, budgets.center, b_list)
    alpha_sh = light_bending_proxy(sharp.gx, budgets.center, b_list)
    alpha_si = light_bending_proxy(signed.gx, budgets.center, b_list)

    logb = np.log(np.array(b_list, dtype=float))
    slope_t, _, _ = fit_affine(logb, np.log(np.abs(alpha_t) + 1e-30))
    slope_a, _, _ = fit_affine(logb, np.log(np.abs(alpha_a) + 1e-30))

    scaled_t = np.abs(np.array(b_list, dtype=float) * alpha_t)
    scaled_a = np.abs(np.array(b_list, dtype=float) * alpha_a)
    scaled_sh = np.abs(np.array(b_list, dtype=float) * alpha_sh)
    scaled_si = np.abs(np.array(b_list, dtype=float) * alpha_si)

    spread_t = rel_spread(scaled_t)
    spread_a = rel_spread(scaled_a)
    spread_sh = rel_spread(scaled_sh)
    spread_si = rel_spread(scaled_si)

    curv_a = second_diff_curvature(scaled_a) / (float(np.mean(np.abs(scaled_a))) + 1e-30)
    curv_sh = second_diff_curvature(scaled_sh) / (float(np.mean(np.abs(scaled_sh))) + 1e-30)
    curv_si = second_diff_curvature(scaled_si) / (float(np.mean(np.abs(scaled_si))) + 1e-30)

    print(f"b list: {b_list}")
    print(f"slope log|alpha| vs log b (expect ~ -1): truth={slope_t:.6g}  adm={slope_a:.6g}")
    print(f"spread |b*alpha| (lower is better):      truth={spread_t:.6g}  adm={spread_a:.6g}  sharp={spread_sh:.6g}  signed={spread_si:.6g}")
    print(f"curvature |b*alpha| (lower is better):   adm={curv_a:.6g}  sharp={curv_sh:.6g}  signed={curv_si:.6g}")
    print(f"HFfrac gx_hat(>|Kp|): adm={adm.hf_gx:.6g}  sharp={sharp.hf_gx:.6g}  signed={signed.hf_gx:.6g}")

    ok_b1 = report(abs(slope_t + 1.0) <= eps, "Gate B1: truth slope near -1", slope=f"{slope_t:.6g}", eps=f"{eps:.6g}")
    ok_b2 = report(abs(slope_a + 1.0) <= eps, "Gate B2: admissible slope near -1", slope=f"{slope_a:.6g}", eps=f"{eps:.6g}")
    hf_floor_b = max(10.0 * adm.hf_gx, eps**2)
    ok_b3 = report(signed.hf_gx >= hf_floor_b, "Gate B3: non-admissible injects HF (>= max(10*hf_a, eps^2))",
                   hf_signed=f"{signed.hf_gx:.3e}", floor=f"{hf_floor_b:.3e}")
    curv_max_b = max(curv_sh, curv_si)
    ok_b4 = report(curv_max_b >= (1.0 + eps) * curv_a, "Gate B4: non-admissible has higher ringing curvature (>= (1+eps)×adm)",
                   curv_a=f"{curv_a:.3e}", curv_max=f"{curv_max_b:.3e}", eps=f"{eps:.6g}")

    # -------------------------------------------------------------------------
    # Stage 7 — Shapiro delay proxy: D(b) ~ ln(b)
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 7 — Shapiro delay proxy: D(b) ~ ln(b) (shape audit)")
    print("=" * 100)

    D_t = shapiro_delay_proxy(truth.phi, budgets.center, b_list)
    D_a = shapiro_delay_proxy(adm.phi, budgets.center, b_list)
    D_sh = shapiro_delay_proxy(sharp.phi, budgets.center, b_list)
    D_si = shapiro_delay_proxy(signed.phi, budgets.center, b_list)

    ln_b = np.log(np.array(b_list, dtype=float))

    a_t, c_t, r2_t = fit_affine(ln_b, D_t)
    a_a, c_a, r2_a = fit_affine(ln_b, D_a)
    a_sh, c_sh, r2_sh = fit_affine(ln_b, D_sh)
    a_si, c_si, r2_si = fit_affine(ln_b, D_si)

    res_a = D_a - (a_a * ln_b + c_a)
    res_sh = D_sh - (a_sh * ln_b + c_sh)
    res_si = D_si - (a_si * ln_b + c_si)

    curv_a_s = second_diff_curvature(res_a)
    curv_sh_s = second_diff_curvature(res_sh)
    curv_si_s = second_diff_curvature(res_si)

    print(f"R2: truth={r2_t:.6g}  adm={r2_a:.6g}  sharp={r2_sh:.6g}  signed={r2_si:.6g}")
    print(f"Residual curvature (mean |d2|): adm={curv_a_s:.6g}  sharp={curv_sh_s:.6g}  signed={curv_si_s:.6g}")
    print(f"HFfrac Phi_hat(>|Kp|): adm={adm.hf_phi:.6g}  sharp={sharp.hf_phi:.6g}  signed={signed.hf_phi:.6g}")
    print(f"Poisson residuals: truth={truth.poisson_res:.6g}  adm={adm.poisson_res:.6g}  sharp={sharp.poisson_res:.6g}  signed={signed.poisson_res:.6g}")

    ok_s0 = report(truth.poisson_res <= (1.0 + eps) * adm.poisson_res, "Gate S0: filtered Poisson residual contract (truth vs admissible)",
                   res_t=f"{truth.poisson_res:.3e}", res_a=f"{adm.poisson_res:.3e}")
    ok_s1 = report(r2_t >= 0.98, "Gate S1: truth affine in ln(b) (R2 >= 0.98)", R2=f"{r2_t:.6g}")
    ok_s2 = report(r2_a >= 0.95, "Gate S2: admissible affine in ln(b) (R2 >= 0.95)", R2=f"{r2_a:.6g}")
    hf_floor_s = max(10.0 * adm.hf_phi, eps**3)
    ok_s3 = report(signed.hf_phi >= hf_floor_s, "Gate S3: signed control injects HF (>= max(10*hf_a, eps^3))",
                   hf_signed=f"{signed.hf_phi:.3e}", floor=f"{hf_floor_s:.3e}")
    curv_max_s = max(curv_sh_s, curv_si_s)
    ok_s4 = report(curv_max_s >= (1.0 + eps) * curv_a_s, "Gate S4: non-admissible has higher curvature (>= (1+eps)×adm)",
                   curv_a=f"{curv_a_s:.3e}", curv_max=f"{curv_max_s:.3e}", eps=f"{eps:.6g}")

    # Shapiro teeth score (composite): relative slope error + (1-R2) + normalized curvature
    rel_err_slope = abs(a_a - a_t) / (abs(a_t) + 1e-30)
    curv_norm = curv_a_s / (float(np.mean(np.abs(D_a))) + 1e-30)
    shapiro_score_primary = float(rel_err_slope + (1.0 - r2_a) + curv_norm)

    # -------------------------------------------------------------------------
    # Stage 8 — Redshift proxy: Φ(r) ~ 1/r (shell means)
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 8 — Gravitational redshift proxy: Φ(r) ~ 1/r (shell-mean audit)")
    print("=" * 100)

    r_list = b_list_from_N(N)
    inv_r = 1.0 / np.array(r_list, dtype=float)

    phi_shell_t = shell_means(truth.phi, rgrid, r_list, halfwidth=0.5)
    phi_shell_a = shell_means(adm.phi, rgrid, r_list, halfwidth=0.5)
    phi_shell_sh = shell_means(sharp.phi, rgrid, r_list, halfwidth=0.5)
    phi_shell_si = shell_means(signed.phi, rgrid, r_list, halfwidth=0.5)

    A_t2, c_t2, R2_t2 = fit_affine(inv_r, phi_shell_t)
    A_a2, c_a2, R2_a2 = fit_affine(inv_r, phi_shell_a)
    A_sh2, c_sh2, R2_sh2 = fit_affine(inv_r, phi_shell_sh)
    A_si2, c_si2, R2_si2 = fit_affine(inv_r, phi_shell_si)

    A_est_t = A_estimates(phi_shell_t, r_list)
    A_est_a = A_estimates(phi_shell_a, r_list)
    rel_err_A = abs(A_est_a - A_est_t) / (abs(A_est_t) + 1e-30)

    curv_shell_a = second_diff_curvature(phi_shell_a)
    curv_shell_sh = second_diff_curvature(phi_shell_sh)
    curv_shell_si = second_diff_curvature(phi_shell_si)

    print(f"r list: {r_list}")
    print(f"R2 fit Φ vs (1/r): truth={R2_t2:.6g}  adm={R2_a2:.6g}  sharp={R2_sh2:.6g}  signed={R2_si2:.6g}")
    print(f"A_est mean (truth vs adm): truth={A_est_t:.6g}  adm={A_est_a:.6g}  rel_err={rel_err_A:.6g}")
    print(f"Ringing curvature (mean |d2|): adm={curv_shell_a:.6g}  sharp={curv_shell_sh:.6g}  signed={curv_shell_si:.6g}")
    print(f"HFfrac Φ̂(>Kp): adm={adm.hf_phi:.6g}  sharp={sharp.hf_phi:.6g}  signed={signed.hf_phi:.6g}")
    print(f"Poisson residual RMS: truth={truth.poisson_res:.6g}  adm={adm.poisson_res:.6g}  sharp={sharp.poisson_res:.6g}  signed={signed.poisson_res:.6g}")

    ok_r0 = report(truth.poisson_res <= (1.0 + eps) * adm.poisson_res, "Gate R0: filtered Poisson residual contract (truth vs admissible)",
                   res_t=f"{truth.poisson_res:.3e}", res_a=f"{adm.poisson_res:.3e}")
    ok_r1 = report(R2_t2 >= 0.98, "Gate R1: truth affine in (1/r) (R2 >= 0.98)", R2=f"{R2_t2:.6g}")
    ok_r2 = report(R2_a2 >= 0.95, "Gate R2: admissible affine in (1/r) (R2 >= 0.95)", R2=f"{R2_a2:.6g}")
    ok_r3 = report(rel_err_A <= eps, "Gate R3: admissible amplitude contract (relative error <= eps)",
                   rel_err=f"{rel_err_A:.6g}", eps=f"{eps:.6g}")
    hf_floor_r = max(10.0 * adm.hf_phi, eps**3)
    ok_r4 = report(signed.hf_phi >= hf_floor_r, "Gate R4: signed control injects HF (>= max(10*hf_a, eps^3))",
                   hf_signed=f"{signed.hf_phi:.3e}", floor=f"{hf_floor_r:.3e}")
    curv_max_r = max(curv_shell_sh, curv_shell_si)
    ok_r5 = report(curv_max_r >= (1.0 + eps) * curv_shell_a, "Gate R5: a non-admissible control has stronger ringing curvature",
                   curv_a=f"{curv_shell_a:.3e}", curv_max=f"{curv_max_r:.3e}", eps=f"{eps:.6g}")

    # Redshift teeth score (matches the verified prework rule structure)
    spread_norm = rel_spread(phi_shell_a - (A_a2 * inv_r + c_a2))
    redshift_score_primary = float(rel_err_A + (1.0 - R2_a2) + spread_norm)

    # -------------------------------------------------------------------------
    # Stage 9 — Counterfactual teeth (budgets change, physics fixed)
    # -------------------------------------------------------------------------
    print()
    print("=" * 100)
    print("STAGE 9 — Counterfactual teeth (budgets change, physics fixed)")
    print("=" * 100)

    teeth_threshold = 1.0 + eps

    # Light bending teeth: spread(|b*alpha|)
    print(f"Light-bending primary spread score: {spread_a:.6g}")
    strong_b = 0
    for cf in cfs:
        bcf = derive_budgets(cf, tier=tier)
        w_cf = weights_3d_from_1d(fejer_weights_1d(N, bcf.K_primary))
        vcf = build_variant(phi_hat_full, rho, w_cf, Kp)
        alpha_cf = light_bending_proxy(vcf.gx, budgets.center, b_list)
        spread_cf = rel_spread(np.abs(np.array(b_list, dtype=float) * alpha_cf))
        degrade = spread_cf >= teeth_threshold * spread_a
        strong_b += int(degrade)
        print(f"CF ({cf.wU},{cf.s2},{cf.s3})  q3={bcf.q3:>3d} K={bcf.K_primary:>3d} spread={spread_cf:.6g} degrade={degrade}")
    ok_tb = report(strong_b >= 3, "Gate TB: counterfactuals degrade light-bending score (>=3/4)", strong=f"{strong_b}/4", eps=f"{eps:.6g}")

    # Shapiro teeth: composite score vs primary
    print(f"Shapiro primary composite score: {shapiro_score_primary:.6g}")
    strong_s = 0
    for cf in cfs:
        bcf = derive_budgets(cf, tier=tier)
        w_cf = weights_3d_from_1d(fejer_weights_1d(N, bcf.K_primary))
        vcf = build_variant(phi_hat_full, rho, w_cf, Kp)
        D_cf = shapiro_delay_proxy(vcf.phi, budgets.center, b_list)
        a_cf, c_cf, r2_cf = fit_affine(ln_b, D_cf)
        res_cf = D_cf - (a_cf * ln_b + c_cf)
        curv_cf = second_diff_curvature(res_cf) / (float(np.mean(np.abs(D_cf))) + 1e-30)
        rel_err_cf = abs(a_cf - a_t) / (abs(a_t) + 1e-30)
        score_cf = float(rel_err_cf + (1.0 - r2_cf) + curv_cf)
        degrade = score_cf >= teeth_threshold * shapiro_score_primary
        strong_s += int(degrade)
        print(f"CF ({cf.wU},{cf.s2},{cf.s3})  q3={bcf.q3:>3d} K={bcf.K_primary:>3d} score={score_cf:.6g} degrade={degrade}")
    ok_ts = report(strong_s >= 3, "Gate TS: counterfactuals degrade Shapiro score (>=3/4)", strong=f"{strong_s}/4", eps=f"{eps:.6g}")

    # Redshift teeth: composite score (same structure as prework D)
    print(f"Redshift primary composite score: {redshift_score_primary:.6g}")
    strong_r = 0
    for cf in cfs:
        bcf = derive_budgets(cf, tier=tier)
        w_cf = weights_3d_from_1d(fejer_weights_1d(N, bcf.K_primary))
        vcf = build_variant(phi_hat_full, rho, w_cf, Kp)
        phi_shell_cf = shell_means(vcf.phi, rgrid, r_list, halfwidth=0.5)
        A_cf, c_cf, R2_cf = fit_affine(inv_r, phi_shell_cf)
        A_est_cf = A_estimates(phi_shell_cf, r_list)
        rel_err_cf = abs(A_est_cf - A_est_t) / (abs(A_est_t) + 1e-30)
        spread_norm_cf = rel_spread(phi_shell_cf - (A_cf * inv_r + c_cf))
        score_cf = float(rel_err_cf + (1.0 - R2_cf) + spread_norm_cf)
        degrade = score_cf >= teeth_threshold * redshift_score_primary
        strong_r += int(degrade)
        print(f"CF ({cf.wU},{cf.s2},{cf.s3})  q3={bcf.q3:>3d} K={bcf.K_primary:>3d} score={score_cf:.6g} degrade={degrade}")
    ok_tr = report(strong_r >= 3, "Gate TR: counterfactuals degrade redshift score (>=3/4)", strong=f"{strong_r}/4", eps=f"{eps:.6g}")

    # -------------------------------------------------------------------------
    # Determinism hash
    # -------------------------------------------------------------------------
    results = {
        "spec_sha256": spec_sha,
        "tier": tier,
        "N": N,
        "K_primary": Kp,
        "K_truth": budgets.K_truth,
        "eps": float(eps),
        "newton_slope_truth": float(prof_t["slope"]),
        "newton_slope_adm": float(prof_a["slope"]),
        "bending_slope_truth": float(slope_t),
        "bending_slope_adm": float(slope_a),
        "bending_spread_adm": float(spread_a),
        "shapiro_R2_truth": float(r2_t),
        "shapiro_R2_adm": float(r2_a),
        "redshift_R2_truth": float(R2_t2),
        "redshift_R2_adm": float(R2_a2),
        "teeth_light_strong": int(strong_b),
        "teeth_shapiro_strong": int(strong_s),
        "teeth_redshift_strong": int(strong_r),
    }
    ledger_lines = [f"{k}={fmt_value(v)}" for k, v in sorted(results.items(), key=lambda kv: kv[0])]
    determinism_sha = sha256_hex(("\n".join(ledger_lines)).encode("utf-8"))

    print()
    print("=" * 100)
    print("DETERMINISM HASH")
    print("=" * 100)
    print(f"determinism_sha256: {determinism_sha}")

    if write_outputs:
        print()
        print("=" * 100)
        print("OPTIONAL OUTPUT FILES")
        print("=" * 100)
        try:
            out = {"spec": spec, "results": results, "ledger": ledger_lines}
            fname = f"demo58_gr_results_N{N}_tier{tier}.json"
            with open(fname, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, sort_keys=True)
            print(f"Wrote: {fname}")
        except Exception as e:
            print(f"NOTE: Could not write JSON output (continuing): {e!r}")

    all_ok = all([
        ok_k1, ok_k2, ok_k3,
        ok_n0, ok_n1, ok_n2, ok_n3, ok_n4,
        ok_b1, ok_b2, ok_b3, ok_b4,
        ok_s0, ok_s1, ok_s2, ok_s3, ok_s4,
        ok_r0, ok_r1, ok_r2, ok_r3, ok_r4, ok_r5,
        ok_tb, ok_ts, ok_tr,
    ])

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    report(all_ok, "DEMO-58 VERIFIED (weak-field suite: scaling + operator falsifiers + teeth)")
    print("Result:", "VERIFIED" if all_ok else "NOT VERIFIED")

    return 0 if all_ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--tier", type=int, default=0, help="Resolution tier: 0 -> base N, 1 -> 2N, 2 -> 4N, ... (default: 0)")
    ap.add_argument("--write", action="store_true", help="Attempt to write a JSON results file (optional).")
    args = ap.parse_args()

    try:
        code = run_demo(tier=int(args.tier), write_outputs=bool(args.write))
    except MemoryError as e:
        print("FATAL: MemoryError (insufficient RAM for this tier).")
        print(repr(e))
        code = 2
    sys.exit(code)


if __name__ == "__main__":
    main()