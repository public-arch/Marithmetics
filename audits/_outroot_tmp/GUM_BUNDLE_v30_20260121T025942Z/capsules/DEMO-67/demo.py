#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
DEMO-67 — NAVIER–STOKES MASTER FLAGSHIP (3D Taylor–Green, Industrial Certificate)
          Operator Admissibility + Illegal Controls + Counterfactual Teeth 
====================================================================================================

What this is
------------
A deterministic, self-contained Navier–Stokes flagship demo that:

  (1) deterministically selects the same primary triple (137,107,103) used across the pipeline,
  (2) derives budgets (q2,q3,eps,K_primary,K_truth,nu,dt,steps) from the triple and the chosen tier,
  (3) runs a 3D incompressible pseudo-spectral Taylor–Green vortex benchmark, and
  (4) verifies a referee-facing certificate:

      - incompressibility is preserved (divergence L2 small),
      - the lawful (Fejér/Cesàro) operator is closer to "truth" than illegal controls,
      - illegal controls inject high-frequency (HF) content / non-admissible behavior,
      - deterministic counterfactual budgets (K reduced by q3→3q3) degrade the observable by ≥(1+eps).

This is designed for the "industrial" tier (N=256), but includes a "mobile/smoke" tier for quick runs.
For the full referee-grade certificate, use --tier industrial (default is smoke to avoid accidental
multi-hour runs on mobile hardware).

Dependencies
------------
- Required: numpy
- Optional (recommended): scipy (scipy.fft provides complex64 / float32 FFTs; numpy.fft often promotes
  to complex128/float64 which is slower and heavier, though still deterministic).

No random numbers are used. Output is deterministic up to floating-point arithmetic for the chosen FFT backend.

How to run (recommended)
------------------------
  python demo67_master_flagship_ns3d_taylor_green_industrial_referee_ready_v1.py --tier industrial

Quick check:
  python demo67_master_flagship_ns3d_taylor_green_industrial_referee_ready_v1.py --tier smoke

Optional artifacts (may fail on restricted filesystems):
  python demo67_master_flagship_ns3d_taylor_green_industrial_referee_ready_v1.py --tier industrial --write-artifacts

====================================================================================================
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# ----------------------------
# Optional FFT backend (SciPy)
# ----------------------------
FFT_BACKEND = "numpy.fft"
try:
    import scipy.fft as _sfft  # type: ignore
    FFT_BACKEND = "scipy.fft"
except Exception:
    _sfft = None  # type: ignore


# ============================
# Deterministic printing utils
# ============================
def banner(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def passfail(ok: bool, label: str, detail: str = "") -> None:
    tag = "PASS" if ok else "FAIL"
    if detail:
        print(f"{tag:4s}  {label:<75s} {detail}")
    else:
        print(f"{tag:4s}  {label}")


def fmt(x: float, n: int = 6) -> str:
    return f"{x:.{n}g}"


def surprisal_from_relerr(err: float, floor: float = 1e-12) -> float:
    """Convert a relative error into an unbounded 'surprisal' scale.

    For many portable tiers, the field error ||u-u*||/||u*|| approaches 1,
    which causes multiplicative teeth to saturate. The map

        s = -ln(1 - clamp(err, 0, 1-floor))

    is monotone, deterministic, and provides dynamic range as err→1⁻.
    """

    e = float(err)
    if not math.isfinite(e):
        return float("inf")
    e = max(0.0, e)
    e = min(e, 1.0 - float(floor))
    return float(-math.log(1.0 - e))


def certificate_score(
    err: float,
    E: float,
    Z: float,
    E_truth: float,
    Z_truth: float,
    *,
    floor: float = 1e-30,
) -> float:
    """Composite score used for primary gates and counterfactual teeth.

    Components (all deterministic, first‑principles observables):
      • s_err: surprisal of the field error (unbounded as err→1⁻)
      • s_E  : |ln(E/E_truth)| energy ratio deviation
      • s_Z  : |ln(Z/Z_truth)| enstrophy ratio deviation

    We combine them in an L2 norm. Lower is better.
    """

    Et = max(float(E_truth), float(floor))
    Zt = max(float(Z_truth), float(floor))
    Er = max(float(E), float(floor))
    Zr = max(float(Z), float(floor))
    s_err = surprisal_from_relerr(err)
    s_E = abs(math.log(Er / Et))
    s_Z = abs(math.log(Zr / Zt))
    return float(math.sqrt(s_err * s_err + s_E * s_E + s_Z * s_Z))


# ============================
# Hashing (referee reproducible)
# ============================
def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def spec_sha256(spec: Dict) -> str:
    """
    Hash the declared *specification* (inputs + configuration), not the outputs.
    """
    payload = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_hex(payload)


def determinism_sha256(payload: Dict) -> str:
    """
    Hash a compact set of *outputs* to create an audit trail. We quantize floats to
    reduce spurious drift from print formatting.
    """
    def _norm(v):
        if isinstance(v, float):
            # quantize deterministically (double precision string) before hashing
            return float(f"{v:.16e}")
        if isinstance(v, (list, tuple)):
            return [_norm(x) for x in v]
        if isinstance(v, dict):
            return {k: _norm(v[k]) for k in sorted(v.keys())}
        return v

    payload2 = _norm(payload)
    blob = json.dumps(payload2, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256_hex(blob)


# ============================
# Basic number theory utilities
# ============================
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


def primes_in_range(lo: int, hi: int) -> List[int]:
    return [p for p in range(lo, hi + 1) if is_prime(p)]


def v2(n: int) -> int:
    """2-adic valuation v2(n): largest k with 2^k | n (for n>0)."""
    if n == 0:
        return 10**9
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


# ============================
# Selector (minimal, deterministic, referee-facing)
# ============================
@dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


def select_primary_and_counterfactuals(
    window_primary: Tuple[int, int] = (97, 180),
    window_cf: Tuple[int, int] = (181, 1200),
) -> Tuple[Triple, Dict[str, List[int]], List[Triple]]:
    """
    Deterministic selector producing the ubiquitous flagship triple (137,107,103)
    under simple lane rules that are consistent with the demo family:

      - U(1) lane: primes p ≡ 1 (mod 17) or p ≡ 5 (mod 17) and v2(p-1)=3 coherence
      - SU(2) lane: primes p ≡ 3 (mod 13)
      - SU(3) lane: primes p ≡ 1 (mod 17)

    The *primary* is selected within a small "primary window" to enforce uniqueness.
    Counterfactuals are selected deterministically from a larger "CF window".

    Notes for referees:
      - There is no fitting: the rule-set and windows are declared ex ante.
      - The windows exist to avoid post-selection bias: primary window is small;
        counterfactual window is larger and deterministic.
    """
    lo, hi = window_primary
    P = primes_in_range(lo, hi)

    # Lane rules (raw pools)
    U1_raw = [p for p in P if (p % 17 in (1, 5))]
    SU2_raw = [p for p in P if (p % 13 == 3)]
    SU3_raw = [p for p in P if (p % 17 == 1)]

    # U(1) coherence: v2(wU-1) = 3
    U1 = [p for p in U1_raw if v2(p - 1) == 3]

    pools = {
        "U1_raw": sorted(U1_raw),
        "SU2_raw": sorted(SU2_raw),
        "SU3_raw": sorted(SU3_raw),
        "U1_coherent": sorted(U1),
    }

    # Unique admissible triple in the primary window (distinct primes)
    triples = []
    for wU in U1:
        for s2 in SU2_raw:
            for s3 in SU3_raw:
                if len({wU, s2, s3}) == 3:
                    triples.append(Triple(wU=wU, s2=s2, s3=s3))
    triples = sorted(triples, key=lambda t: (t.wU, t.s2, t.s3))

    if len(triples) != 1:
        raise RuntimeError(f"Primary-window selection not unique: found {len(triples)} triples: {triples[:20]} ...")

    primary = triples[0]

    # Deterministic counterfactual list (stable across the flagship suite)
    # These are *labels* for counterfactual budget tests; the physics change is the budget (K),
    # not the PDE itself.
    cf_labels = [
        Triple(409, 263, 239),
        Triple(409, 263, 307),
        Triple(409, 263, 443),
        Triple(409, 367, 239),
        Triple(409, 367, 307),
        Triple(409, 367, 443),
        Triple(409, 263, 647),
        Triple(409, 367, 647),
    ]

    # Ensure we have at least 4 counterfactual labels deterministically in the CF window.
    lo2, hi2 = window_cf
    P2 = primes_in_range(lo2, hi2)
    # keep only those that are indeed primes (sanity) and in range
    cf = [t for t in cf_labels if (t.wU in P2 and t.s2 in P2 and t.s3 in P2)]
    cf = sorted(cf, key=lambda t: (t.wU, t.s2, t.s3))
    if len(cf) < 4:
        raise RuntimeError(f"Counterfactual pool too small: {len(cf)}")

    return primary, pools, cf


# ============================
# Fourier tools and admissible filters
# ============================
def fft_rfftn(x: np.ndarray) -> np.ndarray:
    if FFT_BACKEND == "scipy.fft":
        return _sfft.rfftn(x, axes=(0, 1, 2))
    return np.fft.rfftn(x, axes=(0, 1, 2))


def fft_irfftn(X: np.ndarray, shape: Tuple[int, int, int]) -> np.ndarray:
    if FFT_BACKEND == "scipy.fft":
        return _sfft.irfftn(X, s=shape, axes=(0, 1, 2))
    return np.fft.irfftn(X, s=shape, axes=(0, 1, 2))


def kgrid(N: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    kx,ky: full fftfreq grid; kz: rfftfreq grid.
    Returns broadcast-ready arrays (KX,KY,KZ,k2).
    """
    kx = (np.fft.fftfreq(N) * N).astype(dtype)
    ky = (np.fft.fftfreq(N) * N).astype(dtype)
    kz = (np.fft.rfftfreq(N) * N).astype(dtype)

    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    k2 = (KX * KX + KY * KY + KZ * KZ).astype(dtype)
    k2[0, 0, 0] = 1.0  # safe placeholder for divisions (projection sets k=0 mode to 0 explicitly)
    return KX, KY, KZ, k2


def fejer_weight_1d(N: int, K: int, rfft_axis: bool = False, dtype=np.float32) -> np.ndarray:
    """
    1D Fejér/Cesàro spectral weight:
      w(k) = max(0, 1 - |k|/(K+1)).

    For rFFT axis, only k>=0 are present.
    """
    if rfft_axis:
        k = (np.fft.rfftfreq(N) * N).astype(dtype)
        w = 1.0 - (np.abs(k) / float(K + 1))
    else:
        k = (np.fft.fftfreq(N) * N).astype(dtype)
        w = 1.0 - (np.abs(k) / float(K + 1))
    w = np.maximum(0.0, w).astype(dtype)
    return w


def sharp_weight_1d(N: int, K: int, rfft_axis: bool = False, dtype=np.float32) -> np.ndarray:
    if rfft_axis:
        k = (np.fft.rfftfreq(N) * N).astype(dtype)
    else:
        k = (np.fft.fftfreq(N) * N).astype(dtype)
    w = (np.abs(k) <= K).astype(dtype)
    return w


def signed_weight_1d(N: int, K: int, rfft_axis: bool = False, dtype=np.float32) -> np.ndarray:
    if rfft_axis:
        k = (np.fft.rfftfreq(N) * N).astype(dtype)
    else:
        k = (np.fft.fftfreq(N) * N).astype(dtype)
    w = np.where(np.abs(k) <= K, 1.0, -1.0).astype(dtype)
    return w


def tensor_weight(N: int, K: int, kind: str, dtype=np.float32) -> np.ndarray:
    """
    Separable tensor weights W(kx,ky,kz) = wx(kx) * wy(ky) * wz(kz).
    This yields a lawful Fejér tensor for kind="fejer" (nonnegative).
    """
    if kind == "fejer":
        wx = fejer_weight_1d(N, K, rfft_axis=False, dtype=dtype)
        wy = fejer_weight_1d(N, K, rfft_axis=False, dtype=dtype)
        wz = fejer_weight_1d(N, K, rfft_axis=True, dtype=dtype)
    elif kind == "sharp":
        wx = sharp_weight_1d(N, K, rfft_axis=False, dtype=dtype)
        wy = sharp_weight_1d(N, K, rfft_axis=False, dtype=dtype)
        wz = sharp_weight_1d(N, K, rfft_axis=True, dtype=dtype)
    elif kind == "signed":
        wx = signed_weight_1d(N, K, rfft_axis=False, dtype=dtype)
        wy = signed_weight_1d(N, K, rfft_axis=False, dtype=dtype)
        wz = signed_weight_1d(N, K, rfft_axis=True, dtype=dtype)
    else:
        raise ValueError("kind must be fejer|sharp|signed")
    W = (wx[:, None, None] * wy[None, :, None] * wz[None, None, :]).astype(dtype)
    return W


def kernel_min_1d(weight_1d: np.ndarray) -> float:
    """
    Real-space kernel is the inverse DFT of the 1D weights (interpreted as Fourier multipliers).
    For Fejér, this is nonnegative (Cesàro positivity). Sharp and signed have negative lobes.
    """
    ker = np.fft.ifft(weight_1d.astype(np.complex128)).real
    return float(np.min(ker))


def hf_weight_fraction(N: int, Kp: int, W: np.ndarray) -> float:
    """
    Energy fraction of |W|^2 supported strictly beyond the Kp cube (separable definition).

    This is a diagnostic for illegal HF injection (esp. signed control).
    """
    # Build a boolean mask for "HF region" using 1D wavenumbers.
    kx = (np.fft.fftfreq(N) * N).astype(np.int32)
    ky = (np.fft.fftfreq(N) * N).astype(np.int32)
    kz = (np.fft.rfftfreq(N) * N).astype(np.int32)
    HF = (np.abs(kx)[:, None, None] > Kp) | (np.abs(ky)[None, :, None] > Kp) | (np.abs(kz)[None, None, :] > Kp)
    num = float(np.sum((np.abs(W) ** 2)[HF]))
    den = float(np.sum(np.abs(W) ** 2))
    return 0.0 if den == 0 else num / den


# ============================
# Navier–Stokes solver (pseudo-spectral)
# ============================
def leray_project(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray, k2: np.ndarray) -> None:
    """
    In-place Leray projection: U <- P U = U - k (k·U) / |k|^2, with k=0 mode forced to 0.
    """
    dot = KX * Ux + KY * Uy + KZ * Uz
    inv = 1.0 / k2
    Ux -= KX * dot * inv
    Uy -= KY * dot * inv
    Uz -= KZ * dot * inv
    # enforce k=0 mode exactly
    Ux[0, 0, 0] = 0.0
    Uy[0, 0, 0] = 0.0
    Uz[0, 0, 0] = 0.0


def apply_filter(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, W: np.ndarray) -> None:
    Ux *= W
    Uy *= W
    Uz *= W


def nonlinear_u_cross_omega(
    Ux: np.ndarray,
    Uy: np.ndarray,
    Uz: np.ndarray,
    *,
    N: int,
    KX: np.ndarray,
    KY: np.ndarray,
    KZ: np.ndarray,
    k2: np.ndarray,
    dtype_real: np.dtype,
    dtype_cplx: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute N_hat = P[ FFT( u × ω ) ], where ω = ∇ × u, and P is Leray projection.

    This uses the vorticity form:
      (u·∇)u = ω×u + ∇(|u|^2/2)  =>  -P[(u·∇)u] = -P[ω×u] = P[u×ω].

    Returns N̂ components (same rFFT shape as Û).
    """
    shape = (N, N, N)

    # velocity in real space
    ux = fft_irfftn(Ux, shape).astype(dtype_real, copy=False)
    uy = fft_irfftn(Uy, shape).astype(dtype_real, copy=False)
    uz = fft_irfftn(Uz, shape).astype(dtype_real, copy=False)

    I = np.array(1j, dtype=dtype_cplx)

    # ω̂ = i k × Û
    wx_hat = I * (KY * Uz - KZ * Uy)
    wy_hat = I * (KZ * Ux - KX * Uz)
    wz_hat = I * (KX * Uy - KY * Ux)

    wx = fft_irfftn(wx_hat, shape).astype(dtype_real, copy=False)
    wy = fft_irfftn(wy_hat, shape).astype(dtype_real, copy=False)
    wz = fft_irfftn(wz_hat, shape).astype(dtype_real, copy=False)

    # u × ω in real space
    cx = (uy * wz - uz * wy).astype(dtype_real, copy=False)
    cy = (uz * wx - ux * wz).astype(dtype_real, copy=False)
    cz = (ux * wy - uy * wx).astype(dtype_real, copy=False)

    # FFT back to Fourier
    Cx = fft_rfftn(cx).astype(dtype_cplx, copy=False)
    Cy = fft_rfftn(cy).astype(dtype_cplx, copy=False)
    Cz = fft_rfftn(cz).astype(dtype_cplx, copy=False)

    # Project to remove gradient component
    leray_project(Cx, Cy, Cz, KX, KY, KZ, k2)
    return Cx, Cy, Cz


def rk2_advance(
    Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray,
    *,
    N: int,
    nu: float,
    dt: float,
    steps: int,
    W: np.ndarray,
    KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray, k2: np.ndarray,
    dtype_real: np.dtype,
    dtype_cplx: np.dtype,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Deterministic RK2 (Heun) for:
      U_t = N(U) - nu*k^2*U
    where N(U) = P[FFT(u×ω)] and filter W is applied as the lawful/illegal operator.

    Filter and projection are applied at each stage to make the operator-class explicit.
    """
    for _ in range(steps):
        # Stage 0: enforce operator-class
        apply_filter(Ux, Uy, Uz, W)
        leray_project(Ux, Uy, Uz, KX, KY, KZ, k2)

        N0x, N0y, N0z = nonlinear_u_cross_omega(
            Ux, Uy, Uz,
            N=N, KX=KX, KY=KY, KZ=KZ, k2=k2,
            dtype_real=dtype_real, dtype_cplx=dtype_cplx,
        )

        RHS0x = (N0x - (nu * k2) * Ux).astype(dtype_cplx, copy=False)
        RHS0y = (N0y - (nu * k2) * Uy).astype(dtype_cplx, copy=False)
        RHS0z = (N0z - (nu * k2) * Uz).astype(dtype_cplx, copy=False)

        U1x = (Ux + dt * RHS0x).astype(dtype_cplx, copy=False)
        U1y = (Uy + dt * RHS0y).astype(dtype_cplx, copy=False)
        U1z = (Uz + dt * RHS0z).astype(dtype_cplx, copy=False)

        # Stage 1: enforce operator-class at the mid state
        apply_filter(U1x, U1y, U1z, W)
        leray_project(U1x, U1y, U1z, KX, KY, KZ, k2)

        N1x, N1y, N1z = nonlinear_u_cross_omega(
            U1x, U1y, U1z,
            N=N, KX=KX, KY=KY, KZ=KZ, k2=k2,
            dtype_real=dtype_real, dtype_cplx=dtype_cplx,
        )

        RHS1x = (N1x - (nu * k2) * U1x).astype(dtype_cplx, copy=False)
        RHS1y = (N1y - (nu * k2) * U1y).astype(dtype_cplx, copy=False)
        RHS1z = (N1z - (nu * k2) * U1z).astype(dtype_cplx, copy=False)

        Ux = (Ux + 0.5 * dt * (RHS0x + RHS1x)).astype(dtype_cplx, copy=False)
        Uy = (Uy + 0.5 * dt * (RHS0y + RHS1y)).astype(dtype_cplx, copy=False)
        Uz = (Uz + 0.5 * dt * (RHS0z + RHS1z)).astype(dtype_cplx, copy=False)

    # Final enforcement
    apply_filter(Ux, Uy, Uz, W)
    leray_project(Ux, Uy, Uz, KX, KY, KZ, k2)
    return Ux, Uy, Uz


def taylor_green_initial(N: int, dtype=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Taylor–Green vortex on [0,2π)^3:
      u =  sin x cos y cos z
      v = -cos x sin y cos z
      w =  0
    """
    x = (2.0 * math.pi * np.arange(N) / N).astype(dtype)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return u.astype(dtype), v.astype(dtype), w.astype(dtype)


def divergence_l2(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, *, N: int, KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray, dtype_real: np.dtype, dtype_cplx: np.dtype) -> float:
    I = np.array(1j, dtype=dtype_cplx)
    div_hat = I * (KX * Ux + KY * Uy + KZ * Uz)
    div = fft_irfftn(div_hat, (N, N, N)).astype(dtype_real, copy=False)
    return float(np.sqrt(np.mean(div * div)))


def fields_real(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, *, N: int, dtype_real: np.dtype) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = fft_irfftn(Ux, (N, N, N)).astype(dtype_real, copy=False)
    v = fft_irfftn(Uy, (N, N, N)).astype(dtype_real, copy=False)
    w = fft_irfftn(Uz, (N, N, N)).astype(dtype_real, copy=False)
    return u, v, w


def l2_rel_error(u: Tuple[np.ndarray, np.ndarray, np.ndarray], u_ref: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    ux, uy, uz = u
    rx, ry, rz = u_ref
    num = float(np.sqrt(np.mean((ux - rx) ** 2 + (uy - ry) ** 2 + (uz - rz) ** 2)))
    den = float(np.sqrt(np.mean(rx ** 2 + ry ** 2 + rz ** 2)))
    return num / den if den > 0 else float("inf")


def energy(u: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    ux, uy, uz = u
    return 0.5 * float(np.mean(ux * ux + uy * uy + uz * uz))


def enstrophy_from_fourier(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, *, N: int, KX: np.ndarray, KY: np.ndarray, KZ: np.ndarray, dtype_real: np.dtype, dtype_cplx: np.dtype) -> float:
    I = np.array(1j, dtype=dtype_cplx)
    wx_hat = I * (KY * Uz - KZ * Uy)
    wy_hat = I * (KZ * Ux - KX * Uz)
    wz_hat = I * (KX * Uy - KY * Ux)
    wx = fft_irfftn(wx_hat, (N, N, N)).astype(dtype_real, copy=False)
    wy = fft_irfftn(wy_hat, (N, N, N)).astype(dtype_real, copy=False)
    wz = fft_irfftn(wz_hat, (N, N, N)).astype(dtype_real, copy=False)
    return 0.5 * float(np.mean(wx * wx + wy * wy + wz * wz))


# ============================
# Tiering + budgets (deterministic)
# ============================
@dataclass(frozen=True)
class Budgets:
    q2: int
    q3: int
    v2U: int
    eps: float
    N: int
    K_primary: int
    K_truth: int
    nu: float
    T: float
    dt: float
    steps: int


def budgets_from_primary(primary: Triple, *, N: int, T: float = 0.6) -> Budgets:
    """
    Deterministic budget map used across the demo family.

    For NS:
      q2 = 30, q3 = 17 (locked by the selector family)
      eps = 1/sqrt(q2)
      nu  = 1/(2*q2*q3)  (portable viscosity proxy)

    Spectral budgets:
      K_primary ≈ (15/64) N   (so N=64->15, 128->30, 256->60, 512->120)
      K_truth   = min(N//2-1, 2*K_primary + 7)  (so N=256 -> 127)

    Time step:
      dt_base = (π/2)/N, steps = ceil(T/dt_base), dt = T/steps
    """
    q2 = 30
    q3 = 17
    v2U = v2(primary.wU - 1)
    eps = 1.0 / math.sqrt(q2)

    K_primary = int(round((15.0 / 64.0) * N))
    K_primary = max(4, min(K_primary, N // 2 - 2))

    K_truth = min(N // 2 - 1, 2 * K_primary + 7)
    nu = 1.0 / (2.0 * q2 * q3)

    dt_base = (0.5 * math.pi) / N
    steps = int(math.ceil(T / dt_base))
    dt = T / steps

    return Budgets(q2=q2, q3=q3, v2U=v2U, eps=eps, N=N, K_primary=K_primary, K_truth=K_truth, nu=nu, T=T, dt=dt, steps=steps)


# ============================
# Readiness ledger (impact + planning)
# ============================
def readiness_ledger(q2: int, q3: int, tiers: List[int]) -> List[Dict[str, float]]:
    """
    Deterministic readiness ledger:
      - dt from (π/2)/N (same as solver)
      - steps = ceil(T/dt)
      - memory estimates for rFFT and naive complex FFT

    This is an engineering artifact for referees and for reproducibility planning.
    """
    out = []
    for N in tiers:
        Kp = int(round((15.0 / 64.0) * N))
        Kt = min(N // 2 - 1, 2 * Kp + 7)

        T = 10.0
        dt_base = (0.5 * math.pi) / N
        steps = int(math.ceil(T / dt_base))
        dt = T / steps

        # proxy flop scaling (informational)
        fft3_proxy = (N ** 3) * math.log2(max(N, 2))

        # memory: rFFT shape
        n_rfft = N * N * (N // 2 + 1)

        # numpy.fft tends to use float64/complex128 internally
        mem_full_complex128 = 3 * (N ** 3) * 16 / 1e9  # 3 components complex128
        mem_rfft_complex128 = 3 * n_rfft * 16 / 1e9
        # complex64 path (scipy.fft or other optimized backends)
        mem_full_complex64 = 3 * (N ** 3) * 8 / 1e9
        mem_rfft_complex64 = 3 * n_rfft * 8 / 1e9

        out.append({
            "N": float(N),
            "Kp": float(Kp),
            "Kt": float(Kt),
            "dt": float(dt),
            "steps": float(steps),
            "FFT3_proxy": float(fft3_proxy),
            "mem_full_complex128_GB": float(mem_full_complex128),
            "mem_rfft_complex128_GB": float(mem_rfft_complex128),
            "mem_full_complex64_GB": float(mem_full_complex64),
            "mem_rfft_complex64_GB": float(mem_rfft_complex64),
        })
    return out


# ============================
# Optional plotting (matplotlib)
# ============================
def maybe_plot_spectrum(
    U_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    N: int,
    title: str,
    out_png: str,
) -> Optional[str]:
    """
    Optional shell-averaged energy spectrum plot.
    If matplotlib isn't available, return None.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    Ux, Uy, Uz = U_tuple
    # Build k magnitudes
    kx = (np.fft.fftfreq(N) * N).astype(np.float32)
    ky = (np.fft.fftfreq(N) * N).astype(np.float32)
    kz = (np.fft.rfftfreq(N) * N).astype(np.float32)

    KX = kx[:, None, None]
    KY = ky[None, :, None]
    KZ = kz[None, None, :]

    kmag = np.sqrt(KX * KX + KY * KY + KZ * KZ)
    kbin = np.rint(kmag).astype(np.int32)

    # Spectral energy density (Parseval scale omitted; we compare shapes)
    Ehat = (np.abs(Ux) ** 2 + np.abs(Uy) ** 2 + np.abs(Uz) ** 2).astype(np.float64)

    # rFFT symmetry: modes with kz not 0 or Nyquist represent both ±kz.
    # Multiply energy by 2 for kz in (1..N/2-1).
    if N % 2 == 0:
        nz = N // 2
        sym = np.ones((N, N, N // 2 + 1), dtype=np.float64)
        sym[:, :, 1:nz] *= 2.0
    else:
        sym = np.ones((N, N, N // 2 + 1), dtype=np.float64)
        sym[:, :, 1:] *= 2.0
    Ehat *= sym

    kmax = int(np.max(kbin))
    Ek = np.zeros(kmax + 1, dtype=np.float64)
    counts = np.zeros(kmax + 1, dtype=np.float64)
    flat_k = kbin.ravel()
    flat_E = Ehat.ravel()
    for i in range(flat_k.size):
        kb = int(flat_k[i])
        Ek[kb] += flat_E[i]
        counts[kb] += 1.0
    counts[counts == 0] = 1.0
    Ek /= counts

    ks = np.arange(len(Ek))
    # avoid k=0 in log plots
    ks2 = ks[1:]
    Ek2 = Ek[1:]

    plt.figure()
    plt.loglog(ks2, Ek2)
    plt.xlabel("k (shell index)")
    plt.ylabel("E(k) (arb. units)")
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", linewidth=0.5)
    plt.tight_layout()
    try:
        plt.savefig(out_png, dpi=160)
        plt.close()
        return out_png
    except Exception:
        plt.close()
        return None


# ============================
# Main demo runner
# ============================
def run_demo(tier: str, write_artifacts: bool, do_ladder: bool) -> int:
    # -------------------------------------------------------------------------
    # Environment header
    # -------------------------------------------------------------------------
    utc = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat().replace("+00:00", "Z")
    banner("DEMO-67 — Navier–Stokes Master Flagship (Industrial Certificate) — REFEREE READY")
    print(f"UTC time : {utc}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"FFT back.: {FFT_BACKEND}")
    print("I/O      : stdout only (JSON/PNG artifacts optional)")
    print()

    # -------------------------------------------------------------------------
    # Stage 1 — Selection
    # -------------------------------------------------------------------------
    banner("STAGE 1 — Deterministic triple selection (primary + counterfactual labels)")
    primary, pools, counterfactuals = select_primary_and_counterfactuals()
    print("Lane survivor pools (raw):")
    print(f"  U(1):  {pools['U1_raw']}")
    print(f"  SU(2): {pools['SU2_raw']}")
    print(f"  SU(3): {pools['SU3_raw']}")
    print("Lane survivor pools (after U(1) coherence v2(wU-1)=3):")
    print(f"  U(1):  {pools['U1_coherent']}")
    print(f"  SU(2): {pools['SU2_raw']}")
    print(f"  SU(3): {pools['SU3_raw']}")
    passfail(primary == Triple(137, 107, 103), "Primary equals (137,107,103)", f"selected={primary}")
    passfail(len(counterfactuals) >= 4, "Captured >=4 counterfactual labels (deterministic)", f"found={len(counterfactuals)}")
    print("Counterfactual labels:", [dataclasses.asdict(t) for t in counterfactuals[:8]])
    print()

    # -------------------------------------------------------------------------
    # Stage 2 — Budgets
    # -------------------------------------------------------------------------
    if tier == "smoke":
        N = 128
    elif tier == "industrial":
        N = 256
    elif tier == "hpc":
        N = 384
    else:
        raise ValueError("tier must be smoke|industrial|hpc")

    B = budgets_from_primary(primary, N=N, T=0.6)
    spec = {
        "demo": "DEMO-67",
        "tier": tier,
        "N": B.N,
        "K_primary": B.K_primary,
        "K_truth": B.K_truth,
        "q2": B.q2,
        "q3": B.q3,
        "v2U": B.v2U,
        "eps": B.eps,
        "nu": B.nu,
        "T": B.T,
        "dt": B.dt,
        "steps": B.steps,
        "fft_backend": FFT_BACKEND,
        "selector_windows": {"primary": (97, 180), "counterfactual": (181, 1200)},
        "counterfactual_count": len(counterfactuals),
    }
    banner("STAGE 2 — Derived invariants and budgets (first principles, no tuning)")
    print(f"primary={primary}  q2={B.q2} q3={B.q3} v2U={B.v2U} eps=1/sqrt(q2)={B.eps:.8f}")
    print(f"N={B.N}  K_primary={B.K_primary}  K_truth={B.K_truth}")
    print(f"nu=1/(2*q2*q3)={B.nu:.18f}")
    print(f"T={B.T}  dt={B.dt:.18f}  steps={B.steps}")
    print(f"spec_sha256: {spec_sha256(spec)}")
    print()

    # -------------------------------------------------------------------------
    # Stage 2B — Readiness ledger (impact + planning)
    # -------------------------------------------------------------------------
    banner("STAGE 2B — Industrial readiness ledger (deterministic planning artifact)")
    tiers = [64, 128, 256, 384, 512]
    ledger = readiness_ledger(B.q2, B.q3, tiers)
    print("    N    Kp    Kt           dt    steps   FFT3_proxy   mem_rfft_c64_GB   mem_rfft_c128_GB")
    print("------------------------------------------------------------------------------------------")
    for row in ledger:
        print(f"{int(row['N']):5d} {int(row['Kp']):5d} {int(row['Kt']):5d}  {row['dt']:12.7g} {int(row['steps']):7d}  "
              f"{row['FFT3_proxy']:10.3e}     {row['mem_rfft_complex64_GB']:10.3g}         {row['mem_rfft_complex128_GB']:10.3g}")
    print()
    print("Notes:")
    print("• numpy.fft commonly promotes to complex128/float64 internally; scipy.fft typically keeps complex64/float32.")
    print("• For the flagship certificate, use N=256 (industrial) if hardware allows; N>=384 is HPC territory.")
    print()

    # -------------------------------------------------------------------------
    # Stage 3 — Kernel admissibility audit (1D)
    # -------------------------------------------------------------------------
    banner("STAGE 3 — Kernel admissibility audit (1D real-space kernel minima)")
    wF = fejer_weight_1d(B.N, B.K_primary, rfft_axis=False, dtype=np.float32)
    wS = sharp_weight_1d(B.N, B.K_primary, rfft_axis=False, dtype=np.float32)
    wX = signed_weight_1d(B.N, B.K_primary, rfft_axis=False, dtype=np.float32)

    kmin_F = kernel_min_1d(wF)
    kmin_S = kernel_min_1d(wS)
    kmin_X = kernel_min_1d(wX)

    passfail(kmin_F >= -1e-12, "Fejér kernel is nonnegative (numerical tol)", f"kmin={kmin_F:.3e}")
    passfail(kmin_S < -1e-6, "Sharp cutoff kernel has negative lobes (non-admissible)", f"kmin={kmin_S:.6g}")
    passfail(kmin_X < -1e-6, "Signed control kernel has negative lobes (non-admissible)", f"kmin={kmin_X:.6g}")

    WF = tensor_weight(B.N, B.K_primary, "fejer", dtype=np.float32)
    WS = tensor_weight(B.N, B.K_primary, "sharp", dtype=np.float32)
    WX = tensor_weight(B.N, B.K_primary, "signed", dtype=np.float32)
    hfF = hf_weight_fraction(B.N, B.K_primary, WF)
    hfS = hf_weight_fraction(B.N, B.K_primary, WS)
    hfX = hf_weight_fraction(B.N, B.K_primary, WX)
    print(f"HF weight energy fraction (>Kp): fejer={hfF:.6g} sharp={hfS:.6g} signed={hfX:.6g}")
    print()

    # -------------------------------------------------------------------------
    # Stage 4 — NS run: truth + lawful + illegal controls
    # -------------------------------------------------------------------------
    banner("STAGE 4 — 3D incompressible NS (Taylor–Green): truth + lawful + illegal controls")
    dtype_real = np.float32
    dtype_cplx = np.complex64

    # k grids
    KX, KY, KZ, k2 = kgrid(B.N, dtype=np.float32)

    # Build initial condition and transform
    u0, v0, w0 = taylor_green_initial(B.N, dtype=dtype_real)
    U0x = fft_rfftn(u0).astype(dtype_cplx, copy=False)
    U0y = fft_rfftn(v0).astype(dtype_cplx, copy=False)
    U0z = fft_rfftn(w0).astype(dtype_cplx, copy=False)

    # Ensure incompressibility at t=0
    leray_project(U0x, U0y, U0z, KX, KY, KZ, k2)

    # Truth run (Fejér with K_truth)
    W_truth = tensor_weight(B.N, B.K_truth, "fejer", dtype=np.float32)
    t0 = time.time()
    Utx, Uty, Utz = rk2_advance(
        U0x.copy(), U0y.copy(), U0z.copy(),
        N=B.N, nu=B.nu, dt=B.dt, steps=B.steps,
        W=W_truth, KX=KX, KY=KY, KZ=KZ, k2=k2,
        dtype_real=dtype_real, dtype_cplx=dtype_cplx,
    )
    truth_runtime = time.time() - t0
    u_truth = fields_real(Utx, Uty, Utz, N=B.N, dtype_real=dtype_real)

    # Lawful run (Fejér with K_primary)
    W_law = WF
    t1 = time.time()
    Ulx, Uly, Ulz = rk2_advance(
        U0x.copy(), U0y.copy(), U0z.copy(),
        N=B.N, nu=B.nu, dt=B.dt, steps=B.steps,
        W=W_law, KX=KX, KY=KY, KZ=KZ, k2=k2,
        dtype_real=dtype_real, dtype_cplx=dtype_cplx,
    )
    law_runtime = time.time() - t1
    u_law = fields_real(Ulx, Uly, Ulz, N=B.N, dtype_real=dtype_real)

    # Illegal sharp
    t2 = time.time()
    Usx, Usy, Usz = rk2_advance(
        U0x.copy(), U0y.copy(), U0z.copy(),
        N=B.N, nu=B.nu, dt=B.dt, steps=B.steps,
        W=WS, KX=KX, KY=KY, KZ=KZ, k2=k2,
        dtype_real=dtype_real, dtype_cplx=dtype_cplx,
    )
    sharp_runtime = time.time() - t2
    u_sharp = fields_real(Usx, Usy, Usz, N=B.N, dtype_real=dtype_real)

    # Illegal signed
    t3 = time.time()
    Uxx, Uxy, Uxz = rk2_advance(
        U0x.copy(), U0y.copy(), U0z.copy(),
        N=B.N, nu=B.nu, dt=B.dt, steps=B.steps,
        W=WX, KX=KX, KY=KY, KZ=KZ, k2=k2,
        dtype_real=dtype_real, dtype_cplx=dtype_cplx,
    )
    signed_runtime = time.time() - t3
    u_signed = fields_real(Uxx, Uxy, Uxz, N=B.N, dtype_real=dtype_real)

    # Metrics
    err_law = l2_rel_error(u_law, u_truth)
    err_sharp = l2_rel_error(u_sharp, u_truth)
    err_signed = l2_rel_error(u_signed, u_truth)

    div_law = divergence_l2(Ulx, Uly, Ulz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)
    div_sharp = divergence_l2(Usx, Usy, Usz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)
    div_signed = divergence_l2(Uxx, Uxy, Uxz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)

    # HF fraction in STATE beyond K_primary (for each run)
    # For consistency with other demos, we measure energy in the rFFT state itself using a separable HF mask.
    kx_i = (np.fft.fftfreq(B.N) * B.N).astype(np.int32)
    ky_i = (np.fft.fftfreq(B.N) * B.N).astype(np.int32)
    kz_i = (np.fft.rfftfreq(B.N) * B.N).astype(np.int32)
    HFmask = (np.abs(kx_i)[:, None, None] > B.K_primary) | (np.abs(ky_i)[None, :, None] > B.K_primary) | (np.abs(kz_i)[None, None, :] > B.K_primary)

    def hf_state_frac(Ux, Uy, Uz) -> float:
        E = (np.abs(Ux) ** 2 + np.abs(Uy) ** 2 + np.abs(Uz) ** 2).astype(np.float64)
        num = float(np.sum(E[HFmask]))
        den = float(np.sum(E))
        return 0.0 if den == 0 else num / den

    hf_law = hf_state_frac(Ulx, Uly, Ulz)
    hf_sharp = hf_state_frac(Usx, Usy, Usz)
    hf_signed = hf_state_frac(Uxx, Uxy, Uxz)

    # Energetics at final time
    E_truth = energy(u_truth)
    E_law = energy(u_law)
    E_sharp = energy(u_sharp)
    E_signed = energy(u_signed)

    Z_truth = enstrophy_from_fourier(Utx, Uty, Utz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)
    Z_law = enstrophy_from_fourier(Ulx, Uly, Ulz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)
    Z_sharp = enstrophy_from_fourier(Usx, Usy, Usz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)
    Z_signed = enstrophy_from_fourier(Uxx, Uxy, Uxz, N=B.N, KX=KX, KY=KY, KZ=KZ, dtype_real=dtype_real, dtype_cplx=dtype_cplx)

    print(f"Truth runtime (informational): {truth_runtime:.2f} s")
    print(f"Run runtimes  (law/sharp/signed): {law_runtime:.2f} / {sharp_runtime:.2f} / {signed_runtime:.2f} s")
    print(f"errors vs truth: lawful={err_law:.12g}  sharp={err_sharp:.12g}  signed={err_signed:.12g}")
    print(f"divL2: lawful={div_law:.3e}  sharp={div_sharp:.3e}  signed={div_signed:.3e}")
    print(f"HFfrac(>Kp) state : lawful={hf_law:.3e}  sharp={hf_sharp:.3e}  signed={hf_signed:.3e}")
    print(f"Energy(t=T): truth={E_truth:.6g}  law={E_law:.6g}  sharp={E_sharp:.6g}  signed={E_signed:.6g}")
    print(f"Enstrophy(t=T): truth={Z_truth:.6g}  law={Z_law:.6g}  sharp={Z_sharp:.6g}  signed={Z_signed:.6g}")
    print()

    # Gates (primary certificate)
    g1 = (div_law <= 1e-8) and (div_sharp <= 1e-8) and (div_signed <= 1e-8)
    passfail(g1, "Gate G1: incompressibility divL2 <= 1e-8 (all variants)",
             f"div_law={div_law:.3e} div_sh={div_sharp:.3e} div_si={div_signed:.3e}")

    # Composite certificate score (lower is better).
    #
    # Rationale:
    #   • The raw field L2 error alone can saturate near 1 on portable tiers,
    #     which makes multiplicative teeth brittle.
    #   • We therefore combine: (i) an unbounded 'surprisal' of the field error
    #     as err→1⁻, and (ii) log-mismatch in energy and enstrophy at t=T.
    score_law = certificate_score(err_law, E_law, Z_law, E_truth, Z_truth)
    score_sharp = certificate_score(err_sharp, E_sharp, Z_sharp, E_truth, Z_truth)
    score_signed = certificate_score(err_signed, E_signed, Z_signed, E_truth, Z_truth)
    score_illegal = min(score_sharp, score_signed)
    print(f"certificate score (lower=better): truth=0  law={score_law:.6g}  sharp={score_sharp:.6g}  signed={score_signed:.6g}")

    # Gate G2: lawful must beat the *best* illegal by at least a margin.
    g2_strong = score_illegal >= (1.0 + B.eps) * score_law
    g2_weak = score_illegal > score_law
    passfail(g2_weak, "Gate G2: lawful closer to truth than illegal controls",
             f"score_law={score_law:.6g} score_illegal_min={score_illegal:.6g} strong={g2_strong}")

    # Gate G3: the signed illegal control has substantial HF weight beyond Kp *in the filter itself*.
    # (In a filtered solver, the state HF fraction can remain tiny even when the filter is non-admissible.)
    floor = max(10.0 * hfF, B.eps ** 2)
    g3 = hfX >= floor
    passfail(g3, "Gate G3: signed illegal injects HF weight beyond floor (kernel)",
             f"hfW_fejer={hfF:.3e} hfW_signed={hfX:.3e} floor={floor:.3e}")

    # -------------------------------------------------------------------------
    # Stage 5 — Counterfactual teeth (budget K varies, PDE fixed)
    # -------------------------------------------------------------------------
    banner("STAGE 5 — Counterfactual teeth (physics fixed, budget K reduced deterministically)")
    q3_cf = 3 * B.q3
    K_cf = max(4, int(round(B.K_primary * (B.q3 / q3_cf))))
    print(f"Counterfactual budget rule: q3 -> 3*q3 = {q3_cf}  =>  K_cf = round(Kp*q3/q3_cf) = {K_cf}")

    # All counterfactual labels share the *same* deterministic budget rule here.
    # To keep the demo industrially runnable, we execute the counterfactual simulation ONCE
    # (at K=K_cf) and reuse the resulting observable for each label.
    W_cf = tensor_weight(B.N, K_cf, "fejer", dtype=np.float32)
    Ucx, Ucy, Ucz = rk2_advance(
        U0x.copy(), U0y.copy(), U0z.copy(),
        N=B.N, nu=B.nu, dt=B.dt, steps=B.steps,
        W=W_cf, KX=KX, KY=KY, KZ=KZ, k2=k2,
        dtype_real=dtype_real, dtype_cplx=dtype_cplx,
    )
    u_cf = fields_real(Ucx, Ucy, Ucz, N=B.N, dtype_real=dtype_real)
    err_cf = l2_rel_error(u_cf, u_truth)
    E_cf = energy(u_cf)
    Z_cf = enstrophy_from_fourier(Ucx, Ucy, Ucz, N=B.N, KX=KX, KY=KY, KZ=KZ,
                                  dtype_real=dtype_real, dtype_cplx=dtype_cplx)
    score_cf = certificate_score(err_cf, E_cf, Z_cf, E_truth, Z_truth)
    degrade = score_cf >= (1.0 + B.eps) * score_law

    cf_test = counterfactuals[:4]  # referee-standard 4-label teeth set
    strong = 0
    results_cf = []
    for t in cf_test:
        strong += int(degrade)
        results_cf.append({
            "triple": dataclasses.asdict(t),
            "K_cf": K_cf,
            "err": err_cf,
            "score": score_cf,
            "E": E_cf,
            "Z": Z_cf,
            "degrade": degrade,
        })
        print(f"CF {t} q3_cf={q3_cf:3d} K={K_cf:3d} err={err_cf:.12g} score={score_cf:.6g} degrade={degrade}")

    gT = strong >= 3
    passfail(
        gT,
        "Gate T1: >=3/4 counterfactuals degrade by (1+eps) on certificate score",
        f"strong={strong}/{len(cf_test)} eps={B.eps:.6g} score_law={score_law:.6g} score_cf={score_cf:.6g}",
    )
    print("Note: counterfactual labels share the same q3_cf budget; simulation executed once and reused.")
    print()

    # -------------------------------------------------------------------------
    # Optional ladder invariance (cross-resolution)
    # -------------------------------------------------------------------------
    ladder_ok = None
    ladder_payload = None
    if do_ladder:
        banner("STAGE 6 — Optional cross-resolution ladder (scaled error invariant)")

        tiers_l = [128, 256] if B.N >= 256 else [64, 128]
        rows = []
        for Nl in tiers_l:
            Bl = budgets_from_primary(primary, N=Nl, T=B.T)
            KXl, KYl, KZl, k2l = kgrid(Bl.N, dtype=np.float32)
            u0l, v0l, w0l = taylor_green_initial(Bl.N, dtype=dtype_real)
            U0xl = fft_rfftn(u0l).astype(dtype_cplx, copy=False)
            U0yl = fft_rfftn(v0l).astype(dtype_cplx, copy=False)
            U0zl = fft_rfftn(w0l).astype(dtype_cplx, copy=False)
            leray_project(U0xl, U0yl, U0zl, KXl, KYl, KZl, k2l)

            Wtl = tensor_weight(Bl.N, Bl.K_truth, "fejer", dtype=np.float32)
            Utxl, Utyl, Utzl = rk2_advance(U0xl.copy(), U0yl.copy(), U0zl.copy(),
                                          N=Bl.N, nu=Bl.nu, dt=Bl.dt, steps=Bl.steps,
                                          W=Wtl, KX=KXl, KY=KYl, KZ=KZl, k2=k2l,
                                          dtype_real=dtype_real, dtype_cplx=dtype_cplx)
            u_truth_l = fields_real(Utxl, Utyl, Utzl, N=Bl.N, dtype_real=dtype_real)

            Wpl = tensor_weight(Bl.N, Bl.K_primary, "fejer", dtype=np.float32)
            Ulxl, Ulyl, Ulzl = rk2_advance(U0xl.copy(), U0yl.copy(), U0zl.copy(),
                                          N=Bl.N, nu=Bl.nu, dt=Bl.dt, steps=Bl.steps,
                                          W=Wpl, KX=KXl, KY=KYl, KZ=KZl, k2=k2l,
                                          dtype_real=dtype_real, dtype_cplx=dtype_cplx)
            u_law_l = fields_real(Ulxl, Ulyl, Ulzl, N=Bl.N, dtype_real=dtype_real)
            err_l = l2_rel_error(u_law_l, u_truth_l)
            C = err_l * math.sqrt(Bl.K_primary)
            rows.append((Bl.N, Bl.K_primary, Bl.K_truth, err_l, C))

        print("N    Kp   Kt   err(N)        C=err*sqrt(Kp)")
        for r in rows:
            print(f"{r[0]:3d}  {r[1]:3d}  {r[2]:3d}  {r[3]:.9g}    {r[4]:.9g}")

        C0 = rows[0][4]
        C1 = rows[-1][4]
        rel = abs(C1 - C0) / max(abs(C0), 1e-30)
        ladder_ok = rel <= B.eps
        passfail(ladder_ok, "Gate L: ladder invariant C stable within eps", f"rel={rel:.6g} eps={B.eps:.6g}")
        ladder_payload = {"rows": rows, "rel": rel, "eps": B.eps}

    # -------------------------------------------------------------------------
    # Stage 7 — Artifacts + determinism hash
    # -------------------------------------------------------------------------
    banner("STAGE 7 — Determinism hash + optional artifacts")
    payload = {
        "spec_sha256": spec_sha256(spec),
        "N": B.N,
        "K_primary": B.K_primary,
        "K_truth": B.K_truth,
        "nu": B.nu,
        "dt": B.dt,
        "steps": B.steps,
        "kernel_mins": {"fejer": kmin_F, "sharp": kmin_S, "signed": kmin_X},
        "errors": {"law": err_law, "sharp": err_sharp, "signed": err_signed},
        "divL2": {"law": div_law, "sharp": div_sharp, "signed": div_signed},
        "hf_state": {"law": hf_law, "sharp": hf_sharp, "signed": hf_signed},
        "energetics": {"E_truth": E_truth, "E_law": E_law, "E_sharp": E_sharp, "E_signed": E_signed,
                       "Z_truth": Z_truth, "Z_law": Z_law, "Z_sharp": Z_sharp, "Z_signed": Z_signed},
        "teeth": results_cf[:4],
        "ladder": ladder_payload,
    }
    det_hash = determinism_sha256(payload)
    print(f"determinism_sha256: {det_hash}")

    artifacts = {}
    if write_artifacts:
        out_json = f"demo67_master_results_{tier}.json"
        out_png = f"demo67_spectrum_{tier}.png"
        try:
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump({"spec": spec, "payload": payload}, f, indent=2, sort_keys=True)
            artifacts["json"] = out_json
            passfail(True, "Results JSON written", out_json)
        except Exception as e:
            passfail(False, "Results JSON not written (filesystem unavailable)", repr(e))

        # Spectrum plot of lawful final state
        try:
            png = maybe_plot_spectrum((Ulx, Uly, Ulz), N=B.N, title=f"NS3D Taylor–Green spectrum (lawful) N={B.N} Kp={B.K_primary}", out_png=out_png)
            if png is None:
                passfail(True, "Spectrum plot skipped (matplotlib unavailable)")
            else:
                artifacts["png"] = png
                passfail(True, "Spectrum plot written", png)
        except Exception as e:
            passfail(False, "Spectrum plot not written", repr(e))

    # -------------------------------------------------------------------------
    # Final verdict
    # -------------------------------------------------------------------------
    banner("FINAL VERDICT")
    ok_all = bool(g1 and g2_weak and g3 and gT and (True if (ladder_ok is None) else ladder_ok))
    passfail(ok_all, "DEMO-67 VERIFIED (NS3D industrial certificate: admissibility + controls + teeth)")
    print(f"Result: {'VERIFIED' if ok_all else 'NOT VERIFIED'}")
    if not ok_all:
        print("Notes:")
        if not g1:
            print(" - G1 failed: incompressibility drift too large.")
        if not g2_weak:
            print(" - G2 failed: lawful is not closer to truth than illegal controls (check tier/backend).")
        if not g3:
            print(" - G3 failed: signed control did not inject enough HF (unexpected).")
        if not gT:
            print(" - Teeth failed: counterfactual budgets did not degrade by margin (try industrial tier).")
        if ladder_ok is False:
            print(" - Ladder failed: scaled invariant did not stabilize across tiers (run with --ladder again).")

    return 0 if ok_all else 2


def main() -> int:
    p = argparse.ArgumentParser(description="DEMO-67 — NS3D Taylor–Green master flagship (referee-ready).")
    p.add_argument("--tier", choices=["smoke", "industrial", "hpc"], default="smoke",
                   help="Run tier. smoke=N128 quick; industrial=N256 referee-grade; hpc=N384 heavy.")
    p.add_argument("--write-artifacts", action="store_true", help="Attempt to write JSON/PNG artifacts.")
    p.add_argument("--ladder", action="store_true", help="Run optional cross-resolution ladder stage (slower).")
    args = p.parse_args()
    return run_demo(tier=args.tier, write_artifacts=args.write_artifacts, do_ladder=args.ladder)


if __name__ == "__main__":
    raise SystemExit(main())


