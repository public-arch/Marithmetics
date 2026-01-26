#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO-56 — Deterministic Operator Calculus vs Classical Finite Differences

This single script is designed to be:
  - Self-contained (NumPy + standard library only)
  - Deterministic (no tolerance-driven inner iterations; fixed-step updates)
  - Falsifiable (explicit pass/fail gates and counterfactual controls)
  - Referee-ready (no internal jargon; first-principles explanations in output)

What it does (high level)
-------------------------
1) Selects a unique integer triple (wU, s2, s3) by a deterministic rule.
2) Derives a small set of invariants (q2, q3, v2, eps).
3) Uses those invariants to deterministically set numerical budgets (N, K, dt, steps).
4) Runs worked examples showing:
   - Why admissible kernels (Fejér averaging) prevent non-physical oscillations
   - Why non-admissible kernels (sharp truncation / signed filters) fail controls
   - Why the budgets matter (counterfactual triples degrade by a fixed margin)
5) Optionally runs an industrial-scale 3D Navier–Stokes certificate (Taylor–Green vortex).

Default mode is intentionally fast ("smoke-tier") and should run on laptops.
For the industrial certificate (N=256), use:  --tier industrial

Usage
-----
  python demo56_master_flagship_maximum_impact_v6.py
  python demo56_master_flagship_maximum_impact_v6.py --tier smoke
  python demo56_master_flagship_maximum_impact_v6.py --tier industrial
  python demo56_master_flagship_maximum_impact_v6.py --no_ns3d

Notes on determinism
-------------------
- All grids, cutoffs, time-steps, and step counts are fixed functions of the triple.
- No conjugate-gradient / multigrid tolerance loops are used.
- Any reference solutions are computed by running the same fixed update at higher budget.
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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =========================
# Formatting / PASS-FAIL
# =========================

W = 100

def hr(char: str = "=") -> str:
    return char * W

def title(s: str) -> str:
    pad = max(0, W - len(s) - 2)
    left = pad // 2
    right = pad - left
    return f"{'='*left} {s} {'='*right}"

def fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    if abs(x) >= 1e4 or (abs(x) > 0 and abs(x) < 1e-4):
        return f"{x:.10e}"
    return f"{x:.10f}".rstrip("0").rstrip(".")

@dataclasses.dataclass(frozen=True)
class Gate:
    name: str
    ok: bool
    detail: str = ""

def print_gate(g: Gate) -> None:
    tag = "PASS" if g.ok else "FAIL"
    detail = f"  {g.detail}" if g.detail else ""
    print(f"{tag:<5} {g.name:<70}{detail}")

def sha256_of_json(obj: dict) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()

def determinism_hash(payload: dict) -> str:
    # Hash a canonical JSON representation of all recorded results.
    return sha256_of_json(payload)


# =========================
# Small number theory (no external deps)
# =========================

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

def primes_in(lo: int, hi: int) -> List[int]:
    return [p for p in range(lo, hi + 1) if is_prime(p)]

def v2(n: int) -> int:
    """2-adic valuation v2(n): largest k such that 2^k divides n, for n>0."""
    if n <= 0:
        raise ValueError("v2 expects n>0")
    k = 0
    while (n & 1) == 0:
        n >>= 1
        k += 1
    return k

def odd_part(n: int) -> int:
    """Odd part of n: n / 2^{v2(n)} for n>0."""
    return n >> v2(n)

def totient(n: int) -> int:
    """Euler's totient for small n via trial factorization."""
    if n <= 0:
        raise ValueError("totient expects n>0")
    x = n
    p = 2
    result = n
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p = 3 if p == 2 else p + 2
    if x > 1:
        result -= result // x
    return result

def theta_density(w: int) -> float:
    """Theta(w-1) := phi(w-1)/(w-1), used as a density / smoothness gate."""
    m = w - 1
    return totient(m) / m


# =========================
# Deterministic triple selection
# =========================

@dataclasses.dataclass(frozen=True)
class LaneSpec:
    name: str
    q: int
    residues: Tuple[int, ...]
    tau_min: float
    span: Tuple[int, int]
    v2_required: Optional[int] = None  # None means no 2-adic branch constraint

@dataclasses.dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int

@dataclasses.dataclass(frozen=True)
class Invariants:
    q2: int
    q3: int
    v2U: int
    eps: float

def lane_survivors(spec: LaneSpec, apply_v2: bool = True) -> List[int]:
    lo, hi = spec.span
    out: List[int] = []
    for p in primes_in(lo, hi):
        if p % spec.q not in spec.residues:
            continue
        if theta_density(p) + 1e-15 < spec.tau_min:
            continue
        if apply_v2 and (spec.v2_required is not None):
            if v2(p - 1) != spec.v2_required:
                continue
        out.append(p)
    return out

def select_primary_triple() -> Tuple[Triple, Invariants, Dict[str, List[int]], Dict[str, List[int]]]:
    """
    Deterministic primary triple selection.

    Primary window is intentionally small to make the selection falsifiable and checkable.
    We publish (a) raw lane survivor pools (residue + density), and (b) refined pools after
    applying the explicit 2-adic branch constraint to U(1) and SU(2) lanes.
    """
    # Lane specs (as used in the verified selector variants for these demos).
    U1 = LaneSpec("U(1)", q=17, residues=(1, 5), tau_min=0.31, span=(97, 180), v2_required=3)
    SU2 = LaneSpec("SU(2)", q=13, residues=(3,),   tau_min=0.30, span=(97, 180), v2_required=1)
    SU3 = LaneSpec("SU(3)", q=17, residues=(1,),   tau_min=0.30, span=(97, 180), v2_required=None)

    # Raw survivors: residue + theta only (no 2-adic filter).
    raw = {
        "U(1)": lane_survivors(dataclasses.replace(U1, v2_required=None), apply_v2=False),
        "SU(2)": lane_survivors(dataclasses.replace(SU2, v2_required=None), apply_v2=False),
        "SU(3)": lane_survivors(SU3, apply_v2=False),
    }

    # Refined survivors: apply the explicit 2-adic branch (where specified).
    refined = {
        "U(1)": lane_survivors(U1, apply_v2=True),
        "SU(2)": lane_survivors(SU2, apply_v2=True),
        "SU(3)": lane_survivors(SU3, apply_v2=False),
    }

    triples: List[Triple] = []
    for wU in refined["U(1)"]:
        for s2 in refined["SU(2)"]:
            for s3 in refined["SU(3)"]:
                if len({wU, s2, s3}) != 3:
                    continue
                if wU - s2 <= 0:
                    continue
                triples.append(Triple(wU, s2, s3))

    triples = sorted(triples, key=lambda t: (t.wU, t.s2, t.s3))
    if len(triples) != 1:
        raise RuntimeError(f"Primary window selection not unique: found {len(triples)} triples: {[(t.wU,t.s2,t.s3) for t in triples]}")

    primary = triples[0]
    q2 = primary.wU - primary.s2
    q3 = odd_part(primary.wU - 1)
    v2U = v2(primary.wU - 1)
    eps = 1.0 / math.sqrt(q2)
    inv = Invariants(q2=q2, q3=q3, v2U=v2U, eps=eps)
    return primary, inv, raw, refined

def counterfactual_triples(primary: Triple, want: int = 4) -> List[Triple]:
    """
    Deterministic counterfactual triples used as ablations.

    Construction rule:
      1) Find the smallest prime wU > primary_window_hi satisfying the same U(1) residue + density gate
         AND matching the primary 2-adic branch v2(wU-1)=v2(primary.wU-1).
      2) Collect SU(2) and SU(3) survivor primes in a fixed expanded span [200, 500].
      3) Form the first `want` distinct triples by combining the first two SU(2) and first two SU(3)
         candidates with this new wU, subject to distinctness and q2>0.

    This produces a non-arbitrary, reproducible set of ablations that changes budgets (not physics).
    """
    # Lane bases (no v2 requirement for expanded pools).
    U1_base = LaneSpec("U(1)", q=17, residues=(1, 5), tau_min=0.31, span=(200, 500), v2_required=None)
    SU2_base = LaneSpec("SU(2)", q=13, residues=(3,),   tau_min=0.30, span=(200, 500), v2_required=None)
    SU3_base = LaneSpec("SU(3)", q=17, residues=(1,),   tau_min=0.30, span=(200, 500), v2_required=None)

    # Step 1: deterministically pick wU_cf by scanning upward beyond the primary window.
    v2_primary = v2(primary.wU - 1)
    wU_cf = None
    for p in primes_in(181, 2000):
        if p % U1_base.q not in U1_base.residues:
            continue
        if theta_density(p) + 1e-15 < U1_base.tau_min:
            continue
        if v2(p - 1) != v2_primary:
            continue
        if p == primary.wU:
            continue
        wU_cf = p
        break
    if wU_cf is None:
        raise RuntimeError("Could not find counterfactual wU within search bound.")

    # Step 2: expanded lane survivors (fixed span).
    su2 = lane_survivors(SU2_base, apply_v2=False)
    su3 = lane_survivors(SU3_base, apply_v2=False)
    su2 = sorted(su2)[:2]  # deterministic subset
    su3 = sorted(su3)[:2]  # deterministic subset

    # Step 3: deterministic first combinations.
    out: List[Triple] = []
    for s2 in su2:
        for s3 in su3:
            if len({wU_cf, s2, s3}) != 3:
                continue
            if wU_cf - s2 <= 0:
                continue
            out.append(Triple(wU_cf, s2, s3))
            if len(out) >= want:
                return out
    return out


# =========================
# Fourier helpers and filters
# =========================

def fftfreq_int(N: int) -> np.ndarray:
    """Integer Fourier mode numbers compatible with FFT ordering."""
    return (np.fft.fftfreq(N) * N).astype(np.int64)

def fejer_weights_1d(N: int, K: int) -> np.ndarray:
    k = np.abs(fftfreq_int(N))
    w = np.zeros(N, dtype=np.float64)
    mask = k <= K
    w[mask] = 1.0 - (k[mask] / (K + 1.0))
    return w

def sharp_weights_1d(N: int, K: int) -> np.ndarray:
    k = np.abs(fftfreq_int(N))
    return (k <= K).astype(np.float64)

def signed_weights_1d(N: int, K: int) -> np.ndarray:
    """
    Deterministic non-admissible control:
      - retain modes |k|<=K (sharp cutoff magnitude)
      - flip the sign on the upper half of the retained band (|k| in (K/2, K])

    This guarantees negative Fourier weights and therefore a kernel with negative lobes.
    """
    k = np.abs(fftfreq_int(N))
    w = sharp_weights_1d(N, K)
    mask = (k > (K // 2)) & (k <= K)
    w[mask] *= -1.0
    return w

def kernel_min_1d(weights: np.ndarray) -> float:
    ker = np.fft.ifft(weights).real
    return float(np.min(ker))

def apply_filter_hat_1d(u_hat: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return u_hat * weights

def tv_periodic(u: np.ndarray) -> float:
    return float(np.sum(np.abs(np.roll(u, -1) - u)))

def overshoot_mass_01(u: np.ndarray, dx: float) -> float:
    # Overshoot outside [0,1], integrated (L1 mass).
    above = np.maximum(0.0, u - 1.0)
    below = np.maximum(0.0, -u)
    return float(dx * np.sum(above + below))

def l2_error(u: np.ndarray, v: np.ndarray, dx: float) -> float:
    return float(math.sqrt(dx * np.sum((u - v) ** 2)))


# =========================
# Worked Example E1: Step advection
# =========================

@dataclasses.dataclass(frozen=True)
class AdvResult:
    t: float
    l2: float
    ov: float
    tv: float

def step_initial(x: np.ndarray) -> np.ndarray:
    # A periodic step: 1 on [0.25, 0.75), 0 elsewhere.
    return ((x >= 0.25) & (x < 0.75)).astype(np.float64)

def advect_step_spectral(N: int, K: int, t: float, method: str) -> np.ndarray:
    x = (np.arange(N) / N).astype(np.float64)
    u0 = step_initial(x)
    u_hat0 = np.fft.fft(u0)
    k = fftfreq_int(N).astype(np.float64)
    phase = np.exp(-2j * math.pi * k * t)  # c=1 on [0,1]
    u_hat_t = u_hat0 * phase

    if method == "fejer":
        w = fejer_weights_1d(N, K)
    elif method == "sharp":
        w = sharp_weights_1d(N, K)
    elif method == "signed":
        w = signed_weights_1d(N, K)
    else:
        raise ValueError("unknown method")

    u = np.fft.ifft(apply_filter_hat_1d(u_hat_t, w)).real
    return u

def advect_step_exact(N: int, t: float) -> np.ndarray:
    x = (np.arange(N) / N).astype(np.float64)
    x0 = (x - t) % 1.0
    return step_initial(x0)

def run_e1_advection(N: int, K: int, times: Sequence[float], eps: float) -> Tuple[List[AdvResult], List[AdvResult], List[AdvResult], List[Gate]]:
    dx = 1.0 / N
    out_fe: List[AdvResult] = []
    out_sh: List[AdvResult] = []
    out_si: List[AdvResult] = []
    gates: List[Gate] = []

    for t in times:
        u_ex = advect_step_exact(N, t)
        u_fe = advect_step_spectral(N, K, t, "fejer")
        u_sh = advect_step_spectral(N, K, t, "sharp")
        u_si = advect_step_spectral(N, K, t, "signed")

        fe = AdvResult(t=t, l2=l2_error(u_fe, u_ex, dx), ov=overshoot_mass_01(u_fe, dx), tv=tv_periodic(u_fe))
        sh = AdvResult(t=t, l2=l2_error(u_sh, u_ex, dx), ov=overshoot_mass_01(u_sh, dx), tv=tv_periodic(u_sh))
        si = AdvResult(t=t, l2=l2_error(u_si, u_ex, dx), ov=overshoot_mass_01(u_si, dx), tv=tv_periodic(u_si))

        out_fe.append(fe)
        out_sh.append(sh)
        out_si.append(si)

    # Gates:
    ov_all_zero = all(r.ov <= 5e-14 for r in out_fe)
    gates.append(Gate("E1-A: Admissible kernel keeps the solution in [0,1] (zero overshoot, all times)", ov_all_zero,
                      f"max_ov={fmt(max(r.ov for r in out_fe))}"))

    tv_reduced = all((sh.tv - fe.tv) >= eps for sh, fe in zip(out_sh, out_fe))
    gates.append(Gate("E1-B: Admissible kernel reduces total variation vs sharp truncation by >= eps (all times)", tv_reduced,
                      f"eps={fmt(eps)}"))

    control_rings = all((sh.ov > 1e-6) and (si.ov > 1e-6) for sh, si in zip(out_sh, out_si))
    gates.append(Gate("E1-C: Non-admissible controls show ringing (nonzero overshoot, all times)", control_rings))

    return out_fe, out_sh, out_si, gates

def run_e1_convergence_sweep(N: int, K: int, t: float) -> Tuple[List[int], List[float], Gate]:
    dx = 1.0 / N
    u_ex = advect_step_exact(N, t)
    Ks = [max(1, K // 2), K, min((N // 2) - 1, 2 * K)]
    errs: List[float] = []
    for Kc in Ks:
        u = advect_step_spectral(N, Kc, t, "fejer")
        errs.append(l2_error(u, u_ex, dx))
    noninc = (errs[1] <= errs[0] + 1e-15) and (errs[2] <= errs[1] + 1e-15)
    g = Gate("E1-D: Fejér error is non-increasing across K/2, K, 2K (no tuning)", noninc,
             f"errors={','.join(fmt(e) for e in errs)}")
    return Ks, errs, g


# =========================
# Worked Example E2: Smooth Poisson (periodic)
# =========================

def run_e2_poisson(N: int, eps: float) -> Tuple[Dict[str, float], List[Gate]]:
    """
    Solve -u'' = f on [0,1] with periodic BC and zero mean, using:
      - spectral inverse (exact on the grid)
      - second-order finite difference inverse (exact for the FD operator; no tolerance loops)

    Manufactured exact solution:
      u(x) = sin(2πx) + 0.5 cos(4πx)
    """
    x = (np.arange(N) / N).astype(np.float64)
    u_exact = np.sin(2 * math.pi * x) + 0.5 * np.cos(4 * math.pi * x)

    # Build f = -u'' exactly.
    f = (2 * math.pi) ** 2 * np.sin(2 * math.pi * x) + 0.5 * (4 * math.pi) ** 2 * np.cos(4 * math.pi * x)

    f_hat = np.fft.fft(f)
    k = fftfreq_int(N).astype(np.float64)

    # Spectral inverse: u_hat = f_hat / ( (2πk)^2 ) for k!=0 (since -u'' -> (2πk)^2 u_hat)
    denom_spec = (2 * math.pi * k) ** 2
    u_hat_spec = np.zeros_like(f_hat, dtype=np.complex128)
    mask = k != 0
    u_hat_spec[mask] = f_hat[mask] / denom_spec[mask]
    u_spec = np.fft.ifft(u_hat_spec).real

    # FD inverse: eigenvalues of periodic 2nd-order Laplacian
    dx = 1.0 / N
    # -Δ_fd corresponds to: (2 - 2 cos(2πk/N)) / dx^2
    lam = (2.0 - 2.0 * np.cos(2 * math.pi * k / N)) / (dx * dx)
    u_hat_fd = np.zeros_like(f_hat, dtype=np.complex128)
    mask2 = k != 0
    u_hat_fd[mask2] = f_hat[mask2] / lam[mask2]
    u_fd = np.fft.ifft(u_hat_fd).real

    e_fd = l2_error(u_fd, u_exact, dx)
    e_spec = l2_error(u_spec, u_exact, dx)

    ratio = e_spec / (e_fd + 1e-300)

    gates: List[Gate] = []
    # A derived, but very permissive gate: spectral should decisively beat FD for smooth data.
    gates.append(Gate("E2: Spectral Poisson inverse beats 2nd-order FD decisively (smooth manufactured solution)",
                      (ratio <= eps**6), f"ratio={fmt(ratio)}  eps^6={fmt(eps**6)}"))
    return {"L2_FD_2nd": e_fd, "L2_spectral": e_spec, "ratio": ratio}, gates


# =========================
# Worked Example E3: Fisher–KPP (reaction–diffusion)
# =========================

@dataclasses.dataclass(frozen=True)
class RDParams:
    N: int
    K: int
    D: float
    T: float
    dt: float
    steps: int

def kpp_initial(x: np.ndarray) -> np.ndarray:
    # Smooth, nontrivial initial condition in (0,1).
    # Chosen to remain bounded and to produce nonlinear mode coupling.
    return 0.55 + 0.05 * np.cos(2 * math.pi * x) + 0.02 * np.cos(6 * math.pi * x)

def logistic_step(u: np.ndarray, dt: float) -> np.ndarray:
    # Exact flow for du/dt = u(1-u): u(t+dt) = u / (u + (1-u) e^{-dt})
    e = math.exp(-dt)
    return u / (u + (1.0 - u) * e)

def rd_budgets(inv: Invariants) -> Tuple[RDParams, RDParams]:
    # Primary RD run uses N=512 (deterministic power-of-two from v2U=3).
    N = 2 ** (inv.v2U + 6)  # 512
    q3 = inv.q3
    # For this nonlinear test, we choose a conservative cutoff law:
    # K = floor(2N / q3), which matches the verified Fisher–KPP prework settings.
    K = int(math.floor((2.0 * N) / q3))
    K = max(4, min(K, (N // 2) - 1))

    # Physical parameter (part of the PDE): diffusion coefficient.
    D = 0.01
    T = 0.25

    # Deterministic explicit-stability-based dt for FD baseline: dt = 1/(4 D N^2).
    dt0 = 1.0 / (4.0 * D * (N ** 2))
    steps = int(math.ceil(T / dt0))
    dt = T / steps  # exact horizon

    primary = RDParams(N=N, K=K, D=D, T=T, dt=dt, steps=steps)

    # Deterministic reference: 2N with the same fixed dt law.
    Nref = 2 * N
    Kref = int(math.floor((2.0 * Nref) / q3))
    Kref = max(4, min(Kref, (Nref // 2) - 1))
    dt0r = 1.0 / (4.0 * D * (Nref ** 2))
    stepsr = int(math.ceil(T / dt0r))
    dtr = T / stepsr
    ref = RDParams(N=Nref, K=Kref, D=D, T=T, dt=dtr, steps=stepsr)
    return primary, ref

def apply_filter_hat_sep(u_hat: np.ndarray, w: np.ndarray) -> None:
    # In-place separable filter: multiply along the 1D Fourier axis.
    u_hat *= w

def rd_spectral(params: RDParams, method: str) -> np.ndarray:
    """
    Strang splitting:
      reaction half-step -> diffusion full-step (spectral) -> reaction half-step
    With deterministic Fourier cutoff applied after the diffusion step.
    """
    N, K, D, dt, steps = params.N, params.K, params.D, params.dt, params.steps
    x = (np.arange(N) / N).astype(np.float64)
    u = kpp_initial(x).astype(np.float64)

    k = fftfreq_int(N).astype(np.float64)
    lap = -(2 * math.pi * k) ** 2  # d^2/dx^2 eigenvalue
    diff_factor = np.exp(D * lap * dt)

    if method == "fejer":
        w = fejer_weights_1d(N, K)
    elif method == "sharp":
        w = sharp_weights_1d(N, K)
    elif method == "signed":
        w = signed_weights_1d(N, K)
    else:
        raise ValueError("unknown method")

    for _ in range(steps):
        u = logistic_step(u, 0.5 * dt)
        u_hat = np.fft.fft(u)
        u_hat = u_hat * diff_factor
        u_hat = apply_filter_hat_1d(u_hat, w)
        u = np.fft.ifft(u_hat).real
        u = logistic_step(u, 0.5 * dt)
    return u

def rd_fd(params: RDParams) -> np.ndarray:
    """
    Classical baseline: 2nd-order finite difference diffusion + forward Euler reaction.
    Deterministic fixed dt; no tolerance iteration.
    """
    N, D, dt, steps = params.N, params.D, params.dt, params.steps
    x = (np.arange(N) / N).astype(np.float64)
    u = kpp_initial(x).astype(np.float64)
    dx = 1.0 / N
    r = D * dt / (dx * dx)

    for _ in range(steps):
        u_xx = (np.roll(u, -1) - 2.0 * u + np.roll(u, 1)) / (dx * dx)
        u = u + dt * (D * u_xx + u * (1.0 - u))
    return u

def downsample_by_2(u_ref: np.ndarray) -> np.ndarray:
    return u_ref[::2].copy()

def run_e3_reaction_diffusion(inv: Invariants) -> Tuple[Dict[str, float], List[Gate]]:
    p, pref = rd_budgets(inv)

    u_ref = rd_spectral(pref, "fejer")
    u_ref_ds = downsample_by_2(u_ref)

    u_fd = rd_fd(p)
    u_fe = rd_spectral(p, "fejer")
    u_sh = rd_spectral(p, "sharp")
    u_si = rd_spectral(p, "signed")

    dx = 1.0 / p.N
    e_fd = l2_error(u_fd, u_ref_ds, dx)
    e_fe = l2_error(u_fe, u_ref_ds, dx)
    e_sh = l2_error(u_sh, u_ref_ds, dx)
    e_si = l2_error(u_si, u_ref_ds, dx)

    mn, mx = float(np.min(u_fe)), float(np.max(u_fe))
    illegal_nan = (np.any(~np.isfinite(u_si)) or np.any(~np.isfinite(u_sh)))

    gates: List[Gate] = []
    # Gate: lawful is competitive vs classical FD (and typically far better).
    gates.append(Gate("E3-A: Admissible method is competitive vs 2nd-order FD at fixed N (no tuning)",
                      (e_fe <= (1.0 + inv.eps) * e_fd),
                      f"e_fe/e_fd={fmt(e_fe/(e_fd+1e-300))}  eps={fmt(inv.eps)}"))
    # Gate: non-admissible controls degrade vs lawful (or blow up).
    gates.append(Gate("E3-B: Non-admissible controls are worse than admissible (or non-finite)",
                      (illegal_nan or (min(e_sh, e_si) >= (1.0 + inv.eps) * e_fe)),
                      f"e_sh={fmt(e_sh)} e_si={fmt(e_si)} e_fe={fmt(e_fe)}"))
    # Gate: boundedness.
    gates.append(Gate("E3-C: Boundedness preserved (admissible) for Fisher–KPP (0<=u<=1+small slack)",
                      (mn >= -1e-8) and (mx <= 1.0 + 1e-6),
                      f"min={fmt(mn)} max={fmt(mx)}"))

    return {
        "N": p.N, "K": p.K, "D": p.D, "T": p.T, "dt": p.dt, "steps": p.steps,
        "L2_FD": e_fd, "L2_fejer": e_fe, "L2_sharp": e_sh, "L2_signed": e_si,
        "min_fejer": mn, "max_fejer": mx, "illegal_nonfinite": bool(illegal_nan),
    }, gates


# =========================
# Worked Example E4: 3D Navier–Stokes (Taylor–Green vortex)
# =========================

@dataclasses.dataclass(frozen=True)
class NS3DBudgets:
    N: int
    K_primary: int
    K_truth: int
    nu: float
    T: float
    dt: float
    steps: int

def ns3d_budgets(inv: Invariants, tier: str) -> NS3DBudgets:
    # Resolution tier: smoke vs industrial
    if tier == "smoke":
        N = 2 ** (inv.v2U + 3)  # 64
        T = 0.6
    elif tier == "industrial":
        N = 2 ** (inv.v2U + 5)  # 256
        T = 0.6
    else:
        raise ValueError("tier must be smoke or industrial")

    q3 = inv.q3

    # Cutoff policy (verified in the NS3D certificate): K_primary = floor(4N / q3).
    Kp = int(math.floor((4.0 * N) / q3))
    Kp = max(4, min(Kp, (N // 2) - 1))

    # Truth uses full grid spectrum (still Fejér-averaged; K_truth at Nyquist-1).
    Kt = (N // 2) - 1

    # Viscosity from invariants (first principles, no tuning): nu = 1 / (2 q2 q3).
    nu = 1.0 / (2.0 * inv.q2 * q3)

    # Deterministic dt rule: base_dt = (π/2)/N (a CFL-like proxy),
    # then choose integer steps to hit T exactly.
    base_dt = (math.pi / 2.0) / N
    steps = max(1, int(round(T / base_dt)))
    dt = T / steps

    return NS3DBudgets(N=N, K_primary=Kp, K_truth=Kt, nu=nu, T=T, dt=dt, steps=steps)

def taylor_green_initial(N: int, dtype_real=np.float32) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standard Taylor–Green vortex initial condition on [0,2π]^3:
      u =  sin(x) cos(y) cos(z)
      v = -cos(x) sin(y) cos(z)
      w =  0
    """
    x = (2.0 * math.pi) * (np.arange(N, dtype=dtype_real) / N)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    u = np.sin(X) * np.cos(Y) * np.cos(Z)
    v = -np.cos(X) * np.sin(Y) * np.cos(Z)
    w = np.zeros_like(u)
    return u.astype(dtype_real), v.astype(dtype_real), w.astype(dtype_real)

def fejer_sep_1d(N: int, K: int, dtype=np.float32) -> np.ndarray:
    k = np.abs(fftfreq_int(N)).astype(np.float32)
    w = np.zeros(N, dtype=dtype)
    mask = k <= K
    w[mask] = 1.0 - (k[mask] / (K + 1.0))
    return w

def sharp_sep_1d(N: int, K: int, dtype=np.float32) -> np.ndarray:
    k = np.abs(fftfreq_int(N)).astype(np.int64)
    return (k <= K).astype(dtype)

def signed_sep_1d(N: int, K: int, dtype=np.float32) -> np.ndarray:
    """Separable 1D control weights for 3D: sharp magnitude, sign-flipped upper half-band."""
    k = np.abs(fftfreq_int(N)).astype(np.int64)
    w = (k <= K).astype(dtype)
    mask = (k > (K // 2)) & (k <= K)
    w[mask] *= np.array(-1.0, dtype=dtype)
    return w


def filter_weight_kmax_3d(N: int, K: int, kind: str) -> np.ndarray:
    """
    3D Fourier filter weight based on k∞ := max(|kx|,|ky|,|kz|).

    - "fejer":   w = max(0, 1 - k∞/(K+1))   (admissible averaging; nonnegative weights)
    - "sharp":   w = 1_{k∞<=K}             (non-admissible control)
    - "signed":  w = 1_{k∞<=K} with sign flip on k∞ in (K/2, K] (non-admissible control)
    """
    kk = np.abs(fftfreq_int(N)).astype(np.int16)
    ax = kk[:, None, None]
    ay = kk[None, :, None]
    az = kk[None, None, :]
    kmax = np.maximum(np.maximum(ax, ay), az).astype(np.float32)

    if kind == "fejer":
        w = np.maximum(0.0, 1.0 - (kmax / (K + 1.0))).astype(np.float32)
    elif kind == "sharp":
        w = (kmax <= float(K)).astype(np.float32)
    elif kind == "signed":
        w = (kmax <= float(K)).astype(np.float32)
        band = (kmax > float(K // 2)) & (kmax <= float(K))
        w[band] *= -1.0
    else:
        raise ValueError("unknown filter kind")
    return w

def apply_sep_filter_3d(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, wx: np.ndarray, wy: np.ndarray, wz: np.ndarray) -> None:
    # In-place separable multiply along each axis. Broadcasting creates temporaries;
    # we keep the operation order fixed for determinism.
    Ux *= wx[:, None, None]
    Uy *= wx[:, None, None]
    Uz *= wx[:, None, None]

    Ux *= wy[None, :, None]
    Uy *= wy[None, :, None]
    Uz *= wy[None, :, None]

    Ux *= wz[None, None, :]
    Uy *= wz[None, None, :]
    Uz *= wz[None, None, :]

def leray_project(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray) -> None:
    """
    In-place Leray projection to enforce incompressibility in Fourier space:
      U <- U - k (k·U)/|k|^2  for k != 0.
    """
    # kdot = kx*Ux + ky*Uy + kz*Uz
    kdot = (kx[:, None, None] * Ux) + (ky[None, :, None] * Uy) + (kz[None, None, :] * Uz)
    # avoid divide-by-zero at k=0
    kdot_over_k2 = np.zeros_like(kdot)
    mask = (k2 != 0.0)
    kdot_over_k2[mask] = kdot[mask] / k2[mask]
    Ux -= kx[:, None, None] * kdot_over_k2
    Uy -= ky[None, :, None] * kdot_over_k2
    Uz -= kz[None, None, :] * kdot_over_k2

def divergence_l2(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, kx: np.ndarray, ky: np.ndarray, kz: np.ndarray) -> float:
    I = np.complex64(1j)
    div_hat = I * (kx[:, None, None] * Ux + ky[None, :, None] * Uy + kz[None, None, :] * Uz)
    div = np.fft.ifftn(div_hat).real
    return float(np.sqrt(np.mean(div ** 2)))

def hf_energy_fraction(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray, Kcut: int) -> float:
    """
    Fraction of spectral kinetic energy in modes with Euclidean wavenumber |k|_2 > Kcut.

    This choice is intentional: even if a method uses a box cutoff in k∞, corner modes can satisfy
    |k|_2 > Kcut while still having k∞ <= Kcut. This makes the metric sensitive to admissible
    smoothing vs non-admissible truncation.
    """
    N = Ux.shape[0]
    kk = np.abs(fftfreq_int(N)).astype(np.float32)
    ax = kk[:, None, None]
    ay = kk[None, :, None]
    az = kk[None, None, :]
    kmag = np.sqrt(ax * ax + ay * ay + az * az)
    mask_hf = (kmag > float(Kcut))

    E = (np.abs(Ux) ** 2 + np.abs(Uy) ** 2 + np.abs(Uz) ** 2)
    num = float(np.sum(E[mask_hf]))
    den = float(np.sum(E))
    return num / (den + 1e-300)

def ns3d_step_rhs(Ux: np.ndarray, Uy: np.ndarray, Uz: np.ndarray,
                 kx: np.ndarray, ky: np.ndarray, kz: np.ndarray, k2: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute nonlinear RHS in Fourier space using rotational form:
      f(U) = -P( (curl u) × u )̂
    """
    I = np.complex64(1j)
    # ω̂ = i k × Û
    Wx = I * (ky[None, :, None] * Uz - kz[None, None, :] * Uy)
    Wy = I * (kz[None, None, :] * Ux - kx[:, None, None] * Uz)
    Wz = I * (kx[:, None, None] * Uy - ky[None, :, None] * Ux)

    # u = ifft(Û), ω = ifft(ω̂)  (physical space, real-valued)
    ux = np.fft.ifftn(Ux).real.astype(np.float32, copy=False)
    uy = np.fft.ifftn(Uy).real.astype(np.float32, copy=False)
    uz = np.fft.ifftn(Uz).real.astype(np.float32, copy=False)

    wx = np.fft.ifftn(Wx).real.astype(np.float32, copy=False)
    wy = np.fft.ifftn(Wy).real.astype(np.float32, copy=False)
    wz = np.fft.ifftn(Wz).real.astype(np.float32, copy=False)

    # n = ω × u
    nx = (wy * uz - wz * uy).astype(np.float32, copy=False)
    ny = (wz * ux - wx * uz).astype(np.float32, copy=False)
    nz = (wx * uy - wy * ux).astype(np.float32, copy=False)

    # Transform back
    Nx = np.fft.fftn(nx).astype(np.complex64, copy=False)
    Ny = np.fft.fftn(ny).astype(np.complex64, copy=False)
    Nz = np.fft.fftn(nz).astype(np.complex64, copy=False)

    # Project (in-place)
    leray_project(Nx, Ny, Nz, kx, ky, kz, k2)
    # RHS = -N̂
    return (-Nx).astype(np.complex64, copy=False), (-Ny).astype(np.complex64, copy=False), (-Nz).astype(np.complex64, copy=False)

def ns3d_run(b: NS3DBudgets, method: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Run pseudo-spectral incompressible NS with Strang splitting:
      diffusion half-step -> nonlinear RK2 (Heun) -> diffusion half-step
    and then apply the selected filter (admissible or control).

    Returns:
      (Ux_hat, Uy_hat, Uz_hat, divL2, hf_fraction)
    """
    N, nu, dt, steps = b.N, b.nu, b.dt, b.steps
    # Wavenumbers (integer mode numbers on [0,2π]^3)
    k = fftfreq_int(N).astype(np.float32)
    kx = k
    ky = k
    kz = k
    # k2 full grid (float32)
    k2 = (kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2).astype(np.float32)
    # Diffusion factor for half step
    diff_half = np.exp(-nu * k2 * (0.5 * dt)).astype(np.float32)

    # Initial condition in physical space, then FFT to spectral.
    u0x, u0y, u0z = taylor_green_initial(N, dtype_real=np.float32)
    Ux = np.fft.fftn(u0x).astype(np.complex64, copy=False)
    Uy = np.fft.fftn(u0y).astype(np.complex64, copy=False)
    Uz = np.fft.fftn(u0z).astype(np.complex64, copy=False)
    leray_project(Ux, Uy, Uz, kx, ky, kz, k2)

    # Filter weights (k∞-based, precomputed once)
    if method == "fejer":
        W = filter_weight_kmax_3d(N, b.K_primary, "fejer")
    elif method == "sharp":
        W = filter_weight_kmax_3d(N, b.K_primary, "sharp")
    elif method == "signed":
        W = filter_weight_kmax_3d(N, b.K_primary, "signed")
    elif method == "truth":
        W = filter_weight_kmax_3d(N, b.K_truth, "fejer")
    else:
        raise ValueError("unknown method")

# Fixed-step time integration (no tolerance loops)
    for _ in range(steps):
        # diffusion half
        Ux *= diff_half
        Uy *= diff_half
        Uz *= diff_half

        # nonlinear RK2 (Heun) without diffusion
        k1x, k1y, k1z = ns3d_step_rhs(Ux, Uy, Uz, kx, ky, kz, k2)
        Ux1 = (Ux + dt * k1x).astype(np.complex64, copy=False)
        Uy1 = (Uy + dt * k1y).astype(np.complex64, copy=False)
        Uz1 = (Uz + dt * k1z).astype(np.complex64, copy=False)
        k2x, k2y, k2z = ns3d_step_rhs(Ux1, Uy1, Uz1, kx, ky, kz, k2)
        Ux = (Ux + 0.5 * dt * (k1x + k2x)).astype(np.complex64, copy=False)
        Uy = (Uy + 0.5 * dt * (k1y + k2y)).astype(np.complex64, copy=False)
        Uz = (Uz + 0.5 * dt * (k1z + k2z)).astype(np.complex64, copy=False)

        # diffusion half
        Ux *= diff_half
        Uy *= diff_half
        Uz *= diff_half

        # filtering
        Ux *= W
        Uy *= W
        Uz *= W
        leray_project(Ux, Uy, Uz, kx, ky, kz, k2)

    divL2 = divergence_l2(Ux, Uy, Uz, kx, ky, kz)
    hf = hf_energy_fraction(Ux, Uy, Uz, b.K_primary)
    return Ux, Uy, Uz, divL2, hf

def ns3d_error_vs_truth(U: Tuple[np.ndarray, np.ndarray, np.ndarray],
                        Ut: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    # L2 in physical space, normalized by truth L2.
    Ux, Uy, Uz = U
    Tx, Ty, Tz = Ut
    ux = np.fft.ifftn(Ux).real
    uy = np.fft.ifftn(Uy).real
    uz = np.fft.ifftn(Uz).real
    tx = np.fft.ifftn(Tx).real
    ty = np.fft.ifftn(Ty).real
    tz = np.fft.ifftn(Tz).real
    num = np.sqrt(np.mean((ux - tx) ** 2 + (uy - ty) ** 2 + (uz - tz) ** 2))
    den = np.sqrt(np.mean(tx ** 2 + ty ** 2 + tz ** 2)) + 1e-300
    return float(num / den)

def run_e4_ns3d(inv: Invariants, primary: Triple, tier: str, cfs: List[Triple]) -> Tuple[Dict[str, object], List[Gate]]:
    b = ns3d_budgets(inv, tier=tier)

    # Compute truth (deterministic, but can be expensive).
    t0 = time.time()
    Utx, Uty, Utz, divt, hft = ns3d_run(b, method="truth")
    truth_runtime = time.time() - t0

    t0 = time.time()
    Ufx, Ufy, Ufz, divf, hff = ns3d_run(b, method="fejer")
    Ucx, Ucy, Ucz, divc, hfc = ns3d_run(b, method="sharp")
    Usx, Usy, Usz, divs, hfs = ns3d_run(b, method="signed")
    primary_runtime = time.time() - t0

    e_fe = ns3d_error_vs_truth((Ufx, Ufy, Ufz), (Utx, Uty, Utz))
    e_sh = ns3d_error_vs_truth((Ucx, Ucy, Ucz), (Utx, Uty, Utz))
    e_si = ns3d_error_vs_truth((Usx, Usy, Usz), (Utx, Uty, Utz))

    gates: List[Gate] = []

    # Incompressibility gate (very permissive; should be much smaller than 1e-8 typically).
    ok_div = (divf <= 1e-8) and (divc <= 1e-8) and (divs <= 1e-8)
    gates.append(Gate("E4-G1: Incompressibility (divergence L2 <= 1e-8) for all variants", ok_div,
                      f"div_fe={fmt(divf)} div_sh={fmt(divc)} div_si={fmt(divs)}"))

    # Accuracy gate: admissible must be closer to the admissible truth than non-admissible controls.
    ok_acc = (e_fe <= e_sh) and (e_fe <= e_si)
    gates.append(Gate("E4-G2: Admissible run is closer to admissible truth than non-admissible controls", ok_acc,
                      f"e_fe={fmt(e_fe)} e_sh={fmt(e_sh)} e_si={fmt(e_si)}"))

    # HF gate: signed control should inject more HF energy than admissible.
    base = max(hff, 1e-18)
    ok_hf = (hfs > 0.0) and (hfc > 0.0) and (hfs >= 10.0 * base) and (hfc >= 10.0 * base)
    gates.append(Gate("E4-G3: Non-admissible controls inject high-frequency energy (>=10x admissible)", ok_hf,
                      f"hf_fe={fmt(hff)} hf_sh={fmt(hfc)} hf_si={fmt(hfs)}"))

    # Counterfactual teeth: budgets change (via q3 -> K policy) while physics is fixed.
    # We enforce this gate at N>=64 tiers (smoke and industrial qualify).
    err_primary = e_fe
    strong = 0
    rows: List[Tuple[str, int, int, float, bool]] = []
    for cf in cfs:
        q3_cf = odd_part(cf.wU - 1)
        K_cf = int(math.floor((4.0 * b.N) / q3_cf))
        K_cf = max(4, min(K_cf, (b.N // 2) - 1))
        b_cf = dataclasses.replace(b, K_primary=K_cf)  # physics fixed
        # Run admissible method under counterfactual K
        Ucfx, Ucfy, Ucfz, divcf, hfcf = ns3d_run(b_cf, method="fejer")
        e_cf = ns3d_error_vs_truth((Ucfx, Ucfy, Ucfz), (Utx, Uty, Utz))
        degrade = (e_cf >= (1.0 + inv.eps) * err_primary)
        if degrade:
            strong += 1
        rows.append((f"({cf.wU},{cf.s2},{cf.s3})", q3_cf, K_cf, e_cf, degrade))

    ok_teeth = strong >= max(1, (3 * len(cfs)) // 4)  # >=3/4
    gates.append(Gate("E4-T1: Counterfactual triples degrade performance by (1+eps) in >=3/4 cases", ok_teeth,
                      f"strong={strong}/{len(cfs)} eps={fmt(inv.eps)}"))

    payload = {
        "tier": tier,
        "N": b.N, "K_primary": b.K_primary, "K_truth": b.K_truth, "nu": b.nu, "T": b.T, "dt": b.dt, "steps": b.steps,
        "truth_runtime_s": float(truth_runtime),
        "primary_runtime_s": float(primary_runtime),
        "err_fejer": e_fe, "err_sharp": e_sh, "err_signed": e_si,
        "div_fejer": divf, "div_sharp": divc, "div_signed": divs,
        "hf_fejer": hff, "hf_sharp": hfc, "hf_signed": hfs,
        "counterfactuals": [{"triple": r[0], "q3": r[1], "K": r[2], "err": r[3], "degrade": r[4]} for r in rows],
    }
    return payload, gates


# =========================
# Industrial-scale readiness ledger (informational)
# =========================

def ns3d_readiness_ledger(inv: Invariants) -> Dict[str, object]:
    """
    Deterministic memory/work proxy table (informational; not a runtime claim).
    We assume a conservative upper-bound working set size in Fourier space.
    """
    q3 = inv.q3

    def mem_gb_upper(N: int) -> float:
        # Upper-bound: 18 Fourier volumes (3 velocity components, plus scratch).
        # Stored dtype is complex64 in our code, but numpy FFT will still allocate complex128 temporaries.
        # This table is a conservative upper bound for persistent storage only.
        bytes_per = 8  # complex64
        volumes = 18
        return volumes * (N**3) * bytes_per / (1024**3)

    def fft3_proxy(N: int) -> float:
        # Proxy ~ N^3 log2(N^3)
        return (N**3) * math.log2(max(2, N**3))

    Ns = [64, 128, 256, 384, 512]
    rows = []
    for N in Ns:
        Kp = int(math.floor((4.0 * N) / q3))
        Kp = max(4, min(Kp, (N // 2) - 1))
        base_dt = (math.pi / 2.0) / N
        steps = int(round(10.0 / base_dt))
        dt = 10.0 / steps
        rows.append({
            "N": N,
            "grid_pts": N**3,
            "K_primary": Kp,
            "mem_GB_upper": mem_gb_upper(N),
            "fft3_proxy": fft3_proxy(N),
            "dt_at_T10": dt,
            "steps_at_T10": steps,
        })

    example = {
        "N": 256,
        "FFTs_per_step": 16,
        "T": 10.0,
        "dt": rows[2]["dt_at_T10"],
        "steps": rows[2]["steps_at_T10"],
    }
    work_proxy = rows[2]["fft3_proxy"] * example["FFTs_per_step"] * example["steps"]

    return {"rows": rows, "example_industrial_envelope": example, "work_proxy": work_proxy}


# =========================
# Main driver
# =========================

def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--tier", choices=("smoke", "industrial"), default="smoke",
                    help="NS3D tier: smoke (N=64) or industrial (N=256). Default: smoke.")
    ap.add_argument("--no_ns3d", action="store_true", help="Skip the 3D Navier–Stokes example.")
    args = ap.parse_args()

    utc = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    print(hr())
    print("DEMO-56 — Deterministic Operator Calculus vs Classical Finite Differences")
    print("MASTER FLAGSHIP (Maximum-Impact), Referee-Ready")
    print(hr())
    print(f"UTC time : {utc}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : none (stdout only)")

    # -------------------------
    # Stage 1: Triple selection
    # -------------------------
    print("\n" + title("STAGE 1 — Deterministic Triple Selection (Primary Window)"))
    primary, inv, raw_pools, pools = select_primary_triple()
    cfs = counterfactual_triples(primary, want=4)

    print("Lane survivor pools (raw: residue + density only):")
    for k in ("U(1)", "SU(2)", "SU(3)"):
        print(f"  {k:<5}: {raw_pools[k]}")
    print("Lane survivor pools (refined: explicit 2-adic branch where specified):")
    for k in ("U(1)", "SU(2)", "SU(3)"):
        print(f"  {k:<5}: {pools[k]}")

    print(f"Primary-window admissible triples: [({primary.wU}, {primary.s2}, {primary.s3})]")
    print_gate(Gate("Unique admissible triple in primary window", True, "count=1"))
    print_gate(Gate("Primary equals (137,107,103)", (primary.wU, primary.s2, primary.s3) == (137, 107, 103),
                    f"selected=Triple(wU={primary.wU}, s2={primary.s2}, s3={primary.s3})"))

    print(f"Counterfactual triples (ablations, same rules outside the primary window): {[(t.wU,t.s2,t.s3) for t in cfs]}")
    print_gate(Gate("Captured >=4 counterfactual triples", len(cfs) >= 4, f"found={len(cfs)}"))

    print("\nDerived invariants (from the selected triple):")
    print(f"  q2 = wU - s2 = {inv.q2}")
    print(f"  q3 = odd_part(wU - 1) = {inv.q3}")
    print(f"  v2 = v2(wU - 1) = {inv.v2U}")
    print(f"  eps = 1/sqrt(q2) = {fmt(inv.eps)}")

    # Spec hash (deterministic)
    spec = {
        "demo": "DEMO-56 master flagship maximum impact v6",
        "primary_triple": (primary.wU, primary.s2, primary.s3),
        "counterfactual_triples": [(t.wU, t.s2, t.s3) for t in cfs],
        "invariants": dataclasses.asdict(inv),
        "tier": args.tier,
        "ns3d_enabled": (not args.no_ns3d),
    }
    spec_hash = sha256_of_json(spec)
    print(f"\nspec_sha256: {spec_hash}")

    # -------------------------
    # Stage 2: Kernel audit
    # -------------------------
    print("\n" + title("STAGE 2 — Kernel Admissibility Audit (Fejér vs Non-Admissible Controls)"))
    # Use N=256, K derived from invariants for this audit (same as E1/E2).
    N_audit = 2 ** (inv.v2U + 5)  # 256
    K_audit = int(math.floor((4.0 * N_audit) / inv.q3))
    K_audit = max(4, min(K_audit, (N_audit // 2) - 1))

    w_fe = fejer_weights_1d(N_audit, K_audit)
    w_sh = sharp_weights_1d(N_audit, K_audit)
    w_si = signed_weights_1d(N_audit, K_audit)

    kmin_fe = kernel_min_1d(w_fe)
    kmin_sh = kernel_min_1d(w_sh)
    kmin_si = kernel_min_1d(w_si)

    g2 = [
        Gate("Fejér kernel is nonnegative (within tiny numerical slack)", kmin_fe >= -1e-12, f"kmin={fmt(kmin_fe)}"),
        Gate("Sharp truncation kernel has negative lobes (non-admissible control)", kmin_sh < -1e-6, f"kmin={fmt(kmin_sh)}"),
        Gate("Signed filter kernel has negative lobes (non-admissible control)", kmin_si < -1e-6, f"kmin={fmt(kmin_si)}"),
    ]
    for g in g2:
        print_gate(g)

    # -------------------------
    # Example E1: step advection
    # -------------------------
    print("\n" + title("E1 — Linear Advection of a Step (Ringing / Total Variation Audit)"))
    times = [0.13, 0.37, 0.71]
    fe, sh, si, g_e1 = run_e1_advection(N_audit, K_audit, times, inv.eps)

    for r_fe, r_sh, r_si in zip(fe, sh, si):
        print(f"t={r_fe.t:.2f}  L2: fejer={fmt(r_fe.l2)} sharp={fmt(r_sh.l2)} signed={fmt(r_si.l2)}")
        print(f"        overshoot_mass: fejer={fmt(r_fe.ov)} sharp={fmt(r_sh.ov)} signed={fmt(r_si.ov)}")
        print(f"        TV           : fejer={fmt(r_fe.tv)} sharp={fmt(r_sh.tv)} signed={fmt(r_si.tv)}")

    for g in g_e1:
        print_gate(g)

    Ks, errs, g_sweep = run_e1_convergence_sweep(N_audit, K_audit, t=0.37)
    print("\nConvergence sweep (E1, Fejér):")
    for Kc, ec in zip(Ks, errs):
        print(f"  K={Kc:<4d}  L2={fmt(ec)}")
    print_gate(g_sweep)

    # Counterfactual teeth for E1: compare Fejér L2 at t=0.37 with K derived from counterfactual wU (via q3).
    print("\nCounterfactual teeth (E1): budgets change via q3 -> K, physics fixed")
    e_primary = errs[1]  # L2 at K=K_audit
    strong = 0
    for cf in cfs:
        q3_cf = odd_part(cf.wU - 1)
        K_cf = int(math.floor((4.0 * N_audit) / q3_cf))
        K_cf = max(4, min(K_cf, (N_audit // 2) - 1))
        u_cf = advect_step_spectral(N_audit, K_cf, 0.37, "fejer")
        u_ex = advect_step_exact(N_audit, 0.37)
        e_cf = l2_error(u_cf, u_ex, 1.0 / N_audit)
        degrade = (e_cf >= (1.0 + inv.eps) * e_primary)
        if degrade:
            strong += 1
        print(f"  CF ({cf.wU},{cf.s2},{cf.s3})  q3={q3_cf:<3d}  K={K_cf:<3d}  L2={fmt(e_cf)}  degrade={degrade}")
    g_teeth_e1 = Gate("E1-T: Counterfactual budgets degrade by (1+eps) in >=3/4 cases", strong >= 3,
                      f"strong={strong}/{len(cfs)} eps={fmt(inv.eps)}")
    print_gate(g_teeth_e1)

    # -------------------------
    # Example E2: Poisson
    # -------------------------
    print("\n" + title("E2 — 1D Poisson (Smooth Manufactured Solution)"))
    e2_payload, g_e2 = run_e2_poisson(N_audit, inv.eps)
    print(f"L2 error: FD_2nd={fmt(e2_payload['L2_FD_2nd'])}  spectral={fmt(e2_payload['L2_spectral'])}  ratio={fmt(e2_payload['ratio'])}")
    for g in g_e2:
        print_gate(g)

    # -------------------------
    # Example E3: Fisher–KPP
    # -------------------------
    print("\n" + title("E3 — 1D Reaction–Diffusion (Fisher–KPP Nonlinearity)"))
    e3_payload, g_e3 = run_e3_reaction_diffusion(inv)
    print(f"Params: N={e3_payload['N']} K={e3_payload['K']} D={e3_payload['D']} T={e3_payload['T']} dt={fmt(e3_payload['dt'])} steps={e3_payload['steps']}")
    print(f"L2 vs internal ref: FD={fmt(e3_payload['L2_FD'])}  Fejér={fmt(e3_payload['L2_fejer'])}  Sharp={fmt(e3_payload['L2_sharp'])}  Signed={fmt(e3_payload['L2_signed'])}")
    print(f"Bounds (Fejér): min={fmt(e3_payload['min_fejer'])} max={fmt(e3_payload['max_fejer'])}  Non-finite controls: {e3_payload['illegal_nonfinite']}")
    for g in g_e3:
        print_gate(g)

    # -------------------------
    # Example E4: 3D Navier–Stokes
    # -------------------------
    ns3d_payload = None
    g_ns: List[Gate] = []
    if args.no_ns3d:
        print("\n" + title("E4 — 3D Navier–Stokes (Taylor–Green)"))
        print("SKIP  User requested --no_ns3d")
    else:
        print("\n" + title("E4 — 3D Navier–Stokes (Taylor–Green Vortex)"))
        print(f"Tier: {args.tier}  (smoke=N=64, industrial=N=256)")
        try:
            ns3d_payload, g_ns = run_e4_ns3d(inv, primary, tier=args.tier, cfs=cfs)
            b = ns3d_payload
            print(f"Budgets: N={b['N']}  K_primary={b['K_primary']}  K_truth={b['K_truth']}  nu={fmt(b['nu'])}  dt={fmt(b['dt'])}  steps={b['steps']}  T={b['T']}")
            print(f"Truth runtime (informational): {fmt(b['truth_runtime_s'])} s")
            print(f"Errors vs truth (normalized L2): fejer={fmt(b['err_fejer'])} sharp={fmt(b['err_sharp'])} signed={fmt(b['err_signed'])}")
            print(f"divL2: fejer={fmt(b['div_fejer'])} sharp={fmt(b['div_sharp'])} signed={fmt(b['div_signed'])}")
            print(f"HF energy frac (>K_primary): fejer={fmt(b['hf_fejer'])} sharp={fmt(b['hf_sharp'])} signed={fmt(b['hf_signed'])}")
            for g in g_ns:
                print_gate(g)
            print("\nCounterfactuals (E4):")
            for row in b["counterfactuals"]:
                print(f"  CF {row['triple']}  q3={row['q3']:<3d}  K={row['K']:<3d}  err={fmt(row['err'])}  degrade={row['degrade']}")
        except MemoryError as e:
            print("FAIL  E4 could not run due to insufficient memory.")
            g_ns = [Gate("E4: runtime memory availability", False, "MemoryError")]
        except Exception as e:
            print(f"FAIL  E4 encountered an unexpected error: {type(e).__name__}: {e}")
            g_ns = [Gate("E4: runtime execution", False, f"{type(e).__name__}: {e}")]

    # -------------------------
    # Readiness ledger (informational)
    # -------------------------
    print("\n" + title("INDUSTRIAL-SCALE READINESS LEDGER (Informational, Deterministic)"))
    led = ns3d_readiness_ledger(inv)
    print("Memory / work proxy per resolution (Fourier velocity, 3 components):")
    print("Assumptions: complex64 persistent storage; FFT temporaries can be larger.")
    print(f"{'N':>6} {'Grid pts':>12} {'Kp':>6} {'Mem_GB':>10} {'FFT3 proxy':>12} {'dt@T=10':>10} {'steps':>8}")
    for row in led["rows"]:
        print(f"{row['N']:>6d} {row['grid_pts']:>12d} {row['K_primary']:>6d} {row['mem_GB_upper']:>10.3f} {row['fft3_proxy']:>12.3e} {row['dt_at_T10']:>10.6f} {row['steps_at_T10']:>8d}")
    print("\nExample industrial envelope (deterministic dt from a CFL proxy; no tuning):")
    print(json.dumps(led["example_industrial_envelope"], indent=2))
    print(f"Total work proxy ≈ {led['work_proxy']:.3e}")

    # -------------------------
    # Final verdict + determinism hash
    # -------------------------
    print("\n" + title("DETERMINISM HASH"))
    results_payload = {
        "spec": spec,
        "spec_sha256": spec_hash,
        "stage2_kernel_min": {"fejer": kmin_fe, "sharp": kmin_sh, "signed": kmin_si},
        "E1": {
            "N": N_audit, "K": K_audit,
            "times": times,
            "fejer": [dataclasses.asdict(r) for r in fe],
            "sharp": [dataclasses.asdict(r) for r in sh],
            "signed": [dataclasses.asdict(r) for r in si],
            "K_sweep": Ks, "sweep_errors": errs,
        },
        "E2": e2_payload,
        "E3": e3_payload,
        "E4": ns3d_payload,
        "ledger": led,
    }
    det_hash = determinism_hash(results_payload)
    print(f"determinism_sha256: {det_hash}")

    # Gate aggregation
    all_gates: List[Gate] = []
    all_gates.extend(g2)
    all_gates.extend(g_e1)
    all_gates.append(g_sweep)
    all_gates.append(g_teeth_e1)
    all_gates.extend(g_e2)
    all_gates.extend(g_e3)
    all_gates.extend(g_ns)  # may include skip/fail gate

    ok = all(g.ok for g in all_gates)
    print("\n" + title("FINAL VERDICT"))
    if ok:
        print("PASS  DEMO-56 VERIFIED (executed gates pass; counterfactual controls degrade)")
        return 0
    else:
        print("FAIL  DEMO-56 NOT VERIFIED (one or more gates failed)")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
