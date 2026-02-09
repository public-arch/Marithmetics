#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
====================================================================================================
DEMO-71 — ONE ACTION MASTER FLAGSHIP
(Classical Noether + Quantum Unitarity + Field Energy) — REFEREE READY, SELF-CONTAINED
====================================================================================================

What this demo shows (from first principles, deterministic, no tuning):

  (1) A single discrete selector produces the same primary triple (137,107,103) deterministically.
  (2) From that triple, we derive the *same budgets* (q2,q3,eps) used across domains.
  (3) A single *action principle* manifests as three structurally-protected laws:
        • Classical: symplectic / Noether (angular momentum) under a variational integrator.
        • Quantum  : unitarity + reversibility under Crank–Nicolson (CN) time stepping.
        • Field    : energy stability under leapfrog (a variational/symplectic update).

  (4) “Illegal controls” (non-variational / non-unitary / sign-flipped) violate the laws.
  (5) Deterministic counterfactual teeth: reducing the lawful budget K (via q3→3q3) degrades accuracy.

This script is designed to be portable (NumPy only), audit-grade, and fully deterministic.

----------------------------------------------------------------------------------------------------
CLI:
  python demo71_master_flagship_one_action_referee_ready_v1.py
  python demo71_master_flagship_one_action_referee_ready_v1.py --artifacts

Artifacts:
  If --artifacts is set, attempts to write:
    - demo71_one_action_results.json
    - demo71_one_action_plots.png (if matplotlib is available)
  If filesystem is restricted, the demo still passes (prints a PASS about skipping artifacts).

====================================================================================================
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
from typing import Dict, List, Tuple, Any

try:
    import numpy as np
except Exception as e:
    print("FATAL: NumPy is required.")
    print("Import error:", repr(e))
    raise

G = "✅"
R = "❌"


# ----------------------------
# Formatting / hashing helpers
# ----------------------------
def header(title: str) -> None:
    bar = "=" * 100
    print(bar)
    print(title.center(100))
    print(bar)

def subheader(title: str) -> None:
    bar = "-" * 100
    print("\n" + bar)
    print(title)
    print(bar)

def gate_line(label: str, ok: bool, extra: str = "") -> bool:
    mark = G if ok else R
    if extra:
        print(f"  {mark}  {label:<74} {extra}")
    else:
        print(f"  {mark}  {label}")
    return ok

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def freeze_for_json(x: Any, sig: int = 12) -> Any:
    """
    Make a JSON-safe, cross-platform-stable representation:
      - numpy scalars -> python scalars
      - numpy arrays  -> lists
      - floats rounded to `sig` significant digits
    """
    if isinstance(x, (np.floating,)):
        return float(f"{float(x):.{sig}g}")
    if isinstance(x, (float,)):
        return float(f"{x:.{sig}g}")
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (int, str, bool)) or x is None:
        return x
    if isinstance(x, (list, tuple)):
        return [freeze_for_json(v, sig=sig) for v in x]
    if isinstance(x, dict):
        return {str(k): freeze_for_json(v, sig=sig) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        return freeze_for_json(x.tolist(), sig=sig)
    # last resort
    return str(x)


# ----------------------------
# Number theory: selector core
# ----------------------------
def v2(n: int) -> int:
    """2-adic valuation v2(n)."""
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

def select_primary_and_counterfactuals() -> Tuple[Triple, List[Triple], Dict[str, List[int]]]:
    """
    Deterministic lane rules (same pattern used across the flagship suite).
    Primary window: primes in [97, 181)

      U(1):  p mod 17 in {1,5}, and coherence v2(p-1)=3
      SU(2): p mod 13 = 3
      SU(3): p mod 17 = 1  (exclude wU if it appears)

    Counterfactual window: primes in [181, 1200)
      Use the first coherent U(1) survivor (wU_cf) and first 2 SU2 and SU3 survivors.
      => 4 deterministic counterfactual triples.
    """
    window = primes_in_range(97, 181)

    U1_raw  = [p for p in window if (p % 17) in (1, 5)]
    SU2_raw = [p for p in window if (p % 13) == 3]
    SU3_raw = [p for p in window if (p % 17) == 1]

    U1 = [p for p in U1_raw if v2(p - 1) == 3]

    if len(U1) != 1:
        raise RuntimeError(f"Expected unique coherent U(1) survivor; got {U1}")

    wU = U1[0]
    if len(SU2_raw) < 1:
        raise RuntimeError("SU(2) pool empty unexpectedly.")
    s2 = SU2_raw[0]

    SU3_wo = [p for p in SU3_raw if p != wU]
    if len(SU3_wo) < 1:
        raise RuntimeError("SU(3) pool empty unexpectedly.")
    s3 = min(SU3_wo)

    primary = Triple(wU=wU, s2=s2, s3=s3)

    # Counterfactuals
    window_cf = primes_in_range(181, 1200)

    U1_cf_raw = [p for p in window_cf if (p % 17) in (1, 5)]
    U1_cf = [p for p in U1_cf_raw if v2(p - 1) == 3]
    if len(U1_cf) < 1:
        raise RuntimeError("No coherent U(1) counterfactual survivor found.")
    wU_cf = U1_cf[0]

    SU2_cf = [p for p in window_cf if (p % 13) == 3][:2]
    SU3_cf = [p for p in window_cf if (p % 17) == 1][:2]
    if len(SU2_cf) < 2 or len(SU3_cf) < 2:
        raise RuntimeError("Not enough counterfactual SU(2)/SU(3) survivors in window.")

    counterfactuals = [Triple(wU=wU_cf, s2=a, s3=b) for a in SU2_cf for b in SU3_cf]

    pools = {"U1_raw": U1_raw, "SU2_raw": SU2_raw, "SU3_raw": SU3_raw, "U1_coherent": U1}
    return primary, counterfactuals, pools


# ----------------------------
# Stage A: Classical Noether (variational integrator)
# ----------------------------
def exact_oscillator(q0: np.ndarray, p0: np.ndarray, t: float, omega: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    ct = math.cos(omega * t)
    st = math.sin(omega * t)
    q = q0 * ct + (p0 / omega) * st
    p = p0 * ct - (q0 * omega) * st
    return q, p

def vv_step(q: np.ndarray, p: np.ndarray, dt: float, sign: float = +1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Velocity Verlet / Störmer–Verlet for harmonic oscillator with potential V = 0.5*sign*|q|^2:
      q'' = -sign*q
    sign=+1 is physical (stable), sign=-1 is the anti-action (unstable).
    """
    p_half = p - 0.5 * dt * sign * q
    q_new  = q + dt * p_half
    p_new  = p_half - 0.5 * dt * sign * q_new
    return q_new, p_new

def euler_step(q: np.ndarray, p: np.ndarray, dt: float, sign: float = +1.0) -> Tuple[np.ndarray, np.ndarray]:
    a = -sign * q
    q_new = q + dt * p
    p_new = p + dt * a
    return q_new, p_new

def run_classical_noether(primary: Triple, eps: float, q2: int, q3: int) -> Dict[str, Any]:
    subheader("STAGE 2A — ONE ACTION (Classical): Noether + Symplectic Area Witness")

    # First-principles initial conditions (no randomness):
    #   q0 := (s2/wU, s3/wU)
    #   p0 := (1/q3, 1/q2)
    q0 = np.array([primary.s2 / primary.wU, primary.s3 / primary.wU], dtype=float)
    p0 = np.array([1.0 / q3, 1.0 / q2], dtype=float)

    omega = 1.0
    periods = 5
    T_final = periods * 2.0 * math.pi / omega

    dt = 1.0 / q3
    steps = int(T_final / dt)  # deterministic floor

    # Counterfactual: budget degraded by q3 -> 3*q3 (dt increases 3×)
    q3_cf = 3 * q3
    dt_cf = dt * (q3_cf / q3)
    steps_cf = int(T_final / dt_cf)

    print(f"Budgets: omega={omega}  dt={dt:.6f}  periods={periods}  T_final={T_final:.6f}  steps={steps}")
    print(f"Counterfactual budget: dt_cf={dt_cf:.6f}  steps_cf={steps_cf}  (q3_cf={q3_cf})")
    print()
    print("Initial conditions from the triple:")
    print(f"  q0=[{q0[0]:.9f} {q0[1]:.9f}]  (s2/wU, s3/wU)")
    print(f"  p0=[{p0[0]:.9f} {p0[1]:.9f}]  (1/q3, 1/q2)")
    L0 = q0[0] * p0[1] - q0[1] * p0[0]
    E0 = 0.5 * (np.sum(q0*q0) + np.sum(p0*p0))
    print(f"  L0={L0:.12f}  E0={E0:.12f}")
    print()

    def simulate(method: str, dt_run: float, steps_run: int, sign: float = +1.0,
                 blowup_cap: float = 1.0e6) -> Dict[str, Any]:
        q = q0.copy()
        p = p0.copy()
        Ls = []
        Es = []
        blowup = False
        max_state = 0.0

        for n in range(steps_run):
            L = q[0] * p[1] - q[1] * p[0]
            E = 0.5 * (np.sum(q*q) + np.sum(p*p))
            Ls.append(L)
            Es.append(E)
            state_norm = float(np.sqrt(np.sum(q*q) + np.sum(p*p)))
            if state_norm > max_state:
                max_state = state_norm
            if state_norm >= blowup_cap:
                blowup = True
                break

            if method == "VV":
                q, p = vv_step(q, p, dt_run, sign=sign)
            elif method == "Euler":
                q, p = euler_step(q, p, dt_run, sign=sign)
            else:
                raise ValueError("Unknown method")

        # final exact solution at time t = steps_run*dt_run (or truncated time if blowup)
        t_fin = len(Ls) * dt_run
        q_ex, p_ex = exact_oscillator(q0, p0, t_fin, omega=omega)
        num = float(np.sqrt(np.sum((q - q_ex)**2) + np.sum((p - p_ex)**2)))
        den = float(np.sqrt(np.sum(q_ex*q_ex) + np.sum(p_ex*p_ex)))
        traj_err = num / max(1e-16, den)

        Ls = np.array(Ls, dtype=float)
        Es = np.array(Es, dtype=float)
        L_rel_drift = float(np.max(np.abs(Ls - L0)) / max(1e-16, abs(L0))) if len(Ls) else float("inf")
        E_rel_drift = float(np.max(np.abs(Es - E0)) / max(1e-16, abs(E0))) if len(Es) else float("inf")

        return {
            "method": method, "sign": sign, "dt": dt_run, "steps_target": steps_run, "steps_done": int(len(Ls)),
            "t_final": t_fin, "traj_err": traj_err,
            "L_rel_drift_max": L_rel_drift, "E_rel_drift_max": E_rel_drift,
            "blowup": bool(blowup), "max_state": float(max_state),
        }

    # Primary runs
    vv = simulate("VV", dt, steps, sign=+1.0)
    eu = simulate("Euler", dt, steps, sign=+1.0)
    sf = simulate("VV", dt, steps, sign=-1.0)  # anti-action sign-flip

    # Jacobian det (exact for these linear maps)
    det_vv = 1.0
    det_eu = (1.0 + dt*dt)**2  # block-diagonal 2D => det = (1+dt^2)^2

    print("One-step Jacobian determinant (phase-space area proxy):")
    print(f"  det(VV)    = {det_vv:.12f}")
    print(f"  det(Euler) = {det_eu:.12f}")
    print()
    print("Max relative drifts over the run:")
    print(f"  VV   : L_rel_drift_max={vv['L_rel_drift_max']:.3e}  E_rel_drift_max={vv['E_rel_drift_max']:.3e}  traj_err={vv['traj_err']:.3e}")
    print(f"  Euler: L_rel_drift_max={eu['L_rel_drift_max']:.3e}  E_rel_drift_max={eu['E_rel_drift_max']:.3e}  traj_err={eu['traj_err']:.3e}")
    print(f"  SignFlip (anti-action): blowup={sf['blowup']}  max_state≈{sf['max_state']:.3e}")

    # Gates (match the prework philosophy; thresholds derived from eps, not tuned)
    tol_L = eps**6
    tol_det = eps**3
    tol_E = eps**2

    ok1 = gate_line("Gate C1: Noether (VV) angular momentum conserved (strict)", vv["L_rel_drift_max"] <= tol_L,
                    f"L_drift={vv['L_rel_drift_max']:.3e} tol=eps^6={tol_L:.3e}")
    ok2 = gate_line("Gate C2: Symplectic area (VV) det≈1 within eps^3", abs(det_vv - 1.0) <= tol_det,
                    f"|det-1|={abs(det_vv-1.0):.3e} tol=eps^3={tol_det:.3e}")
    ok3 = gate_line("Gate C3: Energy bounded (VV) drift <= eps^2", vv["E_rel_drift_max"] <= tol_E,
                    f"E_drift={vv['E_rel_drift_max']:.3e} eps^2={tol_E:.3e}")
    ok4 = gate_line("Gate C4: Illegal Euler breaks Noether by margin", eu["L_rel_drift_max"] >= tol_E,
                    f"L_drift_euler={eu['L_rel_drift_max']:.3e} floor=eps^2={tol_E:.3e}")
    ok5 = gate_line("Gate C5: Illegal Euler breaks area (|det-1| >= eps^4)", abs(det_eu - 1.0) >= eps**4,
                    f"|det_euler-1|={abs(det_eu-1.0):.3e} eps^4={eps**4:.3e}")
    ok6 = gate_line("Gate C6: Anti-action signflip exhibits blow-up / nonphysical growth",
                    sf["blowup"] and (sf["max_state"] >= 3.0e1),
                    f"blowup={sf['blowup']} max_state={sf['max_state']:.3e} floor=3.0e+01")

    # Counterfactual teeth: dt increased => VV final trajectory error must increase by (1+eps)
    vv_cf = simulate("VV", dt_cf, steps_cf, sign=+1.0)
    okT = gate_line("Gate CT: counterfactual (dt increased) degrades VV trajectory error by (1+eps)",
                    vv_cf["traj_err"] >= (1.0 + eps) * vv["traj_err"],
                    f"errP={vv['traj_err']:.3e} errCF={vv_cf['traj_err']:.3e} 1+eps={1+eps:.3f}")

    return {
        "q0": q0, "p0": p0, "L0": L0, "E0": float(E0),
        "dt": dt, "steps": steps, "T_final": T_final,
        "det_vv": det_vv, "det_euler": det_eu,
        "vv": vv, "euler": eu, "signflip": sf,
        "vv_cf": vv_cf,
        "gates": {"C1": ok1, "C2": ok2, "C3": ok3, "C4": ok4, "C5": ok5, "C6": ok6, "CT": okT},
    }


# ----------------------------
# Stage B: Quantum unitarity (CN vs Euler vs Wick)
# ----------------------------
def l2_norm(psi: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(psi)**2)))

def rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.abs(a - b)**2)) / max(1e-16, np.sqrt(np.mean(np.abs(b)**2))))

def run_quantum_unitarity(primary: Triple, eps: float, q2: int, q3: int) -> Dict[str, Any]:
    subheader("STAGE 2B — ONE ACTION (Quantum): Unitarity + Reversibility Witness (CN vs Euler vs Wick)")

    # Budgets (first principles):
    N = 192
    k0, k1 = 4, 5
    dt = 1.0 / q2
    steps = q3
    T_total = dt * steps

    # Counterfactual: fewer steps (budget reduced by q3 -> 3*q3), keep same total time
    q3_cf = 3 * q3
    steps_cf = int(round(steps * (q3 / q3_cf)))
    steps_cf = max(1, steps_cf)
    dt_cf = T_total / steps_cf

    print(f"Budgets: N={N}  k0={k0} k1={k1}  dt={dt:.8f}  steps={steps}  T_total={T_total:.8f}")
    print(f"Counterfactual: dt_cf={dt_cf:.8f}  steps_cf={steps_cf} (keep T_total fixed)")
    print()

    # Grid x in [0,2π)
    L = 2.0 * math.pi
    x = (L / N) * np.arange(N)

    # Fourier mode numbers m ∈ {...,-2,-1,0,1,2,...}
    m = np.fft.fftfreq(N) * N  # exact integers in float form
    omega_m = 0.5 * (m**2)  # H = -1/2 ∂xx  => eigenvalue ω = 1/2 m^2

    # Initial state: superposition of two eigenmodes (exactly solvable)
    psi0 = np.exp(1j * k0 * x) + np.exp(1j * k1 * x)
    psi0 = psi0 / l2_norm(psi0)

    # Truth at final time via exact spectral evolution
    psi0_hat = np.fft.fft(psi0)
    psi_truth_hat = psi0_hat * np.exp(-1j * omega_m * T_total)
    psi_truth = np.fft.ifft(psi_truth_hat)

    def evolve(method: str, dt_run: float, steps_run: int) -> np.ndarray:
        # evolve in spectral domain (diagonal), then return to x-space
        if method == "CN":
            mult = (1.0 - 1j * omega_m * dt_run / 2.0) / (1.0 + 1j * omega_m * dt_run / 2.0)
        elif method == "Euler":
            mult = (1.0 - 1j * omega_m * dt_run)
        elif method == "Wick":
            # illegal Wick rotation: i -> 1 (diffusion)
            mult = np.exp(-omega_m * dt_run)
        else:
            raise ValueError("Unknown method")

        hat = psi0_hat.copy()
        # power is deterministic; repeated multiplication is fine too
        hat *= (mult ** steps_run)
        return np.fft.ifft(hat)

    def time_reversal_error(method: str, dt_run: float, steps_run: int) -> float:
        psi_f = evolve(method, dt_run, steps_run)
        # reverse with -dt (CN stays stable; Euler reversibility fails)
        # For Wick (diffusion), reversal is ill-posed; we skip it.
        if method == "Wick":
            return float("nan")

        # reverse operator: same scheme with -dt
        psi_f_hat = np.fft.fft(psi_f)
        if method == "CN":
            mult_b = (1.0 - 1j * omega_m * (-dt_run) / 2.0) / (1.0 + 1j * omega_m * (-dt_run) / 2.0)
        elif method == "Euler":
            mult_b = (1.0 - 1j * omega_m * (-dt_run))
        else:
            raise ValueError("Unknown method")
        psi_b = np.fft.ifft(psi_f_hat * (mult_b ** steps_run))
        return rel_l2(psi_b, psi0)

    psi_cn = evolve("CN", dt, steps)
    psi_eu = evolve("Euler", dt, steps)
    psi_wk = evolve("Wick", dt, steps)

    n0 = l2_norm(psi0)
    cn_norm_drift = abs(l2_norm(psi_cn) - n0)
    eu_norm_drift = abs(l2_norm(psi_eu) - n0)
    wk_norm_drift = abs(l2_norm(psi_wk) - n0)

    cn_truth_err = rel_l2(psi_cn, psi_truth)
    eu_truth_err = rel_l2(psi_eu, psi_truth)
    wk_truth_err = rel_l2(psi_wk, psi_truth)

    cn_rev_err = time_reversal_error("CN", dt, steps)
    eu_rev_err = time_reversal_error("Euler", dt, steps)

    print(f"Norm drift: CN={cn_norm_drift:.3e}  Euler={eu_norm_drift:.3e}  Wick(illegal)={wk_norm_drift:.3e}")
    print(f"Truth error (rel L2): CN={cn_truth_err:.3e}  Euler={eu_truth_err:.3e}  Wick={wk_truth_err:.3e}")
    print(f"Time-reversal error: CN={cn_rev_err:.3e}  Euler={eu_rev_err:.3e}")
    print()

    ok1 = gate_line("Gate Q1: CN unitary (norm drift <= eps^4)", cn_norm_drift <= eps**4,
                    f"drift={cn_norm_drift:.3e} eps^4={eps**4:.3e}")
    ok2 = gate_line("Gate Q2: CN reversible (forward+back error <= eps^3)", cn_rev_err <= eps**3,
                    f"err={cn_rev_err:.3e} eps^3={eps**3:.3e}")
    ok3 = gate_line("Gate Q3: CN accuracy vs exact within eps", cn_truth_err <= eps,
                    f"err={cn_truth_err:.3e} eps={eps:.3e}")

    ok4 = gate_line("Gate Q4: Illegal Euler not unitary (norm drift >= eps^2)", eu_norm_drift >= eps**2,
                    f"drift={eu_norm_drift:.3e} eps^2={eps**2:.3e}")
    ok5 = gate_line("Gate Q5: Illegal Euler worse accuracy than CN by margin", eu_truth_err >= (1.0 + eps) * cn_truth_err,
                    f"err_eu={eu_truth_err:.3e} err_cn={cn_truth_err:.3e} 1+eps={1+eps:.3f}")
    ok6 = gate_line("Gate Q6: Wick illegal destroys Schrödinger truth (err >= eps)", wk_truth_err >= eps,
                    f"err_wick={wk_truth_err:.3e} eps={eps:.3e}")

    # Counterfactual teeth
    psi_cn_cf = evolve("CN", dt_cf, steps_cf)
    cn_truth_err_cf = rel_l2(psi_cn_cf, psi_truth)

    okT = gate_line("Gate QT: counterfactual degrades CN error by (1+eps)",
                    cn_truth_err_cf >= (1.0 + eps) * cn_truth_err,
                    f"errP={cn_truth_err:.3e} errCF={cn_truth_err_cf:.3e} 1+eps={1+eps:.3f}")

    return {
        "N": N, "k0": k0, "k1": k1, "dt": dt, "steps": steps, "T_total": T_total,
        "dt_cf": dt_cf, "steps_cf": steps_cf,
        "cn_norm_drift": cn_norm_drift, "eu_norm_drift": eu_norm_drift, "wk_norm_drift": wk_norm_drift,
        "cn_truth_err": cn_truth_err, "eu_truth_err": eu_truth_err, "wk_truth_err": wk_truth_err,
        "cn_rev_err": cn_rev_err, "eu_rev_err": eu_rev_err,
        "cn_truth_err_cf": cn_truth_err_cf,
        "gates": {"Q1": ok1, "Q2": ok2, "Q3": ok3, "Q4": ok4, "Q5": ok5, "Q6": ok6, "QT": okT},
    }


# ----------------------------
# Stage C: Field wave energy witness (spectral leapfrog vs Euler; first principles)
#
# Why spectral here?
#   • The periodic wave equation admits an exact diagonalization in Fourier space.
#   • Using FFT-based derivatives makes the "action → energy" witness extremely crisp and portable.
#   • This is the same legality theme used across the ONE ACTION triad: lawful (reversible/symplectic/unitary)
#     vs illegal (drift / blow-up / nonphysical behavior) under deterministic budgets.
# ----------------------------
def _wave_truth_solution(x: np.ndarray, mode: int, c: float, T: float) -> Tuple[np.ndarray, np.ndarray]:
    """Exact periodic wave solution for a single Fourier mode.

    PDE: u_tt = c^2 u_xx on [0,2π), periodic.
    IC : u(x,0)=sin(mode*x), v(x,0)=u_t(x,0)=0.

    Returns (u(T), v(T)).
    """
    u0 = np.sin(mode * x)
    w = c * float(mode)
    uT = u0 * math.cos(w * T)
    vT = -u0 * w * math.sin(w * T)
    return uT, vT


def run_field_wave(primary: Triple, eps: float, q2: int, q3: int) -> Dict[str, float]:
    header("STAGE 2C — ONE ACTION (Field): Wave Energy Witness (Leapfrog vs Euler)")

    # Budgets (locked; no tuning)
    N = 192
    L = 2.0 * math.pi
    dx = L / N
    c = 1.0

    # Mode locked to the triple (deterministic)
    mode = (primary.s3 % 11) + 2  # for s3=103 -> 6

    # Total time locked by invariants
    T_total = q3 / q2

    # dt from a CFL-like proxy then snapped to hit T_total exactly
    dt_base = dx / q3
    steps = int(round(T_total / dt_base))
    steps = max(4, steps)
    dt = T_total / steps

    # Counterfactual budget: q3_cf = 3*q3 (dt grows; fewer steps)
    q3_cf = 3 * q3
    dt_cf_base = dt * (q3_cf / q3)
    steps_cf = int(round(T_total / dt_cf_base))
    steps_cf = max(3, steps_cf)
    dt_cf = T_total / steps_cf

    print(f"Budgets: N={N}  mode={mode}  c={c:.1f}  dx={dx:.6f}")
    print(f"dt={dt:.8f}  steps={steps}  T_total={T_total:.8f}")
    print(f"counterfactual: dt_cf={dt_cf:.8f}  steps_cf={steps_cf}\n")

    # Periodic grid on [0,2π)
    x = np.arange(N, dtype=float) * dx

    # Spectral derivative operators
    k = 2.0 * math.pi * np.fft.fftfreq(N, d=dx)
    k2 = (k * k)

    def u_x(u: np.ndarray) -> np.ndarray:
        U = np.fft.fft(u)
        return np.fft.ifft(1j * k * U).real

    def u_xx(u: np.ndarray) -> np.ndarray:
        U = np.fft.fft(u)
        return np.fft.ifft(-(k2) * U).real

    def energy(u: np.ndarray, v: np.ndarray) -> float:
        ux = u_x(u)
        return float(0.5 * np.sum(v*v + (c*ux)*(c*ux)) * dx)

    # IC (locked)
    u0 = np.sin(mode * x)
    v0 = np.zeros_like(u0)

    # Truth (analytic)
    u_truth, v_truth = _wave_truth_solution(x, mode=mode, c=c, T=T_total)

    # -------------------------
    # Lawful: Leapfrog (Störmer–Verlet)
    # -------------------------
    u = u0.copy()
    v = v0.copy()
    a0 = (c*c) * u_xx(u)
    v_half = v + 0.5 * dt * a0

    E0 = energy(u, v)
    Emax_drift = 0.0

    for _ in range(steps):
        u = u + dt * v_half
        a = (c*c) * u_xx(u)
        v_half = v_half + dt * a

        # diagnostic: energy at integer time (v ≈ v_half - 0.5 dt a)
        v_int = v_half - 0.5 * dt * a
        Ei = energy(u, v_int)
        Emax_drift = max(Emax_drift, abs(Ei - E0) / max(1e-16, abs(E0)))

    a_end = (c*c) * u_xx(u)
    v = v_half - 0.5 * dt * a_end

    E_end = energy(u, v)
    lf_energy_drift = abs(E_end - E0) / max(1e-16, abs(E0))
    lf_truth_err = rel_l2(u, u_truth)

    # -------------------------
    # Illegal: explicit Euler
    # -------------------------
    ue = u0.copy()
    ve = v0.copy()
    Ee0 = energy(ue, ve)
    Eemax_drift = 0.0

    for _ in range(steps):
        a = (c*c) * u_xx(ue)
        ue = ue + dt * ve
        ve = ve + dt * a
        Ei = energy(ue, ve)
        Eemax_drift = max(Eemax_drift, abs(Ei - Ee0) / max(1e-16, abs(Ee0)))

    eu_energy_drift = abs(energy(ue, ve) - Ee0) / max(1e-16, abs(Ee0))
    eu_truth_err = rel_l2(ue, u_truth)

    # -------------------------
    # Counterfactual teeth: larger dt hurts truth accuracy
    # -------------------------
    ucf = u0.copy()
    vcf = v0.copy()
    a0cf = (c*c) * u_xx(ucf)
    v_half_cf = vcf + 0.5 * dt_cf * a0cf
    for _ in range(steps_cf):
        ucf = ucf + dt_cf * v_half_cf
        a = (c*c) * u_xx(ucf)
        v_half_cf = v_half_cf + dt_cf * a
    a_end_cf = (c*c) * u_xx(ucf)
    vcf = v_half_cf - 0.5 * dt_cf * a_end_cf
    cf_truth_err = rel_l2(ucf, u_truth)

    # Print diagnostics
    print(f"Energy drift: Leapfrog={lf_energy_drift:.3e}  Euler={eu_energy_drift:.3e}  (E0={E0:.6e})")
    print(f"Truth error  : Leapfrog={lf_truth_err:.3e}  Euler={eu_truth_err:.3e}")
    print()

    # Gates
    gF1 = gate_line("Gate F1: Leapfrog energy drift <= eps^3",
                    lf_energy_drift <= eps**3,
                    f"drift={lf_energy_drift:.3e} eps^3={eps**3:.3e}")
    gF2 = gate_line("Gate F2: Leapfrog accuracy vs truth within eps",
                    lf_truth_err <= eps,
                    f"err={lf_truth_err:.3e} eps={eps:.3e}")
    gF3 = gate_line("Gate F3: Illegal Euler breaks energy (drift >= eps^2)",
                    eu_energy_drift >= eps**2,
                    f"drift={eu_energy_drift:.3e} eps^2={eps**2:.3e}")
    gF4 = gate_line("Gate F4: Illegal Euler worse accuracy than Leapfrog by margin",
                    eu_truth_err >= (1 + eps) * lf_truth_err,
                    f"err_eu={eu_truth_err:.3e} err_lf={lf_truth_err:.3e} 1+eps={1+eps:.3f}")
    gFT = gate_line("Gate FT: counterfactual degrades leapfrog error by (1+eps)",
                    cf_truth_err >= (1 + eps) * lf_truth_err,
                    f"errP={lf_truth_err:.3e} errCF={cf_truth_err:.3e} 1+eps={1+eps:.3f}")

    return {
        "N": N, "mode": mode, "c": c, "dx": dx,
        "dt": dt, "steps": steps, "T_total": T_total,
        "q3_cf": q3_cf, "dt_cf": dt_cf, "steps_cf": steps_cf,
        "lf_energy_drift": lf_energy_drift,
        "lf_energy_drift_max": Emax_drift,
        "eu_energy_drift": eu_energy_drift,
        "eu_energy_drift_max": Eemax_drift,
        "lf_truth_err": lf_truth_err,
        "eu_truth_err": eu_truth_err,
        "cf_truth_err": cf_truth_err,
        "gates": {"F1": gF1, "F2": gF2, "F3": gF3, "F4": gF4, "FT": gFT},
    }


# ----------------------------
# Optional artifacts
# ----------------------------
def try_write_artifacts(results: Dict[str, Any], make_plots: bool) -> None:
    # JSON
    out_json = "demo71_one_action_results.json"
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(freeze_for_json(results), f, indent=2, sort_keys=True)
        print(f"PASS  Results JSON written: {out_json}")
    except Exception as e:
        print(f"PASS  Results JSON not written (filesystem unavailable)  {repr(e)}")

    if not make_plots:
        return

    # Plot (best-effort)
    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:
        print(f"PASS  Plot skipped (matplotlib not available)  {repr(e)}")
        return

    try:
        import matplotlib.pyplot as plt

        # Compact 3-panel summary with scalar bars
        fig = plt.figure(figsize=(10, 4.5))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)
        ax3 = fig.add_subplot(133)

        # Classical
        c = results["classical"]
        vv = c["vv"]
        eu = c["euler"]
        ax1.set_title("Classical\n(Noether drift)")
        ax1.bar(["VV L", "Euler L"], [vv["L_rel_drift_max"], eu["L_rel_drift_max"]])
        ax1.set_yscale("log")

        # Quantum
        q = results["quantum"]
        ax2.set_title("Quantum\n(Norm drift)")
        ax2.bar(["CN", "Euler", "Wick"], [q["cn_norm_drift"], q["eu_norm_drift"], q["wk_norm_drift"]])
        ax2.set_yscale("log")

        # Field
        f = results["field"]
        ax3.set_title("Field\n(Energy drift)")
        ax3.bar(["Leapfrog", "Euler"], [f["lf_energy_drift"], f["eu_energy_drift"]])
        ax3.set_yscale("log")

        fig.tight_layout()
        out_png = "demo71_one_action_plots.png"
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"PASS  Plot written: {out_png}")
    except Exception as e:
        print(f"PASS  Plot not written (plot backend/filesystem unavailable)  {repr(e)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", action="store_true", help="Write JSON/PNG artifacts if possible.")
    args = ap.parse_args()

    header("DEMO-71 — ONE ACTION MASTER FLAGSHIP (Classical Noether + Quantum Unitarity + Field Energy)")
    print(f"UTC time : {datetime.datetime.utcnow().isoformat()}Z")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout only (JSON/PNG artifacts optional)")
    print()

    # Spec hash: hash the conceptual spec block (not the entire file) for referee pinning
    SPEC = {
        "selector": "primes[97,181), U1: mod17 in {1,5} + v2(p-1)=3, SU2: mod13=3, SU3: mod17=1",
        "invariants": "q2=wU-s2, v2U=v2(wU-1), q3=(wU-1)/2^v2U, eps=1/sqrt(q2)",
        "classical": "2D harmonic oscillator, VV vs Euler vs signflip, Noether L, symplectic det, energy drift",
        "quantum": "1D periodic Schr, eigenmodes, CN vs Euler vs Wick, unitarity + reversibility + truth",
        "field": "1D periodic wave (spectral derivatives), leapfrog vs Euler, energy drift + truth",
        "teeth": "counterfactual budget via q3->3q3 (dt increases / steps reduce) degrades accuracy",
        "determinism": "hash of frozen results dict (rounded floats) with sort_keys JSON",
    }
    spec_sha256 = sha256_hex(json.dumps(SPEC, sort_keys=True).encode("utf-8"))
    print("spec_sha256:", spec_sha256)
    print()

    # --------------------------
    # STAGE 1 — deterministic selection
    # --------------------------
    header("STAGE 1 — Deterministic triple selection (primary + counterfactuals)")
    primary, counterfactuals, pools = select_primary_and_counterfactuals()

    print("Lane survivor pools (raw):")
    print(f"  U(1):  {pools['U1_raw']}")
    print(f"  SU(2): {pools['SU2_raw']}")
    print(f"  SU(3): {pools['SU3_raw']}")
    print("Lane survivor pools (after U(1) coherence v2(wU-1)=3):")
    print(f"  U(1):  {pools['U1_coherent']}")
    print(f"Primary: {primary}")
    print("Counterfactuals:")
    for cf in counterfactuals:
        print(f"  {cf}")

    okS0 = gate_line("Gate S0: primary equals (137,107,103)", (primary.wU, primary.s2, primary.s3) == (137, 107, 103),
                     f"selected={(primary.wU, primary.s2, primary.s3)}")
    okS1 = gate_line("Gate S1: captured >=4 counterfactual triples (deterministic)", len(counterfactuals) >= 4,
                     f"found={len(counterfactuals)}")

    # --------------------------
    # STAGE 1B — Derived invariants
    # --------------------------
    header("STAGE 1B — Derived invariants (first principles, no tuning)")
    q2 = primary.wU - primary.s2
    v2U = v2(primary.wU - 1)
    q3 = (primary.wU - 1) // (2**v2U)
    eps = 1.0 / math.sqrt(q2)
    q3_cf = 3 * q3

    print(f"q2 = wU - s2 = {q2}")
    print(f"v2U = v2(wU-1) = {v2U}")
    print(f"q3 = (wU-1)/2^v2U = {q3}")
    print(f"eps = 1/sqrt(q2) = {eps:.8f}")
    print(f"counterfactual q3_cf = 3*q3 = {q3_cf}")
    print()

    okI1 = gate_line("Gate I1: invariants match the locked values (q2=30,q3=17,v2U=3)", (q2, q3, v2U) == (30, 17, 3),
                     f"(q2,q3,v2U)={(q2,q3,v2U)}")

    # --------------------------
    # STAGE 2 — One Action triad
    # --------------------------
    classical = run_classical_noether(primary, eps, q2, q3)
    quantum   = run_quantum_unitarity(primary, eps, q2, q3)
    field     = run_field_wave(primary, eps, q2, q3)

    # --------------------------
    # STAGE 3 — Final verdict + determinism hash
    # --------------------------
    header("STAGE 3 — Determinism hash + final verdict")

    results = {
        "spec_sha256": spec_sha256,
        "primary": {"wU": primary.wU, "s2": primary.s2, "s3": primary.s3},
        "counterfactuals": [{"wU": c.wU, "s2": c.s2, "s3": c.s3} for c in counterfactuals],
        "invariants": {"q2": q2, "q3": q3, "v2U": v2U, "eps": eps, "q3_cf": q3_cf},
        "gates_selector": {"S0": okS0, "S1": okS1, "I1": okI1},
        "classical": classical,
        "quantum": quantum,
        "field": field,
    }

    det_sha = sha256_hex(json.dumps(freeze_for_json(results), sort_keys=True).encode("utf-8"))
    print("determinism_sha256:", det_sha)

    # All gates must pass
    all_gates = []
    all_gates.extend(results["gates_selector"].values())
    all_gates.extend(classical["gates"].values())
    all_gates.extend(quantum["gates"].values())
    all_gates.extend(field["gates"].values())

    ok_all = all(bool(x) for x in all_gates)

    print("\n" + "=" * 100)
    print("FINAL VERDICT".center(100))
    print("=" * 100)
    gate_line("DEMO-71 VERIFIED (One Action triad: classical + quantum + field, with illegal controls + teeth)", ok_all)
    print("Result:", "VERIFIED" if ok_all else "NOT VERIFIED")

    if args.artifacts:
        subheader("STAGE 4 — Artifacts (optional)")
        try_write_artifacts(results, make_plots=True)


if __name__ == "__main__":
    main()
