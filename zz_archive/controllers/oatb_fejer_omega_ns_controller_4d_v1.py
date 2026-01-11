#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fejér–Ω NS-like Controller Demo — 4D Vector Field (tuned thresholds)

We simulate a 4D NS-like velocity field u(x,t) on a periodic hypercube and run:

  • Uncontrolled:
      u_t = nu * Δu - (u · ∇)u

  • Controlled:
      u_t = nu * Δu - (u · ∇)u + gamma * Ω_fejer(u_target - u)

Where:
  - Δu is the 4D Laplacian with periodic BCs.
  - (u · ∇)u is an NS-like convective term.
  - Ω_fejer is a separable 4D Fejér-based global controller applied component-wise.
  - We subtract the mean of Ω_fejer(error) per component to avoid adding net momentum.

Target:
  u_target = 0 (we damp the flow toward rest).

Gates:
  G1: Incompressibility: controlled divergence RMS ≤ 0.7× uncontrolled (≥ 30% improvement).
  G2: Energy damping:    controlled KE ≤ 0.7× uncontrolled (≥ 30% drop).
  G3: HF KE damping:     controlled HF energy ≤ 0.9× uncontrolled (≥ 10% drop).

Run:
  python fejer_omega_ns4d.py
  python fejer_omega_ns4d.py --quick
"""

import sys, math, argparse
import numpy as np

G = "🟢 ✅"
R = "🔴 ❌"

# ------------------------------------------------------
# Helpers
# ------------------------------------------------------
def header(title: str):
    bar = "=" * 70
    print(bar)
    print(f" {title}")
    print(bar)

def gate_line(label: str, ok: bool, extra: str = ""):
    mark = G if ok else R
    if extra:
        print(f"  {mark}  {label} {extra}")
    else:
        print(f"  {mark}  {label}")

# ------------------------------------------------------
# Fejér in 4D
# ------------------------------------------------------
def fejer_mult_1d(N: int, r: int) -> np.ndarray:
    """1D Fejér multipliers on integer frequencies m in [-N/2..N/2-1]."""
    freqs = np.fft.fftfreq(N) * N
    H = np.zeros(N, dtype=float)
    mask = np.abs(freqs) <= r
    H[mask] = 1.0 - np.abs(freqs[mask])/(r+1.0)
    return H

def fejer_mult_4d(shape, r: int) -> np.ndarray:
    """Separable 4D Fejér kernel H4D = Hx * Hy * Hz * Hw."""
    Nx, Ny, Nz, Nw = shape
    Hx = fejer_mult_1d(Nx, r).reshape(Nx, 1, 1, 1)
    Hy = fejer_mult_1d(Ny, r).reshape(1, Ny, 1, 1)
    Hz = fejer_mult_1d(Nz, r).reshape(1, 1, Nz, 1)
    Hw = fejer_mult_1d(Nw, r).reshape(1, 1, 1, Nw)
    return Hx * Hy * Hz * Hw

def apply_fejer_omega_4d_vec(u: np.ndarray, r: int, strength: float = 1.0) -> np.ndarray:
    """
    Apply Fejér Ω to a 4D vector field u with shape (4, N, N, N, N):
      Ω(u) = IFFT(H^strength * FFT(u)) component-wise.
    """
    out = np.empty_like(u)
    H4 = fejer_mult_4d(u.shape[1:], r)**strength
    for c in range(4):
        F = np.fft.fftn(u[c])
        F *= H4
        out[c] = np.fft.ifftn(F).real
    return out

def hf_energy_4d_vec(u: np.ndarray, L: float, cutoff_fraction: float = 0.25) -> float:
    """
    High-frequency kinetic energy of a 4D vector field u (4,N,N,N,N):
      E_HF = sum_{|k| >= kcut} |u_hat(k)|^2 / N^4, summed over components.
    """
    _, Nx, Ny, Nz, Nw = u.shape
    dx = L/Nx
    kx = 2.0*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2.0*np.pi*np.fft.fftfreq(Ny, d=dx)
    kz = 2.0*np.pi*np.fft.fftfreq(Nz, d=dx)
    kw = 2.0*np.pi*np.fft.fftfreq(Nw, d=dx)
    KX, KY, KZ, KW = np.meshgrid(kx, ky, kz, kw, indexing="ij")
    kmag = np.sqrt(KX*KX + KY*KY + KZ*KZ + KW*KW)
    kmax = float(np.max(kmag))
    if kmax <= 0.0:
        return 0.0
    kcut = cutoff_fraction*kmax
    mask = kmag >= kcut

    E = 0.0
    for c in range(4):
        F = np.fft.fftn(u[c])
        E += np.sum(np.abs(F[mask])**2)
    return float(E/(Nx*Ny*Nz*Nw))

# ------------------------------------------------------
# NS-like operators in 4D
# ------------------------------------------------------
def grad_4d_scalar(f: np.ndarray, dx: float) -> np.ndarray:
    """
    Gradient of scalar f on a 4D periodic grid:
      returns array of shape (4, N,N,N,N).
    Uses centered differences.
    """
    grad = np.empty((4,) + f.shape, dtype=float)
    # axis 0
    f_ip = np.roll(f, -1, axis=0)
    f_im = np.roll(f,  1, axis=0)
    grad[0] = (f_ip - f_im)/(2.0*dx)
    # axis 1
    f_ip = np.roll(f, -1, axis=1)
    f_im = np.roll(f,  1, axis=1)
    grad[1] = (f_ip - f_im)/(2.0*dx)
    # axis 2
    f_ip = np.roll(f, -1, axis=2)
    f_im = np.roll(f,  1, axis=2)
    grad[2] = (f_ip - f_im)/(2.0*dx)
    # axis 3
    f_ip = np.roll(f, -1, axis=3)
    f_im = np.roll(f,  1, axis=3)
    grad[3] = (f_ip - f_im)/(2.0*dx)
    return grad

def div_4d_vec(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Divergence of vector field u (shape: 4 x N x N x N x N):
      div u = sum_i ∂_i u_i.
    Returns scalar field (N,N,N,N).
    """
    div = np.zeros_like(u[0])
    for i in range(4):
        f = u[i]
        f_ip = np.roll(f, -1, axis=i)
        f_im = np.roll(f,  1, axis=i)
        div += (f_ip - f_im)/(2.0*dx)
    return div

def laplacian_4d_vec(u: np.ndarray, dx: float) -> np.ndarray:
    """
    Laplacian of vector field u, component-wise.
    """
    out = np.empty_like(u)
    for c in range(4):
        f = u[c]
        f_xp = np.roll(f, -1, axis=0)
        f_xm = np.roll(f,  1, axis=0)
        f_yp = np.roll(f, -1, axis=1)
        f_ym = np.roll(f,  1, axis=1)
        f_zp = np.roll(f, -1, axis=2)
        f_zm = np.roll(f,  1, axis=2)
        f_wp = np.roll(f, -1, axis=3)
        f_wm = np.roll(f,  1, axis=3)
        lap = (f_xp + f_xm + f_yp + f_ym + f_zp + f_zm + f_wp + f_wm - 8.0*f)/(dx*dx)
        out[c] = lap
    return out

def convective_term_4d(u: np.ndarray, dx: float) -> np.ndarray:
    """
    NS-like convective term (u · ∇)u for a 4D vector field u (4,N,N,N,N).
    """
    _, Nx, Ny, Nz, Nw = u.shape
    conv = np.zeros_like(u)
    for c in range(4):
        grad_uc = grad_4d_scalar(u[c], dx)  # shape (4,N,N,N,N)
        conv_c = np.zeros((Nx,Ny,Nz,Nw), dtype=float)
        for i in range(4):
            conv_c += u[i] * grad_uc[i]
        conv[c] = conv_c
    return conv

# ------------------------------------------------------
# CLI
# ------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description="Fejér–Ω NS-like Controller Demo (4D)")
    p.add_argument("--quick", action="store_true", help="smaller grid, fewer steps")
    p.add_argument("--N", type=int, default=10, help="grid size per axis (N^4)")
    p.add_argument("--L", type=float, default=40.0, help="domain size per axis")
    p.add_argument("--dt", type=float, default=0.002, help="time step")
    p.add_argument("--T", type=float, default=2.0, help="total sim time")
    p.add_argument("--nu", type=float, default=0.5, help="viscosity")
    p.add_argument("--r_fejer", type=int, default=3, help="Fejér span")
    p.add_argument("--gamma", type=float, default=0.5, help="Omega gain")
    return p

def apply_quick(args):
    if args.quick:
        args.N = 8
        args.L = 32.0
        args.dt = 0.002
        args.T  = 1.5
        args.nu = 0.6
        args.r_fejer = 3
        args.gamma = 0.6

# ------------------------------------------------------
# main
# ------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_quick(args)

    header("Fejér–Ω NS-like Controller Demo — 4D Vector Field")

    N = args.N
    L = args.L
    dt = args.dt
    T  = args.T
    nu = args.nu
    r  = args.r_fejer
    gamma = args.gamma

    dx = L/N
    steps = int(T/dt)
    print(f"Grid: N={N}^4 (= {N**4} points), L={L:.1f}, dx={dx:.3f}, dt={dt}, steps={steps}, nu={nu}")
    print(f"Omega: r_fejer={r}, gamma={gamma}")

    coords = np.linspace(-L/2.0, L/2.0, N, endpoint=False)
    W, X, Y, Z = np.meshgrid(coords, coords, coords, coords, indexing="ij")

    rng = np.random.default_rng(0)

    # Initial velocity: swirl-like structure plus noise.
    u0 = np.zeros((4, N, N, N, N), dtype=float)
    u0[0] = np.sin(Y/L*2*np.pi) + 0.2*rng.normal(size=(N,N,N,N))
    u0[1] = np.sin(Z/L*2*np.pi) + 0.2*rng.normal(size=(N,N,N,N))
    u0[2] = np.sin(W/L*2*np.pi) + 0.2*rng.normal(size=(N,N,N,N))
    u0[3] = np.sin(X/L*2*np.pi) + 0.2*rng.normal(size=(N,N,N,N))

    u_target = np.zeros_like(u0)  # drive to rest

    u_un = u0.copy()
    u_ct = u0.copy()

    for _ in range(steps):
        # Uncontrolled NS-like step
        lap_un  = laplacian_4d_vec(u_un, dx)
        conv_un = convective_term_4d(u_un, dx)
        u_un = u_un + dt*(nu*lap_un - conv_un)

        # Controlled NS-like step
        lap_ct  = laplacian_4d_vec(u_ct, dx)
        conv_ct = convective_term_4d(u_ct, dx)
        u_ct = u_ct + dt*(nu*lap_ct - conv_ct)

        # Fejér–Ω control toward u_target (0)
        err = u_target - u_ct
        omega_corr = apply_fejer_omega_4d_vec(err, r, strength=1.0)
        # enforce zero mean per component
        for c in range(4):
            omega_corr[c] -= float(np.mean(omega_corr[c]))
        u_ct = u_ct + gamma*omega_corr

    # Diagnostics
    # Divergence RMS:
    div_un = div_4d_vec(u_un, dx)
    div_ct = div_4d_vec(u_ct, dx)
    div_un_rms = float(np.sqrt(np.mean(div_un**2)))
    div_ct_rms = float(np.sqrt(np.mean(div_ct**2)))
    div_ratio  = div_ct_rms / max(1e-16, div_un_rms)
    div_gate   = (div_ratio <= 0.7)  # ≥ 30% improvement

    # Kinetic energies:
    ke_un = float(np.mean(np.sum(u_un*u_un, axis=0)))
    ke_ct = float(np.mean(np.sum(u_ct*u_ct, axis=0)))
    ke_ratio = ke_ct / max(1e-16, ke_un)
    ke_gate  = (ke_ratio <= 0.7)

    # HF kinetic energies:
    hf_un = hf_energy_4d_vec(u_un, L, cutoff_fraction=0.25)
    hf_ct = hf_energy_4d_vec(u_ct, L, cutoff_fraction=0.25)
    hf_ratio = hf_ct / max(1e-16, hf_un)
    hf_gate  = (hf_ratio <= 0.9)  # ≥ 10% HF KE reduction

    print("\nFinal diagnostics:")
    gate_line("Divergence RMS (uncontrolled)", True, f"rms={div_un_rms:.3e}")
    gate_line("Divergence RMS (controlled)",   True, f"rms={div_ct_rms:.3e}")
    gate_line("G1 (incompressibility improved: div_ctrl <= 0.7× div_un)",
              div_gate, f"ratio≈{div_ratio:.3f}")
    gate_line("Kinetic energy (uncontrolled)", True, f"KE_un={ke_un:.3e}")
    gate_line("Kinetic energy (controlled)",   True, f"KE_ctrl={ke_ct:.3e}")
    gate_line("G2 (energy damped: KE_ctrl <= 0.7× KE_un)",
              ke_gate, f"ratio≈{ke_ratio:.3f}")
    gate_line("G3 (HF KE damped: HF_ctrl <= 0.9× HF_un)",
              hf_gate, f"ratio≈{hf_ratio:.3f}, HF_un={hf_un:.3e}, HF_ctrl={hf_ct:.3e}")

    print("\nSummary:")
    gate_line("G1 (Fejér–Ω improves incompressibility)", div_gate)
    gate_line("G2 (One Transfer: turbulence energy damped)", ke_gate)
    gate_line("G3 (One Budget: HF KE drop under Ω)", hf_gate)

if __name__ == "__main__":
    main()