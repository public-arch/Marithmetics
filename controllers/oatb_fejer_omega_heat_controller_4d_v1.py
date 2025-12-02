#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fejér–Ω Heat Controller Demo — 4D Physical Model (tuned)

We simulate 4D heat diffusion on a periodic hypercube and run two fields:

  • Uncontrolled: u_t = kappa * Δu
  • Controlled:   u_t = kappa * Δu + gamma * Ω_fejer(target - u)

Where:
  - Δu is the 4D Laplacian with periodic BCs.
  - Ω_fejer is a separable 4D Fejér-based global controller in k-space.
  - We subtract the mean of Ω_fejer(target - u) so Ω does not change total heat.

Gates:
  G1: Fejér–Ω is lawful — total heat (mass) preserved (uncontrolled & controlled).
  G2: One Transfer — controlled field is ≥ 1.3× closer (L²) to target than uncontrolled
      (≥ ~30% reduction in L² error).
  G3: One Budget — HF energy of error (u - target) is ≤ 0.7× in controlled vs uncontrolled
      (≥ ~30% reduction in high-frequency error modes).

Run:
  python fejer_omega_heat4d.py
  python fejer_omega_heat4d.py --quick
"""

import sys, math, argparse
import numpy as np

G = "🟢 ✅"
R = "🔴 ❌"

# --------------------- helpers ---------------------
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

# --------------------- Fejér in 4D ---------------------
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

def apply_fejer_omega_4d(arr: np.ndarray, r: int, strength: float = 1.0) -> np.ndarray:
    """4D Fejér-based Ω controller: Ω(arr) = IFFT(H^strength * FFT(arr))."""
    H4 = fejer_mult_4d(arr.shape, r)**strength
    F = np.fft.fftn(arr)
    F *= H4
    return np.fft.ifftn(F).real

def hf_energy_4d(arr: np.ndarray, L: float, cutoff_fraction: float = 0.25) -> float:
    """High-frequency energy of a 4D field (periodic)."""
    Nx, Ny, Nz, Nw = arr.shape
    dx = L/Nx
    F = np.fft.fftn(arr)
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
    return float(np.sum(np.abs(F[mask])**2)/(Nx*Ny*Nz*Nw))

# --------------------- Heat equation in 4D ---------------------
def heat_step_4d(u: np.ndarray, dt: float, dx: float, kappa: float) -> np.ndarray:
    """Explicit 4D heat step (periodic)."""
    u_xp = np.roll(u, -1, axis=0)
    u_xm = np.roll(u,  1, axis=0)
    u_yp = np.roll(u, -1, axis=1)
    u_ym = np.roll(u,  1, axis=1)
    u_zp = np.roll(u, -1, axis=2)
    u_zm = np.roll(u,  1, axis=2)
    u_wp = np.roll(u, -1, axis=3)
    u_wm = np.roll(u,  1, axis=3)
    lap = (u_xp + u_xm + u_yp + u_ym + u_zp + u_zm + u_wp + u_wm - 8.0*u)/(dx*dx)
    return u + dt*kappa*lap

def gaussian_4d(W, X, Y, Z, mu, sigma, amp=1.0):
    R2 = (W-mu[0])**2 + (X-mu[1])**2 + (Y-mu[2])**2 + (Z-mu[3])**2
    return amp*np.exp(-0.5*R2/(sigma*sigma))

# --------------------- CLI ---------------------
def build_parser():
    p = argparse.ArgumentParser(description="Fejér–Ω Heat Controller Demo (4D)")
    p.add_argument("--quick", action="store_true", help="smaller grid, fewer steps")
    p.add_argument("--N", type=int, default=16, help="grid size per axis (N^4)")
    p.add_argument("--L", type=float, default=64.0, help="domain size per axis")
    p.add_argument("--dt", type=float, default=0.004, help="time step")
    p.add_argument("--T", type=float, default=4.0, help="total simulation time")
    p.add_argument("--kappa", type=float, default=1.0, help="diffusivity")
    p.add_argument("--r_fejer", type=int, default=4, help="Fejér span")
    p.add_argument("--gamma", type=float, default=0.5, help="Omega gain (stronger control)")
    return p

def apply_quick(args):
    if args.quick:
        args.N = 12
        args.L = 48.0
        args.dt = 0.006
        args.T  = 3.0
        args.kappa = 1.0
        args.r_fejer = 3
        args.gamma = 0.5

# --------------------- main ---------------------
def main():
    parser = build_parser()
    args = parser.parse_args()
    apply_quick(args)

    header("Fejér–Ω Heat Controller Demo — 4D Physical Model")

    N = args.N
    L = args.L
    dt = args.dt
    T  = args.T
    kappa = args.kappa
    r = args.r_fejer
    gamma = args.gamma

    dx = L/N
    steps = int(T/dt)

    print(f"Grid: N={N}^4 (= {N**4} points), L={L:.1f}, dx={dx:.3f}, dt={dt}, steps={steps}, kappa={kappa}")
    print(f"Omega: r_fejer={r}, gamma={gamma}")

    coords = np.linspace(-L/2.0, L/2.0, N, endpoint=False)
    W, X, Y, Z = np.meshgrid(coords, coords, coords, coords, indexing="ij")

    rng = np.random.default_rng(0)
    u0 = gaussian_4d(W, X, Y, Z, mu=(0.0,0.0,0.0,0.0), sigma=4.0, amp=5.0)
    u0 += 0.5 * rng.normal(size=(N,N,N,N))

    target = gaussian_4d(W, X, Y, Z, mu=(0.0,0.0,0.0,0.0), sigma=6.0, amp=3.0)

    u_un = u0.copy()
    u_ct = u0.copy()

    vol = dx**4
    mass0_un = float(np.sum(u_un)*vol)
    mass0_ct = float(np.sum(u_ct)*vol)

    for _ in range(steps):
        u_un = heat_step_4d(u_un, dt, dx, kappa)

        u_ct = heat_step_4d(u_ct, dt, dx, kappa)
        err = target - u_ct
        omega_corr = apply_fejer_omega_4d(err, r, strength=1.0)
        omega_corr -= float(np.mean(omega_corr))  # zero-mean => mass-preserving
        u_ct = u_ct + gamma*omega_corr

    mass_un_final = float(np.sum(u_un)*vol)
    mass_ct_final = float(np.sum(u_ct)*vol)
    mass_un_ok = abs(mass_un_final - mass0_un) < 1e-3*max(1.0, abs(mass0_un))
    mass_ct_ok = abs(mass_ct_final - mass0_ct) < 1e-3*max(1.0, abs(mass0_ct))

    err_un = float(np.sqrt(np.mean((u_un - target)**2)))
    err_ct = float(np.sqrt(np.mean((u_ct - target)**2)))
    improvement = err_un / max(1e-12, err_ct)
    err_gate = (improvement >= 1.3)  # ≥ 30% L2 reduction

    hf_err_un = hf_energy_4d(u_un - target, L, 0.25)
    hf_err_ct = hf_energy_4d(u_ct - target, L, 0.25)
    hf_ratio  = hf_err_ct / max(1e-16, hf_err_un)
    hf_gate   = (hf_ratio <= 0.7)  # ≥ 30% HF error reduction

    print("\nFinal diagnostics:")
    gate_line("Mass (uncontrolled) conserved",
              mass_un_ok, f"Δmass={mass_un_final - mass0_un:.3e}")
    gate_line("Mass (controlled) conserved",
              mass_ct_ok, f"Δmass={mass_ct_final - mass0_ct:.3e}")
    gate_line("Target tracking (L²): controlled < uncontrolled",
              err_gate,
              f"err_un={err_un:.3e}, err_ctrl={err_ct:.3e}, factor≈{improvement:.2f}")
    gate_line("HF error suppression (4D E_HF): controlled ≤ 0.7× uncontrolled",
              hf_gate,
              f"ratio≈{hf_ratio:.3f}, HF_err_un={hf_err_un:.3e}, HF_err_ctrl={hf_err_ct:.3e}")

    print("\nSummary:")
    gate_line("G1 (Fejér–Ω lawful: 4D mass preserved)", mass_un_ok and mass_ct_ok)
    gate_line("G2 (One Transfer: 4D profile enforced)", err_gate)
    gate_line("G3 (One Budget: 4D HF drop in error under Ω)", hf_gate)

if __name__ == "__main__":
    main()