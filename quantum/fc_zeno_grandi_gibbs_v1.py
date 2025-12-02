
#!/usr/bin/env python3
# Finite–Continuum Paradox Pack — Demo 1 (Zeno + Grandi + Gibbs) v1.1
#
# One demo, multiple “paradoxes”:
#   • Fejer legality + UFET kernel universality (K ~ 2/3)
#   • Zeno: 1/2 + 1/4 + 1/8 + ... → 1 (finite geometric sum, no paradox)
#   • Grandi series: raw divergence vs Fejer/Cesaro summation to 1/2
#   • Gibbs: hard cutoff vs Fejer smoothing on a square wave
#   • UV tail: Fejer strongly attenuates high-frequency energy vs the true square wave
#
# All gates are tuned to be scientifically honest and numerically stable.

import sys
import math
import argparse
import numpy as np

def EMJ(ok): return "🟢 ✅" if ok else "🔴 ❌"
def log(s):  print(s, flush=True)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--N_fejer", type=int, default=256, help="Grid for 1D Fejer UFET check")
    p.add_argument("--r",       type=int, default=16,  help="Fejer span (window radius)")
    p.add_argument("--Nx",      type=int, default=1024,help="Grid for square wave / Gibbs demo")
    p.add_argument("--M",       type=int, default=40,  help="Half-bandwidth for spectral truncation")
    args, _ = p.parse_known_args()
    return args

# ---------------- Fejer kernel ----------------
def fejer_symbol_1d(N, r):
    H = np.ones(N, dtype=float)
    for k in range(1, N):
        s_num = math.sin(math.pi * r * k / N)
        s_den = math.sin(math.pi * k / N)
        H[k] = (s_num / (max(1e-300, r * s_den)))**2
    return np.clip(H, 0.0, 1.0)

def hard_box_symbol_1d(N, M):
    H = np.zeros(N, dtype=float)
    k = np.fft.fftfreq(N) * N
    H[np.abs(k) <= M] = 1.0
    return H

# ---------------- Stage 1: Fejer legality + UFET constant ----------------
def stage1_fejer_ufet(N_fejer, r):
    log("\n[Stage 1] Fejer legality + UFET constant")
    H = fejer_symbol_1d(N_fejer, r)
    ok_bounds = (np.min(H) >= -1e-12) and (np.max(H) <= 1.0+1e-12)
    ok_dc     = (abs(H[0] - 1.0) < 1e-12)
    log(f"  DFT symbol: min={np.min(H):.6f}, max={np.max(H):.6f}, H[0]={H[0]:.12f}")
    log(f"  {EMJ(ok_bounds and ok_dc)}  G1: 0 <= H <= 1 and DC≈1.")

    # UFET constant K(r) = mean(H^2) * r ~ 2/3 across r
    r_vals = [max(4, r//2), r, min(N_fejer//3, max(r+4, 2*r))]
    K_vals = []
    for rv in r_vals:
        Hv = fejer_symbol_1d(N_fejer, rv)
        K_vals.append(float(np.mean(Hv**2)*rv))
    K_vals = np.array(K_vals)
    spread = (np.max(K_vals) - np.min(K_vals))/max(1e-12, np.min(K_vals))
    ok_K = (spread <= 0.06)
    log(f"  UFET K(r)=mean(H^2)*r for r={r_vals}: {list(K_vals)}")
    log(f"  spread across r in K = {100*spread:.2f}% → {EMJ(ok_K)}  (UFET kernel-universality)")
    return ok_bounds and ok_dc and ok_K

# ---------------- Stage 2: Zeno geometric series ----------------
def stage2_zeno():
    log("\n[Stage 2] Zeno geometric series: 1/2 + 1/4 + 1/8 + ... → 1")
    N_terms = 30
    powers  = np.array([0.5**(k+1) for k in range(N_terms)], dtype=float)
    partial = np.cumsum(powers)
    final   = partial[-1]
    err     = abs(1.0 - final)
    ok_geo  = (err <= 1e-9)
    log(f"  partial sum with {N_terms} terms = {final:.12f}, error vs 1 = {err:.3e}")
    log(f"  {EMJ(ok_geo)}  G2: geometric 'infinite halves' are just a finite sum with exact limit.")
    return ok_geo

# ---------------- Stage 3: Grandi series + Fejer/Cesaro ----------------
def stage3_grandi():
    log("\n[Stage 3] Grandi series: 1 - 1 + 1 - 1 + ...")
    N_terms = 101  # odd, so last partial sum is 1
    a = np.array([(-1)**k for k in range(N_terms)], dtype=float)
    partial = np.cumsum(a)

    # Raw partial sums oscillate
    osc_amp = partial.max() - partial.min()
    ok_osc  = (osc_amp >= 1.0 - 1e-12)

    # Cesaro/Fejer: average the partial sums
    cesaro = np.array([np.mean(partial[:n+1]) for n in range(N_terms)], dtype=float)
    cesaro_final = cesaro[-1]
    err_ces = abs(cesaro_final - 0.5)
    ok_ces = (err_ces <= 5e-3)

    log(f"  raw partial sums range: min={partial.min():.3f}, max={partial.max():.3f}  → {EMJ(ok_osc)} (non-convergent)")
    log(f"  Cesaro/Fejer sum after {N_terms} terms ~ {cesaro_final:.6f}, error vs 0.5 = {err_ces:.3e} → {EMJ(ok_ces)}")
    return ok_osc and ok_ces

# ---------------- Stage 4: Gibbs phenomenon (Dirichlet vs Fejer) ----------------
def stage4_gibbs(Nx, M, r):
    log("\n[Stage 4] Gibbs phenomenon: hard cutoff vs Fejer smoothing on a square wave")

    # 2π-periodic square wave: 1 on [0,π), -1 on [π,2π)
    x = np.linspace(0, 2*np.pi, Nx, endpoint=False)
    f_true = np.sign(np.sin(x)).astype(float)

    F = np.fft.fft(f_true)
    k = np.fft.fftfreq(Nx) * Nx

    # Hard cutoff (Dirichlet): spectral partial sum up to |k| <= M
    box = (np.abs(k) <= M).astype(float)
    F_dir = F * box
    f_dir = np.fft.ifft(F_dir).real

    # Fejer smoothing on full spectrum
    H_fe = fejer_symbol_1d(Nx, r)
    F_fejer = F * H_fe
    f_fejer = np.fft.ifft(F_fejer).real

    # Region around discontinuity at x=π
    idx_jump = Nx//2
    window   = 8
    sl = slice(idx_jump-window, idx_jump+window+1)
    seg_true  = f_true[sl]
    seg_dir   = f_dir[sl]
    seg_fejer = f_fejer[sl]

    # overshoot: above 1 or below -1
    over_dir_plus  = max(seg_dir.max() - 1.0, 0.0)
    over_dir_minus = max(-1.0 - seg_dir.min(), 0.0)
    over_fe_plus   = max(seg_fejer.max() - 1.0, 0.0)
    over_fe_minus  = max(-1.0 - seg_fejer.min(), 0.0)

    dir_over_max = max(over_dir_plus, over_dir_minus)
    fe_over_max  = max(over_fe_plus,  over_fe_minus)

    # Dirichlet should have visible overshoot; Fejer should reduce it
    ok_dir_has_overshoot = (dir_over_max >= 0.02)  # 2% overshoot is enough as an indicator
    ok_fe_reduces        = (fe_over_max <= 0.5*dir_over_max + 1e-12)

    log(f"  Dirichlet overshoot: +{over_dir_plus:.3f}, -{over_dir_minus:.3f} → {EMJ(ok_dir_has_overshoot)}")
    log(f"  Fejer overshoot:     +{over_fe_plus:.3f}, -{over_fe_minus:.3f} → {EMJ(ok_fe_reduces)}")

    return ok_dir_has_overshoot and ok_fe_reduces, f_true, f_dir, f_fejer

# ---------------- Stage 5: UV tail (Fejer vs true) ----------------
def stage5_uv_tail(f_true, f_dir, f_fejer):
    log("\n[Stage 5] UV tail: high-frequency energy (Fejer vs true square)")
    Nx = f_true.shape[0]
    F_true  = np.fft.fft(f_true)
    F_dir   = np.fft.fft(f_dir)
    F_fejer = np.fft.fft(f_fejer)

    k = np.fft.fftfreq(Nx) * Nx
    uv_mask = (np.abs(k) >= Nx/4)

    E_true_uv  = float(np.sum(np.abs(F_true[uv_mask])**2))
    E_dir_uv   = float(np.sum(np.abs(F_dir[uv_mask])**2))
    E_fejer_uv = float(np.sum(np.abs(F_fejer[uv_mask])**2))

    # Fejer should significantly reduce UV energy vs the true square wave.
    ok_uv = (E_fejer_uv <= 0.5 * E_true_uv + 1e-12)

    log(f"  UV energy (true):   {E_true_uv:.3e}")
    log(f"  UV energy (Dir):    {E_dir_uv:.3e}")
    log(f"  UV energy (Fejer):  {E_fejer_uv:.3e} → {EMJ(ok_uv)} (should be much smaller than true)")
    return ok_uv

# ---------------- main ----------------
def main():
    args = parse_args()
    Nf, r, Nx, M = args.N_fejer, args.r, args.Nx, args.M

    log("="*69)
    log(" Finite–Continuum Paradox Pack — Demo 1 (Zeno + Grandi + Gibbs) v1.1")
    log("="*69)
    log(f"[Setup] N_fejer={Nf}, r={r}, Nx={Nx}, M={M}")

    g1 = stage1_fejer_ufet(Nf, r)
    g2 = stage2_zeno()
    g3 = stage3_grandi()
    g4, f_true, f_dir, f_fejer = stage4_gibbs(Nx, M, r)
    g5 = stage5_uv_tail(f_true, f_dir, f_fejer)

    log("\n" + "="*69)
    log(" Demo Summary")
    log("="*69)
    log(f"  Stage 1 (Fejer legality + UFET): {EMJ(g1)}")
    log(f"  Stage 2 (Zeno geometric sum):    {EMJ(g2)}")
    log(f"  Stage 3 (Grandi + Fejer/Cesaro): {EMJ(g3)}")
    log(f"  Stage 4 (Gibbs, Fejer vs Dir):   {EMJ(g4)}")
    log(f"  Stage 5 (UV tail suppression):   {EMJ(g5)}")
    log("")

if __name__ == "__main__":
    main()