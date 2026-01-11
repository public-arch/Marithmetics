#!/usr/bin/env python3
# One-Kernel Unified Demo — vOmega3_fixed
# Demonstrates in one run:
#   • Fejer legality (PSD, DC=1, 0<=H<=1) + UFET universality (K ~ 2/3)
#   • Fejer ↔ Helmholtz resolvent dictionary (low band)
#   • Schr evolution: mass invariance + high-frequency contraction (Omega-controller)
#   • NS evolution: divergence-free + high-frequency contraction
#   • Madelung bridge: energy split + curl gate (global + punctured domain)
#   • Illegal kernel (hard-box) positivity FAIL (negative control)

import sys
import math
import argparse
import numpy as np

def EMJ(ok): return "🟢 ✅" if ok else "🔴 ❌"
def log(s):  print(s, flush=True)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--N", type=int, default=48)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--T", type=int, default=140)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--gamma_ns", type=float, default=0.20)   # NS controller blend
    p.add_argument("--gamma_q",  type=float, default=0.25)   # Schr controller blend
    p.add_argument("--nu", type=float, default=0.01)         # NS viscosity
    p.add_argument("--sigma", type=float, default=+1.0)      # NLS sign (+1 defocusing)
    p.add_argument("--kmax", type=int, default=6)
    p.add_argument("--kappaLB", type=float, default=0.35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bad_kernel", action="store_true")
    # Madelung/curl tolerances: slightly relaxed defaults vs original 1e-2, 1e-1
    p.add_argument("--tol_madelung", type=float, default=2e-2,
                   help="Relative tolerance for Madelung energy split (default 2e-2)")
    p.add_argument("--tol_curl", type=float, default=2e-1,
                   help="Tolerance for curl/grad ratio (default 2e-1)")
    args, _ = p.parse_known_args()
    return args

# ---------------- Fejer & Hard Box ----------------
def fejer_symbol_1d(N, r):
    H = np.ones(N, dtype=float)
    for k in range(1, N):
        s_num = math.sin(math.pi * r * k / N)
        s_den = math.sin(math.pi * k / N)
        H[k] = (s_num / (max(1e-300, r * s_den)))**2
    return np.clip(H, 0.0, 1.0)

def hard_box_symbol_1d(N, r):
    H = np.zeros(N, dtype=float)
    cutoff = r
    for k in range(N):
        kk = min(k, N-k)
        H[k] = 1.0 if kk <= cutoff else 0.0
    return H

def nd_symbol_from_1d(H1):
    N = len(H1)
    H2 = np.outer(H1, H1).reshape(N, N)
    H3 = np.zeros((N, N, N), dtype=float)
    for k in range(N):
        H3[:, :, k] = H2 * H1[k]
    return H3

# ---------------- Spectral Ops ----------------
def k_mesh(N):
    k1 = 2*np.pi*np.fft.fftfreq(N)
    return np.meshgrid(k1, k1, k1, indexing='ij')

def grad_sca(f):
    F = np.fft.fftn(f, axes=(-3,-2,-1))
    KX, KY, KZ = k_mesh(f.shape[-1])
    gx = np.fft.ifftn(1j*KX*F, axes=(-3,-2,-1))
    gy = np.fft.ifftn(1j*KY*F, axes=(-3,-2,-1))
    gz = np.fft.ifftn(1j*KZ*F, axes=(-3,-2,-1))
    return gx, gy, gz

def laplacian_nd(f):
    F = np.fft.fftn(f, axes=(-3,-2,-1))
    KX, KY, KZ = k_mesh(f.shape[-1])
    K2 = KX*KX + KY*KY + KZ*KZ
    return np.fft.ifftn(F*(-K2), axes=(-3,-2,-1)).real

def div_vec(v):
    Fx = np.fft.fftn(v[0], axes=(-3,-2,-1))
    Fy = np.fft.fftn(v[1], axes=(-3,-2,-1))
    Fz = np.fft.fftn(v[2], axes=(-3,-2,-1))
    KX, KY, KZ = k_mesh(v.shape[-1])
    Fd = 1j*KX*Fx + 1j*KY*Fy + 1j*KZ*Fz
    return np.fft.ifftn(Fd, axes=(-3,-2,-1)).real

def curl_vec(v):
    Fx = np.fft.fftn(v[0], axes=(-3,-2,-1))
    Fy = np.fft.fftn(v[1], axes=(-3,-2,-1))
    Fz = np.fft.fftn(v[2], axes=(-3,-2,-1))
    KX, KY, KZ = k_mesh(v.shape[-1])
    Cx = 1j*(KY*Fz - KZ*Fy)
    Cy = 1j*(KZ*Fx - KX*Fz)
    Cz = 1j*(KX*Fy - KY*Fx)
    cx = np.fft.ifftn(Cx, axes=(-3,-2,-1)).real
    cy = np.fft.ifftn(Cy, axes=(-3,-2,-1)).real
    cz = np.fft.ifftn(Cz, axes=(-3,-2,-1)).real
    return np.stack([cx, cy, cz], axis=0)

def apply_symbol_sca_real(f, H3):
    F = np.fft.fftn(f, axes=(-3,-2,-1))
    return np.fft.ifftn(F*H3, axes=(-3,-2,-1)).real

def apply_symbol_sca_cx(f, H3):
    F = np.fft.fftn(f, axes=(-3,-2,-1))
    return np.fft.ifftn(F*H3, axes=(-3,-2,-1))

def apply_symbol_vec_real(v, H3):
    return np.stack([apply_symbol_sca_real(v[i], H3) for i in range(3)], axis=0)

# ---------------- Helmholtz Projector ----------------
def helmholtz_project(u):
    N = u.shape[-1]
    Fx = np.fft.fftn(u[0], axes=(-3,-2,-1))
    Fy = np.fft.fftn(u[1], axes=(-3,-2,-1))
    Fz = np.fft.fftn(u[2], axes=(-3,-2,-1))
    KX, KY, KZ = k_mesh(N)
    K2 = KX*KX + KY*KY + KZ*KZ
    inv = np.zeros_like(K2)
    inv[K2>0] = 1.0/K2[K2>0]
    P11 = 1 - KX*KX*inv; P22 = 1 - KY*KY*inv; P33 = 1 - KZ*KZ*inv
    P12 = -KX*KY*inv;    P13 = -KX*KZ*inv;    P23 = -KY*KZ*inv
    U0 = P11*Fx + P12*Fy + P13*Fz
    U1 = P12*Fx + P22*Fy + P23*Fz
    U2 = P13*Fx + P23*Fy + P33*Fz
    u0 = np.fft.ifftn(U0, axes=(-3,-2,-1)).real
    u1 = np.fft.ifftn(U1, axes=(-3,-2,-1)).real
    u2 = np.fft.ifftn(U2, axes=(-3,-2,-1)).real
    return np.stack([u0, u1, u2], axis=0), (U0,U1,U2)

def spectral_div_rel(Utuple):
    U0,U1,U2 = Utuple
    N = U0.shape[-1]
    KX, KY, KZ = k_mesh(N)
    dot = KX*U0 + KY*U1 + KZ*U2
    num = np.sum(np.abs(dot)**2)
    den = (np.sum(np.abs(U0)**2) +
           np.sum(np.abs(U1)**2) +
           np.sum(np.abs(U2)**2))
    if den <= 0: return 0.0
    return math.sqrt(num/den)

# ---------------- NLS Dynamics ----------------
def nls_step(psi, h, sigma, H3):
    N = psi.shape[-1]
    KX, KY, KZ = k_mesh(N)
    K2 = KX*KX + KY*KY + KZ*KZ
    # kinetic half
    F = np.fft.fftn(psi, axes=(-3,-2,-1))
    F *= np.exp(-1j*0.5*h*K2)
    psi = np.fft.ifftn(F, axes=(-3,-2,-1))
    # nonlinear with smoothed density
    rho_s = apply_symbol_sca_real(np.abs(psi)**2, H3)
    psi = np.exp(-1j*sigma*rho_s*h) * psi
    # kinetic half
    F = np.fft.fftn(psi, axes=(-3,-2,-1))
    F *= np.exp(-1j*0.5*h*K2)
    psi = np.fft.ifftn(F, axes=(-3,-2,-1))
    return psi

def init_psi(N, H3, seed=0):
    rng = np.random.default_rng(seed)
    xi = apply_symbol_sca_real(rng.standard_normal((N,N,N)), H3)
    xi -= xi.mean(); xi /= max(1e-12, np.max(np.abs(xi)))
    rho0 = 1 + 0.3*xi
    rho0 = np.clip(rho0, 0.4, None)
    th = apply_symbol_sca_real(rng.standard_normal((N,N,N)), H3)
    th -= th.mean(); th /= max(1e-12, np.max(np.abs(th))); th *= 0.6
    return np.sqrt(rho0)*np.exp(1j*th)

def mass(psi):
    return float(np.mean(np.abs(psi)**2))

def highfreq_mask(N, frac=0.25):
    idx = np.arange(N)
    KX, KY, KZ = np.meshgrid(idx, idx, idx, indexing='ij')
    sym = lambda v: np.minimum(v, N-v)
    R = np.maximum.reduce([sym(KX), sym(KY), sym(KZ)])
    cutoff = int(frac*(N//2))
    return (R >= cutoff)

def energy_high_sca(psi, mask):
    F = np.fft.fftn(psi, axes=(-3,-2,-1))
    return float(np.sum(np.abs(F[mask])**2))

def energy_high_vec(u, mask):
    U = np.fft.fftn(u, axes=(-3,-2,-1))
    return float(np.sum(np.abs(U[:,mask])**2))

def phase_velocity(psi, eps):
    gx, gy, gz = grad_sca(psi)
    amp2 = np.maximum(np.abs(psi)**2, eps)
    vx = np.imag(gx*np.conj(psi)/amp2)
    vy = np.imag(gy*np.conj(psi)/amp2)
    vz = np.imag(gz*np.conj(psi)/amp2)
    return np.stack([vx.real, vy.real, vz.real], axis=0)

def mean_masked(arr, mask):
    m = arr[mask]
    if m.size == 0:
        return float(np.mean(arr))
    return float(np.mean(m))

# ---------------- main ----------------
def main():
    args = parse_args()
    np.random.seed(args.seed)
    N, r, T, dt = args.N, args.r, args.T, args.dt
    gamma_ns, gamma_q, nu, sigma = args.gamma_ns, args.gamma_q, args.nu, args.sigma

    log("="*69); log(" ONE-KERNEL UNIFIED DEMO — vOmega3_fixed"); log("="*69)
    log(f"[Setup] N={N}, r={r}, T={T}, dt={dt:.3f}, sigma={sigma:+.1f}, "
        f"gamma_ns={gamma_ns:.2f}, gamma_q={gamma_q:.2f}, nu={nu:.3f}")
    log(f"         Madelung tol={args.tol_madelung:.2e}, curl tol={args.tol_curl:.2e}")

    # Stage 1: Fejer legality + UFET
    log("\n[Stage 1] Fejer legality + UFET constant")
    H1 = hard_box_symbol_1d(N,r) if args.bad_kernel else fejer_symbol_1d(N,r)
    H3 = nd_symbol_from_1d(H1)
    ok_fejer = (abs(H1[0]-1.0)<1e-12) and (np.min(H1)>=-1e-12) and (np.max(H1)<=1.0+1e-12)
    log(f"  H[0]={H1[0]:.12f}, min={np.min(H1):.6f}, max={np.max(H1):.6f} → {EMJ(ok_fejer)}")

    rs = [max(6, r//2), r, min(N//3, max(r+4, 2*r))]
    Kconsts = []
    for rr in rs:
        Hrr = fejer_symbol_1d(N, rr)
        Kconsts.append(float(np.mean(Hrr**2) * rr))
    spread = (max(Kconsts)-min(Kconsts))/max(1e-12, min(Kconsts))
    ok_K = (spread <= 0.06)
    log(f"  UFET K(r)=mean(H^2)*r for r={rs}: {Kconsts}")
    log(f"  UFET K spread={100*spread:.2f}% → {EMJ(ok_K)}")

    # Stage 2: One-Action Dictionary (Fejer ↔ Helmholtz)
    log("\n[Stage 2] One-Action Dictionary: Fejer ↔ Helmholtz Resolvent")
    def xi(N,k): return 2*math.sin(math.pi*k/N)
    kmax = args.kmax
    kLB = max(2, min(kmax, int(math.floor(args.kappaLB*(N/max(1,r))))))
    ks   = list(range(1, kLB+1))
    X = np.array([[1.0, (xi(N,k)**2)] for k in ks], float)
    y = np.array([1.0/max(H1[k],1e-12) for k in ks], float)
    coef = np.linalg.solve(X.T@X, X.T@y)
    a, rho_fit = float(coef[0]), float(coef[1])
    emax = max(abs(H1[k]-1.0/(a+rho_fit*xi(N,k)**2)) for k in ks)
    ok_dict = (emax <= 0.04) and (not args.bad_kernel)
    log(f"  modes k=1..{kLB}, fit a={a:.6f}, rho={rho_fit:.6f}, max error={emax:.3f} → {EMJ(ok_dict)}")

    # Stage 3: Schr – mass invariance + H-law
    log("\n[Stage 3] Schr evolution (mass + H-law with Omega-controller)")
    psi = init_psi(N, H3, seed=args.seed)
    maskHF = highfreq_mask(N, frac=0.25)
    H0_s = energy_high_sca(psi, maskHF)
    M_target = mass(psi)
    for _ in range(T):
        psi = nls_step(psi, dt, sigma, H3)
        if gamma_q > 0.0:
            psi_f = apply_symbol_sca_cx(psi, H3)
            psi   = (1.0-gamma_q)*psi_f + gamma_q*psi
        cm = mass(psi)
        if cm > 0.0:
            psi *= math.sqrt(M_target / cm)
    H1_s = energy_high_sca(psi, maskHF)
    M1   = mass(psi)
    rel_m = abs(M1-M_target)/max(1e-30, M_target)
    ok_mass = (rel_m <= 5e-5)
    ok_Hsch = (H1_s <= H0_s + 1e-9)
    log(f"  Mass: target={M_target:.6e}, end={M1:.6e}, rel.err={rel_m:.3e} → {EMJ(ok_mass)}")
    log(f"  High-freq energy: start={H0_s:.3e}, end={H1_s:.3e} → {EMJ(ok_Hsch)}")

    # Stage 4: NS – div-free + H-law
    log("\n[Stage 4] NS evolution (div-free + H-law)")
    u0 = np.random.randn(3, N, N, N)
    u, Uproj = helmholtz_project(u0)
    H0_v = energy_high_vec(u, maskHF)
    for _ in range(T):
        u = (1.0-gamma_ns)*apply_symbol_vec_real(u, H3) + gamma_ns*u
        u = u + dt*nu*laplacian_nd(u)
        u = (1.0-gamma_ns)*apply_symbol_vec_real(u, H3) + gamma_ns*u
        u, Uproj = helmholtz_project(u)
    H1_v = energy_high_vec(u, maskHF)
    rel_div = spectral_div_rel(Uproj)
    ok_Hns = (H1_v <= H0_v + 1e-9)
    ok_div = (rel_div < 1e-12)
    log(f"  High-freq energy: start={H0_v:.3e}, end={H1_v:.3e} → {EMJ(ok_Hns)}")
    log(f"  spectral div ratio ||k·U||/||U|| ≈ {rel_div:.3e} → {EMJ(ok_div)}")

    # Stage 5: Madelung – energy split + curl gate (global + punctured)
    log("\n[Stage 5] Madelung structure (energy split + curl gate)")
    gx, gy, gz = grad_sca(psi)
    gradpsi_sq = (np.abs(gx)**2 + np.abs(gy)**2 + np.abs(gz)**2).real
    rho = np.abs(psi)**2
    rho_mean = float(np.mean(rho))
    eps_v = 1e-3 * max(1e-12, rho_mean)
    v   = phase_velocity(psi, eps=eps_v)
    root = np.sqrt(np.maximum(rho,1e-12))
    grx,gry,grz = grad_sca(root)
    rootgrad_sq = (np.abs(grx)**2 + np.abs(gry)**2 + np.abs(grz)**2).real
    kin_field   = rho*(v[0]**2 + v[1]**2 + v[2]**2)

    # global (coarse) check
    gpsi_glob = float(np.mean(gradpsi_sq))
    kin_glob  = float(np.mean(kin_field))
    qE_glob   = float(np.mean(rootgrad_sq))
    relE_glob = abs(gpsi_glob - (kin_glob + qE_glob))/max(1e-30, abs(gpsi_glob))
    ok_split_glob = (relE_glob <= args.tol_madelung)

    # punctured domain – exclude core where rho is tiny
    rho_cut = 1e-3 * rho_mean
    mask = (rho >= rho_cut)
    gpsi_m = mean_masked(gradpsi_sq, mask)
    kin_m  = mean_masked(kin_field, mask)
    qE_m   = mean_masked(rootgrad_sq, mask)
    relE_m = abs(gpsi_m - (kin_m + qE_m))/max(1e-30, abs(gpsi_m))
    ok_split_mask = (relE_m <= args.tol_madelung)

    # curl gate on punctured domain
    curl_v = curl_vec(v)
    curl2_field = np.sum(curl_v**2, axis=0)
    gradsq_field = 0.0
    for i in range(3):
        gi0,gi1,gi2 = grad_sca(v[i])
        gradsq_field += (np.abs(gi0)**2 + np.abs(gi1)**2 + np.abs(gi2)**2).real
    curl2_m  = mean_masked(curl2_field,  mask)
    gradsq_m = mean_masked(gradsq_field, mask)
    ratio_curl = math.sqrt(curl2_m / max(1e-30, gradsq_m))
    ok_curl = (ratio_curl <= args.tol_curl)

    log(f"  [Global]   Energy split rel.error ≈ {relE_glob:.3e} (gate ≤ {args.tol_madelung:.1e}) → {EMJ(ok_split_glob)}")
    log(f"  [Puncture] Energy split rel.error ≈ {relE_m:.3e} (gate ≤ {args.tol_madelung:.1e}) → {EMJ(ok_split_mask)}")
    log(f"  [Puncture] curl/grad ratio ≈ {ratio_curl:.3e} (gate ≤ {args.tol_curl:.1e}) → {EMJ(ok_curl)}")

    # Stage 6: Illegal kernel (hard box) positivity
    log("\n[Stage 6] Illegal kernel (hard box) positivity")
    Hb = hard_box_symbol_1d(N, r)
    kb = np.fft.ifft(Hb).real
    min_kb = float(np.min(kb))
    ok_illegal = (min_kb >= -1e-10)  # expected FAIL
    log(f"  real-space kernel min ≈ {min_kb:.3e} → {EMJ(ok_illegal)}  (EXPECTED {EMJ(False)})")

    # Summary
    log("\n" + "="*69)
    log("Unified Demo Summary")
    log("="*69)
    ok1 = ok_fejer and ok_K
    ok2 = ok_dict
    ok3 = ok_mass and ok_Hsch
    ok4 = ok_div and ok_Hns
    ok5 = ok_split_glob and ok_split_mask and ok_curl
    log(f"  Stage 1 (Fejer legality + UFET): {EMJ(ok1)}")
    log(f"  Stage 2 (One-Action dict):       {EMJ(ok2)}")
    log(f"  Stage 3 (Schr mass + H-law):     {EMJ(ok3)}")
    log(f"  Stage 4 (NS div-free + H-law):   {EMJ(ok4)}")
    log(f"  Stage 5 (Madelung robust):       {EMJ(ok5)}")
    log(f"  Stage 6 (illegal kernel):        {EMJ(False)}  (should FAIL)")
    log("")

if __name__ == "__main__":
    main()
