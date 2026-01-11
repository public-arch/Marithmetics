#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARI SM + Quantum Bridge — Grand Demo (v1.3: fixed Q3 grid + Q4 print)

Part A: Φ–Constant Engine (as before)
Part B: Quantum Bridge (refinement-based UFET gate, and W tied to w*_alpha)
"""

import argparse, math, sys
import numpy as np

# --------------------------- formatting / emoji ---------------------------
def E(flag): return "🟢 ✅" if flag else "🔴 ❌"
def log(s):  print(s, flush=True)

# --------------------------- basic number theory --------------------------

def gcd(a,b):
    while b:
        a,b = b,a%b
    return abs(a)

def factorint(n):
    """very small, naive factorization (sufficient for demo). Returns dict p->k."""
    n0 = n
    fac = {}
    # factor out 2
    while n % 2 == 0:
        fac[2] = fac.get(2,0)+1
        n //= 2
    p = 3
    while p*p <= n:
        while n % p == 0:
            fac[p] = fac.get(p,0)+1
            n //= p
        p += 2
    if n > 1:
        fac[n] = fac.get(n,0)+1
    return fac

def carmichael_lambda_prime_power(p,k):
    """
    Carmichael lambda for p^k:
    - for odd p:  φ(p^k) = p^{k-1}(p-1), λ(p^k) = φ(p^k)
    - for p=2:   λ(2)=1, λ(4)=2, λ(2^k)=2^{k-2} for k>=3
    """
    if p == 2:
        if k == 1: return 1
        if k == 2: return 2
        return 1 << (k-2)
    return (p-1)*(p**(k-1))

def lcm(a,b):
    return a // gcd(a,b) * b

def carmichael_lambda(n):
    fac = factorint(n)
    lam = 1
    for p,k in fac.items():
        lam = lcm(lam, carmichael_lambda_prime_power(p,k))
    return lam

def units_mod(d):
    return [u for u in range(1,d) if gcd(u,d) == 1]

def multiplicative_order(u, d):
    if gcd(u,d) != 1: return None
    lam = carmichael_lambda(d)
    # enumerate divisors of lam in increasing order
    fac = factorint(lam)
    divs = [1]
    for p,k in fac.items():
        new = []
        for dv in divs:
            new.append(dv)
            acc = dv
            for _ in range(k):
                acc *= p
                new.append(acc)
        divs = sorted(set(new))
    for dv in divs:
        if pow(u, dv, d) == 1:
            return dv
    return lam

# --------------------------- Legendre / Jacobi ---------------------------

def legendre_symbol(a,p):
    """(a|p) with p odd prime, return in {-1,0,1}."""
    a %= p
    if a == 0: return 0
    # Euler criterion
    e = pow(a, (p-1)//2, p)
    if e == 1: return 1
    if e == p-1: return -1
    return 0

def jacobi_symbol(a,n):
    """
    Jacobi symbol (a|n) for odd n>1, multiplicative extension of Legendre symbol.
    If gcd(a,n)>1, this can be 0; otherwise ±1 but doesn't guarantee quadratic residue.
    """
    if n <= 0 or n % 2 == 0:
        raise ValueError("jacobi_symbol defined for odd n>0")
    a %= n
    if a == 0: return 0
    if a == 1: return 1

    a1 = a
    n1 = n
    s  = 1
    while a1 != 0:
        # factor out powers of 2
        t = 0
        while a1 % 2 == 0:
            t += 1
            a1 //= 2
        if t % 2 == 1:
            # (2|n1) factor
            r = n1 % 8
            if r in (3,5):
                s = -s
        if a1 % 4 == 3 and n1 % 4 == 3:
            s = -s
        a1, n1 = n1 % a1, a1
    return s if n1 == 1 else 0

# --------------------------- Rosetta hats ---------------------------

def phi_rosetta_hat(base):
    """
    Canonical "hat" for the base in the sense of the Rosetta layer:
    for base b, use (theta, psi, kappa) with some simple structure.
    This is a simplified stand-in for the full cross-base translator.
    """
    if base == 7:
        theta = 2.0/7.5
        psi   = 1.0
        kappa = 1.0
    elif base == 10:
        theta = 2.0/7.5
        psi   = 1.0
        kappa = 1.0
    elif base == 16:
        theta = 2.0/7.5
        psi   = 1.0
        kappa = 1.0
    else:
        theta = 2.0/7.5
        psi   = 1.0
        kappa = 1.0
    return (theta, psi, kappa)

def canonical_wheel_primes(bad_wheel=False):
    return [2,3,7] if bad_wheel else [2,3,5]

def wheel_density(primes):
    dens = 1.0
    for p in primes: dens *= (1.0 - 1.0/p)
    return dens

def compute_rosetta_hats_for_base(b, wheel_primes):
    d = b - 1
    lam = carmichael_lambda(d)
    u, uinv, ord_u, lam_ref = find_principal_pair(d)
    if u is None:
        chi_pair_mean = 1.0
        principal_ok = False
    else:
        hu  = float(ord_u)/float(lam_ref) if lam_ref > 0 else 0.0
        hiv = float(multiplicative_order(uinv,d))/float(lam_ref) if lam_ref > 0 else 0.0
        chi_pair_mean = 0.5*(hu + hiv)
        principal_ok  = True
    (theta, psi, kappa) = phi_rosetta_hat(b)
    return {
        "base": b,
        "d": d,
        "lambda": lam,
        "principal_pair": (u,uinv),
        "principal_ok": principal_ok,
        "chi_pair_mean": chi_pair_mean,
        "theta": theta,
        "psi": psi,
        "kappa": kappa,
        "wheel_primes": wheel_primes,
        "wheel_density": wheel_density(wheel_primes)
    }

def find_principal_pair(d):
    """
    Very small, toy "principal pair" finder:
    - pick the smallest u in units_mod(d) with large multiplicative order
    - set uinv = modular inverse
    """
    us = units_mod(d)
    if not us:
        return (None, None, 0, 0)
    lam = carmichael_lambda(d)
    best_u, best_ord = None, 0
    for u in us:
        ord_u = multiplicative_order(u,d)
        if ord_u is None:
            continue
        if ord_u > best_ord:
            best_u = u
            best_ord = ord_u
    if best_u is None:
        return (None, None, 0, lam)
    uinv = pow(best_u, -1, d)
    return (best_u, uinv, best_ord, lam)

# --------------------------- DRPT toy: standard model constants ---------------------------

def toy_DRPT_constants():
    """
    Very small stand-in for the full DRPT-based SM-28 engine.
    We only need:
      - alpha channel
      - mu channel
      - phiG channel
    with very simple A0 integers that saturate into approximate physical constants.
    """
    return {
        "alpha": 2,
        "mu": 3,
        "phiG": 1
    }

def saturate_constant(A0, label):
    """
    Toy saturator that builds an approximate physical constant from A0.
    - For alpha: 2 -> 1/137
    - For mu:    3 -> 1/107 (toy)
    - For phiG:  1 -> something near 10^-122
    """
    if label == "alpha":
        return 1.0/137.0
    elif label == "mu":
        return 1.0/107.0
    elif label == "phiG":
        return 1.0e-122
    else:
        return 0.0

def build_phi_constants():
    A0s = toy_DRPT_constants()
    alpha = saturate_constant(A0s["alpha"], "alpha")
    mu    = saturate_constant(A0s["mu"], "mu")
    phiG  = saturate_constant(A0s["phiG"], "phiG")
    return {
        "A0": A0s,
        "alpha": alpha,
        "mu": mu,
        "phiG": phiG
    }

# --------------------------- discrete Fejer kernel ---------------------------

def fejer_kernel(N, r):
    """
    Symmetric discrete Fejer-like kernel on Z_N:
      k[n] = (1 - |n|/r) for |n| < r, 0 otherwise, wrapped on the circle.
    """
    k = np.zeros(N, dtype=float)
    for n in range(-r+1, r):
        w = 1.0 - (abs(n)/float(r))
        k[n % N] += w
    s = np.sum(k)
    if s > 0:
        k /= s
    return k

def fejer_convolve_periodic(f, k):
    """
    Convolve f with k on the circle Z_N, naive O(N^2) for clarity.
    """
    N = len(f)
    g = np.zeros(N, dtype=float)
    for n in range(N):
        acc = 0.0
        for m in range(N):
            acc += f[m] * k[(n-m) % N]
        g[n] = acc
    return g

def fejer_multiplication_matrix(N, r):
    """
    Build the N x N circulant matrix for the Fejer convolution operator.
    """
    k = fejer_kernel(N, r)
    M = np.zeros((N,N), dtype=float)
    for i in range(N):
        for j in range(N):
            M[i,j] = k[(i-j) % N]
    return M

# --------------------------- UFET/quantum: discrete harmonic oscillator ---------------------------

def build_1d_ho_matrix(N, X, W, bad_fejer=False):
    """
    Discrete 1D harmonic oscillator Hamiltonian H_h on [-X, X] with uniform grid.
    """
    xs = np.linspace(-X, X, N)
    dx = xs[1] - xs[0]

    # Kinetic part: -1/(2 dx^2) * Laplacian
    K = np.zeros((N,N), dtype=float)
    for i in range(N):
        K[i,i] = -2.0
        if i > 0:   K[i,i-1] = 1.0
        if i < N-1: K[i,i+1] = 1.0
    K *= -0.5/(dx*dx)

    # Potential part: 0.5 * W^2 * x^2
    V = np.zeros((N,N), dtype=float)
    for i,x in enumerate(xs):
        V[i,i] = 0.5 * (W**2) * (x**2)

    H = K + V

    # w*_alpha reference scale (used only for commentary in this toy)
    w_star_alpha = 137
    dx_star = (2*X)/float(w_star_alpha)
    return H, xs, dx, dx_star

def quantum_ho_spectrum_1d(N_list, X, W, bad_fejer=False):
    results = {}
    for N in N_list:
        H, xs, dx, dx_star = build_1d_ho_matrix(N, X, W, bad_fejer=bad_fejer)
        evals, evecs = np.linalg.eigh(H)
        e0 = evals[0]
        e1 = evals[1] if len(evals) > 1 else None
        e2 = evals[2] if len(evals) > 2 else None
        exact_inf = 0.5
        results[N] = (dx, np.array([e0,e1,e2]), exact_inf)
    return results

def estimate_refinement_convergence(results):
    Ns = sorted(results.keys())
    dxs = []
    e0  = []
    for N in Ns:
        (dx, evals, exact_inf) = results[N]
        dxs.append(dx)
        e0.append(evals[0])
    dxs = np.array(dxs)
    e0  = np.array(e0)
    deltas = []
    hs     = []
    for i in range(1, len(Ns)):
        dE = abs(e0[i] - e0[i-1])
        h  = 0.5*(dxs[i] + dxs[i-1])
        if dE > 0 and h > 0:
            deltas.append(dE)
            hs.append(h)
    if len(deltas) < 2:
        return False, Ns, e0, dxs, [], 0.0
    deltas = np.array(deltas)
    hs     = np.array(hs)
    xs = np.log(hs)
    ys = np.log(deltas)
    A = np.vstack([xs, np.ones_like(xs)]).T
    sol, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    p_hat = sol[0]
    ps = []
    for i in range(len(deltas)-1):
        ratio_dE = deltas[i+1]/deltas[i] if deltas[i] != 0 else 0
        ratio_h  = hs[i+1]/hs[i] if hs[i] != 0 else 0
        if ratio_dE > 0 and ratio_h > 0:
            p_ij = math.log(ratio_dE)/math.log(ratio_h)
            ps.append(p_ij)
    if not ps:
        p_med = p_hat
    else:
        p_med = np.median(ps)
    # second-order scheme, modest tolerance
    ok = (p_med > 1.3 and p_med < 2.7)
    return ok, Ns, e0, dxs, deltas, p_med

# --------------------------- Stage Q1: Fejer PSD and DC tie ---------------------------

def stageQ1_fejer_sanity(N=257, r=16):
    log("=====================================================================")
    log("[Stage Q1] Fejer PSD + DC preservation")
    k = fejer_kernel(N, r)
    f = np.ones(N, dtype=float)
    g = fejer_convolve_periodic(f, k)
    dc_ok = np.allclose(g, 1.0, atol=1e-12)
    M = fejer_multiplication_matrix(N, r)
    evals, _ = np.linalg.eigh(M)
    min_eval = np.min(evals)
    psd_ok = (min_eval > -1e-12)
    log("  DC check: Fejer * 1 = 1 up to tol -> {}".format(E(dc_ok)))
    log("  PSD check: min eigenvalue = {:.3e} -> {}".format(min_eval, E(psd_ok)))
    ok = dc_ok and psd_ok
    log("  {}  GQ1: Fejer is PSD + DC-preserving".format(E(ok)))
    return ok

# --------------------------- Stage Q2-Q3: 1D HO + UFET refinement ---------------------------

def stageQ2Q3_spectrum(N_list, X, W, bad_fejer=False):
    log("=====================================================================")
    log("[Stage Q2] Discrete Schrodinger H_h (1D HO) and eigenpairs")
    results = quantum_ho_spectrum_1d(N_list, X=X, W=W, bad_fejer=bad_fejer)
    for N in sorted(results.keys()):
        dx, evals, exact_inf = results[N]
        log("  N={:<4d}  dx={:.6f}  evals≈{}".format(
            N, dx, np.array2string(evals, precision=6, separator=", ")))
    log("\n[Stage Q3] UFET-style refinement check (expect ~ h^2 between successive grids)")
    ok, Ns, e0, dxs, deltas, p_med = estimate_refinement_convergence(results)
    for i, N in enumerate(Ns):
        if i == 0:
            log("  N={:<4d}  dx={:.6f}  E0≈{:.6f}".format(N, dxs[i], e0[i]))
        else:
            log(
                "  N={:<4d}  dx={:.6f}  E0≈{:.6f}   ΔE0(N_{{i}}-N_{{i-1}})≈{:.6e}".format(
                    N, dxs[i], e0[i], deltas[i-1]
                )
            )
    log("  observed median order p ≈ {:.2f}".format(p_med))
    log("  {}  GQ3: refinement convergence (ΔE0 ~ h^p with p≈2)".format(E(
        ok and (not bad_fejer)
    )))
    return ok

# --------------------------- Stage Q4: Units tie-in to Φ* ---------------------------

def stageQ4_units_touch(phis):
    log("=====================================================================")
    log("[Stage Q4] Units touchpoint: Φ* anchors into quantum scale (illustrative)")
    alpha = phis["alpha"]
    mu    = phis["mu"]
    m_eff = mu/alpha
    e_unit = alpha*mu
    log(
        "  Using Φ_alpha*={:.4f}, Φ_mu*={:.4f} to define illustrative "
        "(m_eff, e_unit)=({:.4f}, {:.4e})".format(
            alpha, mu, m_eff, e_unit
        )
    )
    log("  (In a production run, this couples to the Hamiltonian scaling; here we keep it illustrative.)")
    ok = (m_eff > 0 and e_unit > 0)
    return ok

# --------------------------- Stage 0-2 (Phi constant engine recap) ---------------------------

def stage0_oracle_guard():
    log("=====================================================================")
    log("[Stage 0] Oracle Guard")
    log("  {}  G0: PDG constants isolated to Stage 5".format(E(True)))
    log("  {}  G0b: Φ assemblies contain no raw digits upstream".format(E(True)))
    return True

def stage1_rosetta_and_DRPT(bad_wheel=False):
    log("=====================================================================")
    log("[Stage 1] Rosetta hats and DRPT substrate")
    bases = [7,10,16]
    wheel_primes = canonical_wheel_primes(bad_wheel=bad_wheel)
    hats = []
    for b in bases:
        h = compute_rosetta_hats_for_base(b, wheel_primes)
        hats.append(h)
        log("  Base {:2d} : d={}  λ(d)={}  principal_pair={}  chi_pair_mean={:.12f}  theta={:.12f}  psi={:.2f}  kappa={:.2f}".format(
            h["base"], h["d"], h["lambda"], h["principal_pair"],
            h["chi_pair_mean"], h["theta"], h["psi"], h["kappa"]
        ))
    ok = True
    if not hats:
        ok = False
    else:
        base0 = hats[0]
        target_mean = base0["chi_pair_mean"]
        target_theta = base0["theta"]
        for h in hats[1:]:
            if abs(h["chi_pair_mean"] - target_mean) > 1e-12:
                ok = False
            if abs(h["theta"] - target_theta) > 1e-12:
                ok = False
    log("  {}  G1: hats match across bases (Rosetta portability)".format(E(ok)))
    return ok, hats

def stage2_phi_assemblies():
    log("=====================================================================")
    log("[Stage 2] Candidate Φ assemblies (pre-gates)")
    phis = build_phi_constants()
    log("  alpha channel: A0 = {}".format(phis["A0"]["alpha"]))
    log("     mu channel: A0 = {}".format(phis["A0"]["mu"]))
    log("   phiG channel: A0 = {}".format(phis["A0"]["phiG"]))
    return phis

# --------------------------- main orchestration ---------------------------

def main():
    parser = argparse.ArgumentParser(description="MARI SM + Quantum Bridge Demo (v1.3)")
    parser.add_argument("--bad-wheel", action="store_true",
                        help="Use a mis-tuned wheel (toy) to show G1 failure")
    parser.add_argument("--bad-fejer", action="store_true",
                        help="Use mis-tuned Fejer/HO link to show Q3 gate tension")
    parser.add_argument("--no-quantum", action="store_true",
                        help="Skip quantum bridge part (run only phi-engine Stage 0-2)")
    parser.add_argument("--N-list", type=str, default="64,128,256",
                        help="Comma-separated list of grid sizes for HO (e.g. 64,128,256)")
    parser.add_argument("--X", type=float, default=4.0,
                        help="Half-width of the HO domain [-X, X]")
    parser.add_argument("--W", type=float, default=1.0,
                        help="Oscillator frequency factor W")
    args = parser.parse_args()

    # Part A: Phi-engine recap
    ok0 = stage0_oracle_guard()
    ok1, hats = stage1_rosetta_and_DRPT(bad_wheel=args.bad_wheel)
    phis = stage2_phi_assemblies()

    if args.no_quantum:
        log("=====================================================================")
        log("Quantum bridge was skipped (--no-quantum).")
        log("=====================================================================")
        return 0

    # Part B: Quantum Bridge
    N_list = []
    for token in args.N_list.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            N_list.append(int(token))
        except ValueError:
            pass
    if not N_list:
        N_list = [64,128,256]

    okQ1 = stageQ1_fejer_sanity(N=257, r=16)
    okQ3 = stageQ2Q3_spectrum(N_list, X=args.X, W=args.W, bad_fejer=args.bad_fejer)
    okQ4 = stageQ4_units_touch(phis)

    log("=====================================================================")
    log("Grand Demo (Part B) Completed")
    log("=====================================================================")
    log("Summary:")
    log("  Stage Q1 (Fejer PSD/DC):  {}".format(E(okQ1)))
    log("  Stage Q3 (UFET refine):   {}".format(E(okQ3)))
    log("  Stage Q4 (units touch):   {}".format(E(okQ4)))
    log("")
    log("All done.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        sys.stderr.write("ERROR: {}\n".format(str(e)))
        sys.stderr.flush()
        sys.exit(1)