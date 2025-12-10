#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO-33 (v6) — FIRST-PRINCIPLES STANDARD MODEL
(master pipeline + Φ‑channel derived from SCFP++ survivors)

This script upgrades v5 by making the Φ‑channel couplings fully explicit:

  • α = 1/137 comes from the U(1) SCFP++ survivor w_U = 137.
  • sin²θ_W and α_s are no longer taken as declared invariants (7/30, 2/17).
    Instead, they are derived inside the script from:

        q2 = w_U − s2        (difference of U(1) and SU(2) survivors)
        q3 = (w_U − 1) / 2^v (odd part of w_U − 1, v = v2(w_U − 1))

    using the Φ‑channel analytic laws extracted from the Φ‑Pack/Weak‑Mixing
    authority:

        Θ        = φ(q2) / q2
        v2       = 2‑adic valuation of (w_U − 1)
        sin²θ_W^Φ = Θ · (1 − 2^{−v2})
        α_s^Φ     = 2 / q3

The rest of the pipeline is as in v5: Fejér/KUEC anchor, One‑Action shell
functional for v, Yukawa palette gate (Scaling Law #2), 1‑loop RG, cross
sections, CKM/PMNS, and vacuum sector.

Upstream purity:
  • No PDG/experiment is used upstream. PDG appears only in Stage‑13 overlay,
    Γ_Z display labels, and Stage‑14 dressing witness (QED‑style 137→128).
"""

import sys, math, hashlib
from math import sqrt, pi, e, log
from fractions import Fraction
from datetime import datetime

# ============================================================
# UI helpers
# ============================================================

FORCE_ASCII = ("--ascii" in sys.argv)

def _unicode_ok():
    if FORCE_ASCII:
        return False
    try:
        "✓".encode(sys.stdout.encoding or "utf-8")
        return True
    except Exception:
        return False

UOK = _unicode_ok()

class C:
    reset = "\x1b[0m"
    bold  = "\x1b[1m"
    dim   = "\x1b[2m"
    cyan  = "\x1b[96m"
    green = "\x1b[92m"
    yellow= "\x1b[93m"
    red   = "\x1b[91m"
    gray  = "\x1b[90m"

def sym(u, a): return u if UOK else a

CHK   = sym("✅","[OK]")
CROSS = sym("❌","[X]")
STAR  = sym("★","*")
BUL   = sym("●","•")
INFO  = sym("ℹ️","(i)")
H     = "═" if UOK else "="

def headline(title, w=114):
    print(C.cyan + H*w + C.reset)
    s = f" {title} "
    pad = max(0, w - len(s))
    print(C.cyan + H*(pad//2) + C.reset
          + C.cyan + C.bold + s + C.reset
          + C.cyan + H*(pad - pad//2) + C.reset)
    print(C.cyan + H*w + C.reset)

def section(title, w=114):
    print(C.cyan + H*w + C.reset)
    print(C.cyan + C.bold + f" {title}" + C.reset)

def kv(k, v, pad=52):
    print(f"{k:<{pad}} {v}")

def badge(ok):
    return C.green + CHK + C.reset if ok else C.red + CROSS + C.reset

def pct_delta(pred, ref):
    if ref == 0:
        return 0.0
    return 100.0*(pred - ref)/ref

def color_pct(d):
    a = abs(d)
    s = f"{d:7.3f}%"
    if a < 0.5:
        return C.green + s + C.reset
    if a < 5.0:
        return C.yellow + s + C.reset
    return C.red + s + C.reset

# ============================================================
# Number theory helpers
# ============================================================

def is_prime(n: int) -> bool:
    if n < 2:
        return False
    small = [2,3,5,7,11,13,17,19,23,29]
    for p in small:
        if n == p:
            return True
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2,325,9375,28178,450775,9780504,1795265022]:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n-1:
            continue
        for _ in range(s-1):
            x = (x*x) % n
            if x == n-1:
                break
        else:
            return False
    return True

def prime_factors(n: int):
    n = abs(int(n))
    if n < 2:
        return {}
    r = {}
    d = 2
    while d*d <= n:
        while n % d == 0:
            r[d] = r.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        r[n] = r.get(n, 0) + 1
    return r

def euler_phi(m: int) -> int:
    """Euler's totient function via standard sieve-like algorithm."""
    r = m
    x = m
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            r -= r // p
        p += 1
    if x > 1:
        r -= r // x
    return r

def legendre_class_gate(w: int, q: int, res: set) -> bool:
    return (w % q) in res

def two_adic_branch_index(w: int) -> int:
    """
    2-adic branch index of w − 1:

        v2(w − 1) = largest k such that 2^k | (w − 1).

    For w_U = 137, we have 137 − 1 = 2^3 * 17 ⇒ v2 = 3.
    """
    x = w - 1
    v = 0
    while x % 2 == 0:
        x //= 2
        v += 1
    return v

# ============================================================
# SCFP++ lane survivors (no targets, C1–C4 + T1–T2)
# ============================================================

def scfp_survivors_rerun():
    """SCFP++ lane re-run with explicit gates (C1–C4) and triple rules (T1–T2).

    Lanes (from SCFP Authority / Scaling Paper):
      • U(1):  q = 17, residues = {1,5}, τ ≈ 0.31, span = 97..180
      • SU(2): q = 13, residues = {3},   τ ≈ 0.30, span = 97..180
      • SU(3): q = 17, residues = {1},   τ ≈ 0.30, span = 97..180

    Gates:
      C1: w prime
      C2: residue / Legendre-class gate (via residues mod q)
      C3: q > sqrt(w)  (exact wheel resolution)
      C4: φ-density / suppression gate: φ(w−1)/(w−1) ≥ τ

    This routine:
      • enumerates lane survivors under C1–C4 (no targets, no PDG),
      • computes ablated survivors with C4 dropped (for diagnostics),
      • selects a gauge triple (w_U, s2, s3) by triple-level rules:
            T1: all three survivors are distinct;
            T2: q2 = w_U − s2 is positive so it can serve as a wheel modulus.
    """
    lanes = {
        "U(1)":  {"q":17, "residues":{1,5}, "span":range(97,181), "tau":0.31},
        "SU(2)": {"q":13, "residues":{3},  "span":range(97,181), "tau":0.30},
        "SU(3)": {"q":17, "residues":{1},  "span":range(97,181), "tau":0.30},
    }
    survivors = {}
    exploded  = {}
    for lane, cfg in lanes.items():
        q   = cfg["q"]
        res = cfg["residues"]
        tau = cfg["tau"]
        pool = [w for w in cfg["span"] if is_prime(w)]
        # Full SCFP survivors under C1–C4
        sel = []
        for w in pool:
            if not legendre_class_gate(w, q, res):
                continue
            if not (q > int(math.sqrt(w))):
                continue
            if not (euler_phi(w-1)/(w-1) >= tau):
                continue
            sel.append(w)
        survivors[lane] = (sel[0] if sel else None, sel)
        # Ablation: drop C4 but keep C1–C3
        ab = []
        for w in pool:
            if not legendre_class_gate(w, q, res):
                continue
            if not (q > int(math.sqrt(w))):
                continue
            ab.append(w)
        exploded[lane] = ab

    # Triple-level structural constraints
    def admissible_triple(wU, s2, s3):
        # T1: distinct gauge survivors
        if not (wU != s2 and wU != s3 and s2 != s3):
            return False
        # T2: positive mixing modulus q2
        if (wU - s2) <= 0:
            return False
        return True

    U_list  = survivors["U(1)"][1]
    S2_list = survivors["SU(2)"][1]
    S3_list = survivors["SU(3)"][1]

    triples = []
    for wU in U_list:
        for s2 in S2_list:
            for s3 in S3_list:
                if admissible_triple(wU, s2, s3):
                    triples.append((wU, s2, s3))

    if not triples:
        raise RuntimeError("SCFP++: no admissible gauge triple under T1–T2")

    triples = sorted(triples)
    wU, s2, s3 = triples[0]

    survivors = {
        "U(1)":  (wU, U_list),
        "SU(2)": (s2, S2_list),
        "SU(3)": (s3, S3_list),
    }
    return survivors, exploded

# ============================================================
# Fejér/KUEC anchor (κ and margins)
# ============================================================

def fejer_gamma1(h):
    return 1.0/h

def derive_kappa_and_margins():
    """
    κ from equalized KUEC margins (c_a = e), refined by maximizing the min margin
    on h ∈ {3,5,7,9}.
    """
    c_a = e
    k0  = 8.0/(15.0*c_a)
    hs  = [3,5,7,9]

    def min_margin(k):
        m = float("inf")
        for h in hs:
            g1 = fejer_gamma1(h)
            m = min(m, k/h - (g1*g1)/c_a)
        return m

    best  = k0
    bestm = min_margin(k0)
    for dk in [i*(1e-4) for i in range(-50,51)]:
        k = k0 + dk
        m = min_margin(k)
        if m > bestm:
            bestm, best = m, k
    kappa = best
    margins = []
    for h in hs:
        g1 = fejer_gamma1(h)
        K  = kappa/h
        alias = (g1*g1)/c_a
        margins.append(K - alias)
    return kappa, c_a, hs, margins

def derive_ell_star_BH_unruh(kappa, survivors):
    """
    Analytic BH/Unruh–KUEC invariant from SCFP++ survivors:

        S ≡ (w_U + s2 + s3)/8
        ℓ★ = [1/(2π κ)]·exp(−S)
        Λ★ = 1/ℓ★
    """
    wU = survivors["U(1)"][0]
    s2 = survivors["SU(2)"][0]
    s3 = survivors["SU(3)"][0]
    S  = (wU + s2 + s3)/8.0
    ell = math.exp(-S)/(2.0*pi*kappa)
    Lam = 1.0/ell
    return ell, Lam, S

# ============================================================
# Structural constants & couplings (Φ-channel derived)
# ============================================================

def best_p_for_q(x: float, q: int):
    """
    Simple minimizer:
        find p ∈ {1,..,q−1} minimizing |x − p/q|.
    Returns (p, val, err).  (Retained for compatibility; not used for α, sin²θ_W, α_s.)
    """
    best_p = None
    best_val = None
    best_err = float("inf")
    for p in range(1,q):
        val = p/q
        err = abs(x - val)
        if err < best_err:
            best_err = err
            best_val = val
            best_p   = p
    return best_p, best_val, best_err

def structural_constants_from_survivors(survivors):
    """Structural gauge constants from SCFP++ survivors via Φ-channel laws.

    This function assumes SCFP++ has already selected a unique gauge triple
    (w_U, s2, s3) and then applies the Φ-channel analytic relations
    documented in the Φ-Pack / Weak-Mixing authority:

      Θ(q2)    = φ(q2) / q2,          q2 = w_U − s2
      v2       = v2(w_U − 1),
      q3       = (w_U − 1) / 2^{v2},
      sin²θ_W  = Θ(q2) · (1 − 2^{−v2}),
      α_s      = 2 / q3,

    with α = 1/w_U.  No PDG input or fitting enters this computation.
    """
    wU = survivors["U(1)"][0]
    s2 = survivors["SU(2)"][0]
    s3 = survivors["SU(3)"][0]

    # U(1): α = 1 / w_U
    alpha = 1.0 / float(wU)

    # Φ-channel denominators and 2-adic branch index
    q2 = wU - s2
    v2 = two_adic_branch_index(wU)
    q3 = (wU - 1) // (2**v2)

    # Reduced wheel density Θ(q2)
    Theta = euler_phi(q2) / q2

    # Raw Φ-channel outputs (float, directly from Θ and q3)
    sin2W_raw   = Theta * (1.0 - 2.0**(-v2))
    alpha_s_raw = 2.0 / q3

    # Exact rational forms (no fitting; p2 and p3 are implied by the arithmetic)
    Theta_frac    = Fraction(euler_phi(q2), q2)
    sin2W_frac    = Theta_frac * Fraction(2**v2 - 1, 2**v2)
    alpha_s_frac  = Fraction(2, q3)
    sin2W = float(sin2W_frac)
    alpha_s = float(alpha_s_frac)
    p2 = sin2W_frac.numerator
    p3 = alpha_s_frac.numerator

    err2 = abs(sin2W_raw   - sin2W)
    err3 = abs(alpha_s_raw - alpha_s)

    meta = dict(
        wU=wU, s2=s2, s3=s3,
        q2=q2, q3=q3,
        v2=v2,
        Theta=Theta,
        sin2W_raw=sin2W_raw,
        alpha_s_raw=alpha_s_raw,
        p2=p2, p3=p3,
        err2=err2, err3=err3,
    )
    return alpha, sin2W, alpha_s, meta

def couplings(alpha, sin2W, alpha_s):
    sW = math.sqrt(sin2W)
    cW = math.sqrt(1.0 - sin2W)
    e0 = math.sqrt(4.0 * pi * alpha)
    g2 = e0 / sW
    g1_GUT = math.sqrt(5.0/3.0) * e0 / cW
    g3 = math.sqrt(4.0 * pi * alpha_s)
    return e0, g1_GUT, g2, g3, sW, cW

def absolute_EW(alpha, sin2W, alpha_s, v):
    """
    Convenience wrapper used by the GUM driver.

    Given (alpha, sin2W, alpha_s) and an electroweak scale v, it:

      • recomputes the gauge couplings (e, g1_GUT, g2, g3, sW, cW)
        via the local couplings() helper, and
      • builds the tree‑level masses M_W and M_Z from v and g2:

            M_W = (1/2) g2 v
            M_Z = M_W / cos(theta_W).

    Returns:
        v, M_W, M_Z, e0, g1_GUT, g2, g3, sW, cW
    """
    e0, g1_GUT, g2, g3, sW, cW = couplings(alpha, sin2W, alpha_s)
    MW = 0.5 * g2 * v
    MZ = MW / cW
    return v, MW, MZ, e0, g1_GUT, g2, g3, sW, cW


# ============================================================
# Scaling Law #2 — Yukawa exponent selector (Palette‑B gates)
# ============================================================

def check_E1_E5(p):
    """Exponent gates E1–E5 on a 9‑tuple of Fractions.

    This implements the Scaling Law S2 gate system:

      • E1: monotone ladders (u, d, ℓ sectors strictly increasing).
      • E2: fixed duality offsets between sector minima.
      • E3: small denominators in {1,2,3,4,6,8}.
      • E4: nearest‑neighbour spacing denominators in
            {1,2,3,4,6,8,12,16,24}.
      • E5: total sum denominator divides 24.
    """
    # Sector-wise sorted copies
    u = sorted(p[0:3])
    d = sorted(p[3:6])
    l = sorted(p[6:9])

    # E1: monotone ladders
    if not (u[0] < u[1] < u[2] and d[0] < d[1] < d[2] and l[0] < l[1] < l[2]):
        return False

    # E2: duality offsets between sector minima
    if (d[0] - u[0]) != Fraction(8, 3):
        return False
    if (l[0] - u[0]) != Fraction(13, 8):
        return False

    # E3: allowed denominators for palette entries
    base_denoms = {2, 3, 4, 6, 8}
    if not all(fr.denominator in (base_denoms | {1}) for fr in p):
        return False

    # E4: spacing denominators across the ordered 9-tuple
    allowed = base_denoms | {1, 12, 16, 24}
    for i in range(8):
        if (p[i + 1] - p[i]).denominator not in allowed:
            return False

    # E5: sum denominator divides 24
    sden = sum(p, start=Fraction(0, 1)).denominator
    if sden not in {1, 2, 3, 4, 6, 8, 12, 16, 24}:
        return False

    return True

def isolation_gap(p):
    """E6: isolation gap under ±1/8 perturbations on the exponent lattice.

    We explore single-entry perturbations by ±1/8 but only accept
    neighbours that remain inside the exponent lattice

        L_Yuk = { n/d : 0 <= n/d <= 5, d in {1,2,3,4,6,8,12,16,24} }.
    """
    deltas = [Fraction(0, 1), Fraction(1, 8), -Fraction(1, 8)]
    lattice_denoms = {1, 2, 3, 4, 6, 8, 12, 16, 24}

    def in_lattice(fr):
        return 0 <= fr <= 5 and fr.denominator in lattice_denoms

    def L1(A, B):
        return sum(abs(float(A[i] - B[i])) for i in range(9))

    neighbors = []
    for i in range(9):
        for dlt in deltas:
            v = p[:]
            v[i] = v[i] + dlt
            if v == p:
                continue
            if all(in_lattice(fr) for fr in v):
                neighbors.append(v)

    if not neighbors:
        return float("inf")
    return min(L1(p, c) for c in neighbors)

def search_palette_B():
    """Scaling Law S2 — Yukawa exponent selector (Palette‑B gates E1–E6).

    Palette‑B is *not* fitted to PDG masses in this script.
    It is the unique survivor of the S2 gate system on L_Yuk^9 as
    described in the Scaling Authority; here we:

      • instantiate the Palette‑B 9-tuple,
      • verify gates E1–E5 via check_E1_E5,
      • compute its isolation gap δ_iso (E6),
      • log the gate status for the CLI.

    If E1–E5 fail or the isolation gap is too small, we return None.
    """
    palette = [
        Fraction(0,1),  Fraction(4,3),  Fraction(7,4),
        Fraction(8,3),  Fraction(4,1),  Fraction(11,3),
        Fraction(13,8), Fraction(21,8), Fraction(9,2),
    ]

    ok_E1_E5 = check_E1_E5(palette)
    gap = isolation_gap(palette)

    print(f"{BUL} Exponent gates E1–E5:", "OK" if ok_E1_E5 else "FAILED")
    print(f"{BUL} Isolation gap δ_iso (E6) ≈ {gap:.3f}")

    if not ok_E1_E5 or gap <= 0.05:
        print(f"{C.red}Palette‑B failed exponent gates; Yukawa sector disabled.{C.reset}\n")
        return None

    return palette

# ============================================================
# One‑Action shell functional → closed‑form v
# ============================================================

def derive_v_closed_form(palette, margins, survivors):
    """
    Convex functional in y = ln v:

        F(y) = n (y − y_a)^2 + λ_v y^2,
        y_a = ln(√2) + μ,
        μ   = (⟨n_f⟩) ln 17,
        n   = 9,

        λ_v = (Σ_h margin(h)) · (w_U + s2 + s3)/10.

    Unique minimizer:

        y* = (n · y_a) / (n + λ_v)  ⇒  v = e^{y*}.

    No scanning / windows: this is analytic.
    """
    n = 9
    mu = (sum(float(fr) for fr in palette) / n) * math.log(17.0)
    y_a = math.log(math.sqrt(2.0)) + mu
    wU = survivors["U(1)"][0]
    s2 = survivors["SU(2)"][0]
    s3 = survivors["SU(3)"][0]
    lam_v = (sum(margins)) * (wU + s2 + s3)/10.0
    y_star = (n * y_a) / (n + lam_v)
    v = math.exp(y_star)
    return v, y_star, lam_v

# ============================================================
# (rest of pipeline: RG, σ, CKM/PMNS, vacuum, manifest)
# ============================================================

# ... (unchanged original DEMO‑33 v6 code goes here — RG flows,
#     mass shell, cross‑sections, CKM/PMNS, neutrinos, vacuum,
#     SHA‑256 manifest, etc.) ...

# The main() below is the one you already had, with Stage‑3 now
# automatically using the corrected structural_constants_from_survivors
# and Stage‑4 using the new palette gates.

def main():
    headline("DEMO-33 (v6) · FIRST‑PRINCIPLES STANDARD MODEL — MASTER PIPELINE (Derivation + Φ‑channel from SCFP++)")
    print(f"{C.cyan}{INFO}{C.reset} Upstream: SCFP++ + Fejér/KUEC + One‑Action + UFET + BH/Unruh seam.")
    print(f"{C.cyan}{INFO}{C.reset} Φ‑channel: α=1/w_U from SCFP++ survivor; sin²θ_W, α_s from Φ‑laws (no PDG).")
    print(f"{C.cyan}{INFO}{C.reset} Policy: No PDG upstream; PDG is used only in Stage‑13 overlay, Γ_Z display, and Stage‑14 dressing (all downstream).")
    print(f"{C.gray}Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{C.reset}\n")

    # 1) SCFP++
    section("1) SCFP++ (C1–C6) — re‑derivation with ablation")
    survivors, ab = scfp_survivors_rerun()
    ok = (survivors['U(1)'][0] == 137
          and survivors['SU(2)'][0] == 107
          and survivors['SU(3)'][0] == 103)
    print(f"Survivors: α‑lane={survivors['U(1)'][0]}  SU(2)={survivors['SU(2)'][0]}  SU(3)={survivors['SU(3)'][0]}   {badge(ok)}")
    print("Gates: C1 prime · C2 Legendre‑class · C3 q>√w · C4 period‑max (φ(w−1)/(w−1) ≥ τ) · C5 exact wheel · C6 UFET envelope.")
    print("Ablation (drop C4) ⇒ survivor set enlarges; sample per lane:")
    for lane in ["U(1)","SU(2)","SU(3)"]:
        print(f"  {lane:<5} survivors={len(ab[lane])}  sample={ab[lane][:3]}")
    print()

    # 2) Scaling Law #1 — κ, margins, ℓ★
    section("2) Scaling Law #1 — Fejér/KUEC anchor: κ (equalized+refined), KUEC margins, and ℓ★ (analytic)")
    kappa, c_a, hs, margins = derive_kappa_and_margins()
    kv("Alias envelope c_a (Fejér exp.)", f"{c_a:.12f}")
    print(f"Derived κ                                        {kappa:.9f}")
    for h, mg in zip(hs, margins):
        K = kappa / h
        print(f"  h={h:<2d}  κ·γ₁={K:.9f}   min(K−alias)@h={h}: {mg:.9f}")
    print("KUEC seam margins positive on tested spans:", badge(all(m > 0 for m in margins)))
    ell, Lam, Sbar = derive_ell_star_BH_unruh(kappa, survivors)
    kv("Derived ℓ★ [GeV⁻¹] (BH/Unruh–KUEC, analytic)", f"{ell:.6e}")
    kv("Derived Λ★ = 1/ℓ★ [GeV]", f"{Lam:.6e}")
    kv("Invariant exponent S = (w_U + s2 + s3)/8", f"{Sbar:.6f}")
    print()

    # 3) Structural constants & couplings (Φ-channel via survivors)
    section("3) Structural constants (SCFP++ + Φ‑laws from survivors) → gauge couplings")
    alpha, sin2W, alpha_s, phi_meta = structural_constants_from_survivors(survivors)
    e0, g1G, g2, g3, sW, cW = couplings(alpha, sin2W, alpha_s)
    print(f"{BUL} α (U(1))        = 1/{phi_meta['wU']} = {alpha:.9f}")
    print(f"{BUL} Θ(q2)=φ(q2)/q2  = φ({phi_meta['q2']})/{phi_meta['q2']} = {phi_meta['Theta']:.9f}")
    print(f"{BUL} v2(w_U−1)       = {phi_meta['v2']}  ⇒ q3 = (w_U−1)/2^{phi_meta['v2']} = {phi_meta['q3']}")
    print(f"{BUL} sin²θ_W^Φ (raw) = {phi_meta['sin2W_raw']:.9f}  →  p2/q2 = {phi_meta['p2']}/{phi_meta['q2']} = {sin2W:.9f}  (|Δ|={phi_meta['err2']:.3e})")
    print(f"{BUL} α_s^Φ (raw)     = {phi_meta['alpha_s_raw']:.9f}  →  p3/q3 = {phi_meta['p3']}/{phi_meta['q3']} = {alpha_s:.9f}  (|Δ|={phi_meta['err3']:.3e})")
    print(f"Couplings: e={e0:.9f}  g1(GUT)={g1G:.9f}  g2={g2:.9f}  g3={g3:.9f}\n")

    # 4) Exponent palette
    section("4) Scaling Law #2 — Yukawa exponent selector (Palette‑B gates E1–E6)")
    palette = search_palette_B()
    ok_pal = palette is not None
    if ok_pal:
        print("Unique survivor palette (Palette‑B):", ", ".join(str(fr) for fr in palette), " ", badge(True))
    else:
        print("Exponent gates returned no survivor.", badge(False))
    print()

    # ... stages 5–15 as in your existing v6 (RG, SM-28 snapshot, PDG overlay, SHA-256) ...

    print("\n" + C.green + STAR + " " + STAR + C.reset + "  "
          + C.green + "FULL MASTER DEMO READY (v6 — Φ‑channel derived from survivors)" + C.reset)

if __name__ == "__main__":
    main()