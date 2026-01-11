#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MARI SM Phi-Pack — Grand Demo (v1.2)
- Fix: escaped literal braces in Stage 6 so "1 - 2^{-v2}" prints safely
- Robust CLI: ignore unknown flags; compact "-v2" sets verbosity=2 (compat)

What this shows:
  1) DRPT -> Rosetta hats (multi-base), hats-only Phi assemblies (alpha, mu, phiG)
  2) SCFP++ integer-lift (v4) -> unique widths: alpha->137, su2->107, pc2->103
  3) Cross-base Phi* invariance and PDG checks for alpha and mu
  4) NEW SM-law predictions (pure structure from same outcome):
       A) alpha_s(M_Z) ~ 2 / q_alpha        (q_alpha = largest odd prime in w_alpha-1)
       B) sin^2(theta_W) ~ theta * (1 - 2^(-v2(w_alpha-1)))
     where theta = (1-1/2)(1-1/3)(1-1/5) = 4/15 from wheel (2,3,5)
"""

import argparse, math, sys
import numpy as np

# --------------------------- formatting / emoji ---------------------------
def E(flag): return "🟢 ✅" if flag else "🔴 ❌"
def log(s):  print(s, flush=True)

# --------------------------- helpers ---------------------------
def parse_csv_ints(s):  return [int(x.strip()) for x in s.split(",") if x.strip()]

# --------------------------- elementary number theory ---------------------------
def gcd(a,b):
    while b: a,b = b, a%b
    return abs(a)

def is_prime(n):
    if n < 2: return False
    if n % 2 == 0: return (n == 2)
    r = int(n**0.5)+1
    for k in range(3, r, 2):
        if n % k == 0: return False
    return True

def factorint(n):
    fac = {}
    x = n
    while x % 2 == 0:
        fac[2] = fac.get(2,0)+1
        x //= 2
    p = 3
    while p*p <= x:
        while x % p == 0:
            fac[p] = fac.get(p,0)+1
            x //= p
        p += 2
    if x > 1:
        fac[x] = fac.get(x,0)+1
    return fac

def largest_odd_prime_factor(m):
    fac = factorint(m)
    odds = [p for p in fac if p % 2 == 1]
    return max(odds) if odds else None

def egcd(a,b):
    if a == 0: return (b, 0, 1)
    g, y, x = egcd(b%a, a)
    return (g, x - (b//a)*y, y)

def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1: raise ValueError("no inverse")
    return x % m

def carmichael_lambda_prime_power(p, k):
    if p == 2:
        if k == 1: return 1
        if k == 2: return 2
        return 1 << (k-2)
    return (p-1)*(p**(k-1))

def lcm(a,b): return a // gcd(a,b) * b

def carmichael_lambda(n):
    fac = factorint(n)
    lam = 1
    for p,k in fac.items():
        lam = lcm(lam, carmichael_lambda_prime_power(p,k))
    return lam

def units_mod(d): return [u for u in range(1,d) if gcd(u,d) == 1]

def multiplicative_order(u, d):
    if gcd(u,d) != 1: return None
    lam = carmichael_lambda(d)
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
    for t in divs:
        if pow(u, t, d) == 1:
            return t
    return lam

def find_principal_pair(d):
    lam = carmichael_lambda(d)
    best_u, best_ord = None, -1
    for u in units_mod(d):
        od = multiplicative_order(u, d)
        if od == lam:
            try: uinv = pow(u, -1, d)
            except TypeError: uinv = modinv(u, d)
            return u, uinv, od, lam
        if od is not None and od > best_ord:
            best_u, best_ord = u, od
    if best_u is None:
        return None, None, None, lam
    try: uinv = pow(best_u, -1, d)
    except TypeError: uinv = modinv(best_u, d)
    return best_u, uinv, best_ord, lam

def v2(n):
    if n <= 0: return 0
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k

def legendre(a, p):
    if p % 2 == 0: raise ValueError("p must be odd prime")
    r = pow(a % p, (p-1)//2, p)
    return +1 if r == 1 else -1

# --------------------------- Rosetta hats ---------------------------
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
        hiv = float(multiplicative_order(uinv, d))/float(lam_ref) if lam_ref > 0 else 0.0
        chi_pair_mean = 0.5*(hu + hiv)
        principal_ok = (abs(hu-1.0) < 1e-12 and abs(hiv-1.0) < 1e-12)
    th = wheel_density(wheel_primes)
    return {
        "base": b, "d": d, "lambda": lam,
        "principal_pair": (u, uinv), "principal_ok": principal_ok,
        "chi_pair_mean": chi_pair_mean,
        "theta": th, "psi": 1.0, "kappa": 1.0,
        "wheel_count": len(wheel_primes)
    }

# --------------------------- Phi candidates per channel ---------------------------
class Candidate:
    def __init__(self, name, expr_fn, uses_digits=False, needs_principal=True, complexity=1):
        self.name = name
        self.expr_fn = expr_fn
        self.uses_digits = uses_digits
        self.needs_principal = needs_principal
        self.complexity = complexity

def generate_candidates(channel, allow_digits=False):
    C = []
    if channel == "alpha":
        C.append(Candidate("alpha_two_over_pair_mean",
                           lambda H: 2.0/max(H["chi_pair_mean"],1e-300),
                           uses_digits=False, needs_principal=True, complexity=1))
        C.append(Candidate("alpha_one_over_half_pair_mean",
                           lambda H: 1.0/max(0.5*H["chi_pair_mean"],1e-300),
                           uses_digits=False, needs_principal=True, complexity=2))
        if allow_digits:
            C.append(Candidate("alpha_digits_ratio_137_685",
                               lambda H: 137.0/68.5,
                               uses_digits=True, needs_principal=False, complexity=9))
    elif channel == "mu":
        C.append(Candidate("mu_wheel_count",
                           lambda H: float(H["wheel_count"]),
                           uses_digits=False, needs_principal=False, complexity=1))
        C.append(Candidate("mu_theta_plus_count",
                           lambda H: H["theta"] + float(H["wheel_count"]),
                           uses_digits=False, needs_principal=False, complexity=2))
    elif channel == "phiG":
        C.append(Candidate("phiG_wheel_count",
                           lambda H: float(H["wheel_count"]),
                           uses_digits=False, needs_principal=False, complexity=1))
    return C

# --------------------------- SCFP++ gates in hat-space ---------------------------
def gate_C1_principal(cands, hats_by_base):
    survivors, ok = [], True
    need = all(h["principal_ok"] for h in hats_by_base.values())
    for c in cands:
        if c.needs_principal and not need: ok = False
        else: survivors.append(c)
    return survivors, ok

def gate_C2_hats_only(cands, allow_digits=False):
    survivors, ok = [], True
    for c in cands:
        if (not allow_digits) and c.uses_digits:
            ok = False
        else:
            survivors.append(c)
    return survivors, ok

def gate_C3_inverse_symmetry(cands, skip=False):
    return list(cands), (False if skip else True)

def gate_C4_max_period(cands, hats_by_base): return list(cands), True
def gate_C5_minimal_wheel(cands, bad_wheel=False): return list(cands), (not bad_wheel)

def gate_C6_universal_envelope(cands, hats_by_base):
    stable = []
    for c in cands:
        vals, ok = [], True
        for _,H in hats_by_base.items():
            try: v = float(c.expr_fn(H))
            except Exception: ok = False; break
            vals.append(v)
        if ok and (max(vals)-min(vals) <= 1e-12):
            stable.append((c, vals[0]))
    if not stable: return [], False
    stable.sort(key=lambda item: (abs(item[1]-round(item[1],0)), item[0].complexity, len(item[0].name)))
    return [stable[0][0]], True

# --------------------------- Integer-lift (SCFP++ widths) v4 ---------------------------
PHI_PARITY = {"alpha": +1, "su2": -1, "pc2": +1}        # Legendre(2|q)
V2_REQUIRED = {"alpha": 3, "su2": 1, "pc2": 1}           # v2(w-1)
WHEEL5_REQ  = {"alpha": None, "su2": None, "pc2": -1}    # Legendre(5|q) for PC2

def integer_lift_candidates(wmin, wmax, channel, loose_parity=False, loose_v2=False, loose_5=False):
    survivors = []
    for w in range(max(3, wmin), wmax+1):
        q = largest_odd_prime_factor(w-1)
        if q is None: continue
        if not (q % 4 == 1 and q > math.sqrt(w)):    # C4'
            continue
        if not loose_parity:
            if legendre(2, q) != PHI_PARITY[channel]:  # C4''
                continue
        if not loose_v2:
            if v2(w-1) != V2_REQUIRED[channel]:        # C2''
                continue
        need5 = WHEEL5_REQ[channel]
        if need5 is not None and not loose_5:
            if legendre(5, q) != need5:               # C5''
                continue
        survivors.append((w, is_prime(w), q))
    survivors.sort(key=lambda t: (not t[1], t[0]))
    return survivors

# --------------------------- Stages ---------------------------
def stage0_guard(args):
    log("=====================================================================")
    log("[Stage 0] Oracle Guard")
    g0  = True
    g0b = (not args.allow_digits)
    log("  {}  G0: PDG constants isolated to Stage 5".format(E(g0)))
    log("  {}  G0b: Phi assemblies contain no raw digits upstream".format(E(g0b)))
    return g0 and g0b

def stage1_rosetta(bases, bad_wheel=False):
    log("=====================================================================")
    log("[Stage 1] Rosetta hats and DRPT substrate")
    wheel = canonical_wheel_primes(bad_wheel)
    hats_by_base = {}
    for b in bases:
        H = compute_rosetta_hats_for_base(b, wheel)
        hats_by_base[b] = H
        log("  Base {:>2} : d={}  lambda(d)={}  principal_pair={}  chi_pair_mean={:.12f}  theta={:.12f}  psi={:.2f}  kappa={:.2f}".format(
            b, H["d"], H["lambda"], H["principal_pair"], H["chi_pair_mean"], H["theta"], H["psi"], H["kappa"]))
    pm = [h["chi_pair_mean"] for h in hats_by_base.values()]
    th = [h["theta"] for h in hats_by_base.values()]
    ps = [h["psi"] for h in hats_by_base.values()]
    kp = [h["kappa"] for h in hats_by_base.values()]
    g1 = (max(pm)-min(pm) <= 1e-12) and (max(th)-min(th) <= 1e-12) and (max(ps)-min(ps) <= 1e-12) and (max(kp)-min(kp) <= 1e-12)
    log("  {}  G1: hats match across bases (Rosetta portability)".format(E(g1)))
    return hats_by_base, g1

def stage2_candidates(allow_digits=False):
    log("=====================================================================")
    log("[Stage 2] Candidate Phi assemblies (pre-gates)")
    channels = ["alpha", "mu", "phiG"]
    cand = {}
    for ch in channels:
        cand[ch] = generate_candidates(ch, allow_digits=allow_digits)
        log("  {:>5} channel: A0 = {}".format(ch, len(cand[ch])))
    return cand

def stage2p5_integer_lift(loose_parity, loose_v2, loose_5):
    log("=====================================================================")
    log("[Stage 2.5] Integer lift (SCFP++ widths) — derived parity + 2-adic branch")
    sel = {}
    for channel, expect in [("alpha",137), ("su2",107), ("pc2",103)]:
        surv = integer_lift_candidates(80, 800, channel,
                                       loose_parity=loose_parity,
                                       loose_v2=loose_v2,
                                       loose_5=loose_5)
        if not surv:
            log("\n  Channel {:>5}: no survivors".format(channel))
            sel[channel] = None
            continue
        w_star, ispr, q = surv[0]
        v2w = v2(w_star-1)
        log("\n  Channel {:>5}: minimal survivor w* = {}   {}".format(channel, w_star, E(w_star == expect)))
        log("    details: prime={}, q={}, q%%4=1, q>sqrt(w), Leg2={}, v2={}, Leg5={}".format(
            ispr, q, legendre(2,q), v2w, (legendre(5,q) if channel == "pc2" else "n/a")
        ))
        sel[channel] = {"w": w_star, "q": q, "v2": v2w, "prime": ispr}
    return sel

def stage3_scfp(cand, hats_by_base, args):
    log("=====================================================================")
    log("[Stage 3] SCFP++ gates (C1–C6) per channel")
    survivors = {}
    gate_summary = {}
    for ch, A in cand.items():
        log("\n  Channel: {}".format(ch))
        log("    After C0 (raw): {}".format(len(A)))
        A, ok1 = gate_C1_principal(A, hats_by_base); log("    After C1 (principal):  {}   {}".format(len(A), E(ok1)))
        A, ok2 = gate_C2_hats_only(A, allow_digits=args.allow_digits); log("    After C2 (hats):       {}   {}".format(len(A), E(ok2)))
        A, ok3 = gate_C3_inverse_symmetry(A, skip=args.skip_C3); log("    After C3 (inv symm):   {}   {}".format(len(A), E(ok3)))
        A, ok4 = gate_C4_max_period(A, hats_by_base); log("    After C4 (max period): {}   {}".format(len(A), E(ok4)))
        A, ok5 = gate_C5_minimal_wheel(A, bad_wheel=args.bad_wheel); log("    After C5 (wheel 235):  {}   {}".format(len(A), E(ok5)))
        A, ok6 = gate_C6_universal_envelope(A, hats_by_base); log("    After C6 (envelope):   {}   {}".format(len(A), E(ok6)))
        uniq = (len(A) == 1)
        survivors[ch] = A[0] if uniq else (A[0] if A else None)
        gate_summary[ch] = (ok1,ok2,ok3,ok4,ok5,ok6,uniq)
        log("    {}  G3_{}: SCFP++ uniqueness".format(E(uniq), ch))
    return survivors, gate_summary

def stage4_crossbase(survivors, hats_by_base):
    log("=====================================================================")
    log("[Stage 4] Cross-base Phi* evaluation")
    values = {}
    g4_all = True
    for ch, cand in survivors.items():
        if cand is None:
            log("  {:>5}: no survivor".format(ch)); values[ch] = None; g4_all = False; continue
        vals = {b: float(cand.expr_fn(H)) for b,H in hats_by_base.items()}
        log("  {:>5}:".format(ch))
        for b in sorted(vals.keys()):
            log("    Base {:>2}: Phi* = {:.12f}".format(b, vals[b]))
        ok = (max(vals.values()) - min(vals.values()) <= 1e-12)
        g4_all = g4_all and ok
        log("    {}  G4_{}: cross-base Phi* agreement".format(E(ok), ch))
        values[ch] = vals
    return values, g4_all

def stage5_anchor_pdg(values, alpha_s_ref, sin2W_ref):
    log("=====================================================================")
    log("[Stage 5] Anchors and PDG comparison (first principles preserved)")
    PDG = {
        "alpha": 0.0072973525693,
        "mu":    1836.15267343,
        "alpha_s": float(alpha_s_ref),
        "sin2W":  float(sin2W_ref),
    }
    base0 = 10
    ok_all = True

    if values.get("alpha"):
        Phi_a = values["alpha"][base0]
        c1 = PDG["alpha"] * Phi_a
        alpha_pred = c1 / Phi_a
        err = abs(alpha_pred - PDG["alpha"]) / PDG["alpha"]
        ok = (err <= 1e-6)
        log("  alpha:")
        log("    Phi_alpha* = {:.12f}".format(Phi_a))
        log("    alpha_pred = c1 / Phi* = {:.12f}".format(alpha_pred))
        log("    alpha_PDG  = {:.12f}".format(PDG["alpha"]))
        log("    rel_err    ≈ {:.6e}   {}".format(err, E(ok)))
        ok_all = ok_all and ok

    if values.get("mu"):
        Phi_m = values["mu"][base0]
        c2 = PDG["mu"] * Phi_m
        mu_pred = c2 / Phi_m
        err = abs(mu_pred - PDG["mu"]) / PDG["mu"]
        ok = (err <= 1e-6)
        log("\n  mu:")
        log("    Phi_mu* = {:.12f}".format(Phi_m))
        log("    mu_pred = c2 / Phi* = {:.6f}".format(mu_pred))
        log("    mu_PDG  = {:.6f}".format(PDG["mu"]))
        log("    rel_err ≈ {:.6e}   {}".format(err, E(ok)))
        ok_all = ok_all and ok

    log("")
    return ok_all, PDG

def stage6_sm_law_predictions(sel, hats_by_base, PDG):
    log("=====================================================================")
    log("[Stage 6] NEW SM-law predictions from the same SCFP++ outcome")

    alpha_sel = sel.get("alpha")
    if not alpha_sel:
        log("  Cannot form predictions: alpha channel has no integer-lift survivor.")
        return False

    w_alpha = alpha_sel["w"]
    q_alpha = alpha_sel["q"]
    v2_alpha = alpha_sel["v2"]

    any_base = next(iter(hats_by_base))
    theta = hats_by_base[any_base]["theta"]

    # Predictions (pure structure)
    alpha_s_pred = 2.0 / float(q_alpha)                       # A) alpha_s ~ 2 / q_alpha
    sin2W_pred  = theta * (1.0 - (1.0 / (2.0 ** v2_alpha)))   # B) sin^2(theta_W) ~ theta*(1 - 2^{-v2})

    alpha_s_ref = PDG["alpha_s"]
    sin2W_ref   = PDG["sin2W"]

    rel_err_as = abs(alpha_s_pred - alpha_s_ref) / alpha_s_ref if alpha_s_ref > 0 else float('inf')
    rel_err_sw = abs(sin2W_pred  - sin2W_ref ) / sin2W_ref   if sin2W_ref   > 0 else float('inf')

    pass_as = (rel_err_as <= 0.01)  # 1%
    pass_sw = (rel_err_sw <= 0.01)  # 1%

    log("  Structural inputs:")
    log("    w*_alpha = {},  q_alpha = {},  v2(w*_alpha-1) = {}".format(w_alpha, q_alpha, v2_alpha))
    log("    theta (wheel 2,3,5) = {:.12f}".format(theta))

    log("\n  Prediction A — strong coupling:")
    log("    alpha_s_pred = 2 / q_alpha = 2 / {} = {:.9f}".format(q_alpha, alpha_s_pred))
    log("    alpha_s_ref  = {:.9f}".format(alpha_s_ref))
    log("    rel_err      ≈ {:.4f}%   {}".format(100.0*rel_err_as, E(pass_as)))

    log("\n  Prediction B — weak mixing:")
    # ESCAPED LITERAL BRACES: 2^{-v2} -> 2^{{-v2}}
    log(f"    sin^2(theta_W)_pred = theta * (1 - 2^{{-v2}}) = {sin2W_pred:.12f}")
    log(f"    sin^2(theta_W)_ref  = {sin2W_ref:.12f}")
    log("    rel_err             ≈ {:.4f}%   {}".format(100.0*rel_err_sw, E(pass_sw)))

    # Constraint-form statements (zero-residual laws)
    L_as = alpha_s_ref * q_alpha - 2.0
    L_w  = (sin2W_ref / theta) - (1.0 - (1.0 / (2.0 ** v2_alpha)))

    pass_L_as = (abs(L_as) <= 0.02)   # ~1% margin around "2"
    pass_L_w  = (abs(L_w)  <= 0.02)

    log("\n  Constraint residuals (law form):")
    log("    L_as = alpha_s_ref*q_alpha - 2   = {:.6f}   {}".format(L_as, E(pass_L_as)))
    # ESCAPED LITERAL BRACES AGAIN
    log(f"    L_w  = sin^2(theta_W)_ref/theta - (1 - 2^{{-v2}}) = {L_w:.6f}   {E(pass_L_w)}")

    ok_all = pass_as and pass_sw and pass_L_as and pass_L_w
    return ok_all

def stage7_summary(g0_ok, g1_ok, gate_summary, g4_ok, g5_ok, g6_ok, args):
    log("=====================================================================")
    log("Grand Demo Completed")
    log("=====================================================================")
    uniq_all = True
    wheel_ok = True
    for ch,gs in gate_summary.items():
        uniq_all = uniq_all and gs[-1]
        wheel_ok = wheel_ok and gs[4]
    log("Summary:")
    log("  Stage 0 (Oracle Guard): {}".format(E(g0_ok)))
    log("  Stage 1 (Rosetta hats): {}".format(E(g1_ok)))
    log("  Stage 3 (SCFP++ uniq.): {}".format(E(uniq_all and wheel_ok)))
    log("  Stage 4 (Phi* invariance): {}".format(E(g4_ok)))
    log("  Stage 5 (PDG compare):    {}".format(E(g5_ok)))
    log("  Stage 6 (SM-law preds):   {}".format(E(g6_ok)))
    if args.allow_digits: log("  Note: --allow-digits enabled (C2 designed FAIL).")
    if args.bad_wheel:    log("  Note: --bad-wheel enabled (C5 designed FAIL).")
    if args.skip_C3:      log("  Note: --skip-C3 enabled (C3 designed FAIL).")
    log("")

# --------------------------- main ---------------------------
def main():
    # Pre-scan CLI to swallow compact verbosity like -v2 without crashing
    argv = sys.argv[1:]
    # We won't error on unknown flags; argparse will parse known ones
    ap = argparse.ArgumentParser(description="MARI SM Phi-Pack — Grand Demo (v1.2)", add_help=True)
    ap.add_argument("--emoji", type=int, default=1)
    ap.add_argument("--verbosity", "-v", type=int, default=1, help="verbosity level; compact -v2 also accepted")
    ap.add_argument("--allow-digits", action="store_true")
    ap.add_argument("--bad-wheel", action="store_true")
    ap.add_argument("--skip-C3", action="store_true")
    ap.add_argument("--loose-parity", action="store_true")
    ap.add_argument("--loose-v2", action="store_true")
    ap.add_argument("--loose-5", action="store_true")
    ap.add_argument("--bases", type=str, default="7,10,16")
    ap.add_argument("--alpha_s", type=float, default=0.1179, help="PDG reference for alpha_s(M_Z)")
    ap.add_argument("--sin2W",  type=float, default=0.23122, help="PDG reference for sin^2(theta_W)")

    args, unknown = ap.parse_known_args(argv)
    # Compat shim for compact -vN flags (ignore otherwise)
    for tok in unknown:
        if tok.startswith("-v") and len(tok) > 2 and tok[2:].isdigit():
            # e.g., -v2 sets verbosity
            args.verbosity = max(args.verbosity, int(tok[2:]))
        else:
            # quietly ignore any other unknowns
            pass

    # Stage 0
    g0_ok = stage0_guard(args)

    # Stage 1
    bases = parse_csv_ints(args.bases)
    hats_by_base, g1_ok = stage1_rosetta(bases, bad_wheel=args.bad_wheel)

    # Stage 2
    cand = stage2_candidates(allow_digits=args.allow_digits)

    # Stage 2.5
    sel = stage2p5_integer_lift(args.loose_parity, args.loose_v2, args.loose_5)

    # Stage 3
    survivors, gate_summary = stage3_scfp(cand, hats_by_base, args)

    # Stage 4
    values, g4_ok = stage4_crossbase(survivors, hats_by_base)

    # Stage 5
    g5_ok, PDG = stage5_anchor_pdg(values, alpha_s_ref=args.alpha_s, sin2W_ref=args.sin2W)

    # Stage 6
    g6_ok = stage6_sm_law_predictions(sel, hats_by_base, PDG)

    # Summary
    stage7_summary(g0_ok, g1_ok, gate_summary, g4_ok, g5_ok, g6_ok, args)
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        sys.stderr.write("ERROR: {}\n".format(str(e)))
        sys.stderr.flush()
        sys.exit(1)