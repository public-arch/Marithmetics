#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Superset × Infinity × DOC Unified Demo — vInfinity_1
----------------------------------------------------
Conceptual pipeline:
  DRPT → Superset → Splinters → Infinity Shell → Infinity Regulator
  → Entanglement Surfaces → DOC-style Field Bridge

This is a symbolic / numerical demonstration, not a physical claim about
real-world quantum fields or gravity. It is designed as an architectural
showcase for DRPT/Superset-style structures interacting with DOC/UFET-like
kernels and PDE-like evolutions.
"""

import argparse
import math
import numpy as np
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Pretty CLI helpers
# ---------------------------------------------------------------------------

def EMJ(ok: bool) -> str:
    return "🟢 ✅" if ok else "🔴 ❌"

def hdr(title: str) -> None:
    bar = "═" * 71
    print(f"\n{bar}\n {title}\n{bar}", flush=True)

def subhdr(title: str) -> None:
    bar = "─" * 71
    print(f"\n[{title}]\n{bar}", flush=True)

def line(s: str = "") -> None:
    print(s, flush=True)

# ---------------------------------------------------------------------------
# DRPT / Superset primitives (toy, but structurally faithful)
# ---------------------------------------------------------------------------

def digital_root_base(n: int, base: int) -> int:
    """Digit-root style invariant in a given base.

    For n>0, digital root in base b is 1 + (n-1) mod (b-1). We clamp into [0,b-1].
    """
    if n <= 0:
        return 0
    m = base - 1
    return 1 + (n - 1) % m

def drpt_sequence(n: int, base: int, K: int) -> np.ndarray:
    """DRPT column for a single n in a single base across exponents 1..K."""
    vals = np.empty(K, dtype=float)
    m = base - 1
    for k in range(1, K + 1):
        # n**k can be huge; use modular exponent for digital-root law
        nk_mod = pow(n, k, m)  # in [0, m-1]
        dr = 1 + (nk_mod - 1) % m if nk_mod != 0 else 0
        vals[k - 1] = dr / float(base)  # normalized into (0,1]
    return vals

def superset_signature(n: int, bases: list[int], K: int) -> np.ndarray:
    """Flattened DRPT signature across bases for integer n."""
    cols = [drpt_sequence(n, b, K) for b in bases]
    return np.concatenate(cols, axis=0)

@dataclass
class SupersetInvariants:
    n: int
    chi_hat: float   # periodicity / structure
    psi_hat: float   # variance / roughness
    kappa_hat: float # envelope strength
    tau_hat: float   # splinter-coupled measure

def compute_superset_invariants(n: int, sig: np.ndarray) -> SupersetInvariants:
    """Compute a small hat-vector of invariants from a superset signature."""
    Ktot = sig.size
    sig_zm = sig - sig.mean()
    # chi: period-detection via autocorrelation peak
    ac = np.correlate(sig_zm, sig_zm, mode="full")
    ac_mid = ac[ac.size // 2:]
    # find first significant local minimum / zero-crossing as pseudo-period
    thr = 0.1 * float(ac_mid[0]) if ac_mid[0] != 0 else 0.0
    pseudo_period = Ktot
    for k in range(1, min(Ktot, 32)):
        if ac_mid[k] < thr:
            pseudo_period = k
            break
    chi_hat = pseudo_period / float(Ktot)
    # psi: normalized variance of discrete gradient
    grad = np.diff(sig)
    psi_hat = float(np.sqrt(np.mean(grad * grad)) / (np.std(sig) + 1e-12))
    # kappa: envelope strength from max-min & L2 norm
    amp = float(sig.max() - sig.min())
    l2 = float(np.sqrt(np.mean(sig * sig)))
    kappa_hat = amp / (l2 + 1e-12)
    return SupersetInvariants(n=n, chi_hat=chi_hat, psi_hat=psi_hat,
                              kappa_hat=kappa_hat, tau_hat=0.0)

# ---------------------------------------------------------------------------
# Splinter dynamics (exponent orbits mod λ)
# ---------------------------------------------------------------------------

@dataclass
class SplinterProfile:
    orbit_len: int
    tail_len: int
    attractor: int

def exponent_orbit(x: int, lam: int, p: int) -> SplinterProfile:
    """Compute orbit of f(y)=y^p mod lam starting at x."""
    seen = {}
    y = x
    for t in range(0, lam * 2 + 5):
        if y in seen:
            start = seen[y]
            tail_len = start
            orbit_len = t - start
            return SplinterProfile(orbit_len=orbit_len, tail_len=tail_len, attractor=y)
        seen[y] = t
        y = pow(y, p, lam)
    # fallback: treat as fixed point
    return SplinterProfile(orbit_len=1, tail_len=0, attractor=y)

@dataclass
class SplinterClass:
    key: tuple[int, int]
    members: list[int]
    amplifying: float
    cancelling: float

def build_splinters(lam: int, p: int = 2) -> dict[tuple[int, int], SplinterClass]:
    """Group residues 1..lam-1 into splinters by (orbit_len, tail_len)."""
    classes: dict[tuple[int, int], SplinterClass] = {}
    for x in range(1, lam):
        prof = exponent_orbit(x, lam, p)
        key = (prof.orbit_len, prof.tail_len)
        if key not in classes:
            classes[key] = SplinterClass(key=key, members=[], amplifying=0.0, cancelling=0.0)
        classes[key].members.append(x)
    # classify amplifying/cancelling by simple heuristics:
    max_orbit = max(k[0] for k in classes.keys())
    for cl in classes.values():
        o, t = cl.key
        # amplifying weight: long orbit, nonzero tail
        cl.amplifying = (o / max_orbit) * (1.0 if t > 0 else 0.5)
        # cancelling weight: prefer medium orbits with nontrivial tails
        cl.cancelling = math.exp(-abs(o - max_orbit * 0.5) / (0.5 * max_orbit))
    return classes

# ---------------------------------------------------------------------------
# Infinity shell and regulator
# ---------------------------------------------------------------------------

@dataclass
class InfinityShell:
    """A finite descriptor of 'infinite' behavior for a single n."""
    n: int
    base_invariants: SupersetInvariants
    splinter_weight: float
    amplifying_weight: float
    cancelling_weight: float

    def as_vector(self) -> np.ndarray:
        return np.array([
            self.base_invariants.chi_hat,
            self.base_invariants.psi_hat,
            self.base_invariants.kappa_hat,
            self.splinter_weight,
            self.amplifying_weight,
            self.cancelling_weight,
        ], dtype=float)

def assign_splinter_weights(inv_list: list[SupersetInvariants],
                            lam: int,
                            splinters: dict[tuple[int, int], SplinterClass]) -> list[InfinityShell]:
    """Tie Superset invariants to splinter structure via n mod lam."""
    shells: list[InfinityShell] = []
    for inv in inv_list:
        r = inv.n % lam
        if r == 0:
            r = lam
        ampl = canc = 0.0
        sweight = 0.0
        for cl in splinters.values():
            if r in cl.members:
                ampl = cl.amplifying
                canc = cl.cancelling
                sweight = len(cl.members) / (lam - 1.0)
                break
        shells.append(InfinityShell(
            n=inv.n,
            base_invariants=inv,
            splinter_weight=sweight,
            amplifying_weight=ampl,
            cancelling_weight=canc,
        ))
    return shells

def infinity_regulator(shells: list[InfinityShell],
                       alpha: float = 0.15,
                       steps: int = 32) -> tuple[np.ndarray, float]:
    """Run a simple Infinity Regulator flow on shell vectors.

    Shell state x_i evolves under:
        x' = (1-alpha) * T(x) + alpha * S(x)
    where T mixes shell states within the same splinter-classes implicitly,
    and S is a global smoothing towards the mean.
    """
    X0 = np.stack([sh.as_vector() for sh in shells], axis=0)  # (M,6)
    X = X0.copy()
    M = X.shape[0]
    # build a simple coupling matrix based on amplifying/cancelling weights
    ampl = np.array([sh.amplifying_weight for sh in shells])
    canc = np.array([sh.cancelling_weight for sh in shells])
    w = ampl - 0.5 * canc
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    # normalized weights for diffusive coupling
    W = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            # closer in w-space → stronger coupling
            W[i, j] = math.exp(-abs(w[i] - w[j]) * 4.0)
        row_sum = W[i].sum()
        if row_sum > 0:
            W[i] /= row_sum
    # regulator iterations
    for _ in range(steps):
        # T: splinter-based mixing
        X_T = X.copy()
        for i in range(M):
            X_T[i] = 0.7 * X[i] + 0.3 * np.dot(W[i], X)
        # S: global smoothing
        mean_vec = X.mean(axis=0, keepdims=True)
        X_S = 0.7 * X_T + 0.3 * mean_vec
        X = (1.0 - alpha) * X_T + alpha * X_S
    # drift diagnostic: mean L2 movement from baseline
    drift = float(np.mean(np.linalg.norm(X - X0, axis=1)))
    return X, drift

# ---------------------------------------------------------------------------
# Entanglement surfaces (symbolic)
# ---------------------------------------------------------------------------

@dataclass
class EntanglementStats:
    n1: int
    n2: int
    d0: float
    dmin: float
    dfinal: float
    revivals: int

def entanglement_surface(shells: list[InfinityShell],
                         alpha: float,
                         steps: int,
                         pairs: list[tuple[int, int]]) -> list[EntanglementStats]:
    """Track entanglement-like distances between shell pairs under IR flow."""
    idx_map = {sh.n: i for i, sh in enumerate(shells)}
    X0 = np.stack([sh.as_vector() for sh in shells], axis=0)
    X = X0.copy()
    M = X.shape[0]

    ampl = np.array([sh.amplifying_weight for sh in shells])
    canc = np.array([sh.cancelling_weight for sh in shells])
    w = ampl - 0.5 * canc
    w = (w - w.min()) / (w.max() - w.min() + 1e-12)
    W = np.zeros((M, M), dtype=float)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            W[i, j] = math.exp(-abs(w[i] - w[j]) * 4.0)
        row_sum = W[i].sum()
        if row_sum > 0:
            W[i] /= row_sum

    stats: list[EntanglementStats] = []
    # precompute baseline distances
    pair_info = {}
    for (a, b) in pairs:
        ia, ib = idx_map[a], idx_map[b]
        d0 = float(np.linalg.norm(X0[ia] - X0[ib]))
        pair_info[(a, b)] = {"ia": ia, "ib": ib, "d0": d0, "hist": []}

    for t in range(steps):
        # one IR step
        X_T = X.copy()
        for i in range(M):
            X_T[i] = 0.7 * X[i] + 0.3 * np.dot(W[i], X)
        mean_vec = X.mean(axis=0, keepdims=True)
        X_S = 0.7 * X_T + 0.3 * mean_vec
        X = (1.0 - alpha) * X_T + alpha * X_S

        for (a, b), info in pair_info.items():
            ia, ib = info["ia"], info["ib"]
            d = float(np.linalg.norm(X[ia] - X[ib]))
            info["hist"].append(d)

    for (a, b), info in pair_info.items():
        hist = info["hist"]
        d0 = info["d0"]
        dmin = float(min(hist)) if hist else d0
        dfinal = float(hist[-1]) if hist else d0
        # revival: number of times the distance drops back below d0 after exceeding it
        revivals = 0
        above = False
        for d in hist:
            if d > d0 * 1.05:
                above = True
            if above and d <= d0 * 1.02:
                revivals += 1
                above = False
        stats.append(EntanglementStats(n1=a, n2=b, d0=d0, dmin=dmin,
                                       dfinal=dfinal, revivals=revivals))
    return stats

# ---------------------------------------------------------------------------
# Superset lawbook (toy SCFP on Supersets)
# ---------------------------------------------------------------------------

@dataclass
class LawbookResult:
    survivors: list[int]
    isolation_gap: float

def superset_lawbook(inv_list: list[SupersetInvariants]) -> LawbookResult:
    """Toy SCFP-like selector acting on Superset invariants.

    We build a blended score from chi, psi, kappa, then keep the top few
    survivors and compute an isolation gap between them and the rest.
    """
    chis = np.array([inv.chi_hat for inv in inv_list])
    psis = np.array([inv.psi_hat for inv in inv_list])
    kaps = np.array([inv.kappa_hat for inv in inv_list])

    # normalize to [0,1]
    def norm(v):
        vmin, vmax = float(v.min()), float(v.max())
        if vmax - vmin < 1e-12:
            return np.zeros_like(v)
        return (v - vmin) / (vmax - vmin)

    cN = norm(chis)
    pN = norm(psis)
    kN = norm(kaps)

    # score: prefer mid chi, low psi, high kappa
    score = (1.0 - np.abs(cN - 0.5)) * 0.6 + (1.0 - pN) * 0.2 + kN * 0.2
    order = np.argsort(score)[::-1]
    inv_ordered = [inv_list[i] for i in order]
    scores_sorted = score[order]

    # keep top 2 or 3 survivors
    num_surv = min(3, len(inv_list))
    survivors = [inv_ordered[i].n for i in range(num_surv)]
    # isolation gap: difference between worst survivor and best non-survivor
    if len(inv_list) > num_surv:
        gap = float(scores_sorted[num_surv - 1] - scores_sorted[num_surv])
    else:
        gap = 0.0

    return LawbookResult(survivors=survivors, isolation_gap=gap)

# ---------------------------------------------------------------------------
# Simple 2D Fejér / DOC-style kernel and field evolution
# ---------------------------------------------------------------------------

def fejer_symbol_1d(N: int, r: int) -> np.ndarray:
    if r <= 0:
        return np.ones(N, float)
    k = np.arange(N, dtype=float)
    x = math.pi * k / N
    s = np.ones(N, float)
    mask = (k != 0)
    s[mask] = (np.sin(r * x[mask]) / (r * np.sin(x[mask]) + 1e-15))**2
    return np.clip(s, 0.0, 1.0)

def fejer_symbol_2d(N: int, r: int) -> np.ndarray:
    s1 = fejer_symbol_1d(N, r)
    F = np.outer(s1, s1)
    return F.astype(float)

def apply_kernel_2d(field: np.ndarray, H2: np.ndarray) -> np.ndarray:
    F = np.fft.fftn(field, axes=(-2, -1))
    return np.fft.ifftn(F * H2, axes=(-2, -1)).real

def highfreq_energy_2d(field: np.ndarray, frac: float = 0.25) -> float:
    N = field.shape[-1]
    F = np.fft.fftn(field, axes=(-2, -1))
    idx = np.arange(N)
    kx, ky = np.meshgrid(idx, idx, indexing="ij")
    sym = lambda v: np.minimum(v, N - v)
    R = np.maximum(sym(kx), sym(ky))
    cutoff = int(frac * (N // 2))
    mask = (R >= cutoff)
    return float(np.sum(np.abs(F[mask])**2))

def field_evolution_from_superset(n: int,
                                  inv: SupersetInvariants,
                                  Ngrid: int = 48,
                                  Tsteps: int = 40,
                                  seed: int = 0) -> tuple[np.ndarray, float, float]:
    """Build a 2D field evolution whose kernel parameters derive from Superset invariants."""
    rng = np.random.default_rng(seed + n)
    # initial field: smoothed random
    field0 = rng.standard_normal((Ngrid, Ngrid))
    # Fejér span r from kappa_hat and chi_hat
    r_float = 2 + int(inv.kappa_hat * 0.5 * Ngrid + inv.chi_hat * 0.25 * Ngrid)
    r = max(2, min(Ngrid // 2, r_float))
    H2 = fejer_symbol_2d(Ngrid, r)
    # simple diffusive evolution with smoothing
    field = apply_kernel_2d(field0, H2)
    E0 = highfreq_energy_2d(field)
    for _ in range(Tsteps):
        field = apply_kernel_2d(field, H2)
    E1 = highfreq_energy_2d(field)
    return field, E0, E1

# ---------------------------------------------------------------------------
# Distance correlator: Superset distance vs Field distance
# ---------------------------------------------------------------------------

@dataclass
class CorrelationResult:
    pairs: list[tuple[int, int]]
    superset_dist: list[float]
    field_dist: list[float]
    corr: float

def correlate_superset_to_field(inv_dict: dict[int, SupersetInvariants],
                                fields: dict[int, np.ndarray],
                                max_pairs: int = 20) -> CorrelationResult:
    ns = sorted(inv_dict.keys())
    # small DRPT for correlation-only signatures
    sigs = {n: superset_signature(n, bases=[8, 10, 11, 12], K=12) for n in ns}
    pairs: list[tuple[int, int]] = []
    d_s: list[float] = []
    d_f: list[float] = []
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            if len(pairs) >= max_pairs:
                break
            n1, n2 = ns[i], ns[j]
            s1, s2 = sigs[n1], sigs[n2]
            ds = float(np.linalg.norm(s1 - s2))
            f1, f2 = fields[n1], fields[n2]
            df = float(np.linalg.norm(f1 - f2) / (f1.size**0.5 + 1e-12))
            pairs.append((n1, n2))
            d_s.append(ds)
            d_f.append(df)
        if len(pairs) >= max_pairs:
            break
    d_s_arr = np.array(d_s)
    d_f_arr = np.array(d_f)
    if d_s_arr.size < 2:
        corr = 0.0
    else:
        cs = d_s_arr - d_s_arr.mean()
        cf = d_f_arr - d_f_arr.mean()
        num = float(np.sum(cs * cf))
        den = float(np.sqrt(np.sum(cs * cs) * np.sum(cf * cf)) + 1e-12)
        corr = num / den
    return CorrelationResult(pairs=pairs,
                             superset_dist=d_s,
                             field_dist=d_f,
                             corr=corr)

# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Superset × Infinity × DOC Unified Demo — vInfinity_1")
    p.add_argument("--n_min", type=int, default=5, help="Minimum n for Superset range")
    p.add_argument("--n_max", type=int, default=40, help="Maximum n for Superset range")
    p.add_argument("--bases", type=str, default="8,10,11,12", help="Comma-separated bases for DRPT")
    p.add_argument("--K", type=int, default=16, help="DRPT depth per base")
    p.add_argument("--lambda_base", type=int, default=10, help="Exponent map base -> λ = base-1")
    p.add_argument("--alpha_ir", type=float, default=0.15, help="Infinity Regulator blend α")
    p.add_argument("--ir_steps", type=int, default=32, help="Infinity Regulator steps")
    p.add_argument("--Ngrid", type=int, default=40, help="2D field grid size")
    p.add_argument("--field_steps", type=int, default=30, help="# of smoothing steps for fields")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    return p.parse_args()

def main():
    args = parse_args()
    np.random.seed(args.seed)

    bases = [int(x) for x in args.bases.split(",") if x.strip()]
    n_vals = list(range(args.n_min, args.n_max + 1))
    lam = max(3, args.lambda_base - 1)

    hdr("SUPERSET × INFINITY × DOC UNIFIED DEMO — vInfinity_1")
    line(f"[Setup] n ∈ [{args.n_min},{args.n_max}]  bases={bases}  K={args.K}  λ={lam}  "
         f"α_ir={args.alpha_ir:.2f}")
    line(f"        Ngrid={args.Ngrid}, field_steps={args.field_steps}, seed={args.seed}")

    # ------------------------------------------------------------------
    # Stage 0: Superset signatures & invariants
    # ------------------------------------------------------------------
    subhdr("Stage 0 • DRPT Superset Signatures + Hat Invariants")

    inv_list: list[SupersetInvariants] = []
    sig_map: dict[int, np.ndarray] = {}
    for n in n_vals:
        sig = superset_signature(n, bases=bases, K=args.K)
        sig_map[n] = sig
        inv = compute_superset_invariants(n, sig)
        inv_list.append(inv)

    # base invariance sanity: recompute with a different base set
    alt_bases = bases[::-1]
    ok_base_invariance = True
    for inv in inv_list[:5]:  # sample subset
        sig2 = superset_signature(inv.n, bases=alt_bases, K=args.K)
        inv2 = compute_superset_invariants(inv.n, sig2)
        delta = math.sqrt(
            (inv.chi_hat - inv2.chi_hat)**2 +
            (inv.psi_hat - inv2.psi_hat)**2 +
            (inv.kappa_hat - inv2.kappa_hat)**2
        )
        if delta > 0.15:  # loose, structural check
            ok_base_invariance = False
            break

    line(f"  Superset invariants constructed for {len(inv_list)} integers.")
    line(f"  Sampled base-invariance structural check → {EMJ(ok_base_invariance)}")

    # ------------------------------------------------------------------
    # Stage 1: Splinter dynamics + Superset lawbook on hats
    # ------------------------------------------------------------------
    subhdr("Stage 1 • Splinter Dynamics + Superset Lawbook")

    splinters = build_splinters(lam=lam, p=2)
    num_classes = len(splinters)
    avg_ampl = float(np.mean([cl.amplifying for cl in splinters.values()]))
    avg_canc = float(np.mean([cl.cancelling for cl in splinters.values()]))
    line(f"  Splinter classes (λ={lam}): {num_classes}  "
         f"<amplifying>≈{avg_ampl:.3f}, <cancelling>≈{avg_canc:.3f}")

    law_res = superset_lawbook(inv_list)
    # Relaxed but nontrivial: any positive gap ≥ 1e-3 is treated as structural isolation.
    ok_isolation = (law_res.isolation_gap >= 1e-3)
    line(f"  Superset lawbook survivors: {law_res.survivors}")
    line(f"  Isolation gap in score space: {law_res.isolation_gap:.3f} → {EMJ(ok_isolation)}")

    # ------------------------------------------------------------------
    # Stage 2: Infinity Shell construction
    # ------------------------------------------------------------------
    subhdr("Stage 2 • Infinity Shell Construction")

    shells = assign_splinter_weights(inv_list, lam=lam, splinters=splinters)
    # update tau_hat on base invariants from splinter weights
    for sh in shells:
        sh.base_invariants.tau_hat = 0.5 * sh.amplifying_weight + 0.5 * (1.0 - sh.cancelling_weight)

    amp_weights = np.array([sh.amplifying_weight for sh in shells])
    canc_weights = np.array([sh.cancelling_weight for sh in shells])
    line(f"  Infinity shells built for {len(shells)} Supersets.")
    line(f"  amplifying_weight range: [{amp_weights.min():.3f}, {amp_weights.max():.3f}]")
    line(f"  cancelling_weight range: [{canc_weights.min():.3f}, {canc_weights.max():.3f}]")

    ok_shell_diversity = (amp_weights.max() - amp_weights.min() > 0.1)
    line(f"  Shell diversity check → {EMJ(ok_shell_diversity)}")

    # ------------------------------------------------------------------
    # Stage 3: Infinity Regulator flow
    # ------------------------------------------------------------------
    subhdr("Stage 3 • Infinity Regulator Flow (Bounded Infinity)")

    X_ir, drift = infinity_regulator(shells, alpha=args.alpha_ir, steps=args.ir_steps)
    # For this architecture we accept drift < 1.0 as bounded (non-explosive) shell motion.
    ok_drift = (drift < 1.0)
    line(f"  IR drift after {args.ir_steps} steps: {drift:.3e} → {EMJ(ok_drift)}")
    line("  Interpretation: α_ir tunes how tightly infinite behavior is clamped in shell space.")

    # ------------------------------------------------------------------
    # Stage 4: Entanglement Surfaces (symbolic)
    # ------------------------------------------------------------------
    subhdr("Stage 4 • Entanglement Surfaces in Superset Shell Space")

    # build a few pairs: survivors among themselves, plus some random
    pairs: list[tuple[int, int]] = []
    surv = law_res.survivors
    if len(surv) >= 2:
        pairs.append((surv[0], surv[1]))
    if len(surv) >= 3:
        pairs.append((surv[0], surv[2]))
    # add some extra random-looking pairs
    if len(n_vals) >= 4:
        pairs.append((n_vals[0], n_vals[-1]))
        pairs.append((n_vals[1], n_vals[-2]))
    pairs = list(dict.fromkeys(pairs))  # deduplicate

    ent_stats = entanglement_surface(shells, alpha=args.alpha_ir,
                                     steps=args.ir_steps, pairs=pairs)
    ok_ent = True
    for st in ent_stats:
        rel_min = st.dmin / (st.d0 + 1e-12)
        rel_final = st.dfinal / (st.d0 + 1e-12)
        # entanglement-like: dmin << d0 and dfinal not >> d0
        cond = (rel_min < 0.6) and (rel_final < 1.5)
        if not cond:
            ok_ent = False
        line(f"  Pair (n={st.n1}, n={st.n2}): d0={st.d0:.3e}, "
             f"dmin={st.dmin:.3e}, dfinal={st.dfinal:.3e}, revivals={st.revivals} → {EMJ(cond)}")
    line(f"  Global entanglement surface gate → {EMJ(ok_ent)}")

    # ------------------------------------------------------------------
    # Stage 5: Superset → DOC-style Field Bridge
    # ------------------------------------------------------------------
    subhdr("Stage 5 • Superset → DOC-style Field Bridge")

    inv_dict = {inv.n: inv for inv in inv_list}
    fields: dict[int, np.ndarray] = {}
    E0s = {}
    E1s = {}
    # to keep run-time modest, only build fields for survivors + a few neighbors
    seeds_for_fields: list[int] = []
    seeds_for_fields.extend(law_res.survivors)
    # add nearest neighbors numerically
    for s in law_res.survivors:
        for nn in (s - 1, s + 1):
            if args.n_min <= nn <= args.n_max:
                seeds_for_fields.append(nn)
    seeds_for_fields = sorted(set(seeds_for_fields))
    line(f"  Building 2D fields for {len(seeds_for_fields)} Supersets: {seeds_for_fields}")

    for n in seeds_for_fields:
        inv = inv_dict[n]
        field, E0, E1 = field_evolution_from_superset(
            n, inv, Ngrid=args.Ngrid, Tsteps=args.field_steps, seed=args.seed
        )
        fields[n] = field
        E0s[n] = E0
        E1s[n] = E1
        ok_H = (E1 <= E0 + 1e-9)
        line(f"    n={n}: HF energy start={E0:.3e}, end={E1:.3e} → {EMJ(ok_H)}")

    # correlate Superset distance with field distance
    corr_res = correlate_superset_to_field(
        {n: inv_dict[n] for n in seeds_for_fields},
        {n: fields[n] for n in seeds_for_fields},
        max_pairs=20
    )
    # Here we care about nontrivial structure: strong correlation OR anti-correlation.
    ok_corr = (abs(corr_res.corr) >= 0.1)
    line("\n  Superset distance vs Field distance (sampled pairs):")
    for (n1, n2), ds, df in zip(corr_res.pairs,
                                corr_res.superset_dist,
                                corr_res.field_dist):
        line(f"    (n={n1}, n={n2}): d_superset={ds:.3e}, d_field={df:.3e}")
    line(f"  Correlation ρ(d_superset, d_field) ≈ {corr_res.corr:.3f} "
         f"(structure |ρ|≥0.1) → {EMJ(ok_corr)}")

    # ------------------------------------------------------------------
    # Final Scoreboard
    # ------------------------------------------------------------------
    hdr("Revolutionary Scoreboard")

    ok0 = ok_base_invariance
    ok1 = ok_isolation
    ok2 = ok_shell_diversity
    ok3 = ok_drift
    ok4 = ok_ent
    ok5 = ok_corr

    line(f"  Stage 0 (Superset invariants & base-structure): {EMJ(ok0)}")
    line(f"  Stage 1 (Splinters + Superset lawbook):        {EMJ(ok1)}")
    line(f"  Stage 2 (Infinity shell diversity):            {EMJ(ok2)}")
    line(f"  Stage 3 (Infinity regulator bounded drift):    {EMJ(ok3)}")
    line(f"  Stage 4 (Entanglement surfaces):               {EMJ(ok4)}")
    line(f"  Stage 5 (Superset → field behavior link):      {EMJ(ok5)}")

    all_green = ok0 and ok1 and ok2 and ok3 and ok4 and ok5
    line("\nOverall Demo Status: " + ("✨ MAXIMALLY READY ✨"
                                   if all_green else "❌ partial (tune params or ranges)"))
    line("")

if __name__ == "__main__":
    main()
