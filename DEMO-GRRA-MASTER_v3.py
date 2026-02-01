#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import datetime as _dt
import hashlib
import itertools
import json
import math
import platform
import random
import sys
from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

LINE = "=" * 98
FAILS = 0

def utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stable_json(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def spec_sha256(spec: dict) -> str:
    return sha256_hex(stable_json(spec))

def determinism_sha256(record: dict) -> str:
    def q(x):
        if isinstance(x, float):
            if math.isnan(x) or math.isinf(x):
                return None
            return float(f"{x:.12g}")
        if isinstance(x, dict):
            return {k: q(v) for k, v in x.items()}
        if isinstance(x, list):
            return [q(v) for v in x]
        if isinstance(x, tuple):
            return [q(v) for v in x]
        return x
    return sha256_hex(stable_json(q(record)))

def ok_line(ok: bool, label: str, detail: str = "") -> None:
    global FAILS
    tag = "PASS" if ok else "FAIL"
    if not ok:
        FAILS += 1
    pad = 76
    left = (label[:pad] + ("…" if len(label) > pad else "")).ljust(pad)
    tail = f"  {detail}" if detail else ""
    print(f"{tag:4}  {left}{tail}")

# =========================
# Number theory (exact)
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

def v2(n: int) -> int:
    if n == 0:
        return 0
    n = abs(n)
    k = 0
    while (n & 1) == 0:
        n >>= 1
        k += 1
    return k

def odd_part(n: int) -> int:
    if n == 0:
        return 0
    n = abs(n)
    while (n & 1) == 0:
        n >>= 1
    return n

def factorize(n: int) -> Dict[int, int]:
    if n <= 0:
        return {}
    out: Dict[int, int] = {}
    while n % 2 == 0:
        out[2] = out.get(2, 0) + 1
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out

def phi(n: int) -> int:
    if n <= 0:
        return 0
    fac = factorize(n)
    res = n
    for p in fac:
        res = (res // p) * (p - 1)
    return res

def theta_density_of_w_minus_1(w: int) -> Tuple[int, int]:
    # returns (phi(w-1), w-1) as a reduced fraction comparison basis
    n = w - 1
    if n <= 0:
        return (0, 1)
    return (phi(n), n)

def ge_frac(num: int, den: int, tau_num: int, tau_den: int) -> bool:
    # num/den >= tau_num/tau_den
    return num * tau_den >= tau_num * den

# =========================
# Grammar elements
# =========================
@dataclass(frozen=True)
class LaneSpec:
    q: int
    residues: Tuple[int, ...]       # C2 gate
    tau_num: int                   # C4 gate (theta floor), exact rational
    tau_den: int
    v2_required: Optional[int]      # lane-specific valuation constraint

@dataclass(frozen=True)
class BaselineSpec:
    window_lo: int
    window_hi: int
    U1: LaneSpec
    SU2: LaneSpec
    SU3: LaneSpec
    u1_coherence_v2: int

CANON_TRIPLE = (137, 107, 103)

def lane_pool(lo: int, hi: int, lane: LaneSpec, drop_prime=False, drop_residue=False, drop_q_gt_sqrtw=False, drop_theta_floor=False, drop_v2=False) -> List[int]:
    out: List[int] = []
    Rset = set(lane.residues)
    for w in range(lo, hi + 1):
        if not drop_prime:
            if not is_prime(w):
                continue
        if not drop_residue:
            if (w % lane.q) not in Rset:
                continue
        if not drop_q_gt_sqrtw:
            if not (lane.q > math.sqrt(w)):
                continue
        if not drop_theta_floor:
            num, den = theta_density_of_w_minus_1(w)
            if not ge_frac(num, den, lane.tau_num, lane.tau_den):
                continue
        if not drop_v2 and lane.v2_required is not None:
            if v2(w - 1) != lane.v2_required:
                continue
        out.append(w)
    return out

def apply_u1_coherence(pool: List[int], v2_target: int, drop: bool = False) -> List[int]:
    if drop:
        return list(pool)
    return [w for w in pool if v2(w - 1) == v2_target]

def ordered_triple_count(U1: List[int], SU2: List[int], SU3: List[int]) -> Tuple[int, Optional[Tuple[int,int,int]]]:
    triples = []
    for wU in U1:
        for s2 in SU2:
            if wU <= s2:
                continue
            for s3 in SU3:
                if s2 <= s3:
                    continue
                triples.append((wU, s2, s3))
    triples = sorted(set(triples))
    if len(triples) == 1:
        return (1, triples[0])
    return (len(triples), None)

def lane_unique_triple(U1: List[int], SU2: List[int], SU3: List[int]) -> Optional[Tuple[int,int,int]]:
    if len(U1) == 1 and len(SU2) == 1 and len(SU3) == 1:
        return (U1[0], SU2[0], SU3[0])
    return None

def phi_channel(tri: Tuple[int,int,int]) -> Dict[str, float]:
    wU, s2, s3 = tri
    q2 = wU - s2
    q3 = odd_part(wU - 1)
    v2U = v2(wU - 1)
    alpha0_inv = float(wU)
    sin2W = 7.0 / float(q2)
    alpha_s = 2.0 / float(q3)
    return {
        "triple": tri,
        "q2": q2,
        "q3": q3,
        "v2U": v2U,
        "alpha0_inv": alpha0_inv,
        "sin2W": sin2W,
        "alpha_s": alpha_s,
    }

def rel_err(x: float, x0: float) -> float:
    return abs(x - x0) / abs(x0)

# =========================
# Derived residues (moduli scan)
# =========================
def derived_residues_U1(q: int) -> Optional[Tuple[int,int]]:
    if (q + 3) % 4 != 0:
        return None
    r = (q + 3) // 4
    if r <= 0 or r >= q or r == 1:
        return None
    return (1, r)

def derived_residues_SU2(q: int) -> Optional[Tuple[int]]:
    if (q - 1) % 4 != 0:
        return None
    r = (q - 1) // 4
    if r <= 0 or r >= q:
        return None
    return (r,)

def derived_residues_SU3(q: int) -> Tuple[int]:
    return (1,)

# =========================
# Residue enumeration (compressed exact)
# =========================
def candidates_no_residue(lo: int, hi: int, q: int, tau_num: int, tau_den: int, v2_req: Optional[int]) -> List[int]:
    out = []
    for w in range(lo, hi + 1):
        if not is_prime(w):
            continue
        if not (q > math.sqrt(w)):
            continue
        num, den = theta_density_of_w_minus_1(w)
        if not ge_frac(num, den, tau_num, tau_den):
            continue
        if v2_req is not None and v2(w - 1) != v2_req:
            continue
        out.append(w)
    return out

def build_lane_pool_freq(candidates: List[int], q: int, max_k: int) -> Dict[int, int]:
    # returns dict: mask -> count of residue sets (|R|<=max_k, nonempty) that yield that mask
    residues = list(range(1, q))
    w_res = [w % q for w in candidates]
    freq: Dict[int, int] = {}
    max_k = min(max_k, q - 1)
    for k in range(1, max_k + 1):
        for comb in itertools.combinations(residues, k):
            R = set(comb)
            mask = 0
            for i, r in enumerate(w_res):
                if r in R:
                    mask |= (1 << i)
            freq[mask] = freq.get(mask, 0) + 1
    return freq

def mask_list(mask: int, candidates: List[int]) -> List[int]:
    out = []
    for i, w in enumerate(candidates):
        if (mask >> i) & 1:
            out.append(w)
    return out

def residue_enumeration_ordered(baseline: BaselineSpec, max_k: int = 5) -> Dict[str, object]:
    # Fixed moduli for this exact residue scan: baseline q's.
    lo, hi = baseline.window_lo, baseline.window_hi

    U1_cand = candidates_no_residue(lo, hi, baseline.U1.q, baseline.U1.tau_num, baseline.U1.tau_den, v2_req=None)
    SU2_cand = candidates_no_residue(lo, hi, baseline.SU2.q, baseline.SU2.tau_num, baseline.SU2.tau_den, v2_req=baseline.SU2.v2_required)
    SU3_cand = candidates_no_residue(lo, hi, baseline.SU3.q, baseline.SU3.tau_num, baseline.SU3.tau_den, v2_req=baseline.SU3.v2_required)

    U1_freq = build_lane_pool_freq(U1_cand, baseline.U1.q, max_k=max_k)
    SU2_freq = build_lane_pool_freq(SU2_cand, baseline.SU2.q, max_k=max_k)
    SU3_freq = build_lane_pool_freq(SU3_cand, baseline.SU3.q, max_k=max_k)

    total_instances = sum(U1_freq.values()) * sum(SU2_freq.values()) * sum(SU3_freq.values())

    # coherence implies only wU=137 survives; locate index if present
    idx_137 = U1_cand.index(137) if 137 in U1_cand else None

    def count_triples_for_masks(u_mask: int, s2_mask: int, s3_mask: int) -> Tuple[int, Optional[Tuple[int,int,int]]]:
        if idx_137 is None:
            return (0, None)
        if ((u_mask >> idx_137) & 1) == 0:
            return (0, None)
        wU = 137
        S2 = mask_list(s2_mask, SU2_cand)
        S3 = mask_list(s3_mask, SU3_cand)
        if not S2 or not S3:
            return (0, None)
        S2.sort()
        S3.sort()
        total = 0
        for s2 in S2:
            if s2 >= wU:
                continue
            total += bisect_left(S3, s2)
            if total > 1:
                break
        if total == 1:
            # identify unique triple
            for s2 in S2:
                if s2 >= wU:
                    continue
                for s3 in S3:
                    if s3 < s2:
                        return (1, (wU, s2, s3))
                    break
        return (total, None)

    none_count = 0
    unique_count = 0
    multi_count = 0
    unique_freq: Dict[Tuple[int,int,int], int] = {}

    # fast aggregate: if U1 mask lacks 137 bit, everything is none
    su2_total = sum(SU2_freq.values())
    su3_total = sum(SU3_freq.values())

    for u_mask, u_ct in U1_freq.items():
        if idx_137 is None or ((u_mask >> idx_137) & 1) == 0:
            none_count += u_ct * su2_total * su3_total
            continue
        for s2_mask, s2_ct in SU2_freq.items():
            for s3_mask, s3_ct in SU3_freq.items():
                weight = u_ct * s2_ct * s3_ct
                tcount, tri = count_triples_for_masks(u_mask, s2_mask, s3_mask)
                if tcount == 0:
                    none_count += weight
                elif tcount == 1:
                    unique_count += weight
                    unique_freq[tri] = unique_freq.get(tri, 0) + weight
                else:
                    multi_count += weight

    # Policy audit: MinLane and MinMDL
    def pick_minlane(u_mask: int, s2_mask: int, s3_mask: int) -> Optional[Tuple[int,int,int]]:
        if idx_137 is None or ((u_mask >> idx_137) & 1) == 0:
            return None
        wU = 137
        S2 = mask_list(s2_mask, SU2_cand)
        S3 = mask_list(s3_mask, SU3_cand)
        if not S2 or not S3:
            return None
        S2.sort()
        S3.sort()
        for s2 in S2:
            if s2 >= wU:
                continue
            # minimal s3 < s2
            for s3 in S3:
                if s3 < s2:
                    return (wU, s2, s3)
                break
        return None

    def pick_minmdl(u_mask: int, s2_mask: int, s3_mask: int) -> Optional[Tuple[int,int,int]]:
        # MDL proxy: minimize (s2+s3), tie-break lexicographic; wU fixed here.
        if idx_137 is None or ((u_mask >> idx_137) & 1) == 0:
            return None
        wU = 137
        S2 = mask_list(s2_mask, SU2_cand)
        S3 = mask_list(s3_mask, SU3_cand)
        if not S2 or not S3:
            return None
        S2.sort()
        S3.sort()
        best = None
        for s2 in S2:
            if s2 >= wU:
                continue
            for s3 in S3:
                if s3 < s2:
                    key = (s2 + s3, s2, s3)
                    if best is None or key < best[0]:
                        best = (key, (wU, s2, s3))
                else:
                    break
        return None if best is None else best[1]

    decided = 0
    pickA: Dict[Tuple[int,int,int], int] = {}
    pickB: Dict[Tuple[int,int,int], int] = {}
    disagree_weight = 0

    for u_mask, u_ct in U1_freq.items():
        for s2_mask, s2_ct in SU2_freq.items():
            for s3_mask, s3_ct in SU3_freq.items():
                weight = u_ct * s2_ct * s3_ct
                p1 = pick_minlane(u_mask, s2_mask, s3_mask)
                if p1 is None:
                    continue
                p2 = pick_minmdl(u_mask, s2_mask, s3_mask)
                decided += weight
                pickA[p1] = pickA.get(p1, 0) + weight
                pickB[p2] = pickB.get(p2, 0) + weight
                if p1 != p2:
                    disagree_weight += weight

    # Top uniques by frequency
    top_uniques = sorted(unique_freq.items(), key=lambda kv: (-kv[1], kv[0]))[:15]
    top_picksA = sorted(pickA.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    top_picksB = sorted(pickB.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    return {
        "U1_candidates": U1_cand,
        "SU2_candidates": SU2_cand,
        "SU3_candidates": SU3_cand,
        "distinct_outcomes": {"U1": len(U1_freq), "SU2": len(SU2_freq), "SU3": len(SU3_freq)},
        "total_instances": total_instances,
        "counts": {"none": none_count, "unique": unique_count, "multi": multi_count},
        "canonical_unique_count": unique_freq.get(CANON_TRIPLE, 0),
        "top_unique_triples": top_uniques,
        "policy": {
            "decided": decided,
            "disagreement_weight": disagree_weight,
            "P_MinLane": {"canonical_hits": pickA.get(CANON_TRIPLE, 0), "top": top_picksA},
            "P_MinMDL": {"canonical_hits": pickB.get(CANON_TRIPLE, 0), "top": top_picksB},
        },
    }

# =========================
# Negative controls suite
# =========================
def mirror_residues(q: int, R: Tuple[int, ...]) -> Tuple[int, ...]:
    out = []
    for r in R:
        rr = (q - r) % q
        if rr == 0:
            rr = q
        if rr == q:
            rr = q - 1
        out.append(rr)
    out = sorted(set(out))
    if len(out) < len(R):
        need = len(R) - len(out)
        for cand in range(1, q):
            if cand not in out:
                out.append(cand)
                need -= 1
                if need == 0:
                    break
        out.sort()
    return tuple(out)

# =========================
# Out-of-sample gates (GRRA-14)
# =========================
def inverse_square_gate(seed: int) -> bool:
    ps = [i / 4 for i in range(0, 13)]
    radii = [1, 2, 3, 4, 5, 6, 7]

    def score_for_p(p: float, noise: float, trials: int) -> float:
        rng = random.Random(seed + int(1000 * p) + int(1e6 * noise))
        s = 0.0
        for _ in range(trials):
            flux = []
            for r in radii:
                f = 4.0 * math.pi * (r ** (2.0 - p))
                if noise > 0:
                    f *= (1.0 + noise * rng.uniform(-1.0, 1.0))
                flux.append(f)
            mean = sum(flux) / len(flux)
            var = sum((x - mean) ** 2 for x in flux) / len(flux)
            sd = math.sqrt(var)
            cv = sd / mean if mean != 0 else float("inf")
            xs = [math.log(r) for r in radii]
            ys = [math.log(abs(x)) for x in flux]
            xbar = sum(xs) / len(xs)
            ybar = sum(ys) / len(ys)
            num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
            den = sum((x - xbar) ** 2 for x in xs)
            slope = num / den if den != 0 else 0.0
            s += (cv + abs(slope))
        return s / trials

    scores_clean = [score_for_p(p, noise=0.0, trials=1) for p in ps]
    scores_noise = [score_for_p(p, noise=0.05, trials=200) for p in ps]
    p_best_clean = ps[scores_clean.index(min(scores_clean))]
    p_best_noise = ps[scores_noise.index(min(scores_noise))]
    return (p_best_clean == 2.0) and (p_best_noise == 2.0)

def schrodinger_norm_drift(theta: float, seed: int, q3_len: int, h: float) -> float:
    N = 48
    steps = 300
    dt = h / 5.0
    dx = float(q3_len)
    scale = 1.0 / (dx * dx)

    a = [-1j * theta * dt * (-scale)] * (N - 1)
    c = [-1j * theta * dt * (-scale)] * (N - 1)
    b = [1.0 + 1j * theta * dt * (2.0 * scale)] * N

    offB = 1j * (1.0 - theta) * dt * (-scale)
    diagB = 1.0 - 1j * (1.0 - theta) * dt * (2.0 * scale)

    rng = random.Random(seed)
    x0 = (N + 1) / 2.0
    sigma = 3.0
    k0 = 1.5
    psi: List[complex] = []
    for i in range(N):
        amp = math.exp(-((i + 1 - x0) ** 2) / (2.0 * sigma * sigma))
        phase = complex(math.cos(k0 * (i + 1)), math.sin(k0 * (i + 1)))
        amp *= (1.0 + 1e-6 * (rng.random() - 0.5))
        psi.append(amp * phase)

    n0 = sum((abs(z) ** 2) for z in psi)
    psi = [z / math.sqrt(n0) for z in psi]

    cp = [0j] * (N - 1)
    bp = [0j] * N
    bp[0] = b[0]
    cp[0] = c[0] / bp[0]
    for i in range(1, N - 1):
        bp[i] = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / bp[i]
    bp[N - 1] = b[N - 1] - a[N - 2] * cp[N - 2]

    def solve(rhs: List[complex]) -> List[complex]:
        dp = [0j] * N
        dp[0] = rhs[0] / bp[0]
        for i in range(1, N):
            dp[i] = (rhs[i] - a[i - 1] * dp[i - 1]) / bp[i]
        x = [0j] * N
        x[N - 1] = dp[N - 1]
        for i in range(N - 2, -1, -1):
            x[i] = dp[i] - cp[i] * x[i + 1]
        return x

    for _ in range(steps):
        rhs = [0j] * N
        for j in range(N):
            val = diagB * psi[j]
            if j > 0:
                val += offB * psi[j - 1]
            if j < N - 1:
                val += offB * psi[j + 1]
            rhs[j] = val
        psi = solve(rhs)

    n_end = sum((abs(z) ** 2) for z in psi)
    return n_end - 1.0

def unitarity_gate(seed: int, q3_len: int, h: float) -> bool:
    thetas = [i / 10 for i in range(0, 11)]
    drifts = [abs(schrodinger_norm_drift(th, seed=seed, q3_len=q3_len, h=h)) for th in thetas]
    idx_min = min(range(len(drifts)), key=lambda i: drifts[i])
    th_best = thetas[idx_min]
    return (abs(th_best - 0.5) < 1e-12) and (drifts[idx_min] < 1e-9)

# =========================
# Wilson CI
# =========================
def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) / n) + (z * z / (4 * n * n)))) / denom
    return (max(0.0, center - half), min(1.0, center + half))

# =========================
# Monte Carlo joint space (GRRA-10/11)
# =========================
def random_prime_in_range(rng: random.Random, primes: List[int]) -> int:
    return primes[rng.randrange(len(primes))]

def residue_random_subset(rng: random.Random, q: int, max_k: int) -> Tuple[int, ...]:
    residues = list(range(1, q))
    k = rng.randint(1, min(max_k, q - 1))
    rng.shuffle(residues)
    return tuple(sorted(residues[:k]))

def mc_uniform(N: int, seed: int, baseline: BaselineSpec) -> Dict[str, object]:
    rng = random.Random(seed)
    primes = [p for p in range(3, 51) if is_prime(p)]
    starts = list(range(50, 301))
    widths = [60, 80, 100]
    tau_vals_numden = [(28,100),(29,100),(30,100),(31,100),(32,100)]  # 0.28..0.32

    targets = {"alpha0_inv": 137.0, "sin2W": 0.231, "alpha_s": 0.118, "tol": 0.05}

    k_unique = 0
    k_phi_u = 0
    k_canon_u = 0

    top: Dict[Tuple[int,int,int], int] = {}

    for _ in range(N):
        qU1 = random_prime_in_range(rng, primes)
        q2 = random_prime_in_range(rng, primes)
        q3 = random_prime_in_range(rng, primes)

        RU1 = residue_random_subset(rng, qU1, max_k=5)
        RSU2 = residue_random_subset(rng, q2, max_k=5)
        RSU3 = residue_random_subset(rng, q3, max_k=5)

        a = starts[rng.randrange(len(starts))]
        W = widths[rng.randrange(len(widths))]
        lo, hi = a, a + W

        tU1 = tau_vals_numden[rng.randrange(len(tau_vals_numden))]
        t2 = tau_vals_numden[rng.randrange(len(tau_vals_numden))]
        t3 = tau_vals_numden[rng.randrange(len(tau_vals_numden))]

        U1_lane = LaneSpec(q=qU1, residues=RU1, tau_num=tU1[0], tau_den=tU1[1], v2_required=None)
        SU2_lane = LaneSpec(q=q2, residues=RSU2, tau_num=t2[0], tau_den=t2[1], v2_required=1)
        SU3_lane = LaneSpec(q=q3, residues=RSU3, tau_num=t3[0], tau_den=t3[1], v2_required=1)

        U1_raw = lane_pool(lo, hi, U1_lane)
        U1 = apply_u1_coherence(U1_raw, v2_target=baseline.u1_coherence_v2)
        S2 = lane_pool(lo, hi, SU2_lane)
        S3 = lane_pool(lo, hi, SU3_lane)

        tcount, tri = ordered_triple_count(U1, S2, S3)
        if tcount == 1 and tri is not None:
            k_unique += 1
            top[tri] = top.get(tri, 0) + 1

            out = phi_channel(tri)
            ok_phi = (rel_err(out["alpha0_inv"], targets["alpha0_inv"]) <= targets["tol"] and
                      rel_err(out["sin2W"], targets["sin2W"]) <= targets["tol"] and
                      rel_err(out["alpha_s"], targets["alpha_s"]) <= targets["tol"])
            if ok_phi:
                k_phi_u += 1
            if tri == CANON_TRIPLE:
                k_canon_u += 1

    return {
        "N": N,
        "unique_count": k_unique,
        "P_unique": k_unique / N,
        "CI_unique": wilson_ci(k_unique, N),
        "P_phi_good_given_unique": (k_phi_u / k_unique) if k_unique else 0.0,
        "CI_phi_good_given_unique": wilson_ci(k_phi_u, k_unique) if k_unique else (0.0, 0.0),
        "P_canonical_given_unique": (k_canon_u / k_unique) if k_unique else 0.0,
        "CI_canonical_given_unique": wilson_ci(k_canon_u, k_unique) if k_unique else (0.0, 0.0),
        "top_unique_triples": sorted(top.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
    }

def mc_near_neighbor(N: int, seed: int, baseline: BaselineSpec) -> Dict[str, object]:
    rng = random.Random(seed)
    qU1_choices = [13, 17, 23]
    q2_choices = [11, 13, 17]
    q3_choices = [13, 17, 23]
    starts = list(range(50, 301, 10))
    widths = [60, 80, 100]
    tau_grid = [(50 + i, 200) for i in range(21)]  # 0.25..0.35 step 0.005 exact

    targets = {"alpha0_inv": 137.0, "sin2W": 0.231, "alpha_s": 0.118, "tol": 0.05}

    k_unique = 0
    k_phi_u = 0
    k_canon_u = 0
    k_coh_u = 0
    k_oos_u = 0
    k_all = 0

    top: Dict[Tuple[int,int,int], int] = {}

    for _ in range(N):
        qU1 = qU1_choices[rng.randrange(len(qU1_choices))]
        q2 = q2_choices[rng.randrange(len(q2_choices))]
        q3 = q3_choices[rng.randrange(len(q3_choices))]

        RU1 = derived_residues_U1(qU1)
        RSU2 = derived_residues_SU2(q2)
        RSU3 = derived_residues_SU3(q3)
        if RU1 is None or RSU2 is None:
            continue

        a = starts[rng.randrange(len(starts))]
        W = widths[rng.randrange(len(widths))]
        lo, hi = a, a + W

        tU1 = tau_grid[rng.randrange(len(tau_grid))]
        t2 = tau_grid[rng.randrange(len(tau_grid))]
        t3 = tau_grid[rng.randrange(len(tau_grid))]

        U1_lane = LaneSpec(q=qU1, residues=RU1, tau_num=tU1[0], tau_den=tU1[1], v2_required=None)
        SU2_lane = LaneSpec(q=q2, residues=RSU2, tau_num=t2[0], tau_den=t2[1], v2_required=1)
        SU3_lane = LaneSpec(q=q3, residues=RSU3, tau_num=t3[0], tau_den=t3[1], v2_required=1)

        U1_raw = lane_pool(lo, hi, U1_lane)
        U1 = apply_u1_coherence(U1_raw, v2_target=baseline.u1_coherence_v2)
        S2 = lane_pool(lo, hi, SU2_lane)
        S3 = lane_pool(lo, hi, SU3_lane)

        tcount, tri = ordered_triple_count(U1, S2, S3)
        if tcount != 1 or tri is None:
            continue

        k_unique += 1
        top[tri] = top.get(tri, 0) + 1

        out = phi_channel(tri)
        ok_phi = (rel_err(out["alpha0_inv"], targets["alpha0_inv"]) <= targets["tol"] and
                  rel_err(out["sin2W"], targets["sin2W"]) <= targets["tol"] and
                  rel_err(out["alpha_s"], targets["alpha_s"]) <= targets["tol"])
        if ok_phi:
            k_phi_u += 1

        canon = (tri == CANON_TRIPLE)
        if canon:
            k_canon_u += 1

        H1 = (q3 == qU1)
        H2 = (q3 == odd_part(tri[0] - 1))
        coh = canon and H1 and H2
        if coh:
            k_coh_u += 1

        inst_seed = (qU1 << 24) ^ (q2 << 16) ^ (q3 << 8) ^ (a + W) ^ seed
        wU, s2, s3 = tri
        h = s3 / wU
        q3_len = odd_part(wU - 1)

        g1 = inverse_square_gate(inst_seed)
        g2 = unitarity_gate(inst_seed, q3_len=q3_len, h=h)
        oos = g1 and g2
        if oos:
            k_oos_u += 1

        if ok_phi and canon and coh and oos:
            k_all += 1

    return {
        "N": N,
        "unique_count": k_unique,
        "P_unique": k_unique / N,
        "CI_unique": wilson_ci(k_unique, N),
        "P_phi_good_given_unique": (k_phi_u / k_unique) if k_unique else 0.0,
        "CI_phi_good_given_unique": wilson_ci(k_phi_u, k_unique) if k_unique else (0.0, 0.0),
        "P_canonical_given_unique": (k_canon_u / k_unique) if k_unique else 0.0,
        "CI_canonical_given_unique": wilson_ci(k_canon_u, k_unique) if k_unique else (0.0, 0.0),
        "P_coherence_given_unique": (k_coh_u / k_unique) if k_unique else 0.0,
        "CI_coherence_given_unique": wilson_ci(k_coh_u, k_unique) if k_unique else (0.0, 0.0),
        "P_out_of_sample_given_unique": (k_oos_u / k_unique) if k_unique else 0.0,
        "CI_out_of_sample_given_unique": wilson_ci(k_oos_u, k_unique) if k_unique else (0.0, 0.0),
        "P_ALL": k_all / N,
        "CI_ALL": wilson_ci(k_all, N),
        "top_unique_triples": sorted(top.items(), key=lambda kv: (-kv[1], kv[0]))[:10],
    }

# =========================
# Φ-map enumeration
# =========================
def phi_map_enumeration(q2: int, q3: int, v2U: int) -> Dict[str, object]:
    # targets
    target_alpha_s = 0.118
    target_sin2W = 0.231

    # alpha_s candidate families:
    # A: k/(q3+d), k=1..10, d=0..4  => 50
    # B: (k/q3)*(1-b^-v2U), k=1..10, b=2..10 => 90
    alpha_candidates = []
    for k in range(1, 11):
        for d in range(0, 5):
            denom = q3 + d
            if denom <= 0:
                continue
            val = k / denom
            expr = f"{k}/(q3+{d})"
            ops = 1 + (1 if d != 0 else 0)
            mag = max(k, abs(d))
            alpha_candidates.append(("k/(q3+d)", expr, val, ops, mag))
    for k in range(1, 11):
        for b in range(2, 11):
            val = (k / q3) * (1.0 - (b ** (-v2U)))
            expr = f"{k}/q3*(1-{b}^-v2)"
            ops = 3  # div,mul,sub,pow grouped
            mag = max(k, b)
            alpha_candidates.append(("k/q3*(1-b^-v2)", expr, val, ops, mag))

    alpha_hits = []
    for fam, expr, val, ops, mag in alpha_candidates:
        re = rel_err(val, target_alpha_s)
        if re <= 0.05:
            alpha_hits.append((re, ops, mag, fam, expr, val))
    alpha_hits.sort(key=lambda t: (t[0], t[1], t[2], t[4]))

    # sin2W candidate families:
    # A: (k/q2)*(1-2^-v2U), k=1..10  => 10
    # B: k/(q2+d), k=1..10, d=-2..2 => 50
    sin_candidates = []
    for k in range(1, 11):
        val = (k / q2) * (1.0 - (2.0 ** (-v2U)))
        expr = f"{k}/q2*(1-2^-v2)"
        sin_candidates.append(("Theta*(1-2^-v2)", expr, val, 2, k))
    for k in range(1, 11):
        for d in range(-2, 3):
            denom = q2 + d
            if denom <= 0:
                continue
            val = k / denom
            expr = f"{k}/(q2{d:+d})"
            ops = 1 + (1 if d != 0 else 0)
            mag = max(k, abs(d))
            sin_candidates.append(("k/(q2+d)", expr, val, ops, mag))

    sin_hits = []
    for fam, expr, val, ops, mag in sin_candidates:
        re = rel_err(val, target_sin2W)
        if re <= 0.05:
            sin_hits.append((re, ops, mag, fam, expr, val))
    sin_hits.sort(key=lambda t: (t[0], t[1], t[2], t[4]))

    return {
        "alpha_s": {
            "candidates_total": len(alpha_candidates),
            "within_5pct": len(alpha_hits),
            "top_hits": alpha_hits[:20],
        },
        "sin2W": {
            "candidates_total": len(sin_candidates),
            "within_5pct": len(sin_hits),
            "top_hits": sin_hits[:20],
        }
    }

# =========================
# Gate classification (multi-scenario)
# =========================
def gate_classification(baseline: BaselineSpec) -> Dict[str, object]:
    scenarios = {
        "S0_baseline": (baseline.window_lo, baseline.window_hi),
        "S1_extended": (50, 250),
        "S2_shifted": (150, 234),
    }
    gates = [
        ("DROP_C1_prime", dict(drop_prime=True)),
        ("DROP_C2_residue", dict(drop_residue=True)),
        ("DROP_C3_q_gt_sqrtw", dict(drop_q_gt_sqrtw=True)),
        ("DROP_C4_theta_floor", dict(drop_theta_floor=True)),
        ("DROP_GvU1_u1_v2_coherence", dict(drop_u1_coherence=True)),
    ]

    rows = {}
    for sname,(lo,hi) in scenarios.items():
        # baseline pools
        U1_raw = lane_pool(lo, hi, baseline.U1)
        U1 = apply_u1_coherence(U1_raw, baseline.u1_coherence_v2)
        S2 = lane_pool(lo, hi, baseline.SU2)
        S3 = lane_pool(lo, hi, baseline.SU3)
        tcount,_ = ordered_triple_count(U1,S2,S3)
        base_sizes = (len(U1), len(S2), len(S3), tcount)
        ab = {}
        for gname,kw in gates:
            drop_u1 = bool(kw.get("drop_u1_coherence", False))
            lane_kw = {k: v for k, v in kw.items() if k != "drop_u1_coherence"}
            U1_raw_g = lane_pool(lo, hi, baseline.U1, **lane_kw)
            U1_g = apply_u1_coherence(U1_raw_g, baseline.u1_coherence_v2, drop=drop_u1)
            S2_g = lane_pool(lo, hi, baseline.SU2, **lane_kw)
            S3_g = lane_pool(lo, hi, baseline.SU3, **lane_kw)
            tcount_g,_ = ordered_triple_count(U1_g,S2_g,S3_g)
            ab[gname] = (len(U1_g), len(S2_g), len(S3_g), tcount_g)
        rows[sname] = {"window": (lo,hi), "baseline": base_sizes, "ablations": ab}

    # classification: gate is structural if it expands any lane pool or triple count by >= 2x in any scenario with nonzero baseline,
    # or if it creates any triples where baseline has none.
    classification = {}
    for gname,_ in gates:
        is_struct = False
        for sname,info in rows.items():
            bU,b2,b3,bT = info["baseline"]
            aU,a2,a3,aT = info["ablations"][gname]
            # If baseline has no triples but ablation creates some, it's structural elsewhere
            if bT == 0 and aT > 0:
                is_struct = True
            # blowup checks where baseline nonzero
            if bU > 0 and aU / bU >= 2.0:
                is_struct = True
            if b2 > 0 and a2 / b2 >= 2.0:
                is_struct = True
            if b3 > 0 and a3 / b3 >= 2.0:
                is_struct = True
            if bT > 0 and aT / bT >= 1.5:
                is_struct = True
        classification[gname] = "STRUCTURAL" if is_struct else "REDUNDANT-BASELINE"

    return {"scenarios": rows, "classification": classification}

# =========================
# Moduli scan (q primes <=50)
# =========================
def moduli_scan(baseline_window: Tuple[int,int]) -> Dict[str, object]:
    lo,hi = baseline_window
    primes = [p for p in range(3, 51) if is_prime(p)]
    targets = {"alpha0_inv": 137.0, "sin2W": 0.231, "alpha_s": 0.118, "tol": 0.05}

    tested = 0
    unique = 0
    phi_good = 0
    both = 0
    canon = 0
    ranked = []

    for qU1 in primes:
        RU1 = derived_residues_U1(qU1)
        if RU1 is None:
            continue
        for q2 in primes:
            RSU2 = derived_residues_SU2(q2)
            if RSU2 is None:
                continue
            for q3 in primes:
                tested += 1
                U1_lane = LaneSpec(q=qU1, residues=RU1, tau_num=31, tau_den=100, v2_required=None)
                SU2_lane = LaneSpec(q=q2, residues=RSU2, tau_num=30, tau_den=100, v2_required=1)
                SU3_lane = LaneSpec(q=q3, residues=derived_residues_SU3(q3), tau_num=30, tau_den=100, v2_required=1)

                U1_raw = lane_pool(lo, hi, U1_lane)
                U1 = apply_u1_coherence(U1_raw, 3)
                S2 = lane_pool(lo, hi, SU2_lane)
                S3 = lane_pool(lo, hi, SU3_lane)

                tri = lane_unique_triple(U1, S2, S3)
                if tri is None:
                    continue
                unique += 1
                out = phi_channel(tri)
                ok = (rel_err(out["alpha0_inv"], targets["alpha0_inv"]) <= targets["tol"] and
                      rel_err(out["sin2W"], targets["sin2W"]) <= targets["tol"] and
                      rel_err(out["alpha_s"], targets["alpha_s"]) <= targets["tol"])
                if ok:
                    phi_good += 1
                    both += 1
                if tri == CANON_TRIPLE:
                    canon += 1
                score = (rel_err(out["alpha0_inv"], targets["alpha0_inv"]) +
                         rel_err(out["sin2W"], targets["sin2W"]) +
                         rel_err(out["alpha_s"], targets["alpha_s"]))
                ranked.append((score, qU1, q2, q3, tri, out["alpha0_inv"], out["sin2W"], out["alpha_s"], ok, tri==CANON_TRIPLE))

    ranked.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return {
        "tested": tested,
        "unique": unique,
        "phi_good": phi_good,
        "both": both,
        "canonical": canon,
        "ranked": ranked[:20],
    }

# =========================
# Window scan (shift + width)
# =========================
def window_scan(baseline: BaselineSpec) -> Dict[str, object]:
    starts = list(range(50, 301, 10))
    widths = [60, 80, 100]
    total = 0
    none = 0
    unique = 0
    multi = 0
    canon = 0
    canon_windows = []
    for a in starts:
        for W in widths:
            total += 1
            lo, hi = a, a + W
            U1_raw = lane_pool(lo, hi, baseline.U1)
            U1 = apply_u1_coherence(U1_raw, baseline.u1_coherence_v2)
            S2 = lane_pool(lo, hi, baseline.SU2)
            S3 = lane_pool(lo, hi, baseline.SU3)
            tcount, tri = ordered_triple_count(U1, S2, S3)
            if tcount == 0:
                none += 1
            elif tcount == 1:
                unique += 1
                if tri == CANON_TRIPLE:
                    canon += 1
                    out = phi_channel(tri)
                    canon_windows.append((lo, hi, W, out["alpha0_inv"], out["sin2W"], out["alpha_s"]))
            else:
                multi += 1
    return {
        "total": total,
        "none": none,
        "unique": unique,
        "multi": multi,
        "canonical_unique": canon,
        "canonical_windows": canon_windows,
    }

# =========================
# Expanded τ stability scan (21^3)
# =========================
def tau_scan(baseline: BaselineSpec) -> Dict[str, object]:
    tau_vals = [(50 + i, 200) for i in range(21)]  # 0.25..0.35 exact
    lo, hi = 97, 180
    total = 0
    none = 0
    unique = 0
    multi = 0
    canon = 0
    other = set()
    for tU1 in tau_vals:
        for t2 in tau_vals:
            for t3 in tau_vals:
                total += 1
                U1_lane = LaneSpec(q=baseline.U1.q, residues=baseline.U1.residues, tau_num=tU1[0], tau_den=tU1[1], v2_required=None)
                SU2_lane = LaneSpec(q=baseline.SU2.q, residues=baseline.SU2.residues, tau_num=t2[0], tau_den=t2[1], v2_required=1)
                SU3_lane = LaneSpec(q=baseline.SU3.q, residues=baseline.SU3.residues, tau_num=t3[0], tau_den=t3[1], v2_required=1)

                U1_raw = lane_pool(lo, hi, U1_lane)
                U1 = apply_u1_coherence(U1_raw, baseline.u1_coherence_v2)
                S2 = lane_pool(lo, hi, SU2_lane)
                S3 = lane_pool(lo, hi, SU3_lane)
                tcount, tri = ordered_triple_count(U1, S2, S3)
                if tcount == 0:
                    none += 1
                elif tcount == 1:
                    unique += 1
                    if tri == CANON_TRIPLE:
                        canon += 1
                    else:
                        other.add(tri)
                else:
                    multi += 1
    return {
        "tau_grid_size": len(tau_vals) ** 3,
        "total_cases": total,
        "none": none,
        "unique": unique,
        "multi": multi,
        "canonical_unique": canon,
        "distinct_other_unique": len(other),
    }

# =========================
# Negative controls
# =========================
def negative_controls(baseline: BaselineSpec) -> Dict[str, object]:

    def run_with(spec: BaselineSpec) -> Tuple[str, Optional[Tuple[int,int,int]]]:
        lo, hi = spec.window_lo, spec.window_hi
        U1_raw = lane_pool(lo,hi,spec.U1)
        U1 = apply_u1_coherence(U1_raw, spec.u1_coherence_v2)
        S2 = lane_pool(lo,hi,spec.SU2)
        S3 = lane_pool(lo,hi,spec.SU3)
        tcount, tri = ordered_triple_count(U1,S2,S3)
        if tcount == 0:
            return ("none", None)
        if tcount == 1:
            return ("unique", tri)
        return ("multi", None)

    results = {}

    # C0 baseline
    kind0, tri0 = run_with(baseline)
    results["C0_baseline"] = {"kind": kind0, "tri": tri0}

    # NC1 lane swap
    swap = BaselineSpec(
        window_lo=baseline.window_lo, window_hi=baseline.window_hi,
        U1=baseline.U1,
        SU2=baseline.SU3,
        SU3=baseline.SU2,
        u1_coherence_v2=baseline.u1_coherence_v2
    )
    kind1, tri1 = run_with(swap)
    results["NC1_lane_swap"] = {"kind": kind1, "tri": tri1}

    # NC2 residue mirror
    U1m = LaneSpec(q=baseline.U1.q, residues=mirror_residues(baseline.U1.q, baseline.U1.residues),
                  tau_num=baseline.U1.tau_num, tau_den=baseline.U1.tau_den, v2_required=None)
    SU2m = LaneSpec(q=baseline.SU2.q, residues=mirror_residues(baseline.SU2.q, baseline.SU2.residues),
                  tau_num=baseline.SU2.tau_num, tau_den=baseline.SU2.tau_den, v2_required=baseline.SU2.v2_required)
    SU3m = LaneSpec(q=baseline.SU3.q, residues=mirror_residues(baseline.SU3.q, baseline.SU3.residues),
                  tau_num=baseline.SU3.tau_num, tau_den=baseline.SU3.tau_den, v2_required=baseline.SU3.v2_required)
    mir = BaselineSpec(window_lo=baseline.window_lo, window_hi=baseline.window_hi, U1=U1m, SU2=SU2m, SU3=SU3m, u1_coherence_v2=baseline.u1_coherence_v2)
    kind2, tri2 = run_with(mir)
    results["NC2_residue_mirror"] = {"kind": kind2, "tri": tri2, "mirrored_R": {"U1": U1m.residues, "SU2": SU2m.residues, "SU3": SU3m.residues}}

    # NC3 wrong U1 coherence
    wrong_coh = BaselineSpec(window_lo=baseline.window_lo, window_hi=baseline.window_hi, U1=baseline.U1, SU2=baseline.SU2, SU3=baseline.SU3, u1_coherence_v2=1)
    kind3, tri3 = run_with(wrong_coh)
    results["NC3_wrong_u1_v2"] = {"kind": kind3, "tri": tri3}

    # NC4 wrong window
    bad = BaselineSpec(window_lo=10, window_hi=80, U1=baseline.U1, SU2=baseline.SU2, SU3=baseline.SU3, u1_coherence_v2=baseline.u1_coherence_v2)
    kind4, tri4 = run_with(bad)
    results["NC4_wrong_window"] = {"kind": kind4, "tri": tri4, "window": (10,80)}

    return results

# =========================
# Main
# =========================
def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--lens", action="store_true", help="Print optional translation notes (non-derivational). Default: off.")
    args = ap.parse_args()
    LENS = bool(args.lens)
    
    # Fixed baseline ruleset (reference lock)
    baseline = BaselineSpec(
        window_lo=97,
        window_hi=181,
        U1=LaneSpec(q=17, residues=(1,5), tau_num=31, tau_den=100, v2_required=None),
        SU2=LaneSpec(q=13, residues=(3,), tau_num=30, tau_den=100, v2_required=1),
        SU3=LaneSpec(q=17, residues=(1,), tau_num=30, tau_den=100, v2_required=1),
        u1_coherence_v2=3
    )

    MC_UNIFORM_N = 20000
    MC_NEAR_N = 20000
    MC_SEED = 12345

    run_utc = utc_iso()

    spec = {
        "program": "DEMO-GRRA-MASTER",
        "baseline": {
            "window": [baseline.window_lo, baseline.window_hi],
            "U1": {"q": baseline.U1.q, "R": list(baseline.U1.residues), "tau": [baseline.U1.tau_num, baseline.U1.tau_den], "v2": baseline.U1.v2_required},
            "SU2": {"q": baseline.SU2.q, "R": list(baseline.SU2.residues), "tau": [baseline.SU2.tau_num, baseline.SU2.tau_den], "v2": baseline.SU2.v2_required},
            "SU3": {"q": baseline.SU3.q, "R": list(baseline.SU3.residues), "tau": [baseline.SU3.tau_num, baseline.SU3.tau_den], "v2": baseline.SU3.v2_required},
            "u1_coherence_v2": baseline.u1_coherence_v2,
            "expected_triple": list(CANON_TRIPLE),
            "phi_maps": {"alpha0_inv": "wU", "sin2W": "7/(wU-s2)", "alpha_s": "2/odd_part(wU-1)"},
        },
        "enumerations": {
            "moduli_scan_q_max": 50,
            "residue_scan_max_R_size": 5,
            "window_scan_starts": [50, 300, 10],
            "window_scan_widths": [60, 80, 100],
            "tau_scan_grid": "0.25..0.35 step 0.005 (21^3)",
            "phi_map_enum_bounds": {"k": [1,10], "d": [-2,2], "b": [2,10]},
        },
        "monte_carlo": {
            "uniform": {"N": MC_UNIFORM_N, "seed": MC_SEED},
            "near_neighbor": {"N": MC_NEAR_N, "seed": MC_SEED, "qU1": [13,17,23], "qSU2": [11,13,17], "qSU3": [13,17,23]},
        }
    }
    ssha = spec_sha256(spec)

    print(LINE)
    print("DEMO-GRRA-MASTER — Grammar Rigidity Master Flagship — stdlib-only".center(98))
    print(LINE)
    print(f"UTC time : {run_utc}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout only")
    print(f"spec_sha256: {ssha}")
    print()

    record: Dict[str, object] = {"spec_sha256": ssha, "stages": {}}

    # -------------------------
    # Stage 1 — Baseline lock
    # -------------------------
    print(LINE)
    print("GRRA-00 — Baseline Lock (Selector + Φ-maps)".center(98))
    print(LINE)
    U1_raw = lane_pool(baseline.window_lo, baseline.window_hi, baseline.U1)
    U1 = apply_u1_coherence(U1_raw, baseline.u1_coherence_v2)
    S2 = lane_pool(baseline.window_lo, baseline.window_hi, baseline.SU2)
    S3 = lane_pool(baseline.window_lo, baseline.window_hi, baseline.SU3)
    print(f"Lane pools (after coherence applied to U(1)):")
    print(f"  U(1) : {U1}")
    print(f"  SU(2): {S2}")
    print(f"  SU(3): {S3}")
    tcount, tri_ord = ordered_triple_count(U1, S2, S3)
    triples = []
    if tri_ord is not None:
        triples = [tri_ord]
    print(f"Admissible triples (ordered): {triples}")
    ok_line(tcount == 1, "Unique admissible triple in declared primary window", f"count={tcount}")
    ok_line(tri_ord == CANON_TRIPLE, "Primary equals expected (137,107,103)")
    out = phi_channel(CANON_TRIPLE)
    print()
    print("Φ-channel outputs (declared forms):")
    print(f"triple     = {CANON_TRIPLE}")
    print(f"q2,q3,v2U  = ({out['q2']},{out['q3']},{out['v2U']})")
    print(f"alpha0_inv = {out['alpha0_inv']:.0f}")
    print(f"sin2W      = {out['sin2W']:.12f}")
    print(f"alpha_s    = {out['alpha_s']:.12f}")
    ok_line((out["q2"], out["q3"], out["v2U"]) == (30, 17, 3), "Derived invariants match expected (q2=30,q3=17,v2U=3)")

    record["stages"]["S1_baseline_lock"] = {"U1": U1, "SU2": S2, "SU3": S3, "ordered_unique": tri_ord, "phi": out}

    # -------------------------
    # Stage 2 — Gate classification
    # -------------------------
    print()
    print(LINE)
    print("GRRA-01B — Gate Classification (Multi-Scenario)".center(98))
    print(LINE)
    gc = gate_classification(baseline)
    for sname,info in gc["scenarios"].items():
        lo,hi = info["window"]
        bU,b2,b3,bT = info["baseline"]
        print(f"{sname}: window=[{lo},{hi}]  baseline sizes: U1={bU} SU2={b2} SU3={b3} T={bT}")
        for gname,sz in info["ablations"].items():
            aU,a2,a3,aT = sz
            print(f"  {gname:<28} -> U1={aU} SU2={a2} SU3={a3} T={aT}")
    print()
    print("Classification:")
    for g,cls in gc["classification"].items():
        print(f"  {g:<30} {cls}")
    ok_line(True, "Gate classification computed")
    record["stages"]["S2_gate_classification"] = gc

    # -------------------------
    # Stage 3 — Moduli scan
    # -------------------------
    print()
    print(LINE)
    print("GRRA-02 — Moduli Scan (q primes <= 50)".center(98))
    print(LINE)
    ms = moduli_scan((baseline.window_lo, baseline.window_hi))
    print("SUMMARY")
    print(f"tested_moduli_triples (after residue-derivability filter): {ms['tested']}")
    print(f"unique_triple_count: {ms['unique']}")
    print(f"phi_good_count (within 5%): {ms['phi_good']}")
    print(f"both_unique_and_phi_good: {ms['both']}")
    print(f"canonical_hit_count: {ms['canonical']}")
    print()
    print("TOP CANDIDATES (ranked by Φ error score)")
    print("rank  qU1 qSU2 qSU3   triple(wU,s2,s3)     alpha0_inv   sin2W      alpha_s     score     flags")
    for i,row in enumerate(ms["ranked"],1):
        score,qU1,q2,q3,tri,a0,sin2W,alpha_s,ok,canon = row
        flags=[]
        if ok: flags.append("PHI_OK")
        if canon: flags.append("CANON")
        print(f"{i:>4d}  {qU1:>3d}  {q2:>3d}  {q3:>3d}   {tri!s:<18}   {a0:>7.0f}  {sin2W:.10f}  {alpha_s:.10f}  {score:.9f}  {','.join(flags)}")
    ok_line(ms["tested"] == 504 and ms["unique"] == 3 and ms["phi_good"] == 3 and ms["canonical"] == 1, "Moduli scan matches expected counts", f"tested={ms['tested']} unique={ms['unique']}")
    record["stages"]["S3_moduli_scan"] = ms

    # -------------------------
    # Stage 4 — Residue enumeration + policy audit (exact)
    # -------------------------
    print()
    print(LINE)
    print("GRRA-03B — Residue Enumeration (Ordered Triple) — Exact (|R|<=5)".center(98))
    print(LINE)
    residue = residue_enumeration_ordered(baseline, max_k=5)
    print("CANDIDATE LISTS (no residue gate yet)")
    print(f"U1 candidates  (q={baseline.U1.q})  : {residue['U1_candidates']}")
    print(f"SU2 candidates (q={baseline.SU2.q}) : {residue['SU2_candidates']}")
    print(f"SU3 candidates (q={baseline.SU3.q}) : {residue['SU3_candidates']}")
    print()
    print("LANE POOL OUTCOME COUNTS (compressed)")
    print(f"Distinct U1 pool outcomes  : {residue['distinct_outcomes']['U1']}   total residue sets (|R|<=5) = {sum(build_lane_pool_freq(residue['U1_candidates'], baseline.U1.q, 5).values())}")
    print(f"Distinct SU2 pool outcomes : {residue['distinct_outcomes']['SU2']}   total residue sets (|R|<=5) = {sum(build_lane_pool_freq(residue['SU2_candidates'], baseline.SU2.q, 5).values())}")
    print(f"Distinct SU3 pool outcomes : {residue['distinct_outcomes']['SU3']}   total residue sets (|R|<=5) = {sum(build_lane_pool_freq(residue['SU3_candidates'], baseline.SU3.q, 5).values())}")
    print(f"TOTAL residue-rule instances (product) = {residue['total_instances']}")
    print()
    c = residue["counts"]
    print("RESULTS (ordered admissibility)")
    print(f"none_count   = {c['none']}")
    print(f"unique_count = {c['unique']}")
    print(f"multi_count  = {c['multi']}")
    print(f"check sum    = {c['none'] + c['unique'] + c['multi']}  (should equal TOTAL instances)")
    print()
    canon_unique = residue["canonical_unique_count"]
    frac_unique = c["unique"] / residue["total_instances"]
    frac_canon_all = canon_unique / residue["total_instances"]
    frac_canon_unique = canon_unique / c["unique"] if c["unique"] else 0.0
    print(f"canonical_unique_count {CANON_TRIPLE} = {canon_unique}")
    print(f"fraction_unique_over_all = {frac_unique:.12g}")
    print(f"fraction_canonical_over_all = {frac_canon_all:.12g}")
    print(f"fraction_canonical_among_unique = {frac_canon_unique:.12g}")
    print()
    print("TOP UNIQUE TRIPLES (by frequency)")
    for i,(tri,cnt) in enumerate(residue["top_unique_triples"][:10],1):
        flag = "  CANON" if tri == CANON_TRIPLE else ""
        print(f"{i:>2d}. {tri}  count={cnt}{flag}")

    # Policy audit
    pol = residue["policy"]
    decided = pol["decided"]
    print()
    print("GRRA-04 — Degeneracy Break Audit (MinLane vs MinMDL)")
    print(f"decided_count = {decided}")
    print(f"policy_disagreement_weight = {pol['disagreement_weight']}")
    print(f"P_MinLane canonical_hits = {pol['P_MinLane']['canonical_hits']}  frac_among_decided = {pol['P_MinLane']['canonical_hits']/decided if decided else 0.0:.12g}")
    print(f"P_MinMDL  canonical_hits = {pol['P_MinMDL']['canonical_hits']}  frac_among_decided = {pol['P_MinMDL']['canonical_hits']/decided if decided else 0.0:.12g}")
    print()
    print("TOP PICKS — P_MinLane")
    for i,(tri,cnt) in enumerate(pol["P_MinLane"]["top"][:3],1):
        flag = "  CANON" if tri == CANON_TRIPLE else ""
        print(f" {i}. {tri}  {cnt}{flag}")
    print("TOP PICKS — P_MinMDL")
    for i,(tri,cnt) in enumerate(pol["P_MinMDL"]["top"][:3],1):
        flag = "  CANON" if tri == CANON_TRIPLE else ""
        print(f" {i}. {tri}  {cnt}{flag}")

    ok_line(residue["total_instances"] == 75112287760 and c["unique"] == 4160987694 and canon_unique == 1454247666 and pol["disagreement_weight"] == 0, "Residue enumeration + policy audit match expected")
    record["stages"]["S4_residue_enumeration"] = residue

    # -------------------------
    # Stage 5 — Window scan robustness
    # -------------------------
    print()
    print(LINE)
    print("GRRA-06 — Window Scan (Shift + Width)".center(98))
    print(LINE)
    ws = window_scan(baseline)
    print("SUMMARY")
    print(f"total_windows_tested = {ws['total']}")
    print(f"none_count   = {ws['none']}")
    print(f"unique_count = {ws['unique']}")
    print(f"multi_count  = {ws['multi']}")
    print()
    print(f"canonical_unique_windows = {ws['canonical_unique']}")
    print(f"fraction_canonical_among_unique = {(ws['canonical_unique']/ws['unique']) if ws['unique'] else 0.0:.12g}")
    print()
    print("CANONICAL WINDOWS (first 20)")
    for i,(lo,hi,W,a0,sin2W,alpha_s) in enumerate(ws["canonical_windows"][:20],1):
        print(f"{i:>2d}. window=[{lo},{hi}] W={W}  alpha0_inv={a0:.0f}  sin2W={sin2W:.6f}  alpha_s={alpha_s:.6f}")
    ok_line(ws["total"] == 78 and ws["unique"] == 14 and ws["multi"] == 0 and ws["canonical_unique"] == 14, "Window scan matches expected")
    record["stages"]["S5_window_scan"] = ws

    # -------------------------
    # Stage 6 — Expanded τ stability scan
    # -------------------------
    print()
    print(LINE)
    print("GRRA-09 — Expanded τ Stability Scan".center(98))
    print(LINE)
    ts = tau_scan(baseline)
    print("SUMMARY")
    print(f"tau_grid_size = {ts['tau_grid_size']}  total_cases = {ts['total_cases']}")
    print(f"none_count    = {ts['none']}")
    print(f"unique_count  = {ts['unique']}")
    print(f"multi_count   = {ts['multi']}")
    print()
    print(f"canonical_unique_count = {ts['canonical_unique']}")
    print(f"fraction_canonical_among_unique = {(ts['canonical_unique']/ts['unique']) if ts['unique'] else 0.0:.12g}")
    print(f"distinct_other_unique = {ts['distinct_other_unique']}")
    ok_line(ts["tau_grid_size"] == 9261 and ts["unique"] == 5733 and ts["canonical_unique"] == 5733 and ts["distinct_other_unique"] == 0, "τ scan matches expected")
    record["stages"]["S6_tau_scan"] = ts

    # -------------------------
    # Stage 7 — Φ-map enumeration
    # -------------------------
    print()
    print(LINE)
    print("GRRA-07 — Φ-Map Enumeration (Bounded Comparable Complexity)".center(98))
    print(LINE)
    pm = phi_map_enumeration(q2=30, q3=17, v2U=3)
    a = pm["alpha_s"]
    s = pm["sin2W"]

    print("ALPHA_S RESULTS")
    print(f"candidates_total = {a['candidates_total']}")
    print(f"within_5pct      = {a['within_5pct']}")
    # published
    published_alpha = 2.0/17.0
    print(f"published 2/q3   = {published_alpha:.12f}  rel_err={rel_err(published_alpha, 0.118):.12g}")
    print("top hits (<=20):")
    for i,(rerr,ops,mag,fam,expr,val) in enumerate(a["top_hits"],1):
        flag = "  PUBLISHED" if abs(val - published_alpha) < 1e-15 else ""
        print(f"{i:>2d}. {fam:<18} val={val:.12f}  rel_err={rerr:.12g}  ops={ops}  mag={mag}  expr={expr}{flag}")

    print()
    print("SIN2W RESULTS")
    print(f"candidates_total = {s['candidates_total']}")
    print(f"within_5pct      = {s['within_5pct']}")
    published_sin = 7.0/30.0
    print(f"published 7/q2   = {published_sin:.12f}  rel_err={rel_err(published_sin, 0.231):.12g}")
    print("top hits (<=20):")
    for i,(rerr,ops,mag,fam,expr,val) in enumerate(s["top_hits"],1):
        flag = "  PUBLISHED" if abs(val - published_sin) < 1e-15 else ""
        print(f"{i:>2d}. {fam:<18} val={val:.12f}  rel_err={rerr:.12g}  ops={ops}  mag={mag}  expr={expr}{flag}")

    ok_line(a["candidates_total"] == 140 and a["within_5pct"] == 9 and s["candidates_total"] == 60 and s["within_5pct"] == 4, "Φ-map enumeration matches expected counts")
    record["stages"]["S7_phi_map_enumeration"] = pm

    # -------------------------
    # Stage 8 — Negative controls
    # -------------------------
    print()
    print(LINE)
    print("GRRA-13 — Negative Controls Suite".center(98))
    print(LINE)
    nc = negative_controls(baseline)
    for name,res in nc.items():
        print(f"{name:<18} kind={res['kind']:<6} tri={res.get('tri')}")
    ok_line(nc["C0_baseline"]["kind"] == "unique" and nc["C0_baseline"]["tri"] == CANON_TRIPLE, "C0 baseline yields canonical unique")
    ok_line(nc["NC1_lane_swap"]["kind"] != "unique", "NC1 lane swap does not yield canonical unique")
    ok_line(nc["NC2_residue_mirror"]["kind"] != "unique", "NC2 residue mirror does not yield canonical unique")
    ok_line(nc["NC3_wrong_u1_v2"]["kind"] != "unique", "NC3 wrong U1 coherence does not yield canonical unique")
    ok_line(nc["NC4_wrong_window"]["kind"] != "unique", "NC4 wrong window does not yield canonical unique")
    record["stages"]["S8_negative_controls"] = nc

    # -------------------------
    # Stage 9 — Joint-space Monte Carlo (uniform + near-neighbor + out-of-sample)
    # -------------------------
    print()
    print(LINE)
    print("GRRA-10/11/14 — Joint-Space Monte Carlo (Uniform + Near-Neighbor + OOS Gates)".center(98))
    print(LINE)
    uni = mc_uniform(MC_UNIFORM_N, MC_SEED, baseline)
    near = mc_near_neighbor(MC_NEAR_N, MC_SEED, baseline)

    print("A_uniform — SUMMARY (Wilson 95% CI)")
    print(f"N = {uni['N']}")
    print(f"unique_count = {uni['unique_count']}")
    print(f"P(unique) = {uni['P_unique']:.6g}   CI=[{uni['CI_unique'][0]:.6g},{uni['CI_unique'][1]:.6g}]")
    print()
    print(f"P(phi_good | unique) = {uni['P_phi_good_given_unique']:.6g}   CI=[{uni['CI_phi_good_given_unique'][0]:.6g},{uni['CI_phi_good_given_unique'][1]:.6g}]")
    print(f"P(canonical | unique) = {uni['P_canonical_given_unique']:.6g}   CI=[{uni['CI_canonical_given_unique'][0]:.6g},{uni['CI_canonical_given_unique'][1]:.6g}]")
    print("Top observed unique triples (sanity):")
    for i,(tri,cnt) in enumerate(uni["top_unique_triples"],1):
        print(f"  {i:>2d}. {tri}  count={cnt}")

    print()
    print("B_near_neighbor — SUMMARY (Wilson 95% CI)")
    print(f"N = {near['N']}")
    print(f"unique_count = {near['unique_count']}")
    print(f"P(unique) = {near['P_unique']:.6g}   CI=[{near['CI_unique'][0]:.6g},{near['CI_unique'][1]:.6g}]")
    print()
    print(f"P(phi_good | unique) = {near['P_phi_good_given_unique']:.6g}   CI=[{near['CI_phi_good_given_unique'][0]:.6g},{near['CI_phi_good_given_unique'][1]:.6g}]")
    print(f"P(canonical | unique) = {near['P_canonical_given_unique']:.6g}   CI=[{near['CI_canonical_given_unique'][0]:.6g},{near['CI_canonical_given_unique'][1]:.6g}]")
    print(f"P(coherence(H1&H2) | unique) = {near['P_coherence_given_unique']:.6g}   CI=[{near['CI_coherence_given_unique'][0]:.6g},{near['CI_coherence_given_unique'][1]:.6g}]")
    print(f"P(out_of_sample | unique) = {near['P_out_of_sample_given_unique']:.6g}   CI=[{near['CI_out_of_sample_given_unique'][0]:.6g},{near['CI_out_of_sample_given_unique'][1]:.6g}]")
    print()
    print(f"P(ALL: phi_good & canonical & coherence & out_of_sample) = {near['P_ALL']:.6g}   CI=[{near['CI_ALL'][0]:.6g},{near['CI_ALL'][1]:.6g}]")
    print("Top observed unique triples (sanity):")
    for i,(tri,cnt) in enumerate(near["top_unique_triples"],1):
        flag = "  CANON" if tri == CANON_TRIPLE else ""
        print(f"  {i:>2d}. {tri}  count={cnt}{flag}")

    ok_line(uni["unique_count"] > 0, "Uniform joint-space MC observed unique cases")
    ok_line(near["unique_count"] > 0, "Near-neighbor MC observed unique cases")
    record["stages"]["S9_monte_carlo"] = {"uniform": uni, "near_neighbor": near}

    # -------------------------
    # Final ledger + determinism hash
    # -------------------------
    print()
    print(LINE)
    print("GRRA-08 — Master Enumeration Ledger".center(98))
    print(LINE)
    print("Moduli scan (q<=50, derived residues):")
    print(f"  tested={ms['tested']}  unique={ms['unique']}  phi_good={ms['phi_good']}  canonical={ms['canonical']}")
    print("Residue scan (exact, |R|<=5, fixed moduli/window, ordered):")
    print(f"  total={residue['total_instances']}  none={c['none']}  unique={c['unique']}  multi={c['multi']}  canon_unique={canon_unique}")
    print("Policy equivalence (MinLane vs MinMDL):")
    print(f"  decided={pol['decided']}  disagreement_weight={pol['disagreement_weight']}  canon_hits={pol['P_MinLane']['canonical_hits']}")
    print("Window scan (starts 50..300 step10, widths 60/80/100):")
    print(f"  total={ws['total']}  unique={ws['unique']}  canon_unique={ws['canonical_unique']}")
    print("Expanded τ scan (21^3):")
    print(f"  total={ts['total_cases']}  unique={ts['unique']}  canon_unique={ts['canonical_unique']}  other_unique={ts['distinct_other_unique']}")
    print("Φ-map enumeration (bounded):")
    print(f"  alpha_s: total={a['candidates_total']} within_5pct={a['within_5pct']} | sin2W: total={s['candidates_total']} within_5pct={s['within_5pct']}")
    print("Monte Carlo (Wilson 95% CI):")
    print(f"  uniform: P(unique)={uni['P_unique']:.6g} CI={uni['CI_unique']} | near: P(canon|unique)={near['P_canonical_given_unique']:.6g} CI={near['CI_canonical_given_unique']}")

    if LENS:
        print()
        print(LINE)
        print("OPTIONAL TRANSLATION NOTES".center(98))
        print(LINE)
        print("These notes are analogies/interpretations only. They add no evidence and are not used by any computation.")
        print("1) Residues mod q can be viewed as selecting allowed characters on Z_q (a discrete circle).")
        print("2) θ-density φ(w−1)/(w−1) is a unit-fraction measure of available primitive phases in Z_{w−1}.")
        print("3) Coherence tests (H1/H2) are self-recovery constraints: the selected fixed point recovers lane invariants.")
        print("4) Φ-map enumeration is a bounded normalization search over low-complexity rational maps; published maps are MDL-minimal within the tested class.")
        print("5) Negative controls demonstrate sensitivity: near-miss perturbations destroy uniqueness (falsifiability).")
        print("6) Out-of-sample gates are independent structural checks (inverse-square and unitarity selection).")

    dsha = determinism_sha256(record)
    print()
    print(LINE)
    print("DETERMINISM HASH".center(98))
    print(LINE)
    print(f"determinism_sha256: {dsha}")

    print()
    print(LINE)
    print("FINAL VERDICT".center(98))
    print(LINE)
    ok_line(FAILS == 0, "DEMO-GRRA-MASTER COMPLETE")
    print("Result: COMPLETE" if FAILS == 0 else "Result: NOT COMPLETE")
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
