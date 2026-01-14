#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO-36 (BB) — Big Bang Master Flagship
Structural Cosmology Closure + Two Independent Spectrum Bridges + Deterministic Teeth
Referee-ready, portable, deterministic, no tuning.

SUMMARY (one paragraph)
-----------------------
A unique admissible prime triple (wU,s2,s3) is selected in a declared primary window by a transparent
lane filter and an explicit coherence constraint. All numerical budgets/tolerances are then derived
from that triple (q2,q3,eps,N,K). Using fixed BB-36 monomials, the triple deterministically generates
a full "structural cosmology" parameter set (H0, Ω's, A_s, n_s, τ, ℓ1, δ_CMB). Two *independent*
spectrum-level observables are then constructed from first principles (a debiased tilt proxy and a
power-sum amplitude proxy), each audited against an admissible operator (Fejér tensor; nonnegative
kernel) versus two illegal controls (sharp cutoff; signed HF-injecting control). Finally, deterministic
counterfactual triples induce budget shifts and must degrade the primary signature ("teeth").

NOTES FOR REFEREES
------------------
- This is not a cosmological inference engine; it is an internally consistent, falsifiable demonstration.
- No external data files are required.
- Optional: CAMB (if installed) is used only for an informational TT first-peak check.

Dependencies
------------
Required: numpy
Optional: camb, matplotlib

Artifacts
---------
If filesystem is writable:
  - bb36_master_results.json
  - bb36_master_plot.png (if matplotlib available)

Run
---
  python demo36_bb36_master_flagship_900k_referee_ready_v2.py
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import datetime as _dt
import hashlib
import json
import math
import platform
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np


# ============================================================
# Utilities: printing + stable hashing
# ============================================================

def utc_now_iso() -> str:
    return _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat().replace("+00:00", "Z")


def banner(title: str, ch: str = "=") -> None:
    line = ch * max(96, len(title))
    print(line)
    print(title)
    print(line)


def passfail(ok: bool, label: str, detail: str = "") -> bool:
    tag = "PASS" if ok else "FAIL"
    if detail:
        print(f"{tag:4s}  {label:<74s} {detail}")
    else:
        print(f"{tag:4s}  {label}")
    return ok


def stable_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_of_obj(obj) -> str:
    return sha256_bytes(stable_json(obj).encode("utf-8"))


def dataclass_asdict(x):
    return dataclasses.asdict(x)


def try_file_sha256(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            return sha256_bytes(f.read())
    except Exception:
        return None


# ============================================================
# Deterministic selection engine (primary + counterfactuals)
# ============================================================

@dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


def primes_in(lo: int, hi: int) -> List[int]:
    """Deterministic prime sieve for the small windows used here."""
    if hi < 2 or hi <= lo:
        return []
    n = hi + 1
    sieve = bytearray(b"\x01") * n
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(math.isqrt(hi)) + 1):
        if sieve[p]:
            start = p * p
            step = p
            sieve[start: hi + 1: step] = b"\x00" * (((hi - start) // step) + 1)
    return [x for x in range(max(2, lo), hi + 1) if sieve[x]]


def v2(n: int) -> int:
    """2-adic valuation v2(n) for n>0."""
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k


def odd_part(n: int) -> int:
    """Odd part n / 2^{v2(n)}."""
    while n % 2 == 0:
        n //= 2
    return n


def totient_ratio(p: int) -> float:
    """phi(p-1)/(p-1) for prime p."""
    n = p - 1
    nn = n
    phi = n
    d = 2
    while d * d <= nn:
        if nn % d == 0:
            while nn % d == 0:
                nn //= d
            phi = phi // d * (d - 1)
        d += 1
    if nn > 1:
        phi = phi // nn * (nn - 1)
    return phi / n


@dataclass(frozen=True)
class LaneSpec:
    name: str
    mod: int
    residues: Tuple[int, ...]
    window: Tuple[int, int]
    tot_ratio_min: float
    v2_required: Optional[int] = None


def lane_pool(spec: LaneSpec) -> List[int]:
    lo, hi = spec.window
    ps = primes_in(lo, hi)
    out = []
    for p in ps:
        if (p % spec.mod) not in spec.residues:
            continue
        if totient_ratio(p) < spec.tot_ratio_min:
            continue
        if spec.v2_required is not None and v2(p - 1) != spec.v2_required:
            continue
        out.append(p)
    return out


def select_primary_triple(window: Tuple[int, int] = (97, 200),
                          v2U_req: int = 3) -> Tuple[Triple, Dict[str, List[int]]]:
    """
    Deterministic primary selection (exactly as in the verified BB-1 v3 / BB-2 v3 prework).

    Lane specs (transparent, fixed):
      U(1):  p mod 17 in {1,5}, totient_ratio(p) >= 0.20
      SU(2): p mod 13 in {3},   totient_ratio(p) >= 0.20
      SU(3): p mod 17 in {1},   totient_ratio(p) >= 0.20, and v2(p-1)=1

    Then impose U(1) coherence: v2(wU-1)=v2U_req, and demand uniqueness.
    """
    pool_u1 = lane_pool(LaneSpec("U(1)", 17, (1, 5), window, 0.20))
    pool_s2 = lane_pool(LaneSpec("SU(2)", 13, (3,), window, 0.20))
    pool_s3 = lane_pool(LaneSpec("SU(3)", 17, (1,), window, 0.20, v2_required=1))

    U1_coherent = [p for p in pool_u1 if v2(p - 1) == v2U_req]
    if len(U1_coherent) != 1:
        raise RuntimeError(f"Primary U(1) coherence not unique: {U1_coherent}")

    if not pool_s2 or not pool_s3:
        raise RuntimeError(f"Primary lane pools unexpectedly empty: SU2={pool_s2} SU3={pool_s3}")

    primary = Triple(wU=U1_coherent[0], s2=pool_s2[0], s3=pool_s3[0])

    pools = {
        "U1_raw": pool_u1,
        "SU2_raw": pool_s2,
        "SU3_raw": pool_s3,
        "U1_coherent": U1_coherent,
        # legacy keys (compat)
        "SU2": pool_s2,
        "SU3": pool_s3,
    }
    return primary, pools


def counterfactual_triples(primary: Triple,
                           window: Tuple[int, int] = (181, 1200),
                           need: int = 4) -> List[Triple]:
    """
    Deterministic counterfactual set (matches BB-1 v3 / BB-2 v3 prework).

    Construction:
      pool_u1: primes p in window with p mod 17 in {1,5}, totient_ratio(p) >= 0.31, and v2(p-1)=3
      pool_s2: primes p in window with p mod 13 in {3},   totient_ratio(p) >= 0.30, and v2(p-1)=1
      pool_s3: primes p in window with p mod 17 in {1},   totient_ratio(p) >= 0.30, and v2(p-1)=1

      Choose wU = pool_u1[0], s2s = first 2 of pool_s2, s3s = first 2 of pool_s3
      => 4 counterfactual triples (wU,s2,s3).

    Counterfactuals act as "teeth": they change budgets (K) while the observable definitions stay fixed.
    """
    lo, hi = window
    ps = primes_in(lo, hi)

    def pool(mod: int, residues: Tuple[int, ...], tot_min: float, v2_req: Optional[int]) -> List[int]:
        out: List[int] = []
        for p in ps:
            if (p % mod) not in residues:
                continue
            if totient_ratio(p) < tot_min:
                continue
            if v2_req is not None and v2(p - 1) != v2_req:
                continue
            out.append(p)
        return out

    pool_u1 = pool(17, (1, 5), 0.31, 3)
    pool_s2 = pool(13, (3,), 0.30, 1)
    pool_s3 = pool(17, (1,), 0.30, 1)

    if len(pool_u1) < 1 or len(pool_s2) < 2 or len(pool_s3) < 2:
        return []

    wU = pool_u1[0]
    s2s = pool_s2[:2]
    s3s = pool_s3[:2]

    triples: List[Triple] = []
    for s2 in s2s:
        for s3 in s3s:
            t = Triple(wU=wU, s2=s2, s3=s3)
            if t != primary:
                triples.append(t)

    return triples[:need]


# ============================================================
# Budgets (derived from triple; no tuning)
# ============================================================

@dataclass(frozen=True)
class Budgets:
    q2: int
    q3: int
    v2U: int
    eps: float
    N: int
    K_primary: int
    K_truth: int


def budgets_from_triple(t: Triple, enforce_minN: int = 64) -> Budgets:
    """
    Budget law (identical to verified BB-1 v3 / BB-2 v3 prework).
    """
    q2 = t.wU - t.s2
    q3 = odd_part(t.wU - 1)
    v2U = v2(t.wU - 1)
    eps = 1.0 / math.sqrt(q2)

    N = max(enforce_minN, 2 ** (v2U + 3))
    K_truth = (N // 2) - 1
    K_max = (N // 4) - 1
    K_primary = int(round(K_max * (17.0 / q3)))
    K_primary = max(3, min(K_max, K_primary))

    return Budgets(q2=q2, q3=q3, v2U=v2U, eps=eps, N=N, K_primary=K_primary, K_truth=K_truth)


# ============================================================
# Structural cosmology closure (BB-36 monomials)
# ============================================================

def structural_cosmo_params(t: Triple) -> Dict[str, float]:
    """
    Structural cosmology monomials (identical to PREWORK BB-0 v1).

    Returns a dictionary with:
      H0, Omega_b, Omega_c, Omega_L, ombh2, omch2,
      A_s, n_s, tau, ell1, deltaCMB, delta0, F_CMB
    """
    wU, s2, s3 = t.wU, t.s2, t.s3
    q3 = odd_part(wU - 1)

    pi = math.pi
    e = math.e

    # Structural "monomials"
    H0 = (wU ** -6) * (s2 ** 1) * (s3 ** 2) * (q3 ** 7)
    Omega_b = (1.0 / e) * (s2 ** -1) * (s3 ** 3) * (q3 ** -4)
    Omega_c = (1.0 / e) * (s2 ** -1) * (s3 ** 2) * (q3 ** -2)
    Omega_L = 1.0 - Omega_b - Omega_c

    h = H0 / 100.0
    ombh2 = Omega_b * h * h
    omch2 = Omega_c * h * h

    A_s = (1.0 / (4.0 * pi)) * (wU ** 5) * (s2 ** -2) * (s3 ** -4) * (q3 ** -5)
    n_s = (1.0 / (4.0 * pi)) * (s2 ** -2) * (s3 ** 5) * (q3 ** -4)
    tau = (wU ** -3) * (s3 ** 5) * (q3 ** -4)

    ell1 = (1.0 / e) * (wU ** -7) * (s2 ** 4) * (s3 ** 6) * (q3 ** -2)

    delta0 = e * (wU ** -3) * (s2 ** 2) * (s3 ** -2)
    F_CMB = (1.0 / e) * (s2 ** 2) * (s3 ** -5) * (q3 ** 6)
    deltaCMB = F_CMB * delta0

    return {
        "H0": H0,
        "Omega_b": Omega_b,
        "Omega_c": Omega_c,
        "Omega_L": Omega_L,
        "ombh2": ombh2,
        "omch2": omch2,
        "A_s": A_s,
        "n_s": n_s,
        "tau": tau,
        "ell1": ell1,
        "deltaCMB": deltaCMB,
        "delta0": delta0,
        "F_CMB": F_CMB,
        "q3": float(q3),
    }


# ============================================================
# Spectrum utilities (tilt + deltaCMB proxies)
# ============================================================

def make_kgrid(N: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k1 = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(k1, k1, k1, indexing="ij")
    kr = np.sqrt(kx * kx + ky * ky + kz * kz)
    return kx, ky, kz, kr


def fejer_weights_1d(N: int, Kp: int) -> np.ndarray:
    k = np.fft.fftfreq(N) * N
    w = np.maximum(0.0, 1.0 - (np.abs(k) / (Kp + 1.0)))
    return w


def weights_3d(N: int, Kp: int, variant: str) -> np.ndarray:
    _, _, _, kr = make_kgrid(N)

    if variant == "fejer":
        w1 = fejer_weights_1d(N, Kp)
        return w1[:, None, None] * w1[None, :, None] * w1[None, None, :]

    if variant == "sharp":
        return (kr <= Kp).astype(np.float64)

    if variant == "signed":
        return np.where(kr <= Kp, 1.0, -1.0).astype(np.float64)

    raise ValueError(variant)


def observed_shell_means_3d(N: int, Kp: int, n_s: float, variant: str) -> Dict[int, float]:
    """
    Observed shell means of Delta^2_obs(k) on *integer shells*.

    This matches the verified BB-1 v3 / BB-2 v3 convention:
      shell = floor(|k| + 1e-12)  (so shell m collects modes with m <= |k| < m+1)

    Returns a dict {m: mean Delta^2_obs on shell m} for m>=1 up to max shell on the grid.
    """
    k = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    K = np.sqrt(kx * kx + ky * ky + kz * kz)

    P_base = np.where(K > 0, (K + 1e-9) ** (n_s - 4.0), 0.0)
    W = weights_3d(N, Kp, variant)
    P_obs = P_base * (W * W)
    D2_obs = (K ** 3) * P_obs

    shell = np.floor(K + 1e-12).astype(int)
    Kmax = int(shell.max())

    means: Dict[int, float] = {}
    for m in range(1, Kmax + 1):
        mask = (shell == m)
        if np.any(mask):
            means[m] = float(np.mean(D2_obs[mask]))
    return means


def fit_loglog_slope(means: Dict[int, float], kmin: int, kmax: int) -> float:
    """
    Fit slope of log(means[k]) vs log(k) over integer k in [kmin,kmax].

    Matches BB-1 v3 / BB-2 v3:
      use only ks present with positive means; require at least 3 points; solve by least squares.
    """
    ks = [k for k in range(kmin, kmax + 1) if (k in means) and (means[k] > 0)]
    if len(ks) < 3:
        return float("nan")
    x = np.log(np.array(ks, dtype=np.float64))
    y = np.log(np.array([means[k] for k in ks], dtype=np.float64))
    A = np.column_stack([x, np.ones_like(x)])
    a, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(a)


def curvature_log(means: Dict[int, float], kmin: int, kmax: int) -> float:
    """
    Mean absolute second-difference of log(shell_mean) over k in [kmin,kmax].

    Matches BB-1 v3 / BB-2 v3:
      require at least 5 valid points; otherwise return 0.0.
    """
    ks = [k for k in range(kmin, kmax + 1) if (k in means) and (means[k] > 0)]
    if len(ks) < 5:
        return 0.0
    y = np.log(np.array([means[k] for k in ks], dtype=np.float64))
    d2 = y[2:] - 2.0 * y[1:-1] + y[:-2]
    return float(np.mean(np.abs(d2)))


def tilt_debiased(N: int, Kp: int, n_s: float, variant: str, wmin: float = 0.2) -> Dict[str, float]:
    """
    Debiased tilt proxy (matches BB-1 v3 / BB-2 v3).

    Conventions:
      - Shell index is floor(|k| + 1e-12).
      - Debias per-mode on a hard mask W^2 >= wmin^2 (and |k|>0), then average on shells.
    """
    k = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(k, k, k, indexing="ij")
    K = np.sqrt(kx * kx + ky * ky + kz * kz)

    P_base = np.where(K > 0, (K + 1e-9) ** (n_s - 4.0), 0.0)

    W = weights_3d(N, Kp, variant)
    W2 = W * W

    P_obs = P_base * W2

    mask = (W2 >= (wmin ** 2)) & (K > 0)

    P_hat = np.zeros_like(P_obs)
    P_hat[mask] = P_obs[mask] / W2[mask]

    D_hat = (K ** 3) * P_hat

    shell = np.floor(K + 1e-12).astype(int)
    Kmax = int(shell.max())

    means: Dict[int, float] = {}
    for m in range(1, Kmax + 1):
        msk = (shell == m) & mask
        if np.any(msk):
            means[m] = float(np.mean(D_hat[msk]))

    kmin = max(2, int(round(0.25 * Kp)))
    kmax = min(Kp, max(kmin + 2, int(round(0.75 * Kp))))
    band = (kmin, kmax)

    slope = fit_loglog_slope(means, band[0], band[1])

    kern = np.fft.ifftn(W).real
    kmin_kernel = float(np.min(kern))

    tot = float(np.sum(P_obs[K > 0]))
    hf = float(np.sum(P_obs[K > Kp]) / tot)

    mask_frac = float(np.mean(mask))

    return {
        "tilt": slope,
        "hf": hf,
        "mask_frac": mask_frac,
        "kmin_kernel": kmin_kernel,
        "band": band,
    }


def power_sum_3d(N: int, Kp: int, n_s: float, variant: str) -> Dict[str, float]:
    """
    Power-sum observable (BB-2 v3): tot and HF fraction.
    """
    _, _, _, kr = make_kgrid(N)
    W = weights_3d(N, Kp, variant)
    W2 = W * W
    P_base = np.where(kr > 0, (kr + 1e-9) ** (n_s - 4.0), 0.0)
    P_hat = P_base * W2
    tot = float(np.sum(P_hat))
    hf = float(np.sum(P_hat[kr > Kp]) / tot)
    return {"tot": tot, "hf": hf}


# ============================================================
# Optional CAMB closure (informational)
# ============================================================

def try_camb_first_peak(cosmo: Dict[str, float]) -> Tuple[bool, str, Optional[float]]:
    try:
        import camb  # type: ignore
    except Exception as e:
        return False, f"CAMB not available: {repr(e)}", None

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo["H0"], ombh2=cosmo["ombh2"], omch2=cosmo["omch2"], tau=cosmo["tau"])
    pars.InitPower.set_params(As=cosmo["A_s"], ns=cosmo["n_s"])
    pars.set_for_lmax(2500, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    tt = powers["total"][:, 0]
    ells = np.arange(tt.shape[0])

    lo, hi = 100, 400
    idx = int(lo + np.argmax(tt[lo:hi]))
    return True, "CAMB TT computed", float(ells[idx])


# ============================================================
# Optional plotting (matplotlib)
# ============================================================

def try_write_plot_png(path_png: str, N: int, Kp: int, n_s: float) -> Tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        return False, f"matplotlib not available: {repr(e)}"

    variants = ["fejer", "sharp", "signed"]
    means = {v: observed_shell_means_3d(N, Kp, n_s, v) for v in variants}

    plt.figure()
    for v in variants:
        ks = np.array(sorted(means[v].keys()), dtype=np.float64)
        ys = np.array([means[v][int(k)] for k in ks], dtype=np.float64)
        plt.plot(ks[1:], ys[1:], marker="o", label=v)  # skip k=0
    plt.yscale("log")
    plt.xlabel("k (shell index)")
    plt.ylabel("mean Delta^2_hat(k) (log scale)")
    plt.title("BB-36 observed spectrum proxy (admissible vs controls)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_png, dpi=180)
    plt.close()
    return True, f"wrote {path_png}"


# ============================================================
# MASTER MAIN (integrated flagship)
# ============================================================

def main() -> int:
    np.set_printoptions(precision=12, suppress=False)

    spec = {
        "demo": "DEMO-36 (BB) master flagship",
        "version": "v2",
        "primary_window": [97, 200],
        "cf_window": [181, 1200],
        "enforce_minN": 64,
        "wmin": 0.2,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
    }
    spec_sha = sha256_of_obj(spec)

    banner("DEMO-36 — BIG BANG MASTER FLAGSHIP (BB-36 closure + bridges + teeth)")
    print(f"UTC time : {utc_now_iso()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout + optional JSON/PNG artifacts")
    script_sha = try_file_sha256(__file__)
    if script_sha:
        print(f"Script SHA256 : {script_sha}")
    print()
    print(f"spec_sha256: {spec_sha}\n")

    results: Dict[str, object] = {"spec": spec, "spec_sha256": spec_sha}

    # -------------------------------
    # STAGE 1 — Selection
    # -------------------------------
    banner("STAGE 1 — Selection (primary + deterministic counterfactuals)")
    primary, pools = select_primary_triple(tuple(spec["primary_window"]), v2U_req=3)
    cfs = counterfactual_triples(primary, tuple(spec["cf_window"]), need=4)

    print("Lane survivor pools (raw):")
    print("  U(1): ", pools["U1_raw"])
    print("  SU(2):", pools["SU2_raw"])
    print("  SU(3):", pools["SU3_raw"])
    print("After U(1) coherence v2(wU-1)=3:", pools["U1_coherent"])

    ok_cf = passfail(len(cfs) >= 4, "Captured >=4 counterfactual triples (deterministic)",
                     f"found={len(cfs)} window={tuple(spec['cf_window'])}")
    ok_primary = passfail(primary == Triple(137, 107, 103), "Primary equals (137,107,103)", f"selected={primary}")
    print("Counterfactuals:", cfs, "\n")

    results["primary"] = dataclass_asdict(primary)
    results["pools"] = pools
    results["counterfactuals"] = [dataclass_asdict(t) for t in cfs]

    # -------------------------------
    # STAGE 2 — Budgets + structural closure
    # -------------------------------
    banner("STAGE 2 — Structural closure (parameters) + budgets (derived)")
    b = budgets_from_triple(primary, enforce_minN=int(spec["enforce_minN"]))
    cosmo = structural_cosmo_params(primary)

    print("PRIMARY")
    print(f"primary: {primary}")
    print(f"q2={b.q2} q3={b.q3} v2U={b.v2U} eps={b.eps:.8f}")
    print(f"N={b.N} K_primary={b.K_primary} K_truth={b.K_truth}\n")

    print("STRUCTURAL COSMO OUTPUTS (DEMO-36 monomials)")
    for k in ["H0", "Omega_b", "Omega_c", "Omega_L", "ombh2", "omch2",
              "A_s", "n_s", "tau", "ell1", "deltaCMB", "delta0", "F_CMB"]:
        v = cosmo[k]
        if abs(v) < 1e-2 or abs(v) > 1e3:
            print(f"{k:10s}: {v:.12e}")
        else:
            print(f"{k:10s}: {v:.12f}")
    print()

    g_struct = {}
    g_struct["H0"] = passfail(50.0 < cosmo["H0"] < 80.0, "Gate S1: H0 in (50,80) km/s/Mpc", f"H0={cosmo['H0']:.3f}")
    g_struct["ombh2"] = passfail(0.015 < cosmo["ombh2"] < 0.035, "Gate S2: omega_b h^2 in (0.015,0.035)", f"ombh2={cosmo['ombh2']:.6f}")
    g_struct["omch2"] = passfail(0.05 < cosmo["omch2"] < 0.20, "Gate S3: omega_c h^2 in (0.05,0.20)", f"omch2={cosmo['omch2']:.6f}")
    g_struct["As"] = passfail(1e-9 < cosmo["A_s"] < 5e-9, "Gate S4: A_s in (1e-9,5e-9)", f"A_s={cosmo['A_s']:.3e}")
    g_struct["ns"] = passfail(0.90 < cosmo["n_s"] < 1.05, "Gate S5: n_s in (0.90,1.05)", f"n_s={cosmo['n_s']:.6f}")
    g_struct["tau"] = passfail(0.01 < cosmo["tau"] < 0.10, "Gate S6: tau in (0.01,0.10)", f"tau={cosmo['tau']:.6f}")
    g_struct["ell1"] = passfail(150.0 < cosmo["ell1"] < 350.0, "Gate S7: ell1 in (150,350)", f"ell1={cosmo['ell1']:.3f}")
    g_struct["delta"] = passfail(0.5e-5 < cosmo["deltaCMB"] < 2.0e-5, "Gate S8: deltaCMB in O(1e-5) band", f"delta={cosmo['deltaCMB']:.3e}")

    results["budgets_primary"] = dataclass_asdict(b)
    results["cosmo_structural"] = cosmo
    results["gates_structural"] = g_struct
    print()

    # -------------------------------
    # STAGE 3 — Tilt bridge
    # -------------------------------
    banner("STAGE 3 — Tilt bridge (admissible vs illegal controls + teeth)")
    N = b.N
    Kp = b.K_primary
    eps = b.eps

    tilt_target = cosmo["n_s"] - 1.0
    wmin = float(spec["wmin"])

    adm = tilt_debiased(N, Kp, cosmo["n_s"], "fejer", wmin=wmin)
    sharp = tilt_debiased(N, Kp, cosmo["n_s"], "sharp", wmin=wmin)
    signed = tilt_debiased(N, Kp, cosmo["n_s"], "signed", wmin=wmin)

    # Observed curvature around the cutoff (uses Delta^2_hat means, not debiased)
    means_adm_obs = observed_shell_means_3d(N, Kp, cosmo["n_s"], "fejer")
    means_sharp_obs = observed_shell_means_3d(N, Kp, cosmo["n_s"], "sharp")
    means_signed_obs = observed_shell_means_3d(N, Kp, cosmo["n_s"], "signed")
    cmin = max(2, Kp - 4)
    cmax = Kp + 4
    curv_adm = curvature_log(means_adm_obs, cmin, cmax)
    curv_sharp = curvature_log(means_sharp_obs, cmin, cmax)
    curv_signed = curvature_log(means_signed_obs, cmin, cmax)

    print(f"band(tilt fit) adm={adm['band']} sharp={sharp['band']} signed={signed['band']}")
    print(f"n_s_struct={cosmo['n_s']:.9f}  tilt_target=n_s-1={tilt_target:.9f}\n")
    print(f"admissible: tilt={adm['tilt']:+.9f}  kmin_kernel={adm['kmin_kernel']:+.3e}  hf={adm['hf']:.6f}  mask={adm['mask_frac']:.6f}")
    print(f"sharp     : tilt={sharp['tilt']:+.9f}  kmin_kernel={sharp['kmin_kernel']:+.3e}  hf={sharp['hf']:.6f}  mask={sharp['mask_frac']:.6f}")
    print(f"signed    : tilt={signed['tilt']:+.9f}  kmin_kernel={signed['kmin_kernel']:+.3e}  hf={signed['hf']:.6f}  mask={signed['mask_frac']:.6f}")
    print(f"cutoff curvature (observed Delta^2): adm={curv_adm:.6f} sharp={curv_sharp:.6f} signed={curv_signed:.6f}\n")

    g_tilt = {}
    tol_kernel = 1e-12
    tol_tilt = eps ** 3
    floor_hf = eps ** 2

    g_tilt["G1"] = passfail(adm["kmin_kernel"] >= -tol_kernel, "Gate T1: admissible kernel nonnegative (Fejer tensor)",
                            f"kmin={adm['kmin_kernel']:.3e} tol={tol_kernel:g}")
    g_tilt["G2"] = passfail((sharp["kmin_kernel"] < -1e-6) and (signed["kmin_kernel"] < -1e-6),
                            "Gate T2: illegal kernels have negative lobes (sharp + signed)",
                            f"kmin_sharp={sharp['kmin_kernel']:.3e} kmin_signed={signed['kmin_kernel']:.3e}")
    g_tilt["G3"] = passfail(abs(adm["tilt"] - tilt_target) <= tol_tilt, "Gate T3: admissible tilt matches structural target (tol=eps^3)",
                            f"|Δ|={abs(adm['tilt']-tilt_target):.6f} tol={tol_tilt:.6f}")
    g_tilt["G4"] = passfail(curv_sharp >= (1.0 + eps) * curv_adm, "Gate T4: sharp cutoff increases cutoff-band curvature",
                            f"curv_adm={curv_adm:.6f} curv_sharp={curv_sharp:.6f} eps={eps:.6f}")
    g_tilt["G5"] = passfail(signed["hf"] >= max(10.0 * adm["hf"], floor_hf), "Gate T5: signed illegal injects HF beyond floor",
                            f"hf_signed={signed['hf']:.6f} floor={max(10.0*adm['hf'], floor_hf):.6f}")

    # Teeth: counterfactual budgets must degrade score
    scoreP = abs(adm["tilt"] - tilt_target) + curv_adm
    print("COUNTERFACTUAL TEETH (budget K)")
    print(f"Primary score: K={Kp:2d} tilt={adm['tilt']:+.6f} curv={curv_adm:.6f} score={scoreP:.6f}")

    strong = 0
    cf_rows = []
    for tcf in cfs:
        bcf = budgets_from_triple(tcf, enforce_minN=int(spec["enforce_minN"]))
        Kcf = bcf.K_primary
        adm_cf = tilt_debiased(N, Kcf, cosmo["n_s"], "fejer", wmin=wmin)
        means_cf = observed_shell_means_3d(N, Kcf, cosmo["n_s"], "fejer")
        cminK = max(2, Kcf - 4)
        cmaxK = Kcf + 4
        curv_cf = curvature_log(means_cf, cminK, cmaxK)
        score_cf = abs(adm_cf["tilt"] - tilt_target) + curv_cf
        miss = score_cf >= scoreP * (1.0 + eps)
        strong += int(miss)
        cf_rows.append({"triple": dataclass_asdict(tcf), "K": Kcf, "tilt": adm_cf["tilt"], "curv": curv_cf, "score": score_cf, "miss": miss})
        print(f"CF {tcf}  K={Kcf:2d}  tilt={adm_cf['tilt']:+.6f}  curv={curv_cf:.6f}  score={score_cf:.6f}  miss={miss}")

    g_tilt["TEETH"] = passfail(strong >= 3, "Gate T6: >=3/4 counterfactual budgets miss by score margin",
                               f"strong={strong}/4 eps={eps:.6f}")

    results["tilt_bridge"] = {
        "tilt_target": tilt_target,
        "primary": {"admissible": adm, "sharp": sharp, "signed": signed,
                    "curv_adm": curv_adm, "curv_sharp": curv_sharp, "curv_signed": curv_signed,
                    "score": scoreP},
        "counterfactuals": cf_rows,
        "gates": g_tilt,
    }
    print()

    # -------------------------------
    # STAGE 4 — deltaCMB amplitude bridge
    # -------------------------------
    banner("STAGE 4 — delta_CMB amplitude bridge (power-sum proxy + controls + teeth)")
    truth = power_sum_3d(N, b.K_truth, cosmo["n_s"], "fejer")
    tot_truth = truth["tot"]

    def delta_proxy(tot: float) -> float:
        return cosmo["deltaCMB"] * math.sqrt(tot / tot_truth)

    adm2 = power_sum_3d(N, Kp, cosmo["n_s"], "fejer")
    sh2 = power_sum_3d(N, Kp, cosmo["n_s"], "sharp")
    si2 = power_sum_3d(N, Kp, cosmo["n_s"], "signed")

    delta_adm = delta_proxy(adm2["tot"])
    delta_sh = delta_proxy(sh2["tot"])
    delta_si = delta_proxy(si2["tot"])

    rel_adm = abs(delta_adm / cosmo["deltaCMB"] - 1.0)
    rel_sh = abs(delta_sh / cosmo["deltaCMB"] - 1.0)
    rel_si = abs(delta_si / cosmo["deltaCMB"] - 1.0)

    print(f"deltaCMB_struct = {cosmo['deltaCMB']:.12e}  (ratio to 1e-5: {cosmo['deltaCMB']/1e-5:.6f})")
    print(f"Truth tier (Fejer@K_truth={b.K_truth}): tot={truth['tot']:.6f}  hf={truth['hf']:.6f}  (delta matches by construction)")
    print(f"admissible (Fejer@K={Kp}): delta={delta_adm:.12e}  rel_err={rel_adm:.6f}  hf={adm2['hf']:.6f}")
    print(f"sharp      (cut@K={Kp}):   delta={delta_sh:.12e}  rel_err={rel_sh:.6f}  hf={sh2['hf']:.6f}")
    print(f"signed     (+/-@K={Kp}):   delta={delta_si:.12e}  rel_err={rel_si:.6f}  hf={si2['hf']:.6f}\n")

    g_amp = {}
    g_amp["G1"] = passfail(0.5e-5 < cosmo["deltaCMB"] < 2.0e-5, "Gate A1: deltaCMB_struct in plausible band (order 1e-5)",
                           f"delta={cosmo['deltaCMB']:.3e}")
    g_amp["G2"] = passfail(rel_adm <= eps, "Gate A2: admissible proxy within eps", f"rel_err={rel_adm:.6f} eps={eps:.6f}")
    g_amp["G3"] = passfail(rel_si >= rel_adm, "Gate A3: signed illegal worsens error vs admissible", f"err_adm={rel_adm:.6f} err_signed={rel_si:.6f}")
    g_amp["G4"] = passfail(si2["hf"] >= max(10.0 * adm2["hf"], floor_hf), "Gate A4: signed illegal injects HF beyond floor",
                           f"hf_signed={si2['hf']:.6f} floor={max(10.0*adm2['hf'], floor_hf):.6f}")

    scoreA = rel_adm + adm2["hf"]
    print("COUNTERFACTUAL TEETH (budget K)")
    print(f"Primary score: K={Kp:2d} rel_err={rel_adm:.6f} hf={adm2['hf']:.6f} score={scoreA:.6f}")

    strongA = 0
    cf_rows2 = []
    for tcf in cfs:
        bcf = budgets_from_triple(tcf, enforce_minN=int(spec["enforce_minN"]))
        Kcf = bcf.K_primary
        cf = power_sum_3d(N, Kcf, cosmo["n_s"], "fejer")
        delta_cf = delta_proxy(cf["tot"])
        rel_cf = abs(delta_cf / cosmo["deltaCMB"] - 1.0)
        score_cf = rel_cf + cf["hf"]
        miss = score_cf >= scoreA * (1.0 + eps)
        strongA += int(miss)
        cf_rows2.append({"triple": dataclass_asdict(tcf), "K": Kcf, "rel_err": rel_cf, "hf": cf["hf"], "score": score_cf, "miss": miss})
        print(f"CF {tcf}  K={Kcf:2d}  rel_err={rel_cf:.6f}  hf={cf['hf']:.6f}  score={score_cf:.6f}  miss={miss}")

    g_amp["TEETH"] = passfail(strongA >= 3, "Gate A5: >=3/4 counterfactual budgets miss by score margin",
                              f"strong={strongA}/4 eps={eps:.6f}")

    results["deltaCMB_bridge"] = {
        "truth": truth,
        "primary": {
            "admissible": {"delta": delta_adm, "rel_err": rel_adm, **adm2},
            "sharp": {"delta": delta_sh, "rel_err": rel_sh, **sh2},
            "signed": {"delta": delta_si, "rel_err": rel_si, **si2},
            "score": scoreA,
        },
        "counterfactuals": cf_rows2,
        "gates": g_amp,
    }
    print()

    # -------------------------------
    # STAGE 5 — Optional CAMB
    # -------------------------------
    banner("STAGE 5 — Optional CAMB TT closure (first peak vs ell1_struct)")
    camb_ok, camb_msg, ell_peak = try_camb_first_peak(cosmo)
    print(camb_msg)
    if camb_ok and ell_peak is not None:
        print(f"ell1_struct = {cosmo['ell1']:.6f}")
        print(f"ell1_camb   = {ell_peak:.6f}")
        print(f"Delta_ell1  = {abs(ell_peak - cosmo['ell1']):.6f}")
        passfail(True, "Gate C0: CAMB stage executed (informational)")
        results["camb"] = {"available": True, "ell_peak": ell_peak}
    else:
        passfail(True, "Gate C0: CAMB stage skipped (install camb to enable)")
        results["camb"] = {"available": False, "ell_peak": None}
    print()

    # -------------------------------
    # STAGE 6 — Artifacts + determinism
    # -------------------------------
    banner("STAGE 6 — Artifacts (JSON + optional plot) + determinism hash")
    determinism_sha = sha256_of_obj(results)
    results["determinism_sha256"] = determinism_sha

    out_json = "bb36_master_results.json"
    out_png = "bb36_master_plot.png"

    try:
        with open(out_json, "w", encoding="utf-8") as f:
            f.write(json.dumps(results, sort_keys=True, indent=2))
        passfail(True, "Wrote results JSON", out_json)
    except Exception as e:
        passfail(True, "Results JSON not written (filesystem unavailable)", repr(e))

    ok_plot, msg_plot = try_write_plot_png(out_png, N, Kp, cosmo["n_s"])
    passfail(ok_plot, "Plot (optional)", msg_plot)
    print()

    banner("DETERMINISM HASH")
    print(f"determinism_sha256: {determinism_sha}\n")

    # -------------------------------
    # FINAL VERDICT
    # -------------------------------
    banner("FINAL VERDICT")
    all_required = (
        ok_cf and ok_primary and
        all(g_struct.values()) and
        all(g_tilt.values()) and
        all(g_amp.values())
    )
    passfail(all_required, "DEMO-36 VERIFIED (structural closure + 2 bridges + controls + teeth)")
    print("Result:", "VERIFIED" if all_required else "NOT VERIFIED")
    return 0 if all_required else 1


if __name__ == "__main__":
    raise SystemExit(main())




# Evaluation-only: generate CAMB overlay artifact (must not feed upstream selection)
try:
    import subprocess, sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    _script = _here / "camb_overlay.py"
    print(f"[CAMB_OVERLAY] running: {sys.executable} {_script}")
    subprocess.run([sys.executable, str(_script)], cwd=str(_here), check=True)
except Exception as e:
    print("[CAMB_OVERLAY] FAILED:", repr(e))
    raise



# Evaluation-only: generate CAMB overlay artifact (must not feed upstream selection)
try:
    import subprocess, sys
    from pathlib import Path
    _here = Path(__file__).resolve().parent
    _script = _here / "camb_overlay.py"
    print(f"[CAMB_OVERLAY] running: {sys.executable} {_script}")
    subprocess.run([sys.executable, str(_script)], cwd=str(_here), check=True)
except Exception as e:
    print("[CAMB_OVERLAY] FAILED:", repr(e))
    raise

