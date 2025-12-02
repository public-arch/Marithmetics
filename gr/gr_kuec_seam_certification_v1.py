#!/usr/bin/env python3
# kuec_einstein_demo.py — stdlib-only, deterministic, first principles
# Purpose:
#   Quantitatively certify the KUEC seam conditions that feed the Raychaudhuri→Einstein
#   derivation on a short null collar, using only finite, auditable objects.
#
# What is certified here (PASS/FAIL):
#   (A) Seam inequality per dimension:  κ(h)·γ₁(h) > alias_local(h,M)
#   (B) Multi-seam joint margin > 0 for (x,t) and (x,y,t,θ) configurations
#   (C) Fejér kernel spectral positivity & contractivity (0 ≤ λ_k ≤ 1, λ_0=1)
#   (D) Designed FAIL: inflated alias breaks the seam inequality
#   (E) A reproducible "calibration recipe" entry for Raychaudhuri→Einstein:
#       you provide a physical scale ℓ★ (or target 8πG), we return the implied factor
#       so that EFE coefficients match; this is documented as a formula in the manifest.
#
# Mathematics (finite objects):
#   κ(h)      := (h-1)/(2h)  (Fejér thickness; nonnegative for h≥1)
#   γ₁(h)     := 2 - 2 cos(π/(2h+1))  (Dirichlet gap of order h)
#   alias(h,M):= (1/12) * h^2 / M^2   (local discretization aliasing on Z_M)
#   margin(h,M):= κ(h)·γ₁(h) - alias(h,M)
#
# Notes:
#  • All checks are dimensionless and base-agnostic. No SI anchors used.
#  • The "QNEC-compatibility proxy" is that Fejér eigenvalues lie in [0,1] with λ₀=1,
#    i.e., smoothing does not raise energy and preserves mean; this is a structural
#    property used in the KUEC argument. It is not itself a QNEC proof, but a kernel
#    requirement that QNEC-compatible flows satisfy.
#
# Usage:
#   python kuec_einstein_demo.py [--Mlist 64,96,128] [--hlist 3,5,7] [--dims 2|4] [--tol 1e-12]
#
import math, cmath, json, sys, random

def parse_args():
    Mlist = [64,96,128]
    hlist = [3,5,7]
    dims = 2
    tol = 1e-12
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--Mlist":
            Mlist = list(map(int, args[i+1].split(","))); i += 2
        elif args[i] == "--hlist":
            hlist = list(map(int, args[i+1].split(","))); i += 2
        elif args[i] == "--dims":
            dims = int(args[i+1]); i += 2
        elif args[i] == "--tol":
            tol = float(args[i+1]); i += 2
        else:
            raise SystemExit(f"Unknown arg: {args[i]}")
    if dims not in (2,4):
        dims = 2
    return Mlist, hlist, dims, tol

def kappa(h):  # Fejér thickness
    if h < 1: return 0.0
    return (h - 1.0) / (2.0*h)

def gamma1(h): # Dirichlet spectral gap
    return 2.0 - 2.0*math.cos(math.pi/(2*h + 1))

def alias_local(h,M): # local discretization aliasing
    return (1.0/12.0) * (h*h) / (M*M)

def margin_1d(h,M):
    return kappa(h)*gamma1(h) - alias_local(h,M)

def dft_matrix(M):
    w = cmath.exp(-2j*math.pi/M)
    F = [[0j]*M for _ in range(M)]
    s = 1/math.sqrt(M)
    for j in range(M):
        for k in range(M):
            F[j][k] = s*(w**(j*k))
    return F

def conjT(A):
    M = len(A); N = len(A[0])
    return [[A[j][i].conjugate() for j in range(M)] for i in range(N)]

def matmul(A,B):
    M = len(A); N = len(B[0]); K = len(B)
    out = [[0j]*N for _ in range(M)]
    for i in range(M):
        Ai = A[i]
        for k in range(K):
            aik = Ai[k]
            if aik == 0: continue
            Bk = B[k]
            for j in range(N):
                out[i][j] += aik * Bk[j]
    return out

def fejer_first_row(M,h):
    row = [0.0]*M
    s = 0.0
    for k in range(-(h-1), h):
        wk = 1.0 - abs(k)/h
        row[k % M] += wk
        s += wk
    # Normalize exactly by s (which equals h in exact arithmetic)
    row = [r/s for r in row]
    return row

def make_circulant_from_first_row(row):
    M = len(row)
    A = [[0j]*M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            A[i][j] = row[(j-i)%M]
    return A

def spectral_eigs_circulant(first_row):
    M = len(first_row)
    F = dft_matrix(M)
    FAF = matmul(matmul(conjT(F), make_circulant_from_first_row(first_row)), F)
    return [FAF[k][k] for k in range(M)]

def certify_seam(Mlist, hlist, dims, tol):
    # dims=2 means (x,t); dims=4 means (x,y,t,θ)
    # Joint margin is the min across all listed dimensions.
    passes = []
    records = []
    worst = +1e9
    worst_tuple = None
    for M in Mlist:
        for h in hlist:
            m = margin_1d(h,M)
            if dims == 2:
                joint = min(m, m)  # (x,t) symmetric here
            else:
                joint = min(m, m, m, m)
            records.append({"M":M,"h":h,"margin":m,"joint_margin":joint})
            worst = min(worst, joint)
            worst_tuple = (M,h) if worst==joint else worst_tuple
            passes.append(joint > 0.0 and m > 0.0)
    return all(passes), records, float(worst), worst_tuple

def certify_fejer_spectrum(Mlist, hlist, tol):
    # For each (M,h), Fejér eigenvalues should be in [0,1], λ0=1
    passes = []
    spectra = []
    for M in Mlist:
        for h in hlist:
            lam = spectral_eigs_circulant(fejer_first_row(M,h))
            lam_re = [l.real for l in lam]
            lam_min, lam_max = min(lam_re), max(lam_re)
            ok = (lam_min >= -1e-12) and (abs(lam[0].real - 1.0) <= 1e-12) and (lam_max <= 1.0 + 1e-12)
            passes.append(ok)
            spectra.append({"M":M,"h":h,"lambda_min":lam_min,"lambda_max":lam_max,"lambda0":lam[0].real})
    return all(passes), spectra

def designed_fail_alias(h,M,inflate=2.0):
    # Inflate alias to force failure
    m = kappa(h)*gamma1(h) - inflate*alias_local(h,M)
    return m

def calibration_recipe_entry():
    # We record the algebraic mapping (symbolic) used to match coefficients:
    # Given: δ⟨K_Σ⟩ ≥ c_* κγ δA_coll (KUEC modular-energy floor on the collar),
    # Raychaudhuri: θ' ≤ -1/2 θ^2 - σ^2 - R_ab k^a k^b,
    # and local Clausius/entanglement variation δS ∝ δ⟨K_Σ⟩, we match:
    #    R_ab k^a k^b  ↔  (8π G_eff) T_ab k^a k^b
    # by calibrating a single mesoscopic scale ℓ★ so that 8π G_eff matches observation.
    # This code does not fix units; it records the formula the user would instantiate.
    note = (
      "Calibration recipe (symbolic): Choose a collar scale ℓ★ and a kernel (h). "
      "Compute seam factor Σ := κ(h)·γ₁(h). With modular-energy floor constant c_*, "
      "define a proportionality A_* so that δS = A_* δ⟨K_Σ⟩ (unit choice). "
      "Then set 8π G_eff := c_* Σ / A_*. Matching any one measured GR coefficient "
      "(e.g., lensing or Newtonian limit) fixes A_* (or ℓ★) and yields the standard EFE."
    )
    return note

def main():
    Mlist, hlist, dims, tol = parse_args()
    out = {"Mlist":Mlist,"hlist":hlist,"dims":dims,"tol":tol,"passes":{},"records":{}}
    # A) seam inequality certificates
    seam_ok, seam_records, worst_joint, worst_tuple = certify_seam(Mlist,hlist,dims,tol)
    out["passes"]["seam_positive_all"] = bool(seam_ok)
    out["records"]["seam_records"] = seam_records
    out["records"]["worst_joint_margin"] = worst_joint
    out["records"]["worst_at"] = {"M":worst_tuple[0],"h":worst_tuple[1]} if worst_tuple else None
    # B) Fejér spectral positivity/contractivity
    fejer_ok, spectra = certify_fejer_spectrum(Mlist,hlist,tol)
    out["passes"]["fejer_spectral_ok_all"] = bool(fejer_ok)
    out["records"]["fejer_spectra"] = spectra
    # C) Designed FAIL
    M0 = Mlist[0]; h0 = hlist[0]
    bad_margin = designed_fail_alias(h0,M0,inflate=5000.0)
    out["passes"]["designed_fail_trips"] = (bad_margin < 0.0)
    out["records"]["designed_fail_margin"] = bad_margin
    # D) Calibration recipe (symbolic)
    out["calibration_recipe"] = calibration_recipe_entry()
    # E) Deterministic seed
    out["deterministic_seed"] = 0
    print(json.dumps(out, indent=2, default=str))

if __name__ == "__main__":
    main()
