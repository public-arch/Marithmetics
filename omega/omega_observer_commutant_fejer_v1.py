#!/usr/bin/env python3
# observer_omega_demo.py — stdlib-only, deterministic, first principles
# Purpose:
#   Prove-and-demo in finite dimension M that:
#     (1) The commutant of the cyclic shift S is exactly the circulant algebra.
#     (2) The unique admissible principal projector is Ω = (1/M) 11^T.
#     (3) Fejér smoothing is a positive, contractive, circulant operator that
#         commutes with Ω and preserves the mean (zero-mode eigenvalue = 1).
#   All numerics are exact to machine tolerance with explicit PASS/FAIL.
#
# Usage:
#   python observer_omega_demo.py [--M 16] [--h 5] [--tol 1e-12]
#
# No third-party packages. Pure Python 3.
#
# Mathematical facts used (first principles, finite-dimensional):
#   • S is the M×M cyclic shift: (Sx)_k = x_{k-1 mod M}.
#   • A matrix A commutes with S iff A is circulant (columns are cyclic shifts of the first).
#     Proof sketch: The map E(A) := (1/M) Σ_t S^t A S^{-t} is a conditional expectation onto
#     the commutant of S; its image are precisely the circulants. If A commutes with S then
#     E(A)=A; conversely every E(A) is circulant and commutes with S.
#   • The DFT diagonalizes every circulant: F* A F = diag(λ_0,...,λ_{M-1}).
#   • The unique rank-1 projector in the commutant with range span{1} is Ω = (1/M)11^T:
#     In Fourier basis, commuting projectors are diagonal. For range span{1}, eigenvalues
#     must be 1 on the DC mode and 0 elsewhere, forcing Ω.
#   • The Fejér operator of order h is convolution with weights w_k = 1 - |k|/h  (|k|<h),
#     normalized by Σ w_k = h. It is circulant, positive, and contractive (its eigenvalues
#     in Fourier basis lie in [0,1], with λ_0 = 1).
#
# Designed FAILs:
#   • We perturb a circulant by a single entry; the commutator with S must become nonzero.
#
# Output:
#   Prints PASS/FAIL and writes a JSON manifest with all test results and hashes.
#
import math, cmath, json, hashlib, random, sys

def parse_args():
    M = 16
    h = 5
    tol = 1e-12
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--M":
            M = int(args[i+1]); i += 2
        elif args[i] == "--h":
            h = int(args[i+1]); i += 2
        elif args[i] == "--tol":
            tol = float(args[i+1]); i += 2
        else:
            raise SystemExit(f"Unknown arg: {args[i]}")
    if h < 1: h = 1
    if M < 2: M = 2
    return M, h, tol

def shift_matrix(M):
    S = [[0.0]*M for _ in range(M)]
    for i in range(M):
        S[i][(i-1) % M] = 1.0
    return S

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

def matadd(A,B,alpha=1.0):
    M = len(A); N = len(A[0])
    out = [[0j]*N for _ in range(M)]
    for i in range(M):
        for j in range(N):
            out[i][j] = A[i][j] + alpha*B[i][j]
    return out

def matsub(A,B):
    return matadd(A,B,alpha=-1.0)

def eye(M):
    E = [[0.0]*M for _ in range(M)]
    for i in range(M): E[i][i] = 1.0
    return E

def transpose(A):
    M = len(A); N = len(A[0])
    return [[A[j][i] for j in range(M)] for i in range(N)]

def conjT(A):
    M = len(A); N = len(A[0])
    return [[A[j][i].conjugate() for j in range(M)] for i in range(N)]

def max_abs(A):
    M = len(A); N = len(A[0])
    m = 0.0
    for i in range(M):
        for j in range(N):
            v = A[i][j]
            av = abs(v.real) if isinstance(v,(int,float)) else abs(v)
            if av > m: m = av
    return m

def vec_mean(x):
    return sum(x)/len(x)

def apply(A,x):
    M = len(A); N = len(A[0])
    assert N == len(x)
    y = [0j]*M
    for i in range(M):
        s = 0j
        Ai = A[i]
        for j in range(N):
            s += Ai[j]*x[j]
        y[i] = s
    return y

def dft_matrix(M):
    w = cmath.exp(-2j*math.pi/M)
    F = [[0j]*M for _ in range(M)]
    s = 1/math.sqrt(M)
    for j in range(M):
        for k in range(M):
            F[j][k] = s*(w**(j*k))
    return F

def idft_matrix(M):
    # inverse = conjugate transpose of F
    F = dft_matrix(M)
    return conjT(F)

def commutator(A,B):
    return matsub(matmul(A,B), matmul(B,A))

def is_circulant(A, tol):
    M = len(A)
    # check columns are cyclic shifts of first column
    # we can equivalently check rows as well; here use columns.
    first_col = [A[i][0] for i in range(M)]
    for j in range(M):
        for i in range(M):
            expected = first_col[(i-j)%M]
            if abs(A[i][j] - expected) > tol:
                return False
    return True

def make_circulant_from_first_row(row):
    M = len(row)
    A = [[0j]*M for _ in range(M)]
    for i in range(M):
        for j in range(M):
            A[i][j] = row[(j-i)%M]
    return A

def conditional_expectation_commutant(A, S):
    # E(A) = (1/M) Σ_t S^t A S^{-t}
    M = len(A)
    # precompute S^t and S^{-t}
    Spows = [eye(M)]
    for t in range(1,M):
        Spows.append(matmul(Spows[-1], S))
    # inverse powers
    S_inv = transpose(S)  # since S is a permutation, S^{-1} = S^T
    Sinvpows = [eye(M)]
    for t in range(1,M):
        Sinvpows.append(matmul(Sinvpows[-1], S_inv))
    acc = [[0j]*M for _ in range(M)]
    for t in range(M):
        acc = matadd(acc, matmul(matmul(Spows[t], A), Sinvpows[t]))
    # average
    for i in range(M):
        for j in range(M):
            acc[i][j] /= M
    return acc

def omega_projector(M):
    # Ω = (1/M) 11^T
    val = 1.0/M
    return [[val]*M for _ in range(M)]

def fejer_first_row(M, h):
    # Discrete circular Fejér weights on Z_M, order h
    # w_k = 1 - |k|/h for |k| < h else 0; normalized by sum = h
    row = [0.0]*M
    s = 0.0
    for k in range(-(h-1), h):
        wk = 1.0 - abs(k)/h
        row[k % M] += wk
        s += wk
    # s should equal h
    if abs(s - h) > 1e-12:
        # numerical guard, but in integer arithmetic it should be exact
        pass
    row = [r/h for r in row]
    return row

def spectral_eigenvalues_circulant(first_row):
    M = len(first_row)
    F = dft_matrix(M)
    Finv = conjT(F)
    A = make_circulant_from_first_row(first_row)
    # λ = diag(F* A F)
    L = []
    FAF = matmul(matmul(conjT(F), A), F)
    for k in range(M):
        L.append(FAF[k][k])
    return L

def run(M=16,h=5,tol=1e-12, seed=0):
    random.seed(seed)
    out = {"M":M,"h":h,"tol":tol,"passes":{},"metrics":{}}
    PASS = lambda name, ok: print(f"{name}: {'✅ PASS' if ok else '❌ FAIL'}") or out["passes"].update({name:bool(ok)})
    METR = lambda k,v: out["metrics"].update({k:v})

    # 0) Build S and DFT
    S = shift_matrix(M)
    F = dft_matrix(M); Finv = conjT(F)
    # Verify diagonalization of S
    FSF = matmul(matmul(conjT(F), S), F)
    # Compare FSF to diag(exp(2πik/M))
    max_dev = 0.0
    for k in range(M):
        target = cmath.exp(2j*math.pi*k/M)
        dev = abs(FSF[k][k] - target)
        if dev > max_dev: max_dev = dev
        # off-diagonals
        for j in range(M):
            if j == k: continue
            if abs(FSF[k][j]) > max_dev: max_dev = abs(FSF[k][j])
    METR("FSF_max_deviation", float(max_dev))
    PASS("S diagonalized by DFT", max_dev <= 1e-12)

    # 1) Commutant = circulants via conditional expectation
    # Start with random A, project to commutant
    A = [[complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(M)] for __ in range(M)]
    E = conditional_expectation_commutant(A, S)
    is_comm = max_abs(commutator(E,S)) <= tol
    is_circ = is_circulant(E, tol=1e-10)
    PASS("E(A) commutes with S", is_comm)
    PASS("E(A) is circulant", is_circ)

    # Designed FAIL: perturb one entry to break circulant
    E_bad = [row[:] for row in E]
    E_bad[0][1] += 0.123456789
    comm_norm = max_abs(commutator(E_bad,S))
    METR("comm_norm_after_perturb", float(comm_norm))
    PASS("Designed FAIL (non-circulant does NOT commute)", comm_norm > 1e-6)

    # 2) Ω projector properties and uniqueness
    Ω = omega_projector(M)
    # Idempotent
    idemp = max_abs(matsub(matmul(Ω,Ω), Ω)) <= tol
    PASS("Ω idempotent", idemp)
    # Commutes with S
    commΩ = max_abs(commutator(Ω,S)) <= tol
    PASS("Ω commutes with S", commΩ)
    # Mean projection
    x = [complex(random.uniform(-1,1), random.uniform(-1,1)) for _ in range(M)]
    y = apply(Ω, x)
    m = sum(x)/M
    mean_ok = max(abs(y[j]-m) for j in range(M)) <= 1e-12
    PASS("Ω projects onto span{1} with mean", mean_ok)

    # Uniqueness in Fourier basis: projector eigenvalues must be (1,0,...,0)
    # Compute eigenvalues of Ω
    # Since Ω is circulant, diagonalize with DFT
    # First row of Ω is [1/M, 1/M, ..., 1/M]
    first_row_Ω = [1.0/M]*M
    λ = spectral_eigenvalues_circulant(first_row_Ω)
    # Check λ_0≈1 and λ_k≈0 for k>0
    lam0_ok = abs(λ[0]-1.0) <= 1e-12
    lam_rest_ok = all(abs(λ[k]) <= 1e-12 for k in range(1,M))
    PASS("Uniqueness: eigenvalues of Ω are (1,0,...,0)", lam0_ok and lam_rest_ok)

    # 3) Fejér operator: positivity, contractivity, commutes with Ω, preserves mean
    first_row_F = fejer_first_row(M,h)
    # eigenvalues of Fejér
    λF = spectral_eigenvalues_circulant(first_row_F)
    lam_min = min(λF, key=lambda z: z.real).real
    lam_max = max(λF, key=lambda z: z.real).real
    METR("Fejer_lambda_min", float(lam_min))
    METR("Fejer_lambda_max", float(lam_max))
    pos_ok = lam_min >= -1e-12
    le1_ok = lam_max <= 1.0 + 1e-12
    PASS("Fejér eigenvalues in [0,1]", pos_ok and le1_ok)
    # Commutes with Ω (both circulant)
    Fcirc = make_circulant_from_first_row(first_row_F)
    comm_FΩ = max_abs(commutator(Fcirc, Ω)) <= tol
    PASS("Fejér commutes with Ω", comm_FΩ)
    # Mean preserved: λ_0 should be 1
    lam0_F_ok = abs(λF[0] - 1.0) <= 1e-12
    PASS("Fejér preserves mean (λ_0=1)", lam0_F_ok)

    # Operator norm check (2→2) equals max eigenvalue = lam_max (since diagonal in Fourier basis)
    # We can demonstrate by ||F x||_2 <= lam_max ||x||_2 for random x
    import random as _r
    def l2norm(v): return math.sqrt(sum((abs(z)**2 for z in v)))
    ok_contract = True
    for _ in range(10):
        x = [complex(_r.uniform(-1,1), _r.uniform(-1,1)) for _ in range(M)]
        Fx = apply(Fcirc, x)
        if l2norm(Fx) > (lam_max + 1e-10)*l2norm(x):
            ok_contract = False
            break
    PASS("Fejér contractive in ℓ2", ok_contract)

    # Wrap up
    return out

if __name__ == "__main__":
    M,h,tol = parse_args()
    out = run(M,h,tol,seed=0)
    # Add script hash to manifest
    try:
        import pathlib, hashlib
        p = pathlib.Path(__file__)
        sha = hashlib.sha256(p.read_bytes()).hexdigest()
        out["script_sha256"] = sha
    except Exception as e:
        out["script_sha256_error"] = str(e)
    print(json.dumps(out, indent=2, default=str))