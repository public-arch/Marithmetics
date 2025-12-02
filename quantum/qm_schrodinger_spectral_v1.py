#!/usr/bin/env python3
# qm_schrodinger_spectral_v1.py — stdlib-only, deterministic
import math, cmath, json, random, sys

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

def fft_vec(F, x):
    # x as column vector (M)
    M = len(F)
    return [sum(F[j][k]*x[k] for k in range(M)) for j in range(M)]

def ifft_vec(F, X):
    Finv = conjT(F)
    M = len(F)
    return [sum(Finv[j][k]*X[k] for k in range(M)) for j in range(M)]

def l2norm2(vec):
    return sum((abs(z)**2 for z in vec))

def energy_grad_half(psi, F):
    # compute E = 1/2 ∫ |∂x ψ|^2 dx via spectral derivative
    M = len(psi)
    # Fourier coefficients
    Psi = fft_vec(F, psi)
    # k indices: 0..M-1 mapped to −M/2..M/2-1 like
    def k_index(n):
        return n if n <= M//2 else n-M
    # ∂x ψ ↔ i k ψ_k with k = 2π * k_index
    gradPsi = [1j*(2*math.pi*k_index(n))*Psi[n] for n in range(M)]
    # back to physical
    dpsi = ifft_vec(F, gradPsi)
    # energy ~ average of |dpsi|^2 / 2
    return 0.5*sum((abs(z)**2 for z in dpsi))/M

def init_wave(M, seed=0):
    random.seed(seed)
    # Gaussian wave packet modulated by a mode
    x = [j/M for j in range(M)]
    k0 = 6  # carrier mode
    width = 0.12
    psi = []
    for j in range(M):
        xc = x[j]-0.5
        env = math.exp(-(xc*xc)/(2*width*width))
        phase = cmath.exp(2j*math.pi*k0*x[j])
        psi.append(env*phase)
    # normalize
    n2 = l2norm2(psi)/M
    psi = [z/math.sqrt(n2) for z in psi]
    return psi


def residual_cn_norm(Psi_prev, Psi_next, dt, M):
    # CN-style residual in Fourier: (Ψ_{n+1}-Ψ_n)/dt + i (1/2) k^2 * (Ψ_{n+1}+Ψ_n)/2
    def k_index(n): return n if n <= M//2 else n-M
    res2 = 0.0
    for n in range(M):
        k2 = (2*math.pi*k_index(n))**2
        lhs = (Psi_next[n] - Psi_prev[n]) / dt
        rhs = -1j*0.5*k2*(Psi_next[n] + Psi_prev[n])/2.0
        r = lhs - rhs
        res2 += (abs(r)**2)
    return (res2/M)**0.5


def evolve_unitary(psi0, steps=200, dt=0.01):
    M = len(psi0)
    F = dft_matrix(M)
    Psi = fft_vec(F, psi0)
    def k_index(n): return n if n <= M//2 else n-M
    phase = [cmath.exp(-1j*0.5*(2*math.pi*k_index(n))**2 * dt) for n in range(M)]
    norms, energies, residuals = [], [], []
    psi = psi0[:]
    E0 = energy_grad_half(psi, F)
    for _ in range(steps):
        Psi_next = [Psi[n]*phase[n] for n in range(M)]
        psi = ifft_vec(F, Psi_next)
        norms.append(l2norm2(psi)/M)
        energies.append(energy_grad_half(psi, F))
        residuals.append(residual_cn_norm(Psi, Psi_next, dt, M))
        Psi = Psi_next
    return {"norms":norms, "energies":energies, "residuals":residuals, "E0":E0}

def evolve_wrong_phase(psi0, steps=100, dt=0.01, eps=0.05):
    M = len(psi0)
    F = dft_matrix(M)
    Psi = fft_vec(F, psi0)
    def k_index(n): return n if n <= M//2 else n-M
    phase = [cmath.exp(-1j*0.5*(1+eps)*(2*math.pi*k_index(n))**2 * dt) for n in range(M)]
    norms, energies, residuals = [], [], []
    psi = psi0[:]
    E0 = energy_grad_half(psi, F)
    for _ in range(steps):
        Psi_next = [Psi[n]*phase[n] for n in range(M)]
        psi = ifft_vec(F, Psi_next)
        norms.append(l2norm2(psi)/M)
        energies.append(energy_grad_half(psi, F))
        residuals.append(residual_cn_norm(Psi, Psi_next, dt, M))
        Psi = Psi_next
    return {"norms":norms, "energies":energies, "residuals":residuals, "E0":E0}

def evolve_diffusive(psi0, steps=60, dt=0.01):
    # non-unitary: real diffusion factor
    M = len(psi0)
    F = dft_matrix(M)
    Psi = fft_vec(F, psi0)
    def k_index(n): return n if n <= M//2 else n-M
    damp = [math.exp(-0.5*(2*math.pi*k_index(n))**2 * dt) for n in range(M)]
    norms = []
    psi = psi0[:]
    for _ in range(steps):
        Psi = [Psi[n]*damp[n] for n in range(M)]
        psi = ifft_vec(F, Psi)
        norms.append(l2norm2(psi)/M)
    return {"norms":norms}


def run_suite(M=64, steps=200, dt=0.01):
    psi0 = init_wave(M, seed=0)
    unit = evolve_unitary(psi0, steps=steps, dt=dt)
    wrong = evolve_wrong_phase(psi0, steps=steps//2, dt=dt, eps=0.05)
    diffu = evolve_diffusive(psi0, steps=steps//4, dt=dt)
    norm_dev = max(abs(n-1.0) for n in unit["norms"])
    E0 = unit["E0"]
    E_dev = max(abs(E - E0) for E in unit["energies"])/max(1.0, abs(E0))
    wrong_norm_dev = max(abs(n-1.0) for n in wrong["norms"])
    wrong_E_dev = max(abs(E - wrong["E0"]) for E in wrong["energies"])/max(1.0, abs(wrong["E0"]))
    diff_norm_drop = 1.0 - diffu["norms"][-1]
    passes = {
        "unitary_norm_preserved": bool(norm_dev <= 1e-12),
        "unitary_energy_constant": bool(E_dev <= 1e-6),
        "designed_fail_wrong_phase_trips": bool(max(wrong["residuals"]) >= 1e-2 and max(wrong["norms"])<=1.0+1e-12 and min(wrong["norms"])>=1.0-1e-12),
        "designed_fail_diffusion_trips": bool(diff_norm_drop >= 1e-2)
    }
    return {"params":{"M":M,"steps":steps,"dt":dt},
            "unitary":{"norm_dev":norm_dev,"E_rel_dev":E_dev,"residual_max":max(unit['residuals'])},
            "wrong_phase":{"norm_dev":wrong_norm_dev,"E_rel_dev":wrong_E_dev,"residual_max":max(wrong['residuals'])},
            "diffusion":{"norm_drop":diff_norm_drop},
            "passes":passes}
def main():
    out = run_suite()
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()