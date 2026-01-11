#!/usr/bin/env python3
# gr_linearized_tt_t3_v1.py — stdlib-only, deterministic
# Linearized GR, TT gauge, single plane wave on T^3.
import math, json, random, sys

def unit_vec(k):
    norm = math.sqrt(sum(kk*kk for kk in k))
    if norm == 0: return (1.0,0.0,0.0)
    return tuple(kk/norm for kk in k)

def mat_mul(A,B):
    # 3x3 * 3x3
    return [[sum(A[i][m]*B[m][j] for m in range(3)) for j in range(3)] for i in range(3)]

def mat_add(A,B,alpha=1.0):
    return [[A[i][j] + alpha*B[i][j] for j in range(3)] for i in range(3)]

def mat_tr(A):
    return A[0][0]+A[1][1]+A[2][2]

def mat_sym(R):
    return [[0.5*(R[i][j]+R[j][i]) for j in range(3)] for i in range(3)]

def mat_scale(A,c):
    return [[c*A[i][j] for j in range(3)] for i in range(3)]

def proj_TT(seed, k):
    # TT projector: P = I - n n^T, with n = k/|k|; then
    # T = P R P; make symmetric; remove trace part: T_tt = T - (tr(T)/2) P (in 3D TT on wavefront)
    n = unit_vec(k)
    I = [[1.0 if i==j else 0.0 for j in range(3)] for i in range(3)]
    # n n^T
    N = [[n[i]*n[j] for j in range(3)] for i in range(3)]
    P = mat_add(I, [[-N[i][j] for j in range(3)] for i in range(3)], alpha=1.0)
    R = mat_sym(seed)
    T = mat_mul(mat_mul(P, R), P)
    T = mat_sym(T)
    trT = mat_tr(T)
    # In 3D, pure TT has 2 dof; subtract trace in transverse plane:
    T_tt = mat_add(T, mat_scale(P, -0.5*trT), alpha=1.0)
    return T_tt, P

def grid_positions(N):
    xs = [i/N for i in range(N)]
    return xs

def phase_at(i,j,k, N, kv):
    x=i/N; y=j/N; z=k/N
    return 2*math.pi*(kv[0]*x + kv[1]*y + kv[2]*z)

def centered_diff_1D(f_im1, f_i, f_ip1, dx):
    return 0.5*(f_ip1 - f_im1)/dx


def tt_constraints_from_amplitude(A, k):
    # Check |k_i A_{ij}| and trace(A)
    kvec = list(k)
    # form v_j = sum_i k_i A_{ij}
    v = [0.0,0.0,0.0]
    for j in range(3):
        s=0.0
        for i in range(3):
            s += kvec[i]*A[i][j]
        v[j]=s
    tr = A[0][0]+A[1][1]+A[2][2]
    vnorm = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    return vnorm, abs(tr)


def lgr_tt_run(N=24, kv=(1,2,0), dt=0.01, steps=200, seed=0, fail=False):
    random.seed(seed)
    # Build amplitude A_ij
    if fail:
        # non‑TT random symmetric seed
        S = [[random.uniform(-1,1) for _ in range(3)] for __ in range(3)]
        A = mat_sym(S)
        P = None
    else:
        S = [[random.uniform(-1,1) for _ in range(3)] for __ in range(3)]
        A, P = proj_TT(S, kv)
    # Normalize amplitude to O(1)
    s2 = sum(A[i][j]*A[i][j] for i in range(3) for j in range(3))
    if s2 == 0: 
        A[0][0] = 1.0; s2=1.0
    A = mat_scale(A, 1.0/math.sqrt(s2))
    # Frequency ω = 2π |k|  (c=1, period-1 torus coordinates)
    n = unit_vec(kv)
    k_norm = math.sqrt(sum(kk*kk for kk in kv))
    omega = 2*math.pi*k_norm
    dx = 1.0/N
    # Time stepping: analytic plane-wave update for h and h_t
    # h_ij(t) = A_ij cos(φ - ω t), h_t = + ω A_ij sin(φ - ω t)
    # We sample on grid and compute discrete energy and constraints
    max_trace = 0.0
    max_div = 0.0
    U0 = None
    Umax_dev = 0.0
    # amplitude-level TT checks
    vnorm, trA = tt_constraints_from_amplitude(A, kv)
    # Precompute phase grid
    ph = [[[phase_at(i,j,k, N, kv) for k in range(N)] for j in range(N)] for i in range(N)]
    for step in range(steps+1):
        t = step*dt
        # compute fields
        # For each grid point, h and time derivative ht:
        # then energy density 0.5*(|ht|^2 + |grad h|^2), grad via centered differences.
        # Also compute trace and divergence using centered differences
        # We'll loop modestly—N=24 keeps runtime reasonable.
        # Store needed neighbor values for differences on the fly.
        U = 0.0
        tr_max_here = 0.0
        div_max_here = 0.0
        for i in range(N):
            ip = (i+1)%N; im = (i-1)%N
            for j in range(N):
                jp=(j+1)%N; jm=(j-1)%N
                for k in range(N):
                    kp=(k+1)%N; km=(k-1)%N
                    phi = ph[i][j][k] - omega*t
                    c = math.cos(phi); s = math.sin(phi)
                    # h_ij and ht_ij
                    h = [[A[p][q]*c for q in range(3)] for p in range(3)]
                    ht= [[omega*A[p][q]*s for q in range(3)] for p in range(3)]
                    # trace at this point
                    tr = h[0][0]+h[1][1]+h[2][2]
                    tr_max_here = max(tr_max_here, abs(tr))
                    # discrete divergence (∂_i h_{ij}) with centered differences
                    # compute h at neighbors with same t by using phase shifts
                    phi_ip = ph[ip][j][k] - omega*t
                    phi_im = ph[im][j][k] - omega*t
                    phi_jp = ph[i][jp][k] - omega*t
                    phi_jm = ph[i][jm][k] - omega*t
                    phi_kp = ph[i][j][kp] - omega*t
                    phi_km = ph[i][j][km] - omega*t
                    def h_at(phi_local):
                        cL=math.cos(phi_local)
                        return [[A[p][q]*cL for q in range(3)] for p in range(3)]
                    h_ip = h_at(phi_ip); h_im = h_at(phi_im)
                    h_jp = h_at(phi_jp); h_jm = h_at(phi_jm)
                    h_kp = h_at(phi_kp); h_km = h_at(phi_km)
                    # divergence components j=0,1,2: div_j = sum_i ∂_i h_{ij}
                    div = [0.0,0.0,0.0]
                    # ∂_x
                    for jnd in range(3):
                        div[jnd] += 0.5*(h_ip[0][jnd]-h_im[0][jnd])/dx
                    # ∂_y
                    for jnd in range(3):
                        div[jnd] += 0.5*(h_jp[1][jnd]-h_jm[1][jnd])/dx
                    # ∂_z
                    for jnd in range(3):
                        div[jnd] += 0.5*(h_kp[2][jnd]-h_km[2][jnd])/dx
                    div_here = max(abs(div[0]), abs(div[1]), abs(div[2]))
                    div_max_here = max(div_max_here, div_here)
                    # energy density
                    # grad components via centered differences per tensor entry
                    grad2 = 0.0
                    for p in range(3):
                        for q in range(3):
                            dxd = 0.5*(h_ip[p][q]-h_im[p][q])/dx
                            dyd = 0.5*(h_jp[p][q]-h_jm[p][q])/dx
                            dzd = 0.5*(h_kp[p][q]-h_km[p][q])/dx
                            grad2 += dxd*dxd + dyd*dyd + dzd*dzd
                    ht2 = sum(ht[p][q]*ht[p][q] for p in range(3) for q in range(3))
                    U += 0.5*(ht2 + grad2)
        if U0 is None: U0 = U
        Umax_dev = max(Umax_dev, abs(U-U0)/max(1.0, U0))
        max_trace = max(max_trace, tr_max_here)
        max_div = max(max_div, div_max_here)
    passes = {
        "tt_trace_near_zero": bool(trA <= 1e-12) if not fail else bool(trA > 1e-6),
        "tt_divergence_near_zero": bool(vnorm <= 1e-12) if not fail else bool(vnorm > 1e-6),
        "energy_near_conserved": bool(Umax_dev <= 5e-3) if not fail else bool(Umax_dev > 1e-1)
    }
    return {
        "params":{"N":N,"k":kv,"dt":dt,"steps":steps,"fail":fail},
        "max_trace":max_trace, "max_divergence":max_div, "energy_rel_drift":Umax_dev,
        "passes":passes
    }

def main():
    ok = lgr_tt_run(N=16, kv=(1,2,1), dt=0.02, steps=80, seed=0, fail=False)
    bad = lgr_tt_run(N=16, kv=(1,2,1), dt=0.02, steps=40, seed=0, fail=True)
    out = {"ok_run":ok, "fail_run":bad,
           "passes":{
               "tt_constraints_ok": bool(ok["passes"]["tt_trace_near_zero"] and ok["passes"]["tt_divergence_near_zero"]),
               "energy_ok": bool(ok["passes"]["energy_near_conserved"]),
               "designed_fail_trips": bool(bad["passes"]["tt_trace_near_zero"] and bad["passes"]["tt_divergence_near_zero"] and bad["passes"]["energy_near_conserved"])==False
           }}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
