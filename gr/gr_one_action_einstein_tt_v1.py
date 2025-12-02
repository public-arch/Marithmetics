#!/usr/bin/env python3
# gr_one_action_einstein_tt_v1.py — stdlib-only, deterministic
import math, json, random, sys

def unit_vec(k):
    n = math.sqrt(sum(kk*kk for kk in k))
    if n==0: return (1.0,0.0,0.0)
    return tuple(kk/n for kk in k)

def mat_sym(R): return [[0.5*(R[i][j]+R[j][i]) for j in range(3)] for i in range(3)]
def mat_add(A,B,alpha=1.0): return [[A[i][j]+alpha*B[i][j] for j in range(3)] for i in range(3)]
def mat_scale(A,c): return [[c*A[i][j] for j in range(3)] for i in range(3)]
def mat_tr(A): return A[0][0]+A[1][1]+A[2][2]

def proj_TT(seed, k):
    # P = I - n n^T; T = P R P; symmetrize; remove transverse trace
    n = unit_vec(k)
    I = [[1.0 if i==j else 0.0 for j in range(3)] for i in range(3)]
    N = [[n[i]*n[j] for j in range(3)] for i in range(3)]
    P = [[I[i][j]-N[i][j] for j in range(3)] for i in range(3)]
    R = mat_sym(seed)
    PR = [[sum(P[i][m]*R[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    T  = [[sum(PR[i][m]*P[m][j] for m in range(3)) for j in range(3)] for i in range(3)]
    # sym + remove trace in transverse plane
    T = mat_sym(T); trT = mat_tr(T)
    T_tt = mat_add(T, mat_scale(P, -0.5*trT), alpha=1.0)
    return T_tt

def amp_norm2(A): return sum(A[i][j]*A[i][j] for i in range(3) for j in range(3))

def zeros3(N): 
    return [[ [0.0]*N for _ in range(N)] for __ in range(N)]

def lap3(u):
    # 3D periodic 2nd-order Laplacian
    N=len(u); out=zeros3(N)
    for i in range(N):
        ip=(i+1)%N; im=(i-1)%N
        for j in range(N):
            jp=(j+1)%N; jm=(j-1)%N
            ui=u[i]; uip=u[ip]; uim=u[im]
            for k in range(N):
                kp=(k+1)%N; km=(k-1)%N
                out[i][j][k] = (uip[j][k]+uim[j][k]+ui[jp][k]+ui[jm][k]+ui[j][kp]+ui[j][km]-6.0*ui[j][k])
    return out

def add3(a,b,alpha=1.0):
    N=len(a); out=zeros3(N)
    for i in range(N):
        ai=a[i]; bi=b[i]
        for j in range(N):
            aij=ai[j]; bij=bi[j]; o=out[i][j]
            for k in range(N):
                o[k] = aij[k] + alpha*bij[k]
    return out

def scale3(a,c):
    N=len(a); out=zeros3(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                out[i][j][k] = c*a[i][j][k]
    return out

def l2norm3(a):
    N=len(a); s=0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                s += a[i][j][k]*a[i][j][k]
    return math.sqrt(s/(N*N*N))

def energy_wave(h_prev, h_curr, c2, dt):
    # Discrete energy for wave eq: 1/2 [ |(h^n - h^{n-1})/dt|^2 + c^2 |∇_h h^n|^2 ]
    N=len(h_curr)
    # velocity term
    vel = add3(h_curr, h_prev, alpha=-1.0)
    vel = scale3(vel, 1.0/dt)
    vel2 = l2norm3(vel)**2
    # grad^2 via laplacian identity: ∑|∇h|^2 ≈ - ∑ h Δh
    Lap = lap3(h_curr)
    # inner product <h, -Δh>
    s=0.0
    for i in range(N):
        for j in range(N):
            for k in range(N):
                s += h_curr[i][j][k]*(-Lap[i][j][k])
    grad2 = s/(N*N*N)
    return 0.5*(vel2 + c2*grad2)

def initialize_tt_wave(N=16, kv=(1,2,1), amp=1.0, seed=0):
    random.seed(seed)
    S = [[random.uniform(-1,1) for _ in range(3)] for __ in range(3)]
    A = proj_TT(S, kv)
    n2 = amp_norm2(A); 
    if n2==0.0: A[0][0]=1.0; n2=1.0
    A = mat_scale(A, amp/math.sqrt(n2))
    # phase grid
    ph = [[[2*math.pi*(kv[0]*i/N + kv[1]*j/N + kv[2]*k/N) for k in range(N)] for j in range(N)] for i in range(N)]
    return A, ph

def field_from_amp(A, ph, phase_shift=0.0):
    N=len(ph)
    u=zeros3(N)
    for i in range(N):
        for j in range(N):
            for k in range(N):
                u[i][j][k] = sum(A[p][q]*math.cos(ph[i][j][k]-phase_shift) for p in range(3) for q in range(3))
    return u

def simulate(N=16, kv=(1,2,1), dt=0.02, steps=80, c=1.0, fail=False):
    # Initialize TT amplitude and plane wave samples at t=0 and t=dt (analytic)
    A, ph = initialize_tt_wave(N, kv, amp=1.0, seed=0)
    omega = 2*math.pi*math.sqrt(kv[0]**2+kv[1]**2+kv[2]**2)*c
    h0 = field_from_amp(A, ph, phase_shift=0.0)
    h1 = field_from_amp(A, ph, phase_shift=omega*dt)
    # Evolve by discrete EL (centered second-difference wave scheme from action)
    c2 = c*c; tol = 5e-3
    max_res = 0.0
    U0 = energy_wave(h0, h1, c2, dt)
    Umax_dev = 0.0
    hnm1, hn = h0, h1
    for n in range(1, steps):
        # Compute Laplacian at hn
        Lhn = lap3(hn)
        # Variational update: h^{n+1} = 2h^n - h^{n-1} + c^2 dt^2 Δ_h h^n
        if not fail:
            hnp1 = add3(add3(hn, hnm1, alpha=-1.0), scale3(Lhn, c2*dt*dt), alpha=1.0)
            hnp1 = add3(hnp1, hn, alpha=1.0)  # add back 2*hn
        else:
            # wrong sign on Laplacian (designed FAIL)
            hnp1 = add3(add3(hn, hnm1, alpha=-1.0), scale3(Lhn, -c2*dt*dt), alpha=1.0)
            hnp1 = add3(hnp1, hn, alpha=1.0)
        # EL residual at hn: r = h^{n+1} - 2hn + h^{n-1} - c^2 dt^2 Δ_h hn
        r = add3(add3(hnp1, hn, alpha=-2.0), add3(hnm1, scale3(Lhn, -c2*dt*dt), alpha=1.0), alpha=1.0)
        res = l2norm3(r)
        if res > max_res: max_res = res
        # energy
        U = energy_wave(hn, hnp1, c2, dt)
        Umax_dev = max(Umax_dev, abs(U-U0)/max(1.0, U0))
        hnm1, hn = hn, hnp1
    # TT constraints at amplitude level
    kvec = list(kv)
    v = [sum(kvec[i]*A[i][j] for i in range(3)) for j in range(3)]
    vnorm = math.sqrt(sum(vi*vi for vi in v))
    trA = abs(mat_tr(A))
    passes = {
        "EL_residual_small": bool(max_res <= 1e-9) if not fail else bool(max_res > 1e-3),
        "energy_near_conserved": bool(Umax_dev <= 3e-3) if not fail else bool(Umax_dev > 1e-1),
        "TT_constraints": bool(vnorm <= 1e-12 and trA <= 1e-12) if not fail else bool(vnorm > 1e-6 or trA > 1e-6)
    }
    return {
        "params":{"N":N,"k":kv,"dt":dt,"steps":steps,"c":c,"fail":fail},
        "omega":omega,"EL_residual_max":max_res,"energy_rel_drift":Umax_dev,
        "vnorm_k_dot_A":vnorm,"trace_A":trA,"passes":passes
    }

def main():
    ok = simulate(N=14, kv=(1,1,1), dt=0.015, steps=90, c=1.0, fail=False)
    bad= simulate(N=14, kv=(1,1,1), dt=0.015, steps=45, c=1.0, fail=True)
    out = {"ok_run":ok,"fail_run":bad,
           "passes":{
               "EL_ok": bool(ok["passes"]["EL_residual_small"]),
               "energy_ok": bool(ok["passes"]["energy_near_conserved"]),
               "TT_ok": bool(ok["passes"]["TT_constraints"]),
               "designed_fail_trips": bool(bad["passes"]["EL_residual_small"]==False and bad["passes"]["energy_near_conserved"]==False)
           }}
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()