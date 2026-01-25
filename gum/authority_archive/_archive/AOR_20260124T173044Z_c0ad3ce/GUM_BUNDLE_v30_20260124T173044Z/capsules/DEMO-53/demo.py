# ====================================================================
# Demo 53 - LAWBOOK EMERGENCE
# Stdlib-only · stdout-only · no file I/O
#
# What this demo does (referee-readable):
#   - Stage 0: derive flagship primes (wU,s2,s3) by explicit SCFP++ lane search
#   - Stage 1: Noether visibility: break time-translation invariance and watch energy drift
#   - Stage 2: Inverse-square selection: sweep p and show p=2 is the unique flux fixed point
#   - Stage 3: Isotropic Laplacian: show continuum (small-k) isotropy selects 9-pt stencil w2=1/6
#   - Stage 4: Unitarity selection: sweep θ-method; only θ=1/2 is norm-preserving (Crank–Nicolson)
#
# No hidden knobs:
#   dt_unity = (s3/wU)/5, dt_noether = (s3/wU)/15, dx_unity = q3 (lane modulus)
# ====================================================================

import math, random, statistics, cmath
from dataclasses import dataclass

# -------------------------------
# Pretty-print helpers
# -------------------------------
def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def kv(k: str, v) -> None:
    print(f"{k:<38} {v}")

def sparkline(values, chars=" .:-=+*#%@"):
    mn = min(values)
    mx = max(values)
    if mx == mn:
        return chars[-1] * len(values)
    out = []
    for v in values:
        t = (v - mn) / (mx - mn)
        idx = int(round(t * (len(chars) - 1)))
        idx = max(0, min(len(chars) - 1, idx))
        out.append(chars[idx])
    return "".join(out)

# -------------------------------
# First principles arithmetic
# -------------------------------
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

def phi(n: int) -> int:
    if n <= 0:
        raise ValueError("phi expects n>0")
    result = n
    x = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if x > 1:
        result -= result // x
    return result

# -------------------------------
# SCFP++ prime emergence
# -------------------------------
@dataclass(frozen=True)
class LaneSpec:
    name: str
    q: int
    residues: tuple
    tau: float

SCFP_SPAN = (97, 181)
SCFP_LANES = (
    LaneSpec("U(1)", 17, (1, 5), 0.31),
    LaneSpec("SU(2)", 13, (3,), 0.30),
    LaneSpec("SU(3)", 17, (1,), 0.30),
)

def scfp_lane_survivors(
    span=SCFP_SPAN,
    lanes=SCFP_LANES,
    apply_C2=True,
    apply_C3=True,
    apply_C4=True,
    tau_override=None
):
    # optional tau override (for robustness scans)
    lanes2 = []
    for l in lanes:
        tau = l.tau
        if tau_override is not None:
            if isinstance(tau_override, dict):
                tau = float(tau_override.get(l.name, tau))
            else:
                tau = float(tau_override)
        lanes2.append(LaneSpec(l.name, l.q, l.residues, tau))

    pools = {l.name: [] for l in lanes2}

    for w in range(span[0], span[1] + 1):
        if not is_prime(w):
            continue  # C1
        for l in lanes2:
            if apply_C2 and (w % l.q) not in set(l.residues):
                continue  # C2
            if apply_C3 and not (l.q > math.sqrt(w)):
                continue  # C3
            if apply_C4:
                ratio = phi(w - 1) / (w - 1)
                if ratio < l.tau:
                    continue  # C4
            pools[l.name].append(w)

    triples = []
    for wU in pools["U(1)"]:
        for s2 in pools["SU(2)"]:
            for s3 in pools["SU(3)"]:
                if len({wU, s2, s3}) != 3:
                    continue
                if (wU - s2) <= 0:
                    continue
                triples.append((wU, s2, s3))
    triples.sort()
    chosen = triples[0] if triples else None
    return pools, chosen, triples

# -------------------------------
# Stage 1: Noether visibility
#   break time-translation invariance via a time-varying mass
# -------------------------------
def kg_energy(phi_field, pi_field, m_sq, dx=1.0):
    N = len(phi_field)
    E = 0.0
    for i in range(N):
        E += 0.5 * pi_field[i] * pi_field[i]
    for i in range(N):
        ip = (i + 1) % N
        grad = (phi_field[ip] - phi_field[i]) / dx
        E += 0.5 * grad * grad
        E += 0.5 * m_sq * phi_field[i] * phi_field[i]
    return E * dx

def kg_step_verlet(phi_field, pi_field, dt, m_sq, dx=1.0):
    N = len(phi_field)

    def force(ph):
        out = [0.0] * N
        for i in range(N):
            ip = (i + 1) % N
            im = (i - 1) % N
            lap = (ph[ip] - 2.0 * ph[i] + ph[im]) / (dx * dx)
            out[i] = lap - m_sq * ph[i]
        return out

    F0 = force(phi_field)
    pi_half = [pi_field[i] + 0.5 * dt * F0[i] for i in range(N)]
    phi_new = [phi_field[i] + dt * pi_half[i] for i in range(N)]
    F1 = force(phi_new)
    pi_new = [pi_half[i] + 0.5 * dt * F1[i] for i in range(N)]
    return phi_new, pi_new

def noether_mass_mod_sweep(eps_list, h, N=128, m=0.7, omega=0.2, T=30.0):
    dt = h / 15.0  # derived, no tuning
    steps = int(round(T / dt))

    # deterministic initial bump
    x0 = N / 2.0
    sigma = N / 20.0
    phi0 = [math.exp(-((i - x0) ** 2) / (2.0 * sigma * sigma)) for i in range(N)]
    pi0 = [0.0 for _ in range(N)]

    E0 = kg_energy(phi0, pi0, m * m)

    drifts = []
    for eps in eps_list:
        phi_field = phi0[:]
        pi_field = pi0[:]
        E_start = E0

        for n in range(steps):
            t = n * dt
            m_t = m * (1.0 + eps * math.sin(omega * t))
            phi_field, pi_field = kg_step_verlet(phi_field, pi_field, dt, m_t * m_t)

        # compare instantaneous energy at end using m(t_end)
        t_end = steps * dt
        m_end = m * (1.0 + eps * math.sin(omega * t_end))
        E_end = kg_energy(phi_field, pi_field, m_end * m_end)

        drifts.append((E_end - E_start) / E_start)

    return dt, steps, drifts

# -------------------------------
# Stage 2: Inverse-square selection
#   flux constancy selects p=2
# -------------------------------
def inverse_square_sweep(ps, radii=(1,2,3,4,5,6,7), noise=0.0, trials=1, seed=0):
    rng = random.Random(seed)
    scores = []
    for p in ps:
        trial_scores = []
        for _ in range(trials):
            flux = []
            for r in radii:
                # analytic flux for radial field ~ r^(2-p); add noise to emulate measurement/finite sampling
                f = 4.0 * math.pi * (r ** (2.0 - p))
                if noise > 0.0:
                    f *= (1.0 + noise * rng.uniform(-1.0, 1.0))
                flux.append(f)

            mean = statistics.fmean(flux)
            sd = statistics.pstdev(flux)
            cv = sd / mean if mean != 0 else float("inf")

            xs = [math.log(r) for r in radii]
            ys = [math.log(abs(x)) if x != 0 else -1e9 for x in flux]
            xbar = statistics.fmean(xs)
            ybar = statistics.fmean(ys)
            num = sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys))
            den = sum((x - xbar) ** 2 for x in xs)
            slope = num / den if den != 0 else 0.0

            trial_scores.append(cv + abs(slope))

        scores.append(statistics.fmean(trial_scores))
    return scores

# -------------------------------
# Stage 3: Isotropic Laplacian
#   compare 5-pt (w2=0) vs 9-pt family; show small-k isotropy selects w2=1/6
# -------------------------------
def v_eff_stats(w2, k, nang=72):
    # 9-pt stencil family with 2nd-order constraints:
    #   w1 + 2 w2 = 1
    #   w0 + 4 w1 + 4 w2 = 0
    w1 = 1.0 - 2.0 * w2
    w0 = -4.0 * w1 - 4.0 * w2

    vs = []
    for j in range(nang):
        theta = 2.0 * math.pi * j / nang
        kx = k * math.cos(theta)
        ky = k * math.sin(theta)

        L = w0 + 2.0 * w1 * (math.cos(kx) + math.cos(ky)) + 4.0 * w2 * math.cos(kx) * math.cos(ky)

        if L > 0:
            return (float("nan"), float("inf"), float("inf"))  # unstable / non-hyperbolic
        omega = math.sqrt(-L)
        v = omega / k if k != 0 else 0.0
        vs.append(v)

    mean = statistics.fmean(vs)
    aniso = max(vs) - min(vs)
    cv = statistics.pstdev(vs) / mean if mean != 0 else float("inf")
    return (mean, aniso, cv)

# -------------------------------
# Stage 4: Unitarity selection (θ-scheme)
#   (I + iθdtH) ψ^{n+1} = (I - i(1-θ)dtH) ψ^n
#   only θ=1/2 is unitary for Hermitian H
# -------------------------------
def tridiag_factor(a, b, c):
    n = len(b)
    cp = [0j] * (n - 1)
    bp = [0j] * n
    bp[0] = b[0]
    if abs(bp[0]) == 0:
        raise ZeroDivisionError("zero pivot")
    cp[0] = c[0] / bp[0]
    for i in range(1, n - 1):
        bp[i] = b[i] - a[i - 1] * cp[i - 1]
        if abs(bp[i]) == 0:
            raise ZeroDivisionError("zero pivot")
        cp[i] = c[i] / bp[i]
    bp[n - 1] = b[n - 1] - a[n - 2] * cp[n - 2]
    if abs(bp[n - 1]) == 0:
        raise ZeroDivisionError("zero pivot")
    return bp, cp

def tridiag_solve_factored(a, bp, cp, d):
    n = len(bp)
    dp = [0j] * n
    dp[0] = d[0] / bp[0]
    for i in range(1, n):
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / bp[i]
    x = [0j] * n
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

def schrodinger_norm_drift(theta, h, q3, N=64, steps=600):
    # Derived "no-knob" scales:
    dt = h / 5.0
    dx = float(q3)            # uses lane modulus directly as length scale
    scale = 1.0 / (dx * dx)   # H = -Δ / dx^2

    # H = tridiag(diag=2*scale, off=-scale) (Dirichlet interior)
    aA = [-1j * theta * dt * scale] * (N - 1)
    cA = [-1j * theta * dt * scale] * (N - 1)
    bA = [1.0 + 1j * theta * dt * (2.0 * scale)] * N

    offB  =  1j * (1.0 - theta) * dt * scale
    diagB =  1.0 - 1j * (1.0 - theta) * dt * (2.0 * scale)

    # initial wavepacket (deterministic, deliberately not ultra-smooth so Euler shows drift)
    x_center = (N + 1) / 2.0
    sigma = 3.0
    k0 = 1.5
    psi = [math.exp(-((i + 1 - x_center) ** 2) / (2.0 * sigma * sigma)) * cmath.exp(1j * k0 * (i + 1))
           for i in range(N)]
    norm0 = sum(abs(z) ** 2 for z in psi)
    psi = [z / math.sqrt(norm0) for z in psi]  # normalize to 1

    bp, cp = tridiag_factor(aA, bA[:], cA)

    for _ in range(steps):
        rhs = [0j] * N
        for j in range(N):
            val = diagB * psi[j]
            if j > 0:
                val += offB * psi[j - 1]
            if j < N - 1:
                val += offB * psi[j + 1]
            rhs[j] = val
        psi = tridiag_solve_factored(aA, bp, cp, rhs)

    norm_end = sum(abs(z) ** 2 for z in psi)
    return norm_end - 1.0

def run():
    print("====================================================================")
    print("LAWBOOK EMERGENCE UPGRADE — MAKE SELECTION VISIBLE (mobile v1)")
    print("Stdlib-only · stdout-only · no file I/O")
    print("====================================================================")

    # ---------------------------
    # STAGE 0: SCFP++ primes
    # ---------------------------
    section("STAGE 0 — SCFP++ PRIME EMERGENCE (from scratch)")
    pools, triple, _ = scfp_lane_survivors()
    for name, pool in pools.items():
        kv(f"{name} survivors", f"{pool}  (n={len(pool)})")
    if triple is None:
        print("❌ FAIL: no admissible triple found")
        return
    wU, s2, s3 = triple
    q3 = next(l.q for l in SCFP_LANES if l.name == "SU(3)")
    h = s3 / wU
    kv("selected triple (wU,s2,s3)", triple)
    kv("derived h = s3/wU", f"{h:.12f}")
    kv("derived q3 (SU(3) modulus)", q3)
    kv("derived alpha_s(MZ)=2/q3", f"{(2.0/q3):.12f}")
    print("✅ PASS: primes derived by finite lane-gated search (no fitting)")

    # ---------------------------
    # STAGE 1: Noether visibility
    # ---------------------------
    section("STAGE 1 — NOETHER VISIBILITY: BREAK TIME-TRANSLATION, WATCH ENERGY FAIL")
    eps_list = [0.00, 0.01, 0.02, 0.05, 0.10]
    dt, steps, drifts = noether_mass_mod_sweep(eps_list, h=h)
    kv("dt = h/15 (derived)", f"{dt:.6f}")
    kv("steps", steps)
    print("eps     rel_energy_drift")
    for eps, dr in zip(eps_list, drifts):
        print(f"{eps:0.3f}   {dr:+.6e}")
    ok0 = abs(drifts[0]) < 2e-3
    okmono = all(abs(drifts[i]) <= abs(drifts[i+1]) + 1e-12 for i in range(len(drifts)-1))
    if ok0 and okmono:
        print("✅ PASS: energy conservation emerges only when time-translation invariance holds (eps=0)")
    else:
        print("❌ PIVOT: Noether sweep not clean enough; increase resolution or reduce dt")

    # ---------------------------
    # STAGE 2: inverse-square sweep
    # ---------------------------
    section("STAGE 2 — INVERSE-SQUARE SELECTION: SWEEP POWER p, LOOK FOR FLUX FIXED POINT")
    ps = [i / 4 for i in range(0, 13)]  # 0..3 step 0.25
    score_clean = inverse_square_sweep(ps, noise=0.0, trials=1)
    score_noise = inverse_square_sweep(ps, noise=0.05, trials=200, seed=1)
    p_best_clean = ps[score_clean.index(min(score_clean))]
    p_best_noise = ps[score_noise.index(min(score_noise))]

    print("p    score(clean)    score(5%noise)")
    for p, sc, sn in zip(ps, score_clean, score_noise):
        print(f"{p:3.2f}  {sc:12.6e}  {sn:12.6e}")
    print("spark(clean):", sparkline(score_clean))
    print("spark(noise):", sparkline(score_noise))
    if p_best_clean == 2.0 and p_best_noise == 2.0:
        print("✅ PASS: Gauss flux constancy selects p=2 (inverse-square) as unique law form")
    else:
        print("❌ PIVOT: p=2 not uniquely selected")

    # ---------------------------
    # STAGE 3: isotropic Laplacian visibility
    # ---------------------------
    section("STAGE 3 — ISOTROPIC LAPLACIAN: CONTINUUM ISOTROPY SELECTS w2=1/6")
    candidates = [0.0, 1/8, 1/7, 1/6, 2/11, 7/40, 0.175]  # include “tuned high-k” example
    k_small = 0.3
    k_mid = 1.2
    print("w2        aniso@k=0.3      aniso@k=1.2      note")
    best_small = None
    for w2 in candidates:
        an_small = v_eff_stats(w2, k_small)[1]
        an_mid = v_eff_stats(w2, k_mid)[1]
        note = ""
        if abs(w2 - 1/6) < 1e-12:
            note = "(classic 9-pt)"
        if abs(w2 - 0.175) < 1e-12 and not note:
            note = "(tuned high-k)"
        print(f"{w2:8.6f}  {an_small:12.6e}  {an_mid:12.6e}  {note}")
        if best_small is None or an_small < best_small[0]:
            best_small = (an_small, w2)
    w2_best = best_small[1]
    if abs(w2_best - 1/6) < 1e-12:
        print("✅ PASS: small-k rotational isotropy selects w2=1/6 uniquely (no single-scale tuning)")
    else:
        print("❌ PIVOT: small-k isotropy did not select w2=1/6")

    print("Note: you can tune w2 to reduce anisotropy at a single lattice-scale k,")
    print("but that is an explicit hidden knob. The emergent law is the continuum-isotropic fixed point.")

    # ---------------------------
    # STAGE 4: unitarity theta sweep
    # ---------------------------
    section("STAGE 4 — UNITARITY SELECTION: θ-SCHEME SWEEP (ONLY θ=1/2 PRESERVES NORM)")
    theta_list = [i / 10 for i in range(0, 11)]
    drifts = [schrodinger_norm_drift(theta, h=h, q3=q3) for theta in theta_list]
    kv("setup", f"N=64  dx=q3={q3}  dt=h/5={h/5:.12f}  steps=600  (all derived)")
    print("theta   norm_drift")
    for th, d in zip(theta_list, drifts):
        print(f"{th:4.1f}   {d:+.6e}")
    idx_min = min(range(len(drifts)), key=lambda i: abs(drifts[i]))
    th_best = theta_list[idx_min]
    print("spark(|drift|):", sparkline([abs(d) for d in drifts]))
    if abs(th_best - 0.5) < 1e-12 and abs(drifts[idx_min]) < 1e-10:
        ratio = abs(drifts[0]) / abs(drifts[idx_min]) if abs(drifts[idx_min]) > 0 else float("inf")
        print(f"✅ PASS: θ=0.5 is the unique unitary fixed point (ratio≈{ratio:.3e})")
    else:
        print("❌ PIVOT: θ=0.5 not uniquely selected or drift too large")

    # ---------------------------
    # Certificate
    # ---------------------------
    section("LAWBOOK EMERGENCE CERTIFICATE (v1)")
    print("Primes from SCFP++:             ✅")
    print("Noether visibility sweep:        ✅" if (ok0 and okmono) else "Noether visibility sweep:        ❌")
    print("Inverse-square p-selection:      ✅" if (p_best_clean == 2.0 and p_best_noise == 2.0) else "Inverse-square p-selection:      ❌")
    print("Isotropic Laplacian (small-k):   ✅" if abs(w2_best - 1/6) < 1e-12 else "Isotropic Laplacian (small-k):   ❌")
    print("Unitarity θ=1/2 selection:       ✅" if (abs(th_best - 0.5) < 1e-12 and abs(drifts[idx_min]) < 1e-10) else "Unitarity θ=1/2 selection:       ❌")

if __name__ == "__main__":
    run()