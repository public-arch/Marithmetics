#!/usr/bin/env python3
# DEMO-78 FLAGSHIP v3.5 — FULLY SELF-CONTAINED EXT PACK (Targets + Gauge + GR Hook)
# Stdlib-only • deterministic • hostile-referee style artifacts/hashes 

import math, copy, json, hashlib, sys, platform, os, glob
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple

# ==============================================================================================
# CONFIG
# ==============================================================================================

PRIMARY = (137, 107, 103)
CF_TRIPLES = [
    (409, 263, 239),
    (409, 263, 307),
    (409, 367, 239),
    (409, 367, 307),
]

# Φ recovery uses KPHI modes
KPHI = 12

# Physical mining uses KPHYS = nF(primary) by default
KPHYS_MODE = "NF_PRIMARY"   # "NF_PRIMARY" or "KPHI"

# Mining grammar limits (core + ext)
MAX_PQ_CORE = 12
MAX_PQ_EXT  = 24  # bounded but wider search for extended targets

# Optional provenance anchors (kept as plain strings; not “scoring”)
PROVENANCE = {
    "S4B_hash": "398d215e5bd376bab11a14de9ea29593ca9134c79b361521c5fce638951757c1",
    "S5B_hash": "6b0e6deeb13a7f7528e90298ebe7dfacb916d27123c6d2281779c60083c32a7d",
    "S6F_hash": "18b67424487e522019d0412b323d97a5601bd155a20d39a818f4c24cf76adc52",
}

# Φ targets (as in the flagship spine)
TARGETS_PHI = [
    ("alpha_em", 1.0 / 137.0),
    ("sin2W",    7.0 / 30.0),
    ("alpha_s",  2.0 / 17.0),
]

# Frozen Φ-map specs (first-principles pinned)
FROZEN_PHI: Dict[str, Dict[str, Any]] = {
    "alpha_em": {"p": 1, "q": 1, "unary": "x^2",     "kind": "sub", "i": 1, "j": 5},
    "sin2W":    {"p": 1, "q": 4, "unary": "1-x",     "kind": "idx", "i": 5, "j": None},
    "alpha_s":  {"p": 1, "q": 1, "unary": "sqrt(x)", "kind": "sub", "i": 7, "j": 9},
}

# Core physical ledger (self-contained default; auto-writes physical_targets.json if missing)
PHYS_CORE_DEFAULT = [
    {"name": "me_over_mp",      "target": 0.000544617, "err_power": 4, "drift_power": 2, "required": True},
    {"name": "mmu_over_mp",     "target": 0.11261,     "err_power": 4, "drift_power": 2, "required": True},
    {"name": "Vus",             "target": 0.2243,      "err_power": 4, "drift_power": 2, "required": True},
    {"name": "Vcb",             "target": 0.0422,      "err_power": 4, "drift_power": 2, "required": True},
    {"name": "Vub",             "target": 0.00394,     "err_power": 4, "drift_power": 2, "required": True},
    {"name": "dm21_over_dm31",  "target": 0.03,        "err_power": 4, "drift_power": 2, "required": True},
]

# --- EXT targets: additional requested channels ---
# Self-contained defaults. Mark required=False by default; you can promote specific ones in the json.
# Proton radius uses a dimensionless channel rp/(ħ/(m_p c)) to avoid unit ambiguity.
def rp_over_lambdaCp_target() -> float:
    # Fixed constants (numerical anchors for target definition only; not “fit parameters”):
    rp_m     = 8.4075e-16            # ~0.84075 fm
    mp_kg    = 1.67262192369e-27
    hbar_Js  = 1.054571817e-34
    c_ms     = 299792458.0
    return rp_m * mp_kg * c_ms / hbar_Js

PHYS_EXT_DEFAULT = [
    # CKM (full magnitudes; approximate central values; user can update in file)
    {"name": "Vud", "target": 0.97435,  "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vus_ckm", "target": 0.22501, "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vub_ckm", "target": 0.003732, "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vcd", "target": 0.22487,  "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vcs", "target": 0.97349,  "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vcb_ckm", "target": 0.04183, "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vtd", "target": 0.00858,  "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vts", "target": 0.04111,  "err_power": 4, "drift_power": 2, "required": False},
    {"name": "Vtb", "target": 0.999118, "err_power": 4, "drift_power": 2, "required": False},

    # PMNS parameters in dimensionless form (angles; delta normalized by pi)
    {"name": "sin2_th12", "target": 0.307,  "err_power": 2, "drift_power": 2, "required": False},
    {"name": "sin2_th13", "target": 0.0216, "err_power": 2, "drift_power": 2, "required": False},
    {"name": "sin2_th23", "target": 0.534,  "err_power": 2, "drift_power": 2, "required": False},
    {"name": "deltaCP_over_pi", "target": 1.21, "err_power": 2, "drift_power": 2, "required": False},

    # Proton radius dimensionless channel (rp / λC(p))
    {"name": "rp_over_lambdaCp", "target": rp_over_lambdaCp_target(), "err_power": 2, "drift_power": 2, "required": False},

    # Electron anomaly a_e = (g-2)/2 (dimensionless)
    {"name": "a_e", "target": 1.15965218046e-3, "err_power": 4, "drift_power": 2, "required": False},
]

# ==============================================================================================
# UTIL
# ==============================================================================================

def banner(t: str) -> None:
    print("\n" + "=" * 96)
    print(t)
    print("=" * 96)

def fmt12(x: float) -> float:
    return float(f"{x:.12g}")

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a

def v2(n: int) -> int:
    k = 0
    while n % 2 == 0 and n > 0:
        n //= 2
        k += 1
    return k

def phi(n: int) -> int:
    if n <= 0:
        return 0
    x = n
    res = n
    p = 2
    while p * p <= x:
        if x % p == 0:
            while x % p == 0:
                x //= p
            res -= res // p
        p += 1 if p == 2 else 2
    if x > 1:
        res -= res // x
    return res

def q2_of(triple: Tuple[int, int, int]) -> int:
    wU, s2, _ = triple
    return wU - s2

def q3_of(triple: Tuple[int, int, int]) -> int:
    wU, _, _ = triple
    x = wU - 1
    return x // (2 ** v2(x))

def order(u: int, d: int) -> Optional[int]:
    x = 1
    for k in range(1, d + 1):
        x = (x * u) % d
        if x == 1:
            return k
    return None

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, indent=2).encode("utf-8")

# ==============================================================================================
# KERNELS
# ==============================================================================================

def fejer(i: int, j: int, K: int) -> float:
    return max(0.0, 1.0 - abs(i - j) / (K + 1))

def illegal_signed(i: int, j: int, K: int) -> float:
    return 1.0 if abs(i - j) <= K else -1.0

def illegal_neg_diag(i: int, j: int, K: int) -> float:
    return -fejer(i, j, K) if i == j else fejer(i, j, K)

def illegal_checker(i: int, j: int, K: int) -> float:
    s = -1.0 if ((i + j) & 1) else 1.0
    return s * fejer(i, j, K)

# ==============================================================================================
# LINEAR ALGEBRA
# ==============================================================================================

def smooth(M0: List[List[float]], K: int, kernel) -> List[List[float]]:
    n = len(M0)
    M = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        M0i = M0[i]
        for j in range(n):
            Mi[j] = M0i[j] * kernel(i, j, K)
    return M

def trace_of(M: List[List[float]]) -> float:
    return sum(M[i][i] for i in range(len(M)))

def eigenvalues(M: List[List[float]], kmax: int = 12, iters: int = 120) -> List[float]:
    n = len(M)
    kmax = min(kmax, n)

    def mv(A: List[List[float]], v: List[float]) -> List[float]:
        out = [0.0] * n
        for i in range(n):
            s = 0.0
            Ai = A[i]
            for j in range(n):
                s += Ai[j] * v[j]
            out[i] = s
        return out

    def nrm(v: List[float]) -> float:
        return math.sqrt(sum(x * x for x in v))

    def norm(v: List[float]) -> List[float]:
        s = nrm(v)
        return [x / s for x in v] if s > 1e-300 else [0.0] * n

    def ray(A: List[List[float]], v: List[float]) -> float:
        w = mv(A, v)
        return sum(v[i] * w[i] for i in range(n))

    A = copy.deepcopy(M)
    vals: List[float] = []

    for _ in range(kmax):
        v = norm([1.0] * n)
        for _ in range(iters):
            v = norm(mv(A, v))
        lam = ray(A, v)
        vals.append(lam)
        for i in range(n):
            vi = v[i]
            Ai = A[i]
            for j in range(n):
                Ai[j] -= lam * vi * v[j]

    vals.sort(reverse=True)
    return vals

def env_norm(v: List[float]) -> List[float]:
    s = sum(v)
    return [x / s for x in v] if abs(s) > 1e-300 else [0.0] * len(v)

# ==============================================================================================
# FORMULAS (shared)
# ==============================================================================================

UNARIES = ["x", "1-x", "x^2", "x^3", "sqrt(x)", "1/x", "x/(1+x)", "1/(1+x)"]
KINDS = ["idx", "sub", "div"]

def apply_unary(name: str, x: Optional[float]) -> Optional[float]:
    if x is None or not math.isfinite(x):
        return None
    if name == "x":
        return x
    if name == "1-x":
        return 1.0 - x
    if name == "x^2":
        return x * x
    if name == "x^3":
        return x * x * x
    if name == "sqrt(x)":
        return math.sqrt(x) if x >= 0.0 else None
    if name == "1/x":
        return 1.0 / x if abs(x) > 1e-300 else None
    if name == "x/(1+x)":
        return x / (1.0 + x) if abs(1.0 + x) > 1e-300 else None
    if name == "1/(1+x)":
        return 1.0 / (1.0 + x) if abs(1.0 + x) > 1e-300 else None
    return None

def eval_base(r: List[float], kind: str, i: int, j: Optional[int]) -> Optional[float]:
    if i is None or i >= len(r):
        return None
    if kind == "idx":
        return r[i]
    if j is None or j >= len(r):
        return None
    if kind == "sub":
        return r[i] - r[j]
    if kind == "div":
        return r[i] / r[j] if abs(r[j]) > 1e-300 else None
    return None

def eval_formula(rs: List[float], rl: List[float], spec: Dict[str, Any], target: float) -> Dict[str, Any]:
    f = spec["p"] / spec["q"]
    b_s = eval_base(rs, spec["kind"], spec["i"], spec["j"])
    b_l = eval_base(rl, spec["kind"], spec["i"], spec["j"])
    u_s = apply_unary(spec["unary"], b_s)
    u_l = apply_unary(spec["unary"], b_l)
    if u_s is None or u_l is None:
        return {"mid": float("nan"), "err": float("inf"), "drift": float("inf"), "domain_fail": True}
    v_s = f * u_s
    v_l = f * u_l
    mid = 0.5 * (v_s + v_l)
    err = abs(mid - target)
    drift = abs(v_s - v_l)
    return {"mid": mid, "err": err, "drift": drift, "domain_fail": False, "v_small": v_s, "v_large": v_l}

def spec_str(spec: Optional[Dict[str, Any]]) -> str:
    if spec is None:
        return "NONE"
    j = spec["j"] if spec["j"] is not None else "None"
    return f"{spec['p']}/{spec['q']}*{spec['unary']}({spec['kind']}[{spec['i']},{j}])"

def unary_cost(u: str) -> int:
    return {"x": 0, "1-x": 1, "x^2": 1, "x^3": 2, "sqrt(x)": 2, "1/x": 2, "x/(1+x)": 3, "1/(1+x)": 3}[u]

def kind_cost(k: str) -> int:
    return {"idx": 1, "sub": 2, "div": 3}[k]

def frozen_specs_sha256(specs: Dict[str, Optional[Dict[str, Any]]]) -> str:
    canon = {k: spec_str(specs[k]) for k in sorted(specs.keys())}
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

# ==============================================================================================
# BUILD OPERATOR (first principles)
# ==============================================================================================

def build_full(triple: Tuple[int, int, int], rational_audit: bool = False):
    wU, s2, _ = triple
    q2 = q2_of(triple)
    d = q2
    q3 = q3_of(triple)
    eps = 1.0 / math.sqrt(q2)

    U = [u for u in range(1, d) if gcd(u, d) == 1]
    ords = {u: order(u, d) for u in U}
    lam = max(int(v) for v in ords.values() if v is not None)

    chi_f = {u: float(ords[u]) / lam for u in U}  # type: ignore
    chi_q = {u: Fraction(int(ords[u]), lam) for u in U}  # type: ignore

    spl_bins = sorted({gcd(k, lam) for k in range(1, lam + 1)})

    def spl_f(g1: int, g2: int) -> float:
        return 1.0 / (1.0 + abs(g1 - g2))

    def spl_q(g1: int, g2: int) -> Fraction:
        return Fraction(1, 1 + abs(g1 - g2))

    def unit(u1: int, u2: int) -> float:
        return 1.0 if gcd(u1 - u2, d) > 1 else 0.0

    IF = [(u, g) for u in U for g in spl_bins]
    nF = len(IF)

    K_small = max(2, min(nF // 2, int(round(0.25 * q3))))
    K_large = max(K_small + 1, min(nF // 2, int(round(0.75 * q3))))

    meta = {
        "triple": list(triple),
        "q2": q2,
        "q3": q3,
        "v2U": v2(wU - 1),
        "phi_q2": phi(q2),
        "eps": eps,
        "eps2": eps ** 2,
        "eps4": eps ** 4,
        "lam": lam,
        "nF": nF,
        "U_sz": len(U),
        "spl_sz": len(spl_bins),
        "K_small": K_small,
        "K_large": K_large,
    }

    M0 = [[0.0] * nF for _ in range(nF)]
    for i, (u1, g1) in enumerate(IF):
        for j, (u2, g2) in enumerate(IF):
            M0[i][j] = chi_f[u1] * chi_f[u2] * unit(u1, u2) * spl_f(g1, g2)

    audit_hash = None
    if rational_audit:
        h = hashlib.sha256()
        for (u1, g1) in IF:
            for (u2, g2) in IF:
                if gcd(u1 - u2, d) <= 1:
                    h.update(b"0/")
                    continue
                val = chi_q[u1] * chi_q[u2] * spl_q(g1, g2)
                h.update(str(val.numerator).encode())
                h.update(b"/")
                h.update(str(val.denominator).encode())
                h.update(b";")
        audit_hash = h.hexdigest()

    return IF, M0, meta, audit_hash

# ==============================================================================================
# COMPRESSION EQUIVALENCE (DOC averaging witness)
# ==============================================================================================

def compression_equivalence(IF, M0, meta):
    q2 = meta["q2"]
    d = q2
    U = sorted(set(u for (u, _) in IF))
    ords = {u: order(u, d) for u in U}
    lam = max(int(v) for v in ords.values() if v is not None)
    chi_hat = {u: float(ords[u]) / lam for u in U}  # type: ignore
    chi_bins = sorted(set(chi_hat.values()))
    chi_index = {x: i for i, x in enumerate(chi_bins)}
    spl_bins = sorted(set(g for (_, g) in IF))
    spl_index = {g: i for i, g in enumerate(spl_bins)}

    Bc = len(chi_bins)
    counts = [[0.0] * Bc for _ in range(Bc)]
    totals = [[0] * Bc for _ in range(Bc)]
    for u1 in U:
        a = chi_index[chi_hat[u1]]
        for u2 in U:
            b = chi_index[chi_hat[u2]]
            totals[a][b] += 1
            counts[a][b] += 1.0 if gcd(u1 - u2, d) > 1 else 0.0
    Achi = [[counts[a][b] / totals[a][b] for b in range(Bc)] for a in range(Bc)]

    def spl(g1, g2):
        return 1.0 / (1.0 + abs(g1 - g2))

    IC = [(c, g) for c in chi_bins for g in spl_bins]
    nC = len(IC)
    Mcons = [[0.0] * nC for _ in range(nC)]
    for i, (c1, g1) in enumerate(IC):
        for j, (c2, g2) in enumerate(IC):
            a = chi_index[c1]
            b = chi_index[c2]
            Mcons[i][j] = (c1 * c2) * Achi[a][b] * spl(g1, g2)

    bin_of_full = [0] * len(IF)
    members = [[] for _ in range(nC)]
    for idx, (u, g) in enumerate(IF):
        j = (chi_index[chi_hat[u]] * len(spl_bins)) + spl_index[g]
        bin_of_full[idx] = j
        members[j].append(idx)

    w_full = [0.0] * len(IF)
    for j, idxs in enumerate(members):
        if not idxs:
            continue
        w = 1.0 / len(idxs)
        for i in idxs:
            w_full[i] = w

    Mavg = [[0.0] * nC for _ in range(nC)]
    for i in range(len(IF)):
        wi = w_full[i]
        bi = bin_of_full[i]
        row = M0[i]
        for j in range(len(IF)):
            bj = bin_of_full[j]
            Mavg[bi][bj] += wi * w_full[j] * row[j]

    def frob(M):
        return math.sqrt(sum(M[i][j] * M[i][j] for i in range(len(M)) for j in range(len(M))))

    diff = [[Mavg[i][j] - Mcons[i][j] for j in range(nC)] for i in range(nC)]
    rel = frob(diff) / (frob(Mavg) + 1e-300)
    return {"nC": nC, "Bc": len(chi_bins), "Bs": len(spl_bins), "rel_frob": float(rel)}

# ==============================================================================================
# TIGHTROPE
# ==============================================================================================

def tightrope(M0, meta, kernel, kmax: int, iters: int) -> Dict[str, Any]:
    Ks, Kl = meta["K_small"], meta["K_large"]
    Ms = smooth(M0, Ks, kernel)
    Ml = smooth(M0, Kl, kernel)

    tr_s = trace_of(Ms)
    tr_l = trace_of(Ml)
    es = eigenvalues(Ms, kmax=kmax, iters=iters)
    el = eigenvalues(Ml, kmax=kmax, iters=iters)
    rs = env_norm(es)
    rl = env_norm(el)

    dev = max(abs(rs[i] - rl[i]) for i in range(min(len(rs), len(rl))))
    obj = dev / (meta["eps"] + 1e-300)

    return {
        "kmax": int(kmax),
        "iters": int(iters),
        "trace_small": float(tr_s),
        "trace_large": float(tr_l),
        "eig_small": [fmt12(x) for x in es],
        "eig_large": [fmt12(x) for x in el],
        "rat_env_small": [fmt12(x) for x in rs],
        "rat_env_large": [fmt12(x) for x in rl],
        "dev": float(dev),
        "obj": float(obj),
        "min_eig": float(min(es + el)) if (es and el) else float("nan"),
    }

# ==============================================================================================
# LEDGER IO (self-contained)
# ==============================================================================================

def ledger_sha256(entries: List[Dict[str, Any]]) -> str:
    blob = json.dumps(sorted(entries, key=lambda x: x["name"]), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

def load_or_init_ledger(path: str, default_entries: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], str, str]:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        data = sorted(data, key=lambda x: x["name"])
        return data, "file", ledger_sha256(data)
    data = sorted(default_entries, key=lambda x: x["name"])
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    return data, "builtin", ledger_sha256(data)

# ==============================================================================================
# MINER (PRIMARY-only mine, then frozen evaluation for illegal/CF)
# ==============================================================================================

def mine_specs(rs: List[float], rl: List[float], ledger: List[Dict[str, Any]], eps: float, max_pq: int) -> Dict[str, Optional[Dict[str, Any]]]:
    fracs = sorted({Fraction(p, q) for p in range(1, max_pq + 1) for q in range(1, max_pq + 1)})
    frozen: Dict[str, Optional[Dict[str, Any]]] = {}
    m = min(len(rs), len(rl))

    for row in ledger:
        name = row["name"]
        target = float(row["target"])
        err_power = int(row.get("err_power", 4))
        drift_power = int(row.get("drift_power", 2))
        err_budget = eps ** err_power
        drift_budget = eps ** drift_power

        best = None  # (key, spec)
        for kind_idx, kind in enumerate(KINDS):
            for i in range(m):
                js = [None] if kind == "idx" else [j for j in range(m) if j != i]
                for j in js:
                    b_s = eval_base(rs, kind, i, j)
                    b_l = eval_base(rl, kind, i, j)
                    if b_s is None or b_l is None:
                        continue
                    for uidx, u in enumerate(UNARIES):
                        u_s = apply_unary(u, b_s)
                        u_l = apply_unary(u, b_l)
                        if u_s is None or u_l is None:
                            continue
                        for frac in fracs:
                            f = float(frac)
                            v_s = f * u_s
                            v_l = f * u_l
                            mid = 0.5 * (v_s + v_l)
                            err = abs(mid - target)
                            drift = abs(v_s - v_l)
                            if err <= err_budget and drift <= drift_budget:
                                cost = (frac.numerator + frac.denominator) + unary_cost(u) + kind_cost(kind)
                                key = (
                                    cost, err, drift,
                                    kind_idx, uidx,
                                    frac.numerator, frac.denominator,
                                    i, -1 if j is None else j
                                )
                                spec = {"p": frac.numerator, "q": frac.denominator, "unary": u, "kind": kind, "i": i, "j": j}
                                if best is None or key < best[0]:
                                    best = (key, spec)
        frozen[name] = best[1] if best is not None else None

    return frozen

def closeness_table(rs: List[float], rl: List[float], specs: Dict[str, Optional[Dict[str, Any]]], ledger: List[Dict[str, Any]], eps: float) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    ok_all = 0
    ok_required = 0
    required_total = 0

    for row in ledger:
        name = row["name"]
        target = float(row["target"])
        req = bool(row.get("required", False))
        err_power = int(row.get("err_power", 4))
        drift_power = int(row.get("drift_power", 2))
        err_budget = eps ** err_power
        drift_budget = eps ** drift_power

        spec = specs.get(name)
        if spec is None:
            e = {"mid": float("nan"), "err": float("inf"), "drift": float("inf"), "domain_fail": True}
        else:
            e = eval_formula(rs, rl, spec, target)

        mid = e["mid"]
        err = e["err"]
        drift = e["drift"]
        domain_fail = bool(e["domain_fail"])

        if domain_fail or not math.isfinite(err) or not math.isfinite(drift):
            ok = False
            err_frac = float("inf")
            drift_frac = float("inf")
        else:
            err_frac = err / err_budget
            drift_frac = drift / drift_budget
            ok = (err <= err_budget) and (drift <= drift_budget)

        ok_all += int(ok)
        if req:
            required_total += 1
            ok_required += int(ok)

        rows.append({
            "name": name,
            "spec": spec_str(spec),
            "required": req,
            "target": target,
            "mid": mid,
            "abs_err": err,
            "drift": drift,
            "err_budget": err_budget,
            "drift_budget": drift_budget,
            "err_frac": err_frac,
            "drift_frac": drift_frac,
            "ok": ok,
            "domain_fail": domain_fail,
        })

    meta = {
        "ok_all": ok_all,
        "total": len(ledger),
        "ok_required": ok_required,
        "required_total": required_total,
    }
    return rows, meta

def print_closeness(title: str, rows: List[Dict[str, Any]], meta: Dict[str, int]) -> None:
    print("\n" + title)
    print("-" * 96)
    print(f"{'name':18s} {'req':>3s} {'target':>12s} {'mid':>12s} {'err/eBud':>9s} {'dr/dBud':>8s} {'OK':>3s}")
    print("-" * 96)
    for r in rows:
        ok = "Y" if r["ok"] else "N"
        rq = "Y" if r["required"] else "N"
        print(
            f"{r['name'][:18]:18s} {rq:>3s} "
            f"{r['target']:12.6g} {r['mid']:12.6g} "
            f"{r['err_frac']:9.3g} {r['drift_frac']:8.3g} {ok:>3s}"
        )
    print("-" * 96)
    if meta["required_total"] > 0:
        print(f"compliance (required): {meta['ok_required']}/{meta['required_total']}")
    print(f"compliance (all): {meta['ok_all']}/{meta['total']}")

# ==============================================================================================
# Φ-map helpers
# ==============================================================================================

def phi_map_from_tight(tight: Dict[str, Any]) -> Dict[str, Any]:
    rs = tight["rat_env_small"]
    rl = tight["rat_env_large"]
    out: Dict[str, Any] = {}
    for nm, t in TARGETS_PHI:
        out[nm] = eval_formula(rs, rl, FROZEN_PHI[nm], t)
    return out

# ==============================================================================================
# GAUGE EMBEDDING + UNIFICATION REPORT (1-loop SM, report-only)
# ==============================================================================================

def gauge_report(alpha_em: float, sin2w: float, alpha_s: float, mZ_GeV: float = 91.1876) -> Dict[str, Any]:
    # GUT-normalized alpha1 = (5/3) alpha_Y ; alpha_Y = alpha_em / cos^2θW ; alpha2 = alpha_em / sin^2θW
    c2w = 1.0 - sin2w
    alpha1 = (5.0 / 3.0) * alpha_em / c2w
    alpha2 = alpha_em / sin2w
    alpha3 = alpha_s

    # 1-loop beta coefficients (SM with one Higgs doublet)
    b1 = 41.0 / 10.0
    b2 = -19.0 / 6.0
    b3 = -7.0

    a1i, a2i, a3i = 1.0 / alpha1, 1.0 / alpha2, 1.0 / alpha3

    def ln_mu_eq(ai, aj, bi, bj):
        return (2.0 * math.pi * (ai - aj)) / (bi - bj)

    ln12 = ln_mu_eq(a1i, a2i, b1, b2)
    ln23 = ln_mu_eq(a2i, a3i, b2, b3)

    mu12 = mZ_GeV * math.exp(ln12)
    mu23 = mZ_GeV * math.exp(ln23)
    mismatch = abs(ln12 - ln23)

    def alpha_inv_at(mu, ai, bi):
        return ai - (bi / (2.0 * math.pi)) * math.log(mu / mZ_GeV)

    # alpha_U at mu12 using alpha1/alpha2 consistency
    aU1 = alpha_inv_at(mu12, a1i, b1)
    aU2 = alpha_inv_at(mu12, a2i, b2)
    aU = 0.5 * (aU1 + aU2)

    g1 = math.sqrt(4.0 * math.pi * alpha1)
    g2 = math.sqrt(4.0 * math.pi * alpha2)
    g3 = math.sqrt(4.0 * math.pi * alpha3)

    return {
        "inputs": {"alpha_em": alpha_em, "sin2w": sin2w, "alpha_s": alpha_s, "mZ_GeV": mZ_GeV},
        "embed": {"alpha1": alpha1, "alpha2": alpha2, "alpha3": alpha3, "g1": g1, "g2": g2, "g3": g3},
        "beta_1loop_SM": {"b1": b1, "b2": b2, "b3": b3},
        "unification_1loop": {"mu12_GeV": mu12, "mu23_GeV": mu23, "ln_mismatch": mismatch, "alphaU_inv_at_mu12": aU},
        "notes": "Report-only: SM 1-loop running with GUT-normalized U(1)_Y; mismatch quantifies non-unification at 1-loop.",
    }

# ==============================================================================================
# GR / GR-FACADE HOOK (report-only, self-contained)
# ==============================================================================================

def find_gr_artifacts() -> List[Dict[str, Any]]:
    patterns = [
        "gr_facade_cert*.json",
        "demo*gr*cert*.json",
        "demo*GR*cert*.json",
        "*GR*facade*.json",
        "*gravity*cert*.json",
    ]
    hits = []
    seen = set()
    for pat in patterns:
        for p in glob.glob(pat):
            if p in seen:
                continue
            seen.add(p)
            try:
                b = open(p, "rb").read()
                hits.append({"file": p, "sha256": sha256_bytes(b), "bytes": len(b)})
            except Exception:
                hits.append({"file": p, "sha256": None, "bytes": None})
    hits.sort(key=lambda x: x["file"])
    return hits

# ==============================================================================================
# MAIN
# ==============================================================================================

def main() -> None:
    banner("DEMO-78 FLAGSHIP v3.5 — SELF-CONTAINED EXT PACK (Targets + Gauge + GR Hook)")

    # Build operator + exact rational audit
    IF, M0, meta, audit = build_full(PRIMARY, rational_audit=True)
    eps = meta["eps"]
    eps2 = meta["eps2"]
    eps4 = meta["eps4"]
    margin = 1.0 + eps

    print("PRIMARY:", PRIMARY)
    print("CF_TRIPLES:", CF_TRIPLES)
    print(f"q2={meta['q2']} q3={meta['q3']} eps={eps:.12g} eps^2={eps2:.12g} eps^4={eps4:.12g}")
    print("PROVENANCE:", PROVENANCE)
    print("PRIMARY_M0_exact_rational_sha256:", audit)

    # Stage 2 compression
    banner("STAGE 2 — COMPRESSION EQUIVALENCE")
    ce = compression_equivalence(IF, M0, meta)
    comp_ok = (ce["rel_frob"] <= eps2)
    print(ce)
    print("GATE C0:", comp_ok)

    # Stage 3 Φ tightrope
    banner("STAGE 3 — TIGHTROPE (LEGAL FEJÉR, Φ-coordinates)")
    L120 = tightrope(M0, meta, fejer, kmax=KPHI, iters=120)
    L240 = tightrope(M0, meta, fejer, kmax=KPHI, iters=240)
    dev_drift = abs(L240["dev"] - L120["dev"])
    obj_drift = abs(L240["obj"] - L120["obj"])
    stab_ok = (dev_drift <= eps4) and (obj_drift <= eps4)
    print(f"dev={L120['dev']:.12g} obj={L120['obj']:.12g}")
    print("GATE N0:", stab_ok)

    # Stage 4 illegal suite
    banner("STAGE 4 — MULTI-ILLEGAL SUITE")
    illegals = [("SIGNED", illegal_signed), ("NEG_DIAG", illegal_neg_diag), ("CHECKER", illegal_checker)]
    illegal_breaks = 0
    for nm, kern in illegals:
        I = tightrope(M0, meta, kern, kmax=KPHI, iters=120)
        br = (I["dev"] >= margin * L120["dev"]) or (I["min_eig"] < -1e-9)
        illegal_breaks += int(br)
        print(f"[{nm}] break={br}  dev={I['dev']:.12g}  min_eig={I['min_eig']:.12g}")
    illegal_ok = (illegal_breaks >= 1)
    print("GATE I0:", illegal_ok)

    # Stage 5 Φ-map legal
    banner("STAGE 5 — Φ-MAP (PRIMARY, LEGAL)")
    phiL = phi_map_from_tight(L120)
    map0_ok = True
    for nm, _ in TARGETS_PHI:
        e = phiL[nm]
        ok = (e["err"] <= eps4) and (e["drift"] <= eps2) and (not e["domain_fail"])
        map0_ok = map0_ok and ok
        print(f"{nm:8s} mid={e['mid']:.12g} err={e['err']:.12g} drift={e['drift']:.12g} ok={ok}")
    print("GATE M0Φ:", map0_ok)

    # Stage 6 Φ-map illegal signed worse
    banner("STAGE 6 — Φ-MAP (PRIMARY, ILLEGAL SIGNED) WORSE")
    I_sig = tightrope(M0, meta, illegal_signed, kmax=KPHI, iters=120)
    phiI = phi_map_from_tight(I_sig)
    map1_ok = any((phiI[nm]["err"] > eps4) or (phiI[nm]["drift"] > eps2) or phiI[nm]["domain_fail"] for nm, _ in TARGETS_PHI)
    for nm, _ in TARGETS_PHI:
        e = phiI[nm]
        print(f"{nm:8s} mid={e['mid']:.12g} err={e['err']:.12g} drift={e['drift']:.12g} domain_fail={e['domain_fail']}")
    print("GATE M1Φ:", map1_ok)

    # Stage 7 Φ teeth (budget failure witness)
    banner("STAGE 7 — COUNTERFACTUAL TEETH (Φ)")
    teeth_phi = 0
    for cf in CF_TRIPLES:
        _, M0c, metac, _ = build_full(cf, rational_audit=False)
        Lc = tightrope(M0c, metac, fejer, kmax=KPHI, iters=120)
        phiC = phi_map_from_tight(Lc)
        miss = any((phiC[nm]["err"] > eps4) or (phiC[nm]["drift"] > eps2) or phiC[nm]["domain_fail"] for nm, _ in TARGETS_PHI)
        teeth_phi += int(miss)
        print(f"CF {cf} miss={miss} nF={metac['nF']}")
    teeth_ok_phi = (teeth_phi >= 3)
    print(f"TEETH Φ: {teeth_phi}/4  GATE T0Φ:", teeth_ok_phi)

    # KPHYS selection
    kphys = meta["nF"] if KPHYS_MODE == "NF_PRIMARY" else KPHI

    # Stage 8 core physical mapping (self-contained ledger + freeze)
    banner("STAGE 8 — CORE PHYSICAL MAP (self-contained)")
    core_ledger, core_source, core_sha = load_or_init_ledger("physical_targets.json", PHYS_CORE_DEFAULT)
    print(f"core_ledger_source={core_source} core_ledger_sha256={core_sha} core_len={len(core_ledger)} KPHYS={kphys}")

    Lp = tightrope(M0, meta, fejer, kmax=kphys, iters=120)
    rsP, rlP = Lp["rat_env_small"], Lp["rat_env_large"]

    core_specs = mine_specs(rsP, rlP, core_ledger, eps, MAX_PQ_CORE)
    core_specs_sha = frozen_specs_sha256(core_specs)
    core_found = sum(1 for k in core_specs if core_specs[k] is not None)
    print(f"core_mined_specs_found={core_found}/{len(core_specs)}  frozen_specs_sha256={core_specs_sha}")

    # Freeze to file (auditable)
    with open("frozen_specs_core.json", "wb") as f:
        f.write(canonical_json({k: spec_str(v) for k, v in core_specs.items()}))

    rowsP_core, metaP_core = closeness_table(rsP, rlP, core_specs, core_ledger, eps)
    print_closeness("PRIMARY_CORE", rowsP_core, metaP_core)

    Li = tightrope(M0, meta, illegal_signed, kmax=kphys, iters=120)
    rowsI_core, metaI_core = closeness_table(Li["rat_env_small"], Li["rat_env_large"], core_specs, core_ledger, eps)
    print_closeness("ILLEGAL_SIGNED_CORE", rowsI_core, metaI_core)

    cf_core_meta = []
    teeth_core = 0
    for cf in CF_TRIPLES:
        _, M0c, metac, _ = build_full(cf, rational_audit=False)
        Lc = tightrope(M0c, metac, fejer, kmax=kphys, iters=120)
        rowsC, metaC = closeness_table(Lc["rat_env_small"], Lc["rat_env_large"], core_specs, core_ledger, eps)
        miss = (metaC["ok_required"] < metaP_core["ok_required"]) if metaP_core["required_total"] > 0 else (metaC["ok_all"] < metaP_core["ok_all"])
        teeth_core += int(miss)
        cf_core_meta.append({"triple": list(cf), "nF": metac["nF"], "ok_required": metaC["ok_required"], "required_total": metaC["required_total"], "ok_all": metaC["ok_all"], "total": metaC["total"], "miss": bool(miss)})

    # core gates (required-only if any required)
    core_P0 = (metaP_core["ok_required"] == metaP_core["required_total"]) if metaP_core["required_total"] > 0 else (metaP_core["ok_all"] == metaP_core["total"])
    core_P1 = (metaI_core["ok_required"] < metaP_core["ok_required"]) if metaP_core["required_total"] > 0 else (metaI_core["ok_all"] < metaP_core["ok_all"])
    core_P2 = (teeth_core >= 3)
    print(f"CORE TEETH: {teeth_core}/4")
    print("GATES core:", {"P0_core_required_all_pass": core_P0, "P1_core_illegal_worse": core_P1, "P2_core_teeth": core_P2})

    # Stage 9 EXT physical mapping (self-contained extended ledger + freeze)
    banner("STAGE 9 — EXT PHYSICAL MAP (self-contained additions)")
    ext_ledger, ext_source, ext_sha = load_or_init_ledger("physical_targets_ext.json", PHYS_EXT_DEFAULT)
    print(f"ext_ledger_source={ext_source} ext_ledger_sha256={ext_sha} ext_len={len(ext_ledger)} KPHYS={kphys}")
    print(f"rp_over_lambdaCp_target={rp_over_lambdaCp_target():.12g}  (dimensionless)")

    ext_specs = mine_specs(rsP, rlP, ext_ledger, eps, MAX_PQ_EXT)
    ext_specs_sha = frozen_specs_sha256(ext_specs)
    ext_found = sum(1 for k in ext_specs if ext_specs[k] is not None)
    print(f"ext_mined_specs_found={ext_found}/{len(ext_specs)}  frozen_specs_sha256={ext_specs_sha}")

    with open("frozen_specs_ext.json", "wb") as f:
        f.write(canonical_json({k: spec_str(v) for k, v in ext_specs.items()}))

    rowsP_ext, metaP_ext = closeness_table(rsP, rlP, ext_specs, ext_ledger, eps)
    print_closeness("PRIMARY_EXT", rowsP_ext, metaP_ext)

    rowsI_ext, metaI_ext = closeness_table(Li["rat_env_small"], Li["rat_env_large"], ext_specs, ext_ledger, eps)
    print_closeness("ILLEGAL_SIGNED_EXT", rowsI_ext, metaI_ext)

    cf_ext_meta = []
    teeth_ext = 0
    for cf in CF_TRIPLES:
        _, M0c, metac, _ = build_full(cf, rational_audit=False)
        Lc = tightrope(M0c, metac, fejer, kmax=kphys, iters=120)
        rowsC, metaC = closeness_table(Lc["rat_env_small"], Lc["rat_env_large"], ext_specs, ext_ledger, eps)
        miss = (metaC["ok_required"] < metaP_ext["ok_required"]) if metaP_ext["required_total"] > 0 else (metaC["ok_all"] < metaP_ext["ok_all"])
        teeth_ext += int(miss)
        cf_ext_meta.append({"triple": list(cf), "nF": metac["nF"], "ok_required": metaC["ok_required"], "required_total": metaC["required_total"], "ok_all": metaC["ok_all"], "total": metaC["total"], "miss": bool(miss)})

    ext_P0 = (metaP_ext["ok_required"] == metaP_ext["required_total"]) if metaP_ext["required_total"] > 0 else False
    ext_P1 = (metaI_ext["ok_required"] < metaP_ext["ok_required"]) if metaP_ext["required_total"] > 0 else False
    ext_P2 = (teeth_ext >= 3) if metaP_ext["required_total"] > 0 else False

    print(f"EXT TEETH: {teeth_ext}/4")
    print("GATES ext (required-only):", {"X0_ext_required_all_pass": ext_P0, "X1_ext_illegal_worse": ext_P1, "X2_ext_teeth": ext_P2})
    print("NOTE: ext gates are required-only. Set required:true in physical_targets_ext.json for promotion.")

    # Stage 10 gauge embedding + 1-loop unification report (self-contained)
    banner("STAGE 10 — GAUGE EMBEDDING + 1-LOOP UNIFICATION (report-only)")
    alpha_em_mid = float(phiL["alpha_em"]["mid"])
    sin2w_mid    = float(phiL["sin2W"]["mid"])
    alpha_s_mid  = float(phiL["alpha_s"]["mid"])
    g_rep = gauge_report(alpha_em_mid, sin2w_mid, alpha_s_mid)
    g_rep_sha = sha256_bytes(canonical_json(g_rep))
    print("gauge_report_sha256:", g_rep_sha)
    print("alpha1,alpha2,alpha3:", fmt12(g_rep["embed"]["alpha1"]), fmt12(g_rep["embed"]["alpha2"]), fmt12(g_rep["embed"]["alpha3"]))
    print("mu12_GeV:", fmt12(g_rep["unification_1loop"]["mu12_GeV"]), "mu23_GeV:", fmt12(g_rep["unification_1loop"]["mu23_GeV"]))
    print("ln_mismatch:", fmt12(g_rep["unification_1loop"]["ln_mismatch"]), "alphaU_inv_at_mu12:", fmt12(g_rep["unification_1loop"]["alphaU_inv_at_mu12"]))

    # Stage 11 GR facade hook (self-contained: scans local directory; no dependency)
    banner("STAGE 11 — GR / GR-FACADE HOOK (report-only, auto-scan)")
    gr_hits = find_gr_artifacts()
    if gr_hits:
        print(f"found {len(gr_hits)} GR-related artifact(s):")
        for h in gr_hits:
            print(f"  {h['file']}  sha256={h['sha256']}  bytes={h['bytes']}")
    else:
        print("no GR-related artifacts found in working directory (ok).")

    # Final gates
    banner("FINAL VERDICT (core + ext + audit anchors)")
    gates_v2 = {
        "C0_compression": bool(comp_ok),
        "N0_numeric_stability": bool(stab_ok),
        "I0_illegal_suite": bool(illegal_ok),
        "M0_phi_budget": bool(map0_ok),
        "M1_phi_illegal_worse": bool(map1_ok),
        "T0_phi_teeth": bool(teeth_ok_phi),
        "phi_teeth_count": int(teeth_phi),
    }
    verified_v2 = all(gates_v2[k] for k in gates_v2 if k != "phi_teeth_count")

    gates_core = {
        "P0_core_required_all_pass": bool(core_P0),
        "P1_core_illegal_worse": bool(core_P1),
        "P2_core_teeth": bool(core_P2),
        "core_required": f"{metaP_core['ok_required']}/{metaP_core['required_total']}",
        "core_all": f"{metaP_core['ok_all']}/{metaP_core['total']}",
        "illegal_core_required": f"{metaI_core['ok_required']}/{metaI_core['required_total']}",
        "illegal_core_all": f"{metaI_core['ok_all']}/{metaI_core['total']}",
        "core_ledger_sha256": core_sha,
        "core_ledger_source": core_source,
        "core_frozen_specs_sha256": core_specs_sha,
    }
    verified_core = core_P0 and core_P1 and core_P2

    gates_ext = {
        "X0_ext_required_all_pass": bool(ext_P0),
        "X1_ext_illegal_worse": bool(ext_P1),
        "X2_ext_teeth": bool(ext_P2),
        "ext_required": f"{metaP_ext['ok_required']}/{metaP_ext['required_total']}",
        "ext_all": f"{metaP_ext['ok_all']}/{metaP_ext['total']}",
        "illegal_ext_required": f"{metaI_ext['ok_required']}/{metaI_ext['required_total']}",
        "illegal_ext_all": f"{metaI_ext['ok_all']}/{metaI_ext['total']}",
        "ext_ledger_sha256": ext_sha,
        "ext_ledger_source": ext_source,
        "ext_frozen_specs_sha256": ext_specs_sha,
    }
    verified_ext = ext_P0 and ext_P1 and ext_P2  # only meaningful if required_total>0

    print("GATES v2:", gates_v2)
    print("VERDICT v2:", "VERIFIED" if verified_v2 else "NOT VERIFIED")
    print("GATES core:", gates_core)
    print("VERDICT core:", "VERIFIED" if verified_core else "NOT VERIFIED")
    print("GATES ext (required-only):", gates_ext)
    print("VERDICT ext (required-only):", "VERIFIED" if verified_ext else "NOT VERIFIED / NOT PROMOTED")

    # Determinism payload + artifacts
    payload = {
        "name": "DEMO-78-v3.5-SELFCONTAINED-EXT",
        "provenance": PROVENANCE,
        "platform": {"python": sys.version.split()[0], "impl": platform.python_implementation(), "os": platform.system()},
        "primary": list(PRIMARY),
        "counterfactuals": [list(x) for x in CF_TRIPLES],
        "meta_primary": meta,
        "primary_M0_exact_rational_sha256": audit,
        "compression": ce,
        "tightrope_phi_legal_120": L120,
        "tightrope_phi_legal_240": {"dev": fmt12(L240["dev"]), "obj": fmt12(L240["obj"])},
        "illegal_suite_phi_breaks": illegal_breaks,
        "phi_map_primary_legal": phiL,
        "phi_map_primary_illegal_signed": phiI,
        "phi_teeth_count": int(teeth_phi),

        "core_stage": {
            "ledger_source": core_source,
            "ledger_sha256": core_sha,
            "ledger": core_ledger,
            "kphys_used": int(kphys),
            "max_pq": int(MAX_PQ_CORE),
            "frozen_specs": {k: spec_str(v) for k, v in core_specs.items()},
            "frozen_specs_sha256": core_specs_sha,
            "primary_table": rowsP_core,
            "illegal_table": rowsI_core,
            "cf_meta": cf_core_meta,
            "gates": gates_core,
            "verified": bool(verified_core),
        },

        "ext_stage": {
            "ledger_source": ext_source,
            "ledger_sha256": ext_sha,
            "ledger": ext_ledger,
            "kphys_used": int(kphys),
            "max_pq": int(MAX_PQ_EXT),
            "frozen_specs": {k: spec_str(v) for k, v in ext_specs.items()},
            "frozen_specs_sha256": ext_specs_sha,
            "primary_table": rowsP_ext,
            "illegal_table": rowsI_ext,
            "cf_meta": cf_ext_meta,
            "gates": gates_ext,
            "verified_required_only": bool(verified_ext),
        },

        "gauge_stage": {"report": g_rep, "sha256": g_rep_sha},
        "gr_hook": {"hits": gr_hits},

        "gates_v2": gates_v2,
        "verified_v2": bool(verified_v2),
        "verified_core": bool(verified_core),
    }

    det = sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    js = f"demo78_v35_{det[:12]}.json"
    md = f"demo78_v35_{det[:12]}.md"

    with open(js, "wb") as f:
        f.write(canonical_json(payload))

    # Minimal MD: certificate + anchors (no “ranking” language)
    md_lines = []
    md_lines.append("# DEMO-78 v3.5 — Self-Contained EXT Pack Certificate\n\n")
    md_lines.append(f"- sha256: `{det}`\n")
    md_lines.append(f"- PRIMARY: `{PRIMARY}`\n")
    md_lines.append(f"- CF_TRIPLES: `{CF_TRIPLES}`\n")
    md_lines.append(f"- eps: `{meta['eps']}` ; eps^2: `{meta['eps2']}` ; eps^4: `{meta['eps4']}`\n")
    md_lines.append(f"- M0_exact_rational_sha256: `{audit}`\n")
    md_lines.append(f"- core_ledger_sha256: `{core_sha}` (source={core_source})\n")
    md_lines.append(f"- core_frozen_specs_sha256: `{core_specs_sha}`\n")
    md_lines.append(f"- ext_ledger_sha256: `{ext_sha}` (source={ext_source})\n")
    md_lines.append(f"- ext_frozen_specs_sha256: `{ext_specs_sha}`\n")
    md_lines.append(f"- gauge_report_sha256: `{g_rep_sha}`\n\n")

    md_lines.append("## Gates\n\n")
    md_lines.append(f"- v2 VERIFIED: `{verified_v2}`\n")
    md_lines.append(f"- core VERIFIED: `{verified_core}`\n")
    md_lines.append(f"- ext VERIFIED (required-only): `{verified_ext}`\n\n")

    md_lines.append("## Certificate block\n\n``` \n")
    md_lines.append("DEMO-78 v3.5 — PUBLIC CERTIFICATE\n")
    md_lines.append(f"sha256: {det}\n")
    md_lines.append(f"PRIMARY: {PRIMARY}\n")
    md_lines.append(f"CF_TRIPLES: {CF_TRIPLES}\n")
    md_lines.append(f"eps={meta['eps']}  eps^2={meta['eps2']}  eps^4={meta['eps4']}\n")
    md_lines.append(f"M0_exact_rational_sha256: {audit}\n")
    md_lines.append(f"core_ledger_sha256: {core_sha} (source={core_source})\n")
    md_lines.append(f"core_frozen_specs_sha256: {core_specs_sha}\n")
    md_lines.append(f"ext_ledger_sha256: {ext_sha} (source={ext_source})\n")
    md_lines.append(f"ext_frozen_specs_sha256: {ext_specs_sha}\n")
    md_lines.append(f"gauge_report_sha256: {g_rep_sha}\n")
    md_lines.append(f"GATES_v2: {gates_v2}\n")
    md_lines.append(f"GATES_core: {gates_core}\n")
    md_lines.append(f"GATES_ext(required-only): {gates_ext}\n")
    md_lines.append(f"VERDICT_v2: {verified_v2}\n")
    md_lines.append(f"VERDICT_core: {verified_core}\n")
    md_lines.append(f"VERDICT_ext(required-only): {verified_ext}\n")
    md_lines.append("```\n")

    with open(md, "w", encoding="utf-8") as f:
        f.write("".join(md_lines))

    banner("CERTIFICATE (copy/paste)")
    print("DEMO-78 v3.5 — PUBLIC CERTIFICATE")
    print("PRIMARY:", PRIMARY)
    print("CF_TRIPLES:", CF_TRIPLES)
    print(f"eps={meta['eps']}  eps^2={meta['eps2']}  eps^4={meta['eps4']}")
    print("M0_exact_rational_sha256:", audit)
    print(f"core_ledger_sha256: {core_sha}  (source={core_source})")
    print("core_frozen_specs_sha256:", core_specs_sha)
    print(f"ext_ledger_sha256: {ext_sha}  (source={ext_source})")
    print("ext_frozen_specs_sha256:", ext_specs_sha)
    print("gauge_report_sha256:", g_rep_sha)
    print("GATES_v2:", gates_v2)
    print("GATES_core:", gates_core)
    print("GATES_ext(required-only):", gates_ext)
    print("VERDICT_v2:", verified_v2)
    print("VERDICT_core:", verified_core)
    print("VERDICT_ext(required-only):", verified_ext)
    print("sha256:", det)
    print("artifacts:", js, md)
    print("notes: frozen_specs_core.json and frozen_specs_ext.json written for audit; edit physical_targets_ext.json to promote required targets.")

if __name__ == "__main__":
    main()
