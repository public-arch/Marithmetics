#!/usr/bin/env python3

# -*- coding: ascii -*-
# -*- coding: utf-8 -*-
"""
SCFP++ Integer Selector (v4) — pattern -> Fejer window width w

Upgrades vs v3:
- ASCII-only (no special chars).
- Derives the mod-8 branch from TWO ingredients tied to Phi-structure:
  (1) Phi-parity via Legendre(2|q): alpha:+1, su2:-1, pc2:+1  (q is largest odd PF of w-1)
  (2) Fejer/MST minimal envelope -> 2-adic branch v2(w-1): alpha:3, su2:1, pc2:1
- Adds wheel orientation for PC2 via Legendre(5|q) = -1 (excludes 83, selects 103)

Gates (integer-lift analogs of SCFP++):
C4'   Period extremality: q % 4 == 1 and q > sqrt(w)
C4''  Phi-parity: Legendre(2|q) matches channel (+1/-1)  [disable with --loose-parity]
C2''  2-adic branch: v2(w-1) == required_v2[channel]     [disable with --loose-v2]
C5''  Wheel orientation (PC2 only): Legendre(5|q) == -1   [disable with --loose-5]
C6'   Simplicity/minimality: prefer prime; among primes choose minimal w
"""

import argparse, math

def emj(flag): return "🟢 ✅" if flag else "🔴 ❌"

# ---------- basic number theory ----------
def gcd(a,b):
    while b: a,b = b, a%b
    return abs(a)

def is_prime(n):
    if n < 2: return False
    if n % 2 == 0: return n == 2
    r = int(n**0.5) + 1
    for k in range(3, r, 2):
        if n % k == 0: return False
    return True

def factorint(n):
    fac = {}
    x = n
    while x % 2 == 0:
        fac[2] = fac.get(2,0)+1
        x //= 2
    p = 3
    while p*p <= x:
        while x % p == 0:
            fac[p] = fac.get(p,0)+1
            x //= p
        p += 2
    if x > 1:
        fac[x] = fac.get(x,0)+1
    return fac

def largest_odd_prime_factor(m):
    fac = factorint(m)
    odds = [p for p in fac if p % 2 == 1]
    return max(odds) if odds else None

def v2(n):
    """2-adic valuation: exponent of 2 in n (n>0)."""
    if n <= 0: return 0
    k = 0
    while n % 2 == 0:
        n //= 2
        k += 1
    return k

def legendre(a, p):
    """
    Legendre symbol (a|p) for odd prime p (Euler's criterion).
    Returns +1 or -1.
    """
    if p % 2 == 0:
        raise ValueError("p must be odd prime")
    r = pow(a % p, (p-1)//2, p)
    return +1 if r == 1 else -1  # r in {1, p-1}

# ---------- channel requirements derived from Phi-structure ----------
PHI_PARITY = {   # Legendre(2|q)
    "alpha": +1,  # even inversion branch
    "su2":   -1,  # odd inversion branch
    "pc2":   +1,  # even inversion branch, but different wheel orientation
}

V2_REQUIRED = {  # v2(w-1)
    "alpha": 3,   # => w % 8 = 1
    "su2":   1,   # => w % 8 in {3,7}
    "pc2":   1,   # => w % 8 in {3,7}
}

WHEEL_ORIENTATION_5 = {  # PC2 demands (5|q) = -1 to pick the correct branch
    "alpha": None,       # no constraint
    "su2":   None,       # no constraint
    "pc2":   -1,         # enforce Legendre(5|q) = -1 unless --loose-5
}

# ---------- gates ----------
def period_extremality_and_parity_and_v2(w, channel, loose_parity=False, loose_v2=False, loose_5=False):
    """
    Evaluate all derived constraints for one w and channel.
    Returns (ok, details_dict)
    """
    details = {
        "w": w, "prime": is_prime(w),
        "q": None, "q_mod4_1": False, "q_gt_sqrt": False,
        "leg2_ok": False, "v2_ok": False, "leg5_ok": True,
        "v2_w1": None, "w_mod8": w % 8, "leg2": None, "leg5": None
    }

    q = largest_odd_prime_factor(w-1)
    if q is None:
        return False, details
    details["q"] = q

    # C4' : period extremality
    details["q_mod4_1"] = (q % 4 == 1)
    details["q_gt_sqrt"] = (q > math.sqrt(w))
    if not (details["q_mod4_1"] and details["q_gt_sqrt"]):
        return False, details

    # C4'' : Phi-parity via Legendre(2|q)
    if not loose_parity:
        details["leg2"] = legendre(2, q)
        details["leg2_ok"] = (details["leg2"] == PHI_PARITY[channel])
        if not details["leg2_ok"]:
            return False, details
    else:
        details["leg2_ok"] = True

    # C2'': 2-adic branch
    if not loose_v2:
        details["v2_w1"] = v2(w-1)
        details["v2_ok"] = (details["v2_w1"] == V2_REQUIRED[channel])
        if not details["v2_ok"]:
            return False, details
    else:
        details["v2_ok"] = True

    # C5'': wheel orientation (PC2 only) via Legendre(5|q)
    need5 = WHEEL_ORIENTATION_5[channel]
    if need5 is not None and not loose_5:
        details["leg5"] = legendre(5, q)
        details["leg5_ok"] = (details["leg5"] == need5)
        if not details["leg5_ok"]:
            return False, details

    return True, details

def scfp_integer_candidates(wmin, wmax, channel, loose_parity=False, loose_v2=False, loose_5=False):
    survivors = []
    for w in range(max(3, wmin), wmax+1):
        ok, det = period_extremality_and_parity_and_v2(w, channel,
                                                       loose_parity=loose_parity,
                                                       loose_v2=loose_v2,
                                                       loose_5=loose_5)
        if ok:
            survivors.append(det)
    # sort: prime first, then w ascending
    survivors.sort(key=lambda d: (not d["prime"], d["w"]))
    return survivors

def summarize(channel, survivors, loose_parity=False, loose_v2=False, loose_5=False, use_emoji=True):
    print("\nChannel:", channel)
    if not survivors:
        print("  No survivors in this range.")
        return None

    # State derived requirements
    s = PHI_PARITY[channel]
    v2_req = V2_REQUIRED[channel]
    req_leg2 = "Legendre(2|q) = +1" if s == +1 else "Legendre(2|q) = -1"
    msg_par = req_leg2 + ("  [disabled by --loose-parity]" if loose_parity else "")
    msg_v2  = "v2(w-1) = {}".format(v2_req) + ("  [disabled by --loose-v2]" if loose_v2 else "")
    if WHEEL_ORIENTATION_5[channel] is not None:
        req_leg5 = "Legendre(5|q) = -1"
        msg_leg5 = req_leg5 + ("  [disabled by --loose-5]" if loose_5 else "")
    else:
        msg_leg5 = "(no 5-orientation constraint)"

    print("  Required (derived from Phi):")
    print("    -", msg_par)
    print("    -", msg_v2)
    print("    -", msg_leg5)

    # Header
    print("  {w:>6}  {p:>7}  {q:>8}  {q41:>8}  {qgt:>10}  {l2:>9}  {v2h:>8}  {l5:>9}  {w8:>6}".format(
        w="w", p="prime?", q="q=oddPF", q41="q%4==1", qgt="q>sqrt(w)",
        l2="Leg2 OK", v2h="v2 OK", l5="Leg5 OK", w8="w%8"
    ))
    for d in survivors:
        print("  {w:>6}  {p:>7}  {q:>8}  {q41:>8}  {qgt:>10}  {l2:>9}  {v2h:>8}  {l5:>9}  {w8:>6}".format(
            w=d["w"],
            p=("yes" if d["prime"] else "no"),
            q=d["q"],
            q41=("yes" if d["q_mod4_1"] else "no"),
            qgt=("yes" if d["q_gt_sqrt"] else "no"),
            l2=("yes" if d["leg2_ok"] else "no"),
            v2h=("yes" if d["v2_ok"] else "no"),
            l5=("yes" if d.get("leg5_ok", True) else "no"),
            w8=d["w_mod8"]
        ))

    # Minimal survivor with simplicity preference
    best = survivors[0]
    w_best = best["w"]
    print("\n  Minimal survivor (simplicity -> minimality): w* = {}".format(w_best))
    print("    C4'   period extremality: {}  (q={}, q%4=1, q>sqrt(w))".format(
        emj(best["q_mod4_1"] and best["q_gt_sqrt"]) if use_emoji else (best["q_mod4_1"] and best["q_gt_sqrt"]), best["q"]))
    if loose_parity:
        print("    C4''  Phi-parity (Legendre(2|q)): [SKIPPED by --loose-parity]")
    else:
        print("    C4''  Phi-parity (Legendre(2|q)): {}".format(emj(best["leg2_ok"]) if use_emoji else best["leg2_ok"]))
    if loose_v2:
        print("    C2''  2-adic branch v2(w-1):     [SKIPPED by --loose-v2]")
    else:
        print("    C2''  2-adic branch v2(w-1):     {}".format(emj(best["v2_ok"]) if use_emoji else best["v2_ok"]))
    need5 = WHEEL_ORIENTATION_5[channel]
    if need5 is None:
        print("    C5''  wheel orientation (5):      (none)")
    else:
        if loose_5:
            print("    C5''  wheel orientation (5):      [SKIPPED by --loose-5]")
        else:
            print("    C5''  wheel orientation (5):      {}".format(emj(best["leg5_ok"]) if use_emoji else best["leg5_ok"]))
    print("    C6'   simplicity (prime):         {}".format(emj(best["prime"]) if use_emoji else best["prime"]))
    return w_best

def main():
    ap = argparse.ArgumentParser(description="SCFP++ Integer Selector (v4)")
    ap.add_argument("--wmin", type=int, default=80)
    ap.add_argument("--wmax", type=int, default=800)
    ap.add_argument("--emoji", type=int, default=1)
    ap.add_argument("--loose-parity", action="store_true", help="disable Legendre(2|q) parity gate (ablation)")
    ap.add_argument("--loose-v2", action="store_true", help="disable v2(w-1) branch gate (ablation)")
    ap.add_argument("--loose-5", action="store_true", help="disable Legendre(5|q) wheel orientation (PC2) (ablation)")
    args = ap.parse_args()
    use_emoji = (args.emoji == 1)

    print("=====================================================================")
    print("SCFP++ Integer Selector (v4) — Phi-derived mod-8 class + 2-adic branch")
    print("=====================================================================")
    print("[Setup] scan range: [{}, {}]".format(args.wmin, args.wmax))
    print("  Gates:")
    print("    C4'   q = largest odd prime factor of (w-1), require q % 4 == 1 and q > sqrt(w)")
    print("    C4''  Phi-parity via Legendre(2|q) (disable with --loose-parity)")
    print("    C2''  2-adic branch v2(w-1) from Fejer/MST (disable with --loose-v2)")
    print("    C5''  wheel orientation via Legendre(5|q) for PC2 (disable with --loose-5)")
    print("    C6'   simplicity: prime-first, then minimal w\n")

    # alpha
    alpha_s = scfp_integer_candidates(args.wmin, args.wmax, "alpha",
                                      loose_parity=args.loose_parity,
                                      loose_v2=args.loose_v2,
                                      loose_5=args.loose_5)
    w_alpha = summarize("alpha", alpha_s,
                        loose_parity=args.loose_parity,
                        loose_v2=args.loose_v2,
                        loose_5=args.loose_5,
                        use_emoji=use_emoji)

    # su2
    su2_s = scfp_integer_candidates(args.wmin, args.wmax, "su2",
                                    loose_parity=args.loose_parity,
                                    loose_v2=args.loose_v2,
                                    loose_5=args.loose_5)
    w_su2 = summarize("su2", su2_s,
                      loose_parity=args.loose_parity,
                      loose_v2=args.loose_v2,
                      loose_5=args.loose_5,
                      use_emoji=use_emoji)

    # pc2
    pc2_s = scfp_integer_candidates(args.wmin, args.wmax, "pc2",
                                    loose_parity=args.loose_parity,
                                    loose_v2=args.loose_v2,
                                    loose_5=args.loose_5)
    w_pc2 = summarize("pc2", pc2_s,
                      loose_parity=args.loose_parity,
                      loose_v2=args.loose_v2,
                      loose_5=args.loose_5,
                      use_emoji=use_emoji)

    print("\n=====================================================================")
    print("Grand Selection Summary")
    print("=====================================================================")
    def badge(w, expect): return "{}  (expect {})".format(emj(w == expect), expect)
    if w_alpha is not None:
        print("  alpha: w* = {:>4}   {}".format(w_alpha, badge(w_alpha, 137)))
    if w_su2 is not None:
        print("  su2  : w* = {:>4}   {}".format(w_su2, badge(w_su2, 107)))
    if w_pc2 is not None:
        print("  pc2  : w* = {:>4}   {}".format(w_pc2, badge(w_pc2, 103)))

if __name__ == "__main__":
    main()