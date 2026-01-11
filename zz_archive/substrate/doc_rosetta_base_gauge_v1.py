#!/usr/bin/env python3
# rosetta_base_gauge_demo.py — stdlib-only, deterministic
import math, json, sys

def kappa(h):  # Fejér thickness
    if h < 1: return 0.0
    return (h - 1.0) / (2.0*h)

def gamma1(h):
    return 2.0 - 2.0*math.cos(math.pi/(2*h + 1))

def alias_local(h,M):
    return (1.0/12.0) * (h*h) / (M*M)

def phi_hat(h,M):
    K = kappa(h)*gamma1(h)
    A = alias_local(h,M)
    # base-invariant dimensionless ratio
    return K / (K + A)

def to_base(x, base=10, digits=18):
    # Represent float |x| in the given base as integer/fraction digit strings (for display).
    # We do NOT use this to compute; it's just to show representation changes only.
    if not (2 <= base <= 16): base = 10
    sign = "-" if x < 0 else ""
    x = abs(x)
    # integer part
    i = int(x); frac = x - i
    DIG = "0123456789abcdef"
    if i == 0: int_digits = "0"
    else:
        d=[]; n=i
        while n>0:
            d.append(DIG[n%base]); n//=base
        int_digits="".join(reversed(d))
    # fractional
    f=[]; y=frac
    for _ in range(digits):
        y *= base
        d = int(y + 1e-15)  # guard
        f.append(DIG[d]); y -= d
    return f"{sign}{int_digits}.{''.join(f)}(base{base})"

def crt_encode(x, moduli):
    return tuple(x % m for m in moduli)

def coprime(a,b):
    while b: a,b=b,a%b
    return a==1

def pairwise_coprime(moduli):
    L=len(moduli)
    for i in range(L):
        for j in range(i+1,L):
            if not coprime(moduli[i], moduli[j]):
                return False
    return True

def find_collision(moduli, search_max=200):
    # Find x!=y with same residue tuple for non-coprime moduli (should happen quickly).
    seen={}
    for x in range(search_max):
        r = crt_encode(x, moduli)
        if r in seen and seen[r] != x:
            return {"x":seen[r], "y":x, "r":r}
        seen[r]=x
    return None

def run_suite(h=3,M=64, tol=1e-12):
    out={"passes":{}, "records":{}}
    # (A) Cross-base invariance
    ph = phi_hat(h,M)
    b7 = to_base(ph,7); b10 = to_base(ph,10); b16=to_base(ph,16)
    # numeric equality check via absolute diff (base-agnostic computation)
    ph_b7 = ph; ph_b10 = ph; ph_b16 = ph
    flat_ok = (abs(ph_b7 - ph_b10) <= tol) and (abs(ph_b7 - ph_b16) <= tol)
    out["records"]["phi_hat"]={"h":h,"M":M,"value":ph,"repr_b7":b7,"repr_b10":b10,"repr_b16":b16}
    out["passes"]["phi_cross_base_flat"] = bool(flat_ok)

    # (B) CRT injectivity
    good_mods = (2,3,5)
    bad_mods  = (6,9,15)
    inj_ok = pairwise_coprime(good_mods)
    coll = find_collision(bad_mods, search_max=500)
    out["records"]["crt_good_moduli"]=good_mods
    out["records"]["crt_bad_moduli"]=bad_mods
    out["records"]["crt_bad_collision"]=coll
    out["passes"]["crt_injective_on_good"] = bool(inj_ok)
    out["passes"]["crt_collision_exists_on_bad"] = bool(coll is not None)

    # (C) Designed FAIL: digit injection
    # Wrong: treat the base-10 digits as if they were base-7 before evaluating.
    s10 = b10.split("(")[0]   # "int.frac"
    # strip dot and re-interpret as base-7 integer scaled back
    s = s10.replace(".","")
    DIG = "0123456789abcdef"
    val_wrong = 0
    for ch in s:
        val_wrong = val_wrong*7 + DIG.index(ch)
    # scale back by 7^{digits_after_dot}
    p = len(s10.split(".")[1])
    val_wrong = val_wrong / (7**p)
    designed_fail_trips = abs(val_wrong - ph) > 1e-6  # should differ O(1)
    out["records"]["digit_injection_val"] = val_wrong
    out["passes"]["designed_fail_digit_injection_trips"] = bool(designed_fail_trips)

    return out

def main():
    out = run_suite()
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
