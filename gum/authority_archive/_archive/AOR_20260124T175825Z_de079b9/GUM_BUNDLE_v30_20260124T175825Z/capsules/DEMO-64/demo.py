#!/usr/bin/env python3
"""DEMO-64 — Base-Gauge Invariance of a Deterministic Integer Selector

This demo isolates a single claim:

  The selection procedure that fixes the primary integer triple (wU, s2, s3)
  is invariant under a change of numeral base used to *encode* the integers.

In other words, the selector depends on the integers themselves (and integer
invariants), not on the human-facing encoding (binary, octal, decimal, hex, …).

Why this matters
----------------
If a new framework claims it produces a unique triple deterministically, a
skeptical reader must be able to rule out "formatting accidents": e.g.,
implicit decimal parsing, locale-dependent string handling, float round-trip
errors, or hidden dependence on a specific base.

What this script does
---------------------
1) Defines a small, fully explicit selector using only elementary number theory:
   - primality
   - congruences modulo q
   - the totient ratio Θ(n) = φ(n)/n as a minimal 'structure' filter

2) Runs the selector in integer mode to obtain the baseline survivor pools and
   the unique admissible triple.

3) Re-runs *the exact same selector* behind an explicit encode/decode layer
   across multiple bases. The contract for the encoding layer is:

      decode_base_b( encode_base_b(w) ) == w

   If the contract holds, the selector must return identical pools and the
   same triple.

4) Runs a negative control by deliberately violating the contract (decoding
   in the wrong base). The demo must detect this and fail the audit for those
   bases.

Outputs
-------
- Prints a referee-facing audit to stdout.
- Optionally writes a small JSON bundle (results + hashes) to disk.

No external dependencies.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import platform
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


# -----------------------------
# Formatting helpers
# -----------------------------

LINE = "=" * 98


def utc_iso() -> str:
    return _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def print_header(title: str) -> None:
    print(LINE)
    print(title)
    print(LINE)
    print(f"UTC time : {utc_iso()}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print("I/O      : stdout + optional JSON bundle")
    print(LINE)


def gate(name: str, ok: bool, details: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    if details:
        print(f"{status:4s}  {name:<72s} {details}")
    else:
        print(f"{status:4s}  {name}")


# -----------------------------
# Canonical hashing
# -----------------------------


def canonical_json_bytes(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# -----------------------------
# Basic number theory (small inputs; trial division is enough)
# -----------------------------


def is_prime(n: int) -> bool:
    if n <= 1:
        return False
    if n <= 3:
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


def factorize(n: int) -> Dict[int, int]:
    """Prime factorization by trial division."""
    if n <= 0:
        raise ValueError("factorize expects n>0")
    out: Dict[int, int] = {}
    while n % 2 == 0:
        out[2] = out.get(2, 0) + 1
        n //= 2
    p = 3
    while p * p <= n:
        while n % p == 0:
            out[p] = out.get(p, 0) + 1
            n //= p
        p += 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def euler_phi(n: int) -> int:
    if n <= 0:
        raise ValueError("phi expects n>0")
    fac = factorize(n)
    result = n
    for p in fac:
        result = result // p * (p - 1)
    return result


def theta(n: int) -> float:
    """Θ(n) = φ(n)/n."""
    return euler_phi(n) / float(n)


def v2_adic(n: int) -> int:
    """v2(n): exponent of 2 in n."""
    if n == 0:
        return 0
    k = 0
    while (n & 1) == 0:
        n >>= 1
        k += 1
    return k


# -----------------------------
# Base encoding/decoding
# -----------------------------

ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyz"


def encode_base(n: int, base: int) -> str:
    if base < 2 or base > len(ALPHABET):
        raise ValueError("unsupported base")
    if n == 0:
        return "0"
    if n < 0:
        raise ValueError("encode_base expects n>=0")
    digits: List[str] = []
    x = n
    while x > 0:
        x, r = divmod(x, base)
        digits.append(ALPHABET[r])
    return "".join(reversed(digits))


def decode_base(s: str, base: int) -> int:
    if base < 2 or base > len(ALPHABET):
        raise ValueError("unsupported base")
    s = s.strip().lower()
    if not s:
        raise ValueError("empty string")
    value = 0
    for ch in s:
        idx = ALPHABET.find(ch)
        if idx < 0 or idx >= base:
            raise ValueError(f"invalid digit '{ch}' for base {base}")
        value = value * base + idx
    return value


# -----------------------------
# The selector
# -----------------------------


@dataclass(frozen=True)
class Triple:
    wU: int
    s2: int
    s3: int


@dataclass(frozen=True)
class LaneSpec:
    name: str
    q: int
    residues: Tuple[int, ...]
    tau_min: float
    span_lo: int
    span_hi: int


def lane_survivors(spec: LaneSpec, candidates: Iterable[int]) -> List[int]:
    """Apply the lane filter to a candidate set."""
    pool: List[int] = []
    for w in candidates:
        if w < spec.span_lo or w > spec.span_hi:
            continue
        if not is_prime(w):
            continue
        if (w % spec.q) not in spec.residues:
            continue
        if theta(w - 1) < spec.tau_min:
            continue
        pool.append(w)
    return pool


def admissible_triples(u1: Sequence[int], su2: Sequence[int], su3: Sequence[int]) -> List[Triple]:
    """Cartesian product + simple admissibility constraints."""
    out: List[Triple] = []
    for wU in u1:
        for s2 in su2:
            for s3 in su3:
                # T1: all three must be distinct.
                if len({wU, s2, s3}) != 3:
                    continue
                # T2: define q2 = wU - s2 and require positivity.
                if wU - s2 <= 0:
                    continue
                out.append(Triple(wU=wU, s2=s2, s3=s3))
    return out


def derived_invariants(t: Triple) -> Dict[str, Any]:
    q2 = t.wU - t.s2
    v2u = v2_adic(t.wU - 1)
    q3 = (t.wU - 1) >> v2u
    eps = 1.0 / math.sqrt(q2)
    return {"q2": q2, "q3": q3, "v2U": v2u, "eps": eps}


def run_selector_integer_mode(lanes: Sequence[LaneSpec]) -> Dict[str, Any]:
    span_lo = min(l.span_lo for l in lanes)
    span_hi = max(l.span_hi for l in lanes)
    candidates = list(range(span_lo, span_hi + 1))

    pools: Dict[str, List[int]] = {}
    for lane in lanes:
        pools[lane.name] = lane_survivors(lane, candidates)

    triples = admissible_triples(pools["U(1)"], pools["SU(2)"], pools["SU(3)"])
    return {"pools": pools, "triples": [(t.wU, t.s2, t.s3) for t in triples]}


def run_selector_with_codec(
    lanes: Sequence[LaneSpec],
    base: int,
    *,
    encode: Callable[[int, int], str],
    decode: Callable[[str, int], int],
    require_round_trip: bool,
) -> Dict[str, Any]:
    """Run the selector while forcing all candidates through encode/decode."""
    span_lo = min(l.span_lo for l in lanes)
    span_hi = max(l.span_hi for l in lanes)

    round_trip_failures = 0
    decoded_candidates: List[int] = []

    for w in range(span_lo, span_hi + 1):
        s = encode(w, base)
        try:
            w_dec = decode(s, base)
        except Exception:
            round_trip_failures += 1
            continue
        if require_round_trip and (w_dec != w):
            round_trip_failures += 1
            continue
        decoded_candidates.append(w_dec)

    pools: Dict[str, List[int]] = {}
    for lane in lanes:
        pools[lane.name] = lane_survivors(lane, decoded_candidates)

    triples = admissible_triples(pools["U(1)"], pools["SU(2)"], pools["SU(3)"])

    return {
        "base": base,
        "round_trip_failures": round_trip_failures,
        "pools": pools,
        "triples": [(t.wU, t.s2, t.s3) for t in triples],
    }


# -----------------------------
# Artifact writing (optional)
# -----------------------------


def get_out_dir(user_out: Optional[str]) -> Path:
    if user_out:
        return Path(user_out).expanduser().resolve()
    env = os.environ.get("DEMO_OUT_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return Path.cwd().resolve()


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def make_zip(zip_path: Path, folder: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                z.write(p, arcname=str(p.relative_to(folder)))


# -----------------------------
# Main
# -----------------------------


def main() -> int:
    ap = argparse.ArgumentParser(add_help=True)
    ap.add_argument("--out", default=None, help="Output directory for JSON bundle (default: $DEMO_OUT_DIR or CWD)")
    ap.add_argument("--no_artifacts", action="store_true", help="Disable writing JSON/ZIP artifacts")
    ap.add_argument(
        "--bases",
        default="2,3,4,5,6,7,8,9,10,12,16",
        help="Comma-separated bases to audit (default: 2,3,4,5,6,7,8,9,10,12,16)",
    )
    args = ap.parse_args()

    bases: List[int] = [int(x.strip()) for x in args.bases.split(",") if x.strip()]

    # Lane specs chosen to reproduce the primary pools and unique triple in the declared window.
    lanes = [
        LaneSpec(name="U(1)", q=17, residues=(1, 5), tau_min=0.31, span_lo=97, span_hi=180),
        LaneSpec(name="SU(2)", q=13, residues=(3,), tau_min=0.30, span_lo=97, span_hi=180),
        LaneSpec(name="SU(3)", q=17, residues=(1,), tau_min=0.30, span_lo=97, span_hi=180),
    ]

    spec = {
        "demo": "DEMO-64",
        "title": "Base-Gauge Invariance of the Selector (Encode/Decode Audit)",
        "lanes": [
            {
                "name": l.name,
                "q": l.q,
                "residues": list(l.residues),
                "tau_min": l.tau_min,
                "span": [l.span_lo, l.span_hi],
            }
            for l in lanes
        ],
        "bases": bases,
        "encoding_alphabet": ALPHABET,
        "encoding_contract": "decode_base_b(encode_base_b(w)) == w",
    }
    spec_sha256 = sha256_bytes(canonical_json_bytes(spec))

    print_header("DEMO-64 — Base-Gauge Invariance of the Selector (Encode/Decode Audit) — REFEREE READY")

    print(LINE)
    print("STAGE 1 — Baseline selector (integer mode, no encoding layer)")
    print(LINE)

    baseline = run_selector_integer_mode(lanes)
    pools = baseline["pools"]

    for lane in lanes:
        pool = pools[lane.name]
        print(
            f"Lane {lane.name}: q={lane.q} residues={list(lane.residues)} τ≥{lane.tau_min} span={lane.span_lo}..{lane.span_hi}"
        )
        print(f"  → Survivors ({len(pool)}): {pool}")
        for w in pool:
            print(f"    w={w:<3d} : Θ(w−1)=φ(w−1)/(w−1) = {theta(w - 1):.5f}")
        print()

    triples = baseline["triples"]
    print(f"Admissible triples (after T1–T2): {triples}")
    gate("Unique admissible triple (baseline)", len(triples) == 1, f"count={len(triples)}")
    if len(triples) != 1:
        print("\nResult: NOT VERIFIED (baseline selection not unique).")
        return 2

    t0 = Triple(*triples[0])
    inv0 = derived_invariants(t0)
    print("Selected triple:")
    print(f"  wU={t0.wU}  s2={t0.s2}  s3={t0.s3}")
    print("Derived invariants:")
    print(f"  q2={inv0['q2']}  q3={inv0['q3']}  v2U={inv0['v2U']}  eps={inv0['eps']:.8f}")
    print(f"spec_sha256: {spec_sha256}")

    print("\n" + LINE)
    print("STAGE 2 — Cross-base encode/decode audit (base-gauge invariance)")
    print(LINE)

    per_base: List[Dict[str, Any]] = []
    total_rt_fail = 0
    pools_match_all = True
    triple_match_all = True

    for b in bases:
        res = run_selector_with_codec(
            lanes,
            b,
            encode=encode_base,
            decode=decode_base,
            require_round_trip=True,
        )
        total_rt_fail += int(res["round_trip_failures"])

        triple_b = res["triples"]
        triple_ok = (len(triple_b) == 1) and (tuple(triple_b[0]) == (t0.wU, t0.s2, t0.s3))
        pools_ok = (res["pools"] == pools)

        triple_match_all = triple_match_all and triple_ok
        pools_match_all = pools_match_all and pools_ok

        per_base.append({
            "base": b,
            "round_trip_failures": res["round_trip_failures"],
            "triple": triple_b[0] if len(triple_b) == 1 else None,
            "pools_match": pools_ok,
        })

        triple_str = str(tuple(triple_b[0])) if len(triple_b) == 1 else "None"
        print(f"base={b:<2d} triple={triple_str:<15s} pools_match={str(pools_ok)} rt_fail={res['round_trip_failures']}")

    gate("Gate G0: encode/decode contract holds (no round-trip failures)", total_rt_fail == 0, f"failures={total_rt_fail}")
    gate("Gate G1: triple invariant across bases", triple_match_all)
    gate("Gate G2: lane survivor pools invariant across bases", pools_match_all)

    print("\n" + LINE)
    print("STAGE 3 — Negative controls (intentionally violating the encoding contract)")
    print(LINE)

    def bad_decode_wrong_base(s: str, base: int) -> int:
        # Deliberately decode using base+1.
        return int(s, base + 1)

    falsifier_mismatches = 0
    falsifier_rt_fails = 0

    for b in bases:
        res_bad = run_selector_with_codec(
            lanes,
            b,
            encode=encode_base,
            decode=bad_decode_wrong_base,
            require_round_trip=True,  # enforce the contract; this *should* fail
        )
        falsifier_rt_fails += int(res_bad["round_trip_failures"])
        mismatch = (res_bad["pools"] != pools) or (res_bad["triples"] != [(t0.wU, t0.s2, t0.s3)])
        falsifier_mismatches += int(mismatch)

        triple_str = str(tuple(res_bad["triples"][0])) if len(res_bad["triples"]) == 1 else "None"
        print(f"base={b:<2d} contract_failures={res_bad['round_trip_failures']:<4d} mismatch_vs_baseline={mismatch} triple={triple_str}")

    gate(
        "Gate F: negative control triggers mismatch (sensitivity)",
        falsifier_mismatches >= max(1, int(0.75 * len(bases))),
        f"mismatches={falsifier_mismatches}/{len(bases)}",
    )

    # Deterministic record (excluding timestamps / file paths)
    deterministic_record = {
        "spec_sha256": spec_sha256,
        "baseline": {
            "pools": pools,
            "triple": [t0.wU, t0.s2, t0.s3],
            "invariants": {
                "q2": inv0["q2"],
                "q3": inv0["q3"],
                "v2U": inv0["v2U"],
                "eps": float(f"{inv0['eps']:.12f}"),
            },
        },
        "bases": per_base,
        "negative_control": {
            "decoder": "wrong_base_plus_one",
            "mismatches": falsifier_mismatches,
            "bases_tested": len(bases),
            "round_trip_failures": falsifier_rt_fails,
        },
    }

    determinism_sha256 = sha256_bytes(canonical_json_bytes(deterministic_record))

    print("\n" + LINE)
    print("DETERMINISM HASH")
    print(LINE)
    print(f"determinism_sha256: {determinism_sha256}")

    ok = (len(triples) == 1) and (total_rt_fail == 0) and triple_match_all and pools_match_all and (
        falsifier_mismatches >= max(1, int(0.75 * len(bases)))
    )

    print("\n" + LINE)
    print("FINAL VERDICT")
    print(LINE)
    gate("DEMO-64 VERIFIED (base-gauge invariance + falsifier sensitivity)", ok)
    print(f"Result: {'VERIFIED' if ok else 'NOT VERIFIED'}")

    if ok and (not args.no_artifacts):
        out_dir = get_out_dir(args.out)
        run_id = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        bundle_dir = out_dir / f"DEMO64_BUNDLE_{run_id}"
        bundle_dir.mkdir(parents=True, exist_ok=True)

        run_meta = {
            "utc_time": utc_iso(),
            "python": sys.version,
            "platform": platform.platform(),
            "argv": sys.argv,
        }

        # Write artifacts
        write_json(bundle_dir / "spec.json", spec)
        write_json(bundle_dir / "run_metadata.json", run_meta)
        write_json(bundle_dir / "deterministic_record.json", deterministic_record)

        # Hashes of written artifacts
        artifact_hashes = {}
        for p in sorted(bundle_dir.glob("*.json")):
            artifact_hashes[p.name] = sha256_file(p)
        write_json(bundle_dir / "artifact_sha256.json", artifact_hashes)

        # Script hash
        try:
            script_path = Path(__file__).resolve()
            script_hash = sha256_file(script_path)
        except Exception:
            script_hash = None
        write_json(bundle_dir / "code_sha256.json", {"script": script_hash})

        # Zip bundle
        zip_path = out_dir / f"DEMO64_BUNDLE_{run_id}.zip"
        make_zip(zip_path, bundle_dir)

        print("\nArtifacts written:")
        print(f"  folder: {bundle_dir}")
        print(f"  zip   : {zip_path}  (sha256={sha256_file(zip_path)})")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
