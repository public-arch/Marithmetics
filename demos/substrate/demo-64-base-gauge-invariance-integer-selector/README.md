# DEMO-64 — Base–Gauge Invariance of a Deterministic Integer Selector

> **Claim:** The selection procedure that produces the primary integer triple (wU, s2, s3) = (137, 107, 103) is invariant under change of numeral base encoding; the selector depends only on integer invariants, not on human-facing representation.

---

## What this demo computes

A deterministic audit that:
1. Defines a fully explicit integer selector using only elementary number theory (primality, modular congruences, totient ratio Θ(n) = φ(n)/n)
2. Runs the baseline selector in integer mode to obtain survivor pools and the unique admissible triple
3. Re-runs the same selector behind an explicit encode/decode layer across multiple bases (binary through hexadecimal)
4. Validates the encode/decode contract: decode_base_b(encode_base_b(w)) = w for all bases
5. Confirms triple and survivor pools remain identical across all bases when encoding contract holds
6. Runs a negative control by deliberately decoding in the wrong base; the demo must detect and reject these violations

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. The admissible triple must equal (137, 107, 103); any other value falsifies.
3. Encode/decode contract (G0) must have zero round-trip failures; any encoding error falsifies.
4. The triple must be invariant across all encoding bases (G1); any base-dependent result falsifies.
5. Survivor pools must be invariant across all encoding bases (G2); any pool mismatch falsifies.
6. The negative control (F) must detect all 11 deliberate mismatches; fewer than 11 mismatches falsify.
7. Determinism hash must match reference; any change falsifies.

## Controls

- **Negative control:** Deliberately decode in wrong base; demo must reject all 11 such violations
- **Ablations:** Removing encode/decode layers must preserve selector output
- **Sensitivity test:** Contract violations must be detected and reported

## Dependencies

Python 3.10+ (stdlib-only; no third-party packages)

## Run

```bash
python demo.py
```
