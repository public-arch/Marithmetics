# DEMO-18 — Integer Triple Uniqueness

> **Claim:** The primary integer triple (wU, s2, s3) = (137, 107, 103) is the unique output of five number-theoretic gates derived from Phi-structure. The triple is a necessity, not a design choice.

---

## What this demo computes

A deterministic scan over a wide integer range [80, 800] that applies five gates per channel (alpha, su2, pc2) and shows that exactly one prime survives each channel:

1. **C4' Period extremality** — the largest odd prime factor q of (w-1) must satisfy q % 4 == 1 and q > sqrt(w)
2. **C4'' Phi-parity** — Legendre(2|q) must match the channel's required sign (alpha: +1, su2: -1, pc2: +1)
3. **C2'' 2-adic branch** — the 2-adic valuation v2(w-1) must equal the channel's required depth (alpha: 3, su2: 1, pc2: 1)
4. **C5'' Wheel orientation** — for PC2 only, Legendre(5|q) must equal -1 (excludes 83, selects 103)
5. **C6' Simplicity/minimality** — prefer prime survivors; among primes choose minimal w

The mod-8 class and 2-adic branch are derived from two ingredients tied to Phi-structure: Phi-parity via Legendre symbols and the Fejer/MST minimal envelope.

## Falsification contract

1. Any channel that fails to produce the expected survivor falsifies the demo.
2. The alpha channel must select w* = 137; any other value falsifies.
3. The su2 channel must select w* = 107; any other value falsifies.
4. The pc2 channel must select w* = 103; any other value falsifies.
5. All five gates must pass for each selected survivor; any gate failure falsifies.

## Controls

- **Ablation (--loose-parity):** Disabling the Phi-parity gate admits additional survivors, breaking uniqueness
- **Ablation (--loose-v2):** Disabling the 2-adic branch gate admits additional survivors, breaking uniqueness
- **Ablation (--loose-5):** Disabling wheel orientation admits w=83 alongside w=103 for PC2, breaking uniqueness

## Dependencies

Python 3.10+ (stdlib-only; no third-party packages)

## Run

```bash
python demo.py
```

Ablation examples:

```bash
python demo.py --loose-parity
python demo.py --loose-v2
python demo.py --loose-5
```
