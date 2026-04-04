# DEMO-18 — Integer Triple Uniqueness

> **Claim:** Five number-theoretic gates derived from Phi-structure — period extremality, Phi-parity (Legendre symbol), 2-adic branching, wheel orientation, and simplicity — select the primary triple (137, 107, 103) as the minimal prime survivor per channel over [80, 800].

---

## What this demo computes

A deterministic scan over integers [80, 800] that applies five gates independently to three channels (alpha, su2, pc2):

1. **C4' Period extremality** — the largest odd prime factor q of (w-1) must satisfy q % 4 == 1 and q > sqrt(w)
2. **C4'' Phi-parity** — Legendre(2|q) must match the channel's required sign (alpha: +1, su2: -1, pc2: +1)
3. **C2'' 2-adic branch** — the 2-adic valuation v2(w-1) must equal the channel's required depth (alpha: 3, su2: 1, pc2: 1)
4. **C5'' Wheel orientation** — for PC2 only, Legendre(5|q) must equal -1 (excludes 83, selects 103)
5. **C6' Simplicity/minimality** — prefer prime survivors; among primes choose minimal w

Gates C4'–C5'' are hard number-theoretic constraints derived from Phi-structure. C6' is a simplicity ranking applied to the surviving candidates.

## Falsification contract

1. The alpha channel must select w* = 137; any other value falsifies.
2. The su2 channel must select w* = 107; any other value falsifies.
3. The pc2 channel must select w* = 103; any other value falsifies.
4. Gates C4'–C5'' must pass for each selected survivor; any gate failure falsifies.

## Ablations

Each gate can be independently disabled to demonstrate its necessity:

- **--loose-parity:** Disabling the Phi-parity gate admits additional survivors, breaking minimal-prime selection
- **--loose-v2:** Disabling the 2-adic branch gate admits additional survivors, breaking minimal-prime selection
- **--loose-5:** Disabling wheel orientation admits w=83 alongside w=103 for PC2, breaking selection

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
