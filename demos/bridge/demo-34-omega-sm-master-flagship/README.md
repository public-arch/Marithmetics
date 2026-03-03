# DEMO-34 — Omega to Standard Model Bridge (Master Flagship)

> **Claim:** A unique admissible integer triple (wU, s2, s3) = (137, 107, 103) is selected deterministically under declared modular constraints and produces a certified bridge between Omega and Standard Model observables.

---

## What this demo computes

A Tier-A₁ joint-triple certificate that:
1. Selects the primary (wU, s2, s3) triple in a declared band [80, 1,000,000] by transparent lane filter and coherence constraints
2. Derives budgets (q2, q3, eps, N, K) from the triple
3. Demonstrates necessity via ablation—removing load-bearing gates causes the survivor pool to explode (loss of uniqueness)
4. Compares against PDG reference values to measure fit (evaluation-only; no feedback to selection)
5. Emits a determinism hash and VERIFIED verdict when all gates pass

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. Any ablation marked "required" must increase the survivor count (explosion) as declared.
3. The selected triple must equal (137, 107, 103); any other value falsifies.
4. Determinism hash must match the reference value; any change falsifies.
5. A missing or invalid certificate section falsifies.

## Controls

- **Illegal operators:** Non-Fejér kernels (sharp cutoff with Dirichlet ringing; signed HF injection)
- **Counterfactuals:** Ablated gates (lane removal, coherence drop) must cause explosion
- **Necessity tests:** Mandatory gates (T1, T2) whose removal breaks uniqueness

## Dependencies

Python 3.10+ (stdlib-only core; numpy/matplotlib optional for diagnostics)

## Run

```bash
python demo.py
```
