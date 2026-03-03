# DEMO-36 — Big Bang Master Flagship

> **Claim:** A unique admissible prime triple (wU, s2, s3) = (137, 107, 103) selected by transparent lane filter and coherence constraint deterministically generates a complete structural cosmology parameter set that satisfies all spectral closure gates.

---

## What this demo computes

A deterministic cosmological exhibit that:
1. Selects the primary triple by declared prime window and coherence constraints
2. Derives budgets (q2, q3, eps, N, K) from the triple
3. Generates a full structural cosmology set (H0, Ω_b h², Ω_c h², A_s, n_s, τ, ℓ1, δ_CMB) via fixed BB-36 monomials
4. Constructs two independent spectrum-level observables (tilt proxy and power-sum amplitude proxy)
5. Audits each against admissible operator (Fejér/Cesàro; nonnegative kernel) versus illegal controls (sharp cutoff; signed HF injection)
6. Demonstrates counterfactual teeth—deterministic budget shifts degrade all observable scores by at least (1+eps)

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. The selected triple must equal (137, 107, 103); any other value falsifies.
3. All structural gates (S1–S8: H0, Ω's, A_s, n_s, τ, ℓ1, δ_CMB) must pass within stated ranges.
4. All spectral gates (T1–T6, A1–A4) must pass; illegal controls must show negative lobes or HF injection.
5. At least 3/4 counterfactual budgets must degrade by (1+eps); any exceptions falsify.
6. Determinism hash must match reference; any change falsifies.

## Controls

- **Illegal operators:** Sharp cutoff (Dirichlet ringing); signed HF-injecting kernel
- **Counterfactuals:** Deterministic budget shifts (K reduced by q3→3q3) must degrade scores
- **Ablations:** Removing Fejér constraint must cause counterfactuals to perform worse

## Dependencies

Python 3.10+, numpy. Optional: matplotlib, CAMB (for informational TT first-peak check only).

## Run

```bash
python demo.py
```
