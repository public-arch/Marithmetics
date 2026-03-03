# DEMO-68 — General Relativity Master Flagship

> **Claim:** A deterministic weak-field GR demo using the primary triple (137, 107, 103) reconstructs four classic GR tests as spectral witnesses, completes Einstein's geometric-optics closure via Fermat compatibility, and demonstrates that lawful (Fejér) operators satisfy all gates while illegal controls fail, with counterfactual budgets degrading performance by at least (1+eps).

---

## What this demo computes

A deterministic, audit-grade GR demonstration that:
1. Selects the primary triple (wU, s2, s3) = (137, 107, 103) deterministically
2. Derives budgets (q2, q3, eps) from the triple
3. Reconstructs four classic weak-field GR tests as discrete/spectral witnesses:
   - Light bending: α(b) ∝ 1/b
   - Shapiro delay: Δt(b) ≈ a ln b + c
   - Redshift proxy: Φ(r) ≈ A(1/r) + C (shell means)
   - Perihelion proxy: Φ(r) ≈ -M/r (near-field; rΦ(r) ≈ const)
4. Enforces DOC-style admissibility: Fejér kernel (positive; no Gibbs negativity) versus illegal controls (sharp cutoff with Dirichlet ringing; signed HF injection)
5. Validates mass closure, near-field 1/r slope, and Fermat compatibility (α(b) ≈ dΔt/db within eps for lawful, violated by illegal)
6. Tests resolution ladder invariance (max/min distortion ratio ≤ 1+eps across tiers)
7. Demonstrates counterfactual teeth: at least 3/4 deterministic budget shifts degrade all scores by ≥(1+eps)

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. All light-bending, Shapiro, redshift, and perihelion subtest gates must pass.
3. Perihelion gates (P1–P5): Fejér mass closure, near-field slope, illegal filters must increase ringing and worsen deviation.
4. Einstein completion gates (E1–E5): Fermat compatibility must hold within eps for Fejér; illegal filters must break it beyond margin.
5. At least 3/4 counterfactual budgets must degrade all scores by (1+eps); fewer strong counterfactuals falsify.
6. Ladder invariance gates (L1–L3): tier distortion bounded by eps; max/min ratio ≤ 1+eps; designed FAIL must increase distortion.
7. At least 3/4 counterfactual budgets must degrade tier distortion by (1+eps).
8. Determinism hash must match reference; any change falsifies.

## Controls

- **Illegal operators:** Sharp cutoff (Dirichlet ringing); signed HF-injecting kernel
- **Counterfactuals:** Deterministic budget shifts must degrade all perturbed scores
- **Ladder tests:** Resolution ladder invariance; designed FAIL gate for sensitivity validation

## Dependencies

Python 3.10+, numpy

## Run

```bash
python demo.py
```
