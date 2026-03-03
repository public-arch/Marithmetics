# DEMO-67 — Navier–Stokes Master Flagship (3D Taylor–Green certificate)

> **Claim:** A deterministic 3D incompressible pseudo-spectral Navier–Stokes simulation using the primary triple (137, 107, 103) and derived budgets preserves incompressibility and demonstrates that the lawful (Fejér/Cesàro) operator is closer to truth than illegal controls, with counterfactual budgets degrading performance by at least (1+eps).

---

## What this demo computes

A deterministic industrial-grade Navier–Stokes flagship demo that:
1. Selects the primary triple (wU, s2, s3) = (137, 107, 103) deterministically
2. Derives budgets (q2, q3, eps, K_primary, K_truth, ν, dt, steps) from the triple and chosen tier
3. Runs a 3D incompressible pseudo-spectral Taylor–Green vortex benchmark with configurable resolution (smoke tier: ~64³; industrial tier: 256³)
4. Verifies incompressibility: divergence L2 must remain ≤ 1e-8 across all operator variants
5. Audits lawful (Fejér/Cesàro) operator against two illegal controls: sharp cutoff (Dirichlet ringing) and signed HF injection
6. Confirms lawful operator score is lower (closer to truth) than illegal control scores
7. Demonstrates counterfactual teeth: deterministic budget shifts (K reduced by q3→3q3) must degrade performance by ≥(1+eps)

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. The selected triple must equal (137, 107, 103); any other value falsifies.
3. Incompressibility gate (G1) must pass: all divergence L2 norms ≤ 1e-8 (lawful, sharp, signed variants).
4. Admissibility gate (G2) must pass: lawful operator score must be lower than all illegal control scores.
5. HF injection gate (G3) must pass: signed illegal control must inject HF weight beyond floor (3.333e-02).
6. At least 3/4 counterfactual budgets (T1) must degrade by (1+eps); fewer strong counterfactuals falsify.
7. Determinism hash must match reference; any change falsifies.

## Controls

- **Illegal operators:** Sharp cutoff (Dirichlet; high-frequency ringing); signed HF-injecting kernel
- **Counterfactuals:** Deterministic budget shifts (q3→3q3) reducing K must degrade score by ≥(1+eps)
- **Tier options:** Smoke (quick validation, ~64³) and industrial (full referee-grade, 256³)

## Dependencies

Python 3.10+, numpy. Optional: scipy (for faster FFT backend; numpy.fft used as fallback), matplotlib.

## Run

```bash
python demo.py [--tier industrial]
```
