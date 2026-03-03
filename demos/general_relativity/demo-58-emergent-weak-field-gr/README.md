# DEMO-58 — Emergent Weak-Field General Relativity (Poisson / Shapiro / redshift)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) yields budgets (eps, N, K_primary, K_truth); exact FFT-based discrete Poisson solver on 3D periodic lattice shows weak-field observables (Newtonian slope |g(r)| ~ 1/r², light-bending α(b) ~ 1/b, Shapiro delay D(b) ~ ln(b), redshift Φ(r) ~ 1/r) are recovered by admissible Fejér operator but corrupted by illegal controls, with counterfactual triples degrading by fixed eps margin.

---

## What this demo computes

Pipeline (all deterministic):
- Deterministic selection: unique prime triple (wU, s2, s3) = (137, 107, 103) via fixed congruence, totient-density, and 2-adic rules.
- Budget derivation: eps = 1/√q2 ≈ 0.1826; N (grid size), K_primary (primary cutoff), K_truth (truth cutoff) from triple invariants.
- Discrete Poisson solver: solve ΔΦ = ρ on 3D periodic lattice using exact FFT eigenvalues of discrete Laplacian.
- Operator comparison on Φ̂ (Fourier domain):
  - Admissible: Fejér triangle weights → nonnegative real-space kernel.
  - Illegal: sharp spectral cutoff (negative kernel lobes), signed HF injection (stronger lobes).
- Weak-field observables extracted from Φ:
  - Newtonian limit: |g(r)| ~ 1/r² slope recovery (truth slope ≈ -2.01, admissible ≈ -1.94).
  - Light bending: α(b) ~ 1/b affine fit (truth R² > 0.98, admissible R² > 0.95).
  - Shapiro delay: D(b) ~ ln(b) affine fit (truth R² > 0.99, admissible R² > 0.99).
  - Redshift: Φ(r) ~ 1/r shell-mean fits (truth R² > 0.99, admissible R² > 0.99).
- Counterfactual teeth: alternative triples (same rules, larger window) must degrade slopes and fits by (1+eps) margin.
- Illegal controls must inject HF or increase ringing curvature beyond admissible.

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Admissible slopes/fits must be near-truth within eps tolerance.
5. Illegal controls must have stronger ringing curvature than admissible.
6. Counterfactual triples must degrade by (1+eps) in all observables.
7. Residual contract: filtered Poisson residuals (truth vs admissible) must match.

## Controls

- **Illegal operators:** Sharp spectral cutoff (Gate N2, N3, N4), signed HF injection (Gate N3, N4) with negative kernel lobes; tested across all observable suites (Newtonian/N, light-bending/B, Shapiro/S, redshift/R).
- **Counterfactuals:** Alternative triples from extended window; must degrade all slopes/fits by (1+eps).
- **Ablations:** Truth-tier (higher K_truth) provides reference without external data.

## Dependencies

Python 3.10+ with NumPy.

## Run

```bash
python demo.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window  count=1
PASS  Gate N0: filtered Poisson residual contract (truth vs admissible)  res_t=1.733e-03 res_a=1.927e-03
PASS  Gate N1: truth slope near -2  slope=-2.01171 eps=0.182574
PASS  Gate N2: admissible slope near -2  slope=-1.94229 eps=0.182574
PASS  Gate N3: signed control injects HF (>= max(10*hf_a, eps^3))  hf_signed=3.549e-01 floor=6.086e-03
PASS  Gate N4: a non-admissible control has stronger ringing curvature  curv_a=8.799e-03 curv_max=5.889e-02 eps=0.182574
PASS  Gate B1: truth slope near -1  slope=-1.0535 eps=0.182574
PASS  Gate B2: admissible slope near -1  slope=-0.950942 eps=0.182574
PASS  Gate B3: non-admissible injects HF (>= max(10*hf_a, eps^2))  hf_signed=3.549e-01 floor=3.333e-02
PASS  Gate B4: non-admissible has higher ringing curvature (>= (1+eps)×adm)  curv_a=2.429e-02 curv_max=1.595e+00 eps=0.182574
PASS  Gate S0: filtered Poisson residual contract (truth vs admissible)  res_t=1.733e-03 res_a=1.927e-03
PASS  Gate S1: truth affine in ln(b) (R2 >= 0.98)  R2=0.999969
PASS  Gate S2: admissible affine in ln(b) (R2 >= 0.95)  R2=0.999584
PASS  Gate S3: signed control injects HF (>= max(10*hf_a, eps^3))  hf_signed=4.364e-02 floor=6.086e-03
PASS  Gate S4: non-admissible has higher curvature (>= (1+eps)×adm)  curv_a=2.222e-03 curv_max=1.609e-02 eps=0.182574
PASS  Gate R0: filtered Poisson residual contract (truth vs admissible)  res_t=1.733e-03 res_a=1.927e-03
PASS  Gate R1: truth affine in (1/r) (R2 >= 0.98)  R2=0.999845
PASS  Gate R2: admissible affine in (1/r) (R2 >= 0.95)  R2=0.999164
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `9991051fd4c5043945221abab354a5b23e011a60105e2ee12df26a2ba974d26a`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)  selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate N0: filtered Poisson residual contract (truth vs admissible)  res_t=1.733e-03 res_a=1.927e-03
PASS  Gate N1: truth slope near -2  slope=-2.01171 eps=0.182574
PASS  Gate N2: admissible slope near -2  slope=-1.94229 eps=0.182574
PASS  Gate N3: signed control injects HF (>= max(10*hf_a, eps^3))  hf_signed=3.549e-01 floor=6.086e-03
PASS  Gate N4: a non-admissible control has stronger ringing curvature  curv_a=8.799e-03 curv_max=5.889e-02 eps=0.182574
PASS  Gate B1: truth slope near -1  slope=-1.0535 eps=0.182574
PASS  Gate B2: admissible slope near -1  slope=-0.950942 eps=0.182574
PASS  Gate B3: non-admissible injects HF (>= max(10*hf_a, eps^2))  hf_signed=3.549e-01 floor=3.333e-02
PASS  Gate B4: non-admissible has higher ringing curvature (>= (1+eps)×adm)  curv_a=2.429e-02 curv_max=1.595e+00 eps=0.182574
PASS  Gate S0: filtered Poisson residual contract (truth vs admissible)  res_t=1.733e-03 res_a=1.927e-03
PASS  Gate S1: truth affine in ln(b) (R2 >= 0.98)  R2=0.999969
PASS  Gate S2: admissible affine in ln(b) (R2 >= 0.95)  R2=0.999584
PASS  Gate S3: signed control injects HF (>= max(10*hf_a, eps^3))  hf_signed=4.364e-02 floor=6.086e-03
PASS  Gate S4: non-admissible has higher curvature (>= (1+eps)×adm)  curv_a=2.222e-03 curv_max=1.609e-02 eps=0.182574
PASS  Gate R0: filtered Poisson residual contract (truth vs admissible)  res_t=1.733e-03 res_a=1.927e-03
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
DETERMINISM HASH
====================================================================================================
determinism_sha256: 9991051fd4c5043945221abab354a5b23e011a60105e2ee12df26a2ba974d26a

====================================================================================================
VERDICT
====================================================================================================
PASS  DEMO-58 VERIFIED (weak-field suite: scaling + operator falsifiers + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
