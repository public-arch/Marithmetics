# DEMO-11 — Step tests do not certify admissibility

## Thesis

A spectral operator can make a canonical step input appear well-behaved (overshoot near zero), while still failing admissibility badly on other {0,1} inputs.

This demo constructs a guaranteed lower bound on admissibility violation from kernel negativity, showing that:

- Fejér (Cesàro) low-pass is admissible (kernel nonnegative).
- Sharp cutoff low-pass is non-admissibile (negative kernel mass).
- Signed cutoff control is non-admissible (negative kernel mass).

The step test is a useful diagnostic, but it is not a certificate.

 What the demo produces

For a length-N periodic grid and cutoff K swept from 1 to Nyquist-‒1:

1. Observed step overshoot
   - Max violation outside [0,1] when filtering the canonical step.

2. Guaranteed worst-case violation lower bound (constructive)
   - A witness input x ∈ {0,1}^N constructed directly from negative kernel coefficients.
   - If any convolution row has negative coefficients, admissibility fails.

3. Mechanism visualization
   - Kernel triptych showing Fejér vs Sharp vs Signed kernels in real space.
   - Negative lobes are the admissibility failure mechanism.

4. Paper-ready figures
   - HERO: observed step overshoot vs guaranteed violation LB across K
  - MECHANISM: kernel triptych with zoom
  - PHASE: observed overshoot vs guaranteed failure colored by K

5. Reproducibility hashes
   - spec_sha256: hashes experiment parameters
  - determinism_sha256: hashes numeric outputs
  - per-file sha256 in MANIFEST.json

## How to run

From repo root:

```bash
python demos/controllers/demo-11-step-tests-do-not-certify-admissibility/demo.py --out demo_out/demo-11
```

Outputs:

```
demo_out/demo-11/
  REPORT.md
  MANIFEST.json
  data/demo_data.json
  figures/Fig1_HERO.png + .pdf
  figures/Fig2_KernelTriptych.png + .pdf
  figures/Fig3_ObservedVsGuaranteed.png + .pdf
```

## Expected results

- Fejér:
  - kmin ≥ 0 (within tolerance)
  - worst-case LB ≈ 0 across K

- Sharp cutoff:
  - kmin < 0 for all K ≥ 1
  - worst-case LB > 0 for all K≥ 1
  - At Nyquist-‒1: observed step overshoot can be ~0 while LB stays bounded away from 0

- Signed cutoff:
  - kmin < 0 for all K ≥ 1
  - worst-case LB > 0 for all K≥ 1
  - At Nyquist-‒1: step overshoot ~0 while LB remains large

## Gates

This demo is considered PASS if all of the following hold:

- Mass preservation: sum(kernel) = 1 at K_primary (all operators)
- Fei��r admissible: kernel nonnegative and LB≈0 across K
- Sharp non-admissible: negative kernel mass and LB>0 across K
- Signed non-admissibile: negative kernel mass and LB >0 across K
- Illusion point: step overshoot ≈ 0 at Nyquist‑1 while LB remains > 0
- Witness identity: witness filtering reproduces neg_sum/pos_sum tightly

## Notes

- Deterministic: no random numbers.
- Dependencies: numpy, matplotlib.
- This demo is intended as a governance-quality boundary: “step pass” is not an admissibility certificate.