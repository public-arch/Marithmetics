# DEMO-58 — Emergent Weak-Field General Relativity (Poisson / Shapiro / redshift)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-58-emergent-weak-field-general-relativity-poisson-shapiro-redshift.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-58' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  DEMO-58 VERIFIED (weak-field suite: scaling + operator falsifiers + teeth)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-58 — Emergent Weak-Field General Relativity from a Discrete Poisson Substrate
> Master Flagship Demo (Referee-Ready)

> Summary
> -------
> This is a deterministic, first-principles computational audit. It does not fit parameters and does
> not tune thresholds.

> Pipeline (all deterministic):
>   1) Select a unique prime triple (wU, s2, s3) in a fixed primary window via fixed congruence,
>      totient-density, and 2-adic rules (no external input).
>   2) Derive budgets (eps, N, K_primary, K_truth) deterministically from the selected triple.
>   3) Solve the discrete Poisson equation ΔΦ = ρ on a 3D periodic lattice using exact eigenvalues of
>      the discrete Laplacian (FFT diagonalization).
>   4) Apply three operator classes to Φ̂ (Fourier domain):
>        - Admissible: Fejér smoothing (nonnegative convolution kernel)
>        - Non-admissible control: sharp cutoff (kernel with negative lobes)
>        - Non-admissible control: signed HF injection (kernel with stronger negative lobes)
>   5) Extract weak-field observables from the same Φ:
>        - Newtonian limit: |g(r)| ~ 1/r^2

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-58-emergent-weak-field-general-relativity-poisson-shapiro-redshift.py
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
