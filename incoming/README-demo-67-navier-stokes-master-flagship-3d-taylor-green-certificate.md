# DEMO-67 — Navier–Stokes Master Flagship (3D Taylor–Green certificate)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-67-navier-stokes-master-flagship-3d-taylor-green-certificate.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-67' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  DEMO-67 VERIFIED (NS3D industrial certificate: admissibility + controls + teeth)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-67 — NAVIER–STOKES MASTER FLAGSHIP (3D Taylor–Green, Industrial Certificate)
>           Operator Admissibility + Illegal Controls + Counterfactual Teeth

> What this is
> A deterministic, self-contained Navier–Stokes flagship demo that:

>   (1) deterministically selects the same primary triple (137,107,103) used across the pipeline,
>   (2) derives budgets (q2,q3,eps,K_primary,K_truth,nu,dt,steps) from the triple and the chosen tier,
>   (3) runs a 3D incompressible pseudo-spectral Taylor–Green vortex benchmark, and
>   (4) verifies a referee-facing certificate:

>       - incompressibility is preserved (divergence L2 small),
>       - the lawful (Fejér/Cesàro) operator is closer to "truth" than illegal controls,
>       - illegal controls inject high-frequency (HF) content / non-admissible behavior,
>       - deterministic counterfactual budgets (K reduced by q3→3q3) degrade the observable by ≥(1+eps).

> This is designed for the "industrial" tier (N=256), but includes a "mobile/smoke" tier for quick runs.
> For the full referee-grade certificate, use --tier industrial (default is smoke to avoid accidental
> multi-hour runs on mobile hardware).

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-67-navier-stokes-master-flagship-3d-taylor-green-certificate.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Gate G1: incompressibility divL2 <= 1e-8 (all variants)                     div_law=1.395e-17 div_sh=3.011e-10 div_si=2.663e-10
PASS  Gate G2: lawful closer to truth than illegal controls                       score_law=7.66841 score_illegal_min=28.4338 strong=True
PASS  Gate G3: signed illegal injects HF weight beyond floor (kernel)             hfW_fejer=0.000e+00 hfW_signed=8.917e-01 floor=3.333e-02
PASS  Gate T1: >=3/4 counterfactuals degrade by (1+eps) on certificate score      strong=4/4 eps=0.182574 score_law=7.66841 score_cf=35.8173
PASS  DEMO-67 VERIFIED (NS3D industrial certificate: admissibility + controls + teeth)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `19af0a06619a6792e639713b480419795f32160576eca3487ddbcdb9f6febe68`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                                selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate G1: incompressibility divL2 <= 1e-8 (all variants)                     div_law=1.395e-17 div_sh=3.011e-10 div_si=2.663e-10
PASS  Gate G2: lawful closer to truth than illegal controls                       score_law=7.66841 score_illegal_min=28.4338 strong=True
PASS  Gate G3: signed illegal injects HF weight beyond floor (kernel)             hfW_fejer=0.000e+00 hfW_signed=8.917e-01 floor=3.333e-02
PASS  Gate T1: >=3/4 counterfactuals degrade by (1+eps) on certificate score      strong=4/4 eps=0.182574 score_law=7.66841 score_cf=35.8173
determinism_sha256: 19af0a06619a6792e639713b480419795f32160576eca3487ddbcdb9f6febe68
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
STAGE 7 — Determinism hash + optional artifacts
====================================================================================================
determinism_sha256: 19af0a06619a6792e639713b480419795f32160576eca3487ddbcdb9f6febe68

====================================================================================================
FINAL VERDICT
====================================================================================================
PASS  DEMO-67 VERIFIED (NS3D industrial certificate: admissibility + controls + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
