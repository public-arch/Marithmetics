# DEMO-68 — General Relativity Master Flagship

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-68-general-relativity-master-flagship.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-68' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  DEMO-68 VERIFIED (GR master flagship: DOC legality + 4 tests + Einstein completion + ladder + teeth)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-68 — GENERAL RELATIVITY MASTER FLAGSHIP (DOC-Admissible Weak-Field GR + Einstein Completion)

> Purpose (referee-facing)
> This script is a single, deterministic, audit-grade demonstration that:

>   (1) Uses the same deterministic primary triple used across the program:
>         Triple(wU, s2, s3) = (137, 107, 103)

>   (2) Enforces a DOC-style admissibility contract:
>         - "Legal" operator = Fejér (positive kernel; no Gibbs negativity)
>         - "Illegal" operators = sharp cutoff (Dirichlet ringing) and signed high-pass (HF injection)

>   (3) Reconstructs the *four classic weak-field GR tests* as discrete / spectral witnesses:
>         A. Light bending:     α(b) ∝ 1/b
>         B. Shapiro delay:     Δt(b) ≈ a ln b + c
>         C. Redshift proxy:    Φ(r) ≈ A(1/r) + C   (shell means)
>         D. Perihelion proxy:  Φ(r) ≈ -M/r (near-field line; rΦ(r) ≈ const)

>   (4) Completes Einstein's geometric-optics closure via a Fermat compatibility witness:
>         α(b) ≈ d(Δt)/db     (within eps for Fejér; violated by illegal filters)

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-68-general-relativity-master-flagship.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Light-bending subtest gates
PASS  Shapiro subtest gates
PASS  Redshift subtest gates
PASS  Gate P1: Fejér mass closure within eps
PASS  Gate P2: near-field 1/r log-slope within eps (Fejér)
PASS  Gate P3: illegal filters increase r*phi spread (ringing)
PASS  Gate P4: signed-kernel HF injection beyond floor
PASS  Gate P5: illegal filters worsen slope deviation
PASS  Gate E1: Fejér Fermat-consistency within eps
PASS  Gate E2: illegal filters break Fermat-consistency margin
PASS  Gate E3: Fejér accuracy vs truth within eps
PASS  Gate E4: illegal filters worsen accuracy vs truth
PASS  Gate E5: signed-kernel HF injection beyond floor
PASS  Teeth gate: >=3/4 counterfactuals degrade all scores by (1+eps)  strong=4/4  eps=0.196116
PASS  Gate L1: tier distortion bounded by eps          max_dist=0.077733 eps=0.196116
PASS  Gate L2: ladder invariance (max/min <= 1+eps)    ratio=1.036077 1+eps=1.196116
PASS  Gate L3: designed FAIL increases distortion by (1+eps)   dist_bad=0.245918  (1+eps)*dist0=0.092978
PASS  Gate LT: counterfactual budgets degrade tier distortion   strong=3/3 eps=0.196116
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `edb0feeccf3ad587a5ac702fdc819e85c69863be6f4e5bbf1aa38cee41c597f8`

- Verdict line: `PASS  DEMO-68 VERIFIED (GR master flagship: DOC legality + 4 tests + Einstein completion + ladder + teeth)`

Selected printed checkpoints:

```text
PASS  Gate P1: Fejér mass closure within eps
PASS  Gate P2: near-field 1/r log-slope within eps (Fejér)
PASS  Gate P3: illegal filters increase r*phi spread (ringing)
PASS  Gate P4: signed-kernel HF injection beyond floor
PASS  Gate P5: illegal filters worsen slope deviation
PASS  Gate E1: Fejér Fermat-consistency within eps
PASS  Gate E2: illegal filters break Fermat-consistency margin
PASS  Gate E3: Fejér accuracy vs truth within eps
PASS  Gate E4: illegal filters worsen accuracy vs truth
PASS  Gate E5: signed-kernel HF injection beyond floor
PASS  Gate L1: tier distortion bounded by eps          max_dist=0.077733 eps=0.196116
PASS  Gate L2: ladder invariance (max/min <= 1+eps)    ratio=1.036077 1+eps=1.196116
PASS  Gate L3: designed FAIL increases distortion by (1+eps)   dist_bad=0.245918  (1+eps)*dist0=0.092978
PASS  Gate LT: counterfactual budgets degrade tier distortion   strong=3/3 eps=0.196116
determinism_sha256: edb0feeccf3ad587a5ac702fdc819e85c69863be6f4e5bbf1aa38cee41c597f8
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
                                        DETERMINISM HASH                                        
==================================================================================================
determinism_sha256: edb0feeccf3ad587a5ac702fdc819e85c69863be6f4e5bbf1aa38cee41c597f8

==================================================================================================
                                         FINAL VERDICT                                          
==================================================================================================
PASS  DEMO-68 VERIFIED (GR master flagship: DOC legality + 4 tests + Einstein completion + ladder + teeth)
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
