# DEMO-65 — Continuous Lift Paradox (capstones + GR witnesses)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-65-continuous-lift-paradox-capstones-gr-witnesses.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-65' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  Gate Q3: illegal density negativity (<= -eps^2)                        eps^2=0.0333333
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-65 — CONTINUOUS LIFT PARADOX

> Goal
> ----
> Provide a deterministic, first-principles, audit-grade demonstration of the
> "continuous lift paradox":

>     *Certain discrete operator choices look harmless (or even "sharp") but
>     violate continuum legality classes (positivity / admissibility / invariants).
>     An admissible operator family (Fejér / Cesàro-summed spectral projection)
>     avoids these violations and produces stable, falsifiable signatures.*

> This script is deliberately self-contained:
>   - No I/O required (stdout only by default)
>   - NumPy only (optional JSON/PNG artifacts attempted but never required)
>   - Deterministic selection of a primary triple and deterministic counterfactuals
>   - Explicit legal vs illegal operator classes + "teeth" (counterfactual degradation)

> What this demo *is*:
>   - A reproducible computational certificate: if you run the same code, you obtain

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-65-continuous-lift-paradox-capstones-gr-witnesses.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                             count=1
PASS  Gate P1: Fejér preserves mass within 1e-12                             |Δ|=0
PASS  Gate P2: Fejér preserves nonnegativity (min >= -1e-12)                 min=0.00261178
PASS  Gate P3: illegal produces negative undershoot (<= -eps^2)              eps^2=0.0333333
PASS  Gate P4: illegal increases variation (TV) by >= (1+eps)                eps=0.182574
PASS  Gate P.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate H1: FFT round-trip relative error <= 1e-12                        err=2.60883e-16
PASS  Gate H2: signed retains material HF energy beyond K                    hf=0.505361 floor=0.0333333
PASS  Gate Q1: unitary norm drift <= 1e-10                                   drift=1.04361e-14
PASS  Gate Q2: Fejér density nonnegative (min >= -1e-12)                     min=0.00432282
PASS  Gate Q3: illegal density negativity (<= -eps^2)                        eps^2=0.0333333
PASS  Gate Q.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate N1: legal energy drift <= 1e-10                                   drift=4.34097e-13
PASS  Gate N2: illegal blow-up >= 1e3                                        blow=3.20839e+06
PASS  Gate GR.B1: truth slope near -1 (|Δ|<=0.25)                            slope=-1.02911
PASS  Gate GR.B2: admissible slope near -1 (|Δ|<=0.35)                       slope=-0.940344
PASS  Gate GR.B3: signed illegal retains HF (>= eps^2)                       hf=0.524986 floor=0.0333333
PASS  Gate GR.S1: truth is ln(b)-like (R2>0.98)                              R2=0.999973
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `e8585756d4ada17c788259bbb12a3cf35e57b93ce8138d620f4c03a7cced6141`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                           selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate P1: Fejér preserves mass within 1e-12                             |Δ|=0
PASS  Gate P2: Fejér preserves nonnegativity (min >= -1e-12)                 min=0.00261178
PASS  Gate P3: illegal produces negative undershoot (<= -eps^2)              eps^2=0.0333333
PASS  Gate P4: illegal increases variation (TV) by >= (1+eps)                eps=0.182574
PASS  Gate P.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate H1: FFT round-trip relative error <= 1e-12                        err=2.60883e-16
PASS  Gate H2: signed retains material HF energy beyond K                    hf=0.505361 floor=0.0333333
PASS  Gate Q1: unitary norm drift <= 1e-10                                   drift=1.04361e-14
PASS  Gate Q2: Fejér density nonnegative (min >= -1e-12)                     min=0.00432282
PASS  Gate Q3: illegal density negativity (<= -eps^2)                        eps^2=0.0333333
PASS  Gate Q.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate N1: legal energy drift <= 1e-10                                   drift=4.34097e-13
PASS  Gate N2: illegal blow-up >= 1e3                                        blow=3.20839e+06
PASS  Gate GR.B1: truth slope near -1 (|Δ|<=0.25)                            slope=-1.02911
PASS  Gate GR.B2: admissible slope near -1 (|Δ|<=0.35)                       slope=-0.940344
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
STAGE 7 — Determinism hash + optional artifacts
==================================================================================================
determinism_sha256: e8585756d4ada17c788259bbb12a3cf35e57b93ce8138d620f4c03a7cced6141

==================================================================================================
FINAL VERDICT
==================================================================================================
PASS  DEMO-65 VERIFIED (continuous lift paradox + capstones + GR witnesses + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
