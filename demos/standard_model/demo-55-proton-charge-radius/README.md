# DEMO-55 — Proton Charge Radius (from substrate selection)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-55-proton-charge-radius-from-substrate-selection.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-55' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
r_p(2-loop Lambda_5) = 0.8430139282 fm
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO 55 - Proton Charge Radius from Substrate Selection

> Purpose
> -------
> A zero-knob, zero-tuning, first-principles audit that:
>   1) selects the unique SCFP triple in the primary prime window (97..180),
>   2) derives alpha_s(MZ) from the selected triple,
>   3) computes the QCD scale Lambda_5 in a fixed 2-loop MS-bar scheme (nf=5, mu=MZ),
>   4) maps Lambda_5 to the proton rms charge radius via a fixed dressing factor,
>   5) falsifies via counterfactual admissible triples under a reduced gate set.

> Notes
> -----
> - No file I/O is required.
> - Any external reference values are evaluation-only and do not affect selection.
> - This script is designed to run on minimal Python installs (math/json/hashlib/time/platform).

## Run instructions

- Dependencies: Stdlib-only (no third-party packages).

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-55-proton-charge-radius-from-substrate-selection.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window  count=1
PASS  Primary proton radius within 1% (evaluation-only gate)  rel_err=+0.269275%
PASS  DEMO 55 VERIFIED (selection + proton radius + counterfactual ablation)
```

## Reference checkpoints (from provided transcript)

- Spec SHA256: `0a46a38ce93a14d9770b8b7a77aca810eb497943d0a4cafef7b7d069454ba1c1`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)  selected=(137, 107, 103)
Admissible triple: (137, 107, 103)
Spec SHA256: 0a46a38ce93a14d9770b8b7a77aca810eb497943d0a4cafef7b7d069454ba1c1
alpha_s(MZ) = 2/q3    = 0.117647058823529
STAGE 3 - LAMBDA_QCD FROM alpha_s(MZ) (NO PDG FIT)
alpha_s(MZ)              = 0.117647058823529
Lambda_5 (1-loop, nf=5)  = 0.0858720264356 GeV
Lambda_5 (2-loop, nf=5)  = 0.221411095276 GeV
PASS  Sanity: Lambda_5(2-loop) in expected ballpark  Lambda5=0.221411
r_p = (hbar*c / Lambda_5) * sqrt(1 / (1 + alpha_s(MZ)))
r_p(1-loop Lambda_5) = 2.17361398 fm
r_p(2-loop Lambda_5) = 0.8430139282 fm
ref r_p              = 0.84075 fm
(277, 263, 239)  alpha_s=0.028985507  Lambda5=1.89784e-10 GeV  r_p=1.025e+09 fm  ratio=1.21914e+09  MISS
(307, 263, 239)  alpha_s=0.013071895  Lambda5=2.72683e-25 GeV  r_p=7.18965e+23 fm  ratio=8.55147e+23  MISS
(311, 263, 239)  alpha_s=0.012903226  Lambda5=1.20638e-25 GeV  r_p=1.62525e+24 fm  ratio=1.93309e+24  MISS
```

Transcript excerpt (for quick visual diff):

```text
(409, 263, 239)  alpha_s=0.039215686  Lambda5=2.76483e-07 GeV  r_p=700109 fm  ratio=832720  MISS
(409, 367, 239)  alpha_s=0.039215686  Lambda5=2.76483e-07 GeV  r_p=700109 fm  ratio=832720  MISS
(409, 367, 307)  alpha_s=0.039215686  Lambda5=2.76483e-07 GeV  r_p=700109 fm  ratio=832720  MISS
PASS  All counterfactuals miss outside fixed ratio band  strong_misses=6/6  band=(0.8, 1.2)

====================================================================================================
                                           FINAL VERDICT                                            
====================================================================================================
PASS  DEMO 55 VERIFIED (selection + proton radius + counterfactual ablation)

Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
