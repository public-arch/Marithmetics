# DEMO-60 — Quantum Master Flagship (unitarity + controls)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-60-quantum-master-flagship-unitarity-controls.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-60' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  DEMO-60 VERIFIED (selection + admissibility + quantum suite + ladder + time-reversal + PDE benchmark)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-60 — Quantum Master Flagship
> Referee-ready, first-principles, deterministic, single-file demo.
> NumPy only.

> What this script demonstrates (computationally, not rhetorically)
> A) Deterministic selection:
>    A fixed selection rule identifies a unique prime triple (wU, s2, s3) in a declared window.
>    The triple deterministically sets budgets (N, K, eps). No runtime tuning.

> B) Operator admissibility (probability-safe coarse graining):
>    The Fejér spectral multiplier has a nonnegative real-space kernel (positivity-preserving on densities).
>    Two non-admissible controls (sharp cutoff and signed filter) have negative kernel lobes.

> C) Quantum worked examples (orthogonal checks):
>    E1) Density admissibility on a discontinuous top-hat:
>        - Fejér preserves mass and nonnegativity.
>        - Non-admissible controls create negative undershoot and higher variation.
>        - Counterfactual triples (same rules, different budgets) degrade distortion by a fixed eps margin.

>    E2) Double-slit interference density:

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-60-quantum-master-flagship-unitarity-controls.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                                 count=1
PASS  Gate E1.1: Fejér preserves mass within 1e-12                               |Δ|=0
PASS  Gate E1.2: Fejér preserves nonnegativity (min >= -1e-12)                   min=0.00261178
PASS  Gate E1.3: illegal produces negative undershoot (<= -eps^2)                eps^2=0.0333333
PASS  Gate E1.4: illegal increases variation (TV) by >= (1+eps)                  eps=0.182574
PASS  Gate E1.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate E2.1: unitary norm drift <= 1e-10                                     drift=8.881784e-16
PASS  Gate E2.2: signed illegal distortion >= (1+eps)×fejer                      eps=0.182574
PASS  Gate E2.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate 60A.L0_tiers_verified
PASS  Gate 60A.L1_E1_C_stable
PASS  Gate 60A.L2_E2_C_stable
PASS  Gate 60A.L3_visibility_stable
PASS  Gate 60B.G1_truth_reversible
PASS  Gate 60B.G2_illegal_breaks
PASS  Gate 60B.T_counterfactual
PASS  Tier N=256 Gate 60C.G1_fd_worse_than_fejer
PASS  Tier N=256 Gate 60C.G2_illegal_distorts_more
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `5ae18adb1184e4e7cdc95edddbff85dc0eb3c852ae723521239d840c24933048`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                               selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate E1.1: Fejér preserves mass within 1e-12                               |Δ|=0
PASS  Gate E1.2: Fejér preserves nonnegativity (min >= -1e-12)                   min=0.00261178
PASS  Gate E1.3: illegal produces negative undershoot (<= -eps^2)                eps^2=0.0333333
PASS  Gate E1.4: illegal increases variation (TV) by >= (1+eps)                  eps=0.182574
PASS  Gate E1.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate E2.1: unitary norm drift <= 1e-10                                     drift=8.881784e-16
PASS  Gate E2.2: signed illegal distortion >= (1+eps)×fejer                      eps=0.182574
PASS  Gate E2.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate 60A.L0_tiers_verified
PASS  Gate 60A.L1_E1_C_stable
PASS  Gate 60A.L2_E2_C_stable
PASS  Gate 60A.L3_visibility_stable
PASS  Gate 60B.G1_truth_reversible
PASS  Gate 60B.G2_illegal_breaks
PASS  Gate 60B.T_counterfactual
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
DETERMINISM HASH
====================================================================================================
determinism_sha256: 5ae18adb1184e4e7cdc95edddbff85dc0eb3c852ae723521239d840c24933048

====================================================================================================
FINAL VERDICT
====================================================================================================
PASS  DEMO-60 VERIFIED (selection + admissibility + quantum suite + ladder + time-reversal + PDE benchmark)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
