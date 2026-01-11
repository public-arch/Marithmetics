# DEMO-69 — OATB Master Flagship (Operator Admissibility Transfer Bridge)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-69-oatb-master-flagship-operator-admissibility-transfer-bridge.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-69' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
Result: VERIFIED
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-69 — OATB MASTER FLAGSHIP (Operator Admissibility Transfer Bridge)

> What this flagship does (first principles, deterministic; no tuning):
>   1) Selects the unique primary triple (137,107,103) via lane rules, plus deterministic counterfactuals.
>   2) Proves the OATB kernel contract: Fejér triangle multipliers are nonnegative, DC-preserving,
>      and exhibit the UFET near-constant K(r) witness (~2/3) across budgets.
>   3) Demonstrates sharp-transfer vs lawful-transfer on a discontinuity:
>        - lawful (Fejér) matches truth within eps and preserves nonnegativity
>        - illegal (sharp/signed) creates Gibbs overshoot and negative density
>        - counterfactual budget reduction degrades accuracy ("teeth")
>   4) Resolves a paradox pack (finite↔continuum + measure + quantum collapse) with the *same* admissible
>      operator class; illegal operators are forced to violate legality.
>   5) Shows Ω reuse across PDEs:
>        - 3D heat controller (mass preserved + HF error suppressed + better tracking)
>        - 4D heat controller (same)
>        - 4D NS-like vector-field controller (Helmholtz projection + Ω admissibility → incompressibility)
>   6) Proves cross-base invariance of the selector (Rosetta-style) and non-ubiquity via rigidity scan.

> Outputs:
>   - Full gate transcript (PASS/FAIL)

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-69-oatb-master-flagship-operator-admissibility-transfer-bridge.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
✅  Gate S1: primary equals (137,107,103)
✅  Gate S2: captured >=4 counterfactual triples                             found=4
✅  Gate K(r) contract @r=8: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=16: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=32: H_min≈1/(r+1) and DC=1
✅  Gate U1: UFET K(r) spread <= 1%                                          spread=0.570%
✅  Gate U2: mean K(r) close to 2/3 (<=2%)                                   |K-2/3|=0.001858
✅  Gate A1: Fejér kernel nonnegative (tol)                                  min=5.177e-10
✅  Gate A2: Sharp kernel has negative lobes                                 min=-3.511e-03
✅  Gate A3: Signed kernel has negative lobes                                min=-9.678e-01
✅  Gate A4: Signed kernel retains large HF weight                           hf=0.984 floor=0.250
✅  Gate T1: Fejér distance vs truth <= eps                                  dist=0.1405 eps=0.1826
✅  Gate T2: illegal filters exhibit Gibbs overshoot (Fejér does not)        ov_fejer=-0.044 ov_sharp=0.101 ov_signed=0.202 floor=eps^2=0.033
✅  Gate T3: Fejér preserves nonnegativity (tol)                             min=7.361e-03
✅  Gate T4: illegal kernels create negative density (undershoot)            floor=-eps^2=-0.033 mins=(-8.101e-02,-1.620e-01)
✅  Gate CF1: budget reduction degrades by (1+eps)                           distP=0.1405 distCF=0.3181 (1+eps)=1.183
✅  Gate Z1: Zeno partial sum close to 1                                     sum=0.999999999069 err=9.313e-10
✅  Gate G1: Grandi Cesàro close to 1/2                                      cesaro=0.500000 err=0.000e+00
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `06947e1258f6b5d3688a38c6ffff954d485e6470b12ac932eb32e60fdd4beb36`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
✅  Gate S1: primary equals (137,107,103)
✅  Gate S2: captured >=4 counterfactual triples                             found=4
✅  Gate K(r) contract @r=8: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=16: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=32: H_min≈1/(r+1) and DC=1
✅  Gate U1: UFET K(r) spread <= 1%                                          spread=0.570%
✅  Gate U2: mean K(r) close to 2/3 (<=2%)                                   |K-2/3|=0.001858
✅  Gate A1: Fejér kernel nonnegative (tol)                                  min=5.177e-10
✅  Gate A2: Sharp kernel has negative lobes                                 min=-3.511e-03
✅  Gate A3: Signed kernel has negative lobes                                min=-9.678e-01
✅  Gate A4: Signed kernel retains large HF weight                           hf=0.984 floor=0.250
✅  Gate T1: Fejér distance vs truth <= eps                                  dist=0.1405 eps=0.1826
✅  Gate T2: illegal filters exhibit Gibbs overshoot (Fejér does not)        ov_fejer=-0.044 ov_sharp=0.101 ov_signed=0.202 floor=eps^2=0.033
✅  Gate T3: Fejér preserves nonnegativity (tol)                             min=7.361e-03
✅  Gate T4: illegal kernels create negative density (undershoot)            floor=-eps^2=-0.033 mins=(-8.101e-02,-1.620e-01)
✅  Gate CF1: budget reduction degrades by (1+eps)                           distP=0.1405 distCF=0.3181 (1+eps)=1.183
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
DETERMINISM HASH
==================================================================================================
determinism_sha256: 06947e1258f6b5d3688a38c6ffff954d485e6470b12ac932eb32e60fdd4beb36

==================================================================================================
FINAL VERDICT
==================================================================================================
✅  DEMO-69 VERIFIED (OATB flagship: admissibility + transfer + paradox + Ω reuse + invariance) score=1000000/1000000  passed_weight=32.00/32.00
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
