# DEMO-37 — Math × Standard Model Master Flagship

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-37-math-standard-model-master-flagship.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-37' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
Primary vector summary: alpha0_inv=137, alpha_s=0.117647, alpha_inv(MZ)=128.148
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-37 — MATH×SM MASTER FLAGSHIP

> What this demo is
> A deterministic, self-contained computational exhibit that:

> 1) Selects a single primary "triple" of integers (wU, s2, s3) by explicit rules
>    (prime windows + modular residue filters + Euler-phi density floors + coherence).
>    Primary (unique) triple:
>         (wU, s2, s3) = (137, 107, 103)

> 2) Enforces *operator admissibility* using a Fejér kernel (nonnegative in real space),
>    and contrasts it with two illegal controls:
>       - sharp cutoff kernel (Dirichlet; negative lobes),
>       - signed control (injects high-frequency energy and negative lobes).

> 3) Builds Standard-Model–adjacent observables *from first principles and without tuning*:
>       - alpha0^{-1}         := wU
>       - alpha_s(MZ)         := 2 / q3, where q3 = odd_part(wU-1)
>       - Lambda_QCD (2-loop) from alpha_s(MZ)
>       - QED running alpha^{-1}(MZ) with *confinement-floor thresholds*

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-37-math-standard-model-master-flagship.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                           count=1
PASS  Gate B1: encode/decode invariance across bases                             bases=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
PASS  Gate S0: structural sanity (q2>0, q3>0, v2U matches coherence)               q2=30 q3=17 v2U=3
PASS  Gate K1: Fejér kernel is nonnegative (admissible)                         kmin=0.000e+00
PASS  Gate K2: illegal kernels have negative lobes (sharp + signed)            kmin_sharp=-1.053e-01 kmin_signed=-3.181e-01
PASS  Gate K3: signed control injects HF beyond eps^2 floor                     hf_signed=1.000 floor=0.033
PASS  Gate A1: lawful prediction matches reference within derived tolerance       |Δ|=0.192580
PASS  Gate A2: illegal model violates closure by an eps-derived margin             |Δ_illegal|=1.410299
PASS  Gate M1: mean relative error <= eps^3                                      mean=5.996e-05 eps^3=6.086e-03
PASS  Gate T: >=3/4 counterfactuals miss by rel_dist>=eps                           strong=6/6 eps=0.182574
PASS  DEMO-37 VERIFIED (selection + base invariance + admissibility + gauge/QED/QCD + math + teeth)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `9da2bedcf1208fcbe0760ccb78483ebd333bf17bb24e49daf68850cb6765b477`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                         selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate B1: encode/decode invariance across bases                             bases=[2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]
PASS  Gate S0: structural sanity (q2>0, q3>0, v2U matches coherence)               q2=30 q3=17 v2U=3
PASS  Gate K1: Fejér kernel is nonnegative (admissible)                         kmin=0.000e+00
PASS  Gate K2: illegal kernels have negative lobes (sharp + signed)            kmin_sharp=-1.053e-01 kmin_signed=-3.181e-01
PASS  Gate K3: signed control injects HF beyond eps^2 floor                     hf_signed=1.000 floor=0.033
alpha_s(MZ):= 2/q3 = 0.117647058824  (q3=17)
PASS  Gate A1: lawful prediction matches reference within derived tolerance       |Δ|=0.192580
PASS  Gate A2: illegal model violates closure by an eps-derived margin             |Δ_illegal|=1.410299
PASS  Gate M1: mean relative error <= eps^3                                      mean=5.996e-05 eps^3=6.086e-03
Primary vector summary: alpha0_inv=137, alpha_s=0.117647, alpha_inv(MZ)=128.148
PASS  Gate T: >=3/4 counterfactuals miss by rel_dist>=eps                           strong=6/6 eps=0.182574
determinism_sha256: 9da2bedcf1208fcbe0760ccb78483ebd333bf17bb24e49daf68850cb6765b477
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
================================================================================================
DETERMINISM HASH
================================================================================================
determinism_sha256: 9da2bedcf1208fcbe0760ccb78483ebd333bf17bb24e49daf68850cb6765b477

================================================================================================
FINAL VERDICT
================================================================================================
PASS  DEMO-37 VERIFIED (selection + base invariance + admissibility + gauge/QED/QCD + math + teeth)
Result: VERIFIED

PASS  Results JSON not written (filesystem unavailable)                      PermissionError(1, 'Operation not permitted')
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
