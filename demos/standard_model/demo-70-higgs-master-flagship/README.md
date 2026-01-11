# DEMO-70 — Higgs Master Flagship (fixed-point mass closure)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-70-higgs-master-flagship-fixed-point-mass-closure.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-70' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
Best mode: d=13 with mH≈123.881359 (|Δ|=1.118641 GeV)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-70 — HIGGS MASTER FLAGSHIP

> This flagship demo is the *single* integrated run that combines:

>   • PREWORK 70A: Exact EW rational locks + lawful "dressed" closure
>                 + illegal control separation + counterfactual teeth.

>   • PREWORK 70B: UV critical edge λ* (solve λ(μ_max)≈0)
>                 + truth tier vs budget tier + illegal controls + teeth.

>   • PREWORK 70C: Mode-ladder / SU(2) lock: best mode d=13
>                 + illegal control + counterfactual budget teeth.

> Goal: a maximally clear, deterministic, referee-facing certificate.
> No fits. No hidden knobs. Everything is fixed by the deterministic triple.

> Outputs:
>   • stdout (primary)
>   • optional JSON + PNG if --write is passed (safe: failures are caught)

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-70-higgs-master-flagship-fixed-point-mass-closure.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Gate S1: primary equals (137,107,103)
PASS  Gate S2: captured >=4 counterfactuals                                    found=4
PASS  Gate A1: lock-gates exact (Theta=4/15, sin^2θW=7/30, alpha0=1/137, alpha_s=2/17)
PASS  Gate A2: plausibility (iters<=250, v∈[200,400], alpha(MZ)∈[0.0075,0.0083], MZ∈[80,100]) iters=44 v=246.286 alpha=0.007800 MZ=91.155
PASS  Gate A3: illegal control is worse (dist_illegal > dist_lawful)
PASS  Gate A4: counterfactual teeth (>=3/4 CF out of [80,100])                 out=4/4
PASS  Gate B1: lambda* in sane band [0.1,0.3]                                  lambda*=0.173398
PASS  Gate B2: primary budget reproduces truth within eps^3                    err=1.144e-05 tol=eps^3=6.086e-03
PASS  Gate B3: illegal controls worse than primary                             err_il/err_p=31.17 res_ratio=8.21e+03
PASS  Gate B4: counterfactual budget degrades by (1+eps)                       err_cf=4.005e-05 err_p=1.144e-05
PASS  Gate C1: best mode is d=13 (SU(2) lock)                                  best_d=13 |Δ|=1.119
PASS  Gate C2: illegal is worse than lawful best by (1+eps)                    best_il_d=15 Δ_law=1.119 Δ_il=1.857
PASS  Gate C3: counterfactual budget degrades by (1+eps)                       Δ_cf=5.365 Δ_law=1.119 eps=0.183
PASS  DEMO-70 VERIFIED (Higgs master flagship)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `29df9e06c202ad60ac47783a1d0d41a69fcab795325a9f4a28a45c47d199ba12`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Gate S1: primary equals (137,107,103)
PASS  Gate S2: captured >=4 counterfactuals                                    found=4
sin^2θW = 7/30 ≈ 0.233333333333
alpha_s = 2/17 ≈ 0.117647058824
PASS  Gate A1: lock-gates exact (Theta=4/15, sin^2θW=7/30, alpha0=1/137, alpha_s=2/17)
PASS  Gate A2: plausibility (iters<=250, v∈[200,400], alpha(MZ)∈[0.0075,0.0083], MZ∈[80,100]) iters=44 v=246.286 alpha=0.007800 MZ=91.155
PASS  Gate A3: illegal control is worse (dist_illegal > dist_lawful)
PASS  Gate A4: counterfactual teeth (>=3/4 CF out of [80,100])                 out=4/4
PASS  Gate B1: lambda* in sane band [0.1,0.3]                                  lambda*=0.173398
PASS  Gate B2: primary budget reproduces truth within eps^3                    err=1.144e-05 tol=eps^3=6.086e-03
PASS  Gate B3: illegal controls worse than primary                             err_il/err_p=31.17 res_ratio=8.21e+03
PASS  Gate B4: counterfactual budget degrades by (1+eps)                       err_cf=4.005e-05 err_p=1.144e-05
d=16  λ0=0.062500  mH_fp=112.5922  |Δ|=12.408  it=17
d=15  λ0=0.066667  mH_fp=115.9006  |Δ|= 9.099  it=18
d=14  λ0=0.071429  mH_fp=119.6347  |Δ|= 5.365  it=18
d=13  λ0=0.076923  mH_fp=123.8814  |Δ|= 1.119  it=19 <-- best
```

Transcript excerpt (for quick visual diff):

```text
STAGE 6 — Determinism hash + score + optional artifacts
----------------------------------------------------------------------------------------------------
determinism_sha256: 29df9e06c202ad60ac47783a1d0d41a69fcab795325a9f4a28a45c47d199ba12
presentation_score: 1000000 / 1,000,000

----------------------------------------------------------------------------------------------------
FINAL VERDICT
----------------------------------------------------------------------------------------------------
PASS  DEMO-70 VERIFIED (Higgs master flagship)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
