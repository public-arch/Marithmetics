# DEMO-51 — QFT+GR Vacuum Suppression (first-principles closure)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-51-qft-gr-vacuum-suppression-first-principles-closure.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-51' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
ratio rho_pred/rho_obs = 0.990650067403
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO 51 — QFT+GR VACUUM SUPPRESSION (MASTER, FIRST-PRINCIPLES, <1% CLOSURE)
> Stdlib-only. CLI-only. No file writes.

> Executive purpose
> A single end-to-end demonstration that supports the GUM report with:

>   1) Deterministic discrete selection (fixed rules, fixed window) yielding a unique
>      admissible triple (wU, s2, s3) = (137, 107, 103).
>   2) Explicit linkage to all three integers:
>        - wU drives alpha0_inv and the odd-part invariant q3
>        - s2 enters via q2 = wU - s2 and sin^2(thetaW) = 7/q2
>        - s3 enters via a consistency check on the active-flavor branch count
>          (derived_nf = 3 + v2(s2-1) + v2(s3-1)), which matches nf=5 at the MZ scale.
>   3) QCD scale extraction: infer Λ5 from alpha_s(MZ)=2/q3 using 2-loop running (numeric inversion).
>   4) Mechanism-grade induced vacuum term (EFT/QFT+GR) with derived loop geometry:
>        rho_pred = (1/(16π^2))^2 * (1/(1+alpha_s(MZ))) * Λ5^6 / M_Pl^2
>      No continuous parameters are tuned.
>   5) <1% agreement with rho_Lambda(obs) computed from (H0, ΩΛ) (evaluation-only).
>   6) Correct robustness tests (no invalid “Λ3 must match Λ5” claim):
>        - μ-sweep at fixed Λ5 (renormalization-scale dependence)

## Run instructions

- Dependencies: Stdlib-only (no third-party packages).

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-51-qft-gr-vacuum-suppression-first-principles-closure.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                                      count=1
```

## Reference checkpoints (from provided transcript)

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
Admissible triples: [(137, 107, 103)]
PASS  Selected triple equals (137,107,103)                                            selected=(137, 107, 103)
sin^2(thetaW) := 7/q2        = 0.233333333333333
alpha_s(MZ) := 2/q3          = 0.117647058823529
rho = (1/(16π^2))^2 * (1/(1+alpha_s)) * Λ5^6 / M_Pl^2
- 1/(1+alpha_s) is a minimal RG dressing (no free constants)
rho_pred [GeV^4]       = 2.83595269166448e-47
ratio rho_pred/rho_obs = 0.990650067403
μ=45.5175 GeV   alpha_s(μ)=0.13139832   ratio=0.978609491317   |ratio-1|=0.021390509
μ=91.0349 GeV   alpha_s(μ)=0.11764706   ratio=0.990650067403   |ratio-1|=0.0093499326
μ=182.07 GeV   alpha_s(μ)=0.10656011   ratio=1.00057568013   |ratio-1|=0.00057568013
recovered alpha_s(MZ) = 0.117647058824
PASS  Threshold audit recovers alpha_s(MZ) (consistency)                              Δ=1.39e-17
(277, 263, 239)  alpha_s=0.0289855  Λ5=1.89784e-10  ratio=4.26755e-55  PASS
(277, 263, 307)  alpha_s=0.0289855  Λ5=1.89784e-10  ratio=4.26755e-55  PASS
(307, 263, 239)  alpha_s=0.0130719  Λ5=2.72683e-25  ratio=3.81365e-144  PASS
```

Transcript excerpt (for quick visual diff):

```text
W := q2/q3 = 1.76470588235
k_struct   = 20
k_eff      = 20.672189
PASS  k_eff approximately equals k_struct (within 1)                                

====================================================================================================
FINAL VERDICT
====================================================================================================
PASS  Verified (accuracy + robustness + ablations) under declared mechanism         

Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
