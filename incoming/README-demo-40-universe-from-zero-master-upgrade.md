# DEMO-40 — Universe-from-Zero (master upgrade)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-40-universe-from-zero-master-upgrade.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-40' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  Gate P: primary equals (137,107,103)                                           selected=(137,107,103)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-40 — Universe-from-Zero

> This is a unified "master upgrade" that preserves the audit-grade determinism of
> the original DEMO-40 while integrating the deeper, first-principles framing
> associated with the MARI-style master upgrade.

> Design goals (strict)
> 1) Deterministic: no stochastic inputs, no external data, no tuning.
> 2) Portable: Python + NumPy only; optional file write is best-effort.
> 3) Referee-facing: explicit stages, gates, illegal controls, and falsifiers.
> 4) Non-regression: keeps *all* DEMO-40 verified components and restores suite-wide
>    invariant definitions (q2, q3, eps, budgets) consistent with the flagship line.

> What is being demonstrated (narrow claim)
> From a finite arithmetic substrate (primes + residue filters + 2-adic coherence),
> a single triple of primes is recovered in a predeclared window:

>     (wU, s2, s3) = (137, 107, 103)

> Then we show:

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-40-universe-from-zero-master-upgrade.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Gate P: primary equals (137,107,103)                                           selected=(137,107,103)
PASS  Gate F: absorbing fixed point (idempotent eliminators)
PASS  Gate CF: captured >=4 deterministic counterfactual triples                     found=4 window=(181, 1200)
PASS  Gate A: PhiAlpha normalization (2/q3)*q3 == 2                                  PhiAlpha=2.000000000000
PASS  Gate R: all residue-from-digits hats match integer residues (all bases, all q)
PASS  Gate G1: triple + pools invariant across bases (encode/decode audit)
PASS  Gate G2: digit-dependent path is not portable                                  freq=0.273 (<0.50 expected)
PASS  Gate R0: variant scan executed (count)                                         total=5832
PASS  Gate R1: at least one variant reproduces primary triple (sanity)
PASS  Gate R2: uniqueness is not generic                                             unique_frac=0.037
PASS  Gate R3: primary is not ubiquitous                                             hit_frac=0.037
PASS  Gate R4: no multi-triple variants (rigidity)                                   multi=0
PASS  Gate S9: >=3/4 counterfactuals fail plausibility gates (teeth)                 fail=4/4
PASS  Gate K1: Fejer kernel nonnegative (admissible)                                 kmin=0.000e+00
PASS  Gate K2: sharp cutoff has negative lobes (illegal)                             kmin=-1.053e-01
PASS  Gate K3: signed HF injector has negative lobes (illegal)                       kmin=-2.107e-01
PASS  Gate C1: Hilbert/DFT round-trip + Parseval consistency                         rt_err=2.578e-16 norm_err=0.000e+00
PASS  Gate C2: Quantum2D Fejer density nonnegative                                   min=1.982e-02
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `b359a4a46dcef4d8f6f42f5c5efa13fe9be4450baf305de55885fd80bfa2a936`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Gate P: primary equals (137,107,103)                                           selected=(137,107,103)
PASS  Gate F: absorbing fixed point (idempotent eliminators)
PASS  Gate CF: captured >=4 deterministic counterfactual triples                     found=4 window=(181, 1200)
PASS  Gate A: PhiAlpha normalization (2/q3)*q3 == 2                                  PhiAlpha=2.000000000000
PASS  Gate R: all residue-from-digits hats match integer residues (all bases, all q)
PASS  Gate G1: triple + pools invariant across bases (encode/decode audit)
PASS  Gate G2: digit-dependent path is not portable                                  freq=0.273 (<0.50 expected)
PASS  Gate R0: variant scan executed (count)                                         total=5832
PASS  Gate R1: at least one variant reproduces primary triple (sanity)
PASS  Gate R2: uniqueness is not generic                                             unique_frac=0.037
PASS  Gate R3: primary is not ubiquitous                                             hit_frac=0.037
PASS  Gate R4: no multi-triple variants (rigidity)                                   multi=0
PASS  Gate S9: >=3/4 counterfactuals fail plausibility gates (teeth)                 fail=4/4
PASS  Gate K1: Fejer kernel nonnegative (admissible)                                 kmin=0.000e+00
PASS  Gate K2: sharp cutoff has negative lobes (illegal)                             kmin=-1.053e-01
PASS  Gate K3: signed HF injector has negative lobes (illegal)                       kmin=-2.107e-01
```

Transcript excerpt (for quick visual diff):

```text
core_sha256: 1436dab79ed74b8d1f248827d3adbc1e3b7901ccb8ce82797cd961fd29d97191
full_sha256: 1ee6920e997dd65a6f8023e0622e0c1eb34fbb7594a14d932a05b6645e0f4f45
determinism_sha256: b359a4a46dcef4d8f6f42f5c5efa13fe9be4450baf305de55885fd80bfa2a936


================================================================================================
FINAL VERDICT
================================================================================================
PASS  DEMO-40 MASTER UPGRADE VERIFIED (determinism + invariance + rigidity + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
