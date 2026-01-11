# DEMO-63 — Gravitational-Wave Inspiral Phasing (observable vector + teeth)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-63-gravitational-wave-inspiral-phasing-observable-vector-teeth.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-63' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  Gate T: >=3/4 counterfactuals miss by eps (vector L2)  strong=11/12  eps=0.182574185835
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-63 — Gravitational-Wave Inspiral Phasing (Deterministic Amplitude + Counterfactual Controls)
> MASTER FLAGSHIP — REFEREE-READY, FIRST-PRINCIPLES, DETERMINISTIC

> What this demo *is*:
>   A self-contained, deterministic audit that starts from an explicitly declared discrete selector
>   (over integer candidates in a primary window), derives *fixed* invariants, and propagates them
>   into a physically motivated inspiral-phasing observable vector.

> What this demo *is not*:
>   - A tuned fit to observational data.
>   - A runtime benchmark.
>   - A full numerical-relativity waveform generator.

> Referee-facing claims (all testable inside this single script):
>   (C1) The primary-window selector is deterministic and yields a unique triple (wU, s2, s3).
>   (C2) From that triple, we derive an eps-margin (eps = 1/sqrt(q2)) and a dimensionless amplitude A,
>        with no degrees of freedom left for tuning.
>   (C3) A leading-order inspiral phasing integral implies a frequency-power-law dependence
>        ~ f^{-5/3}; we use this as the fixed exponent p = 5/3.
>   (C4) Counterfactual triples, chosen by the same deterministic rules in a larger window, change the

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-63-gravitational-wave-inspiral-phasing-observable-vector-teeth.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Gate G1: primary vector finite and nonzero  ||vP||=60755.4
PASS  Gate T: >=3/4 counterfactuals miss by eps (vector L2)  strong=11/12  eps=0.182574185835
PASS  DEMO-63 VERIFIED (selection + first-principles observable vector + teeth)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `14e8d64760f3e69244293081d739d1f51b01ab583e5ade20711b9437155d3443`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)  selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate G1: primary vector finite and nonzero  ||vP||=60755.4
PASS  Gate T: >=3/4 counterfactuals miss by eps (vector L2)  strong=11/12  eps=0.182574185835
determinism_sha256: 14e8d64760f3e69244293081d739d1f51b01ab583e5ade20711b9437155d3443
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
DETERMINISM HASH
==================================================================================================
determinism_sha256: 14e8d64760f3e69244293081d739d1f51b01ab583e5ade20711b9437155d3443

==================================================================================================
FINAL VERDICT
==================================================================================================
PASS  DEMO-63 VERIFIED (selection + first-principles observable vector + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
