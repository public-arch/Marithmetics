# DEMO-56 — Deterministic Operator Calculus (vs classical finite differences)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-56-deterministic-operator-calculus-vs-classical-finite-differences.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-56' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
Errors vs truth (normalized L2): fejer=0.5451457656 sharp=1.153051488 signed=1.153051488
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-56 — Deterministic Operator Calculus vs Classical Finite Differences

> This single script is designed to be:
>   - Self-contained (NumPy + standard library only)
>   - Deterministic (no tolerance-driven inner iterations; fixed-step updates)
>   - Falsifiable (explicit pass/fail gates and counterfactual controls)
>   - Referee-ready (no internal jargon; first-principles explanations in output)

> What it does (high level)
> 1) Selects a unique integer triple (wU, s2, s3) by a deterministic rule.
> 2) Derives a small set of invariants (q2, q3, v2, eps).
> 3) Uses those invariants to deterministically set numerical budgets (N, K, dt, steps).
> 4) Runs worked examples showing:
>    - Why admissible kernels (Fejér averaging) prevent non-physical oscillations
>    - Why non-admissible kernels (sharp truncation / signed filters) fail controls
>    - Why the budgets matter (counterfactual triples degrade by a fixed margin)
> 5) Optionally runs an industrial-scale 3D Navier–Stokes certificate (Taylor–Green vortex).

> Default mode is intentionally fast ("smoke-tier") and should run on laptops.
> For the industrial certificate (N=256), use:  --tier industrial

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-56-deterministic-operator-calculus-vs-classical-finite-differences.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                              count=1
PASS  DEMO-56 VERIFIED (executed gates pass; counterfactual controls degrade)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `028a6c5d4aeccc4e5b3d0516cd47ab5cbaa5b1f33d54814d72a4c4e06b1c9f38`

- Verdict line: `PASS  DEMO-56 VERIFIED (executed gates pass; counterfactual controls degrade)`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                            selected=Triple(wU=137, s2=107, s3=103)
Derived invariants (from the selected triple):
determinism_sha256: 028a6c5d4aeccc4e5b3d0516cd47ab5cbaa5b1f33d54814d72a4c4e06b1c9f38
========================================== FINAL VERDICT ===========================================
```

Transcript excerpt (for quick visual diff):

```text
}
Total work proxy ≈ 1.050e+13

========================================= DETERMINISM HASH =========================================
determinism_sha256: 028a6c5d4aeccc4e5b3d0516cd47ab5cbaa5b1f33d54814d72a4c4e06b1c9f38

========================================== FINAL VERDICT ===========================================
PASS  DEMO-56 VERIFIED (executed gates pass; counterfactual controls degrade)
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
