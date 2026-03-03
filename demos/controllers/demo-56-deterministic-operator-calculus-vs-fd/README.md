# DEMO-56 — Deterministic Operator Calculus (vs classical finite differences)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) yields invariants (q2, q3, v2, eps) that set fixed numerical budgets (N, K, dt, steps); admissible Fejér kernels prevent non-physical oscillations and achieve lower L2 error than sharp/signed illegal controls across all worked examples, with counterfactual triples degrading by fixed eps margin.

---

## What this demo computes

- Deterministic selection: unique integer triple (wU, s2, s3) via fixed congruence rules.
- Invariant derivation: q2 = wU - s2 = 30; q3 = odd_part(wU - 1) = 17; v2 = 2-adic valuation; eps = 1/√q2 ≈ 0.1826.
- Deterministic budgets: N (grid size), K (spectral cutoff), dt (time step), steps (iteration count) from triple invariants.
- Worked examples (fixed-budget, fixed-step, no tolerance loops):
  - Admissible Fejér averaging: prevents non-physical oscillations; nonnegative real-space kernel.
  - Illegal controls: sharp truncation (negative kernel lobes), signed filter (stronger negative lobes).
  - Comparison metric: normalized L2 error vs truth (higher-budget Fejér); Fejér ≈ 0.545, sharp ≈ 1.153, signed ≈ 1.153.
- Counterfactual triples: same deterministic rules, different budgets; expected to degrade by (1+eps) factor.
- Optional industrial certificate: 3D Navier-Stokes (Taylor-Green vortex) with N=256 if --tier industrial passed.

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Admissible kernel error must be ≤ illegal kernel errors in all worked examples.
5. Counterfactual budget degradation must match (1+eps) margin (±tol).
6. Fejer vs sharp/signed error ratio must be ≥1.0 (admissible strictly better or equal).

## Controls

- **Illegal operators:** Sharp spectral cutoff (kernel with negative lobes), signed high-pass filter (kernel with stronger negative lobes).
- **Counterfactuals:** Alternative triples with same selection rules but different budgets; must degrade error by (1+eps).
- **Ablations:** Higher-budget Fejér as truth reference (to avoid external data).

## Dependencies

Python 3.10+ with NumPy.

## Run

```bash
python demo.py
python demo.py --tier industrial  # For full 3D Navier-Stokes certificate (N=256)
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
