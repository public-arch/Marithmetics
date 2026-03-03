# DEMO-63 — Gravitational-Wave Inspiral Phasing (observable vector + teeth)

> **Claim:** Deterministic discrete selector yields unique triple (wU, s2, s3) = (137, 107, 103), from which eps-margin and dimensionless amplitude A are derived with no tuning degrees of freedom; leading-order inspiral phasing integral with fixed exponent p = 5/3 produces observable vector with ||v_P|| = 60755; counterfactual triples from extended window miss this vector by ≥eps in ≥11/12 cases.

---

## What this demo computes

A self-contained deterministic audit:
- Deterministic discrete selector: unique triple (wU, s2, s3) = (137, 107, 103) in primary window via explicit congruence rules.
- Fixed invariants: eps-margin eps = 1/√q2 ≈ 0.1826; dimensionless amplitude A (no tuning degrees of freedom).
- Inspiral phasing observable vector:
  - Leading-order inspiral phasing integral with fixed power-law exponent p = 5/3 (frequency dependence ~ f⁻⁵/³).
  - Observable vector components derived from triple-dependent constants (no external fits).
  - Vector magnitude ||v_P|| ≈ 60755.4.
- Gate G1: vector is finite and nonzero.
- Gate T: counterfactual triples (≥3/4 out of 12) miss primary vector by ≥eps in L2 norm (strong = 11/12).
- Counterfactual selection: same deterministic rules applied to extended window → alternative admissible triples; all tested.

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Primary triple must equal (137, 107, 103).
5. Observable vector must be finite and nonzero (||v_P|| > 0).
6. Counterfactual triples must miss primary vector by ≥eps in ≥3/4 cases (vector L2 distance).
7. No runtime tuning: all parameters fixed by triple selection.

## Controls

- **Illegal operators:** None tested (this is a discrete selector + observable vector audit, not a field-solver demo).
- **Counterfactuals:** Alternative triples from extended window; must miss primary vector by ≥eps in L2 norm (≥3/4 cases).
- **Ablations:** Fixed exponent p = 5/3 cannot be changed (locked by leading-order physics).

## Dependencies

Python 3.10+ with NumPy.

## Run

```bash
python demo.py
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
