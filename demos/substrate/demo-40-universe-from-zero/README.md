# DEMO-40 — Universe-from-Zero (master upgrade)

> **Claim:** From a finite arithmetic substrate (primes + residue filters + 2-adic coherence), a deterministic rule recovers a unique triple (wU, s2, s3) = (137, 107, 103) in a predeclared window, with counterfactual triples failing plausibility gates and proving uniqueness is not generic.

---

## What this demo computes

From a finite arithmetic substrate (primes + residue filters + 2-adic coherence):
- Deterministic selection of the unique admissible triple (wU, s2, s3) = (137, 107, 103) via lane rules and congruence filters.
- Absorbing fixed point (explicit elimination chain, idempotent).
- Base-gauge invariance audit (encode/decode consistency across bases).
- Rosetta/DRPT residue reconstruction from digits (base-independent residues).
- Uniqueness: predeclared neighborhood scan confirms uniqueness is not generic (unique_frac ≈ 0.037).
- Causality capstones: Hilbert/DFT round-trip + Parseval consistency; Fejer kernel nonnegative (admissible) vs illegal controls.
- Deterministic structural cosmology capsule (BB-36 monomials) with counterfactual teeth.

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Primary triple must equal (137, 107, 103).
5. Counterfactual triples must fail plausibility gates (Gate S9: ≥3/4 fail).
6. Rigidity gate (Gate R4) must show zero multi-triple variants.

## Controls

- **Illegal operators:** Gate K2 (sharp cutoff has negative lobes), Gate K3 (signed HF injector has negative lobes).
- **Counterfactuals:** Variant scan (5832 total variants tested); counterfactual triples chosen by same deterministic rules in larger window.
- **Ablations:** Rigidity scan (Gate R4: no multi-triple variants); portability test (Gate G2: digit-dependent path is not portable).

## Dependencies

Python 3.10+ with NumPy.

## Run

```bash
python demo.py
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
