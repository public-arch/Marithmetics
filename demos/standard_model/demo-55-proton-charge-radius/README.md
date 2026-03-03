# DEMO-55 — Proton Charge Radius (from substrate selection)

> **Claim:** A deterministic first-principles audit selects the unique SCFP triple (wU, s2, s3) = (137, 107, 103), derives αₛ(MZ) = 2/q₃, computes Λ₅ via 2-loop MS-bar scheme, and maps to proton rms charge radius r_p = 0.8430 fm via fixed dressing r_p = (ℏc/Λ₅) × √(1/(1+αₛ)), achieving <1% agreement with experimental r_p = 0.84075 fm with no tuning.

---

## What this demo computes

A zero-knob, zero-tuning, first-principles audit:
- Deterministic SCFP triple selection in primary prime window (97..180) → unique (wU, s2, s3) = (137, 107, 103).
- Derive αₛ(MZ) = 2/q₃ = 2/17 from the selected triple.
- QCD scale Λ₅ via fixed 2-loop MS-bar scheme (nf = 5, μ = MZ = 91.03 GeV): Λ₅(2-loop) ≈ 0.2214 GeV.
- Proton rms charge radius mapping: r_p = (ℏc/Λ₅) × √(1/(1+αₛ(MZ))) = 0.8430 fm.
- Evaluation-only comparison with CODATA 2022 reference r_p = 0.84075 fm; relative error +0.27%.
- Counterfactual admissible triples (alternative windows) all fail radius gate (ratio > 1.2 or < 0.8).

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Primary triple must equal (137, 107, 103).
5. Primary proton radius must be within 1% of reference (rel_err ≤ 0.01).
6. All counterfactual triples must miss radius band [0.8, 1.2] × r_p_ref (strong_misses ≥ 6/6).

## Controls

- **Illegal operators:** None tested (this is a QCD scale audit, not a field-solver demo).
- **Counterfactuals:** Alternative triples from extended window (277, 263, 239), (307, 263, 239), (311, 263, 239), etc.; all must miss radius band [0.8×r_p_ref, 1.2×r_p_ref].
- **Ablations:** 1-loop Λ₅ as comparison (expected to overestimate r_p by factor ~2.58).

## Dependencies

Python 3.9+ with standard library only (no third-party packages).

## Run

```bash
python demo.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window  count=1
PASS  Primary proton radius within 1% (evaluation-only gate)  rel_err=+0.269275%
PASS  DEMO 55 VERIFIED (selection + proton radius + counterfactual ablation)
```

## Reference checkpoints (from provided transcript)

- Spec SHA256: `0a46a38ce93a14d9770b8b7a77aca810eb497943d0a4cafef7b7d069454ba1c1`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)  selected=(137, 107, 103)
Admissible triple: (137, 107, 103)
Spec SHA256: 0a46a38ce93a14d9770b8b7a77aca810eb497943d0a4cafef7b7d069454ba1c1
alpha_s(MZ) = 2/q3    = 0.117647058823529
STAGE 3 - LAMBDA_QCD FROM alpha_s(MZ) (NO PDG FIT)
alpha_s(MZ)              = 0.117647058823529
Lambda_5 (1-loop, nf=5)  = 0.0858720264356 GeV
Lambda_5 (2-loop, nf=5)  = 0.221411095276 GeV
PASS  Sanity: Lambda_5(2-loop) in expected ballpark  Lambda5=0.221411
r_p = (hbar*c / Lambda_5) * sqrt(1 / (1 + alpha_s(MZ)))
r_p(1-loop Lambda_5) = 2.17361398 fm
r_p(2-loop Lambda_5) = 0.8430139282 fm
ref r_p              = 0.84075 fm
(277, 263, 239)  alpha_s=0.028985507  Lambda5=1.89784e-10 GeV  r_p=1.025e+09 fm  ratio=1.21914e+09  MISS
(307, 263, 239)  alpha_s=0.013071895  Lambda5=2.72683e-25 GeV  r_p=7.18965e+23 fm  ratio=8.55147e+23  MISS
(311, 263, 239)  alpha_s=0.012903226  Lambda5=1.20638e-25 GeV  r_p=1.62525e+24 fm  ratio=1.93309e+24  MISS
```

Transcript excerpt (for quick visual diff):

```text
(409, 263, 239)  alpha_s=0.039215686  Lambda5=2.76483e-07 GeV  r_p=700109 fm  ratio=832720  MISS
(409, 367, 239)  alpha_s=0.039215686  Lambda5=2.76483e-07 GeV  r_p=700109 fm  ratio=832720  MISS
(409, 367, 307)  alpha_s=0.039215686  Lambda5=2.76483e-07 GeV  r_p=700109 fm  ratio=832720  MISS
PASS  All counterfactuals miss outside fixed ratio band  strong_misses=6/6  band=(0.8, 1.2)

====================================================================================================
                                           FINAL VERDICT                                            
====================================================================================================
PASS  DEMO 55 VERIFIED (selection + proton radius + counterfactual ablation)

Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
