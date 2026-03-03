# DEMO-60 — Quantum Master Flagship (unitarity + controls)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) yields budgets (N, K, eps); admissible Fejér coarse-graining is positivity-preserving and mass-conserving on quantum densities, while illegal controls (sharp/signed) create negative undershoot, higher variation, and unitarity violation; counterfactual triples and cross-resolution ladder tiers degrade by fixed eps margin.

---

## What this demo computes

Deterministic referee-ready, first-principles audit:
- Deterministic selection: unique prime triple (wU, s2, s3) = (137, 107, 103) via fixed rules.
- Budget derivation: eps = 1/√q2 ≈ 0.1826; N, K from triple invariants.
- Operator admissibility: Fejér spectral multiplier → nonnegative real-space kernel (positivity-preserving); sharp/signed → negative kernel lobes.

Example E1 (Density admissibility on discontinuous top-hat):
- Fejér preserves mass (|Δ| ≤ 1e-12) and nonnegativity (min ≥ -1e-12).
- Illegal controls create negative undershoot (≤ -eps²).
- Counterfactual budget reduction degrades TV variation by (1+eps).

Example E2 (Double-slit interference density):
- Unitary spectral evolution: norm drift ≤ 1e-10.
- Illegal control distortion ≥ (1+eps) × Fejér.
- Counterfactual budgets degrade distortion by (1+eps).

PREWORK 60A (Cross-resolution ladder):
- Two tiers (N=256, N=512) jointly stable under Parseval-like scaling invariant C = distortion × √K.

PREWORK 60B (Time-reversal stress test):
- Forward-backward reversibility: return to initial state within machine precision.
- Illegal operator breaks reversibility materially.
- Counterfactual budgets degrade lawful distortion.

PREWORK 60C (Quantum PDE: free Schrödinger):
- Truth: exact spectral phase evolution.
- Baseline FD2: finite-difference Laplacian + Crank-Nicolson stepping (expected to underperform).
- Gate: FD baseline density error ≥ (1+eps) × Fejér measurement distortion.
- Illegal filters distort more; counterfactual budgets degrade.

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Fejér mass preservation: |Δ| = 0.
5. Fejér nonnegativity: min ≥ -1e-12.
6. Illegal undershoot: undershoot ≤ -eps².
7. Illegal TV increase: ≥ (1+eps) × Fejér.
8. All counterfactual tiers must degrade by (1+eps).
9. Reversibility gates must pass (time-reversal error recovery).
10. FD baseline must be worse than Fejér by (1+eps).

## Controls

- **Illegal operators:** Sharp spectral cutoff (negative kernel lobes), signed HF complement (stronger lobes); tested in all examples (E1-E2, ladder, time-reversal, PDE).
- **Counterfactuals:** Alternative triples with same selection rules, different budgets (N tier variations, ladder tiers N=256/512); must degrade by (1+eps).
- **Ablations:** Higher-budget Fejér as truth; FD2 baseline as PDE ablation.

## Dependencies

Python 3.10+ with NumPy.

## Run

```bash
python demo.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                                 count=1
PASS  Gate E1.1: Fejér preserves mass within 1e-12                               |Δ|=0
PASS  Gate E1.2: Fejér preserves nonnegativity (min >= -1e-12)                   min=0.00261178
PASS  Gate E1.3: illegal produces negative undershoot (<= -eps^2)                eps^2=0.0333333
PASS  Gate E1.4: illegal increases variation (TV) by >= (1+eps)                  eps=0.182574
PASS  Gate E1.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate E2.1: unitary norm drift <= 1e-10                                     drift=8.881784e-16
PASS  Gate E2.2: signed illegal distortion >= (1+eps)×fejer                      eps=0.182574
PASS  Gate E2.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate 60A.L0_tiers_verified
PASS  Gate 60A.L1_E1_C_stable
PASS  Gate 60A.L2_E2_C_stable
PASS  Gate 60A.L3_visibility_stable
PASS  Gate 60B.G1_truth_reversible
PASS  Gate 60B.G2_illegal_breaks
PASS  Gate 60B.T_counterfactual
PASS  Tier N=256 Gate 60C.G1_fd_worse_than_fejer
PASS  Tier N=256 Gate 60C.G2_illegal_distorts_more
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `5ae18adb1184e4e7cdc95edddbff85dc0eb3c852ae723521239d840c24933048`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                               selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate E1.1: Fejér preserves mass within 1e-12                               |Δ|=0
PASS  Gate E1.2: Fejér preserves nonnegativity (min >= -1e-12)                   min=0.00261178
PASS  Gate E1.3: illegal produces negative undershoot (<= -eps^2)                eps^2=0.0333333
PASS  Gate E1.4: illegal increases variation (TV) by >= (1+eps)                  eps=0.182574
PASS  Gate E1.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate E2.1: unitary norm drift <= 1e-10                                     drift=8.881784e-16
PASS  Gate E2.2: signed illegal distortion >= (1+eps)×fejer                      eps=0.182574
PASS  Gate E2.T: >=3/4 counterfactuals increase distortion by (1+eps)            strong=4/4 eps=0.182574
PASS  Gate 60A.L0_tiers_verified
PASS  Gate 60A.L1_E1_C_stable
PASS  Gate 60A.L2_E2_C_stable
PASS  Gate 60A.L3_visibility_stable
PASS  Gate 60B.G1_truth_reversible
PASS  Gate 60B.G2_illegal_breaks
PASS  Gate 60B.T_counterfactual
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
DETERMINISM HASH
====================================================================================================
determinism_sha256: 5ae18adb1184e4e7cdc95edddbff85dc0eb3c852ae723521239d840c24933048

====================================================================================================
FINAL VERDICT
====================================================================================================
PASS  DEMO-60 VERIFIED (selection + admissibility + quantum suite + ladder + time-reversal + PDE benchmark)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
