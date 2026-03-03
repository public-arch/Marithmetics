# DEMO-65 — Continuous Lift Paradox (capstones + GR witnesses)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) yields budgets; discrete operators that appear harmless violate continuum legality (positivity, admissibility, invariants); admissible Fejér family (nonnegative kernel) preserves mass, nonnegativity, unitarity, and exhibits linear energy drift, while illegal operators create negative undershoot, TV increase, unitarity violation, and blow-up; counterfactual triples degrade all observables by fixed eps margin.

---

## What this demo computes

Deterministic first-principles audit of the “continuous lift paradox”:
- Deterministic triple selection: (wU, s2, s3) = (137, 107, 103); counterfactual triples via extended window.
- Budget derivation: eps = 1/√q2; N, K from triple invariants.

Stage 1 (Core paradox: 1D probability lift):
- Fejér preserves mass (|Δ| ≤ 1e-12) and nonnegativity (min ≥ -1e-12).
- Illegal operators create negative undershoot (≤ -eps²) and increase TV variation by (1+eps).
- Counterfactual budgets degrade distortion by (1+eps).

Stage 2 (Capstones: Hilbert, Quantum2D, Noether):
- Hilbert: FFT round-trip relative error ≤ 1e-12.
- Quantum2D: unitary norm drift ≤ 1e-10; Fejér density nonnegative; illegal creates negative (≤ -eps²).
- Noether: legal energy drift ≤ 1e-10; illegal blow-up ≥ 1e³.

Stage 3 (GR weak-field witnesses):
- Light-bending proxy: α(b) ~ 1/b affine fit; truth slope ≈ -1.03 ± eps, admissible ≈ -0.94 ± 0.35eps.
- Shapiro delay proxy: D(b) ~ ln(b) affine fit; truth R² > 0.98, admissible R² > 0.95.
- Redshift proxy: Φ(r) ~ 1/r shell means; truth R² > 0.98, admissible R² > 0.95.
- Illegal controls inject HF beyond admissible.
- Counterfactual triples degrade slopes/fits by (1+eps).

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Fejér mass preservation: |Δ| = 0.
5. Fejér nonnegativity: min ≥ -1e-12 (all stages).
6. Illegal undershoot: undershoot ≤ -eps² (Stages 1-2).
7. Illegal TV/distortion increase: ≥ (1+eps) × Fejér.
8. Unitarity: norm drift ≤ 1e-10 (admissible).
9. Energy: legal drift ≤ 1e-10; illegal blow-up ≥ 1e³.
10. GR slopes/fits near-truth within (1+eps)eps margin.
11. Counterfactual triples degrade all observables by (1+eps).

## Controls

- **Illegal operators:** Sharp spectral cutoff (negative kernel lobes), signed HF complement (stronger lobes); tested across all stages (core paradox, capstones, GR witnesses).
- **Counterfactuals:** Alternative triples with same selection rules, different budgets; must degrade by (1+eps) in all observables (mass, nonnegativity, energy, GR slopes).
- **Ablations:** Higher-budget Fejér provides truth reference; illegal controls forced to violate legality.

## Dependencies

Python 3.10+ with NumPy. Matplotlib optional (only for PNG output).

## Run

```bash
python demo.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window                             count=1
PASS  Gate P1: Fejér preserves mass within 1e-12                             |Δ|=0
PASS  Gate P2: Fejér preserves nonnegativity (min >= -1e-12)                 min=0.00261178
PASS  Gate P3: illegal produces negative undershoot (<= -eps^2)              eps^2=0.0333333
PASS  Gate P4: illegal increases variation (TV) by >= (1+eps)                eps=0.182574
PASS  Gate P.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate H1: FFT round-trip relative error <= 1e-12                        err=2.60883e-16
PASS  Gate H2: signed retains material HF energy beyond K                    hf=0.505361 floor=0.0333333
PASS  Gate Q1: unitary norm drift <= 1e-10                                   drift=1.04361e-14
PASS  Gate Q2: Fejér density nonnegative (min >= -1e-12)                     min=0.00432282
PASS  Gate Q3: illegal density negativity (<= -eps^2)                        eps^2=0.0333333
PASS  Gate Q.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate N1: legal energy drift <= 1e-10                                   drift=4.34097e-13
PASS  Gate N2: illegal blow-up >= 1e3                                        blow=3.20839e+06
PASS  Gate GR.B1: truth slope near -1 (|Δ|<=0.25)                            slope=-1.02911
PASS  Gate GR.B2: admissible slope near -1 (|Δ|<=0.35)                       slope=-0.940344
PASS  Gate GR.B3: signed illegal retains HF (>= eps^2)                       hf=0.524986 floor=0.0333333
PASS  Gate GR.S1: truth is ln(b)-like (R2>0.98)                              R2=0.999973
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `e8585756d4ada17c788259bbb12a3cf35e57b93ce8138d620f4c03a7cced6141`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                           selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate P1: Fejér preserves mass within 1e-12                             |Δ|=0
PASS  Gate P2: Fejér preserves nonnegativity (min >= -1e-12)                 min=0.00261178
PASS  Gate P3: illegal produces negative undershoot (<= -eps^2)              eps^2=0.0333333
PASS  Gate P4: illegal increases variation (TV) by >= (1+eps)                eps=0.182574
PASS  Gate P.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate H1: FFT round-trip relative error <= 1e-12                        err=2.60883e-16
PASS  Gate H2: signed retains material HF energy beyond K                    hf=0.505361 floor=0.0333333
PASS  Gate Q1: unitary norm drift <= 1e-10                                   drift=1.04361e-14
PASS  Gate Q2: Fejér density nonnegative (min >= -1e-12)                     min=0.00432282
PASS  Gate Q3: illegal density negativity (<= -eps^2)                        eps^2=0.0333333
PASS  Gate Q.T: >=3/4 counterfactuals increase distortion by (1+eps)         strong=4/4 eps=0.182574
PASS  Gate N1: legal energy drift <= 1e-10                                   drift=4.34097e-13
PASS  Gate N2: illegal blow-up >= 1e3                                        blow=3.20839e+06
PASS  Gate GR.B1: truth slope near -1 (|Δ|<=0.25)                            slope=-1.02911
PASS  Gate GR.B2: admissible slope near -1 (|Δ|<=0.35)                       slope=-0.940344
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
STAGE 7 — Determinism hash + optional artifacts
==================================================================================================
determinism_sha256: e8585756d4ada17c788259bbb12a3cf35e57b93ce8138d620f4c03a7cced6141

==================================================================================================
FINAL VERDICT
==================================================================================================
PASS  DEMO-65 VERIFIED (continuous lift paradox + capstones + GR witnesses + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
