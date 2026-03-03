# DEMO-69 — OATB Master Flagship (Operator Admissibility Transfer Bridge)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) yields budgets; Fejér triangle multipliers are nonnegative, DC-preserving, exhibit UFET K(r) witness (~2/3 across budgets); sharp-transfer vs lawful-transfer on discontinuity shows lawful matches truth within eps while illegal creates Gibbs overshoot and negative density; same admissible operator resolves paradox pack (finite↔continuum, measure, quantum collapse); Ω reuse across PDEs (3D/4D heat, 4D Helmholtz NS); counterfactual triples degrade all observables by fixed eps margin.

---

## What this demo computes

First-principles deterministic audit:
- Deterministic selection: unique triple (137, 107, 103) via lane rules; counterfactual triples from extended window.
- Budget derivation: eps = 1/√q2; N, K from triple invariants.

Stage 1 (OATB kernel contract):
- Fejér triangle multipliers nonnegative (min ≥ tol).
- DC preservation: H_min ≈ 1/(r+1) and DC = 1 at radii r = 8, 16, 32.
- UFET K(r) witness: spread ≤ 1%, mean K(r) ≈ 2/3 (±2%).
- Sharp/signed kernels have negative lobes (illegal).
- Signed retains large HF weight (hf ≥ 0.25).

Stage 2 (Sharp-transfer vs lawful-transfer on discontinuity):
- Fejér distance vs truth ≤ eps.
- Illegal filters exhibit Gibbs overshoot (Fejér does not).
- Fejér preserves nonnegativity (min ≥ tol); illegal creates negative undershoot (≤ -eps²).
- Counterfactual budget reduction degrades by (1+eps).

Stage 3 (Paradox pack resolution):
- Zeno partial sum (geometric series) ≈ 1 (err ≤ 1e-9).
- Grandi Cesàro (harmonic oscillation) ≈ 1/2 (exact).
- Same admissible operator class handles all three paradoxes.

Stage 4 (Ω reuse across PDEs):
- 3D heat controller: mass preserved, HF error suppressed, better tracking than baseline.
- 4D heat controller: same admissibility + mass preservation.
- 4D NS-like vector field: Helmholtz projection + Ω admissibility → incompressibility maintained.

Stage 5 (Cross-base invariance + non-ubiquity):
- Rosetta-style: selector invariant across bases (base-7, base-10, base-16).
- Rigidity scan confirms non-ubiquity (uniqueness not generic).

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Primary triple must equal (137, 107, 103).
5. Fejér kernel nonnegative (min ≥ tol).
6. UFET spread ≤ 1%, mean K(r) ≈ 2/3.
7. Fejér distance to truth ≤ eps.
8. Illegal overshoot ≥ eps² (Fejér does not overshoot).
9. Illegal creates negative density; Fejér does not.
10. Counterfactual budget reduction degrades by (1+eps).
11. Paradox sums resolve within specified bounds.
12. Heat/NS controllers preserve mass and maintain expected properties.
13. Cross-base invariance confirmed; non-ubiquity via rigidity.

## Controls

- **Illegal operators:** Sharp spectral cutoff (negative kernel lobes), signed HF complement (stronger lobes); tested across kernel contract, discontinuity transfer, and all PDE suites.
- **Counterfactuals:** Alternative triples with same selection rules, different budgets; must degrade distance/errors by (1+eps) across all stages.
- **Ablations:** Rigidity scan (fixed rules, extended window) confirms non-ubiquity.

## Dependencies

Python 3.10+ with NumPy. Matplotlib optional (only for PNG output).

## Run

```bash
python demo.py
python demo.py --full  # Larger PDE controller steps (slower)
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
✅  Gate S1: primary equals (137,107,103)
✅  Gate S2: captured >=4 counterfactual triples                             found=4
✅  Gate K(r) contract @r=8: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=16: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=32: H_min≈1/(r+1) and DC=1
✅  Gate U1: UFET K(r) spread <= 1%                                          spread=0.570%
✅  Gate U2: mean K(r) close to 2/3 (<=2%)                                   |K-2/3|=0.001858
✅  Gate A1: Fejér kernel nonnegative (tol)                                  min=5.177e-10
✅  Gate A2: Sharp kernel has negative lobes                                 min=-3.511e-03
✅  Gate A3: Signed kernel has negative lobes                                min=-9.678e-01
✅  Gate A4: Signed kernel retains large HF weight                           hf=0.984 floor=0.250
✅  Gate T1: Fejér distance vs truth <= eps                                  dist=0.1405 eps=0.1826
✅  Gate T2: illegal filters exhibit Gibbs overshoot (Fejér does not)        ov_fejer=-0.044 ov_sharp=0.101 ov_signed=0.202 floor=eps^2=0.033
✅  Gate T3: Fejér preserves nonnegativity (tol)                             min=7.361e-03
✅  Gate T4: illegal kernels create negative density (undershoot)            floor=-eps^2=-0.033 mins=(-8.101e-02,-1.620e-01)
✅  Gate CF1: budget reduction degrades by (1+eps)                           distP=0.1405 distCF=0.3181 (1+eps)=1.183
✅  Gate Z1: Zeno partial sum close to 1                                     sum=0.999999999069 err=9.313e-10
✅  Gate G1: Grandi Cesàro close to 1/2                                      cesaro=0.500000 err=0.000e+00
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `06947e1258f6b5d3688a38c6ffff954d485e6470b12ac932eb32e60fdd4beb36`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
✅  Gate S1: primary equals (137,107,103)
✅  Gate S2: captured >=4 counterfactual triples                             found=4
✅  Gate K(r) contract @r=8: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=16: H_min≈1/(r+1) and DC=1
✅  Gate K(r) contract @r=32: H_min≈1/(r+1) and DC=1
✅  Gate U1: UFET K(r) spread <= 1%                                          spread=0.570%
✅  Gate U2: mean K(r) close to 2/3 (<=2%)                                   |K-2/3|=0.001858
✅  Gate A1: Fejér kernel nonnegative (tol)                                  min=5.177e-10
✅  Gate A2: Sharp kernel has negative lobes                                 min=-3.511e-03
✅  Gate A3: Signed kernel has negative lobes                                min=-9.678e-01
✅  Gate A4: Signed kernel retains large HF weight                           hf=0.984 floor=0.250
✅  Gate T1: Fejér distance vs truth <= eps                                  dist=0.1405 eps=0.1826
✅  Gate T2: illegal filters exhibit Gibbs overshoot (Fejér does not)        ov_fejer=-0.044 ov_sharp=0.101 ov_signed=0.202 floor=eps^2=0.033
✅  Gate T3: Fejér preserves nonnegativity (tol)                             min=7.361e-03
✅  Gate T4: illegal kernels create negative density (undershoot)            floor=-eps^2=-0.033 mins=(-8.101e-02,-1.620e-01)
✅  Gate CF1: budget reduction degrades by (1+eps)                           distP=0.1405 distCF=0.3181 (1+eps)=1.183
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
DETERMINISM HASH
==================================================================================================
determinism_sha256: 06947e1258f6b5d3688a38c6ffff954d485e6470b12ac932eb32e60fdd4beb36

==================================================================================================
FINAL VERDICT
==================================================================================================
✅  DEMO-69 VERIFIED (OATB flagship: admissibility + transfer + paradox + Ω reuse + invariance) score=1000000/1000000  passed_weight=32.00/32.00
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
