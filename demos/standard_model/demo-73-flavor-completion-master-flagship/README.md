# DEMO-73 — Flavor Completion Master Flagship

> **Claim:** A single deterministic integer substrate plus an admissible Fejér kernel budget is sufficient to generate the Yukawa hierarchy, quark mixing (CKM), and lepton mixing (PMNS) — three linked structures from one construction.

---

## What this demo computes

**Selection.** The primary triple (137, 107, 103) is selected by the standard SCFP++ lane gates. No search, no RNG.

**Yukawa hierarchy.** Dimensionless fermion couplings are derived from the kernel budget using the same Fejér-admissible operators that appear across the suite. The hierarchy spans five orders of magnitude (electron to top quark).

**CKM matrix.** Quark mixing angles are extracted from lawful kernel textures. The resulting 3×3 unitary matrix is exported with full angle and phase decomposition.

**PMNS matrix.** Lepton mixing angles are extracted from the same lawful texture construction. The PMNS matrix is exported with θ₁₂, θ₁₃, θ₂₃, and CP phase δ.

---

## Falsification contract

This demo is falsified if:

1. **Admissibility breaks.** The Fejér kernel must have a nonnegative profile. If sharp cutoff or signed-HF controls produce equivalent or better textures, the admissibility logic is vacuous.
2. **Stability fails.** Primary-budget outputs must be close to a higher-budget truth tier. If the budget→∞ limit diverges from the primary, the construction is budget-sensitive.
3. **Counterfactual teeth are absent.** Budget reduction must deterministically degrade outputs. If reduced budgets match primary-budget quality, the budget claim has no teeth.
4. **Any gate prints FAIL.**

---

## Controls

- **Illegal operators:** Sharp cutoff and signed-HF kernels tested against the same texture construction.
- **Budget stability:** Primary budget compared to higher-budget truth tier.
- **Counterfactual teeth:** Budget reduction degrades Yukawa, CKM, and PMNS outputs by explicit margins.

---

## Dependencies

Python 3.11+, NumPy.

## Run

```bash
python demo.py
python demo.py --json    # emit JSON artifact
```
