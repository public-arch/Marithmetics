# DEMO-75 — Prediction Ledger Master Flagship

> **Claim:** The kernel pipeline produces forward predictions for quantities that are poorly measured or unmeasured — neutrino absolute masses, PMNS CP phase, and dark-sector candidates — using the same construction that derives the well-measured Standard Model observables.

---

## What this demo computes

This demo consolidates predictions that extend beyond the well-tested SM closure. Every prediction falls out of the existing pipeline with no additional parameters.

**Section A — Neutrino absolute masses.** Individual masses m₁, m₂, m₃ are derived from the kernel budget. Squared mass splittings Δm²₂₁ and Δm²₃₁ are computed. The sum Σm is exported as a cosmological cross-check target.

**Section B — PMNS + CP phase.** The leptonic CP-violation phase sin(δ) is derived from the same Φ-channel invariants used for the CKM matrix. Effective masses m_β (beta decay) and m_ββ (neutrinoless double-beta) are computed.

**Section C — Dark-sector candidates.** A dark-matter mass proxy m_χ and cross-section proxy σ are derived from the strong-field deviations of the kernel. These are structural predictions, not fitted to any dark-matter search data.

---

## Falsification contract

This demo is falsified if:

1. Any prediction section is missing from the output.
2. The lawful (Fejér/OATB) construction is matched or exceeded by illegal controls (sharp cutoff, sign-flip, Θ-substitution).
3. Counterfactual triples (409-class) or budget reduction fail to degrade the prediction vector.
4. Any printed gate shows FAIL.
5. Any claim in the output cannot be reproduced by rerunning the script.

---

## Controls

- **Illegal operators:** Sharp cutoff, sign-flip, and Θ-substitution tested against every prediction section.
- **Counterfactual triples:** 409-class triples processed through the identical pipeline.
- **Budget teeth:** Reduced budgets must produce measurably worse predictions.

---

## Dependencies

Python 3.10+, NumPy.

## Run

```bash
python demo.py
python demo.py --json    # emit JSON artifact
```
