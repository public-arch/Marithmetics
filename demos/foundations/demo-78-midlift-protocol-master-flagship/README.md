# DEMO-78 — Midlift Protocol Master Flagship

> **Claim:** Starting from the 0R-substrate and residue grammar, the one-dimensional number line appears as a derived lift — not an axiom — and treating 1D as primitive loses structure that the 0R substrate preserves.

---

## What this demo computes

This demo makes the Midlift Protocol explicit and auditable. It addresses whether the foundational move from discrete residue arithmetic to continuous structure is a convenience or a necessity.

**Φ recovery.** From the primary triple (137, 107, 103) and frozen Φ-map specifications, the demo recovers the three anchor rationals (α_em = 1/137, sin²θ_W = 7/30, α_s = 2/17) and verifies them against their targets.

**Physical target mining.** Using the kernel budget (K_PHI modes), the demo mines physical observables from rational combinations of kernel eigenvalues: mass ratios (m_e/m_p, m_μ/m_p), CKM elements (V_us, V_cb, V_ub), neutrino mass splitting ratios, PMNS parameters, proton charge radius, and the electron anomalous magnetic moment.

**Gauge + GR hook.** Extended target mining covers the full CKM matrix, PMNS mixing angles, and gravitational observables — all derived from the same substrate with no additional inputs.

**Counterfactual triples.** The 409-class triples (four variants) are processed through the identical mining protocol. Their scores must degrade relative to the primary.

**Provenance anchors.** SHA-256 hashes from prior authority releases (S4B, S5B, S6F) are checked for continuity — ensuring the current code produces results consistent with the archived Authority-of-Record.

---

## Falsification contract

This demo is falsified if:

1. The Φ recovery fails to reproduce the anchor rationals from the frozen map.
2. Counterfactual triples produce target-mining scores comparable to the primary.
3. The mining protocol requires any input beyond the triple and the declared mode budget.
4. Provenance hashes do not match prior authority releases.
5. Any gate prints FAIL.

---

## Controls

- **Counterfactual triples:** Four 409-class variants processed through identical mining.
- **Provenance chain:** SHA-256 hashes verified against prior authority bundles.
- **Budget modes:** K_PHI and K_PHYS budgets are declared constants, not searched.

---

## Dependencies

Python 3.10+ standard library only (uses `fractions.Fraction` for exact arithmetic).

## Run

```bash
python demo.py
```
