# DEMO-66 — Quantum Gravity Master Flagship

> **Claim:** From the same deterministic integer triple (137, 107, 103), a discrete renormalization group flow, two independent screening witnesses, and a strong-field geometry emerge — all without fitting, external data, or numerical optimization.

---

## What this demo computes

**Selection.** The primary triple is selected deterministically by residue + v₂-coherence rules on primes in [97, 180]. No search, no RNG.

**Discrete scale.** A canonical scale D* = lcm(b−1) is derived from base structure alone — no empirical input.

**Weak-field closure.** ε₀(β, N) is produced from a locked κ*(β, N) ledger (no fit) and closes to ~10⁻⁵. This is the vacuum-energy-scale proxy derived entirely from the triple.

**RG flow.** A locked renormalization group table is fitted to R(D) = R_∞ + a/D², yielding a nonzero effective gravitational coupling g_eff. The fit is to the locked table, not to experimental data.

**Screening witnesses.** Two independent lawful constructions — piecewise saturation and smooth saturation — produce monotone weak→strong behavior and agree to within a declared tolerance. Both are built from Fejér-admissible operators.

**Strong-field geometry.** A Reissner-Nordström–like softening proxy remains lawful (no naked singularity) for the derived coupling. An illegal Θ-palette injection destroys the horizon — the control fails as expected.

**Counterfactual teeth.** Budget-limited counterfactual triples (409-class) deterministically degrade weak-field, RG, and strong-field scores by explicit margins.

---

## Falsification contract

This demo is falsified if:

1. The two screening witnesses disagree beyond the declared tolerance.
2. The illegal Θ-palette control produces a lawful horizon (it should destroy it).
3. Any counterfactual triple matches or exceeds the primary's certification scores.
4. The RG table changes between runs (locked ledger — determinism hash must be stable).
5. Any gate prints FAIL.
6. The canonical JSON payload hash drifts between runs.

---

## Controls

- **Illegal operator:** Θ-palette injection into strong-field geometry — must destroy the horizon.
- **Counterfactual triples:** 409-class triples processed through the identical pipeline — must degrade all scores.
- **Screening cross-check:** Two independent witness constructions must converge.

---

## Dependencies

Python 3.10+ standard library only. No NumPy. No external packages.

## Run

```bash
python demo.py
python demo.py --selftest          # hard-fail on regression
python demo.py --cert --out-dir .  # emit certificate zip
```

Outputs: `demo66_v4_outputs_pure.json` (deterministic, byte-stable, SHA-256 hashed).
