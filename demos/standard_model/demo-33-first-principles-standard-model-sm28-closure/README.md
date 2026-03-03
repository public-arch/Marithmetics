# DEMO-33 — First-Principles Standard Model (SM-28 Closure)

> **Claim:** The full Standard Model — gauge couplings, fermion masses, CKM and PMNS mixing matrices, neutrino masses, vacuum energy, and Λ_QCD — is derived from a single deterministic integer triple through an explicit algebraic pipeline with no fitted parameters and no upstream physical data.

---

## What this demo computes

DEMO-33 is the most comprehensive demo in the suite. It runs a 13-stage pipeline from integer selection to a complete Standard Model manifest (28 observables).

**Stage 1 — SCFP++ selection.** Lane-gated survivor search on the prime window [97, 180]. Three modular residue filters (U(1) mod 17, SU(2) mod 13, SU(3) mod 17) plus a totient-ratio threshold (τ) yield exactly one admissible triple: (137, 107, 103). Full τ-robustness scan and gate ablation included.

**Stage 2 — κ refinement.** A closed-form margin optimizer derives κ_refined and the canonical ℓ★/Λ★ seam (BH/Unruh boundary) from the triple's sum S = (wU + s2 + s3)/8.

**Stage 3 — Φ-channel rationals.** From the triple's arithmetic:
- α₀ = 1/wU = 1/137
- q₂ = wU − s2 = 30, giving Θ = φ(30)/30 = 4/15
- sin²θ_W = Θ·(1 − 1/2^v₂(wU−1)) = 7/30
- q₃ = (wU − 1)/2^v₂(wU−1) = 17, giving α_s = 2/q₃ = 2/17

All rationals are exact (Python `Fraction`). No floating-point rounding enters the derivation.

**Stage 4 — Palette-B.** A declared 9-tuple of Yukawa exponents, verified against five structural gates (E1–E5: sector ordering, fixed offsets, denominator constraints) and a lattice isolation witness (E6). A bounded neighborhood scan certifies local uniqueness.

**Stage 5 — One-action minimizer.** Closed-form derivation of the Higgs VEV (v) from the palette exponents and κ-margins. No numerical optimization.

**Stage 6 — CKM/PMNS.** Mixing-matrix denominators and CP phases derived from Φ-channel invariants. Full 3×3 matrices exported with unitarity defects at machine precision (~10⁻¹⁶ for CKM, ~4×10⁻¹⁷ for PMNS).

**Stage 7 — SM symbolic manifest.** Anomaly cancellation verified in exact arithmetic. Hypercharge sums over all SM fields confirmed to vanish.

**Stage 8 — 1-loop RG.** Gauge coupling running from M_Z to arbitrary scales using β-coefficients derived from the SM field content (not fitted).

**Stage 10 — Γ_Z prediction.** Z-boson total width from tree-level partial widths with leading-order QCD corrections.

**Stage 12 — Neutrinos, vacuum energy, Λ_QCD, G_F.** Structural witnesses for the neutrino mass sector, cosmological constant, QCD scale (2-loop and 4-loop MS̄), and Fermi constant.

**Stage 12B — Authority v1 dressed predictions.** The complete prediction vector with solver-invariance witness (damping invariance certificate).

**Stage 12C — SM-28 tables.** Full manifest comparing structural (pure) outputs to dressed predictions, formatted for the Authority-of-Record.

**Stage 13 — PDG overlay (evaluation only).** Optional `--overlay` flag compares outputs to PDG values. This stage is explicitly excluded from the derivation — it is a post-hoc evaluation.

---

## Key outputs

| Observable | Derived value | PDG reference |
|---|---|---|
| α⁻¹ | 137 (exact) | 137.036 |
| sin²θ_W | 7/30 = 0.2333 | 0.2312 |
| α_s(M_Z) | 2/17 = 0.1176 | 0.1179 |
| v (Higgs VEV) | 245.8 GeV | ~246 GeV |
| M_Z | 91.03 GeV | 91.19 GeV |
| M_W | 79.71 GeV | 80.38 GeV |
| Λ_QCD (4-loop MS̄) | 0.159 GeV | ~0.2 GeV |
| CKM unitarity defect | ~10⁻¹⁶ | 0 (exact) |
| PMNS unitarity defect | ~4×10⁻¹⁷ | 0 (exact) |

Nine fermion masses are derived from the Palette-B exponents through a single mass law: m_f = (v/√2) · 17^(−e_f).

---

## Falsification contract

This demo is falsified if any of the following occur:

1. **The triple is not unique.** Drop any single gate (C2, C3, or C4) and verify that pool sizes expand. If they do not, that gate is not necessary and the selection is overconstrained.
2. **The τ threshold is fragile.** The demo scans Δτ ∈ [−0.020, +0.012]. If the triple changes within this range, the selection is threshold-sensitive.
3. **Palette-B has local competitors.** The bounded lattice scan (±1/8 step in each of 9 components) must find zero E1–E5-passing neighbors. If competitors exist, the palette is not locally unique.
4. **Anomaly cancellation fails.** Hypercharge sums must vanish exactly in `Fraction` arithmetic.
5. **CKM/PMNS unitarity is broken.** Unitarity defects must be at machine precision, not approximate.
6. **Determinism hash changes.** `sm_outputs_pure.json` must be byte-identical across runs. If the SHA-256 drifts, the pipeline has a hidden stochastic dependency.
7. **Selftest regression.** `--selftest` hard-fails if any numeric checkpoint drifts beyond declared tolerances.

---

## Controls

- **Gate ablation:** Drop C2 → pool sizes explode (U(1): 3→14, SU(2): 1→6, SU(3): 2→7). Every gate is necessary.
- **τ robustness:** The triple is stable across a continuous band of threshold perturbations.
- **Palette isolation:** E6 witness confirms the declared palette sits in a gap on the exponent lattice.

---

## Dependencies

Python 3.10+ standard library only. No NumPy. No external packages.

## Run

```bash
python demo.py              # pure mode (default)
python demo.py --overlay    # pure + PDG evaluation overlay
python demo.py --selftest   # hard-fail on any regression
python demo.py --cert       # emit certificate zip
```

Outputs: `sm_outputs_pure.json` (deterministic, byte-stable, SHA-256 hashed).
