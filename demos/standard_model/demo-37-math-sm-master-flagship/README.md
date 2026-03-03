# DEMO-37 — Math × Standard Model Master Flagship

> **Claim:** A unique integer triple (wU, s2, s3) = (137, 107, 103) selected by deterministic modular constraints generates Standard Model–adjacent observables (α₀⁻¹ = 137, α_s(MZ) = 2/q3, α⁻¹(MZ)) and satisfies both mathematical closure (fast-converging constants) and spectral admissibility gates.

---

## What this demo computes

A deterministic, self-contained computational exhibit that:
1. Selects the primary triple (wU, s2, s3) by explicit rules (prime windows + modular residue filters + Euler-phi density + coherence)
2. Derives structural invariants (q2, q3, v2U, eps) from the triple
3. Builds Standard Model observables from first principles: α₀⁻¹ := wU; α_s(MZ) := 2/q3; Lambda_QCD (2-loop); QED running α⁻¹(MZ) with confinement-floor thresholds
4. Enforces operator admissibility via Fejér kernel (nonnegative in real space)
5. Contrasts against two illegal controls: sharp cutoff (Dirichlet ringing) and signed HF injection
6. Validates base-gauge invariance (encode/decode across bases 2–16 must preserve selector)
7. Demonstrates counterfactual teeth—at least 3/4 must degrade by rel_dist ≥ eps

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. The selected triple must equal (137, 107, 103); any other value falsifies.
3. Base invariance gate (B1) must pass across all bases [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16]; encode/decode round-trip failures falsify.
4. Fejér kernel (K1) must be nonnegative; kmin must equal 0 (within numerical tolerance).
5. Illegal kernels (K2) must show negative lobes; sharp cutoff and signed controls must both be negative.
6. Mean relative error (M1) must be ≤ eps³; any violation falsifies.
7. At least 3/4 counterfactuals (T) must miss by rel_dist ≥ eps; fewer strong counterfactuals falsify.
8. Determinism hash must match reference; any change falsifies.

## Controls

- **Illegal operators:** Sharp cutoff (Dirichlet; negative lobes); signed HF-injecting kernel
- **Counterfactuals:** Deterministic triples from separate window; nearby U(1) coherence-drop control
- **Base-gauge invariance:** Encode/decode in multiple bases; deliberately wrong-base decoding must fail

## Dependencies

Python 3.10+, numpy. Optional: matplotlib (for optional plotting only).

## Run

```bash
python demo.py
```
