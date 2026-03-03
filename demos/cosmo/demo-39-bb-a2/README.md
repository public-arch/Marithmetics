# DEMO-39 — BB A2 Archive Master (closure capsule)

> **Claim:** A deterministic stdlib-only audit reproduces the A2/BB-36 closure capsule from first principles, deriving the unique triple (wU, s2, s3) = (137, 107, 103) and all gauge/cosmology/neutrino/amplitude invariants via fixed contracts (prime window, residue filters, 2-adic coherence) with cross-base Rosetta roundtrip equality (base-7/10/16).

---

## What this demo computes

From a finite arithmetic substrate (primes + residue filters + 2-adic coherence):
- Deterministic selection of the unique admissible triple (wU, s2, s3) = (137, 107, 103) via lane filters.
- Gauge lawbook derivation: α₀⁻¹ := wU = 137, sin²θW := 7/(wU - s2) = 7/30, αₛ(MZ) := 2/q₃ = 2/17.
- Yukawa palette closure via D1 local selector + offset sweep → Palette-B as unique best tuple.
- Cosmology Ω-sector closure with near-flatness and H0 rank-1 closure via structural reuse.
- Primordial sector: As, ns, tau rank-1 closures.
- Neutrino sector: Δ₂₁, Δ₃₁, Σmν closures + hierarchy contract.
- Amplitude sector: etaB, YHe, deltaCMB windows + ℓ₁ reuse.
- Cross-base consistency audit (base-7, base-10, base-16): roundtrip to identical numeric values (tol ≤ 1e-15).

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. All closure gates must pass: lawbook, selector, Φ-mapping, Yukawa, cosmology, primordial, neutrino, amplitudes, Rosetta suite.

## Controls

- **Illegal operators:** None tested (this is a pure arithmetic audit, not a field-solver demo).
- **Counterfactuals:** Not applicable (single deterministic path; closure is unique by construction).
- **Ablations:** All closure stages are required to pass; removing any stage would break the monolithic audit.

## Dependencies

Python 3.7+ with standard library only (no third-party packages).

## Run

```bash
python demo.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
✅ CLOSED: unique canonical lawbook under declared contracts.
✅ CLOSED: canonical (137,107,103) and invariants (q2=30, v2=3, Θ=4/15).
✅ CLOSED: α=1/137, αs=2/17, sin^2=7/30.
✅ CLOSED: D1 selects Palette‑B as unique best tuple.
✅ CLOSED: BB‑36 Ω templates + near-flatness (ε<=1e-3).
✅ CLOSED: BB‑36 H0 is rank‑1.
✅ CLOSED: primordial trio is rank‑1.
✅ CLOSED: neutrino templates + hierarchy contract pass.
✅ CLOSED: amplitude windows pass; NC breaks deltaC window.
✅ CLOSED: base representations (b7/b10/b16) roundtrip to the same numeric values (tol=1e-15).
```

## Reference checkpoints (from provided transcript)

Selected printed checkpoints:

```text
unique triple: {'wU': 137, 's2': 107, 's3': 103}
alpha_s: 2/17 = 0.1176470588235294
sin^2: 7/30 = 0.2333333333333333
✅ CLOSED: α=1/137, αs=2/17, sin^2=7/30.
alpha_s  0.11764705882352941      0.055232026114346405(base7)    0.1e1e1e1e1e1e1e0000(base16)   4.72e-16
sin2     0.233333333333333337     0.143014301430143014(base7)    0.3bbbbbbbbbbbbc0000(base16)   3.33e-16
══════════ A2 scoreboard ══════════
```

Transcript excerpt (for quick visual diff):

```text
Om_tot   0.999944398552645386     0.666660313165253433(base7)    0.fffc5b299a1cc80000(base16)   1.11e-16
  As       2.09909411320087293e-09  0.000000000041024402(base7)    0.0000000903fa7774d8(base16)   2.44e-16
  ns       0.964746071154910068     0.651623260544614532(base7)    0.f6f999388e09300000(base16)   2.22e-16
  tau      0.053979484970166157     0.024341426631301623(base7)    0.0dd1997a9a0eb98000(base16)   1.60e-16
  ell1     219.949087324076373307   432.643352113612240534(base7)  db.f2f76309f4c8000000(base16)  0.00e+00
  ✅ CLOSED: base representations (b7/b10/b16) roundtrip to the same numeric values (tol=1e-15).
══════════ A2 scoreboard ══════════
  claim                                PASS?
  H0_BB36_rank1                        PASS 
  a2_constants_roundtrip_parse_ok      PASS 
  a2_numeric_invariant_under_repr      PASS 
  amplitudes_deltaC_in_window          PASS 
  amplitudes_etaB_in_window            PASS 
  cosmology_flatness_eps_1e-3          PASS 
  cosmology_templates_match_BB36       PASS 
  ell1_value_finite                    PASS 
  gauge_invariants_match               PASS 
  gauge_lawbook_unique                 PASS 
  gauge_selector_unique                PASS 
  neutrinos_closed                     PASS 
  phi_mapping_expected_fractions       PASS 
  primordial_As_rank1                  PASS 
  primordial_ns_rank1                  PASS 
  primordial_tau_rank1                 PASS 
  rosetta_all_pass                     PASS 
  yukawa_D1_best_is_palette_B          PASS 
  yukawa_offset_sweep_canonical_found  PASS
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
