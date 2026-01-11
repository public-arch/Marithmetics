# DEMO-39 — BB A2 Archive Master (closure capsule)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-39-bb-a2-archive-master-closure-capsule.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-39' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
alpha: 1/137 = 0.0072992700729927
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> A2 Archive Master Script (v1.3)
> a2_archive_master.py

> Stdlib-only, deterministic, one-file archive.

> What it does
> - Reproduces the A2 / BB-36 closure capsule from first principles (as currently encoded):
>   • Gauge lawbook derivation (contracts + coherence)
>   • Gauge selector (lane filters) → unique (wU,s2,s3) = (137,107,103)
>   • Φ-mapping uniqueness (alpha, alpha_s, sin^2)
>   • Yukawa palette closure (D1 local selector + offset sweep)
>   • Cosmology Ω-sector closure + flatness
>   • H0 closure (structural reuse)
>   • Primordial closures (As, ns, tau)
>   • Neutrino closures (Δ21, Δ31, Σmν)
>   • Amplitude closures (etaB, YHe, deltaCMB) + ℓ1 reuse
>   • Cross-base compatibility / Rosetta suite (base-7/10/16)
>   • Shows explicit cross‑base roundtrip equality for key A2 constants (table + max‑error summary).

> Run:

## Run instructions

- Dependencies: Stdlib-only (no third-party packages).

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-39-bb-a2-archive-master-closure-capsule.py
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
