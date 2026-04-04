# Marithmetics

**A deterministic pipeline that derives physical constants from integer structure alone.**

This repository contains 29 self-auditing computational demos. Each one starts from the same integer triple — **(137, 107, 103)** — selected by number-theoretic admissibility gates with zero free parameters, and derives dimensionless physical observables: coupling constants, mass ratios, cosmological parameters, mixing matrices. No fitting. No tuning. No PDG data upstream.

These are extraordinary claims. The entire architecture of this project exists to make them falsifiable.

---

## What this repository claims

A single deterministic selector, operating on prime structure within the window [97, 180], produces a unique admissible triple. From that triple, through explicit algebraic channels with no adjustable parameters:

| Domain | What is derived | Demo |
|---|---|---|
| **Electroweak** | α⁻¹ = 137 (exact), sin²θ_W = 7/30, α_s(M_Z) = 2/17 | DEMO-33, DEMO-34 |
| **Standard Model** | 9 fermion masses, CKM/PMNS matrices, Γ_Z, v, M_Z, M_W | DEMO-33, DEMO-73 |
| **QCD** | Λ_QCD (2-loop + 4-loop MS̄), proton charge radius | DEMO-37, DEMO-55 |
| **Cosmology** | H₀, Ω_b, Ω_c, A_s, n_s, τ, ℓ₁ | DEMO-36 |
| **General relativity** | Weak-field GR tests (bending, Shapiro, redshift, perihelion), vacuum energy | DEMO-68, DEMO-51 |
| **Quantum gravity** | Discrete RG flow, screening witnesses, strong-field geometry | DEMO-66 |
| **Integer selection** | Triple uniqueness: five number-theoretic gates force (137, 107, 103) | DEMO-18 |
| **Quantum mechanics** | Probability-safe coarse graining, double-slit, dispersion | DEMO-60 |
| **Navier–Stokes** | 3D Taylor–Green vortex benchmark under operator admissibility | DEMO-67 |
| **Neutrino sector** | Absolute masses (m₁, m₂, m₃), Σm, m_β, m_ββ, CP phase | DEMO-75 |
| **Higgs sector** | EW rational locks, UV critical coupling λ*, mode-ladder SU(2) lock | DEMO-70 |

External datasets (Planck, CAMB, PDG) are used **only** as evaluation overlays. They never enter the derivation.

---

## How to break this

Every demo ships its own falsification contract. But the fastest attacks on the framework are:

**"This is numerology — it depends on base 10."**
Run [DEMO-64](demos/substrate/demo-64-base-gauge-invariance-integer-selector/). The selector operates identically in bases 2, 7, 10, and 16. If the triple changes under re-encoding, the claim is dead.

**"You tuned parameters to fit known values."**
Every demo includes counterfactual triples (e.g., the 409-class). These are processed through the identical pipeline. If counterfactuals produce comparable closures, the method is fit-dependent. They do not. Counterfactual certification scores degrade by 6× or more.

**"The operators are chosen to get the right answer."**
Every demo with a spectral operator tests three classes: lawful (Fejér/Cesàro, nonnegative kernel), sharp cutoff (Gibbs artifacts), and signed high-frequency injection. If illegal operators perform as well as lawful ones, the admissibility logic is vacuous. They do not — illegal controls inject measurable artifacts across every domain.

**"One triple is a coincidence."**
The selector is not a search — it is a filter. In the primary window [97, 180], exactly one triple survives all gates. In the extended window [80, 1,000,000], still one. Drop any single gate and survivor counts explode from 1 to hundreds or thousands. The necessity of every constraint is demonstrated by ablation. [DEMO-18](demos/substrate/demo-18-integer-triple-uniqueness/) proves this directly: five Phi-derived gates (period extremality, Legendre parity, 2-adic branching, wheel orientation, simplicity) are each independently necessary, and jointly sufficient, to force the triple.

---

## The 60-second audit

```bash
python -m audits.run_master_suite --verbosity full
```

This runs all 28 demos, captures stdout/stderr, vendors artifacts, seals everything into a cryptographic **Authority-of-Record (AoR)** bundle, and generates the launch report from that bundle.

What you should expect:

- **Deterministic** results — identical outputs across runs, within declared tolerances
- A sealed **AoR bundle** with logs, artifacts, tables, and SHA-256 hashes
- A generated **GUM launch report (v32 PDF)** built from the bundle
- A master release zip for archival or independent verification

---

## Authority-of-Record (AoR)

After `run_master_suite` completes, the AoR is written to `gum/authority_archive/`. Each AoR folder contains:

- `GUM_BUNDLE_v30_.../` — tables, logs, vendored artifacts, hashes
- `report/` — the launch report PDF + manifest
- `claim_ledger.jsonl` — machine-readable claim ledger
- `runner_transcript.txt` — full CLI transcript
- `MARI_MASTER_RELEASE_*.zip` — portable frozen snapshot

**The bundle hash** (`bundle_sha256.txt`) is what you cite. It is the immutable record.

---

## Verification protocol

### Run the full suite and generate the AoR

```bash
python -m audits.run_master_suite --verbosity full
```

### Build a bundle without the full suite runner

```bash
python -m audits.gum_bundle_v30 --outroot audits/results --vendor-artifacts --demos-root demos
```

### Generate the report from a specific bundle

```bash
python gum/gum_report_generator_v32.py --bundle-dir /path/to/GUM_BUNDLE_v30_*
```

### Run individual demos

```bash
python demos/standard_model/demo-33-first-principles-standard-model-sm28-closure/demo.py
python demos/bridge/demo-34-omega-sm-master-flagship/demo.py
python demos/cosmo/demo-36-big-bang-master-flagship/demo.py
python demos/quantum_gravity/demo-66-quantum-gravity-master-flagship-v4/demo.py
python demos/substrate/demo-18-integer-triple-uniqueness/demo.py
python demos/substrate/demo-64-base-gauge-invariance-integer-selector/demo.py
```

Every demo is a single `demo.py` file. No hidden dependencies between demos. No shared state.

---

## How to cite

For any claim, cite:

1. **Demo ID** (e.g., DEMO-36)
2. **AoR bundle hash** (`bundle_sha256.txt`)
3. **Artifact path inside the bundle** (e.g., `vendored_artifacts/<slug>__*.png`)
4. **File hash prefix** (as listed in the report evidence table)

Citations are stable across repository evolution because the AoR is the immutable record.

---

## Repository layout

```
demos/                    29 canonical demos (demo.py per folder)
audits/                   AoR bundler + full suite runner
gum/                      Report generator (v32), AoR archive, report assets
atlas_substrate_visualization/   Interactive DRPT explorer
publication_spine/        Canonical paper spine (governance, number-theory, physics tracks)
website/                  marithmetics.com source (Vite + React)
```

---

## Design principles

This is not a paper making a theoretical argument. It is an execution surface.

Every demo in this suite follows the same discipline:

- **Claim first.** The demo declares what it will compute before it computes it.
- **Zero upstream data.** No physical measurements enter the derivation. PDG/Planck/CAMB appear only in post-hoc evaluation overlays, clearly marked.
- **Deterministic.** Every demo produces a SHA-256 hash of its outputs. Run it twice — if the hash changes, the demo is broken.
- **Self-auditing.** Every demo prints PASS/FAIL gates. A single FAIL anywhere in the suite is a falsification event.
- **Negative controls.** Illegal operators (sharp cutoff, signed kernels) and counterfactual triples (409-class) are tested through the same pipeline. If controls perform as well as the primary, the logic is vacuous.
- **No narrative without execution.** If a claim appears in the README or the report, there is a demo that computes it and a gate that falsifies it.

---

**Website:** [marithmetics.com](https://marithmetics.com)
