# Marithmetics
**A deterministic, audit-grade pipeline for integer-to-physics emergence**

**Narrative is not evidence. Execution is.**  

This repository produces **Authority-of-Record (AoR)** bundles: it runs a deterministic demo suite, captures logs and artifacts, seals them into a cryptographic bundle, and generates the launch report from that bundle.

---

## The 60-second audit

```bash
python -m audits.run_master_suite --verbosity full
```

What you should expect:
- **Deterministic** results (same outputs across runs, within declared tolerances)
- A sealed **AoR bundle** (logs, artifacts, tables, hashes)
- A generated **GUM launch report (v32 PDF)** built from the bundle
- A master release zip for archival or upload

---

## Trophy case: headline outputs (derived + certified)

These are the “fast anchors” a referee can use to orient. Each row points to a demo whose certificate and evidence table are included in the report and AoR.

| Anchor | What you will see | Source |
|---|---|---|
| **Kernel bridge (Ω→SM)** | Cross-domain bridge proof showing coupled constraint flow | **DEMO-34** |
| **Standard Model flagship** | Full structured export of Standard Model constants, closures, and certificates | **DEMO-33** |
| **Weak mixing rational** | Exact rational anchor appears in certificate + exports | **DEMO-33** |
| **Strong coupling rational** | Exact rational anchor appears in certificate + exports | **DEMO-37** |
| **Cosmology closure** | Kernel-derived cosmology parameters + falsification gates | **DEMO-36** |
| **Visual proof anchors** | Planck/CAMB overlay and QG screening witness (bundle artifacts) | **DEMO-36 / DEMO-66** |
| **Base-gauge invariance** | Representation independence across bases (2/7/16) | **DEMO-64** |
| **Counterfactual teeth** | Counterfactual triples fail under the same pipeline (fit-dependence test) | **Multiple** |

**Important:** External datasets (e.g., Planck/CAMB) are used strictly as **evaluation overlays**. They do not feed upstream selection.

---

## Where the Authority-of-Record (AoR) lives

After `run_master_suite` completes, your AoR is written under:

- `gum/authority_archive/`

Only the canonical release AoR is kept in the release surface. Historical runs are archived under `zz_archive/`.


Each AoR folder contains:
- `GUM_BUNDLE_v30_.../` (tables, logs, vendored artifacts, hashes)
- `report/` (the report PDF + manifest copied into the AoR)
- `claim_ledger.jsonl` (machine-readable claim ledger)
- `runner_transcript.txt` (full CLI transcript)
- `MARI_MASTER_RELEASE_*.zip` (portable frozen snapshot)

**Bundle hash (what you cite):**
- `gum/authority_archive/.../GUM_BUNDLE_v30_.../bundle_sha256.txt`

---

## Verification protocol (rebuild everything)

### A) Build a fresh bundle (AoR) without running the full suite

```bash
python -m audits.gum_bundle_v30 --outroot audits/results --vendor-artifacts --demos-root demos
```

This produces a `GUM_BUNDLE_v30_*` directory containing:
- `logs/` (stdout/stderr per demo)
- `vendored_artifacts/` (images/json exported by demos, hashed)
- `tables/` (index tables, falsification matrix, constants tables)
- `bundle_sha256.txt` (the bundle seal)
- `codepack/` (code snapshot for reproducibility)

### B) Generate the report from a specific bundle

```bash
python gum/gum_report_generator_v32.py --bundle-dir /path/to/GUM_BUNDLE_v30_*
```

### C) Generate the report from the latest AoR bundle

```bash
BUNDLE="$(ls -dt gum/authority_archive/*/GUM_BUNDLE_v30_* 2>/dev/null | head -n 1)"
python gum/gum_report_generator_v32.py --bundle-dir "$BUNDLE"
```

### D) Drill into demos individually

```bash
python demos/bridge/demo-34-omega-sm-master-flagship-v1/demo.py
python demos/standard_model/demo-33-first-principles-standard-model-sm28-closure/demo.py
python demos/cosmo/demo-36-big-bang-master-flagship/demo.py
python demos/quantum_gravity/demo-66-quantum-gravity-master-flagship-v4/demo.py
python demos/substrate/demo-64-base-gauge-invariance-integer-selector/demo.py
```

---

## How to cite results (paper rewrite recipe)

For any claim, cite:
1) **Demo ID** (e.g., DEMO-36)  
2) **AoR bundle hash** (`bundle_sha256.txt`)  
3) **Artifact/log path inside the bundle** (e.g., `vendored_artifacts/<slug>__*.png` or `logs/<slug>.out.txt`)  
4) **File hash prefix** (as listed in the report evidence table, or recomputed from the AoR)

This makes citations stable even as the repository evolves, because the AoR is the immutable record.

---

## Repository layout

- `demos/` — canonical demo suite (`demo.py` per demo folder)
- `audits/` — AoR bundler + full suite runner
- `gum/` — report generator (v32), report assets, AoR archive, reports folder
- `atlas_substrate_visualization/` — interactive DRPT explorer
- `publication_spine/` — canonical paper spine (governance, number-theory track, physics track)  
  - DOC is intentionally duplicated for navigation while preserving the canonical DOC anchor path.

---

## Falsification guide (how to break this)

We invite skeptical review. The suite includes negative controls designed to make failure modes obvious.

**Attack: “This is just numerology / base-10 bias.”**  
- **Defense:** run **DEMO-64** (base-gauge invariance). It repeats the derivation across bases (e.g., 2/7/16). If invariants do not hold across representations, the claim fails.

**Attack: “You tuned parameters to fit the data.”**  
- **Defense:** counterfactual teeth appear across the suite (e.g., 409-class triples). If counterfactuals produce comparable closures under the same pipeline, the method is fit-dependent. The certificates include these ablations and their fail margins.

**Attack: “The operators are arbitrary.”**  
- **Defense:** demos with admissibility contracts include illegal controls (sharp/signed kernels). If illegal operators perform as well as lawful admissible ones, the logic is flawed. See operator admissibility sections (e.g., DOC-adjacent demos and PDE/continuum tests).

---

## Notes on scope

- This is a **technical release** focused on reproducible evidence: demos, bundles, and a launch report.
- Interpretive and theoretical papers are being rebuilt to cite the AoR produced by this suite (see `publication_spine/`).
