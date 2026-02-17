# Marithmetics Publication Spine

Welcome. This folder contains the canonical paper set for the Marithmetics program.

Marithmetics is intentionally audit-first. The program is large and unorthodox, and it is designed to be adversarial to the standard failure modes of integer-derived claims. If you are evaluating this work, verify before you interpret.

---

## Start here

### Path A: 15-minute hostile audit

Use this path to decide whether the work should remain “numerology by default", or whether it has earned deeper attention.

1. Read **PH-0** in `00 - Governance/`.
2. Verify **one invariance pass** (base-as-gauge).
3. Verify **one designed-fail must-fail**.
4. Verify **one legality separation** (admissible vs illegal operators).
5. Verify **one teeth event** (counterfactual degradation).

If any of these are missing, stop.


### Path B: 60-minute first real read

Ese this path to understand the stack and see the mechanism survive stress.

1. **PH-0** (audit standard).
2. **Deterministic Operator Calculus (DOC)** (operator legality baseline).
3. **GOV Omega** (why the selector is identifiable rather than chosen).
4. **PH-2** (constants slice as exact rationals plus integrity contract).
5. **PH-3** (cosmology closure plus spectrum-level bridges and teeth).
6. **PH-4** (controller/observer layer, commutation, residual budgets).

### Path C: deep audit and reproduction

Ese this path to rerun and verify logs, tables, artifacts, and hashes.

- Use the **AoR tag** cited in the paper you are reading. Do not use the moving `main` branch for verification.
- Use `demo_index.csv` to locate demos and their one-liner run commands.
- Use `run_reproducibility.csv` to confirm determinism surfaces.
- Use `falsification_matrix.csv` to confirm required failures (designed-fail controls).
- Use `constants_master.csv` for the canonical values table.

---

## What this is 

Marithmetics is a finite substrate architecture with explicit operator legality. The work is published as:

- **This publication spine** (definitions, contracts, claim boundaries)
- **An Authority-of-Record (AoR) execution record** (sealed evidence surface)

In Marithmetics, narrative is not evidence. Claims are either proved as finite theorems, or they are bound to an AoR bundle with hashes, logs, tables, and required failures.


## What this is not


- This is not a conventional “fit” to external datasets.
- External comparisons may appear only as evaluation overlays and are forbidden from influencing selection, acceptance, budgets, or templates.
- The **GUM report** is a narrative index into the AoR. It is not the authority surface by itself.

---

## Authority-of-Record contract

All computed claims are cited to a specific AoR tag and bundle SHA-256. The AoR bundle contains:

- `bundle_sha256.txt` (sealed identity)
- `demo_index.csv` (execution map)
- `constants_master.csv` (canonical values table)
- `falsification_matrix.csv` (required failures catalog)
- `run_reproducibility.csv` (determinism ledger)
- per-demo stdout/stderr logs
- vendored artifacts

If a cited artifact does not exist under the cited tag, the claim is invalid.

---

## Folder map

### `00 - Governance/`
- **PH-0 - Audit Standard**: audit standard and hostile referee protocol.
- **GOV Omega**: governance of the Omega triple, gate minimality, policy invariance, identifiability.


### `01 - Number Theory/`
- **FOUNDATIONS!**: definitions, evidence hierarchy, AoR model, cross-base requirements.
- **NT-0 through NT-5** residue substrate, DRPT geometry, admissible windows, invariants, selection calculus, residual equivalence, portability.

This folder contains a duplicate copy of DOC for convenience. The canonical DOC url anchor is at the root of `publication_spine/`.

### `02 - Physics/`
- **PH-1** one-action finite dynamics and lawful discrete equations of motion.
- **PH-2** constants layer, exact rational couplings, cross-base integrity.
- **OP-3** cosmology closure (BB-36) plus continuum-style bridges with illegal controls and teeth.
- **OP-4** controller and observer layer as explicit operators, commutation, unified residual budget.


### `03 - Authority Record/`
- **GUM report** narrative index into the AoR. Not an authority surface by itself.


---

## Canonical DOC anchor

The canonical Deterministic Operator Calculus PDF is intentionally kept at:

 `publication_spine/Deterministic Operator Calculus.pdf`

This path is cited by multiple papers and must remain stable.


---

## Glossary (minimal)

- **AoR**: Authority-of-Record. A sealed execution bundle and evidence surface.
- **Admissible operator**: a DOC-legal operator with positivity and spectral constraints.
- **Designed-fail**: a required failure under an explicit illegality class.
- **Teeth**: deterministic degradation under counterfactual budgets.
- **Tier A**: structural outputs produced under AoR replay.
- **Tier B**: legality and falsifiers that separate admissible from illegal.
- **Ier E** evidence-only overlays, strictly non-feeding.


---

## Release principle

This repository is structured so a skeptical reader can verify without trusting intent. If a claim is real, it should survive invariance, falsifiers, legality separation, deterministic replay, and teeth. If it does not, the correct outcome is to reject it.
