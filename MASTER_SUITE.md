# MARI / GUM Master Suite
Deterministic Kernel · Cross-Domain Evidence · Authority-of-Record

This repository includes a one-click master demonstration suite for the MARI / GUM program.

The Master Suite runs the current demo set in a curated narrative order and produces a new canonical Authority-of-Record (AoR) suitable for rewriting papers, auditing claims, and public inspection.

This is not a collection of independent experiments. Every result is a consequence of a single deterministic kernel, evaluated under explicit legality and falsification rules.

---

## One-click run

From the repository root:

    python -m audits.run_master_suite

To print every demo’s full metrics inline (very verbose):

    python -m audits.run_master_suite --verbosity full

To fail the run if any demo does not verify:

    python -m audits.run_master_suite --strict

---

## What the suite produces (Authority-of-Record)

Each run writes a timestamped AoR directory under:

    GUM/authority_archive/AOR_<UTC_TIMESTAMP>_<GIT_SHA>/

Inside:

- Bundle directory (GUM_BUNDLE_v30_*)
  - logs (canonical evidence)
  - vendored_artifacts
  - tables (constants, falsification matrix, reproducibility)
  - codepack
  - bundle hash

- Report
  - GUM_Report_v31_<timestamp>.pdf
  - .manifest.json

- SUMMARY.md (human dashboard)
- claim_ledger.jsonl (claim -> evidence -> sha256)
- run_metadata.json (environment + provenance)
- runner_transcript.txt (exact CLI output)
- Master zip: MARI_MASTER_RELEASE_<timestamp>.zip

This directory is the canonical record for citing results and rewriting papers.

---

## Narrative flow

The runner is structured so the story is clear on first reading:

1. Kernel framing (DEMO-34 first).
2. DRPT structure reveal (rows + columns: mirroring, inversion, repetition).
3. Legality and admissibility (survivors, smoothing discipline).
4. Tiling / extension (closed cycles repeat deterministically).
5. Flagship consequences (SM, GR, QG, cosmology) then breadth sweep.

Structure is shown before explanation. Evidence is shown before interpretation.

---

## Per-demo dashboard

Each demo prints a predictable mini-dashboard:

- Verdict: VERIFIED / NOT VERIFIED
- Gate score: PASS / FAIL counts
- Artifacts: vendored evidence count
- Structured exports: present or not
- Runtime
- Citation block: bundle + demo + artifact hash prefixes

All metrics follow, grouped by meaning. Nothing is truncated.

---

## Rewriting papers

Because the demo suite has changed, prior archives are no longer authoritative.

Papers should cite:
- the new AoR bundle seal,
- specific claim IDs from claim_ledger.jsonl,
- the corresponding evidence file and sha256.

This makes every claim traceable to execution.

---

## Public access

A GitHub Actions workflow can run the Master Suite publicly:
- anyone may view the latest run output,
- anyone may download the AoR archive,
- trusted collaborators may trigger runs manually.

---

## Design principles

- Determinism over convenience
- Structure before explanation
- Evidence before interpretation
- Completeness without noise
- Visual clarity without theatrics

The goal is not persuasion. The goal is legibility under scrutiny.
