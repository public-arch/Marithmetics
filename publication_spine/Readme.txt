# Publication Spine

This folder contains the initial public release papers for the Marithmetics program.

## Start here

* `00_Foundations/README_EXEC.md`
  Executive verification guide (Tier-A / A2). This is the fastest way to understand what is certified and how to reproduce and verify the canonical results bundle.

* `00_Foundations/Marithmetics_Foundations_Authority.md`
  Full authority paper (referee-grade). Includes definitions, worked derivations, and commit-pinned citations into the canonical evidence bundle.

## Folder structure

* `00_Foundations/`
  Tier-A / A2 certification layer and reproducibility capsule.

* `01_DOC/`
  Deterministic Operator Calculus (DOC). Formal operator discipline and ZFC-compatible framework used by the certificates.

* `02_GUM/`
  GUM paper(s) and supporting publication artifacts.

* `03_NT_1-4/`
  Number Theory / Mathematics track papers (NT-1 through NT-4). Formal theorems, invariant basis, and residual equivalence framework.

* `04_PH_1-4/`
  Physics track papers (PH-1 through PH-4). Downstream physical façade and observer layer, with explicit tier discipline.

* `05_Supplements/`
  Longer authority monographs and supporting documents not required to verify the core A2 closure.

## Canonical verification record

All Tier-A / A2 claims in this release point to a single canonical results bundle produced by the repo runner and sealed by SHA-256. Each paper cites:

* the exact code at a commit-pinned URL, and
* the exact stdout/stderr artifact inside the canonical results bundle, and
* the bundle seal hash.

## Licensing and usage

Unless otherwise stated, documents and code in this folder are released under the same license terms as the repository root.
