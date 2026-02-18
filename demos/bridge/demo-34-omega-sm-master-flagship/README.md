# DEMO-34 - Omega to Standard Model Bridge (Master Flagship)

A deterministic, self-auditing demo with explicit gates. The claim is operational: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python demos/bridge/demo-34-omega-sm-master-flagship/demo.py
```

If your folder name differs, locate by demo number:

```bash
python "$(ls -d demos/**/demo-34-* 2>/dev/null | head -n 1)/demo.py"
```

Falsification condition: any printed FAIL gate, a missing or invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- Zero tuning and zero fitted parameters. Any external reference values are evaluation-only and do not feed back into selection.
- Deterministic. No randomness; no network; no hidden configuration.
- Governance-first. The selector must be identifiable under declared policies and must fail under ablations.

## Scope (what this demo claims)

DEMO-34 is a bridge certificate connecting the Omega selection channel to Standard-Model-facing outputs under a strict audit posture:

- It produces the canonical ordered triple under the declared selector contract.
- It includes necessity ablations that must explode (loss of uniqueness) when load-bearing gates are removed.
- It prints rational anchor values used downstream (evaluation-only comparisons are fenced).
- It emits a determinism hash and a final VERIFIED verdict when all gates pass.

## Run instructions

- Dependencies: Python 3.10+ (core logic is stdlib-only; optional plotting may require additional packages depending on your environment).
- Execute:

```bash
python "$(ls -d demos/**/demo-34-* 2>/dev/null | head -n 1)/demo.py"
```

## Pass/Fail contract (gates)

Dreat any regression as a hard failure:

- The demo must print the selected triple and confirm it matches the canonical survivor.
- Any ablation marked as required must increase the survivor count (explosion) or break invariance as declared.
- Any printed FAIL gate falsifies the run.

## Reference checkpoints

For audit, use the AoR surfaces:

- Demo stdout log for DEMO-34
- constants_master.csv for rational anchors
- falsification_matrix.csv for required failures
- run_reproducibility.csv for determinism expectations

## Reviewer notes

- This README intentionally focuses on reproducible observables: gates, explosion counts, and deterministic identity surfaces.
- For discrepancies, attach full stdout and your Python version.
