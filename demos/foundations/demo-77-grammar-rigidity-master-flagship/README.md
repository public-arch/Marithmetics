# DEMO-77 — Grammar Rigidity Master Flagship

## What this demo is
DEMO-77 is a deterministic audit of grammar rigidity and selector robustness.

It is designed to answer a hostile-referee question:
If we perturb the admissibility grammar or relax a constraint, does the solution remain rigid,
or does the solution space explode?

This demo is not a new physics claim. It is an auditable structural stress test of the selection
and closure grammar.

## What it produces
- A stdout-first certificate (PASS/FAIL posture)
- Explicit counterfactual and relaxed-grammar outcomes
- Determinism hash (run identity)

AoR bundling treats stdout/stderr as canonical evidence.

## How to run
```bash
python demos/foundations/demo-77-grammar-rigidity-master-flagship/demo.py
```

## AoR notes
This demo is auto-discovered by the AoR bundler via `demos/*/*/demo.py`.

To include it in an AoR, run the Master Suite:

```bash
python -m audits.run_master_suite --verbosity compact
```
