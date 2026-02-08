# DEMO-78 — Midlift Protocol (Master Flagship)

## Purpose
DEMO-78 formalizes and audits the "midlift" correction: the claim that treating the 1D number line as the primitive is a lift, and that the correct primitive is a 0D substrate with lawful residue structure.

This demo exists to:
- Make the midlift claim explicit and testable
- Provide Authority-grade certificate posture for how the program defines 0D substrate vs 1D lift
- Supply designed FAILs and negative controls so the narrative cannot hide behind interpretation

## What this demo is not
- Not a curve-fit to external targets.
- Not a replacement for downstream physics closures.
- Not claiming proof of all physics by itself.

## Run
```bash
python demos/foundations/demo-78-midlift-protocol-master-flagship/demo.py
```

## Expected outputs
- stdout certificate log (canonical evidence)
- optional artifacts (if the demo emits any)

The AoR bundler captures stdout/stderr as canonical evidence.
