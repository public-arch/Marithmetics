# DEMO-66 – Quantum Gravity Master Flagship (v4)

## Purpose
The canonical quantum gravity flagship demo for the repository. This demo is designed to be deterministic, self-auditing, and release-grade.

## Run
```bash
python demos/quantum_gravity/demo-66-quantum-gravity-master-flagship-v4/demo.py
```

## What "PASS" means
- All gates print PASS
- Deterministic outputs match across runs (within stated tolerances)
- Illegal controls fail as expected
- Counterfactuals do not accidentally pass

## What falsifies it
- Any printed FAIL
- Missing certificate sections
- Any material drift in checkpoint values without an explicit tolerance statement
