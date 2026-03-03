# DEMO-71 — One Action Master Flagship

> **Claim:** A single action principle manifests as three structurally-protected laws—classical Noether conservation, quantum unitarity, and field energy stability—deterministically derived from the primary triple (137, 107, 103), with illegal controls violating all laws and counterfactual budgets degrading accuracy by at least (1+eps).

---

## What this demo computes

A comprehensive deterministic demo showing how a single action principle protects three distinct domains:
1. Selects the primary triple (wU, s2, s3) = (137, 107, 103) deterministically
2. Derives structural invariants (q2, q3, eps) and captures at least 4 deterministic counterfactual triples
3. **Classical domain:** Demonstrates symplectic/Noether conservation (angular momentum) under velocity-Verlet (VV) variational integrator; energy drift bounded by eps²; Jacobian determinant ≈ 1 within eps³
4. **Quantum domain:** Demonstrates unitarity (norm drift ≤ eps⁴) and reversibility (forward+backward error ≤ eps³) under Crank–Nicolson (CN) time stepping; accuracy vs exact within eps
5. **Field domain:** Demonstrates energy stability under leapfrog (variational update); energy drift bounded by eps³
6. Contrasts with illegal controls: non-variational Euler breaks Noether; non-unitary Euler breaks unitarity; anti-action sign-flip exhibits blow-up; Wick illegal (wrong Schrödinger evolution) destroys accuracy
7. Demonstrates counterfactual teeth: deterministic budget shifts (K reduced by q3→3q3) degrade all trajectory/norm/energy errors by ≥(1+eps)

## Falsification contract

1. Any printed FAIL gate falsifies the demo.
2. The selected triple must equal (137, 107, 103); any other value falsifies.
3. Must capture ≥4 deterministic counterfactual triples; fewer falsifies.
4. Structural invariants must be locked: q2=30, q3=17, v2U=3; any deviation falsifies.
5. All classical gates (C1–C6, CT) must pass: Noether drift, symplectic area, energy bound, and illegal control violations.
6. All quantum gates (Q1–Q6, QT) must pass: CN unitarity, reversibility, accuracy, and illegal control violations.
7. All field gates (F1+) must pass: leapfrog energy drift ≤ eps³.
8. Counterfactual teeth (CT, QT) must show degradation by ≥(1+eps); weaker degradation falsifies.
9. Determinism hash must match reference; any change falsifies.

## Controls

- **Illegal operators:** Non-variational Euler (breaks Noether); non-unitary Euler (breaks unitarity); anti-action sign-flip (exhibits blow-up); Wick illegal (wrong quantum evolution)
- **Counterfactuals:** Deterministic budget shifts (dt increased; q3→3q3 reducing K) must degrade all observable errors
- **Ablations:** Removing action principle protection must cause illegal methods to violate conservation laws

## Dependencies

Python 3.10+, numpy. Optional: matplotlib (for artifact plotting only).

## Run

```bash
python demo.py [--artifacts]
```
