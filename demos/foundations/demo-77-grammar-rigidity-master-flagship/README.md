# DEMO-77 — Grammar Rigidity Master Flagship

> **Claim:** The admissibility grammar that selects the primary triple is rigid — perturbing any constraint or relaxing any gate produces either the same unique solution or an exploding solution space, never a smooth continuum of alternatives.

---

## What this demo computes

This demo audits the structural rigidity of the selection grammar itself. It answers a foundational question: is the triple (137, 107, 103) the output of a fragile, finely-tuned filter, or is the grammar rigid?

**Constraint perturbation.** Each admissibility gate (C1–C4: primality, residue, modulus bound, totient ratio) is perturbed independently. The demo measures whether the output triple changes and, if so, how the survivor pool responds.

**Relaxed-grammar outcomes.** Gates are dropped one at a time. If dropping a gate preserves the triple, that gate is redundant. If dropping a gate causes pool expansion (1 → many), the gate is necessary and the grammar is tight.

**Counterfactual grammar variants.** Alternative residue sets, alternative moduli, and shifted windows are tested. The demo reports how many variants produce any admissible triple at all, and how many produce the same triple.

**Rigidity scan.** A large variant space (5,832 grammar configurations) is enumerated to confirm the selection is isolated — not embedded in a family of equivalent grammars.

**Determinism certificate.** PASS/FAIL posture with explicit counterfactual and relaxed-grammar outcomes, plus a determinism hash.

---

## Falsification contract

This demo is falsified if:

1. A smooth family of alternative grammars produces the same triple — this would mean the selection is not rigid but embedded in a continuum of equivalent grammars.
2. Dropping a gate does not expand the survivor pool — the gate was not doing work.
3. The determinism hash changes between runs.
4. Any gate prints FAIL.

---

## Controls

- **Gate ablation:** Each gate dropped independently; pool expansion measured.
- **Grammar variants:** Alternative moduli and residue sets tested systematically.
- **Rigidity scan:** 5,832 variants enumerated to confirm grammatical isolation.

---

## Dependencies

Python 3.10+ standard library only.

## Run

```bash
python demo.py
python demo.py --json    # emit JSON artifact
```
