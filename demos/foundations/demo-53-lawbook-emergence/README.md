# DEMO-53 — Lawbook Emergence

> **Claim:** Four foundational physical laws — Noether's theorem, inverse-square force scaling, isotropic diffusion, and unitary time evolution — emerge as the unique selections under discrete structure constraints, not as assumptions imposed from outside.

---

## What this demo computes

This demo derives physical laws the way the rest of the suite derives physical constants: by selection, not postulation.

**Stage 0 — SCFP++ selection.** The primary triple (137, 107, 103) is derived by explicit lane search. All numerical budgets in subsequent stages are set by the triple — no hidden knobs. Specifically: dt_unity = (s3/wU)/5, dt_noether = (s3/wU)/15, dx_unity = q3.

**Stage 1 — Noether visibility.** Time-translation invariance is broken in a discrete Hamiltonian system. Energy drift is measured as a function of the symmetry-breaking parameter ε. The demo shows that energy conservation is not assumed — it emerges when and only when the symmetry holds.

**Stage 2 — Inverse-square selection.** The force-law exponent p is swept continuously. p = 2 is the unique value for which total flux through a closed surface is independent of surface radius — the Gauss's-law fixed point. The inverse-square law is selected, not imposed.

**Stage 3 — Isotropic Laplacian.** On a discrete lattice, the continuum Laplacian has a one-parameter family of finite-difference stencils. Isotropy in the small-k (long-wavelength) limit uniquely selects the 9-point stencil with weight w₂ = 1/6. This is the discrete expression of rotational invariance.

**Stage 4 — Unitarity selection.** The θ-method family for time evolution (θ ∈ [0, 1]) is swept. Only θ = 1/2 (Crank–Nicolson) preserves the L² norm exactly. This is not a numerical preference — it is the unique unitary fixed point of the discretization family.

---

## Falsification contract

This demo is falsified if:

1. A second flux-stable exponent besides p = 2 exists in the sweep.
2. A stencil weight besides w₂ = 1/6 achieves small-k isotropy.
3. A θ-method value besides 1/2 preserves unitarity.
4. Any numerical budget depends on something other than the triple.
5. Any gate prints FAIL.

---

## Controls

Each stage is its own sweep — the wrong answers are computed alongside the right one, making the selection visible rather than assumed.

---

## Dependencies

Python 3.10+ standard library only.

## Run

```bash
python demo.py
```
