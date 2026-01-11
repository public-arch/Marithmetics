# DEMO-53 — Lawbook Emergence (Noether / inverse-square / isotropy / unitarity)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-53-lawbook-emergence-noether-inverse-square-isotropy-unitarity.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-53' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
✅ PASS: small-k rotational isotropy selects w2=1/6 uniquely (no single-scale tuning)
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> ====================================================================
> Demo 53 - LAWBOOK EMERGENCE
> Stdlib-only · stdout-only · no file I/O
> What this demo does (referee-readable):
> - Stage 0: derive flagship primes (wU,s2,s3) by explicit SCFP++ lane search
> - Stage 1: Noether visibility: break time-translation invariance and watch energy drift
> - Stage 2: Inverse-square selection: sweep p and show p=2 is the unique flux fixed point
> - Stage 3: Isotropic Laplacian: show continuum (small-k) isotropy selects 9-pt stencil w2=1/6
> - Stage 4: Unitarity selection: sweep θ-method; only θ=1/2 is norm-preserving (Crank–Nicolson)
> No hidden knobs:
> dt_unity = (s3/wU)/5, dt_noether = (s3/wU)/15, dx_unity = q3 (lane modulus)
> ====================================================================

## Run instructions

- Dependencies: Stdlib-only (no third-party packages).

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-53-lawbook-emergence-noether-inverse-square-isotropy-unitarity.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
✅ PASS: primes derived by finite lane-gated search (no fitting)
✅ PASS: energy conservation emerges only when time-translation invariance holds (eps=0)
✅ PASS: Gauss flux constancy selects p=2 (inverse-square) as unique law form
✅ PASS: small-k rotational isotropy selects w2=1/6 uniquely (no single-scale tuning)
✅ PASS: θ=0.5 is the unique unitary fixed point (ratio≈1.373e+10)
```

## Reference checkpoints (from provided transcript)

Selected printed checkpoints:

```text
selected triple (wU,s2,s3)             (137, 107, 103)
derived alpha_s(MZ)=2/q3               0.117647058824
LAWBOOK EMERGENCE CERTIFICATE (v1)
```

Transcript excerpt (for quick visual diff):

```text
0.9   -4.777539e-04
 1.0   -5.971487e-04
spark(|drift|): @#+=: :=+#@
✅ PASS: θ=0.5 is the unique unitary fixed point (ratio≈1.373e+10)

======================================================================
LAWBOOK EMERGENCE CERTIFICATE (v1)
======================================================================
Primes from SCFP++:             ✅
Noether visibility sweep:        ✅
Inverse-square p-selection:      ✅
Isotropic Laplacian (small-k):   ✅
Unitarity θ=1/2 selection:       ✅
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
