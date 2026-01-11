# DEMO-64 — Base–Gauge Invariance of a Deterministic Integer Selector

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-64-base-gauge-invariance-of-a-deterministic-integer-selector.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-64' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  Gate F: negative control triggers mismatch (sensitivity)                 mismatches=11/11
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-64 — Base-Gauge Invariance of a Deterministic Integer Selector

> This demo isolates a single claim:

>   The selection procedure that fixes the primary integer triple (wU, s2, s3)
>   is invariant under a change of numeral base used to *encode* the integers.

> In other words, the selector depends on the integers themselves (and integer
> invariants), not on the human-facing encoding (binary, octal, decimal, hex, …).

> Why this matters
> If a new framework claims it produces a unique triple deterministically, a
> skeptical reader must be able to rule out "formatting accidents": e.g.,
> implicit decimal parsing, locale-dependent string handling, float round-trip
> errors, or hidden dependence on a specific base.

> What this script does
> 1) Defines a small, fully explicit selector using only elementary number theory:
>    - primality
>    - congruences modulo q

## Run instructions

- Dependencies: Stdlib-only (no third-party packages).

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-64-base-gauge-invariance-of-a-deterministic-integer-selector.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple (baseline)                                      count=1
PASS  Gate G0: encode/decode contract holds (no round-trip failures)           failures=0
PASS  Gate G1: triple invariant across bases
PASS  Gate G2: lane survivor pools invariant across bases
PASS  Gate F: negative control triggers mismatch (sensitivity)                 mismatches=11/11
PASS  DEMO-64 VERIFIED (base-gauge invariance + falsifier sensitivity)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `9daf73e8797ad7b279799c178a9271f99a97af35b5b5e474675e6dd1f6c46d8c`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
Admissible triples (after T1–T2): [(137, 107, 103)]
PASS  Gate G0: encode/decode contract holds (no round-trip failures)           failures=0
PASS  Gate G1: triple invariant across bases
PASS  Gate G2: lane survivor pools invariant across bases
PASS  Gate F: negative control triggers mismatch (sensitivity)                 mismatches=11/11
determinism_sha256: 9daf73e8797ad7b279799c178a9271f99a97af35b5b5e474675e6dd1f6c46d8c
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
DETERMINISM HASH
==================================================================================================
determinism_sha256: 9daf73e8797ad7b279799c178a9271f99a97af35b5b5e474675e6dd1f6c46d8c

==================================================================================================
FINAL VERDICT
==================================================================================================
PASS  DEMO-64 VERIFIED (base-gauge invariance + falsifier sensitivity)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
