# DEMO-54 — Master Flagship Demo (single-file determinism)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-54-master-flagship-demo-single-file-determinism.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-54' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
Result: VERIFIED
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO 54 - MASTER FLAGSHIP DEMO (first‑principles, deterministic, single‑file)

> What this demo does (in a single run, no tuning knobs):

> 1) Symmetry‑Constrained Fixed‑Point (SCFP++) selection (Demo‑33 window)
>    - From first principles (primality, residue classes, Euler totient density),
>      deterministically selects a unique admissible triple (wU, s2, s3).

> 2) Gauge‑sector rationals from the selected triple
>    - α0⁻¹ := wU
>    - sin²θW := 7 / (wU − s2)
>    - αs(MZ) := 2 / odd_part(wU − 1)

> 3) QCD scale from αs(MZ)
>    - Λ_QCD computed both at 1‑loop (closed form) and 2‑loop (numeric inversion).

> 4) Vacuum energy suppression (no tuning)
>    - Uses a fixed Zel’dovich scaling ρ ~ Λ^6 / M_Pl^2,
>      multiplied by a fixed 2‑loop factor (1/(16π²))² and a fixed dressing 1/(1+αs).
>    - Compares to an observational overlay (H0, ΩΛ) for a ratio test.

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-54-master-flagship-demo-single-file-determinism.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple                                                count=1
PASS  Field emergence gate (determinism + ablations)
```

## Reference checkpoints (from provided transcript)

- Spec SHA256: `Spec SHA256 : f097de5783e530a82daff84bfbffdb85c304727d637503a819dc1cba5015e7d9`

- Determinism hash: `4a1a22e3d6f7201385c0b7600f8bf01c79512fb2948c14ca73d44afa50b26eb1`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
Spec SHA256 : f097de5783e530a82daff84bfbffdb85c304727d637503a819dc1cba5015e7d9
PASS  Primary equals (137,107,103)                                            selected=(137, 107, 103)
PASS  sin²θW matches lawbook                                                  sin2W=0.2333333333  ref=0.2333333333
PASS  αs(MZ) matches lawbook                                                  alpha_s=0.1176470588  ref=0.1176470588
PASS  Prediction: ρΛ (GeV⁴)                                                   rho_pred_GeV4=2.83595282e-47
determinism_sha256: 4a1a22e3d6f7201385c0b7600f8bf01c79512fb2948c14ca73d44afa50b26eb1
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
                                     STAGE 9 — DETERMINISM HASH                                     
====================================================================================================
determinism_sha256: 4a1a22e3d6f7201385c0b7600f8bf01c79512fb2948c14ca73d44afa50b26eb1

====================================================================================================
                                           FINAL VERDICT                                            
====================================================================================================
PASS  Gauge sector laws match                                               
PASS  Vacuum suppression (<1% + ablations)                                  
PASS  Neutrino sector closure (Δm² + Σmν)                                   
PASS  Math canaries (δ + C2)                                                
PASS  Field emergence gate (determinism + ablations)                        
PASS  GR emergence (inverse-square + illegal ringing + counterfactual separation)

PASS  FLAGSHIP VERIFIED (selection + vacuum + math + field + GR + counterfactuals)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
