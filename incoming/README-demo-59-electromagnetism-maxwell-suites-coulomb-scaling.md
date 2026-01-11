# DEMO-59 — Electromagnetism (Maxwell suites + Coulomb scaling)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-59-electromagnetism-maxwell-suites-coulomb-scaling.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-59' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
slope log|E| vs log r (expect ~ -2): truth=-1.906371 adm=-1.791637 sharp=-1.766929 signed=-2.686532
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-59 — Electromagnetism

> Scope (what this demo is):
>   A deterministic, referee-facing demonstration of *operator admissibility* for
>   Fourier-domain filters used inside discrete field solvers.

>   The demo consists of two benchmark suites:

>     (A) Electrostatics (3D): Poisson solver with a neutralized point charge on a periodic lattice.
>         Observable: Coulomb scaling |E(r)| ~ r^{-2} and stability of r^2⟨|E|⟩ across shells.

>     (B) Maxwell-class operators (2D): filter admissibility diagnostics that are ubiquitous in
>         wave/field solvers.
>         Observable 1: Gibbs/overshoot on a discontinuous step (sharp cutoff should overshoot).
>         Observable 2: broadband distortion on a smooth Gaussian bump (budget teeth).

> First-principles definitions (used throughout):
>   - Grid: periodic lattice with N points per dimension, unit lattice spacing.
>   - Fourier transform: numpy.fft (deterministic for fixed inputs).
>   - Discrete Laplacian eigenvalues: λ(k) = -4 Σ_d sin^2(π k_d / N), consistent with the

## Run instructions

- Dependencies: Python 3.10+ plus `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-59-electromagnetism-maxwell-suites-coulomb-scaling.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS Unique admissible triple in primary window                                      count=1
PASS Gate E1: truth slope near -2                                                    slope=-1.906371 tol=0.365148
PASS Gate E2: admissible slope near -2                                               slope=-1.791637 tol=0.365148
PASS Gate E3: signed control retains HF beyond Kp (operator falsifier)               hf_adm=0.000000 hf_signed=1.000000 floor=0.033333
PASS Gate E4: some non-admissible control has stronger ringing curvature             curv_adm=0.002522 curv_max=0.029511 eps=0.182574
PASS Gate T_E: >=3/4 counterfactuals degrade by (1+eps)                              strong=4/4 eps=0.182574
PASS Gate M1: Fejér reconstruction is bounded for a step                             overshoot=0.000e+00
PASS Gate M2: Sharp cutoff exhibits Gibbs overshoot                                  overshoot=0.068438 floor=0.033333
PASS Gate T_M: >=3/4 counterfactuals degrade by (1+eps)                              strong=4/4 eps=0.182574
PASS DEMO-59 VERIFIED (electrostatics + maxwell suites + teeth)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `5a74664c0e5719d72eb2b5f8345829aa8531175b1fdb5b39da45264d68d77530`

- Verdict line: `PASS DEMO-59 VERIFIED (electrostatics + maxwell suites + teeth)`

Selected printed checkpoints:

```text
PASS Primary equals (137,107,103)                                                    selected=Triple(wU=137, s2=107, s3=103)
PASS Gate E1: truth slope near -2                                                    slope=-1.906371 tol=0.365148
PASS Gate E2: admissible slope near -2                                               slope=-1.791637 tol=0.365148
PASS Gate E3: signed control retains HF beyond Kp (operator falsifier)               hf_adm=0.000000 hf_signed=1.000000 floor=0.033333
PASS Gate E4: some non-admissible control has stronger ringing curvature             curv_adm=0.002522 curv_max=0.029511 eps=0.182574
PASS Gate T_E: >=3/4 counterfactuals degrade by (1+eps)                              strong=4/4 eps=0.182574
PASS Gate M1: Fejér reconstruction is bounded for a step                             overshoot=0.000e+00
PASS Gate M2: Sharp cutoff exhibits Gibbs overshoot                                  overshoot=0.068438 floor=0.033333
PASS Gate T_M: >=3/4 counterfactuals degrade by (1+eps)                              strong=4/4 eps=0.182574
determinism_sha256: 5a74664c0e5719d72eb2b5f8345829aa8531175b1fdb5b39da45264d68d77530
FINAL VERDICT
```

Transcript excerpt (for quick visual diff):

```text
==================================================================================================
DETERMINISM HASH
==================================================================================================
determinism_sha256: 5a74664c0e5719d72eb2b5f8345829aa8531175b1fdb5b39da45264d68d77530

==================================================================================================
FINAL VERDICT
==================================================================================================
PASS DEMO-59 VERIFIED (electrostatics + maxwell suites + teeth)
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
