# DEMO-59 — Electromagnetism (Maxwell suites + Coulomb scaling)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) yields budgets; two benchmark suites (electrostatics 3D Poisson with point charge → Coulomb |E(r)| ~ r⁻² scaling; Maxwell 2D filters → Gibbs overshoot on step + broadband distortion on bump) demonstrate admissible Fejér filters achieve near-truth slopes/overshoots, illegal controls fail, and counterfactual triples degrade by fixed eps margin.

---

## What this demo computes

Deterministic first-principles computational audit:
- Deterministic selection: unique prime triple (wU, s2, s3) = (137, 107, 103) via fixed rules.
- Budget derivation: eps = 1/√q2; N (grid size), K (spectral cutoff) from triple invariants.

Suite A (Electrostatics 3D):
- Poisson solver: ΔΦ = ρ on 3D periodic lattice with neutralized point charge; exact FFT eigenvalues.
- Observable: Coulomb scaling |E(r)| ~ r⁻² and stability of r²⟨|E|⟩ across shells.
- Operator comparison: admissible Fejér (slope ≈ -1.79, truth ≈ -1.91), illegal sharp/signed controls.
- Gate E1-E4: slope recovery, ringing, HF injection, curvature tests.
- Gate T_E: ≥3/4 counterfactuals degrade by (1+eps).

Suite B (Maxwell 2D):
- Observable 1: Gibbs/overshoot on discontinuous step; Fejér preserves boundedness, sharp shows overshoot ≥ eps².
- Observable 2: broadband distortion on smooth Gaussian bump; Fejér minimizes, illegal controls worsen.
- Gate M1-M2: overshoot contracts, sharp cutoff Gibbs injection.
- Gate T_M: ≥3/4 counterfactuals degrade by (1+eps).

First-principles definitions:
- Grid: periodic lattice, unit spacing.
- Fourier transform: numpy.fft (deterministic for fixed inputs).
- Discrete Laplacian: λ(k) = -4 Σ_d sin²(πk_d/N) (standard 2nd-order periodic Laplacian).
- Admissible: Fejér weights (triangular Fourier multipliers, nonnegative real-space kernel).
- Illegal: sharp cutoff, signed HF complement (both with negative kernel lobes).

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Coulomb slope must be near -2 (truth ≈ -1.91 ± eps).
5. Admissible slopes/overshoots must be ≤ illegal controls.
6. Sharp/signed Gibbs overshoot must exceed Fejér (or zero).
7. Counterfactual triples must degrade in both suites by (1+eps).

## Controls

- **Illegal operators:** Sharp spectral cutoff (negative kernel lobes in all suites), signed HF complement (stronger lobes).
- **Counterfactuals:** Alternative triples from extended window; must degrade slopes and overshoots by (1+eps) in all suites.
- **Ablations:** Higher-budget Fejér provides truth reference without external data.

## Dependencies

Python 3.10+ with NumPy.

## Run

```bash
python demo.py
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
