# DEMO-36 — Big Bang Master Flagship (BB-36 capsule)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-36-big-bang-master-flagship-bb-36-capsule.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-36' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
PASS  Gate S5: n_s in (0.90,1.05)                                                n_s=0.964746
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-36 (BB) — Big Bang Master Flagship
> Structural Cosmology Closure + Two Independent Spectrum Bridges + Deterministic Teeth
> Referee-ready, portable, deterministic, no tuning.

> SUMMARY (one paragraph)
> A unique admissible prime triple (wU,s2,s3) is selected in a declared primary window by a transparent
> lane filter and an explicit coherence constraint. All numerical budgets/tolerances are then derived
> from that triple (q2,q3,eps,N,K). Using fixed BB-36 monomials, the triple deterministically generates
> a full "structural cosmology" parameter set (H0, Ω's, A_s, n_s, τ, ℓ1, δ_CMB). Two *independent*
> spectrum-level observables are then constructed from first principles (a debiased tilt proxy and a
> power-sum amplitude proxy), each audited against an admissible operator (Fejér tensor; nonnegative
> kernel) versus two illegal controls (sharp cutoff; signed HF-injecting control). Finally, deterministic
> counterfactual triples induce budget shifts and must degrade the primary signature ("teeth").

> NOTES FOR REFEREES
> - This is not a cosmological inference engine; it is an internally consistent, falsifiable demonstration.
> - No external data files are required.
> - Optional: CAMB (if installed) is used only for an informational TT first-peak check.

> Dependencies

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-36-big-bang-master-flagship-bb-36-capsule.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Gate S1: H0 in (50,80) km/s/Mpc                                            H0=70.449
PASS  Gate S2: omega_b h^2 in (0.015,0.035)                                      ombh2=0.022325
PASS  Gate S3: omega_c h^2 in (0.05,0.20)                                        omch2=0.062640
PASS  Gate S4: A_s in (1e-9,5e-9)                                                A_s=2.099e-09
PASS  Gate S5: n_s in (0.90,1.05)                                                n_s=0.964746
PASS  Gate S6: tau in (0.01,0.10)                                                tau=0.053979
PASS  Gate S7: ell1 in (150,350)                                                 ell1=219.949
PASS  Gate S8: deltaCMB in O(1e-5) band                                          delta=1.000e-05
PASS  Gate T1: admissible kernel nonnegative (Fejer tensor)                      kmin=-1.735e-18 tol=1e-12
PASS  Gate T2: illegal kernels have negative lobes (sharp + signed)              kmin_sharp=-4.652e-03 kmin_signed=-8.921e-01
PASS  Gate T3: admissible tilt matches structural target (tol=eps^3)             |Δ|=0.002211 tol=0.006086
PASS  Gate T4: sharp cutoff increases cutoff-band curvature                      curv_adm=0.091802 curv_sharp=1.006428 eps=0.182574
PASS  Gate T5: signed illegal injects HF beyond floor                            hf_signed=0.227137 floor=0.033333
PASS  Gate T6: >=3/4 counterfactual budgets miss by score margin                 strong=4/4 eps=0.182574
PASS  Gate A1: deltaCMB_struct in plausible band (order 1e-5)                    delta=1.000e-05
PASS  Gate A2: admissible proxy within eps                                       rel_err=0.156987 eps=0.182574
PASS  Gate A3: signed illegal worsens error vs admissible                        err_adm=0.156987 err_signed=0.366026
PASS  Gate A4: signed illegal injects HF beyond floor                            hf_signed=0.227137 floor=0.033333
```

## Reference checkpoints (from provided transcript)

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)                                               selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate S1: H0 in (50,80) km/s/Mpc                                            H0=70.449
PASS  Gate S2: omega_b h^2 in (0.015,0.035)                                      ombh2=0.022325
PASS  Gate S3: omega_c h^2 in (0.05,0.20)                                        omch2=0.062640
PASS  Gate S4: A_s in (1e-9,5e-9)                                                A_s=2.099e-09
PASS  Gate S5: n_s in (0.90,1.05)                                                n_s=0.964746
PASS  Gate S6: tau in (0.01,0.10)                                                tau=0.053979
PASS  Gate S7: ell1 in (150,350)                                                 ell1=219.949
PASS  Gate S8: deltaCMB in O(1e-5) band                                          delta=1.000e-05
PASS  Gate T1: admissible kernel nonnegative (Fejer tensor)                      kmin=-1.735e-18 tol=1e-12
PASS  Gate T2: illegal kernels have negative lobes (sharp + signed)              kmin_sharp=-4.652e-03 kmin_signed=-8.921e-01
PASS  Gate T3: admissible tilt matches structural target (tol=eps^3)             |Δ|=0.002211 tol=0.006086
PASS  Gate T4: sharp cutoff increases cutoff-band curvature                      curv_adm=0.091802 curv_sharp=1.006428 eps=0.182574
PASS  Gate T5: signed illegal injects HF beyond floor                            hf_signed=0.227137 floor=0.033333
PASS  Gate T6: >=3/4 counterfactual budgets miss by score margin                 strong=4/4 eps=0.182574
PASS  Gate A1: deltaCMB_struct in plausible band (order 1e-5)                    delta=1.000e-05
```

Transcript excerpt (for quick visual diff):

```text
================================================================================================
STAGE 4 — delta_CMB amplitude bridge (power-sum proxy + controls + teeth)
================================================================================================
deltaCMB_struct = 1.000475284975e-05  (ratio to 1e-5: 1.000475)
Truth tier (Fejer@K_truth=31): tot=25.160688  hf=0.000172  (delta matches by construction)
admissible (Fejer@K=15): delta=8.434139561014e-06  rel_err=0.156987  hf=0.000381
sharp      (cut@K=15):   delta=1.201479769851e-05  rel_err=0.200909  hf=0.000000
signed     (+/-@K=15):   delta=1.366675022496e-05  rel_err=0.366026  hf=0.227137

PASS  Gate A1: deltaCMB_struct in plausible band (order 1e-5)                    delta=1.000e-05
PASS  Gate A2: admissible proxy within eps                                       rel_err=0.156987 eps=0.182574
PASS  Gate A3: signed illegal worsens error vs admissible                        err_adm=0.156987 err_signed=0.366026
PASS  Gate A4: signed illegal injects HF beyond floor                            hf_signed=0.227137 floor=0.033333
COUNTERFACTUAL TEETH (budget K)
Primary score: K=15 rel_err=0.156987 hf=0.000381 score=0.157367
CF Triple(wU=409, s2=263, s3=239)  K= 5  rel_err=0.409393  hf=0.003227  score=0.412619  miss=True
CF Triple(wU=409, s2=263, s3=307)  K= 5  rel_err=0.409393  hf=0.003227  score=0.412619  miss=True
CF Triple(wU=409, s2=367, s3=239)  K= 5  rel_err=0.409393  hf=0.003227  score=0.412619  miss=True
CF Triple(wU=409, s2=367, s3=307)  K= 5  rel_err=0.409393  hf=0.003227  score=0.412619  miss=True
PASS  Gate A5: >=3/4 counterfactual budgets miss by score margin                 strong=4/4 eps=0.182574

================================================================================================
STAGE 5 — Optional CAMB TT closure (first peak vs ell1_struct)
================================================================================================
CAMB not available: ModuleNotFoundError("No module named 'camb'")
PASS  Gate C0: CAMB stage skipped (install camb to enable)

================================================================================================
STAGE 6 — Artifacts (JSON + optional plot) + determinism hash
================================================================================================
PASS  Results JSON not written (filesystem unavailable)                          PermissionError(1, 'Operation not permitted')
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
