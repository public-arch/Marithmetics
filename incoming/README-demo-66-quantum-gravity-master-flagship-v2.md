# DEMO-66 — Quantum Gravity Master Flagship (v2)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-66-quantum-gravity-master-flagship-v2.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-66' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
r_plus=2.000000  r_ph=3.000000  b_ph=5.196152
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-66 — QUANTUM GRAVITY MASTER FLAGSHIP (Weak→Strong Field) — REFEREE READY (Self‑Contained)

> Goal
> ----
> A single, deterministic, first‑principles demo that:
>   1) Selects the primary triple (137,107,103) from residue + v2 coherence rules.
>   2) Derives a canonical discrete scale D* from base structure alone (LCM of (b−1)).
>   3) Produces an eps0 scale table from a locked κ*(β,N) ledger (no fitting).
>   4) Fits a locked RG table to extract an effective coupling g_eff (no tuning).
>   5) Builds a lawful screening law α_eff(ε) (weak→strong) and a ringdown proxy.
>   6) Extends to strong‑field geometry (horizon + shadow + ISCO + ringdown) with:
>        - CONTROL_OFF (pure GR baseline)
>        - ILLEGAL control (Θ‑palette blow‑up; loses the horizon)
>        - Counterfactual teeth (budget reduction ⇒ D shrinks ⇒ α inflates ⇒ observables miss)
>   7) Emits a determinism SHA‑256 over the full report.

> Design principles
> • Deterministic: no RNG; all outputs fixed by the lane rules and locked tables.
> • First‑principles: all “knobs” are derived from the triple or from declared canonical bases.
> • Controls + teeth: illegal operators and counterfactual budgets must fail by explicit margins.

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-66-quantum-gravity-master-flagship-v2.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
✅  Gate S1: unique coherent U(1) survivor                                   count=1
✅  Gate S2: primary equals (137,107,103)                                    selected=Triple(wU=137, s2=107, s3=103)
✅  Gate S3: captured >=4 counterfactual triples                             found=4
✅  Gate I: invariants match locked values (q2=30,q3=17,v2U=3)               (q2,q3,v2U)=(30,17,3)
✅  Gate D1: D* equals 1170 for canonical bases                              D*=1170
✅  Gate D2: exp(-sqrt(D*)/3) matches locked value                           computed=1.117586236861e-05 expected=1.117586236861e-05
✅  Gate E1: best (beta,N) achieves <1% closure to 1e-5                      best=(beta=8,N=96) eps0=9.972e-06 |ratio-1|=0.00279887
✅  Gate E2: canonical (beta=8,N=96) also achieves <1% closure               eps0=9.972011253995e-06 |ratio-1|=0.00279887
✅  Gate DF: base‑ablation (drop base 27) breaks eps0 closure by >>1         D_ablate=45 eps0_ablate=9.537e-02
✅  Gate T_eps0: budget‑limited base set degrades score by (1+eps)           scoreP=0.00279887 scoreCF=9535.52 1+eps=1.18257
✅  Gate RG1: R_inf > 1 (nontrivial positive coupling)                       R_inf=1.058064
✅  Gate RG2: SSE small (table consistent with 1/D^2 scaling)                SSE=3.626e-05
✅  Gate RG3: g_eff in sane nonzero band                                     g_eff=4.839e-03
✅  Gate T_rg: reduced RG sample degrades prediction error by (1+eps)        residP=3.980e-03 errCF=9.111e-03 1+eps=1.183
✅  Gate Sc1: alpha_eff monotone in eps (Mercury < DP < BH)                  9.210e-11 < 4.102e-04 < 4.839e-03
✅  Gate Sc2: BH saturates to g_eff (<=1e-12 abs)                            |alpha_bh-g_eff|=0.000e+00
✅  Gate Rd1: delta_f bounded by cap and nonzero                             df=4.558e-03 cap=0.3
✅  Gate Rd2: delta_tau bounded by cap and nonzero                           dtau=4.534e-03 cap=0.3
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `ab0d7c9df5ad5d9c0e599934ed534b9c82cb5a3580934a51810d91146abe59ad`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
✅  Gate S1: unique coherent U(1) survivor                                   count=1
✅  Gate S2: primary equals (137,107,103)                                    selected=Triple(wU=137, s2=107, s3=103)
✅  Gate S3: captured >=4 counterfactual triples                             found=4
✅  Gate I: invariants match locked values (q2=30,q3=17,v2U=3)               (q2,q3,v2U)=(30,17,3)
✅  Gate D1: D* equals 1170 for canonical bases                              D*=1170
✅  Gate D2: exp(-sqrt(D*)/3) matches locked value                           computed=1.117586236861e-05 expected=1.117586236861e-05
✅  Gate E1: best (beta,N) achieves <1% closure to 1e-5                      best=(beta=8,N=96) eps0=9.972e-06 |ratio-1|=0.00279887
✅  Gate E2: canonical (beta=8,N=96) also achieves <1% closure               eps0=9.972011253995e-06 |ratio-1|=0.00279887
✅  Gate DF: base‑ablation (drop base 27) breaks eps0 closure by >>1         D_ablate=45 eps0_ablate=9.537e-02
✅  Gate T_eps0: budget‑limited base set degrades score by (1+eps)           scoreP=0.00279887 scoreCF=9535.52 1+eps=1.18257
✅  Gate RG1: R_inf > 1 (nontrivial positive coupling)                       R_inf=1.058064
✅  Gate RG2: SSE small (table consistent with 1/D^2 scaling)                SSE=3.626e-05
✅  Gate RG3: g_eff in sane nonzero band                                     g_eff=4.839e-03
✅  Gate T_rg: reduced RG sample degrades prediction error by (1+eps)        residP=3.980e-03 errCF=9.111e-03 1+eps=1.183
✅  Gate Sc1: alpha_eff monotone in eps (Mercury < DP < BH)                  9.210e-11 < 4.102e-04 < 4.839e-03
✅  Gate Sc2: BH saturates to g_eff (<=1e-12 abs)                            |alpha_bh-g_eff|=0.000e+00
```

Transcript excerpt (for quick visual diff):

```text
✅  Gate T_sf: >=3/4 counterfactuals degrade strong-field score by (1+eps)   strong=4/4  eps=0.182574
==================================================================================================
                                         DETERMINISM HASH                                         
==================================================================================================
determinism_sha256: ab0d7c9df5ad5d9c0e599934ed534b9c82cb5a3580934a51810d91146abe59ad
==================================================================================================
                                          FINAL VERDICT                                           
==================================================================================================
✅  DEMO-66 VERIFIED (QG weak→strong: scales + RG + screening + strong-field teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
