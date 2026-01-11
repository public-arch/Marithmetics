# DEMO-71 — One Action Master Flagship (Noether + unitarity + field energy)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-71-one-action-master-flagship-noether-unitarity-field-energy.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-71' | head -n 1)"
```

**Teeth (what to check):** the reference run prints, among other lines:

```text
FINAL VERDICT
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> DEMO-71 — ONE ACTION MASTER FLAGSHIP
> (Classical Noether + Quantum Unitarity + Field Energy) — REFEREE READY, SELF-CONTAINED

> What this demo shows (from first principles, deterministic, no tuning):

>   (1) A single discrete selector produces the same primary triple (137,107,103) deterministically.
>   (2) From that triple, we derive the *same budgets* (q2,q3,eps) used across domains.
>   (3) A single *action principle* manifests as three structurally-protected laws:
>         • Classical: symplectic / Noether (angular momentum) under a variational integrator.
>         • Quantum  : unitarity + reversibility under Crank–Nicolson (CN) time stepping.
>         • Field    : energy stability under leapfrog (a variational/symplectic update).

>   (4) “Illegal controls” (non-variational / non-unitary / sign-flipped) violate the laws.
>   (5) Deterministic counterfactual teeth: reducing the lawful budget K (via q3→3q3) degrades accuracy.

> This script is designed to be portable (NumPy only), audit-grade, and fully deterministic.

> CLI:
>   python demo71_master_flagship_one_action_referee_ready_v1.py
>   python demo71_master_flagship_one_action_referee_ready_v1.py --artifacts

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-71-one-action-master-flagship-noether-unitarity-field-energy.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
✅  Gate S0: primary equals (137,107,103)                                      selected=(137, 107, 103)
✅  Gate S1: captured >=4 counterfactual triples (deterministic)               found=4
✅  Gate I1: invariants match the locked values (q2=30,q3=17,v2U=3)            (q2,q3,v2U)=(30, 17, 3)
✅  Gate C1: Noether (VV) angular momentum conserved (strict)                  L_drift=9.536e-14 tol=eps^6=3.704e-05
✅  Gate C2: Symplectic area (VV) det≈1 within eps^3                           |det-1|=0.000e+00 tol=eps^3=6.086e-03
✅  Gate C3: Energy bounded (VV) drift <= eps^2                                E_drift=8.615e-04 eps^2=3.333e-02
✅  Gate C4: Illegal Euler breaks Noether by margin                            L_drift_euler=5.304e+00 floor=eps^2=3.333e-02
✅  Gate C5: Illegal Euler breaks area (|det-1| >= eps^4)                      |det_euler-1|=6.932e-03 eps^4=1.111e-03
✅  Gate C6: Anti-action signflip exhibits blow-up / nonphysical growth        blowup=True max_state=1.035e+06 floor=3.0e+01
✅  Gate CT: counterfactual (dt increased) degrades VV trajectory error by (1+eps) errP=4.530e-03 errCF=4.076e-02 1+eps=1.183
✅  Gate Q1: CN unitary (norm drift <= eps^4)                                  drift=6.661e-16 eps^4=1.111e-03
✅  Gate Q2: CN reversible (forward+back error <= eps^3)                       err=1.499e-15 eps^3=6.086e-03
✅  Gate Q3: CN accuracy vs exact within eps                                   err=7.306e-02 eps=1.826e-01
✅  Gate Q4: Illegal Euler not unitary (norm drift >= eps^2)                   drift=1.071e+21 eps^2=3.333e-02
✅  Gate Q5: Illegal Euler worse accuracy than CN by margin                    err_eu=1.071e+21 err_cn=7.306e-02 1+eps=1.183
✅  Gate Q6: Wick illegal destroys Schrödinger truth (err >= eps)              err_wick=1.001e+00 eps=1.826e-01
✅  Gate QT: counterfactual degrades CN error by (1+eps)                       errP=7.306e-02 errCF=4.949e-01 1+eps=1.183
✅  Gate F1: Leapfrog energy drift <= eps^3                                    drift=2.184e-06 eps^3=6.086e-03
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `5a2527716ad389f567db9398157c012d49c40402ae5566596daf9e40dfc51a7c`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
✅  Gate S0: primary equals (137,107,103)                                      selected=(137, 107, 103)
✅  Gate S1: captured >=4 counterfactual triples (deterministic)               found=4
✅  Gate I1: invariants match the locked values (q2=30,q3=17,v2U=3)            (q2,q3,v2U)=(30, 17, 3)
✅  Gate C1: Noether (VV) angular momentum conserved (strict)                  L_drift=9.536e-14 tol=eps^6=3.704e-05
✅  Gate C2: Symplectic area (VV) det≈1 within eps^3                           |det-1|=0.000e+00 tol=eps^3=6.086e-03
✅  Gate C3: Energy bounded (VV) drift <= eps^2                                E_drift=8.615e-04 eps^2=3.333e-02
✅  Gate C4: Illegal Euler breaks Noether by margin                            L_drift_euler=5.304e+00 floor=eps^2=3.333e-02
✅  Gate C5: Illegal Euler breaks area (|det-1| >= eps^4)                      |det_euler-1|=6.932e-03 eps^4=1.111e-03
✅  Gate C6: Anti-action signflip exhibits blow-up / nonphysical growth        blowup=True max_state=1.035e+06 floor=3.0e+01
✅  Gate CT: counterfactual (dt increased) degrades VV trajectory error by (1+eps) errP=4.530e-03 errCF=4.076e-02 1+eps=1.183
✅  Gate Q1: CN unitary (norm drift <= eps^4)                                  drift=6.661e-16 eps^4=1.111e-03
✅  Gate Q2: CN reversible (forward+back error <= eps^3)                       err=1.499e-15 eps^3=6.086e-03
✅  Gate Q3: CN accuracy vs exact within eps                                   err=7.306e-02 eps=1.826e-01
✅  Gate Q4: Illegal Euler not unitary (norm drift >= eps^2)                   drift=1.071e+21 eps^2=3.333e-02
✅  Gate Q5: Illegal Euler worse accuracy than CN by margin                    err_eu=1.071e+21 err_cn=7.306e-02 1+eps=1.183
✅  Gate Q6: Wick illegal destroys Schrödinger truth (err >= eps)              err_wick=1.001e+00 eps=1.826e-01
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
                             STAGE 3 — Determinism hash + final verdict                             
====================================================================================================
determinism_sha256: 5a2527716ad389f567db9398157c012d49c40402ae5566596daf9e40dfc51a7c

====================================================================================================
                                           FINAL VERDICT                                            
====================================================================================================
  ✅  DEMO-71 VERIFIED (One Action triad: classical + quantum + field, with illegal controls + teeth)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
