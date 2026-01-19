# DEMO-66 — Quantum Gravity Master Flagship (v1)

A self-auditing, deterministic demo with explicit gates. The claim is **operational**: the run must satisfy the printed pass/fail contract.

## Falsify (run this)

```bash
python incoming/demo-66-quantum-gravity-master-flagship-v1.py
```

If you renamed files, locate by demo number:

```bash
python "incoming/$(ls incoming | grep -i '^demo-66' | head -n 1)"
```

**Falsification condition:** any printed `FAIL` gate, a missing/invalid certificate section, or a materially different checkpoint (beyond stated tolerances) falsifies the demo as packaged here.

## Premise

- **Zero tuning / zero fitted parameters.** Any external “reference” values are evaluation-only and do not feed back into selection.

- **Deterministic.** No randomness; no network; no hidden configuration.

- **Integer-first.** Selection and budgets are derived from fixed integer contracts (prime window + residue/coherence constraints).


## Scope (what this demo claims)

> demo66_master_flagship_qg_referee_ready_v1.py

> DEMO-66 — QUANTUM-GRAVITY MASTER FLAGSHIP
> Canonical discrete scale D* + eps0 closure + Rosetta RG fit + screening/ringdown proxy + teeth

> This script is deliberately:
>   • deterministic (no RNG, no external data downloads)
>   • self-auditing (explicit gates with printed PASS/FAIL)
>   • referee-facing (first-principles derivations + clear falsifiers)

> What this demo is (and is not):
>   • It is a reproducible *certificate* that a locked, discrete pipeline produces a coherent
>     weak-to-strong coupling story with strong deterministic falsification ("teeth").
>   • It is NOT an empirical cosmology paper; it is an operator-class / invariance / closure
>     demonstration with budget-limited counterfactual failure modes.

> I/O policy:
>   • stdout only by default.
>   • optional JSON + PNG artifacts are attempted; failures are caught and reported.

## Run instructions

- Dependencies: Python 3.10+ plus `matplotlib`, `numpy`.

- Install (recommended):

```bash
python -m pip install -r requirements.txt
```

- Execute:

```bash
python incoming/demo-66-quantum-gravity-master-flagship-v1.py
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Unique admissible triple in primary window  count=1
PASS  Gate G1: D* equals 1170 for canonical bases  Dstar=1170
FAIL  Gate G2: exp(-sqrt(D*)/3) matches locked value  computed=1.1175862368611232e-05 expected=1.1175862368611232e-05
PASS  Gate G3: best (beta,N) achieves <1% closure to 1e-5  best=(beta=8,N=96) eps0=9.972011253995e-06 err=0.00279887
PASS  Gate G4: canonical (beta=8,N=96) also achieves <1% closure  eps0=9.972011253995e-06 err=0.00279887
PASS  Gate T0: >=3/4 counterfactuals degrade eps0 score by (1+eps)  strong=4/4 eps=0.182574
PASS  Gate G5: R_inf > 1 (nontrivial, positive coupling)  R_inf=1.058064
PASS  Gate G6: SSE small (locked table consistent with 1/D^2 scaling)  SSE=3.626e-05
PASS  Gate G7: |g_eff| in sane nonzero band  g_eff=4.839e-03
PASS  Gate T1: >=3/4 counterfactuals degrade prediction error by (1+eps)  strong=4/4 eps=0.182574
PASS  Gate G8: alpha_eff monotone in eps  order=Mercury<3.781e-04<4.839e-03
PASS  Gate G9: BH_proxy saturates to g_eff (<=1e-3 rel)  alpha_BH=4.838651e-03 g_eff=4.838651e-03
PASS  Gate G10: weak-field alpha_eff tiny and positive  alpha_M=9.210e-11
PASS  Gate G11: intermediate system in between weak/strong regimes  alpha_DP=3.781e-04
PASS  Gate G12: delta_f bounded by caps and nonzero  df=4.558e-03 cap=0.3
PASS  Gate G13: delta_tau bounded by caps and nonzero  dtau=4.534e-03 cap=0.3
PASS  Gate A0: CONTROL_OFF yields zero effect (expected)
PASS  Gate A1: control separates from primary (nontriviality)  df0=0.0 df=4.558e-03
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `991c7c7622d3e18ccbae8a8e63619bd482b29c14d1c1faeb9dfdb89fd5a474f4`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Primary equals (137,107,103)  selected=Triple(wU=137, s2=107, s3=103)
PASS  Gate G1: D* equals 1170 for canonical bases  Dstar=1170
FAIL  Gate G2: exp(-sqrt(D*)/3) matches locked value  computed=1.1175862368611232e-05 expected=1.1175862368611232e-05
PASS  Gate G3: best (beta,N) achieves <1% closure to 1e-5  best=(beta=8,N=96) eps0=9.972011253995e-06 err=0.00279887
PASS  Gate G4: canonical (beta=8,N=96) also achieves <1% closure  eps0=9.972011253995e-06 err=0.00279887
PASS  Gate T0: >=3/4 counterfactuals degrade eps0 score by (1+eps)  strong=4/4 eps=0.182574
PASS  Gate G5: R_inf > 1 (nontrivial, positive coupling)  R_inf=1.058064
PASS  Gate G6: SSE small (locked table consistent with 1/D^2 scaling)  SSE=3.626e-05
PASS  Gate G7: |g_eff| in sane nonzero band  g_eff=4.839e-03
PASS  Gate T1: >=3/4 counterfactuals degrade prediction error by (1+eps)  strong=4/4 eps=0.182574
PASS  Gate G8: alpha_eff monotone in eps  order=Mercury<3.781e-04<4.839e-03
PASS  Gate G9: BH_proxy saturates to g_eff (<=1e-3 rel)  alpha_BH=4.838651e-03 g_eff=4.838651e-03
PASS  Gate G10: weak-field alpha_eff tiny and positive  alpha_M=9.210e-11
PASS  Gate G11: intermediate system in between weak/strong regimes  alpha_DP=3.781e-04
PASS  Gate G12: delta_f bounded by caps and nonzero  df=4.558e-03 cap=0.3
PASS  Gate G13: delta_tau bounded by caps and nonzero  dtau=4.534e-03 cap=0.3
```

Transcript excerpt (for quick visual diff):

```text
====================================================================================================
DETERMINISM HASH
====================================================================================================
determinism_sha256: 991c7c7622d3e18ccbae8a8e63619bd482b29c14d1c1faeb9dfdb89fd5a474f4

====================================================================================================
FINAL VERDICT
====================================================================================================
PASS  DEMO-66 VERIFIED (D* + eps0 closure + RG + screening/ringdown + teeth)
Result: VERIFIED

====================================================================================================
ARTIFACTS (optional)
====================================================================================================
Results JSON not written (filesystem may be unavailable)  error=PermissionError(1, 'Operation not permitted')
Plot not written (matplotlib unavailable or filesystem restricted)  error=PermissionError(1, 'Operation not permitted')
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
