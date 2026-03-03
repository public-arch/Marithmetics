# DEMO-70 — Higgs Master Flagship (fixed-point mass closure)

> **Claim:** Deterministic selection of triple (wU, s2, s3) = (137, 107, 103) fixes all EW parameters (Θ = 4/15, sin²θW = 7/30, α₀⁻¹ = 137, αₛ = 2/17); PREWORK 70A recovers EW vacuum via lawful closed-form closure with illegal controls failing; PREWORK 70B solves λ(μ_max) ≈ 0 for UV critical edge λ* ≈ 0.173 with budget-tier vs truth-tier comparison; PREWORK 70C mode-ladder SU(2) lock selects best mode d=13 with m_H ≈ 123.88 GeV (|Δ| ≈ 1.12 GeV), illegal controls and counterfactual budgets degrading by fixed eps margin.

---

## What this demo computes

Single integrated deterministic audit combining three preworks:

PREWORK 70A (Exact EW rational locks + lawful closure):
- Exact lock gates: Θ = 4/15, sin²θW = 7/30, α₀⁻¹ = 137, αₛ = 2/17.
- Plausibility checks: Higgs mechanism iterations ≤ 250, vacuum v ∈ [200, 400] GeV, α(MZ) ∈ [0.0075, 0.0083], MZ ∈ [80, 100] GeV.
- Lawful “dressed” closure: fixed renormalization scheme, no tuning.
- Illegal control separation: non-admissible operators fail closure gates.
- Counterfactual teeth: ≥3/4 counterfactual triples out of predeclared band [80, 100] GeV.

PREWORK 70B (UV critical edge λ*):
- Solve λ(μ_max) ≈ 0 for UV critical edge λ* via RG running.
- Target band: λ* ∈ [0.1, 0.3] (plausibility); result λ* ≈ 0.173.
- Truth tier (higher budget K_truth) vs budget tier (K from triple): primary budget reproduces truth within eps³.
- Illegal controls perform much worse (err_illegal/err_primary ≈ 31.17).
- Counterfactual budget degrades error by (1+eps).

PREWORK 70C (Mode-ladder SU(2) lock):
- Sweep coupling parameter λ₀ over mode d ∈ [12..20] (SU(2) modes).
- Solve fixed-point equation for Higgs mass m_H(λ₀, d).
- Best mode d = 13: m_H ≈ 123.88 GeV (|Δ| ≈ 1.119 GeV vs PDG m_H ≈ 125 GeV).
- Illegal control: d = 15 with Δ ≈ 9.1 GeV (worse).
- Counterfactual budget: d = 14 with Δ ≈ 5.4 GeV (degrade by (1+eps)).

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Primary triple must equal (137, 107, 103).
5. Lock gates exact: Θ = 4/15, sin²θW = 7/30, α₀⁻¹ = 137, αₛ = 2/17.
6. Plausibility: iters ≤ 250, v ∈ [200, 400], α(MZ) ∈ [0.0075, 0.0083], MZ ∈ [80, 100].
7. λ* in sane band [0.1, 0.3].
8. Primary budget reproduces truth within eps³.
9. Illegal controls significantly worse (err_ill/err_prim > threshold).
10. Counterfactual budgets degrade by (1+eps).
11. Best SU(2) mode d = 13 with m_H offset ≈ 1.119 GeV.
12. Illegal/counterfactual modes degrade m_H offset by (1+eps).

## Controls

- **Illegal operators:** Non-admissible EW closure (fails lock gates), non-optimal SU(2) mode selection.
- **Counterfactuals:** Alternative triples with same selection rules, different budgets; counterfactual modes d ∈ [12..20] must degrade m_H offset and λ* by (1+eps).
- **Ablations:** Truth tier (K_truth) vs budget tier (K) comparison; RG running accuracy checks.

## Dependencies

Python 3.10+ with standard library. Matplotlib optional (only for PNG output).

## Run

```bash
python demo.py
python demo.py --write  # Optional JSON + PNG output
```

## Pass/Fail contract (gates)

Primary gates emitted by the demo (treat any regression as a hard failure):

```text
PASS  Gate S1: primary equals (137,107,103)
PASS  Gate S2: captured >=4 counterfactuals                                    found=4
PASS  Gate A1: lock-gates exact (Theta=4/15, sin^2θW=7/30, alpha0=1/137, alpha_s=2/17)
PASS  Gate A2: plausibility (iters<=250, v∈[200,400], alpha(MZ)∈[0.0075,0.0083], MZ∈[80,100]) iters=44 v=246.286 alpha=0.007800 MZ=91.155
PASS  Gate A3: illegal control is worse (dist_illegal > dist_lawful)
PASS  Gate A4: counterfactual teeth (>=3/4 CF out of [80,100])                 out=4/4
PASS  Gate B1: lambda* in sane band [0.1,0.3]                                  lambda*=0.173398
PASS  Gate B2: primary budget reproduces truth within eps^3                    err=1.144e-05 tol=eps^3=6.086e-03
PASS  Gate B3: illegal controls worse than primary                             err_il/err_p=31.17 res_ratio=8.21e+03
PASS  Gate B4: counterfactual budget degrades by (1+eps)                       err_cf=4.005e-05 err_p=1.144e-05
PASS  Gate C1: best mode is d=13 (SU(2) lock)                                  best_d=13 |Δ|=1.119
PASS  Gate C2: illegal is worse than lawful best by (1+eps)                    best_il_d=15 Δ_law=1.119 Δ_il=1.857
PASS  Gate C3: counterfactual budget degrades by (1+eps)                       Δ_cf=5.365 Δ_law=1.119 eps=0.183
PASS  DEMO-70 VERIFIED (Higgs master flagship)
```

## Reference checkpoints (from provided transcript)

- Determinism hash: `29df9e06c202ad60ac47783a1d0d41a69fcab795325a9f4a28a45c47d199ba12`

- Verdict line: `Result: VERIFIED`

Selected printed checkpoints:

```text
PASS  Gate S1: primary equals (137,107,103)
PASS  Gate S2: captured >=4 counterfactuals                                    found=4
sin^2θW = 7/30 ≈ 0.233333333333
alpha_s = 2/17 ≈ 0.117647058824
PASS  Gate A1: lock-gates exact (Theta=4/15, sin^2θW=7/30, alpha0=1/137, alpha_s=2/17)
PASS  Gate A2: plausibility (iters<=250, v∈[200,400], alpha(MZ)∈[0.0075,0.0083], MZ∈[80,100]) iters=44 v=246.286 alpha=0.007800 MZ=91.155
PASS  Gate A3: illegal control is worse (dist_illegal > dist_lawful)
PASS  Gate A4: counterfactual teeth (>=3/4 CF out of [80,100])                 out=4/4
PASS  Gate B1: lambda* in sane band [0.1,0.3]                                  lambda*=0.173398
PASS  Gate B2: primary budget reproduces truth within eps^3                    err=1.144e-05 tol=eps^3=6.086e-03
PASS  Gate B3: illegal controls worse than primary                             err_il/err_p=31.17 res_ratio=8.21e+03
PASS  Gate B4: counterfactual budget degrades by (1+eps)                       err_cf=4.005e-05 err_p=1.144e-05
d=16  λ0=0.062500  mH_fp=112.5922  |Δ|=12.408  it=17
d=15  λ0=0.066667  mH_fp=115.9006  |Δ|= 9.099  it=18
d=14  λ0=0.071429  mH_fp=119.6347  |Δ|= 5.365  it=18
d=13  λ0=0.076923  mH_fp=123.8814  |Δ|= 1.119  it=19 <-- best
```

Transcript excerpt (for quick visual diff):

```text
STAGE 6 — Determinism hash + score + optional artifacts
----------------------------------------------------------------------------------------------------
determinism_sha256: 29df9e06c202ad60ac47783a1d0d41a69fcab795325a9f4a28a45c47d199ba12
presentation_score: 1000000 / 1,000,000

----------------------------------------------------------------------------------------------------
FINAL VERDICT
----------------------------------------------------------------------------------------------------
PASS  DEMO-70 VERIFIED (Higgs master flagship)
Result: VERIFIED
```

## Reviewer notes

- The README intentionally focuses on **reproducible observables**: gates, checkpoints, and deterministic hashes.

- For discrepancies, attach the full stdout transcript and your Python version; these scripts are designed for line-by-line audit.
