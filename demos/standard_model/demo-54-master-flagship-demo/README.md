# DEMO-54 — Master Flagship Demo (single-file determinism)

> **Claim:** Deterministic first-principles selection of the unique triple (wU, s2, s3) = (137, 107, 103) yields gauge rationals (α₀⁻¹ = 137, sin²θW = 7/30, αₛ = 2/17), QCD scale Λ via 2-loop running, vacuum suppression <1% via fixed Zel’dovich scaling, emergent gravity from discrete Poisson, and mathematical linkages (Feigenbaum δ, twin-prime constant C₂) with counterfactual falsifiers.

---

## What this demo computes

From first principles (primality, residue classes, Euler totient density):
- Symmetry-Constrained Fixed-Point (SCFP++) deterministic selection: unique triple (wU, s2, s3) = (137, 107, 103) in Demo-33 window.
- Gauge-sector rationals: α₀⁻¹ := wU = 137; sin²θW := 7/(wU - s2) = 7/30; αₛ(MZ) := 2/odd_part(wU - 1) = 2/17.
- QCD scale: Λ_QCD via 1-loop closed form and 2-loop numeric inversion at MZ = 91.03 GeV, nf = 5.
- Vacuum energy suppression: ρ ~ Λ⁶/M_Pl² × (1/(16π²))² × 1/(1+αₛ), compared to (H0, ΩΛ) for <1% ratio test.
- Emergent gravity check: discrete Poisson equation on 3D periodic lattice → inverse-square slope verification; lawful (Fejér) vs illegal (sharp/signed) coarse-graining comparison.
- Mathematical linkages: Feigenbaum δ via logistic-map superstable points (root finding + Aitken acceleration); twin-prime constant C₂ via Euler product; compute budgets derived from triple.
- Counterfactual triples (fixed reduced-gate scan) fail as falsifier.

## Falsification contract

1. Any printed `FAIL` gate → demo falsified.
2. Missing or invalid certificate section → demo falsified.
3. Materially different checkpoint beyond stated tolerances → demo falsified.
4. Vacuum suppression ratio |pred/obs - 1| must be <1%.
5. Inverse-square slope must be near -2 (within eps tolerance).
6. All mathematical gates must pass (δ, C₂, emergency gravity).
7. Counterfactual triples must separate from primary in GR emergence.

## Controls

- **Illegal operators:** Sharp spectral cutoff (kernel with negative lobes), signed high-pass complement (stronger negative lobes); tested in emergent gravity suite.
- **Counterfactuals:** Fixed reduced-gate scan of counterfactual triples; must fail GR emergence (inverse-square separation) and vacuum suppression.
- **Ablations:** Baseline budget (expected to miss δ); coarse-graining suite (lawful vs illegal).

## Dependencies

Python 3.9+ with NumPy. Matplotlib optional (only for PNG output).

## Run

```bash
python demo.py
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
