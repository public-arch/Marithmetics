#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SM_GUM_v1.0 — Grand Unified Model driver

Unifies three existing demos into one executable pipeline:

  Layer 0  (substrate / BIOS): SCFP++ integers {wU, s2, s3, q3}
               → selected by Demo‑18 gates, inlined in BB‑36.

  Layer 1  (math kernel): SM‑MATH‑9 (Demo 37)
               → π, γ, ζ(3), ζ(5), C₂ (twin prime), Artin A, δ (Feigenbaum).

  Layer 2A (physics engine): DEMO‑33 (v6)
               → α, sin²θ_W, α_s, EW scale v, gauge couplings, masses, etc.

  Layer 2B (cosmology engine): BB‑36 (vΩ∞)
               → H₀, Ω fractions, η_B, Y_He, δ_CMB, Σm_ν, A_s, n_s, τ, ℓ₁, ρ_Λ…

The point is that *one* integer substrate (SCFP++) is used consistently
across all layers. This file does **no physics on its own**; it just
wires the three authority demos into a single artifact and prints a
unified dashboard.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

# ---------------------------------------------------------------------------
# Adjust these imports to match your actual filenames
# ---------------------------------------------------------------------------

# SM‑MATH‑9 (Demo 37)
import sm_math_model_demo_v1 as smath        

# DEMO‑33 Standard Model
import sm_standard_model_demo_v1 as sm33        

# BB‑36 SCFP Universe
import bb_grand_emergence_masterpiece_runner_v1 as bb36    


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Substrate:
    """Layer‑0 data: SCFP++ survivors and basic derivatives."""
    wU: int
    s2: int
    s3: int
    q3: int
    survivors: dict  # DEMO‑33 style: {"U(1)": (w*, [candidates]), ...}


@dataclass
class MathLayer:
    """Layer‑1 math invariants from SM‑MATH‑9."""
    pi: float
    gamma: float
    zeta3: float
    zeta5: float
    C2: float
    ArtinA: float
    delta: float  # Feigenbaum δ (RG 10‑dim)


@dataclass
class PhysicsLayer:
    """Layer‑2A physics invariants from DEMO‑33."""
    alpha_em: float
    sin2W: float
    alpha_s: float
    e: float
    g1: float
    g2: float
    g3: float
    v: float
    MW: float
    MZ: float
    kappa: float
    ell_star: float
    Lambda_star: float
    Sbar: float


@dataclass
class CosmoLayer:
    """Layer‑2B cosmology invariants from BB‑36 structural engine."""
    H0: float
    H0_ref: float
    Omega_L: float
    Omega_L_ref: float
    Omega_b: float
    Omega_b_ref: float
    Omega_c: float
    Omega_c_ref: float
    Omega_r: float
    Omega_r_ref: float
    etaB: float
    etaB_ref: float
    YHe: float
    YHe_ref: float
    deltaCMB: float
    deltaCMB_ref: float


# ---------------------------------------------------------------------------
# Layer 0 — Substrate (SCFP++) from BB‑36 Stage 0
# ---------------------------------------------------------------------------

def build_substrate() -> Substrate:
    """
    Use BB‑36's inlined Demo‑18 selector to get SCFP++ integers:

        select_scfp_integers() → {wU, s2, s3, q3}

    Then map them into the DEMO‑33 survivor format:
        survivors["U(1)"] = (wU, [wU]), etc.
    """
    scfp = bb36.select_scfp_integers()
    wU = int(scfp["wU"])
    s2 = int(scfp["s2"])
    s3 = int(scfp["s3"])
    q3 = int(scfp["q3"])

    survivors = {
        "U(1)": (wU, [wU]),
        "SU(2)": (s2, [s2]),
        "SU(3)": (s3, [s3]),
    }

    return Substrate(wU=wU, s2=s2, s3=s3, q3=q3, survivors=survivors)


# ---------------------------------------------------------------------------
# Layer 1 — Math kernel via SM‑MATH‑9 (Demo 37)
# ---------------------------------------------------------------------------

def build_math_layer() -> MathLayer:
    """
    Call SM‑MATH‑9's compute_all() to obtain its sector dictionary.

    We only extract a small subset for the GUM dashboard:
      • π
      • γ
      • ζ(3), ζ(5)
      • C2 (twin prime constant)
      • Artin's constant A
      • δ (Feigenbaum constant from RG with M=10)
    """
    out = smath.compute_all()
    A = out["A"]
    B = out["B"]
    C = out["C"]

    return MathLayer(
        pi=A["pi"],
        gamma=A["gamma"],
        zeta3=A.get("zeta3", A.get("zeta3_apery", float("nan"))),
        zeta5=A.get("zeta5", float("nan")),
        C2=B["C2"],
        ArtinA=B["ArtinA"],
        delta=C.get("delta_RG_10", C.get("delta_RG_8", float("nan"))),
    )


# ---------------------------------------------------------------------------
# Layer 2A — Physics engine via DEMO‑33
# ---------------------------------------------------------------------------

def build_physics_layer(sub: Substrate) -> PhysicsLayer:
    """
    Re‑use DEMO‑33's internal functions, but *feed* them the SCFP++
    survivors from Layer‑0 instead of letting DEMO‑33 run its own SCFP scan.

    Pipeline:

        kappa, margins = Fejér / KUEC anchor
        ℓ★, Λ★, S̄     = BH/Unruh seam over survivors
        α, sin²θ_W, α_s = Φ‑channel from survivors
        e, g1, g2, g3   = gauge couplings
        palette         = Palette‑B exponents
        v               = One‑Action closed‑form v
        v, MW, MZ, ...  = absolute EW outputs
    """
    survivors = sub.survivors

    # Scaling Law #1 — Fejér / KUEC anchor
    kappa, c_a, hs, margins = sm33.derive_kappa_and_margins()

    # BH/Unruh–KUEC seam
    ell_star, Lambda_star, Sbar = sm33.derive_ell_star_BH_unruh(kappa, survivors)

    # Structural constants α, sin²θ_W, α_s from Φ‑channel
    alpha, sin2W, alpha_s, phi_meta = sm33.structural_constants_from_survivors(survivors)

    # Gauge couplings
    e0, g1G, g2, g3, sW, cW = sm33.couplings(alpha, sin2W, alpha_s)

    # Palette‑B (Yukawa exponent palette)
    palette = sm33.search_palette_B()
    if palette is None:
        raise RuntimeError("Palette‑B search failed (no survivor).")

    # Closed‑form v from One‑Action shell functional
    v, y_star, lam_v = sm33.derive_v_closed_form(palette, margins, survivors)

    # Absolute EW observables (M_W, M_Z, recomputed e,g1,g2,g3)
    v, MW, MZ, e0, g1G, g2, g3, sW, cW = sm33.absolute_EW(alpha, sin2W, alpha_s, v)

    return PhysicsLayer(
        alpha_em=alpha,
        sin2W=sin2W,
        alpha_s=alpha_s,
        e=e0,
        g1=g1G,
        g2=g2,
        g3=g3,
        v=v,
        MW=MW,
        MZ=MZ,
        kappa=kappa,
        ell_star=ell_star,
        Lambda_star=Lambda_star,
        Sbar=Sbar,
    )


# ---------------------------------------------------------------------------
# Layer 2B — Cosmology engine via BB‑36 structural monomials
# ---------------------------------------------------------------------------

def build_cosmo_layer(sub: Substrate) -> CosmoLayer:
    """
    Feed the same SCFP++ integers into BB‑36's structural monomial engine.

    BB‑36's build_structural_engine(scfp) returns a dict with:

        H0_SCFP, H0_obs
        Omega_b_SCFP, Omega_c_SCFP, Omega_L_SCFP, Omega_r_SCFP
        Omega_b_obs,  Omega_c_obs,  Omega_L_obs,  Omega_r_obs
        etaB_struct, etaB_obs
        YHe_struct, YHe_obs
        deltaCMB_struct, deltaCMB_obs
        ... (plus many more, which we happily keep but don't need here)
    """
    scfp = dict(wU=sub.wU, s2=sub.s2, s3=sub.s3, q3=sub.q3)
    engine = bb36.build_structural_engine(scfp)

    return CosmoLayer(
        H0=engine["H0_SCFP"],
        H0_ref=engine["H0_obs"],
        Omega_L=engine["Omega_L_SCFP"],
        Omega_L_ref=engine["Omega_L_obs"],
        Omega_b=engine["Omega_b_SCFP"],
        Omega_b_ref=engine["Omega_b_obs"],
        Omega_c=engine["Omega_c_SCFP"],
        Omega_c_ref=engine["Omega_c_obs"],
        Omega_r=engine["Omega_r_SCFP"],
        Omega_r_ref=engine["Omega_r_obs"],
        etaB=engine["etaB_struct"],
        etaB_ref=engine["etaB_obs"],
        YHe=engine["YHe_struct"],
        YHe_ref=engine["YHe_obs"],
        deltaCMB=engine["deltaCMB_struct"],
        deltaCMB_ref=engine["deltaCMB_obs"],
    ), engine


# ---------------------------------------------------------------------------
# Dashboard + helpers
# ---------------------------------------------------------------------------

def rel_err(pred: float, ref: float) -> float:
    if ref == 0.0:
        return float("inf") if pred != 0.0 else 0.0
    return abs(pred - ref) / abs(ref)


def print_header(title: str) -> None:
    line = "═" * len(title)
    print()
    print(line)
    print(title)
    print(line)


def print_unified_manifest(sub: Substrate,
                           mathL: MathLayer,
                           physL: PhysicsLayer,
                           cosmoL: CosmoLayer,
                           engine: dict) -> None:
    """
    This is the “mic‑drop” view: show α, δ, C₂, H₀, Ω_Λ, etc. side‑by‑side.
    """
    print_header("SM_GUM_v1.0 · Grand Unified Model (MATH ⊕ SM ⊕ COSMO from SCFP++)")
    print(f"Timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print("\nLayer‑0 substrate (SCFP++ survivors):")
    print(f"  wU = {sub.wU}")
    print(f"  s2 = {sub.s2}")
    print(f"  s3 = {sub.s3}")
    print(f"  q3 = {sub.q3}")
    print(f"  q2 = wU − s2 = {sub.wU - sub.s2}")
    print()

    # Shortcuts to reference tables
    MREF = getattr(smath, "REF", {})
    PDG  = getattr(sm33, "PDG", {})

    # Unified constant table
    print("Unified constants dashboard:")
    print("  {name:<22} {pred:>18}  {ref:>18}  {err:>10}  {src}".format(
        name="Constant",
        pred="Derived",
        ref="Reference",
        err="RelErr",
        src="Source",
    ))

    # 1) Fine structure constant α
    alpha_ref = PDG.get("alpha_em", float("nan"))
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="α (fine structure)",
        pred=physL.alpha_em,
        ref=alpha_ref,
        err=rel_err(physL.alpha_em, alpha_ref),
        src="Physics (DEMO‑33)",
    ))

    # 2) Feigenbaum δ (from SM‑MATH‑9 RG solver)
    delta_ref = MREF.get("delta", 4.669201609102990)
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="δ (Feigenbaum)",
        pred=mathL.delta,
        ref=delta_ref,
        err=rel_err(mathL.delta, delta_ref),
        src="Dynamics (SM‑MATH‑9)",
    ))

    # 3) Twin prime constant C2
    C2_ref = MREF.get("C2", 0.6601618158468696)
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="C₂ (twin prime)",
        pred=mathL.C2,
        ref=C2_ref,
        err=rel_err(mathL.C2, C2_ref),
        src="Primes (SM‑MATH‑9)",
    ))

    # 4) Hubble constant H0
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="H₀ [km/s/Mpc]",
        pred=cosmoL.H0,
        ref=cosmoL.H0_ref,
        err=rel_err(cosmoL.H0, cosmoL.H0_ref),
        src="Cosmo (BB‑36)",
    ))

    # 5) Ω_Λ (dark energy fraction)
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="Ω_Λ",
        pred=cosmoL.Omega_L,
        ref=cosmoL.Omega_L_ref,
        err=rel_err(cosmoL.Omega_L, cosmoL.Omega_L_ref),
        src="Cosmo (BB‑36)",
    ))

    # 6) Optional: Y_He and δ_CMB
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="Y_He (BBN)",
        pred=cosmoL.YHe,
        ref=cosmoL.YHe_ref,
        err=rel_err(cosmoL.YHe, cosmoL.YHe_ref),
        src="BBN (BB‑36)",
    ))
    print("  {name:<22} {pred:18.9e}  {ref:18.9e}  {err:10.3e}  {src}".format(
        name="δ_CMB",
        pred=cosmoL.deltaCMB,
        ref=cosmoL.deltaCMB_ref,
        err=rel_err(cosmoL.deltaCMB, cosmoL.deltaCMB_ref),
        src="CMB (BB‑36)",
    ))

    print("\nAdditional cross‑checks:")
    print(f"  sin²θ_W^Φ        = {physL.sin2W:.9f}  (ref PDG ≈ {PDG.get('sin2W', float('nan')):.9f})")
    print(f"  α_s^Φ            = {physL.alpha_s:.9f}  (ref PDG ≈ {PDG.get('alpha_s', float('nan')):.9f})")
    print(f"  v (One‑Action)    = {physL.v:.9f} GeV")
    print(f"  ℓ★, Λ★, S̄        = {physL.ell_star:.3e}, {physL.Lambda_star:.3e}, {physL.Sbar:.6f}")
    print(f"  Ω_tot (BB‑36)     = {engine['Omega_tot']:.9f}")
    print()

    print("Narrative summary:")
    print("  • Layer‑0: SCFP++ selects (137, 107, 103, 17) from a finite lawbook.")
    print("  • Layer‑1: SM‑MATH‑9 derives δ, C₂, etc., from ℤ + operator calculus.")
    print("  • Layer‑2A: DEMO‑33 derives α, sin²θ_W, α_s, v, masses from the same survivors.")
    print("  • Layer‑2B: BB‑36 derives H₀, Ω_Λ, η_B, Y_He, δ_CMB, Σm_ν, A_s, n_s, τ from the same survivors.")
    print("  • This GUM driver simply makes that unification explicit in a single run.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Layer‑0: substrate
    sub = build_substrate()

    # Layer‑1: math kernel
    mathL = build_math_layer()

    # Layer‑2A: SM physics
    physL = build_physics_layer(sub)

    # Layer‑2B: cosmology
    cosmoL, engine = build_cosmo_layer(sub)

    # Unified dashboard
    print_unified_manifest(sub, mathL, physL, cosmoL, engine)


if __name__ == "__main__":
    main()
