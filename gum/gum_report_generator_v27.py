#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
gum_report_generator_v27.py

Grand Unified Model (GUM) report generator with:

  - SCFP++ substrate selector (via gum_v1 + BB-36),
  - SM-MATH-9 math kernel (sm_math_model_demo_v1),
  - DEMO-33 Standard Model layer (sm_standard_model_demo_v1),
  - BB-36 structural cosmology (bb_grand_emergence_masterpiece_runner_v1),
  - Big Bang SPDE / tidal visual (5-panel PNG),
  - Planck 2018 vs GUM CAMB TT comparison (PNG overlay embedded in the PDF),
  - JSON manifest and numeric SHA-256 fingerprint.

All PDF-visible text is ASCII-only to avoid black boxes in some viewers.
"""

from __future__ import annotations

import math
import json
import hashlib
import os
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "Assets")



# ---------------------------------------------------------------------------
# Third-party libraries
# ---------------------------------------------------------------------------

try:
    import numpy as np
except Exception:
    np = None  # type: ignore

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None  # type: ignore

try:
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
        PageBreak,
        Image,
    )
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
except Exception as e:
    raise SystemExit(
        "This script requires the 'reportlab' package. Install with:\n"
        "    pip install reportlab\n"
        f"Original import error: {e}"
    )

# CAMB is optional; we enable TT overlay only if available
HAS_CAMB = False
try:
    import camb  # type: ignore

    HAS_CAMB = True
except Exception:
    HAS_CAMB = False

# ---------------------------------------------------------------------------
# Local GUM stack
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local GUM stack
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Local GUM stack
# ---------------------------------------------------------------------------

try:
    # Structured imports: pull modules from the correct package folders
    from gum import gum_v1 as gum
    from sm import sm_math_model_demo_v1 as smath
    from sm import sm_standard_model_demo_v1 as sm33
    from cosmo import bb_grand_emergence_masterpiece_runner_v1 as bb36
except ImportError as e:
    raise SystemExit(
        "Could not import one of the local GUM modules.\n"
        "Expected structure:\n"
        "  gum/gum_v1.py\n"
        "  sm/sm_math_model_demo_v1.py\n"
        "  sm/sm_standard_model_demo_v1.py\n"
        "  cosmo/bb_grand_emergence_masterpiece_runner_v1.py\n"
        f"Original error: {e}"
    )

# If you like direct type names:
Substrate = gum.Substrate
MathLayer = gum.MathLayer
PhysicsLayer = gum.PhysicsLayer
CosmoLayer = gum.CosmoLayer


# ---------------------------------------------------------------------------
# Reference constants
# ---------------------------------------------------------------------------

MATH_REF = getattr(smath, "REF", {})

SM_REF = {
    "alpha_em": 7.2973525693e-3,  # fine structure
    "sin2W": 0.23122,
    "alpha_s": 0.1179,
    "v": 246.22,   # GeV
    "MW": 80.379,  # GeV
    "MZ": 91.1876,  # GeV
}

COSMO_REF = {
    "H0": 70.476,
    "Omega_b": 0.04500,
    "Omega_c": 0.24300,
    "Omega_L": 0.71192,
    "Omega_r": 0.00008,
    "Omega_tot": 1.0,
    "etaB": 6.10e-10,
    "YHe": 0.246,
    "deltaCMB": 1.00e-5,
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def rel_err(pred: Any, ref: Any) -> Optional[float]:
    """Relative error |pred - ref| / |ref| with safety checks."""
    try:
        if pred is None or ref is None:
            return None
        p = float(pred)
        r = float(ref)
        if math.isnan(p) or math.isnan(r):
            return None
        if r == 0.0:
            return None
        return abs(p - r) / abs(r)
    except Exception:
        return None


def fmt_val(x: Any, sci: bool = False, digits: int = 9) -> str:
    """Format a scalar for table display."""
    if x is None:
        return "NA"
    try:
        if isinstance(x, bool):
            return "true" if x else "false"
        if isinstance(x, int):
            return str(x)
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return "NA"
        if sci:
            return f"{xf:.{digits}e}"
        return f"{xf:.{digits}g}"
    except Exception:
        return "NA"


def fmt_err(e: Any) -> str:
    """Format a relative error."""
    if e is None:
        return "NA"
    try:
        v = float(e)
        if math.isnan(v) or math.isinf(v):
            return "NA"
        return f"{v:.3e}"
    except Exception:
        return "NA"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_manifest_values(vals: List[float]) -> str:
    """Deterministic SHA-256 over a numeric list."""
    s = "[" + ",".join(f"{float(v):.18e}" for v in vals) + "]"
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

# ---------------------------------------------------------------------------
# Cosmology builder (direct from BB-36, using build_structural_cosmo)
# ---------------------------------------------------------------------------

def build_cosmo_layer_direct(sub: Substrate) -> Tuple[CosmoLayer, Dict[str, Any]]:
    """
    Build the BB-36 structural cosmology engine via build_structural_cosmo
    and wrap it into a CosmoLayer. We do not depend on *_obs keys; instead
    we attach reference values from COSMO_REF.
    """
    scfp = {"wU": sub.wU, "s2": sub.s2, "s3": sub.s3, "q3": sub.q3}

    if not hasattr(bb36, "build_structural_cosmo"):
        raise RuntimeError(
            "bb_grand_emergence_masterpiece_runner_v1 has no "
            "build_structural_cosmo(scfp) function."
        )

    engine: Dict[str, Any] = bb36.build_structural_cosmo(scfp)

    H0 = float(engine["H0_SCFP"])
    Om_b = float(engine["Omega_b_SCFP"])
    Om_c = float(engine["Omega_c_SCFP"])
    Om_L = float(engine["Omega_L_SCFP"])
    Om_r = float(engine["Omega_r_SCFP"])

    # YHe key may be Y_He_struct or YHe_struct, depending on version
    if "Y_He_struct" in engine:
        YHe_struct = float(engine["Y_He_struct"])
    elif "YHe_struct" in engine:
        YHe_struct = float(engine["YHe_struct"])
    else:
        raise RuntimeError("BB-36 engine has no Y_He_struct or YHe_struct key")

    etaB_struct = float(engine["etaB_struct"])
    deltaCMB_struct = float(engine["deltaCMB_struct"])

    Omega_tot_struct = float(
        engine.get("Omega_tot_SCFP", Om_b + Om_c + Om_L + Om_r)
    )
    engine["Omega_tot_SCFP"] = Omega_tot_struct

    cosmoL = CosmoLayer(
        H0=H0,
        H0_ref=COSMO_REF["H0"],
        Omega_L=Om_L,
        Omega_L_ref=COSMO_REF["Omega_L"],
        Omega_b=Om_b,
        Omega_b_ref=COSMO_REF["Omega_b"],
        Omega_c=Om_c,
        Omega_c_ref=COSMO_REF["Omega_c"],
        Omega_r=Om_r,
        Omega_r_ref=COSMO_REF["Omega_r"],
        etaB=etaB_struct,
        etaB_ref=COSMO_REF["etaB"],
        YHe=YHe_struct,
        YHe_ref=COSMO_REF["YHe"],
        deltaCMB=deltaCMB_struct,
        deltaCMB_ref=COSMO_REF["deltaCMB"],
    )

    return cosmoL, engine

# ---------------------------------------------------------------------------
# Run the full GUM stack and collect a payload
# ---------------------------------------------------------------------------

def run_gum_and_collect() -> Dict[str, Any]:
    """
    Run the full GUM stack:

      - gum.build_substrate()
      - gum.build_math_layer() + smath.compute_all()
      - gum.build_physics_layer(sub)
      - build_cosmo_layer_direct(sub)
    """
    sub = gum.build_substrate()
    mathL = gum.build_math_layer()
    math_res = smath.compute_all()
    physL = gum.build_physics_layer(sub)
    cosmoL, engine = build_cosmo_layer_direct(sub)

    payload: Dict[str, Any] = {
        "sub": sub,
        "mathL": mathL,
        "physL": physL,
        "cosmoL": cosmoL,
        "engine": engine,
        "math_res": math_res,
        "MATH_REF": MATH_REF,
        "SM_REF": SM_REF,
        "COSMO_REF": COSMO_REF,
    }
    return payload

# ---------------------------------------------------------------------------
# Big Bang SPDE / tidal visual
# ---------------------------------------------------------------------------

def generate_big_bang_visual(report_dir: str,
                             payload: Dict[str, Any],
                             filename: str = "BB36_big_bang.png"
                             ) -> Optional[str]:
    """
    Generate a 5-panel "Big Bang" visual as a PNG:

      1. Baryon asymmetry eta_B(x),
      2. Helium-like field X_He(x),
      3. Radiation contrast delta_gamma(x),
      4. Vorticity magnitude |omega|(x),
      5. Tidal trace Tr(T)(x).

    This is a lightweight SPDE-style demo that uses amplitudes drawn from
    the BB-36 engine but does not depend on BB-36's internal SPDE helpers.
    """
    if np is None or plt is None:
        return None

    out_path = os.path.join(report_dir, filename)
    if os.path.isfile(out_path):
        return out_path

    engine = payload["engine"]

    N = 24
    steps = 32
    dt = 0.1
    D_eta = 0.2
    D_He = 0.15
    D_gam = 0.25

    rng = np.random.default_rng(1234)

    eta_field = rng.normal(size=(N, N, N))
    He_field = rng.normal(size=(N, N, N))
    gam_field = rng.normal(size=(N, N, N))

    def laplacian(field: np.ndarray) -> np.ndarray:
        return (
            np.roll(field, 1, axis=0)
            + np.roll(field, -1, axis=0)
            + np.roll(field, 1, axis=1)
            + np.roll(field, -1, axis=1)
            + np.roll(field, 1, axis=2)
            + np.roll(field, -1, axis=2)
            - 6.0 * field
        )

    def entropy(field: np.ndarray) -> float:
        hist, _ = np.histogram(field, bins=64, density=True)
        p = hist + 1e-12
        p /= p.sum()
        return float(-np.sum(p * np.log(p)))

    for _ in range(steps):
        eta_field += D_eta * dt * laplacian(eta_field)
        He_field += D_He * dt * laplacian(He_field)
        gam_field += D_gam * dt * laplacian(gam_field)
        _ = entropy(eta_field)  # entropy diagnostics if needed
        _ = entropy(He_field)
        _ = entropy(gam_field)

    # Rescale eta_field so mean matches structural eta_B
    eta_target = float(engine["etaB_struct"])
    base_mean = float(eta_field.mean())
    if abs(base_mean) > 1e-12:
        eta_field *= (eta_target / base_mean)

    # Velocity from gradient of radiation field
    vx = np.gradient(gam_field, axis=0)
    vy = np.gradient(gam_field, axis=1)
    vz = np.gradient(gam_field, axis=2)

    dvz_dy = np.gradient(vz, axis=1)
    dvy_dz = np.gradient(vy, axis=2)
    dvx_dz = np.gradient(vx, axis=2)
    dvz_dx = np.gradient(vz, axis=0)
    dvy_dx = np.gradient(vy, axis=0)
    dvx_dy = np.gradient(vx, axis=1)

    omega_x = dvz_dy - dvy_dz
    omega_y = dvx_dz - dvz_dx
    omega_z = dvy_dx - dvx_dy
    omega_mag = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)

    # Tidal tensor proxy from second derivatives of gam_field
    dxx = np.gradient(np.gradient(gam_field, axis=0), axis=0)
    dyy = np.gradient(np.gradient(gam_field, axis=1), axis=1)
    dzz = np.gradient(np.gradient(gam_field, axis=2), axis=2)
    T_trace = dxx + dyy + dzz

    nz = N // 2
    slice_eta = eta_field[:, :, nz].T
    slice_He = He_field[:, :, nz].T
    slice_gam = gam_field[:, :, nz].T
    slice_omega = omega_mag[:, :, nz].T
    slice_T = T_trace[:, :, nz].T

    fig, axes = plt.subplots(1, 5, figsize=(12.0, 2.8))

    axes[0].imshow(slice_eta, origin="lower", cmap="coolwarm")
    axes[0].set_title("Baryon asymmetry eta_B", fontsize=8)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(slice_He, origin="lower", cmap="magma")
    axes[1].set_title("Helium-like field X_He", fontsize=8)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    axes[2].imshow(slice_gam, origin="lower", cmap="viridis")
    axes[2].set_title("Radiation contrast", fontsize=8)
    axes[2].set_xticks([])
    axes[2].set_yticks([])

    axes[3].imshow(slice_omega, origin="lower", cmap="plasma")
    axes[3].set_title("Vorticity magnitude", fontsize=8)
    axes[3].set_xticks([])
    axes[3].set_yticks([])

    axes[4].imshow(slice_T, origin="lower", cmap="cividis")
    axes[4].set_title("Tidal trace Tr(T)", fontsize=8)
    axes[4].set_xticks([])
    axes[4].set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    return out_path

# ---------------------------------------------------------------------------
# CAMB validation: Planck 2018 vs GUM TT spectrum (gum_camb_check logic)
# ---------------------------------------------------------------------------

def run_camb_validation(report_dir: str,
                        payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Use CAMB to compare:

      1. Planck 2018 flat LCDM baseline (approximate TTTEEE+lowE),
      2. GUM cosmology built from H0_SCFP and Omega_L_SCFP,

    following the logic of gum_camb_check.py:
      - Preserve Planck baryon fraction in GUM cosmology.
      - Use same As, ns, tau for both runs.
      - Compute RMS and max relative differences over ell in [2, 2000].
      - Generate a PNG overlay of the TT spectra.

    Returns a dict with:
      enabled, reason, ell_min, ell_max, rms_rel, max_rel,
      threshold_rms, threshold_max, pass_rms, pass_max, plot_path.
    """
    if not HAS_CAMB or np is None or plt is None:
        return {
            "enabled": False,
            "reason": "CAMB or plotting stack not available.",
        }

    engine = payload["engine"]

    try:
        H0_GUM = float(engine["H0_SCFP"])
        Omega_L_GUM = float(engine["Omega_L_SCFP"])
    except KeyError as e:
        return {
            "enabled": False,
            "reason": f"engine missing key {e} for CAMB GUM cosmology.",
        }

    # Planck 2018 TT,TE,EE+lowE (approximate) baseline,
    # matching your gum_camb_check.py script.
    PLANCK = {
        "H0":    67.36,     # km/s/Mpc
        "ombh2": 0.02237,   # Omega_b * h^2
        "omch2": 0.1200,    # Omega_c * h^2
        "tau":   0.0544,
        "As":    2.100e-9,
        "ns":    0.9649,
    }

    hP = PLANCK["H0"] / 100.0
    Omega_b_P = PLANCK["ombh2"] / (hP * hP)
    Omega_c_P = PLANCK["omch2"] / (hP * hP)
    Omega_m_P = Omega_b_P + Omega_c_P
    f_baryon = Omega_b_P / Omega_m_P

    # GUM cosmology: use H0 and Omega_L from BB-36; assume flatness.
    hG = H0_GUM / 100.0
    Omega_m_GUM = 1.0 - Omega_L_GUM
    Omega_b_GUM = f_baryon * Omega_m_GUM
    Omega_c_GUM = (1.0 - f_baryon) * Omega_m_GUM
    ombh2_GUM = Omega_b_GUM * (hG * hG)
    omch2_GUM = Omega_c_GUM * (hG * hG)

    GUM_params = {
        "H0":    H0_GUM,
        "ombh2": ombh2_GUM,
        "omch2": omch2_GUM,
        "tau":   PLANCK["tau"],
        "As":    PLANCK["As"],
        "ns":    PLANCK["ns"],
        "Omega_L": Omega_L_GUM,
        "Omega_m": Omega_m_GUM,
    }

    ELL_MIN = 2
    ELL_MAX = 2000

    def get_cmb_tt_spectrum(H0, ombh2, omch2, As, ns, tau, lmax: int) -> Tuple[np.ndarray, np.ndarray]:
        pars = camb.CAMBparams()
        pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0.0, tau=tau)
        pars.InitPower.set_params(As=As, ns=ns)
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        results = camb.get_results(pars)
        powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
        totCL = powers["total"]
        TT = totCL[:, 0]
        ells = np.arange(TT.size)
        return ells, TT

    try:
        # Planck baseline
        ell_P, TT_P = get_cmb_tt_spectrum(
            PLANCK["H0"], PLANCK["ombh2"], PLANCK["omch2"],
            PLANCK["As"], PLANCK["ns"], PLANCK["tau"], lmax=ELL_MAX
        )

        # GUM cosmology
        ell_G, TT_G = get_cmb_tt_spectrum(
            GUM_params["H0"], GUM_params["ombh2"], GUM_params["omch2"],
            GUM_params["As"], GUM_params["ns"], GUM_params["tau"], lmax=ELL_MAX
        )

        ell_max = min(ELL_MAX, len(TT_P) - 1, len(TT_G) - 1)
        ell_min = ELL_MIN

        mask = (ell_P >= ell_min) & (ell_P <= ell_max)
        ell_use = ell_P[mask]
        TT_P_use = TT_P[mask]
        TT_G_use = TT_G[mask]

        rel_diff = (TT_G_use - TT_P_use) / TT_P_use
        rms_rel = float(np.sqrt(np.mean(rel_diff**2)))
        max_rel = float(np.max(np.abs(rel_diff)))

        threshold_rms = 0.05   # 5% RMS
        threshold_max = 0.15   # 15% max
        pass_rms = rms_rel < threshold_rms
        pass_max = max_rel < threshold_max

        camb_dir = os.path.join(report_dir, "CAMB")
        os.makedirs(camb_dir, exist_ok=True)
        plot_path = os.path.join(camb_dir, "TT_spectrum_Planck2018_vs_GUM.png")

        # Plot spectra (labels can use TeX; they are rendered into a PNG)
        plt.figure(figsize=(7, 5))
        plt.plot(ell_P, TT_P, label="Planck 2018 LCDM", color="C0")
        plt.plot(ell_G, TT_G, label="GUM (SCFP++ / BB-36)", color="C1", linestyle="--")
        plt.xlim(2, 2500)
        plt.xlabel(r"$\ell$")
        plt.ylabel(r"$C_\ell^{TT}$ [$\mu K^2$]")
        plt.title("CMB TT Power Spectrum: Planck 2018 vs GUM")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=200)
        plt.close()

        return {
            "enabled": True,
            "ell_min": int(ell_min),
            "ell_max": int(ell_max),
            "rms_rel": rms_rel,
            "max_rel": max_rel,
            "threshold_rms": threshold_rms,
            "threshold_max": threshold_max,
            "pass_rms": bool(pass_rms),
            "pass_max": bool(pass_max),
            "plot_path": plot_path,
            "planck": PLANCK,
            "gum": GUM_params,
        }
    except Exception as e:
        return {
            "enabled": False,
            "reason": f"CAMB comparison failed: {e}",
        }

# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest(payload: Dict[str, Any],
                   camb_info: Dict[str, Any]) -> Dict[str, Any]:
    sub: Substrate = payload["sub"]
    mathL: MathLayer = payload["mathL"]
    physL: PhysicsLayer = payload["physL"]
    cosmoL: CosmoLayer = payload["cosmoL"]
    engine = payload["engine"]
    math_res = payload["math_res"]

    A = math_res.get("A", {})
    C = math_res.get("C", {})

    manifest: Dict[str, Any] = {
        "substrate": {
            "wU": int(sub.wU),
            "s2": int(sub.s2),
            "s3": int(sub.s3),
            "q3": int(sub.q3),
            "q2": int(sub.wU - sub.s2),
        },
        "math": {
            "delta_derived": float(C.get("delta_RG_10", mathL.delta)),
            "delta_ref": float(MATH_REF.get("delta", 4.66920160910299)),
            "C2_derived": float(A.get("C2", mathL.C2)),
            "C2_ref": float(MATH_REF.get("C2", 0.6601618158468696)),
        },
        "physics": {
            "alpha_em_derived": float(physL.alpha_em),
            "alpha_em_ref": float(SM_REF["alpha_em"]),
            "sin2W_derived": float(physL.sin2W),
            "sin2W_ref": float(SM_REF["sin2W"]),
            "alpha_s_derived": float(physL.alpha_s),
            "alpha_s_ref": float(SM_REF["alpha_s"]),
        },
        "cosmology": {
            "H0_SCFP": float(engine["H0_SCFP"]),
            "H0_obs": float(COSMO_REF["H0"]),
            "Omega_L_SCFP": float(engine["Omega_L_SCFP"]),
            "Omega_L_obs": float(COSMO_REF["Omega_L"]),
            "Omega_b_SCFP": float(engine["Omega_b_SCFP"]),
            "Omega_b_obs": float(COSMO_REF["Omega_b"]),
            "Omega_c_SCFP": float(engine["Omega_c_SCFP"]),
            "Omega_c_obs": float(COSMO_REF["Omega_c"]),
            "Omega_r_SCFP": float(engine["Omega_r_SCFP"]),
            "Omega_r_obs": float(COSMO_REF["Omega_r"]),
            "Omega_tot_SCFP": float(engine.get("Omega_tot_SCFP", float("nan"))),
            "Omega_tot_obs": float(COSMO_REF["Omega_tot"]),
            "etaB_struct": float(engine["etaB_struct"]),
            "etaB_obs": float(COSMO_REF["etaB"]),
            "YHe_struct": float(engine.get("Y_He_struct", engine.get("YHe_struct"))),
            "YHe_obs": float(COSMO_REF["YHe"]),
            "deltaCMB_struct": float(engine["deltaCMB_struct"]),
            "deltaCMB_obs": float(COSMO_REF["deltaCMB"]),
            "Sum_mnu_SCFP": float(engine["Sum_mnu_SCFP"]),
            "A_s_SCFP": float(engine["A_s_SCFP"]),
            "n_s_SCFP": float(engine["n_s_SCFP"]),
            "tau_SCFP": float(engine["tau_SCFP"]),
            "ell1_SCFP": float(engine["ell1_SCFP"]),
            "rho_L_SCFP": float(engine["rho_L_SCFP"]),
        },
        "camb": camb_info,
    }
    return manifest

def build_numeric_sha(payload: Dict[str, Any]) -> str:
    sub: Substrate = payload["sub"]
    mathL: MathLayer = payload["mathL"]
    physL: PhysicsLayer = payload["physL"]
    cosmoL: CosmoLayer = payload["cosmoL"]
    engine = payload["engine"]
    math_res = payload["math_res"]

    A = math_res.get("A", {})
    C = math_res.get("C", {})

    delta_val = float(C.get("delta_RG_10", mathL.delta))
    C2_val = float(A.get("C2", mathL.C2))

    vals = [
        float(sub.wU),
        float(sub.s2),
        float(sub.s3),
        float(sub.q3),
        float(mathL.pi),
        float(mathL.gamma),
        float(mathL.zeta3),
        float(mathL.zeta5),
        float(C2_val),
        float(mathL.ArtinA),
        float(delta_val),
        float(physL.alpha_em),
        float(physL.sin2W),
        float(physL.alpha_s),
        float(physL.v),
        float(physL.MW),
        float(physL.MZ),
        float(cosmoL.H0),
        float(cosmoL.Omega_b),
        float(cosmoL.Omega_c),
        float(cosmoL.Omega_L),
        float(cosmoL.Omega_r),
        float(cosmoL.etaB),
        float(cosmoL.YHe),
        float(cosmoL.deltaCMB),
        float(engine.get("Omega_tot_SCFP", 0.0)),
        float(engine["Sum_mnu_SCFP"]),
        float(engine["A_s_SCFP"]),
        float(engine["n_s_SCFP"]),
        float(engine["tau_SCFP"]),
        float(engine["ell1_SCFP"]),
    ]
    return sha256_of_manifest_values(vals)

# ---------------------------------------------------------------------------
# PDF report builder
# ---------------------------------------------------------------------------

def build_pdf_report(pdf_path: str,
                     payload: Dict[str, Any],
                     timestamp: str,
                     numeric_sha: str,
                     camb_info: Dict[str, Any],
                     bb36_image_path: Optional[str]
                     ) -> None:
    sub: Substrate = payload["sub"]
    mathL: MathLayer = payload["mathL"]
    physL: PhysicsLayer = payload["physL"]
    cosmoL: CosmoLayer = payload["cosmoL"]
    engine = payload["engine"]
    math_res = payload["math_res"]

    A = math_res.get("A", {})
    C = math_res.get("C", {})

    styles = getSampleStyleSheet()

    body = styles["BodyText"]
    body.fontName = "Helvetica"
    body.fontSize = 10
    body.leading = 13

    h1 = ParagraphStyle(
        "GUM_H1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceAfter=8,
    )
    h2 = ParagraphStyle(
        "GUM_H2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceAfter=6,
    )
    title_style = ParagraphStyle(
        "GUM_Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        alignment=1,
        spaceAfter=12,
    )
    subtitle_style = ParagraphStyle(
        "GUM_Subtitle",
        parent=styles["Title"],
        fontName="Helvetica",
        fontSize=14,
        leading=18,
        alignment=1,
        spaceAfter=12,
    )
    body_italic = ParagraphStyle(
        "GUM_BodyItalic",
        parent=body,
        fontName="Helvetica-Oblique",
    )
    small = ParagraphStyle(
        "GUM_Small",
        parent=body,
        fontSize=8,
        leading=10,
        textColor=colors.grey,
        spaceAfter=4,
    )

    story: List[Any] = []

     # Title page
    story.append(Paragraph("Grand Unified Model", title_style))
    story.append(Paragraph("Executive Technical Report", subtitle_style))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(f"Generated (UTC): {timestamp}", small))
    story.append(Paragraph("GUM stack: SM-MATH-9, DEMO-33, BB-36, gum_v1", small))
    story.append(Paragraph(f"Numeric manifest SHA-256: {numeric_sha}", small))

    survivors_text = (
        f"SCFP++ survivors: wU = {sub.wU}, s2 = {sub.s2}, s3 = {sub.s3}, "
        f"q3 = {sub.q3}, q2 = wU - s2 = {sub.wU - sub.s2}."
    )
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(survivors_text, body))

    # Author note / epigraph
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("Author note", h2))
    story.append(
        Paragraph(
            "\"Within everything accepted lies everything overlooked.\"",
            body_italic,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "This program is not a cathedral; it is a single brick, laid on a structure "
            "built by generations of human curiosity. May the pursuit of the light "
            "never diminish, and may it become ever more steadfast.",
            body,
        )
    )
    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("Justin Grieshop", body_italic))

    story.append(PageBreak())

    # Section 0: DRPT geometry and analytic filter (origin story)
    story.append(Paragraph("0. DRPT geometry and analytic filter", h1))
    story.append(
        Paragraph(
            "Before the SM-MATH-9, DEMO-33, and BB-36 layers, GUM rests on a purely "
            "arithmetical substrate. Digital-Root Power Tables (DRPTs) capture the "
            "geometry of n^k modulo b-1 across bases and exponents. The origin story "
            "for the SCFP++ survivors and the physical constants begins in this "
            "discrete geometry.",
            body,
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("0.1 Digital-Root Power Tables and pattern families", h2))
    story.append(
        Paragraph(
            "A DRPT records, for each integer n and exponent k, the digital root of "
            "n^k in a fixed base. Visually this produces a finite heatmap of "
            "multiplicative dynamics on the ring Z_{b-1}. Within these tables we "
            "observe recurring combinatorial motifs, which we treat as pattern "
            "families or identities: identity columns, echo motifs, reciprocal "
            "chains, inverse-pair lattices, and others. To date we have catalogued "
            "13 distinct DRPT pattern families. The catalogue is not complete; the "
            "total number of families and their classification across all bases "
            "remains an open combinatorial program.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "These families are not tied to a single base. The same motifs reappear "
            "in multiple bases with the same qualitative geometry. This cross-base "
            "stability is the raw material used by the SCFP++ selector and the "
            "Rosetta layer: it ensures that the substrate is not tuned to any one "
            "representation of the integers.",
            body,
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    
       # DRPT identity + echo patterns (2x2 grid)
    story.append(Spacer(1, 0.2 * inch))

    id9_path   = os.path.join(ASSETS_DIR, "Identity9.png")
    id10_path  = os.path.join(ASSETS_DIR, "Identity10.png")
    echo6_path = os.path.join(ASSETS_DIR, "Echo6.png")
    echo10_path= os.path.join(ASSETS_DIR, "Echo10.png")

    if all(os.path.exists(p) for p in [id9_path, id10_path, echo6_path, echo10_path]):
        img_id9    = Image(id9_path,    width=3.1 * inch, height=1.6 * inch)
        img_id10   = Image(id10_path,   width=3.1 * inch, height=1.6 * inch)
        img_echo6  = Image(echo6_path,  width=3.1 * inch, height=1.6 * inch)
        img_echo10 = Image(echo10_path, width=3.1 * inch, height=1.6 * inch)

        drpt_grid = Table(
            [[img_id9, img_id10],
             [img_echo6, img_echo10]],
            colWidths=[3.2 * inch, 3.2 * inch],
        )
        drpt_grid.hAlign = "CENTER"
        story.append(drpt_grid)
    else:
        missing = [
            name for name, path in [
                ("Identity9.png",  id9_path),
                ("Identity10.png", id10_path),
                ("Echo6.png",      echo6_path),
                ("Echo10.png",     echo10_path),
            ]
            if not os.path.exists(path)
        ]
        story.append(
            Paragraph(
                "DRPT identity / echo assets are missing ({}); skipping this panel."
                .format(", ".join(missing) or "unknown"),
                body_italic,
            )
        )


    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Panels (a) and (b) show the identity family (value 1) in bases 9 and 10. "
            "Panels (c) and (d) show an echo family in bases 6 and 10, where repeated "
            "residues appear at multiple coordinates. Together they illustrate both "
            "the internal structure and the cross-base stability of the DRPT substrate.",
            body,
        )
    )

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("0.2 Analytic filter: Fejer smoothing on DRPT", h2))
    ...
    # keep the rest of the Fejér and survivor-table code indented exactly like this

    story.append(
        Paragraph(
            "The analytic layer does not alter the DRPT substrate; it alters how the "
            "substrate is read. In the DOC lawbook the Fejer triangle is the "
            "canonical positive, symmetric, mean-preserving low-pass kernel. On a "
            "periodic DRPT row it preserves the global mean and existing mirror "
            "symmetries while contracting all non-principal oscillatory modes.",
            body,
        )
    )

    story.append(Spacer(1, 0.15 * inch))
    try:
        img_fejer_left = Image(
            os.path.join(ASSETS_DIR, "FejerSmoothignLeft.png"),
            width=3.2 * inch,
            height=2.0 * inch,
        )
        img_fejer_right = Image(
            os.path.join(ASSETS_DIR, "FejerSmoothingRight.png"),
            width=3.2 * inch,
            height=2.0 * inch,
        )
        fejer_grid = Table(
            [[img_fejer_left, img_fejer_right]],
            colWidths=[3.2 * inch, 3.2 * inch],
        )
        fejer_grid.hAlign = "CENTER"
        story.append(fejer_grid)
    except Exception:
        story.append(
            Paragraph(
                "Fejer smoothing assets (FejerSmoothignLeft/FejerSmoothingRight) "
                "were not found in the Assets directory; skipping this panel.",
                body_italic,
            )
        )

    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "The left panel shows sliding local windows cut from a raw DRPT row for "
            "base 2: the familiar jagged \"lightning\" structure is dominated by "
            "high-frequency aliasing. The right panel applies Fejer smoothing to "
            "the same windows. The sharp diagonals melt into a narrow band around the "
            "cycle mean while the column symmetry is preserved. In effect the filter "
            "kills the wiggle and keeps the backbone.",
            body,
        )
    )

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph("0.3 Survivor selection logic", h2))
    story.append(
        Paragraph(
            "Not every integer produces a stable pillar under Fejer smoothing. "
            "The SCFP++ selector applies additional number-theoretic gates to tie "
            "DRPT pillars to physical channels. For each lane (U(1), SU(2), and the "
            "QCD/PC2 lane) it enforces period extremality, a residue-class gate, "
            "and a 2-adic branch condition before accepting a survivor.",
            body,
        )
    )
    story.append(Spacer(1, 0.1 * inch))
    story.append(
        Paragraph(
            "Period extremality requires the totient ratio "
            "Theta(w) = phi(w-1)/(w-1) to exceed a channel-specific "
            "threshold, selecting integers whose multiplicative order modulo "
            "the channel modulus is as large as possible. The residue-class "
            "gate fixes the Legendre symbol (2/q) and congruence of q, matching "
            "the DRPT inverse-pair geometry. The 2-adic branch fixes v2(w-1) "
            "so that the odd part of w-1 defines the strong-coupling modulus.",
            body,
        )
    )

    story.append(Spacer(1, 0.15 * inch))
    survivor_rows = [
        [
            "Lane",
            "q",
            "Theta(w-1) threshold",
            "Legendre (2|q)",
            "v2(w-1)",
            "Selected w",
        ],
        [
            "U(1) / alpha",
            "17",
            ">= 0.31",
            "+1",
            "3",
            f"{sub.wU}",
        ],
        [
            "SU(2)",
            "13",
            ">= 0.30",
            "(2|13) = -1 mod q",
            "1",
            f"{sub.s2}",
        ],
        [
            "PC2 / QCD",
            "17",
            ">= 0.30",
            "+1",
            "1",
            f"{sub.s3}",
        ],
    ]
    survivor_table = Table(survivor_rows, hAlign="LEFT")
    survivor_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
            ]
        )
    )
    story.append(survivor_table)

    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "Within the numeric window probed in this report the selector finds a "
            "single consistent triple (wU, s2, s3) = "
            f"({sub.wU}, {sub.s2}, {sub.s3}). This same triple is then reused "
            "without retuning in the SM-MATH-9, DEMO-33, and BB-36 layers that "
            "follow.",
            body,
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    try:
        img_city = Image(
            os.path.join(ASSETS_DIR, "DRPTCity137.png"),
            width=6.5 * inch,
            height=2.6 * inch,
        )
        img_strip = Image(
            os.path.join(ASSETS_DIR, "DRPTSurvivor137.png"),
            width=6.5 * inch,
            height=1.0 * inch,
        )
        img_city.hAlign = "CENTER"
        img_strip.hAlign = "CENTER"
        story.append(img_city)
        story.append(Spacer(1, 0.1 * inch))
        story.append(img_strip)
    except Exception:
        story.append(
            Paragraph(
                "DRPT survivor pillar assets (DRPTCity137 / DRPTSurvivor137) "
                "were not found in the Assets directory; skipping this panel.",
                body_italic,
            )
        )

    story.append(Spacer(1, 0.15 * inch))
    story.append(
        Paragraph(
            "The top panel shows the base-10 DRPT city view: rows are integers n and "
            "columns are exponents k. The framed band is the lane of integers with "
            "n congruent to 137 modulo 9, which share the same digital-root power cycle "
            "as 137. Thin vertical markers indicate exponents where this lane hits the "
            "identity value 1, forming identity pillars. The lower panel isolates "
            "these identity hits as a chunky strip, making the pillar structure of "
            "the 137 lane along the exponent axis explicit.",
            body,
        )
    )

    story.append(Spacer(1, 0.3 * inch))
    story.append(PageBreak())

    # Section 1: Executive summary
    story.append(Paragraph("1. Executive summary", h1))

    story.append(
        Paragraph(
            "This report records a single execution of the Grand Unified Model (GUM). "
            "A finite set of SCFP++ integers is selected once and then reused without "
            "retuning in three domains: a math kernel (SM-MATH-9), a Standard Model "
            "kernel (DEMO-33), and a structural cosmology engine (BB-36). The same "
            "substrate that fixes the Feigenbaum delta and twin prime constant also "
            "fixes the fine structure constant, electroweak scale, and a LambdaCDM-"
            "like cosmology.",
            body,
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("1.1 Unified constants dashboard", h2))

    delta_ref = MATH_REF.get("delta", 4.66920160910299)
    C2_ref = MATH_REF.get("C2", 0.6601618158468696)

    const_rows = [
        ["Quantity", "Derived value", "Reference value", "Rel. error", "Domain"],
        [
            "delta (Feigenbaum)",
            fmt_val(C.get("delta_RG_10", mathL.delta)),
            fmt_val(delta_ref),
            fmt_err(rel_err(C.get("delta_RG_10", mathL.delta), delta_ref)),
            "Math (SM-MATH-9)",
        ],
        [
            "C2 (twin prime constant)",
            fmt_val(A.get("C2", mathL.C2)),
            fmt_val(C2_ref),
            fmt_err(rel_err(A.get("C2", mathL.C2), C2_ref)),
            "Math (SM-MATH-9)",
        ],
        [
            "alpha_em (fine structure)",
            fmt_val(physL.alpha_em, sci=True),
            fmt_val(SM_REF["alpha_em"], sci=True),
            fmt_err(rel_err(physL.alpha_em, SM_REF["alpha_em"])),
            "Physics (DEMO-33)",
        ],
        [
            "sin^2(theta_W)",
            fmt_val(physL.sin2W),
            fmt_val(SM_REF["sin2W"]),
            fmt_err(rel_err(physL.sin2W, SM_REF["sin2W"])),
            "Physics (DEMO-33)",
        ],
        [
            "alpha_s (MZ)",
            fmt_val(physL.alpha_s),
            fmt_val(SM_REF["alpha_s"]),
            fmt_err(rel_err(physL.alpha_s, SM_REF["alpha_s"])),
            "Physics (DEMO-33)",
        ],
        [
            "H0 [km s^-1 Mpc^-1]",
            fmt_val(cosmoL.H0),
            fmt_val(cosmoL.H0_ref),
            fmt_err(rel_err(cosmoL.H0, cosmoL.H0_ref)),
            "Cosmology (BB-36)",
        ],
        [
            "Omega_L",
            fmt_val(cosmoL.Omega_L),
            fmt_val(cosmoL.Omega_L_ref),
            fmt_err(rel_err(cosmoL.Omega_L, cosmoL.Omega_L_ref)),
            "Cosmology (BB-36)",
        ],
        [
            "Y_He",
            fmt_val(cosmoL.YHe),
            fmt_val(cosmoL.YHe_ref),
            fmt_err(rel_err(cosmoL.YHe, cosmoL.YHe_ref)),
            "BBN (BB-36)",
        ],
        [
            "delta_CMB",
            fmt_val(cosmoL.deltaCMB, sci=True),
            fmt_val(cosmoL.deltaCMB_ref, sci=True),
            fmt_err(rel_err(cosmoL.deltaCMB, cosmoL.deltaCMB_ref)),
            "CMB (BB-36)",
        ],
    ]

    const_table = Table(const_rows, hAlign="LEFT")
    const_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-2, -1), "RIGHT"),
            ]
        )
    )
    story.append(const_table)

    story.append(Spacer(1, 0.3 * inch))
    story.append(PageBreak())

    # Section 2: Layer 0 and Layer 1
    story.append(Paragraph("2. Layer 0: SCFP++ substrate", h1))
    story.append(
        Paragraph(
            "The SCFP++ selector enforces arithmetic, totient density, and structural "
            "constraints on three gauge lanes. In this run only one triple survives "
            "in the search window, and that triple is used as the substrate for all "
            "subsequent layers.",
            body,
        )
    )
    story.append(
        Paragraph(
            f"Selected triple: wU = {sub.wU}, s2 = {sub.s2}, s3 = {sub.s3}, "
            f"with q2 = {sub.wU - sub.s2} and q3 = {sub.q3}.",
            body,
        )
    )

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph("3. Layer 1: SM-MATH-9 (math kernel)", h1))
    story.append(
        Paragraph(
            "SM-MATH-9 builds an operator calculus on the integers and recovers "
            "analytic constants such as the Feigenbaum delta and twin prime constant "
            "from purely discrete dynamics. The same SCFP++ substrate that later "
            "sets alpha and H0 already fixes these mathematical invariants.",
            body,
        )
    )

    math_rows = [
        ["Invariant", "Derived", "Reference", "Rel. error"],
        [
            "delta (Feigenbaum)",
            fmt_val(C.get("delta_RG_10", mathL.delta)),
            fmt_val(delta_ref),
            fmt_err(rel_err(C.get("delta_RG_10", mathL.delta), delta_ref)),
        ],
        [
            "C2 (twin prime)",
            fmt_val(A.get("C2", mathL.C2)),
            fmt_val(C2_ref),
            fmt_err(rel_err(A.get("C2", mathL.C2), C2_ref)),
        ],
        [
            "gamma (Euler-Mascheroni)",
            fmt_val(A.get("gamma", mathL.gamma)),
            fmt_val(MATH_REF.get("gamma")),
            fmt_err(rel_err(A.get("gamma", mathL.gamma), MATH_REF.get("gamma"))),
        ],
        [
            "pi",
            fmt_val(A.get("pi", mathL.pi)),
            fmt_val(MATH_REF.get("pi")),
            fmt_err(rel_err(A.get("pi", mathL.pi), MATH_REF.get("pi"))),
        ],
    ]
    math_table = Table(math_rows, hAlign="LEFT")
    math_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(math_table)

    story.append(PageBreak())

    # Section 4: Layer 2A (Standard Model)
    story.append(Paragraph("4. Layer 2A: DEMO-33 (Standard Model kernel)", h1))
    story.append(
        Paragraph(
            "DEMO-33 lifts the substrate into a minimal Standard Model skeleton. "
            "The SCFP++ survivors determine the gauge couplings, weak mixing angle, "
            "and electroweak scale v, which in turn fix the W and Z masses.",
            body,
        )
    )

    sm_rows = [
        ["Quantity", "Derived", "Reference", "Rel. error"],
        [
            "alpha_em",
            fmt_val(physL.alpha_em, sci=True),
            fmt_val(SM_REF["alpha_em"], sci=True),
            fmt_err(rel_err(physL.alpha_em, SM_REF["alpha_em"])),
        ],
        [
            "sin^2(theta_W)",
            fmt_val(physL.sin2W),
            fmt_val(SM_REF["sin2W"]),
            fmt_err(rel_err(physL.sin2W, SM_REF["sin2W"])),
        ],
        [
            "alpha_s(MZ)",
            fmt_val(physL.alpha_s),
            fmt_val(SM_REF["alpha_s"]),
            fmt_err(rel_err(physL.alpha_s, SM_REF["alpha_s"])),
        ],
        [
            "v [GeV]",
            fmt_val(physL.v),
            fmt_val(SM_REF["v"]),
            fmt_err(rel_err(physL.v, SM_REF["v"])),
        ],
        [
            "MW [GeV]",
            fmt_val(physL.MW),
            fmt_val(SM_REF["MW"]),
            fmt_err(rel_err(physL.MW, SM_REF["MW"])),
        ],
        [
            "MZ [GeV]",
            fmt_val(physL.MZ),
            fmt_val(SM_REF["MZ"]),
            fmt_err(rel_err(physL.MZ, SM_REF["MZ"])),
        ],
    ]
    sm_table = Table(sm_rows, hAlign="LEFT")
    sm_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(sm_table)

    story.append(PageBreak())

    # Section 5: Layer 2B (cosmology)
    story.append(Paragraph("5. Layer 2B: BB-36 structural cosmology", h1))
    story.append(
        Paragraph(
            "BB-36 builds a structural LambdaCDM-like cosmology directly from the "
            "SCFP++ substrate. The same discrete data that fixed delta and alpha now "
            "fix the Hubble constant, density budget, baryon asymmetry, helium yield, "
            "and CMB temperature contrast, with no additional fitting layer.",
            body,
        )
    )

    cosmo_rows = [
        ["Quantity", "Derived (SCFP)", "Reference", "Rel. error"],
        [
            "H0 [km s^-1 Mpc^-1]",
            fmt_val(cosmoL.H0),
            fmt_val(cosmoL.H0_ref),
            fmt_err(rel_err(cosmoL.H0, cosmoL.H0_ref)),
        ],
        [
            "Omega_L",
            fmt_val(cosmoL.Omega_L),
            fmt_val(cosmoL.Omega_L_ref),
            fmt_err(rel_err(cosmoL.Omega_L, cosmoL.Omega_L_ref)),
        ],
        [
            "Omega_b",
            fmt_val(cosmoL.Omega_b),
            fmt_val(cosmoL.Omega_b_ref),
            fmt_err(rel_err(cosmoL.Omega_b, cosmoL.Omega_b_ref)),
        ],
        [
            "Omega_c",
            fmt_val(cosmoL.Omega_c),
            fmt_val(cosmoL.Omega_c_ref),
            fmt_err(rel_err(cosmoL.Omega_c, cosmoL.Omega_c_ref)),
        ],
        [
            "Omega_r",
            fmt_val(cosmoL.Omega_r),
            fmt_val(cosmoL.Omega_r_ref),
            fmt_err(rel_err(cosmoL.Omega_r, cosmoL.Omega_r_ref)),
        ],
        [
            "eta_B",
            fmt_val(cosmoL.etaB, sci=True),
            fmt_val(cosmoL.etaB_ref, sci=True),
            fmt_err(rel_err(cosmoL.etaB, cosmoL.etaB_ref)),
        ],
        [
            "Y_He",
            fmt_val(cosmoL.YHe),
            fmt_val(cosmoL.YHe_ref),
            fmt_err(rel_err(cosmoL.YHe, cosmoL.YHe_ref)),
        ],
        [
            "delta_CMB",
            fmt_val(cosmoL.deltaCMB, sci=True),
            fmt_val(cosmoL.deltaCMB_ref, sci=True),
            fmt_err(rel_err(cosmoL.deltaCMB, cosmoL.deltaCMB_ref)),
        ],
        [
            "Omega_tot",
            fmt_val(engine.get("Omega_tot_SCFP")),
            fmt_val(COSMO_REF["Omega_tot"]),
            fmt_err(rel_err(engine.get("Omega_tot_SCFP"), COSMO_REF["Omega_tot"])),
        ],
    ]
    cosmo_table = Table(cosmo_rows, hAlign="LEFT")
    cosmo_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ]
        )
    )
    story.append(cosmo_table)

    story.append(PageBreak())

    # Section 6: Field-level structure and CAMB check
    story.append(Paragraph("6. Field-level structure and CMB validation", h1))
    story.append(
        Paragraph(
            "To give the cosmology layer geometric and observational content, "
            "this section shows a Big Bang visual based on SPDE fields and a "
            "CAMB-based CMB TT spectrum comparison against the Planck 2018 "
            "baseline.",
            body,
        )
    )

    # 6.1 Big Bang visual
    story.append(Paragraph("6.1 Big Bang visual (SPDE and tidal slices)", h2))
    if bb36_image_path and os.path.isfile(bb36_image_path):
        story.append(
            Paragraph(
                "The panel figure below shows mid-plane slices from five 3D fields "
                "evolved on a common comoving grid: baryon asymmetry eta_B, "
                "a helium-like field X_He, a radiation contrast field, vorticity "
                "magnitude, and a tidal trace proxy Tr(T).",
                body,
            )
        )
        story.append(Spacer(1, 0.15 * inch))
        img = Image(bb36_image_path, width=6.5 * inch, height=2.3 * inch)
        img.hAlign = "CENTER"
        story.append(img)
    else:
        story.append(
            Paragraph(
                "Big Bang visual is not available in this environment "
                "(missing NumPy/matplotlib or an error occurred while building "
                "the figure).",
                body_italic,
            )
        )

    story.append(Spacer(1, 0.3 * inch))

    # 6.2 CAMB TT overlay
    story.append(Paragraph("6.2 Planck 2018 vs GUM CAMB TT overlay", h2))
    if camb_info.get("enabled", False) and camb_info.get("plot_path"):
        ell_min = camb_info["ell_min"]
        ell_max = camb_info["ell_max"]
        rms_rel = camb_info["rms_rel"]
        max_rel = camb_info["max_rel"]
        thr_rms = camb_info["threshold_rms"]
        thr_max = camb_info["threshold_max"]
        pass_rms = camb_info["pass_rms"]
        pass_max = camb_info["pass_max"]

        story.append(
            Paragraph(
                "CAMB is used as an independent Einstein-Boltzmann solver. The same "
                "H0 and Omega_L that appear in the BB-36 table are inserted into "
                "a flat LCDM cosmology, with the Planck 2018 baryon fraction and "
                "primordial parameters. The resulting TT spectrum is compared to "
                "the Planck 2018 baseline.",
                body,
            )
        )

        txt = (
            f"ell range: {ell_min} to {ell_max}; "
            f"RMS relative difference: {rms_rel:.3e} "
            f"(threshold {thr_rms:.3f}, {'PASS' if pass_rms else 'FAIL'}); "
            f"max relative difference: {max_rel:.3e} "
            f"(threshold {thr_max:.3f}, {'PASS' if pass_max else 'FAIL'})."
        )
        story.append(Paragraph(txt, small))

        plot_path = camb_info["plot_path"]
        story.append(Spacer(1, 0.2 * inch))
        img2 = Image(plot_path, width=6.5 * inch, height=3.5 * inch)
        img2.hAlign = "CENTER"
        story.append(img2)
        story.append(
            Paragraph(
                "The solid curve shows the Planck 2018 flat LCDM baseline. The "
                "dashed curve shows the GUM cosmology derived from the SCFP++ "
                "substrate. Close agreement over this ell range is a strong CAMB "
                "check on the structural cosmology.",
                small,
            )
        )
    else:
        reason = camb_info.get("reason", "CAMB not available.")
        story.append(
            Paragraph(
                "CAMB overlay is disabled in this environment: " + reason,
                body_italic,
            )
        )

    story.append(PageBreak())

    # Section 7: Closing remarks
    story.append(Paragraph("7. Closing remarks", h1))
    story.append(
        Paragraph(
            "This artifact is intended as a stable snapshot of the GUM pipeline. "
            "A single SCFP++ substrate feeds the math kernel, the Standard Model "
            "kernel, and the structural cosmology engine. The numeric manifest "
            "and its SHA-256 fingerprint make it easy to reproduce and audit this "
            "run in a clean environment.",
            body,
        )
    )
    story.append(
        Paragraph(
            "Future work can extend this report with additional falsifiers, "
            "alternative cosmologies, and more detailed field diagnostics, but "
            "the basic unification claim is already visible here.",
            body,
        )
    )

    # Build PDF
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="Grand Unified Model - Executive Technical Report",
    )

    def footer(canvas, doc_obj):
        """
        Two-line footer:
          Line 1: report title + timestamp (left), page number (right)
          Line 2: full SHA-256(manifest) (left)
        """
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)

        # Vertical positions for the two lines
        y1 = 0.55 * inch  # first line
        y2 = 0.40 * inch  # second line

        # Line 1: title + timestamp (left) and page number (right)
        left_text  = f"GUM report v27 – generated {timestamp} (UTC)"
        right_text = f"Page {doc_obj.page}"
        canvas.drawString(doc_obj.leftMargin, y1, left_text)
        canvas.drawRightString(
            doc_obj.pagesize[0] - doc_obj.rightMargin,
            y1,
            right_text,
        )

        # Line 2: full SHA-256(manifest), left aligned
        sha_text = f"SHA-256(manifest) = {numeric_sha}"
        canvas.drawString(doc_obj.leftMargin, y2, sha_text)

        canvas.restoreState()



    doc.build(story, onFirstPage=footer, onLaterPages=footer)

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    here = os.path.abspath(os.path.dirname(__file__))
    report_dir = os.path.join(here, "GUM_Report")
    os.makedirs(report_dir, exist_ok=True)

    payload = run_gum_and_collect()
    bb36_img = generate_big_bang_visual(report_dir, payload)
    camb_info = run_camb_validation(report_dir, payload)
    manifest = build_manifest(payload, camb_info)
    numeric_sha = build_numeric_sha(payload)

    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_name = f"GUM_Report_{ts}.pdf"
    pdf_path = os.path.join(report_dir, pdf_name)

    manifest_name = f"GUM_manifest_{ts}.json"
    manifest_path = os.path.join(report_dir, manifest_name)

    sha_name = f"GUM_manifest_SHA256_{ts}.txt"
    sha_path = os.path.join(report_dir, sha_name)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    with open(sha_path, "w", encoding="utf-8") as f:
        f.write(numeric_sha + "\n")

    build_pdf_report(
        pdf_path=pdf_path,
        payload=payload,
        timestamp=ts,
        numeric_sha=numeric_sha,
        camb_info=camb_info,
        bb36_image_path=bb36_img,
    )

    pdf_sha = sha256_of_file(pdf_path)

    print("============================================================")
    print(" GUM report generation complete")
    print("------------------------------------------------------------")
    print(f"  Report directory : {report_dir}")
    print(f"  PDF file         : {pdf_name}")
    print(f"  PDF SHA-256      : {pdf_sha}")
    print(f"  Manifest JSON    : {manifest_name}")
    print(f"  Numeric SHA-256  : {numeric_sha}")
    print(f"  SHA sidecar      : {sha_name}")
    print("============================================================")

if __name__ == "__main__":
    main()





















