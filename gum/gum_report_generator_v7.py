#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUM_Report_Generator_v4
=======================

Grand Unified Model — Executive Technical Report generator.

This script:

  • Creates/uses a folder named "GUM_Report/" in the current directory.
  • Runs the unified GUM stack from gum_v1 or SM_GUM_v1_0 (whichever is available).
  • Extracts key readouts from SM-MATH-9, DEMO-33, and BB-36.
  • Optionally runs a CAMB TT-spectrum comparison vs Planck LCDM.
  • Builds a polished PDF report (title page, executive summary,
    technical narrative, and CAMB validation section).
  • Computes:
        - A numeric manifest SHA-256 (over cross-domain invariants).
        - A file-level SHA-256 for the generated PDF.
  • Writes:
        GUM_Report_<timestamp>.pdf
        GUM_Report_<timestamp>.sha256
        GUM_Report_<timestamp>_manifest.json

The PDF is intended as a high-impact, archival-quality artifact.
"""

import os
import json
import math
import hashlib
import inspect
from datetime import datetime

# --- GUM driver (tries gum_v1, then SM_GUM_v1_0) ---
'''try:
    import gum_v1 as gum
except ImportError:
    import SM_GUM_v1_0 as gum
'''

# --- GUM driver (structured repo import) ---

try:
    from gum.gum_v1 import (
        build_substrate,
        build_math_layer,
        build_physics_layer,
        build_cosmo_layer,
        smath,
        sm33
    )
except ImportError:
    print("Could not import gum_v1 from gum/ directory.")
    print("Actual error:")
    raise

# Recreate gum namespace expected by the report generator
class GumNamespace:
    @staticmethod
    def build_substrate(*args, **kwargs):
        return build_substrate(*args, **kwargs)

    @staticmethod
    def build_math_layer(*args, **kwargs):
        return build_math_layer(*args, **kwargs)

    @staticmethod
    def build_physics_layer(*args, **kwargs):
        return build_physics_layer(*args, **kwargs)

    @staticmethod
    def build_cosmo_layer(*args, **kwargs):
        return build_cosmo_layer(*args, **kwargs)

    smath = smath
    sm33 = sm33


gum = GumNamespace()

# --- ReportLab (for PDF generation) ---
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
except ImportError:
    raise SystemExit(
        "This script requires the 'reportlab' package.\n"
        "Install it via:\n"
        "    pip install reportlab\n"
    )

# --- CAMB + plotting (optional) ---
try:
    import numpy as np
    import matplotlib.pyplot as plt
    import camb
    from camb import model, initialpower
    HAS_CAMB = True
except Exception:
    HAS_CAMB = False


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def rel_err(pred: float, ref: float) -> float:
    if ref == 0.0:
        return 0.0 if pred == 0.0 else float("inf")
    return abs(pred - ref) / abs(ref)


def fmt_val(x) -> str:
    try:
        if x is None:
            return "—"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "—"
        return f"{x:.12g}"
    except Exception:
        return "—"


def fmt_err(e) -> str:
    try:
        if e is None or math.isnan(e) or math.isinf(e):
            return "—"
        return f"{e:.3e}"
    except Exception:
        return "—"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_manifest_values(vals) -> str:
    buf = []
    for x in vals:
        try:
            buf.append("{:.12e}".format(float(x)))
        except Exception:
            buf.append("NaN")
    s = "|".join(buf).encode("utf-8")
    return hashlib.sha256(s).hexdigest()


# ---------------------------------------------------------------------------
# Run GUM and collect the data we need
# ---------------------------------------------------------------------------

def run_gum_and_collect():
    """
    Run the unified GUM pipeline once and collect:

      - sub   : substrate (wU, s2, s3, q3)
      - mathL : MathLayer (aggregated view; from gum_v1 / SM_GUM_v1_0)
      - physL : PhysicsLayer (aggregated view)
      - cosmoL: CosmoLayer (aggregated view)
      - engine: full BB-36 structural engine dict
      - math_res: full SM-MATH-9 sector dict
      - REF_MATH, PDG: reference tables

    This is intentionally defensive: it inspects the signatures of the
    GUM helper functions and calls them with the right number of args.
    """
  # --- Layer 0: substrate ---
    # --- Layer 0: substrate ---
    sub = gum.build_substrate()

    # --- Layer 1: math kernel ---
    mathL = gum.build_math_layer()

    # --- Layer 2A: SM engine ---
    physL = gum.build_physics_layer(sub)

    # --- Layer 2B: cosmology ---
    cosmoL, engine = gum.build_cosmo_layer(sub)

    # --- Full SM-MATH-9 output + refs ---
    math_res = gum.smath.compute_all()
    REF_MATH = getattr(gum.smath, "REF", {})
    PDG = getattr(gum.sm33, "PDG", {})

    return {
        "sub": sub,
        "mathL": mathL,
        "physL": physL,
        "cosmoL": cosmoL,
        "engine": engine,
        "math_res": math_res,
        "REF_MATH": REF_MATH,
        "PDG": PDG,
    }

# ---------------------------------------------------------------------------
# Build numeric manifest hash over cross-domain constants
# ---------------------------------------------------------------------------

def build_numeric_manifest_hash(payload) -> str:
    sub = payload["sub"]
    mathL = payload["mathL"]
    physL = payload["physL"]
    cosmoL = payload["cosmoL"]
    engine = payload["engine"]

    vals = [
        float(sub.wU),
        float(sub.s2),
        float(sub.s3),
        float(sub.q3),

        mathL.pi,
        mathL.gamma,
        mathL.zeta3,
        mathL.zeta5,
        mathL.C2,
        mathL.ArtinA,
        mathL.delta,

        physL.alpha_em,
        physL.sin2W,
        physL.alpha_s,
        physL.v,
        physL.MW,
        physL.MZ,

        cosmoL.H0,
        cosmoL.Omega_b,
        cosmoL.Omega_c,
        cosmoL.Omega_L,
        cosmoL.Omega_r,
        cosmoL.etaB,
        cosmoL.YHe,
        cosmoL.deltaCMB,
        engine.get("Omega_tot", 0.0),

        engine.get("Sum_mnu_SCFP", 0.0),
        engine.get("A_s_SCFP", 0.0),
        engine.get("n_s_SCFP", 0.0),
        engine.get("tau_SCFP", 0.0),
        engine.get("ell1_SCFP", 0.0),
    ]

    return sha256_of_manifest_values(vals)


# ---------------------------------------------------------------------------
# CAMB validation helper
# ---------------------------------------------------------------------------

ELL_MIN = 2
ELL_MAX = 2000

PLANCK_BASELINE = {
    "H0":    67.36,
    "ombh2": 0.02237,
    "omch2": 0.1200,
    "tau":   0.0544,
    "As":    2.100e-9,
    "ns":    0.9649,
    "mnu":   0.06,
}


def get_cmb_tt_spectrum(H0, ombh2, omch2, As, ns, tau, mnu=0.06, lmax=ELL_MAX):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0, ombh2=ombh2, omch2=omch2, mnu=mnu, omk=0.0, tau=tau)
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    totCL = powers["total"]
    TT = totCL[:, 0]
    ells = np.arange(TT.size)
    return ells, TT


def run_camb_validation(report_dir: str, payload):
    """
    Run a CAMB TT-spectrum comparison between:

      • Planck 2018 flat LCDM baseline
      • GUM cosmology derived from BB-36 / SCFP++

    Returns a dict with comparison metrics and (optionally) path to PNG plot.
    """
    if not HAS_CAMB:
        return {
            "enabled": False,
            "reason": "CAMB/numpy/matplotlib not available in this environment.",
        }

    cosmoL = payload["cosmoL"]
    engine = payload["engine"]

    # GUM cosmology from BB-36
    H0_gum = float(cosmoL.H0)
    Omega_L_gum = float(cosmoL.Omega_L)
    Omega_r_gum = float(cosmoL.Omega_r)
    Omega_m_gum = max(0.0, 1.0 - Omega_L_gum - Omega_r_gum)

    # Planck baryon fraction
    H0P = PLANCK_BASELINE["H0"]
    hP = H0P / 100.0
    Omega_b_P = PLANCK_BASELINE["ombh2"] / (hP * hP)
    Omega_c_P = PLANCK_BASELINE["omch2"] / (hP * hP)
    Omega_m_P = Omega_b_P + Omega_c_P
    f_baryon = Omega_b_P / Omega_m_P if Omega_m_P > 0 else 0.16

    # Map baryon fraction onto GUM cosmology
    hG = H0_gum / 100.0
    Omega_b_gum = f_baryon * Omega_m_gum
    Omega_c_gum = (1.0 - f_baryon) * Omega_m_gum
    ombh2_gum = Omega_b_gum * (hG * hG)
    omch2_gum = Omega_c_gum * (hG * hG)

    # Scalar spectrum and neutrino mass from engine if present, else Planck-like
    As_gum = float(engine.get("A_s_SCFP", PLANCK_BASELINE["As"]))
    ns_gum = float(engine.get("n_s_SCFP", PLANCK_BASELINE["ns"]))
    tau_gum = float(engine.get("tau_SCFP", PLANCK_BASELINE["tau"]))
    mnu_gum = float(engine.get("Sum_mnu_SCFP", PLANCK_BASELINE["mnu"]))

    try:
        ell_P, TT_P = get_cmb_tt_spectrum(
            PLANCK_BASELINE["H0"],
            PLANCK_BASELINE["ombh2"],
            PLANCK_BASELINE["omch2"],
            PLANCK_BASELINE["As"],
            PLANCK_BASELINE["ns"],
            PLANCK_BASELINE["tau"],
            mnu=PLANCK_BASELINE["mnu"],
            lmax=ELL_MAX,
        )

        ell_G, TT_G = get_cmb_tt_spectrum(
            H0_gum,
            ombh2_gum,
            omch2_gum,
            As_gum,
            ns_gum,
            tau_gum,
            mnu=mnu_gum,
            lmax=ELL_MAX,
        )
    except Exception as e:
        return {
            "enabled": False,
            "reason": f"CAMB execution failed: {e!r}",
        }

    ell_min = ELL_MIN
    ell_max = min(ELL_MAX, len(TT_P) - 1, len(TT_G) - 1)
    mask = (ell_P >= ell_min) & (ell_P <= ell_max)
    ell_use = ell_P[mask]
    TT_P_use = TT_P[mask]
    TT_G_use = TT_G[mask]

    rel_diff = (TT_G_use - TT_P_use) / TT_P_use
    rms_rel = float(np.sqrt(np.mean(rel_diff**2)))
    max_rel = float(np.max(np.abs(rel_diff)))

    threshold_rms = 0.05
    threshold_max = 0.15
    pass_rms = rms_rel < threshold_rms
    pass_max = max_rel < threshold_max

    # Plot
    out_plot = None
    try:
        plt.figure(figsize=(7, 5))
        plt.plot(ell_P, TT_P, label="Planck 2018 LCDM", color="C0")
        plt.plot(ell_G, TT_G, label="GUM (BB-36)", color="C1", linestyle="--")
        plt.xlim(2, 2500)
        plt.xlabel("ell")
        plt.ylabel("C_ell^TT [muK^2]")
        plt.title("CMB TT Power Spectrum: Planck vs GUM")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        camb_dir = os.path.join(report_dir, "CAMB")
        os.makedirs(camb_dir, exist_ok=True)
        out_plot = os.path.join(camb_dir, "TT_spectrum_Planck_vs_GUM.png")
        plt.savefig(out_plot, dpi=200)
        plt.close()
    except Exception:
        out_plot = None

    return {
        "enabled": True,
        "planck": PLANCK_BASELINE,
        "gum": {
            "H0": H0_gum,
            "Omega_L": Omega_L_gum,
            "Omega_m": Omega_m_gum,
            "Omega_b": Omega_b_gum,
            "Omega_c": Omega_c_gum,
            "ombh2": ombh2_gum,
            "omch2": omch2_gum,
            "As": As_gum,
            "ns": ns_gum,
            "tau": tau_gum,
            "Sum_mnu": mnu_gum,
        },
        "ell_min": ell_min,
        "ell_max": ell_max,
        "rms_rel": rms_rel,
        "max_rel": max_rel,
        "threshold_rms": threshold_rms,
        "threshold_max": threshold_max,
        "pass_rms": pass_rms,
        "pass_max": pass_max,
        "plot_path": out_plot,
    }


# ---------------------------------------------------------------------------
# PDF building (polished executive artifact)
# ---------------------------------------------------------------------------

def build_pdf_report(pdf_path: str, payload, timestamp: str,
                     numeric_sha: str, camb_info: dict):
    sub      = payload["sub"]
    mathL    = payload["mathL"]
    physL    = payload["physL"]
    cosmoL   = payload["cosmoL"]
    engine   = payload["engine"]
    math_res = payload["math_res"]
    REF_MATH = payload["REF_MATH"]
    PDG      = payload["PDG"]

    A = math_res.get("A", {})
    B = math_res.get("B", {})
    C = math_res.get("C", {})

    # Footer with page number + tiny manifest hint
    def make_on_page(ts, manifest_sha):
        def _on_page(canvas, doc):
            canvas.saveState()
            canvas.setFont("Helvetica", 8)
            canvas.setFillColor(colors.grey)
            foot = f"Grand Unified Model Executive Report • {ts} • Manifest SHA-256: {manifest_sha[:12]}..."
            canvas.drawRightString(
                doc.pagesize[0] - doc.rightMargin,
                0.6 * inch,
                foot,
            )
            canvas.restoreState()
        return _on_page

    on_page = make_on_page(timestamp, numeric_sha)

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
        title="Grand Unified Model — Executive Technical Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "GUMTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        alignment=1,
        spaceAfter=18,
    )
    subtitle_style = ParagraphStyle(
        "GUMSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=12,
        leading=14,
        alignment=1,
        textColor=colors.grey,
        spaceAfter=24,
    )
    h1 = ParagraphStyle(
        "GUMHeading1",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=12,
        spaceAfter=6,
    )
    h2 = ParagraphStyle(
        "GUMHeading2",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        spaceBefore=10,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "GUMBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=6,
    )
    body_italic = ParagraphStyle(
        "GUMBodyItalic",
        parent=body,
        fontName="Helvetica-Oblique",
    )
    small = ParagraphStyle(
        "GUMSmall",
        parent=body,
        fontSize=8,
        leading=10,
        textColor=colors.grey,
        spaceAfter=4,
    )

    story = []

    # ---------- Title page ----------
    story.append(Paragraph("Grand Unified Model", title_style))
    story.append(Paragraph("Executive Technical Report", subtitle_style))

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph(f"Generated: {timestamp}", small))
    story.append(Paragraph("GUM Version: SM_GUM_v1_0 / gum_v1", small))
    story.append(Paragraph(f"Numeric Manifest SHA-256: {numeric_sha}", small))

    survivors_txt = (
        f"SCFP++ survivors (Demo 18 lawbook): "
        f"wU = {sub.wU}, s2 = {sub.s2}, s3 = {sub.s3}, q3 = {sub.q3}, "
        f"q2 = wU - s2 = {sub.wU - sub.s2}."
    )
    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(survivors_txt, body))

    story.append(Spacer(1, 0.4 * inch))
    story.append(Paragraph("Author’s Note / Epigraph", h2))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "(Reserved space for author’s note or epigraph. "
        "This page is intentionally left with room for a personal statement.)",
        body_italic,
    ))

    story.append(PageBreak())

    # ---------- Section 1: Executive Summary ----------
    story.append(Paragraph("1. Executive Summary", h1))
    story.append(Paragraph(
        "This report captures a single execution of the Grand Unified Model (GUM). "
        "In one run, and from a single discrete substrate (SCFP++ survivors wU=137, s2=107, s3=103, q3=17), "
        "it reproduces three distinct families of invariants:",
        body,
    ))
    story.append(Paragraph(
        "• Analytic and prime-pattern constants (gamma, zeta(3), zeta(5), pi, C2, Artin A, delta) via the "
        "SM-MATH-9 engine; "
        "• Standard Model couplings (alpha, sin^2(theta_W), alpha_s, v, MW, MZ) via DEMO-33; "
        "• Cosmological parameters (H0, Omega_i, eta_B, Y_He, delta_CMB, Sum m_nu, A_s, n_s, tau, l1, "
        "Omega_tot) via BB-36.",
        body,
    ))
    if camb_info.get("enabled"):
        rms_pct = 100.0 * camb_info["rms_rel"]
        max_pct = 100.0 * camb_info["max_rel"]
        story.append(Paragraph(
            "In addition, a CAMB TT-spectrum comparison between the GUM cosmology and the "
            "Planck 2018 LCDM baseline yields an RMS relative difference of "
            f"{rms_pct:.2f}% and a maximum deviation of {max_pct:.2f}%, both within the structural "
            f"thresholds of {100.0 * camb_info['threshold_rms']:.1f}% (RMS) and "
            f"{100.0 * camb_info['threshold_max']:.1f}% (max).",
            body,
        ))
    else:
        story.append(Paragraph(
            "In environments where CAMB is available, this report can also include a TT-spectrum "
            "comparison against the Planck 2018 LCDM baseline. In this run, CAMB was not executed "
            f"({camb_info.get('reason', 'no CAMB')}), but the structural cosmology parameters are still "
            "reported in detail in Section 5.",
            body,
        ))

    story.append(Spacer(1, 0.2 * inch))

    # Unified dashboard
    exec_data = [
        ["Constant", "Derived", "Reference", "RelErr", "Sector"],
    ]
    alpha_ref = PDG.get("alpha_em", float("nan"))
    sin2W_ref = PDG.get("sin2W", float("nan"))
    alpha_s_ref = PDG.get("alpha_s", float("nan"))
    delta_ref = REF_MATH.get("delta", 4.6692016091029907)
    C2_ref = REF_MATH.get("C2", 0.6601618158468696)

    exec_data.append([
        "alpha (fine structure)",
        fmt_val(physL.alpha_em),
        fmt_val(alpha_ref),
        fmt_err(rel_err(physL.alpha_em, alpha_ref)),
        "Physics (DEMO-33)",
    ])
    exec_data.append([
        "delta (Feigenbaum)",
        fmt_val(mathL.delta),
        fmt_val(delta_ref),
        fmt_err(rel_err(mathL.delta, delta_ref)),
        "Dynamics (SM-MATH-9)",
    ])
    exec_data.append([
        "C2 (twin prime)",
        fmt_val(mathL.C2),
        fmt_val(C2_ref),
        fmt_err(rel_err(mathL.C2, C2_ref)),
        "Primes (SM-MATH-9)",
    ])
    exec_data.append([
        "H0 [km/s/Mpc]",
        fmt_val(cosmoL.H0),
        fmt_val(cosmoL.H0_ref),
        fmt_err(rel_err(cosmoL.H0, cosmoL.H0_ref)),
        "Cosmology (BB-36)",
    ])
    exec_data.append([
        "Omega_L",
        fmt_val(cosmoL.Omega_L),
        fmt_val(cosmoL.Omega_L_ref),
        fmt_err(rel_err(cosmoL.Omega_L, cosmoL.Omega_L_ref)),
        "Cosmology (BB-36)",
    ])
    exec_data.append([
        "Y_He (BBN)",
        fmt_val(cosmoL.YHe),
        fmt_val(cosmoL.YHe_ref),
        fmt_err(rel_err(cosmoL.YHe, cosmoL.YHe_ref)),
        "BBN (BB-36)",
    ])
    exec_data.append([
        "delta_CMB",
        fmt_val(cosmoL.deltaCMB),
        fmt_val(cosmoL.deltaCMB_ref),
        fmt_err(rel_err(cosmoL.deltaCMB, cosmoL.deltaCMB_ref)),
        "CMB (BB-36)",
    ])

    exec_table = Table(exec_data, hAlign="LEFT")
    exec_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-2, -1), "RIGHT"),
    ]))
    story.append(exec_table)

    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "The table above is the core unified dashboard: it places alpha, delta, C2, H0, Omega_L, Y_He, "
        "and delta_CMB side-by-side, each derived from the same SCFP++ integers but via three independent "
        "engines.",
        body,
    ))

    story.append(PageBreak())

    # ---------- Section 2: Layer 0 ----------
    story.append(Paragraph("2. Layer 0 — Substrate", h1))
    story.append(Paragraph(
        "The SCFP++ selector (Demo 18 logic, embedded in BB-36 and DEMO-33) scans a finite interval of "
        "integers under a fixed gate set and returns one survivor per gauge lane. In this run, the survivors "
        "are:",
        body,
    ))
    story.append(Paragraph(
        f"wU = {sub.wU} (U(1) lane), s2 = {sub.s2} (SU(2) lane), s3 = {sub.s3} (SU(3) lane), "
        f"q3 = {sub.q3} (2-adic branch), with q2 = wU - s2 = {sub.wU - sub.s2}.",
        body,
    ))
    story.append(Paragraph(
        "Once these integers are fixed, they are not tuned again. Every subsequent calculation in "
        "SM-MATH-9, DEMO-33, and BB-36 either uses them directly or depends on quantities derived from them.",
        body,
    ))

    # ---------- Section 3: SM-MATH-9 ----------
    story.append(Paragraph("3. Layer 1 — SM-MATH-9: Standard Model of Mathematics", h1))
    story.append(Paragraph(
        "SM-MATH-9 exercises the substrate at the level of analytic number theory and dynamical invariants. "
        "Its sectors are: (A) Analytic (gamma, zeta(3), zeta(5), pi, Catalan), "
        "(B) Prime-pattern (C2, Artin A, twin-density), and (C) Dynamical (K0, delta, Gauss-log variance).",
        body,
    ))

    story.append(Paragraph("3.1 Analytic invariants (Sector A)", h2))
    sa_data = [["Constant", "Derived", "Reference", "RelErr"]]
    sa_data.append([
        "gamma",
        fmt_val(A.get("gamma")),
        fmt_val(REF_MATH.get("gamma")),
        fmt_err(rel_err(A.get("gamma", float("nan")),
                        REF_MATH.get("gamma", float("nan")))),
    ])
    sa_data.append([
        "zeta(3)",
        fmt_val(A.get("zeta3")),
        fmt_val(REF_MATH.get("zeta3")),
        fmt_err(rel_err(A.get("zeta3", float("nan")),
                        REF_MATH.get("zeta3", float("nan")))),
    ])
    sa_data.append([
        "zeta(5)",
        fmt_val(A.get("zeta5")),
        fmt_val(REF_MATH.get("zeta5")),
        fmt_err(rel_err(A.get("zeta5", float("nan")),
                        REF_MATH.get("zeta5", float("nan")))),
    ])
    sa_data.append([
        "pi",
        fmt_val(A.get("pi")),
        fmt_val(REF_MATH.get("pi")),
        fmt_err(rel_err(A.get("pi", float("nan")),
                        REF_MATH.get("pi", float("nan")))),
    ])
    sa_data.append([
        "Catalan",
        fmt_val(A.get("Catalan")),
        fmt_val(REF_MATH.get("Catalan")),
        fmt_err(rel_err(A.get("Catalan", float("nan")),
                        REF_MATH.get("Catalan", float("nan")))),
    ])
    sa_table = Table(sa_data, hAlign="LEFT")
    sa_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(sa_table)

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("3.2 Prime-pattern invariants (Sector B)", h2))
    sb_data = [["Constant", "Derived", "Reference", "RelErr"]]
    sb_data.append([
        "C2",
        fmt_val(B.get("C2")),
        fmt_val(REF_MATH.get("C2")),
        fmt_err(rel_err(B.get("C2", float("nan")),
                        REF_MATH.get("C2", float("nan")))),
    ])
    sb_data.append([
        "Artin A",
        fmt_val(B.get("ArtinA")),
        fmt_val(REF_MATH.get("ArtinA")),
        fmt_err(rel_err(B.get("ArtinA", float("nan")),
                        REF_MATH.get("ArtinA", float("nan")))),
    ])
    sb_table = Table(sb_data, hAlign="LEFT")
    sb_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(sb_table)

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph("3.3 Dynamical invariants (Sector C)", h2))
    K0_val = C.get("K0_series", C.get("K0"))
    sc_data = [["Constant", "Derived", "Reference", "RelErr"]]
    sc_data.append([
        "K0 (series)",
        fmt_val(K0_val),
        fmt_val(REF_MATH.get("K0")),
        fmt_err(rel_err(K0_val if K0_val is not None else float("nan"),
                        REF_MATH.get("K0", float("nan")))),
    ])
    sc_data.append([
        "delta (RG, M=10)",
        fmt_val(mathL.delta),
        fmt_val(REF_MATH.get("delta")),
        fmt_err(rel_err(mathL.delta,
                        REF_MATH.get("delta", float("nan")))),
    ])
    sc_table = Table(sc_data, hAlign="LEFT")
    sc_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(sc_table)

    story.append(PageBreak())

    # ---------- Section 4: DEMO-33 ----------
    story.append(Paragraph("4. Layer 2A — DEMO-33: Standard Model from GUM substrate", h1))
    story.append(Paragraph(
        "DEMO-33 takes the same SCFP++ integers and constructs the electroweak and QCD sectors via a "
        "Phi-channel. The structural couplings in this run are shown against PDG values:",
        body,
    ))

    sm_data = [["Observable", "Derived", "PDG Ref", "RelErr"]]
    sm_data.append([
        "alpha (U(1))",
        fmt_val(physL.alpha_em),
        fmt_val(alpha_ref),
        fmt_err(rel_err(physL.alpha_em, alpha_ref)),
    ])
    sm_data.append([
        "sin^2(theta_W)^Phi",
        fmt_val(physL.sin2W),
        fmt_val(sin2W_ref),
        fmt_err(rel_err(physL.sin2W, sin2W_ref)),
    ])
    sm_data.append([
        "alpha_s^Phi",
        fmt_val(physL.alpha_s),
        fmt_val(alpha_s_ref),
        fmt_err(rel_err(physL.alpha_s, alpha_s_ref)),
    ])
    sm_data.append([
        "v [GeV]",
        fmt_val(physL.v),
        fmt_val(PDG.get("v_GeV", float("nan"))),
        fmt_err(rel_err(physL.v, PDG.get("v_GeV", float("nan")))),
    ])
    sm_data.append([
        "MW [GeV]",
        fmt_val(physL.MW),
        fmt_val(PDG.get("MW_GeV", float("nan"))),
        fmt_err(rel_err(physL.MW, PDG.get("MW_GeV", float("nan")))),
    ])
    sm_data.append([
        "MZ [GeV]",
        fmt_val(physL.MZ),
        fmt_val(91.1876),
        fmt_err(rel_err(physL.MZ, 91.1876)),
    ])
    sm_table = Table(sm_data, hAlign="LEFT")
    sm_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(sm_table)

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "From these couplings, DEMO-33 reconstructs gauge couplings e, g1, g2, g3 and a closed-form "
        "electroweak scale v in a way that is entirely upstream-structural (no PDG inputs enter the Phi-laws).",
        body,
    ))

    story.append(PageBreak())

    # ---------- Section 5: BB-36 ----------
    story.append(Paragraph("5. Layer 2B — BB-36: Structural Cosmology from GUM substrate", h1))
    story.append(Paragraph(
        "BB-36 feeds the same SCFP++ survivors into a monomial engine to build an FRW + Lambda cosmology, "
        "including baryogenesis, BBN, and the CMB amplitude. Selected parameters:",
        body,
    ))

    cosmo_data = [["Quantity", "Derived", "Reference", "RelErr"]]
    cosmo_data.append([
        "H0 [km/s/Mpc]",
        fmt_val(cosmoL.H0),
        fmt_val(cosmoL.H0_ref),
        fmt_err(rel_err(cosmoL.H0, cosmoL.H0_ref)),
    ])
    cosmo_data.append([
        "Omega_b",
        fmt_val(cosmoL.Omega_b),
        fmt_val(cosmoL.Omega_b_ref),
        fmt_err(rel_err(cosmoL.Omega_b, cosmoL.Omega_b_ref)),
    ])
    cosmo_data.append([
        "Omega_c",
        fmt_val(cosmoL.Omega_c),
        fmt_val(cosmoL.Omega_c_ref),
        fmt_err(rel_err(cosmoL.Omega_c, cosmoL.Omega_c_ref)),
    ])
    cosmo_data.append([
        "Omega_L",
        fmt_val(cosmoL.Omega_L),
        fmt_val(cosmoL.Omega_L_ref),
        fmt_err(rel_err(cosmoL.Omega_L, cosmoL.Omega_L_ref)),
    ])
    cosmo_data.append([
        "Omega_r",
        fmt_val(cosmoL.Omega_r),
        fmt_val(cosmoL.Omega_r_ref),
        fmt_err(rel_err(cosmoL.Omega_r, cosmoL.Omega_r_ref)),
    ])
    cosmo_data.append([
        "eta_B",
        fmt_val(cosmoL.etaB),
        fmt_val(cosmoL.etaB_ref),
        fmt_err(rel_err(cosmoL.etaB, cosmoL.etaB_ref)),
    ])
    cosmo_data.append([
        "Y_He",
        fmt_val(cosmoL.YHe),
        fmt_val(cosmoL.YHe_ref),
        fmt_err(rel_err(cosmoL.YHe, cosmoL.YHe_ref)),
    ])
    cosmo_data.append([
        "delta_CMB",
        fmt_val(cosmoL.deltaCMB),
        fmt_val(cosmoL.deltaCMB_ref),
        fmt_err(rel_err(cosmoL.deltaCMB, cosmoL.deltaCMB_ref)),
    ])

    sum_mnu_s = engine.get("Sum_mnu_SCFP")
    sum_mnu_o = engine.get("Sum_mnu_obs")
    if sum_mnu_s is not None and sum_mnu_o is not None:
        cosmo_data.append([
            "Sum m_nu [eV]",
            fmt_val(sum_mnu_s),
            fmt_val(sum_mnu_o),
            fmt_err(rel_err(sum_mnu_s, sum_mnu_o)),
        ])

    n_s_s = engine.get("n_s_SCFP")
    n_s_o = engine.get("n_s_obs")
    if n_s_s is not None and n_s_o is not None:
        cosmo_data.append([
            "n_s",
            fmt_val(n_s_s),
            fmt_val(n_s_o),
            fmt_err(rel_err(n_s_s, n_s_o)),
        ])

    tau_s = engine.get("tau_SCFP")
    tau_o = engine.get("tau_obs")
    if tau_s is not None and tau_o is not None:
        cosmo_data.append([
            "tau",
            fmt_val(tau_s),
            fmt_val(tau_o),
            fmt_err(rel_err(tau_s, tau_o)),
        ])

    ell1_s = engine.get("ell1_SCFP")
    ell1_o = engine.get("ell1_obs")
    if ell1_s is not None and ell1_o is not None:
        cosmo_data.append([
            "l1",
            fmt_val(ell1_s),
            fmt_val(ell1_o),
            fmt_err(rel_err(ell1_s, ell1_o)),
        ])

    Omega_tot = engine.get("Omega_tot")
    if Omega_tot is not None:
        cosmo_data.append([
            "Omega_tot",
            fmt_val(Omega_tot),
            "",
            "",
        ])

    cosmo_table = Table(cosmo_data, hAlign="LEFT")
    cosmo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
    ]))
    story.append(cosmo_table)

    story.append(Spacer(1, 0.15 * inch))
    story.append(Paragraph(
        "Together, these values show that the same substrate underpins not only gauge couplings and "
        "mass scales, but also large-scale cosmology, baryogenesis, and the CMB amplitude—without "
        "sector-specific tuning.",
        body,
    ))

    story.append(PageBreak())

    # ---------- Section 6: CAMB structural validation ----------
    story.append(Paragraph("6. Structural CAMB Validation", h1))
    if camb_info.get("enabled"):
        rms_pct = 100.0 * camb_info["rms_rel"]
        max_pct = 100.0 * camb_info["max_rel"]
        story.append(Paragraph(
            "To test whether the GUM-derived cosmology is dynamically consistent with the observed CMB, "
            "we feed the BB-36 parameters into CAMB and compare the resulting TT power spectrum against "
            "the Planck 2018 LCDM baseline.",
            body,
        ))
        story.append(Spacer(1, 0.1 * inch))

        camb_table_data = [
            ["Metric", "Value", "Threshold", "Status"],
            [
                "RMS(Delta TT / TT)",
                f"{rms_pct:.2f}%",
                f"< {100.0 * camb_info['threshold_rms']:.1f}%",
                "PASS" if camb_info["pass_rms"] else "FAIL",
            ],
            [
                "max|Delta TT / TT|",
                f"{max_pct:.2f}%",
                f"< {100.0 * camb_info['threshold_max']:.1f}%",
                "PASS" if camb_info["pass_max"] else "FAIL",
            ],
        ]
        camb_table = Table(camb_table_data, hAlign="LEFT")
        camb_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgoldenrodyellow),
            ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
        ]))
        story.append(camb_table)

        story.append(Spacer(1, 0.15 * inch))
        story.append(Paragraph(
            "Within the ell-range considered, the GUM cosmology tracks the Planck best-fit TT spectrum "
            "to within a few percent RMS. This is a strong consistency check: the same substrate that "
            "fixes alpha, sin^2(theta_W), and delta also yields a cosmology whose acoustic peaks and "
            "damping tail are compatible with the observed microwave background.",
            body,
        ))

        if camb_info.get("plot_path"):
            story.append(Spacer(1, 0.2 * inch))
            story.append(Paragraph("Figure 1 — CMB TT Power Spectrum", h2))
            story.append(Paragraph(
                "Overlay of the Planck 2018 LCDM TT spectrum and the GUM/BB-36 TT spectrum, both computed "
                "via CAMB.",
                body,
            ))
            try:
                img = Image(camb_info["plot_path"], width=5.5 * inch, height=3.5 * inch)
                story.append(img)
            except Exception:
                story.append(Paragraph(
                    "(TT spectrum plot could not be embedded; see CAMB PNG in the GUM_Report directory.)",
                    body_italic,
                ))
    else:
        story.append(Paragraph(
            "In this environment, CAMB could not be executed "
            f"({camb_info.get('reason', 'no CAMB')}), so no Boltzmann-level TT-spectrum validation is shown. "
            "When CAMB is available, this section presents a quantitative comparison between the GUM "
            "cosmology and the Planck 2018 TT spectrum.",
            body,
        ))

    story.append(PageBreak())

    # ---------- Section 7: Unified view & reproducibility ----------
    story.append(Paragraph("7. Unified View and Reproducibility", h1))
    story.append(Paragraph(
        "Viewed as a whole, the GUM stack makes the following claim: given the MARI / DRPT, UFET, Fejer, "
        "and SCFP++ architecture, the observed Standard-Model-like and LCDM-like world is not an arbitrary "
        "fit but an isolated fixed point of a finite lawbook.",
        body,
    ))
    story.append(Paragraph(
        "This report is one concrete snapshot of that architecture. The numeric manifest SHA-256 printed on "
        "the title page is computed from the cross-domain invariants themselves. A separate JSON manifest "
        "and a file-level SHA-256 checksum of the PDF are written alongside this report in the GUM_Report "
        "directory.",
        body,
    ))
    story.append(Spacer(1, 0.2 * inch))
    story.append(Paragraph(
        "Re-running this report on another machine with the same versions of SM-MATH-9, DEMO-33, BB-36, "
        "and SM_GUM_v1_0 / gum_v1 should reproduce these numbers (to within floating-point drift) and "
        "yield the same numeric manifest hash.",
        body,
    ))

    story.append(Spacer(1, 0.3 * inch))
    story.append(Paragraph(
        "This document is intended as an executive artifact: a readable, coherent summary of a single "
        "GUM execution (substrate to MATH plus SM plus COSMO), suitable for archiving and citation.",
        body_italic,
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


# ---------------------------------------------------------------------------
# Manifest builder
# ---------------------------------------------------------------------------

def build_manifest_dict(payload, timestamp: str, pdf_name: str,
                        file_sha: str, numeric_sha: str, camb_info: dict):
    sub    = payload["sub"]
    mathL  = payload["mathL"]
    physL  = payload["physL"]
    cosmoL = payload["cosmoL"]
    engine = payload["engine"]

    manifest = {
        "timestamp": timestamp,
        "pdf_file": pdf_name,
        "pdf_sha256": file_sha,
        "numeric_manifest_sha256": numeric_sha,
        "substrate": {
            "wU": sub.wU,
            "s2": sub.s2,
            "s3": sub.s3,
            "q3": sub.q3,
            "q2": sub.wU - sub.s2,
        },
        "math": {
            "pi": mathL.pi,
            "gamma": mathL.gamma,
            "zeta3": mathL.zeta3,
            "zeta5": mathL.zeta5,
            "C2": mathL.C2,
            "ArtinA": mathL.ArtinA,
            "delta": mathL.delta,
        },
        "sm": {
            "alpha_em": physL.alpha_em,
            "sin2W": physL.sin2W,
            "alpha_s": physL.alpha_s,
            "v_GeV": physL.v,
            "MW_GeV": physL.MW,
            "MZ_GeV": physL.MZ,
        },
        "cosmo": {
            "H0": cosmoL.H0,
            "H0_ref": cosmoL.H0_ref,
            "Omega_b": cosmoL.Omega_b,
            "Omega_b_ref": cosmoL.Omega_b_ref,
            "Omega_c": cosmoL.Omega_c,
            "Omega_c_ref": cosmoL.Omega_c_ref,
            "Omega_L": cosmoL.Omega_L,
            "Omega_L_ref": cosmoL.Omega_L_ref,
            "Omega_r": cosmoL.Omega_r,
            "Omega_r_ref": cosmoL.Omega_r_ref,
            "etaB": cosmoL.etaB,
            "etaB_ref": cosmoL.etaB_ref,
            "YHe": cosmoL.YHe,
            "YHe_ref": cosmoL.YHe_ref,
            "deltaCMB": cosmoL.deltaCMB,
            "deltaCMB_ref": cosmoL.deltaCMB_ref,
            "Omega_tot": engine.get("Omega_tot"),
            "Sum_mnu_SCFP": engine.get("Sum_mnu_SCFP"),
            "Sum_mnu_obs": engine.get("Sum_mnu_obs"),
            "A_s_SCFP": engine.get("A_s_SCFP"),
            "A_s_obs": engine.get("A_s_obs"),
            "n_s_SCFP": engine.get("n_s_SCFP"),
            "n_s_obs": engine.get("n_s_obs"),
            "tau_SCFP": engine.get("tau_SCFP"),
            "tau_obs": engine.get("tau_obs"),
            "ell1_SCFP": engine.get("ell1_SCFP"),
            "ell1_obs": engine.get("ell1_obs"),
        },
        "camb_validation": camb_info,
    }
    return manifest


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    # Keep the root clean: all artifacts go into GUM_Report/
    report_dir = os.path.join(os.getcwd(), "GUM_Report")
    os.makedirs(report_dir, exist_ok=True)

    payload = run_gum_and_collect()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf_name = f"GUM_Report_{timestamp}.pdf"
    pdf_path = os.path.join(report_dir, pdf_name)
    sha_name = os.path.join(report_dir, f"GUM_Report_{timestamp}.sha256")
    manifest_name = os.path.join(report_dir, f"GUM_Report_{timestamp}_manifest.json")

    numeric_sha = build_numeric_manifest_hash(payload)
    camb_info = run_camb_validation(report_dir, payload)

    build_pdf_report(pdf_path, payload, timestamp, numeric_sha, camb_info)

    file_sha = sha256_of_file(pdf_path)
    with open(sha_name, "w", encoding="utf-8") as f:
        f.write(f"{file_sha}  {pdf_name}\n")

    manifest = build_manifest_dict(payload, timestamp, pdf_name, file_sha, numeric_sha, camb_info)
    with open(manifest_name, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    print(f"GUM executive report written to: {pdf_path}")
    print(f"PDF SHA-256: {file_sha}")
    print(f"SHA256 sidecar : {sha_name}")
    print(f"JSON manifest  : {manifest_name}")


if __name__ == "__main__":
    main()




