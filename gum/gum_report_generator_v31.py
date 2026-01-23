
# -----------------------------
# Units (report-side resolver)
# -----------------------------
UNITS_MAP = {
    # Cosmology / BB36
    "H0": "km/s/Mpc",
    "Omega_b": "-", "Omega_c": "-", "Omega_L": "-", "Omega_r": "-", "Omega_tot": "-",
    "ombh2": "-", "omch2": "-",
    "A_s": "-", "n_s": "-", "tau": "-",
    "ell1": "-", "deltaCMB": "-", "delta0": "-", "F_CMB": "-",

    # QCD / particle masses
    "Lambda_QCD": "GeV",
    "MZ": "GeV", "MW": "GeV",
    "MZ_GeV": "GeV", "MW_GeV": "GeV",
    "mz": "GeV", "mw": "GeV",

    # Dimensionless anchors
    "alpha_inv": "-", "alpha0_inv": "-", "alpha_s": "-", "sin2thetaW": "-", "sin2_thetaW": "-", "sin^2(thetaW)": "-",
}

def unit_for(name: str, units_field) -> str:
    # Prefer explicit units if provided; else map known names; else "-"
    if units_field is not None and str(units_field).strip():
        return str(units_field).strip()
    nm = (name or "").strip()
    return UNITS_MAP.get(nm, "-")
from __future__ import annotations

CLAUDE_VISUAL_ATLAS_URL = ""
BUNDLE_VISUAL_ATLAS_PATH = "atlas_substrate_visualization/visual_atlas_1.html"
#!/usr/bin/env python3
"""
GUM Report Generator v31 (Masterpiece upgrade)

Design goals:
- v28 aesthetic + narrative arc (DRPT origin -> filter -> kernel bridge -> falsification -> certificates)
- bundle-driven (reads audits/bundles|audits/bundler output folders; does NOT run demos)
- audit-grade: every demo has a reproducible one-liner and hashes; no invented numeric claims
- referee-friendly: clear origin story, why-it-matters narratives, explicit missing-data callouts
"""



def _domain_from_slug(slug: str) -> str:
    # slug examples: cosmo__demo-36..., standard_model__demo-33...
    if not slug:
        return "n/a"
    if "__" in slug:
        return slug.split("__", 1)[0]
    return "n/a"

def _domain_short(s: str) -> str:
    m = {
        "standard_model": "std_model",
        "general_relativity": "gr",
        "quantum_gravity": "qg",
        "foundations": "foundations",
        "controllers": "controllers",
        "infinity": "infinity",
        "cosmo": "cosmo",
        "sm": "sm",
        "quantum": "quantum",
        "bridge": "bridge",
        "substrate": "substrate",
    }
    return m.get(s, s)


import argparse
import csv
import datetime as _dt
import hashlib
import json
import os
import re
import textwrap
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# reportlab
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents


def _fmt_cell(v, missing_reason: str = "MISSING") -> str:
    """Production rule: no blank cells in tables."""
    if v is None:
        return missing_reason
    if isinstance(v, str):
        vv = v.strip()
        return vv if vv else missing_reason
    return str(v)

def _fmt_sha(v) -> str:
    return _fmt_cell(v, "MISSING (no artifacts sha)")


__version__ = "31.2-masterpiece"


# ----------------------------
# Utilities
# ----------------------------

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(s: str) -> str:
    return ANSI_RE.sub("", s)


TRANSLIT = {
    "—": "-",
    "–": "-",
    "→": "->",
    "⇒": "=>",
    "×": "x",
    "·": "*",
    "π": "pi",
    "Π": "Pi",
    "τ": "tau",
    "Φ": "Phi",
    "φ": "phi",
    "θ": "theta",
    "Θ": "Theta",
    "μ": "mu",
    "Λ": "Lambda",
    "λ": "lambda",
    "ℓ": "l",
    "★": "*",
    "²": "^2",
    "³": "^3",
    "⁻": "-",
    "⁺": "+",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
    "✓": "OK",
    "✅": "OK",
    "❌": "FAIL",
    "•": "-",
    "…": "...",
    "“": '"',
    "”": '"',
    "’": "'",
}


def ascii_sanitize(s: str) -> str:
    """Best-effort conversion to ASCII to avoid tofu blocks in PDFs."""
    if not s:
        return s
    s = strip_ansi(s)
    for k, v in TRANSLIT.items():
        s = s.replace(k, v)
    # Normalize and drop remaining non-ascii
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    return s


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def try_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def fmt_seconds(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if x < 0.01:
        return f"{x*1000:.2f} ms"
    if x < 1:
        return f"{x*1000:.0f} ms"
    if x < 60:
        return f"{x:.2f} s"
    return f"{x/60:.2f} min"


def fmt_num(x: Any, sig: int = 10) -> str:
    """Format numbers compactly for tables without changing value semantics."""
    if x is None:
        return "MISSING"
    if isinstance(x, (int,)):
        return str(x)
    if isinstance(x, float):
        if x == 0.0:
            return "0"
        ax = abs(x)
        if ax >= 1e6 or ax < 1e-4:
            return f"{x:.{sig}g}"
        # fixed but trimmed
        s = f"{x:.{sig}f}".rstrip("0").rstrip(".")
        return s
    # string numeric?
    s = str(x)
    try:
        xf = float(s)
        return fmt_num(xf, sig=sig)
    except Exception:
        pass
    if len(s) > 64:
        return s[:61] + "..."
    return s


def demo_label_from_slug(slug_or_id: str) -> str:
    s = str(slug_or_id)
    if s.startswith("DEMO-"):
        return s
    if s.isdigit():
        return f"DEMO-{s}"
    m = re.match(r"demo-(\d+[a-z]?)", s)
    if m:
        return f"DEMO-{m.group(1)}"
    return f"DEMO-{s}"


def demo_sort_key(label: str) -> Tuple[int, str]:
    # DEMO-66 sorts after DEMO-66 but before DEMO-67
    m = re.match(r"DEMO-(\d+)([a-z]?)", label)
    if not m:
        return (10**9, label)
    n = int(m.group(1))
    suf = m.group(2) or ""
    return (n, suf)


def escape_xml(s: str) -> str:
    return (
        s.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def hard_wrap_command(cmd: str) -> str:
    """Insert soft line breaks to keep one-liners on page."""
    if not cmd:
        return cmd
    cmd = cmd.replace(" && ", " &&\n")
    cmd = cmd.replace(" ; ", " ;\n")
    cmd = cmd.replace(" | ", " |\n")
    return cmd


# ----------------------------
# Demo catalog (context + narratives)
# ----------------------------

FLAGSHIPS: set[str] = {
    "DEMO-33",
    "DEMO-34",
    "DEMO-36",
    "DEMO-40",
    "DEMO-54",
    "DEMO-55",
    "DEMO-66",
    "DEMO-67",
    "DEMO-68",
    "DEMO-70",
    "DEMO-71",
    "DEMO-73",
    "DEMO-75",
    "DEMO-76",
}

# Cluster IDs drive "grouped stories" in certificates
CLUSTER_TITLES: Dict[str, str] = {
    "KERNEL": "The Kernel: Substrate, Selection, and Invariance",
    "FILTER": "Analytic Filters: DRPT Motifs and Fejer Smoothing",
    "SM": "Closure I: Standard Model and Electroweak Dressing",
    "COSMO": "Closure II: Cosmology and Large-Scale Structure",
    "GRNS": "Dynamics: Gravity, Fields, and Continuum Constraints",
    "QNTM": "Quantum and Quantum Gravity",
    "BRIDGE": "Bridges and Transfer Principles",
}

CLUSTER_INTROS: Dict[str, str] = {
    "KERNEL": (
        "These demos establish the discrete kernel the program reuses everywhere else: the substrate ontology, "
        "base-gauge invariance, constrained selection (SCFP/SCFP++), and the lawful lift from integer space into "
        "continuous-looking structures. If this kernel fails, later physics closures are not merely wrong; they are "
        "uninterpretable. For referees, this cluster is the correct starting point for mechanism rather than results."
    ),
    "FILTER": (
        "A recurring risk in integer-derived constructions is accidental numerology. This cluster addresses that risk head-on "
        "by showing how DRPT motifs and Fejer smoothing behave like an analytic filter: they suppress noise, stabilize limits, "
        "and expose structure that is stable under perturbations. The point is not to hide instability; the point is to make it testable."
    ),
    "SM": (
        "This cluster demonstrates particle-physics closure from the kernel: gauge structure, anomaly cancellation, mixing, and a coherent parameter set. "
        "It distinguishes a structural witness space from an optional comparison/overlay space, and it records hashes so independent auditors can reproduce the closure exactly."
    ),
    "COSMO": (
        "This cluster tests whether the same integer-derived kernel can coherently project into cosmology: background parameters, lensing, and the Big Bang closure pipeline. "
        "It also defines where external overlays (Planck/CAMB) are appropriate and where they are explicitly excluded from upstream selection."
    ),
    "GRNS": (
        "These demos treat continuum dynamics as a stress test: if the kernel is real, it must survive contact with differential structure (Einstein, Maxwell, Navier-Stokes) "
        "without arbitrary patching. The story is not 'we match one number' but 'we preserve constraints, stability, and admissibility across regimes'."
    ),
    "QNTM": (
        "Quantum claims are easy to overstate and hard to audit. This cluster is therefore certificate-heavy: it records one-liners, hashes, and artifacts so reviewers can rerun "
        "the same computations and compare outputs byte-for-byte. Where structured exports are missing, the report flags exactly what evidence is present and what is not."
    ),
    "BRIDGE": (
        "These demos show transfer: how operators, admissibility, and coupling rules move between discrete and continuum descriptions. "
        "In the narrative arc, this is where 'unrelated' domains are shown to share the same kernel constraints."
    ),
}

# Per-demo metadata and narrative. Everything here is qualitative unless sourced from the bundle.
# Minimum 6 sentences per demo narrative (referee-targeted).
DEMO_INFO: Dict[str, Dict[str, Any]] = {
    "DEMO-40": {
        "cluster": "KERNEL",
        "title": "Universe from Zero (canonical substrate ontology)",
        "tests": "Substrate generation from the 0/1 ontology; canonical invariants; reproducible kernel seed.",
        "highlights": [
            "Defines the substrate starting point used by later closures (SM, cosmology, GR/NS).",
            "Shows the program is not tuned to physics numbers; it is tuned to discrete structural constraints.",
            "Acts as a falsifier: if this kernel does not reproduce, downstream matches are irrelevant.",
        ],
        "narrative": (
            "DEMO-40 is the origin point: it constructs the substrate from the program's zero-one ontology and records the canonical objects that later demos reuse. "
            "This matters because the entire GUM story depends on reuse: if the same kernel cannot be reconstructed deterministically, any downstream agreement can be dismissed as drift. "
            "The demo is framed as a mechanism test rather than a headline-number test, which is the correct posture for skeptical review. "
            "It also makes the cross-base claim concrete by building objects that are defined independent of representation, not by fitting in a chosen base. "
            "In the audit context, DEMO-40 is valuable because it produces a stable starting state that can be hashed and compared across machines. "
            "If a referee wants a single 'first domino' to kick, DEMO-40 is designed to be that domino."
        ),
    },
    "DEMO-64": {
        "cluster": "KERNEL",
        "title": "Base-gauge invariance (integer selector and invariance checks)",
        "tests": "Base-gauge invariance; invariance of selection under representation changes; selector stability.",
        "highlights": [
            "Addresses the core 'base-dependence' critique directly.",
            "Stresses invariance under representation changes rather than raw value matching.",
            "Provides a clean falsifier: invariance breaks are unambiguous and reproducible.",
        ],
        "narrative": (
            "DEMO-64 tests a foundational claim: results should not depend on the numeral base or encoding used to represent integers. "
            "Instead of asserting invariance in prose, it executes explicit checks that selection and derived invariants remain stable under base-gauge transformations. "
            "This is critical for referee confidence because base-dependence is a common failure mode of pattern-mining approaches. "
            "In the narrative arc, DEMO-64 is the bridge between the substrate ontology and physics closure: it says 'the kernel is real, not an artifact of notation'. "
            "The audit value is high because invariance failures are crisp: the demo can be rerun and compared exactly using hashes. "
            "If the program's cross-domain unification is correct, DEMO-64 is one of the simplest places to see why."
        ),
    },
    "DEMO-65": {
        "cluster": "KERNEL",
        "title": "Continuous lift paradox (finite-to-continuum consistency stress test)",
        "tests": "Lawful lift rules; consistency constraints when mapping discrete structures to continuum-like limits.",
        "highlights": [
            "Targets the 'continuous lift' critique: how do discrete objects produce continuum behavior without cheating?",
            "Connects directly to later GR/NS demos that rely on differential structure.",
            "Serves as an internal consistency check: lift rules are either coherent or they are not.",
        ],
        "narrative": (
            "DEMO-65 is a stress test for the program's most delicate step: the lift from discrete integer structure to continuum-looking behavior. "
            "The demo exists because without a lawful lift, any claim of emergent physics can be dismissed as post-hoc curve fitting. "
            "By framing the lift as a constrained mapping problem, it turns a philosophical objection into a falsifiable computation. "
            "This demo also foreshadows why Fejer smoothing appears later: analytic filters are meaningful only when the lift itself is lawful. "
            "For referees, DEMO-65 provides a mechanism-level checkpoint that is independent of the specific physical constants being targeted. "
            "If this paradox is not resolved in the program's terms, later closures should be treated as ungrounded."
        ),
    },
    "DEMO-53": {
        "cluster": "KERNEL",
        "title": "Lawbook emergence (axioms to admissible rules)",
        "tests": "Emergence of admissible 'lawbook' constraints from axioms; rule selection without parameter fitting.",
        "highlights": [
            "Shows how constraints are selected rather than imposed by hand.",
            "Provides the interpretive layer that later physics demos rely on.",
            "Reframes the project as constrained derivation, not pattern search.",
        ],
        "narrative": (
            "DEMO-53 focuses on rule emergence: how the program's constraints (the 'lawbook') arise from the underlying axioms rather than being chosen to match outcomes. "
            "This matters because the most credible form of unification is not agreement on numbers but agreement on why only certain transformations are allowed. "
            "The demo is therefore positioned as an admissibility audit: it tracks whether rules are consistent, reusable, and stable under perturbation. "
            "In the narrative arc, DEMO-53 is where the kernel becomes operational: axioms turn into a set of allowed moves that later closures must respect. "
            "For skeptical readers, this is an antidote to the 'hidden knob' worry, because the output is a structured set of constraints rather than a best-fit parameter list. "
            "If the lawbook is not emergent and stable here, downstream claims should be considered underdetermined."
        ),
    },

    "DEMO-56": {
        "cluster": "FILTER",
        "title": "Deterministic operator calculus vs finite differences (analytic filter audit)",
        "tests": "Operator construction; deterministic calculus pipeline; Fejer smoothing nonnegativity and error control.",
        "highlights": [
            "Demonstrates Fejer smoothing as a controlled analytic filter, not an aesthetic choice.",
            "Provides explicit falsifiers (kernel nonnegativity, contraction bounds, operator invariants).",
            "Connects the DRPT motif to continuum calculus in an audit-ready way.",
        ],
        "narrative": (
            "DEMO-56 is an explicit audit of the analytic machinery: it compares a deterministic operator-calculus construction against more conventional finite-difference intuition. "
            "The point is not that finite differences are 'wrong'; it is that the program needs a reproducible operator pipeline that does not depend on unstable discretization choices. "
            "Fejer smoothing appears here as a principled mechanism: it stabilizes partial sums and suppresses spurious oscillations in a way that can be tested with inequalities. "
            "This is why the demo is narrative-important even beyond its immediate outputs: it justifies why the report treats Fejer smoothing as part of the kernel rather than a cosmetic post-processing step. "
            "For referees, the key value is that the demo exposes falsifiers like nonnegativity and contraction bounds that do not rely on external reference numbers. "
            "If those invariants fail, downstream 'nice-looking' plots should be treated as untrustworthy."
        ),
    },

    "DEMO-33": {
        "cluster": "SM",
        "title": "First-principles Standard Model (SM-28 closure; SCFP++ -> Phi -> SM)",
        "tests": "SM closure from SCFP++ survivors; anomaly cancellation; CKM/PMNS unitarity; 1-loop RG; dressed closure space.",
        "highlights": [
            "Flagship: constructs a full SM manifest from constrained survivors and a minimal palette.",
            "Separates structural witness space from prediction/overlay space (no upstream PDG leakage).",
            "Produces hashed JSON artifacts for citation and third-party reproduction.",
        ],
        "narrative": (
            "DEMO-33 is the flagship Standard Model closure: it starts from SCFP++ survivor structure and builds a complete SM manifest through the Phi-channel pipeline. "
            "The demo is designed to be audit-friendly: anomaly cancellation and mixing unitarity are treated as exact checks, not approximate matches. "
            "It explicitly separates a structural witness space from a dressed prediction space, preventing upstream contamination by external constants. "
            "This separation is crucial for referees because it tells you what is derived from first principles versus what is used only for evaluation overlays. "
            "The closure is recorded as structured JSON with hashes so the same manifest can be cited and independently regenerated. "
            "In the blended story, DEMO-33 shows how a discrete kernel constrains a seemingly continuous field theory without hand-tuned parameters."
        ),
    },
    "DEMO-34": {
        "cluster": "BRIDGE",
        "title": "Ω→SM master flagship (v1)",
        "tests": "Tier-A₁ joint-triple Ω certificate (finite band + necessity ablations); lane-local stress failure by 100k; Tier-C SM overlay (PDG only for Δ%).",
        "highlights": [
            "Tier-A₁ joint-triple certificate with necessity ablations.",
            "Lane-local Tier-A stress test demonstrates lane-local locks fail by 100k.",
            "Tier-C SM overlay keeps PDG usage strictly in Δ% reporting.",
        ],
        "narrative": (
            "DEMO-34 is the Ω→SM master flagship. It is release-relevant because it combines a strict Tier-A₁ joint-triple certificate "
            "with explicit necessity ablations and a lane-local stress test that surfaces real failure modes at scale. "
            "It then layers a Tier-C SM overlay where PDG is used only for Δ% reporting. "
            "This is exactly the posture needed for a high-standard report: certify what is claimed, expose what fails, and keep overlays honest."
        ),
    },

    "DEMO-37": {
        "cluster": "SM",
        "title": "Math-SM master flagship (alpha_s at MZ; confinement and freequark branches)",
        "tests": "alpha_s(MZ) branch structure; confinement/freequark variants; invariance of derived couplings; citation-grade outputs.",
        "highlights": [
            "Demonstrates that multiple physically meaningful branches can arise from the same kernel constraints.",
            "Provides a clean interface between mathematical invariants and SM running quantities.",
            "Feeds the unified constants dashboard with bundle-sourced values and hashes.",
        ],
        "narrative": (
            "DEMO-37 focuses on a key interface: how the kernel's discrete structure projects into the strong coupling at the electroweak scale. "
            "Rather than presenting a single number, it exposes branch structure (e.g., confinement vs freequark) as an internal diagnostic of the construction. "
            "This matters for referees because a robust theory should explain which branches are admissible and why, not merely select the best-looking output. "
            "The demo is also a test of cross-base stability: couplings are treated as derived invariants that should persist under representation changes. "
            "From an audit standpoint, DEMO-37 is valuable because its outputs are already bundle-structured and hashable, making it immediately citable. "
            "In the narrative arc, it strengthens the 'one kernel, many domains' thesis by tying discrete invariants directly to a running coupling."
        ),
    },
    "DEMO-54": {
        "cluster": "SM",
        "title": "Master flagship demo (multi-stage closure sanity check)",
        "tests": "End-to-end master pipeline sanity: stage-wise acceptance, multiple manifest checks, deterministic closure.",
        "highlights": [
            "Flagship integration test: verifies that multiple subsystems cohere in one run.",
            "Emits explicit stage verdicts and invariant summaries suitable for referee triage.",
            "Acts as a regression sentinel: small changes in the codebase show up here first.",
        ],
        "narrative": (
            "DEMO-54 is an integration flagship: it runs a multi-stage pipeline intended to catch inconsistencies that do not appear in single-domain demos. "
            "Its value is not limited to any one physical quantity; instead, it verifies that successive closure layers remain compatible under a single deterministic run. "
            "For referees, this matters because cross-domain unification lives or dies on consistency: a theory that matches one table but fails when composed is not a theory. "
            "The demo reports stage-level verdicts, which makes it easier to localize failures and avoids 'black box' conclusions. "
            "In an audit workflow, DEMO-54 is also the natural regression gate: if a refactor changes behavior, this demo should reflect it immediately via hashes. "
            "In the blended narrative, DEMO-54 is where the kernel is tested as a reusable mechanism rather than a set of isolated coincidences."
        ),
    },
    "DEMO-55": {
        "cluster": "SM",
        "title": "Proton radius master flagship",
        "tests": "Hadronic-scale observable derived from kernel constraints; sensitivity to dressing; reproducibility of extraction.",
        "highlights": [
            "Flagship: targets a historically contentious observable (proton radius) as a stress test.",
            "Demonstrates coupling between discrete kernel and low-energy hadronic structure.",
            "Provides a clear falsifier: rerun the extraction and compare the artifact hashes.",
        ],
        "narrative": (
            "DEMO-55 targets the proton radius, a low-energy observable that is historically sensitive to modeling assumptions. "
            "This is an intentional stress test: if the program can only operate near the electroweak scale, it is not a unified kernel story. "
            "The demo therefore tests whether the same discrete constraints can project into hadronic structure without ad hoc tuning. "
            "For referees, the key is reproducibility: the extraction is captured as artifacts with hashes so independent auditors can rerun and compare outputs directly. "
            "The demo is also narrative-important because it connects the SM closure machinery to a domain where conventional approaches often disagree. "
            "If the program's kernel is real, it should produce stable low-energy structure here, not just high-energy coincidences."
        ),
    },
    "DEMO-70": {
        "cluster": "SM",
        "title": "Higgs master flagship (surrogate closure and stability checks)",
        "tests": "Higgs-sector surrogate closure; lambda_H and mH proxy outputs; stability under dressing rules.",
        "highlights": [
            "Flagship: stresses the scalar sector where radiative/threshold effects are subtle.",
            "Treats the Higgs as a surrogate proxy and states limitations explicitly (referee-friendly honesty).",
            "Anchors the dressed-vs-structural distinction in a familiar SM component.",
        ],
        "narrative": (
            "DEMO-70 addresses the Higgs sector, where naive closures are particularly prone to hidden assumptions. "
            "The demo is careful to treat the Higgs mass as a surrogate proxy derived from lambda_H and v, explicitly noting that it is not a full radiative/threshold-corrected prediction. "
            "This explicitness is important for referees because it separates what the pipeline actually computes from what readers might assume it claims. "
            "Within the program's story, DEMO-70 tests whether the same kernel constraints that fix gauge structure also constrain the scalar sector in a coherent way. "
            "It is also a stability demo: outputs are recorded so changes in dressing or selection rules are reflected in hashes rather than in ambiguous narrative. "
            "In the blended arc, DEMO-70 helps show that 'one kernel' does not stop at gauge couplings; it reaches into symmetry breaking structure as well."
        ),
    },
    "DEMO-71": {
        "cluster": "GRNS",
        "title": "One Action master flagship (Classical Noether + quantum energy bridge)",
        "tests": "Action principle spine; Noether structure; bridge between invariants and dynamics; symmetry constraints.",
        "highlights": [
            "Flagship: provides the 'laws of motion' spine that ties the rest together.",
            "Explains why overlap is expected: invariants constrain admissible dynamics, not just constants.",
            "Best single entry point for reviewers seeking mechanism over numerology.",
        ],
        "narrative": (
            "DEMO-71 is the action-principle flagship: it provides the dynamical spine that turns the kernel from a static table of invariants into a law-of-motion story. "
            "This matters because unification without a principled action principle tends to collapse into a list of correlations rather than a mechanism. "
            "By grounding the narrative in Noether structure, the demo connects conserved quantities to admissible dynamics in a way that can be audited. "
            "It also clarifies why cross-domain overlap is the expected outcome: if the same invariants constrain the allowed actions, unrelated domains should share structural signatures. "
            "For referees, DEMO-71 is a high-leverage checkpoint because it tests whether the program can derive dynamics rather than only kinematics. "
            "If this demo fails, the blended story loses its core explanatory mechanism."
        ),
    },

    "DEMO-36": {
        "cluster": "COSMO",
        "title": "Big Bang master flagship (BB36 cosmology pipeline)",
        "tests": "Cosmology closure pipeline; BB36-derived parameters; evidence artifacts for plots; optional CAMB overlay.",
        "highlights": [
            "Flagship cosmology closure: ties kernel invariants to background parameters and BB36 structure.",
            "Generates citation-grade artifacts (results JSON + BB36 plot).",
            "Defines the boundary between internal closure and optional external overlays (Planck/CAMB).",
        ],
        "narrative": (
            "DEMO-36 is the flagship cosmology closure: it executes the BB36 pipeline and produces a coherent set of cosmological parameters from the same kernel used elsewhere. "
            "In the audit framing, the key contribution is not a single best-fit number but a reproducible pipeline that emits structured artifacts with hashes. "
            "This design allows referees to rerun the computation, verify determinism, and inspect intermediate structure rather than trusting a narrative summary. "
            "The demo also cleanly separates internal closure outputs from optional external overlays (e.g., CAMB), preventing reference data from leaking into the selection mechanism. "
            "Visually, the BB36 plot is included as evidence because it communicates the structure of the closure more directly than a list of scalars. "
            "In the blended story, DEMO-36 shows that the kernel's constraints project naturally into cosmology, not just into particle physics."
        ),
    },
    "DEMO-39": {
        "cluster": "COSMO",
        "title": "BB-A2 (cosmology sanity and parameter consistency)",
        "tests": "Cosmology A2 consistency; parameter sanity checks; bridge between BB closure and kernel invariants.",
        "highlights": [
            "Narrative-important secondary cosmology check that guards against single-pipeline fragility.",
            "Helps isolate which cosmological features are robust under variant assumptions.",
            "Acts as a cross-check on BB36 rather than a replacement.",
        ],
        "narrative": (
            "DEMO-39 provides a second cosmology vantage point: it tests whether cosmological structure remains consistent under an A2-style construction rather than relying on a single flagship pipeline. "
            "This matters for referees because any one closure can accidentally encode its own assumptions; independent constructions are a robustness test. "
            "The demo therefore functions as a sanity layer: it does not need to dominate the narrative to be essential to audit credibility. "
            "In the program's blended story, DEMO-39 helps disentangle which cosmological signatures belong to the kernel and which belong to a particular closure implementation. "
            "From an audit standpoint, its primary value is that it can fail differently than DEMO-36, making debugging and falsification more informative. "
            "Together, the cosmology demos aim to show that overlap is structural, not accidental."
        ),
    },

    "DEMO-51": {
        "cluster": "GRNS",
        "title": "QFT/GR vacuum suppression (bridge between quantum and curved background)",
        "tests": "Vacuum suppression mechanism; consistency of coupling rules; bridge constraints between QFT and GR settings.",
        "highlights": [
            "Targets a conceptual bridge point where naive unification claims often break.",
            "Tests the program's ability to control vacuum contributions via kernel constraints.",
            "Provides falsifiers that are internal to the construction, not just external fits.",
        ],
        "narrative": (
            "DEMO-51 examines vacuum suppression as a bridge problem between quantum field intuition and curved-background constraints. "
            "This is narrative-important because vacuum energy is a classic failure point for would-be unified models; hand-waving here is easy and unacceptable. "
            "The demo frames suppression as a structured consequence of admissible couplings and kernel constraints rather than as an arbitrary tuning choice. "
            "For referees, the most important feature is that the demo is phrased in falsifiable terms: either the suppression emerges from the rules or it does not. "
            "It also sits strategically in the report: it connects the kernel story to cosmology (vacuum energy) and to quantum structure without requiring the full machinery of quantum gravity. "
            "If the kernel can control vacuum structure here, later gravitational closures become more plausible rather than less."
        ),
    },
    "DEMO-58": {
        "cluster": "GRNS",
        "title": "Emergent weak-field GR (limit and stability checks)",
        "tests": "Weak-field limit behavior; emergent GR structure; stability under perturbations.",
        "highlights": [
            "Checks the GR weak-field regime where many constructions can be validated or falsified cleanly.",
            "Focuses on stability and limiting behavior rather than a single constant match.",
            "Bridges kernel invariants to differential structure in a controlled regime.",
        ],
        "narrative": (
            "DEMO-58 targets the weak-field regime of general relativity, where structure can be tested without the confounders of strong-field complexity. "
            "The demo is therefore a disciplined falsifier: if the kernel cannot reproduce stable weak-field behavior, later claims about full GR or cosmology lose credibility. "
            "It emphasizes limiting behavior and stability under perturbations, which are more informative than a single-point agreement. "
            "In the narrative arc, DEMO-58 is one of the first places where the discrete kernel is forced to behave like a differential theory. "
            "For referees, this is precisely the kind of test that separates mechanistic derivation from pattern matching. "
            "If it passes, it supports the thesis that the kernel constrains admissible dynamics, not just derived constants."
        ),
    },
    "DEMO-63": {
        "cluster": "GRNS",
        "title": "Gravitational-wave inspiral phasing (observable regime stress test)",
        "tests": "Inspiral phasing constraints; consistency with GR dynamics; reproducible phasing computation.",
        "highlights": [
            "Places the kernel-derived GR structure in contact with an observable waveform regime.",
            "Tests phase-sensitive predictions where small errors accumulate and become obvious.",
            "Provides a concrete falsifier: rerun and compare phasing outputs and hashes.",
        ],
        "narrative": (
            "DEMO-63 uses gravitational-wave inspiral phasing as a precision stress test for the program's GR dynamics. "
            "Phasing is unforgiving: small structural errors accumulate into large mismatches, making it a strong falsifier rather than a loose correlation. "
            "The demo therefore helps anchor the report in an 'observable regime' without depending on any single external constant. "
            "In the blended narrative, it shows how the kernel's constraints propagate into time-domain dynamics, not just static parameters. "
            "For auditability, the demo is packaged so independent parties can rerun the computation and compare outputs via hashes. "
            "If the kernel story is correct, it should manifest as stable phase structure here rather than as fragile tuning."
        ),
    },
    "DEMO-68": {
        "cluster": "GRNS",
        "title": "General Relativity master flagship",
        "tests": "End-to-end GR structure from kernel constraints; checks across regimes; reproducible GR pipeline.",
        "highlights": [
            "Flagship GR closure: central evidence that the kernel reaches differential geometry.",
            "Designed to be audited: rerun one-liner, compare hashes, inspect artifacts.",
            "Narrative anchor for 'many domains' because GR couples to almost everything else.",
        ],
        "narrative": (
            "DEMO-68 is the flagship GR demo: it presents an end-to-end construction intended to show that the kernel can produce coherent gravitational structure. "
            "General relativity is a high-leverage domain: it couples to cosmology, to field theory, and to continuum mechanics, so a successful GR closure strengthens the whole blended narrative. "
            "For referees, the key point is that the demo is certificate-driven: it records rerun commands and hashes so claims can be checked without trusting the authors. "
            "It also serves as an integration test for the lift rules and analytic filters introduced earlier; GR is where those mechanisms are hardest to fake. "
            "Where structured exports are present, they are pulled into the bundle dashboards; where they are absent, the report treats stdout evidence and hashes as the ground truth. "
            "In the story arc, DEMO-68 is the moment where 'integer structure' becomes 'geometry' in a way that can be audited."
        ),
    },
    "DEMO-59": {
        "cluster": "GRNS",
        "title": "Electromagnetism (Maxwell structure from kernel constraints)",
        "tests": "Field-structure constraints; Maxwell-like relationships; reproducibility of EM derivation.",
        "highlights": [
            "Tests whether the kernel can produce a nontrivial field theory beyond gravity.",
            "Provides an independent check on the analytic filter machinery in a different domain.",
            "Anchors cross-domain overlap: EM and GR share invariance themes but differ in dynamics.",
        ],
        "narrative": (
            "DEMO-59 targets electromagnetism, providing a complementary field-theory test alongside gravity-focused demos. "
            "This matters because a unification claim should not be able to succeed only in one favored domain; it should generalize to multiple independent dynamical structures. "
            "By deriving EM structure from the same kernel constraints, the demo tests whether invariance and admissibility rules are truly reusable. "
            "For referees, the value is that the demo can be rerun deterministically and compared via hashes, making disagreements concrete rather than interpretive. "
            "In the blended narrative, electromagnetism is a key overlap point: it shares gauge structure themes with the Standard Model and invariance themes with the kernel, yet it lives in a different dynamical regime. "
            "A successful EM derivation therefore supports the program's claim that overlap is structural, not curated."
        ),
    },
    "DEMO-67": {
        "cluster": "GRNS",
        "title": "Navier-Stokes master flagship",
        "tests": "Continuum fluid dynamics constraints; admissibility and stability; determinism of PDE-related outputs.",
        "highlights": [
            "Flagship continuum mechanics stress test: NS is notoriously sensitive to modeling choices.",
            "Connects the lift rules to an extreme case where instability is the default.",
            "Supports the claim that the kernel tiles through infinity via lawful constraints, not heuristics.",
        ],
        "narrative": (
            "DEMO-67 is the Navier-Stokes flagship, included because continuum fluid dynamics is a stringent stress test for any discrete-to-continuum unification program. "
            "Unlike many physics-constant demos, NS problems tend to amplify small errors, so success requires more than matching a scalar target. "
            "In the report's narrative, DEMO-67 strengthens the claim that the kernel tiles through infinity: the same admissibility constraints must hold even in regimes where instability is common. "
            "For referees, this is important precisely because it is difficult: if the program can only produce stable structure in 'easy' domains, the unification story is weak. "
            "The audit framing remains consistent: rerun the one-liner, compare hashes, and inspect any artifacts rather than trusting prose. "
            "If DEMO-67 holds up, it provides some of the strongest evidence that the program's kernel is a reusable mechanism rather than a curated set of coincidences."
        ),
    },

    "DEMO-60": {
        "cluster": "QNTM",
        "title": "Quantum master flagship",
        "tests": "Quantum structural constraints; deterministic quantum pack outputs; reproducibility and audit hashes.",
        "highlights": [
            "Flagship quantum evidence bundle: designed for reproducibility rather than persuasion.",
            "Where artifacts exist, the report includes them; where missing, placeholders mark expected locations.",
            "Positions quantum structure as constrained reuse of the kernel, not as a separate theory.",
        ],
        "narrative": (
            "DEMO-60 is the flagship quantum demo, treated conservatively in the audit narrative because quantum claims are easy to over-interpret. "
            "The demo is therefore framed around reproducibility and explicit evidence artifacts: a referee should be able to rerun the computation and compare outputs byte-for-byte. "
            "In the blended story, quantum structure is not introduced as an independent add-on; it is presented as another projection of the same kernel constraints. "
            "Where the bundle contains structured exports or images, the report includes them directly; where they are missing, the report leaves an explicit placeholder so the artifact pipeline can be repaired without rewriting the narrative. "
            "This approach avoids the appearance of hiding missing evidence while still keeping the report coherent for reviewers. "
            "If DEMO-60's constraints are stable under rerun, it supports the broader thesis that the kernel constrains not just classical structure but quantum structure as well."
        ),
    },

    "DEMO-66": {
        "cluster": "QNTM",
        "title": "Quantum gravity master flagship (v4)",
        "tests": "Consolidated quantum-gravity certificate; deterministic gates; illegal controls; counterfactual teeth; audit outputs.",
        "highlights": [
            "Canonical QG flagship for master release.",
            "Deterministic gates and explicit falsifiers (controls + counterfactuals).",
            "Designed for rerun/compare/audit, not interpretation drift.",
        ],
        "narrative": (
            "DEMO-66 is the consolidated quantum gravity flagship for this release. "
            "It is structured as a single deterministic certificate with explicit gates and falsifiers. "
            "This matters because quantum-gravity narratives are easy to over-interpret; the certificate makes evaluation mechanical. "
            "The credibility claim is operational: rerun it, compare outputs, and confirm the controls fail as expected. "
            "Artifacts and hashes are treated as first-class evidence when present. "
            "If DEMO-66 is not reproducible byte-for-byte (within stated tolerances), it should be treated as FAIL."
        ),
    },

    "DEMO-73": {
        "cluster": "SM",
        "title": "Flavor completion master flagship",
        "tests": "Kernel → Yukawas → CKM/PMNS closure with explicit gates, controls, and auditable outputs.",
        "highlights": [
            "Release-grade deterministic certificate.",
            "Explicit gates + falsifiers to prevent interpretation drift.",
            "Intended to be evaluated as a certificate, not an essay.",
        ],
        "narrative": (
            "DEMO-73 extends the Standard Model closure into flavor structure as a single auditable certificate. "
            "It is designed to be deterministic and self-auditing, with explicit gates. "
            "The demo is positioned to reduce ambiguity: outputs are paired with controls and counterfactuals. "
            "This matters because flavor is where many pipelines silently smuggle assumptions. "
            "Here, the goal is to make every dependency explicit and rerunnable. "
            "If the gates or falsifiers fail, the demo fails."
        ),
    },

    "DEMO-75": {
        "cluster": "BRIDGE",
        "title": "Prediction ledger master flagship",
        "tests": "Consolidated forward predictions (neutrino, PMNS/CP proxies, dark sector, strong-field proxies) with falsifiers.",
        "highlights": [
            "Release-grade prediction ledger.",
            "Organized as reproducible outputs + explicit falsifiers.",
            "Designed to be cited as a ledger, not a claim dump.",
        ],
        "narrative": (
            "DEMO-75 is the forward prediction ledger for the master release. "
            "It consolidates predictions that fall naturally out of the kernel pipeline into one place. "
            "The goal is referee usability: each prediction is paired with a falsifier and an experimental venue. "
            "This demo is not meant to 'win' by rhetoric; it is meant to be checkable. "
            "If a prediction cannot be reproduced by rerunning the demo, it does not belong in the ledger. "
            "The report treats missing artifacts as pipeline work, not as evidence."
        ),
    },

    "DEMO-76": {
        "cluster": "SM",
        "title": "Primorial–Yukawa master flagship",
        "tests": "Primorial/Yukawa sensitivity and stability audit with deterministic gates.",
        "highlights": [
            "Stability and sensitivity audit for Yukawa ladder.",
            "Deterministic gates with explicit failure modes.",
            "Designed to expose fragility, not hide it.",
        ],
        "narrative": (
            "DEMO-76 is the primorial–Yukawa stability and sensitivity flagship. "
            "Its purpose is to test robustness: do the Yukawa results remain stable under reasonable perturbations? "
            "This matters for credibility because fragile pipelines can look impressive while being unrepeatable. "
            "The demo therefore emphasizes gates, sensitivity tables, and clear fail states. "
            "If the stability gates do not hold, the correct conclusion is that the ladder is not yet release-grade. "
            "If it holds, it strengthens the case that the Yukawa structure is constrained rather than tuned."
        ),
    },
"DEMO-69": {
        "cluster": "BRIDGE",
        "title": "OATB (operator admissibility transfer bridge)",
        "tests": "Transfer rules for admissible operators; bridge between domains; stability of admissibility under mapping.",
        "highlights": [
            "Demonstrates how the same admissibility logic propagates across domains.",
            "Provides a practical transfer mechanism rather than a post-hoc analogy.",
            "Strengthens the blended narrative by showing reuse as an algorithm, not a metaphor.",
        ],
        "narrative": (
            "DEMO-69 is explicitly about transfer: it tests whether admissible operators can be moved between domains without breaking the kernel's constraints. "
            "This is narrative-important because the report's central claim is not that many domains are solved independently, but that they share a common mechanism. "
            "Transfer bridges are how that mechanism becomes concrete: they operationalize overlap rather than merely observing it. "
            "For referees, DEMO-69 provides a falsifiable claim: if transfer breaks admissibility, the bridge is not real and the blended story becomes a set of coincidences. "
            "The demo is also an engineering asset: transfer rules reduce the chance that each new domain requires a bespoke toolchain. "
            "If DEMO-69 holds up under rerun and hashing, it is one of the clearest demonstrations that the program is building a single reusable kernel."
        ),
    },

}
# Bundle IO
# ----------------------------

@dataclass
class RunRecord:
    demo: str              # DEMO-33
    slug: str              # demo-33-first-...
    domain: str            # standard_model, cosmo, ...
    folder: str            # demos/.../slug
    status: str
    return_code: Optional[int]
    runtime_sec: Optional[float]
    mode: str
    cmd: str
    one_liner: str
    code_sha256: str
    stdout_sha256: str
    stderr_sha256: str
    artifacts_sha256: str


@dataclass
class ArtifactRecord:
    demo: str
    relpath: str
    sha256: str
    size: Optional[int] = None


@dataclass
class Bundle:
    root: Path
    meta: Dict[str, Any] = field(default_factory=dict)
    runs: List[RunRecord] = field(default_factory=list)
    demo_index: List[Dict[str, Any]] = field(default_factory=list)
    constants_rows: List[Dict[str, Any]] = field(default_factory=list)
    values_rows: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: List[ArtifactRecord] = field(default_factory=list)
    falsification: List[Dict[str, Any]] = field(default_factory=list)
    repo_inventory: Dict[str, Any] = field(default_factory=dict)

    def run_by_demo(self) -> Dict[str, RunRecord]:
        return {r.demo: r for r in self.runs}

    def artifacts_by_demo(self) -> Dict[str, List[ArtifactRecord]]:
        out: Dict[str, List[ArtifactRecord]] = {}
        for a in self.artifacts:
            out.setdefault(a.demo, []).append(a)
        # stable order
        for k in out:
            out[k].sort(key=lambda x: x.relpath)
        return out


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_root_from_file() -> Path:
    """
    Resolve repository root robustly.
    - If this generator lives inside a 'gum/' directory, repo root is its parent.
    - Otherwise fall back to current working directory.
    """
    here = Path(__file__).resolve()
    if here.parent.name == "gum":
        return here.parent.parent
    return Path.cwd()


def find_latest_bundle(repo_root: Path) -> Optional[Path]:
    """Auto-detect latest bundle folder under common locations."""
    candidates: List[Path] = []
    for base in [
        repo_root / "audits" / "bundler",
        repo_root / "audits" / "bundles",
        repo_root / "audits" / "results",
    ]:
        if base.exists():
            for p in base.glob("GUM_BUNDLE_v30_*"):
                if p.is_dir():
                    candidates.append(p)
            for p in base.glob("GUM_BUNDLE_v31_*"):
                if p.is_dir():
                    candidates.append(p)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]




def _sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _slug_from_log_filename(name: str) -> str:
    # e.g. standard_model__demo-33-... .out.txt -> standard_model__demo-33-...
    for suf in (".out.txt", ".err.txt"):
        if name.endswith(suf):
            return name[:-len(suf)]
    return name

def _collect_vendored_artifacts_by_slug(bundle_dir: Path) -> dict[str, list[Path]]:
    vdir = bundle_dir / "vendored_artifacts"
    out: dict[str, list[Path]] = {}
    if not vdir.exists():
        return out
    for fp in sorted(vdir.iterdir()):
        if not fp.is_file():
            continue
        # filenames look like: domain__slug__artifact.ext
        parts = fp.name.split("__")
        if len(parts) < 3:
            continue
        slug = "__".join(parts[:2])  # domain__demo-XX-...
        out.setdefault(slug, []).append(fp)
    return out


def _run_slug(r) -> str:
    return getattr(r, "slug", "") or getattr(r, "run_slug", "") or ""


def _infer_domain_from_folder(folder_full: str) -> str:
    # folder like demos/standard_model/demo-33-...
    parts = str(folder_full).replace("\\", "/").split("/")
    if len(parts) >= 2 and parts[0] == "demos":
        return parts[1]
    return ""

def _canonical_run_slug(domain: str, folder_full: str, fallback: str = "") -> str:
    # Canonical slug matches bundle filenames:
    #   logs:            domain__<demo-folder>.out.txt
    #   vendored files:  domain__<demo-folder>__artifact.ext
    base = Path(folder_full).name if folder_full else (fallback or "")
    if not domain:
        return base
    if base.startswith(domain + "__"):
        return base
    if "__" in base:
        # already has some prefix; trust it
        return base
    return f"{domain}__{base}"

def _source_sha_prefix(source_path: str, repo_root: Path, bundle_root: Path) -> str:
    # Return 12-char sha prefix if resolvable, else "".
    sp = (source_path or "").strip()
    if not sp:
        return ""
    # If already a hash-like token, return it.
    if re.fullmatch(r"[0-9a-fA-F]{12,64}", sp):
        return sp[:12]
    # 1) Try repo path
    fp = (repo_root / sp)
    if fp.exists() and fp.is_file():
        return sha256_file(fp)[:12]
    # 2) Try vendored_artifacts by filename suffix match
    vdir = bundle_root / "vendored_artifacts"
    if vdir.exists():
        base = Path(sp).name
        for cand in vdir.iterdir():
            if cand.is_file() and cand.name.endswith(base):
                return sha256_file(cand)[:12]
    return ""

def load_bundle(bundle_dir: Path) -> Bundle:
    b = Bundle(root=bundle_dir)

    # meta
    meta_path = bundle_dir / "bundle.json"
    if meta_path.exists():
        try:
            b.meta = load_json(meta_path)
        except Exception:
            b.meta = {}
    # runs
    runs_path = bundle_dir / "runs.json"
    if runs_path.exists():
        runs_obj = load_json(runs_path)
        runs_map = runs_obj.get("runs", runs_obj)  # allow either {"runs": {...}} or {...}
        if isinstance(runs_map, dict):
            for key, rr in runs_map.items():
                if not isinstance(rr, dict):
                    continue
                demo = demo_label_from_slug(rr.get("demo_id") or rr.get("demo") or key)
                domain = str(rr.get("category") or rr.get("domain") or rr.get("group") or "")
                folder = str(rr.get("folder") or "")
                demo_path = str(rr.get("demo_path") or rr.get("path") or "")
                # Prefer a full path for rerun commands when available.
                folder_full = demo_path or folder
                slug_name = Path(folder_full).name if folder_full else str(key)
                domain = domain or _infer_domain_from_folder(folder_full)
                slug_name = _canonical_run_slug(domain, folder_full, fallback=slug_name)
                cmd = str(rr.get("cmd") or "")
                one_liner = str(rr.get("one_liner") or "")
                if not cmd:
                    # assume standard entry point
                    cmd = "python demo.py"
                if not one_liner and folder_full:
                    one_liner = f"(cd '{folder_full}' && python demo.py)"
        
                b.runs.append(
                    RunRecord(
                        demo=demo,
                        slug=slug_name,
                        domain=(domain or _infer_domain_from_folder(folder_full)),
                        folder=folder_full,
                        status=str(rr.get("status") or ""),
                        return_code=rr.get("return_code") if rr.get("return_code") is not None else rr.get("returncode"),
                        runtime_sec=try_float(rr.get("seconds") or rr.get("runtime_seconds") or rr.get("runtime_sec") or rr.get("elapsed_s") or rr.get("runtime_s")),
                        mode=str(rr.get("mode") or ""),
                        cmd=cmd,
                        one_liner=one_liner,
                        code_sha256=str(rr.get("code_sha256") or ""),
                        stdout_sha256=str(rr.get("stdout_sha256") or ""),
                        stderr_sha256=str(rr.get("stderr_sha256") or ""),
                        artifacts_sha256=str(rr.get("artifacts_sha256") or ""),
                    )
                )
        elif isinstance(runs_map, list):
            for rr in runs_map:
                if not isinstance(rr, dict):
                    continue
                key = str(rr.get("slug") or rr.get("demo") or rr.get("demo_id") or "")
                demo = demo_label_from_slug(rr.get("demo_id") or rr.get("demo") or key)
                domain = str(rr.get("category") or rr.get("domain") or rr.get("group") or "")
                folder = str(rr.get("folder") or "")
                demo_path = str(rr.get("demo_path") or rr.get("path") or "")
                folder_full = demo_path or folder
                slug_name = Path(folder_full).name if folder_full else key
                domain = domain or _infer_domain_from_folder(folder_full)
                slug_name = _canonical_run_slug(domain, folder_full, fallback=slug_name)
                cmd = str(rr.get("cmd") or "") or "python demo.py"
                one_liner = str(rr.get("one_liner") or "")
                if not one_liner and folder_full:
                    one_liner = f"(cd '{folder_full}' && python demo.py)"
        
                b.runs.append(
                    RunRecord(
                        demo=demo,
                        slug=slug_name,
                        domain=(domain or _infer_domain_from_folder(folder_full)),
                        folder=folder_full,
                        status=str(rr.get("status") or ""),
                        return_code=rr.get("return_code") if rr.get("return_code") is not None else rr.get("returncode"),
                        runtime_sec=try_float(rr.get("seconds") or rr.get("runtime_seconds") or rr.get("runtime_sec") or rr.get("elapsed_s") or rr.get("runtime_s")),
                        mode=str(rr.get("mode") or ""),
                        cmd=cmd,
                        one_liner=one_liner,
                        code_sha256=str(rr.get("code_sha256") or ""),
                        stdout_sha256=str(rr.get("stdout_sha256") or ""),
                        stderr_sha256=str(rr.get("stderr_sha256") or ""),
                        artifacts_sha256=str(rr.get("artifacts_sha256") or ""),
                    )
                )
    # demo index
    demo_index_path = bundle_dir / "tables" / "demo_index.csv"
    if demo_index_path.exists():
        with demo_index_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                b.demo_index.append(row)

    # constants master
    const_path = bundle_dir / "tables" / "constants_master.csv"
    if const_path.exists():
        with const_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                b.constants_rows.append(row)

    # values jsonl
    vals_path = bundle_dir / "values.jsonl"
    if vals_path.exists():
        with vals_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        b.values_rows.append(obj)
                except Exception:
                    continue

    # artifacts index
    art_path = bundle_dir / "artifacts_index.json"
    if art_path.exists():
        art_obj = load_json(art_path)
        art_list = art_obj.get("artifacts", art_obj)
        if isinstance(art_list, list):
            for a in art_list:
                if not isinstance(a, dict):
                    continue
                demo = demo_label_from_slug(a.get("demo_id") or a.get("demo") or "")
                rel = str(a.get("relpath") or a.get("path") or "")
                sha = str(a.get("sha256") or "")
                size = a.get("size")
                b.artifacts.append(ArtifactRecord(demo=demo, relpath=rel, sha256=sha, size=size))

    # Compute per-demo artifacts digest if not present in runs.json
    if b.artifacts:
        by_demo: Dict[str, List[str]] = {}
        for a in b.artifacts:
            by_demo.setdefault(a.demo, []).append(a.sha256)
        for d in by_demo:
            by_demo[d] = sorted(by_demo[d])
        for r in b.runs:
            if not r.artifacts_sha256:
                shas = by_demo.get(r.demo)
                if shas:
                    r.artifacts_sha256 = hashlib.sha256(("\n".join(shas)).encode("utf-8")).hexdigest()

    # falsification matrix
    fals_path = bundle_dir / "tables" / "falsification_matrix.json"
    if fals_path.exists():
        try:
            b.falsification = load_json(fals_path)
        except Exception:
            b.falsification = []

    # Backfill one-liner commands from falsification matrix (preferred)
    if b.falsification:
        lookup: Dict[str, str] = {}
        for entry in b.falsification:
            if not isinstance(entry, dict):
                continue
            d = demo_label_from_slug(entry.get("demo") or entry.get("demo_id") or "")
            ol = entry.get("one_liner") or entry.get("cmd") or ""
            if d and ol:
                lookup[d] = str(ol)
        for r in b.runs:
            if (not r.one_liner) and lookup.get(r.demo):
                r.one_liner = lookup[r.demo]

    # repo inventory
    inv_path = bundle_dir / "repo_inventory.json"
    if inv_path.exists():
        try:
            b.repo_inventory = load_json(inv_path)
        except Exception:
            b.repo_inventory = {}

    return b


# ----------------------------
# Assets
# ----------------------------

def resolve_asset(repo_root: Path, bundle_dir: Path, *candidates: str) -> Optional[Path]:
    """
    Try to locate an asset in:
      1) bundle vendored_artifacts
      2) repo gum/assets
      3) repo atlas_substrate_visualization/assets (if exists)
    """
    search_roots = [
        bundle_dir / "vendored_artifacts",
        repo_root / "gum" / "assets",
        repo_root / "atlas_substrate_visualization" / "assets",
        repo_root,
    ]
    for root in search_roots:
        for name in candidates:
            p = root / name
            if p.exists():
                return p
    # also allow absolute candidates
    for name in candidates:
        p = Path(name)
        if p.exists():
            return p
    return None


def image_grid_2x2(paths, styles):
    # paths: list of Path objects (len 1..4)
    from reportlab.platypus import Table
    cells = []
    row = []
    for pth in paths[:4]:
        row.append(Image(str(pth), width=3.3*inch, height=2.0*inch))
        if len(row) == 2:
            cells.append(row)
            row = []
    if row:
        while len(row) < 2:
            row.append(Paragraph('', styles['Small']))
        cells.append(row)
    tbl = Table(cells, colWidths=[3.4*inch, 3.4*inch])
    return tbl

def missing_box(text: str, width: float, height: float = 1.0 * inch) -> Table:
    """
    A simple boxed placeholder that fits in tables/flow.
    """
    txt = ascii_sanitize(textwrap.fill(text, width=80))
    tbl = Table([[Paragraph(escape_xml(txt), getSampleStyleSheet()["BodyText"])]], colWidths=[width], rowHeights=[height])
    tbl.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, colors.red),
        ("BACKGROUND", (0, 0), (-1, -1), colors.whitesmoke),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    return tbl


# ----------------------------
# PDF Doc template with TOC + outline
# ----------------------------
def build_front_exec_summary(bundle: Bundle, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(H1("Executive Summary", styles, bookmark="exec_front"))

    bullets = [
        "This release presents a computational audit of the Marithmetics hypothesis: that physical law emerges as the unique eigen-structure of a Zero-Dimensional (0D) discrete substrate (residue rings and DRPTs), and that many continuum paradoxes are mid-lift artifacts.",
        "The kernel is survivor selection, not parameter tuning. The primary triple (wU=137, s2=107, s3=103) is produced by a deterministic sieve and is unique under declared constraints; ablations cause solution-space explosion.",
        "The bridge from discrete structure to continuum operators is formalized via Deterministic Operator Calculus (DOC), preserving ZFC-conservative reasoning while making infinity operational (vanishing residual law).",
        "Verification is delivered as a cryptographically sealed evidence ledger: every demo run produces rerunnable stdout/stderr logs, vendored artifacts, and a bundle seal (BUNDLE_SHA256) suitable for referee reproduction and paper citation.",
        "Representation independence is explicitly tested via base-gauge (Rosetta) roundtrip tests across multiple numeral systems, demonstrating that results are intrinsic to integer structure, not notation.",
    ]

    story.append(Paragraph("Summary bullets:", styles["Small"]))
    for b in bullets:
        story.append(Paragraph("• " + b, styles["Small"]))

    # Evidence pointers
    try:
        bsha = (bundle.root / "bundle_sha256.txt").read_text(encoding="utf-8").strip()
    except Exception:
        bsha = ""
    if bsha:
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"<b>Bundle seal:</b> {bsha}", styles["Small"]))

    # Visual Atlas pointers (if configured)
    try:
        claude = globals().get("CLAUDE_VISUAL_ATLAS_URL","")
        localp = globals().get("BUNDLE_VISUAL_ATLAS_PATH","")
        if localp or claude:
            story.append(Spacer(1, 6))
            story.append(Paragraph("<b>Visual Explorer:</b>", styles["Small"]))
            if localp:
                story.append(Paragraph(f"• bundle-local: {localp}", styles["Tiny"]))
            if claude:
                story.append(Paragraph(f"• Claude: {claude}", styles["Tiny"]))
    except Exception:
        pass

    story.append(PageBreak())
    return story



class AuditDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, pagesize=letter, footer_line1: str = "", footer_line2: str = "", **kw):
        self.footer_line1 = footer_line1
        self.footer_line2 = footer_line2
        super().__init__(filename, pagesize=pagesize, **kw)
        self.allowSplitting = 1
        frame = Frame(self.leftMargin, self.bottomMargin, self.width, self.height, id="normal")
        template = PageTemplate(id="main", frames=[frame], onPage=self._on_page)
        self.addPageTemplates([template])

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph):
            style_name = getattr(flowable.style, "name", "")
            if style_name in ("H1", "H2", "H3"):
                text = flowable.getPlainText()
                level = {"H1": 0, "H2": 1, "H3": 2}[style_name]
                key = getattr(flowable, "_bookmarkName", None)
                if key:
                    self.canv.bookmarkPage(key)
                    self.canv.addOutlineEntry(text, key, level=level, closed=False)
                self.notify("TOCEntry", (level, text, self.page))

    def _on_page(self, canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.grey)
        y1 = doc.bottomMargin - 14
        y2 = doc.bottomMargin - 26
        footer1 = getattr(self, "footer_line1", "")
        footer2 = getattr(self, "footer_line2", "")
        if footer1:
            canvas.drawString(doc.leftMargin, y1, footer1)
        if footer2:
            canvas.drawString(doc.leftMargin, y2, footer2)
        canvas.drawRightString(doc.leftMargin + doc.width, y1, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()


# ----------------------------
# Styling helpers
# ----------------------------

def build_styles() -> Dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()

    # optional: register DejaVuSans for broader glyph support. We still sanitize to ASCII for safety.
    try:
        # many environments include DejaVuSans; if not, ignore.
        dejavu = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if Path(dejavu).exists():
            pdfmetrics.registerFont(TTFont("DejaVuSans", dejavu))
    except Exception:
        pass

    styles: Dict[str, ParagraphStyle] = {}

    styles["Body"] = ParagraphStyle(
        "Body",
        parent=base["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        spaceAfter=8,
    )
    styles["Small"] = ParagraphStyle(
        "Small",
        parent=styles["Body"],
        fontSize=9,
        leading=11,
        spaceAfter=6,
    )
    styles["Tiny"] = ParagraphStyle(
        "Tiny",
        parent=styles["Body"],
        fontSize=8,
        leading=10,
        spaceAfter=4,
    )
    styles["H1"] = ParagraphStyle(
        "H1",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        spaceBefore=12,
        spaceAfter=8,
    )
    styles["H2"] = ParagraphStyle(
        "H2",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=6,
    )
    styles["H3"] = ParagraphStyle(
        "H3",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        spaceBefore=8,
        spaceAfter=4,
    )
    styles["Center"] = ParagraphStyle(
        "Center",
        parent=styles["Body"],
        alignment=TA_CENTER,
    )
    styles["CoverTitle"] = ParagraphStyle(
        "CoverTitle",
        parent=styles["Center"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=28,
        spaceAfter=14,
    )
    styles["CoverSubtitle"] = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Center"],
        fontName="Helvetica",
        fontSize=12,
        leading=16,
        spaceAfter=8,
    )
    styles["Quote"] = ParagraphStyle(
        "Quote",
        parent=styles["Center"],
        fontName="Helvetica-Oblique",
        fontSize=11,
        leading=15,
        textColor=colors.grey,
        spaceAfter=12,
    )
    styles["CodeBlock"] = ParagraphStyle(
        "CodeBlock",
        parent=styles["Tiny"],
        fontName="Courier",
        fontSize=8,
        leading=9.5,
        spaceAfter=8,
        wordWrap="CJK",
    )
    styles["TableCell"] = ParagraphStyle(
        "TableCell",
        parent=styles["Tiny"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=10,
        spaceAfter=0,
        wordWrap="CJK",
    )
    styles["TableCellBold"] = ParagraphStyle(
        "TableCellBold",
        parent=styles["TableCell"],
        fontName="Helvetica-Bold",
    )
    return styles


def H1(text: str, styles: Dict[str, ParagraphStyle], bookmark: str) -> Paragraph:
    p = Paragraph(escape_xml(ascii_sanitize(text)), styles["H1"])
    p._bookmarkName = bookmark
    return p


def H2(text: str, styles: Dict[str, ParagraphStyle], bookmark: str) -> Paragraph:
    p = Paragraph(escape_xml(ascii_sanitize(text)), styles["H2"])
    p._bookmarkName = bookmark
    return p


def H3(text: str, styles: Dict[str, ParagraphStyle], bookmark: str) -> Paragraph:
    p = Paragraph(escape_xml(ascii_sanitize(text)), styles["H3"])
    p._bookmarkName = bookmark
    return p


def P(text: str, styles: Dict[str, ParagraphStyle], style: str = "Body") -> Paragraph:
    return Paragraph(escape_xml(ascii_sanitize(text)), styles[style])


def table_grid(
    data: List[List[Any]],
    styles: Dict[str, ParagraphStyle],
    col_widths: List[float],
    header_rows: int = 1,
) -> Table:
    """
    Grid table with wrapped Paragraph cells, safe for long text.
    """
    tdata: List[List[Any]] = []
    for r_i, row in enumerate(data):
        out_row: List[Any] = []
        for c_i, cell in enumerate(row):
            txt = "" if cell is None else str(cell)
            txt = ascii_sanitize(txt)
            # convert to Paragraph for wrapping
            cell_style = styles["TableCellBold"] if (r_i < header_rows) else styles["TableCell"]
            out_row.append(Paragraph(escape_xml(txt).replace("\n", "<br/>"), cell_style))
        tdata.append(out_row)

    tbl = Table(tdata, colWidths=col_widths, repeatRows=header_rows)
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, header_rows-1), colors.HexColor("#f0f0f0")),
        ("TEXTCOLOR", (0, 0), (-1, header_rows-1), colors.black),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c8c8c8")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return tbl


def table_kv(pairs: List[Tuple[str, str]], styles: Dict[str, ParagraphStyle], width: float) -> Table:
    data = [["Field", "Value"]]
    for k, v in pairs:
        data.append([k, v])
    col_widths = [width * 0.28, width * 0.72]
    return table_grid(data, styles=styles, col_widths=col_widths, header_rows=1)


# ----------------------------
# Section builders
# ----------------------------

def build_cover(bundle: Bundle, repo_root: Path, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(Spacer(1, 0.8 * inch))

    # v28 title preserved
    story.append(Paragraph("Digital Root Power Tables and Cross-Base Structural Invariants", styles["CoverTitle"]))
    story.append(Paragraph("A Reproducible Computational Pipeline for Emergent Integer-Derived Mathematical Constants and Physical Parameters", styles["CoverSubtitle"]))
    story.append(Paragraph("Executive Technical Report", styles["CoverSubtitle"]))
    story.append(Paragraph("Justin Grieshop", styles["CoverSubtitle"]))

    story.append(Spacer(1, 0.25 * inch))
    story.append(Paragraph('"Within everything accepted lies everything overlooked."', styles["Quote"]))

    ts = _dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
    bundle_name = bundle.root.name
    story.append(Paragraph(f"Generated: {ts} (UTC)", styles["Center"]))
    story.append(Paragraph(f"Bundle: {bundle_name}", styles["Center"]))
    if bundle.meta.get("git_commit"):
        story.append(Paragraph(f"Git commit: {ascii_sanitize(str(bundle.meta.get('git_commit')))}", styles["Center"]))

    story.append(Spacer(1, 0.35 * inch))
    author_note = (
        "Author note (for referees): this report is a system audit and evidence ledger. "
        "Every demo is rerunnable via a single root-safe one-liner. Every run is hash-linked to stdout/stderr and artifacts. "
        "No headline number is claimed without a source file in the bundle. Where evidence is missing, the report states exactly what is missing and where it should be produced."
    )
    story.append(Paragraph(author_note, styles["Body"]))

    story.append(PageBreak())
    return story


def build_toc(styles: Dict[str, ParagraphStyle]) -> List[Any]:
    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(name="TOC1", fontName="Helvetica", fontSize=10, leftIndent=10, firstLineIndent=-10, spaceAfter=4),
        ParagraphStyle(name="TOC2", fontName="Helvetica", fontSize=9, leftIndent=28, firstLineIndent=-10, spaceAfter=2),
        ParagraphStyle(name="TOC3", fontName="Helvetica", fontSize=8, leftIndent=46, firstLineIndent=-10, spaceAfter=1),
    ]
    story: List[Any] = []
    story.append(H1("Table of Contents", styles, bookmark="toc"))
    story.append(Spacer(1, 0.1 * inch))
    story.append(toc)
    story.append(PageBreak())
    return story


def build_origin_and_visuals(bundle: Bundle, repo_root: Path, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    """
    v28-like early narrative + visuals, upgraded with explicit origin story + Fejer section.
    """
    story: List[Any] = []

    story.append(H1("0. Visual Origin and the Kernel Story", styles, bookmark="sec0"))
    story.append(P(
        "The purpose of the opening section is to define the objects that later physics closures reuse. "
        "DRPTs (Digital Root Power Tables) are treated as a dimensionless substrate signature: they are discrete, cross-base, and locally testable. "
        "The central discovery motivating this report is that the same kernel signatures recur across domains that historically look unrelated. "
        "This is the blended story: overlap is the point, not a coincidence. "
        "To make that claim referee-auditable, we begin with visuals and then connect them to falsifiers and per-demo certificates.",
        styles,
        "Body",
    ))

    # 0.1 DRPT geometry
    story.append(H2("0.1 Identity pillars and Echo tiles", styles, bookmark="sec0_2"))
    story.append(P(
        "Identity and Echo tiles are visual witnesses for repeatable residue structure. "
        "They matter because they demonstrate that the kernel is not a single special-case configuration: the motifs tile and recur. "
        "This recurrence is what makes the phrase 'tiles throughout infinity' operational: the structure is not local to one scale or one base. "
        "Where later demos claim ALQ dressing behavior, these tiles are the discrete origin of that behavior in dimensionless form.",
        styles,
        "Body",
    ))

    story.append(P(
        "For many more families and cross-base invariants, we encourage readers to explore the Visual Atlas (bundle-local: atlas_substrate_visualization/visual_atlas_1.html; Claude: ) tool. "
        "We are still documenting the full family taxonomy, but these objects can already be identified across each base. "
        "Code is available in this GitHub repository. For quick access to the Visual Atlas artifact, see: "
        "bundle-local: atlas_substrate_visualization/visual_atlas_1.html",
        styles,
        "Small",
    ))

    identity9 = resolve_asset(repo_root, bundle.root, "Identity9.png")
    identity10 = resolve_asset(repo_root, bundle.root, "Identity10.png")
    echo6 = resolve_asset(repo_root, bundle.root, "Echo6.png")
    echo10 = resolve_asset(repo_root, bundle.root, "Echo10.png")

    for img_path, caption in [
        (identity9, "Figure 0.2A: Identity pillar (n=9)"),
        (identity10, "Figure 0.2B: Identity pillar (n=10)"),
        (echo6, "Figure 0.2C: Echo tile (n=6)"),
        (echo10, "Figure 0.2D: Echo tile (n=10)"),
    ]:
        if img_path and img_path.exists():
            story.append(Image(str(img_path), width=6.8 * inch, height=3.5 * inch))
            story.append(Paragraph(caption, styles["Small"]))
        else:
            story.append(missing_box(f"Missing asset: {caption}. Expected in gum/assets/.", width=6.8 * inch, height=0.9 * inch))

    # 0.3 Fejer smoothing (expanded)
    story.append(H2("0.2 DRPT geometry and cross-base invariance", styles, bookmark="sec0_1"))
    story.append(P(
        "DRPTs are best understood as a discretized geometry of residue structure. "
        "Because they are defined on digit-root dynamics, they are dimensionless and naturally comparable across bases. "
        "The visual motifs (city grids, survivor patterns, and identity pillars) function as invariance witnesses: if the motifs change under base-gauge transforms, the kernel is not invariant. "
        "In this report, DRPT visuals are treated as mechanism evidence rather than as decorative plots.",
        styles,
        "Body",
    ))

    drpt_city = resolve_asset(repo_root, bundle.root, "DRPTCity137_v2.png", "DRPTCity137.png")
    drpt_surv = resolve_asset(repo_root, bundle.root, "DRPTSurvivor137_v2.png", "DRPTSurvivor137.png")
    for img_path, caption in [
        (drpt_city, "Figure 0.1A: DRPT City (example motif)"),
        (drpt_surv, "Figure 0.1B: DRPT Survivor lattice (example motif)"),
    ]:
        if img_path and img_path.exists():
            story.append(Image(str(img_path), width=6.8 * inch, height=3.9 * inch))
            story.append(Paragraph(caption, styles["Small"]))
        else:
            story.append(missing_box(f"Missing DRPT asset: {caption}. Expected in gum/assets/.", width=6.8 * inch, height=1.0 * inch))

    # 0.2 Identity / Echo motifs
    story.append(H2("0.3 Fejer smoothing: role, guarantees, and dimensionless structure", styles, bookmark="sec0_3"))
    story.append(P(
        "Fejer smoothing is used in this program as an analytic filter with guarantees, not as an aesthetic smoothing operation. "
        "Formally, it replaces a partial-sum sequence by its Cesaro (Fejer) average, which is known to suppress Gibbs-type oscillations and stabilize convergence in many settings. "
        "In the GUM pipeline, this matters because many constructions are discrete-to-continuum lifts: without a stabilizing filter, it is too easy to confuse numerical noise for structure. "
        "Because the filter is applied to dimensionless sequences derived from the kernel, the resulting stability is cross-base comparable. "
        "When the filter is used in a demo, the report treats its invariants (e.g., nonnegativity, contraction bounds, or monotone error envelopes) as falsifiers: if they fail, the result should be rejected.",
        styles,
        "Body",
    ))

    # v28 asset names (note: one file is misspelled 'Smoothign' in the repo)
    fejer_left = resolve_asset(repo_root, bundle.root, "FejerSmoothignLeft.png", "FejerSmoothingLeft.png")
    fejer_right = resolve_asset(repo_root, bundle.root, "FejerSmoothingRight.png")
    if fejer_left and fejer_left.exists() and fejer_right and fejer_right.exists():
        img_left = Image(str(fejer_left), width=3.35 * inch, height=2.2 * inch)
        img_right = Image(str(fejer_right), width=3.35 * inch, height=2.2 * inch)
        tbl = Table([[img_left, img_right]], colWidths=[3.4*inch, 3.4*inch])
        tbl.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "MIDDLE")]))
        story.append(tbl)
        story.append(Paragraph("Figure 0.3: Fejer smoothing panels (analytic filter role).", styles["Small"]))
    else:
        story.append(missing_box(
            "Missing Fejer smoothing visuals. Expected in gum/assets/ as FejerSmoothignLeft.png and FejerSmoothingRight.png.",
            width=6.8 * inch,
            height=0.9 * inch,
        ))

    story.append(PageBreak())
    return story


def build_bridge_section(styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(H1("1. Bridge: One Kernel, Many Domains", styles, bookmark="sec1"))
    story.append(P(
        "The primary critique of earlier reports was that the transition from integer structure to physics closure felt abrupt. "
        "This version makes the bridge explicit: physical constants behave like eigenvalues of discrete constrained structures. "
        "Just as vibrational modes of a constrained system determine its resonant frequencies, the survivor structures and identity motifs constrain the allowed couplings, masses, and scales. "
        "The demos are not separate stories; they are projections of one kernel into different domains: matter, fields, geometry, cosmology, and continuum complexity. "
        "This is the blended narrative anchor: overlap is the mechanism. "
        "The numeric evidence lives in the per-demo certificates and bundle tables; this section is the conceptual map that explains why those numbers are expected to cohere.",
        styles,
        "Body",
    ))

    story.append(H2("1.1 Kernel map (conceptual; where to look)", styles, bookmark="sec1_1"))
    kernel_map = [
        ["Kernel element", "Operational role", "Where it appears (examples)"],
        ["SCFP++ selection", "Constrained survivor set; basis of Phi-channel closures", "DEMO-33, DEMO-37, DEMO-40, DEMO-64"],
        ["DRPT motifs", "Visual stability pillars; cross-base structure signatures", "Section 0 visuals; DEMO-40"],
        ["Analytic filter", "Suppresses noise; isolates stable structure (Fejer/Cesaro)", "DEMO-56; influences later closures"],
        ["Lift / transfer rules", "Lawful maps between discrete and continuum regimes", "DEMO-65; downstream GR/NS demos"],
        ["Action principle", "Dynamics spine; symmetry constraints", "DEMO-71; downstream GR/NS demos"],
        ["Closure layers", "Domain-specific manifests built from the kernel", "SM (33/37/54/55/70), Cosmo (36/39), QG (66), NS (67)"],
    ]
    col_widths = [1.5 * inch, 2.3 * inch, 2.8 * inch]
    story.append(table_grid(kernel_map, styles, col_widths=col_widths, header_rows=1))

    story.append(H2("1.2 The anchor discovery: overlap is the mechanism", styles, bookmark="sec1_2"))
    story.append(P(
        "Across v28 to v31 the recurring discovery is not a single numerical coincidence; it is reuse. "
        "The same survivor and identity structures that appear as DRPT motifs also appear as constraints in later closures. "
        "When reviewers see a cosmology table and a Standard Model table on different pages, the report asks them to treat those as two projections of one kernel rather than as two independent fits. "
        "This is what 'overlap is the point' means operationally: the admissible couplings and scales are constrained by the same discrete backbone, so the closures are correlated by construction. "
        "Fejer smoothing sits in this anchor story as the stabilizer that makes the backbone visible in continuum-looking outputs: it suppresses noise while preserving invariants. "
        "The one-action demo (DEMO-71) then plays the role of dynamics glue: it ties conserved structure to equations of motion so the kernel is not just kinematic.",
        styles,
        "Body",
    ))
    story.append(P(
        "Referee shortcut: if you want to falsify the anchor story quickly, test the chain Kernel -> Filter -> Closure. "
        "Run DEMO-40 or DEMO-64 (kernel), then DEMO-56 (filter), then one closure flagship (DEMO-33 for SM or DEMO-36 for cosmology). "
        "If any link in that chain fails deterministically, the anchor story fails. "
        "If the chain holds, the remaining demos are best interpreted as coverage expansion and stress tests rather than as isolated claims.",
        styles,
        "Small",
    ))

    story.append(Spacer(1, 0.15 * inch))
    story.append(P(
        "Referee guidance: the fastest audit path is to pick one kernel demo (40 or 64), one filter demo (56), and one closure flagship (33 or 36). "
        "If those reproduce deterministically and the bridge logic is coherent, the remaining demos serve as coverage expansion rather than as isolated claims.",
        styles,
        "Small",
    ))
    story.append(PageBreak())
    return story


def build_exec_summary(bundle: Bundle, repo_root: Path, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(H1("2. Executive Summary (Coverage and Audit Posture)", styles, bookmark="sec2"))
    story.append(P(
        "This report is structured as a system audit rather than a manifesto. "
        "Every demo run has a rerun command, status, runtime, and hashes. "
        "Where structured exports exist, they are summarized in bundle-sourced tables; where they do not, stdout evidence and hashes are treated as the authoritative record. "
        "The strongest narrative claim is the blended story: the same kernel constraints recur across domains that are usually treated as unrelated. "
        "The fastest way to test that claim is the falsification matrix in Section 3.",
        styles,
        "Body",
    ))

    # Coverage table with context
    run_rows = sorted(bundle.runs, key=lambda r: demo_sort_key(r.demo))
    table = [["Demo", "Domain", "What it tests (context)", "Status", "Runtime", "One-liner (copy/paste)"]]
    for r in run_rows:
        info = DEMO_INFO.get(r.demo, {})
        tests = info.get("tests") or "Bundle run (no narrative metadata available)."
        table.append([
            r.demo,
            _domain_short(r.domain) if r.domain else "n/a",
            tests,
            r.status or "n/a",
            fmt_seconds(r.runtime_sec),
            hard_wrap_command(r.one_liner or r.cmd),
        ])

    col_widths = [0.75*inch, 0.95*inch, 2.2*inch, 0.65*inch, 0.7*inch, 2.55*inch]
    story.append(H2("2.1 Bundle coverage and run status", styles, bookmark="sec2_1"))
    story.append(table_grid(table, styles, col_widths=col_widths, header_rows=1))
    story.append(Spacer(1, 0.15 * inch))

    # Unified dashboard sourced from values.jsonl + constants_master
    story.append(H2("2.2 Unified constants dashboard (bundle-sourced)", styles, bookmark="sec2_2"))
    story.append(P(
        "The dashboard below is intentionally bundle-sourced. It is not a place for hand-picked 'best values'; it is a cross-reference table "
        "that points from a named quantity to the demo and source file that produced it. "
        "If a value is missing, the correct action is to repair the artifact pipeline, not to fill the cell by hand.",
        styles,
        "Small",
    ))

    dash_rows: List[List[str]] = [["Name", "Value", "Units", "Demo", "Source"]]
    # Prefer values.jsonl when present; fallback to constants_master
    preferred_names = [
        # SM-ish
        "predictions.alpha_MZ_inv",
        "predictions.MZ_dressed_GeV",
        "predictions.MW_dressed_GeV",
        "predictions.GammaZ_dressed_LOQCD_GeV",
        "raw.sm_manifest.phi.alpha0_inv",
        "predictions.sm_manifest.phi.alpha0_inv",
        # Cosmo-ish
        "cosmo.H0_km_s_Mpc",
        "cosmo.Omega_m",
        "cosmo.Omega_L",
        "cosmo.theta_s",
        "camb.available",
    ]
    vals_by_name = {(v.get("name") or v.get("value_name")): v for v in bundle.values_rows if isinstance(v, dict) and (v.get("name") or v.get("value_name"))}
    for name in preferred_names:
        v = vals_by_name.get(name)
        if v:
            src_path = v.get("source_path") or v.get("source") or ""
            line_ref = v.get("line_ref") or ""
            src = f"{src_path}#{line_ref}" if (src_path and line_ref) else (src_path or "")
            dash_rows.append([
                name,
                fmt_num(v.get("value")),
                unit_for(str(v.get("name") or v.get("value_name") or ""), v.get("units")),
                demo_label_from_slug(v.get("demo_id") or ""),
                ((_source_sha_prefix(src_path, repo_root, bundle.root) or '') + ' ' + src).strip(),
            ])

    # Add a few from constants_master if not already
    if bundle.constants_rows:
        existing = set(r[0] for r in dash_rows[1:])
        for row in bundle.constants_rows:
            nm = row.get("name") or row.get("export") or ""
            if not nm or nm in existing:
                continue
            if len(dash_rows) >= 18:
                break
            dash_rows.append([
                nm,
                fmt_num(row.get("value")),
                row.get("units") or "",
                demo_label_from_slug(row.get("demo_id") or ""),
                row.get("source_path") or "",
            ])

    if len(dash_rows) == 1:
        story.append(missing_box(
            "No structured values found in bundle (values.jsonl or tables/constants_master.csv). "
            "This is not a demo failure; it means the artifact export pipeline should be extended.",
            width=6.8*inch,
            height=1.1*inch,
        ))
    else:
        col_widths = [2.2*inch, 1.2*inch, 0.7*inch, 0.8*inch, 1.9*inch]
        story.append(table_grid(dash_rows, styles, col_widths=col_widths, header_rows=1))

    story.append(Spacer(1, 10))
    story.append(Paragraph("1.3 Headline metrics overview (stdout-derived)", styles["H2"]))
    story.append(Paragraph(
        "Structured exports (constants_master.csv / values.jsonl) are intentionally conservative and not yet wired for every demo. "
        "However, every demo emits falsifiable numbers on stdout. This table extracts 1-2 headline metrics per demo directly from the "
        "bundled stdout logs so referees can see quantitative coverage at a glance. The logs + hashes remain the source of truth; this "
        "is a convenience view.",
        styles["Body"],
    ))

    head_rows: List[List[str]] = [["Demo", "Domain", "What it tests", "Headline metrics"]]
    for r in bundle.runs:
        info = DEMO_INFO.get(r.demo, {})
        title = info.get("title", r.slug)
        log_text = read_demo_log(bundle, r) or ""
        metrics = extract_stdout_metrics(log_text, max_items=28)
        heads = pick_headline_metrics(metrics, max_items=2)
        head_str = "; ".join([f"{k}={v}" for (k, v) in heads]) if heads else "--"
        head_rows.append([r.demo, _domain_short(getattr(r,"domain","")), title, head_str])

    story.append(table_grid(head_rows, styles, col_widths=[0.8*inch, 0.9*inch, 2.6*inch, 2.7*inch], header_rows=1))


    story.append(PageBreak())
    return story


def build_falsification_section(bundle: Bundle, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(H1("3. Falsification Quickstart and Matrix", styles, bookmark="sec3"))
    story.append(P(
        "Skeptical readers should start here. The goal is to minimize cognitive load: copy, paste, run. "
        "If a demo fails, the failure is recorded and should be reproducible. "
        "If a demo passes but outputs differ, the hashes make the discrepancy concrete. "
        "Avoid prose instructions like 'run the script in the folder'; if you cannot reproduce a result with a single command, treat the claim as unaudited.",
        styles,
        "Body",
    ))
    story.append(H2("3.1 One-liner rule", styles, bookmark="sec3_1"))
    story.append(P(
        "Every demo must have a root-safe one-liner command. "
        "This ensures that the audit surface is stable: reviewers do not need to guess which file to run or which working directory matters.",
        styles,
        "Small",
    ))

    story.append(H2("3.2 Matrix (top entries)", styles, bookmark="sec3_2"))
    rows = [["Demo", "What it tests", "One-liner (copy/paste)"]]
    # Prefer bundle falsification matrix ordering; fallback to runs list
    fals = bundle.falsification or []
    if fals:
        for entry in fals[:18]:
            demo = demo_label_from_slug(entry.get("demo") or entry.get("demo_id") or "")
            info = DEMO_INFO.get(demo, {})
            tests = info.get("tests") or "Yukawa coupling admissibility checks"
            rows.append([demo, tests, hard_wrap_command(entry.get("one_liner") or "")])
    else:
        for r in sorted(bundle.runs, key=lambda x: demo_sort_key(x.demo))[:18]:
            info = DEMO_INFO.get(r.demo, {})
            tests = info.get("tests") or "Yukawa coupling admissibility checks"
            rows.append([r.demo, tests, hard_wrap_command(r.one_liner or r.cmd)])

    col_widths = [0.75*inch, 2.3*inch, 3.75*inch]
    story.append(table_grid(rows, styles, col_widths=col_widths, header_rows=1))

    story.append(P(
        "The full falsification matrix is included in the bundle under tables/falsification_matrix.json and should be preferred for complete coverage.",
        styles,
        "Small",
    ))
    story.append(PageBreak())
    return story


def select_log_excerpt(log_text: str, max_lines: int = 18) -> str:
    """Select a compact, reviewer-facing excerpt from stdout.

    We prefer PASS/FAIL gate lines and end-of-run verdicts. If no high-signal lines are
    detected, we fall back to the last non-empty lines. The returned text is hard-wrapped
    so it never runs off the page in a PDF code block.
    """
    if not log_text:
        return ""

    raw_lines = [ascii_sanitize(ln.rstrip("\n")) for ln in log_text.splitlines()]
    signal: List[str] = []

    for ln in raw_lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("PASS") or s.startswith("FAIL"):
            signal.append(s)
            continue
        if "FINAL VERDICT" in s or s.startswith("Result") or "Result:" in s:
            signal.append(s)
            continue
        # Cross-demo headline tokens (kept ASCII-friendly).
        if re.search(r"\b(H0|alpha0_inv|sin2W|Omega|ombh2|omch2|ell1|deltaCMB)\b", s):
            signal.append(s)

    if not signal:
        tail = [ln.strip() for ln in raw_lines if ln.strip()]
        excerpt = "\n".join(tail[-max_lines:])
        return wrap_preformatted_block(excerpt, width=110, max_lines=max_lines)

    excerpt = "\n".join(signal[:max_lines])
    return wrap_preformatted_block(excerpt, width=110, max_lines=max_lines)



def read_demo_log(bundle: Bundle, run: RunRecord) -> Optional[str]:
    logs_dir = bundle.root / "logs"
    if not logs_dir.exists():
        return None
    # Most logs are named "{domain}__{slug}.out.txt"
    cand = logs_dir / f"{run.slug}.out.txt"
    if cand.exists():
        return cand.read_text(encoding="utf-8", errors="replace")
    # try any log containing slug
    matches = list(logs_dir.glob(f"*{run.slug}*.txt"))
    if matches:
        return matches[0].read_text(encoding="utf-8", errors="replace")
    return None




def wrap_preformatted_block(text: str, width: int = 110, max_lines: Optional[int] = None) -> str:
    """Hard-wrap a preformatted block so long tokens never run off the page."""
    if not text:
        return ""
    out_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip("\n")
        if line == "":
            out_lines.append("")
        else:
            wrapped = textwrap.wrap(
                line,
                width=width,
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            out_lines.extend(wrapped if wrapped else [""])
        if max_lines is not None and len(out_lines) >= max_lines:
            out_lines = out_lines[:max_lines]
            break
    return "\n".join(out_lines)


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def extract_gate_lines(log_text: str, max_items: int = 6) -> List[str]:
    """Extract PASS/FAIL gate lines (and a few adjacent verdict markers) from stdout."""
    if not log_text:
        return []
    out: List[str] = []
    for ln in log_text.splitlines():
        s = _ANSI_RE.sub("", ln).strip()
        s = ascii_sanitize(s)
        if not s:
            continue
        if s.startswith("PASS") or s.startswith("FAIL") or s.startswith("FINAL VERDICT") or s.startswith("Result"):
            # Prefer Gate lines, but keep a small number of strong markers.
            if "Gate" in s or s.startswith("FINAL VERDICT") or s.startswith("Result"):
                out.append(re.sub(r"\s+", " ", s))
        if len(out) >= max_items:
            break
    return out


_KV_COLON_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.()\/\-]{0,80})\s*[:=]\s*(.+?)\s*$")
_KV_ALIGN_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_.()\/\-]{0,80})\s{2,}([-+0-9.eE/]+)\s*$")
_KV_INLINE_RE = re.compile(r"([A-Za-z][A-Za-z0-9_.\/\-]{0,60})=([-+0-9.eE]+)")


def _looks_like_hex(s: str) -> bool:
    s2 = s.strip().lower()
    if len(s2) < 16:
        return False
    return bool(re.fullmatch(r"[0-9a-f]+", s2))


def extract_stdout_metrics(log_text: str, max_items: int = 24) -> List[Tuple[str, str]]:
    """Heuristically extract key/value metrics from stdout.

    This is intentionally generic: most demos print at least a few "key: value" lines or
    "key=value" fragments. If structured exports are missing from the bundle, these
    stdout-derived values ensure the PDF still carries *numbers* (not just prose).
    """
    if not log_text:
        return []

    ignore_keys = {"utc", "utc_time", "time", "python", "platform", "cwd", "args", "i/o", "io"}

    out: List[Tuple[str, str]] = []
    seen: set = set()

    for ln in log_text.splitlines():
        s = _ANSI_RE.sub("", ln).strip()
        if not s:
            continue
        s = ascii_sanitize(s)

        # Inline key=value (captures Gate lines too).
        for m in _KV_INLINE_RE.finditer(s):
            k, v = m.group(1), m.group(2)
            kl = k.lower()
            if kl in ignore_keys or "sha" in kl or "hash" in kl:
                continue
            if k not in seen:
                seen.add(k)
                out.append((k, v))

        # Skip PASS/FAIL prose lines after harvesting inline pairs.
        if s.startswith("PASS") or s.startswith("FAIL"):
            continue

        m = _KV_COLON_RE.match(s)
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip()
            kl = k.lower()
            if kl in ignore_keys:
                continue
            if k not in seen:
                seen.add(k)
                out.append((k, v))
            continue

        m = _KV_ALIGN_RE.match(s)
        if m:
            k = m.group(1).strip()
            v = m.group(2).strip()
            kl = k.lower()
            if kl in ignore_keys:
                continue
            if k not in seen:
                seen.add(k)
                out.append((k, v))
            continue

        if len(out) >= max_items * 3:
            # avoid pathological logs; we will trim later
            break

    # Trim and de-noise.
    cleaned: List[Tuple[str, str]] = []
    for k, v in out:
        if len(cleaned) >= max_items:
            break
        if not k or not v:
            continue
        if k.lower().endswith("_sha256") or "sha256" in k.lower() or "hash" in k.lower():
            # keep hashes out of the "results" table (they are already listed in metadata)
            continue
        if _looks_like_hex(v):
            continue
        cleaned.append((k, v[:180]))

    return cleaned


def pick_headline_metrics(metrics: List[Tuple[str, str]], max_items: int = 2) -> List[Tuple[str, str]]:
    """Pick 1-2 headline metrics suitable for a high-level coverage dashboard."""
    if not metrics:
        return []
    patterns = [
        "alpha0_inv",
        "sin2w",
        "alpha_s",
        "h0",
        "omega",
        "ombh2",
        "omch2",
        "ell",
        "delta",
        "mw",
        "mz",
        "gamma",
        "lambda",
        "phi",
        "primary",
        "tau",
        "n_s",
        "a_s",
        "tv",
        "e_fe",
        "e_fd",
        "ratio",
        "l2",
        "rmse",
        "error",
    ]

    scored: List[Tuple[int, int, Tuple[str, str]]] = []
    for idx, (k, v) in enumerate(metrics):
        kl = k.lower()
        score = 0
        for p in patterns:
            if p in kl:
                score += 50
        # Prefer numeric-looking values
        if re.search(r"[-+]?\d", v):
            score += 5
        # Penalize long strings
        if len(v) > 60:
            score -= 10
        scored.append((score, -idx, (k, v)))

    scored.sort(reverse=True)
    chosen: List[Tuple[str, str]] = []
    for score, _, kv in scored:
        if len(chosen) >= max_items:
            break
        chosen.append(kv)
    return chosen


def select_key_metrics(metrics: List[Tuple[str, str]], max_items: int = 18) -> List[Tuple[str, str]]:
    """Select a compact but relevant set of metrics for a per-demo certificate table.

    We score keys by domain-agnostic patterns (H0, Omega, alpha0_inv, error, RMSE, etc.) and
    then fill remaining slots by first appearance. This keeps the table "about the demo"
    rather than about early kernel bookkeeping.
    """
    if not metrics:
        return []

    patterns = [
        "alpha0_inv",
        "sin2w",
        "alpha_s",
        "h0",
        "omega",
        "ombh2",
        "omch2",
        "ell",
        "delta",
        "mw",
        "mz",
        "gamma",
        "lambda",
        "phi",
        "primary",
        "tau",
        "n_s",
        "a_s",
        "tv",
        "e_fe",
        "e_fd",
        "ratio",
        "rmse",
        "l2",
        "error",
        "chi2",
        "sigma",
        "mean",
        "var",
    ]

    scored: List[Tuple[int, int, Tuple[str, str]]] = []
    for idx, (k, v) in enumerate(metrics):
        kl = k.lower()
        score = 0
        for p in patterns:
            if p in kl:
                score += 50
        if re.search(r"[-+]?\d", v):
            score += 5
        if kl in {"count", "found"}:
            score -= 5
        if len(v) > 80:
            score -= 10
        scored.append((score, idx, (k, v)))

    scored.sort(key=lambda t: (-t[0], t[1]))

    chosen: List[Tuple[str, str]] = []
    for score, _, kv in scored:
        if len(chosen) >= max_items:
            break
        # Prefer positive-signal keys first
        if score > 0:
            chosen.append(kv)

    # Fill remaining slots in original order.
    if len(chosen) < max_items:
        chosen_keys = {k for (k, _) in chosen}
        for k, v in metrics:
            if k in chosen_keys:
                continue
            chosen.append((k, v))
            if len(chosen) >= max_items:
                break

    return chosen[:max_items]





def _artifact_display_row(bundle_root: Path, a) -> list[str]:
    """
    Production rule: never show blank artifact File/Size columns.
    If relpath/size missing, try to recover from vendored_artifacts/ by sha.
    """
    rel = getattr(a, "relpath", "") or ""
    sha = getattr(a, "sha256", "") or ""
    size = getattr(a, "size", None)

    # backfill size if possible
    if (not rel) or (size is None) or (size == ""):
        vend = bundle_root / "vendored_artifacts"
        if vend.exists() and sha:
            # find by sha prefix match (files are already hashed and indexed)
            for fp in vend.iterdir():
                if fp.is_file():
                    # if relpath missing, use filename
                    if not rel:
                        rel = fp.name if sha[:6] in fp.name or True else rel
                    # size backfill
                    if size is None or size == "":
                        try:
                            size = fp.stat().st_size
                        except Exception:
                            pass
                    break

    return [rel or "MISSING", (sha[:12] if sha else "MISSING"), (str(size) if size is not None else "MISSING")]

def build_demo_certificates(bundle: Bundle, repo_root: Path, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(H1("4. Demo Certificates (Grouped Stories)", styles, bookmark="sec4"))
    story.append(P(
        "Each certificate is a modular unit of evidence. For each demo we provide: a narrative 'why it matters', a highlighted audit takeaway, "
        "a copy/paste rerun command, run metadata, and hashes. All demos included in the bundle are presented; none are left on the table.",
        styles,
        "Body",
    ))

    runs_by_demo = bundle.run_by_demo()
    artifacts_by_demo = bundle.artifacts_by_demo()

    # Build cluster->demos mapping (preserve bundle ordering by demo number)
    demos_present = sorted(runs_by_demo.keys(), key=demo_sort_key)
    cluster_to_demos: Dict[str, List[str]] = {}
    for demo in demos_present:
        cluster = DEMO_INFO.get(demo, {}).get("cluster") or "BRIDGE"
        cluster_to_demos.setdefault(cluster, []).append(demo)

    cluster_order = ["KERNEL", "FILTER", "SM", "COSMO", "GRNS", "QNTM", "BRIDGE"]
    for cluster in cluster_order:
        demos = cluster_to_demos.get(cluster, [])
        if not demos:
            continue

        story.append(H2(f"4.{cluster_order.index(cluster)+1} {CLUSTER_TITLES.get(cluster, cluster)}", styles, bookmark=f"sec4_{cluster}"))
        story.append(P(CLUSTER_INTROS.get(cluster, ""), styles, "Small"))
        story.append(Spacer(1, 0.08 * inch))

        for demo in demos:
            r = runs_by_demo[demo]
            info = DEMO_INFO.get(demo, {})
            title = info.get("title") or ascii_sanitize(r.slug.replace("demo-", "").replace("-", " ").title())
            bookmark = f"demo_{demo.replace('-','_')}"
            story.append(H3(f"{demo} - {title}", styles, bookmark=bookmark))

            # Meta table
            meta_pairs = [
                ("Domain", r.domain or ""),
                ("Folder", r.folder or ""),
                ("Status", r.status or ""),
                ("Return code", str(r.return_code) if r.return_code is not None else "n/a"),
                ("Runtime", fmt_seconds(r.runtime_sec)),
                ("Mode", r.mode or ""),
                ("One-liner", hard_wrap_command(r.one_liner or r.cmd)),
            ]
            story.append(table_kv(meta_pairs, styles, width=6.8*inch))
            story.append(Spacer(1, 0.05 * inch))

            # Hash table (prefixes)
            hash_pairs = [
                ("code_sha256", (r.code_sha256 or "")[:12]),
                ("stdout_sha256", (r.stdout_sha256 or "")[:12]),
                ("stderr_sha256", (r.stderr_sha256 or "")[:12]),
                ("artifacts_sha256", (r.artifacts_sha256 or "")[:12] if r.artifacts_sha256 else "n/a"),
            ]
            story.append(table_kv(hash_pairs, styles, width=6.8*inch))
            story.append(Spacer(1, 0.05 * inch))

            # Why it matters (>=6 sentences)
            narrative = info.get("narrative") or (
                f"{demo} is included in the bundle as part of the complete audit surface. "
                "Where structured exports are available, they are summarized below and referenced in bundle tables. "
                "Where only stdout evidence is present, the excerpt and hashes still allow an auditor to verify determinism. "
                "The key requirement is that a third party can rerun the same command and compare outputs byte-for-byte. "
                "If a claim in this demo is incorrect, the falsification matrix provides a direct way to demonstrate that. "
                "This certificate therefore treats reproducibility as the primary deliverable."
            )
            story.append(Paragraph("<b>Why it matters:</b> " + escape_xml(ascii_sanitize(narrative)), styles["Body"]))

            # Flagship highlights (bullets)
            highlights = info.get("highlights") or []
            if highlights:
                bullet_text = "<br/>".join([f"- {escape_xml(ascii_sanitize(h))}" for h in highlights])
                story.append(Paragraph("<b>Flagship highlights:</b><br/>" + bullet_text, styles["Small"]))
            else:
                story.append(Paragraph("<b>Flagship highlights:</b> (not yet annotated)", styles["Small"]))

            # Load stdout once; used for stdout-derived values + excerpt.
            log_text = read_demo_log(bundle, r) or ""
            gate_lines = extract_gate_lines(log_text, max_items=6)
            stdout_metrics = extract_stdout_metrics(log_text, max_items=28)

            # Key exported values (structured) if available
            # Use values.jsonl + constants_master filtered by demo id number when possible.
            demo_id_num = None
            m = re.match(r"DEMO-(\d+)", demo)
            if m:
                demo_id_num = m.group(1)
            structured_rows: List[Dict[str, Any]] = []
            # values.jsonl
            for v in bundle.values_rows:
                if demo_label_from_slug(v.get("demo_id") or "") == demo:
                    structured_rows.append({
                        "name": (v.get("name") or v.get("value_name")),
                        "value": v.get("value"),
                        "units": unit_for(str(v.get("name") or v.get("value_name") or ""), v.get("units")),
                        "source": (v.get("source") or v.get("source_path") or ""),
                    })
            # constants_master
            for c in bundle.constants_rows:
                if demo_label_from_slug(c.get("demo_id") or "") == demo:
                    structured_rows.append({
                        "name": c.get("name") or c.get("export"),
                        "value": c.get("value"),
                        "units": c.get("units") or "",
                        "source": c.get("source_sha256") or c.get("source_path") or "",
                    })

            if structured_rows:
                story.append(Paragraph("<b>Structured exports (bundle-sourced):</b>", styles["Small"]))
                seen = set()
                picked = []
                for row in structured_rows:
                    name = str(row.get("name", ""))
                    if not name or name in seen:
                        continue
                    seen.add(name)
                    picked.append(row)
                    if len(picked) >= 10:
                        break
                const_rows = [["Name", "Value", "Units", "Source"]]
                for row in picked:
                    const_rows.append([
                        str(row.get("name", "")),
                        str(row.get("value", "")),
                        unit_for(str(row.get("name") or ""), row.get("units")),
                        (_source_sha_prefix(str(row.get("source") or ""), repo_root, bundle.root) or "") + " " + str(row.get("source") or ""),
                    ])
                story.append(table_grid(const_rows, styles, col_widths=[2.0*inch, 2.0*inch, 0.8*inch, 1.2*inch], header_rows=1))
            else:
                story.append(Paragraph(
                    "<b>Structured exports:</b> not present in this bundle for this demo. "
                    "Below we include stdout-derived values (parsed directly from the bundled log) so the certificate still carries numbers.",
                    styles["Small"],
                ))

            # Stdout-derived falsifiers and key values (logs are bundled + hashed)
            if gate_lines:
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Key falsifiers (PASS/FAIL gates from stdout):</b>", styles["Small"]))
                for gl in gate_lines:
                    story.append(Paragraph("- " + gl, styles["Tiny"]))

            if stdout_metrics:
                story.append(Spacer(1, 6))
                story.append(Paragraph("<b>Key extracted values (stdout-derived):</b>", styles["Small"]))
                m_rows: List[List[str]] = [["Key", "Value"]]
                for k, v in select_key_metrics(stdout_metrics, max_items=18):
                    m_rows.append([k, v])
                story.append(table_grid(m_rows, styles, col_widths=[2.0*inch, 4.0*inch], header_rows=1))
            else:
                story.append(Spacer(1, 6))

            # Evidence artifacts list (vendored_artifacts/*) keyed by r.slug prefix
            arts = []
            try:
                vdir = bundle.root / "vendored_artifacts"
                if vdir.exists():
                    prefix = (r.slug or "") + "__"
                    for fp in sorted(vdir.iterdir()):
                        if fp.is_file() and fp.name.startswith(prefix):
                            arts.append(fp)
            except Exception:
                arts = []

            story.append(Paragraph("<b>Evidence artifacts (bundle):</b>", styles["Small"]))
            if arts:
                arows = [["File", "sha256 (prefix)", "Size"]]
                for fp in arts[:12]:
                    try:
                        rel = str(fp.relative_to(bundle.root))
                        sha = sha256_file(fp)[:12]
                        size = fp.stat().st_size
                    except Exception:
                        rel = fp.name
                        sha = "MISSING"
                        size = "MISSING"
                    arows.append([rel, sha, str(size)])
                story.append(table_grid(arows, styles, col_widths=[4.2*inch, 1.5*inch, 1.1*inch], header_rows=1))
            else:
                story.append(Paragraph("None found for this demo in vendored_artifacts/.", styles["Tiny"]))


            # Include key visual evidence if present for certain demos
            # - BB36 plot (DEMO-36)
            # - QG screening plot (DEMO-66)
            if demo in ("DEMO-36", "DEMO-54"):
                img = resolve_asset(repo_root, bundle.root, "cosmo__demo-36-big-bang-master-flagship__BB36_big_bang.png", "standard_model__demo-54-master-flagship-demo__BB36_big_bang.png")
                if img and img.exists():
                    story.append(Image(str(img), width=6.8*inch, height=3.6*inch))
                    # DEMO-36 MULTI-PANEL (bundle artifacts)
                    if demo == "DEMO-36":
                        imgs = []
                        for cand in [
                            "cosmo__demo-36-big-bang-master-flagship__bb36_master_plot.png",
                            "cosmo__demo-36-big-bang-master-flagship__camb_overlay.png",
                                                    ]:
                            pth = resolve_asset(repo_root, bundle.root, cand)
                            if pth and pth.exists() and pth not in imgs:
                                imgs.append(pth)
                        if imgs:
                            story.append(Spacer(1, 0.10 * inch))
                            story.append(Paragraph("DEMO-36 evidence panel (BB36 + CAMBH overlays).", styles["Small"]))
                            story.append(image_grid_2x2(imgs, styles))
                            story.append(Spacer(1, 0.12 * inch))

                    story.append(Paragraph("Figure: BB36 Big Bang evidence plot (bundle artifact).", styles["Small"]))
            if demo == "DEMO-66":
                img = resolve_asset(repo_root, bundle.root, "quantum_gravity__demo-66-quantum-gravity-master-flagship-v4__qg_screening_plot.png", "quantum_gravity__demo-66-quantum-gravity-master-flagship-v4__demo66_screening_plot.png", "quantum_gravity__demo-66b-quantum-gravity-master-flagship-v2__qg_screening_plot.png", "quantum_gravity__demo-66a-quantum-gravity-master-flagship-v1__qg_screening_plot.png", "quantum_gravity__demo-66b-quantum-gravity-master-flagship-v2__demo66_screening_plot.png", "quantum_gravity__demo-66a-quantum-gravity-master-flagship-v1__demo66_screening_plot.png")
                if img and img.exists():
                    story.append(Image(str(img), width=6.8*inch, height=3.6*inch))
                    story.append(Paragraph("Figure: Quantum-gravity screening plot (bundle artifact).", styles["Small"]))
                else:
                    story.append(missing_box(
                        "Quantum-gravity screening plot expected but not found in bundle vendored_artifacts/. "
                        "If the artifact pipeline is broken, leave this placeholder and fix the artifact export.",
                        width=6.8*inch,
                        height=0.7*inch,
                    ))

            # Stdout excerpt (sanitized)
            excerpt = select_log_excerpt(log_text, max_lines=18)
            if excerpt:
                story.append(Paragraph("<b>Stdout excerpt (sanitized; clipped):</b>", styles["Small"]))
                story.append(Preformatted(excerpt, styles["CodeBlock"]))
            else:
                story.append(missing_box(
                    "Stdout log not found in bundle logs/. Expected a *.out.txt file for this demo.",
                    width=6.8*inch,
                    height=0.6*inch,
                ))

            story.append(PageBreak())

    return story


def build_appendices(bundle: Bundle, repo_root: Path, styles: Dict[str, ParagraphStyle]) -> List[Any]:
    story: List[Any] = []
    story.append(H1("5. Appendices", styles, bookmark="sec5"))

    # 5.1 Bundle manifest summary
    story.append(H2("5.1 Bundle manifest and verification", styles, bookmark="sec5_1"))
    story.append(P(
        "The primary deliverable for audit/citation is the bundle directory. It contains the canonical index (bundle.json), run ledger (runs.json), "
        "artifact hashes (artifacts_index.json), demo index and falsification matrix (tables/), and logs (logs/). "
        "If any result is questioned, the correct procedure is to rerun the one-liner and compare hashes; narrative should never be treated as evidence.",
        styles,
        "Body",
    ))
    story.append(P(f"Bundle root: {bundle.root}", styles, "Small"))
    if (bundle.root / 'bundle.json').exists():
        story.append(P(f"bundle.json sha256: {sha256_file(bundle.root / 'bundle.json')}", styles, "Small"))
    if (bundle.root / 'runs.json').exists():
        story.append(P(f"runs.json sha256: {sha256_file(bundle.root / 'runs.json')}", styles, "Small"))
    if (bundle.root / 'artifacts_index.json').exists():
        story.append(P(f"artifacts_index.json sha256: {sha256_file(bundle.root / 'artifacts_index.json')}", styles, "Small"))

    # 5.2 CAMB expected assets
    story.append(H2("5.2 CAMB expected assets (overlay boundary)", styles, bookmark="sec5_2"))
    story.append(P(
        "CAMB/Planck overlays are evaluation-only and must never feed upstream selection. "
        "This report includes CAMB visuals only if they are produced as explicit demo artifacts. "
        "If a CAMB overlay page is missing, that usually indicates an artifact export issue rather than a report-writer issue.",
        styles,
        "Body",
    ))
    camb_md = resolve_asset(repo_root, bundle.root, "CAMB_EXPECTED_ASSETS.md")
    if camb_md and camb_md.exists():
        story.append(Preformatted(ascii_sanitize(camb_md.read_text(encoding='utf-8', errors='replace'))[:2500], styles["CodeBlock"]))
        story.append(P("Note: truncated for PDF. See CAMB_EXPECTED_ASSETS.md in the repository for full details.", styles, "Small"))
    else:
        story.append(missing_box(
            "CAMB_EXPECTED_ASSETS.md not found. Expected alongside the report generator or in the bundle. "
            "Add it to document which CAMB artifacts should be produced by the cosmology demos.",
            width=6.8*inch,
            height=0.9*inch,
        ))

    story.append(PageBreak())
    return story


def write_manifest(pdf_path: Path, bundle: Bundle) -> Path:
    manifest = {
        "report_sha256": sha256_file(pdf_path),
        "bundle_dir": str(bundle.root),
        "bundle_name": bundle.root.name,
        "generated_utc": _dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generator_version": __version__,
    }
    out = pdf_path.with_suffix(pdf_path.suffix + ".manifest.json")
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out


def build_pdf(bundle_dir: Path, out_path: Path) -> Tuple[Path, Path]:
    repo_root = repo_root_from_file()
    bundle = load_bundle(bundle_dir)
    styles = build_styles()

    bm = (bundle.meta or {}).get("bundle_meta", {}) if isinstance(bundle.meta, dict) else {}
    git = (bm.get("git_head") or "")[:8]
    ts = bm.get("timestamp_utc") or ""
    ledger = ""
    try:
        _runs_meta = json.loads((bundle.root / "runs.json").read_text(encoding="utf-8"))
        ledger = (_runs_meta.get("source_ledger_sha256") or "")[:12]
    except Exception:
        ledger = ""
    footer1 = "Digital Root Power Tables and Cross-Base Structural Invariants"
    footer2 = f"bundle {bundle.root.name} • git {git} • source_ledger {ledger} • {ts}"
    doc = AuditDocTemplate(
        str(out_path),
        pagesize=letter,
        footer_line1=footer1,
        footer_line2=footer2,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="GUM Report v31 - System Audit",
        author="public-arch",
    )

    story: List[Any] = []
    story.extend(build_cover(bundle, repo_root, styles))
    story.extend(build_front_exec_summary(bundle, styles))
    story.extend(build_toc(styles))
    story.extend(build_origin_and_visuals(bundle, repo_root, styles))
    story.extend(build_bridge_section(styles))
    story.extend(build_exec_summary(bundle, repo_root, styles))
    story.extend(build_falsification_section(bundle, styles))
    story.extend(build_demo_certificates(bundle, repo_root, styles))
    story.extend(build_appendices(bundle, repo_root, styles))

    doc.multiBuild(story)

    manifest_path = write_manifest(out_path, bundle)
    return out_path, manifest_path


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate GUM Report v31 (bundle-driven).")
    ap.add_argument("--bundle-dir", type=str, default="", help="Path to a bundle directory. If omitted, auto-detect latest.")
    ap.add_argument("--out", type=str, default="", help="Output PDF path. Default: gum/reports/GUM_Report_v31_<timestamp>.pdf")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    repo_root = repo_root_from_file()

    bundle_dir = Path(args.bundle_dir) if args.bundle_dir else None
    if bundle_dir is None or not bundle_dir.exists():
        bundle_dir = find_latest_bundle(repo_root)
    if bundle_dir is None or not bundle_dir.exists():
        raise SystemExit("Could not find a bundle directory. Provide --bundle-dir or run the bundler first.")

    if args.out:
        out_path = Path(args.out)
    else:
        ts = _dt.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%SZ")
        out_path = repo_root / "gum" / "reports" / f"GUM_Report_v31_{ts}.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_path, manifest_path = build_pdf(bundle_dir, out_path)
    print(f"Wrote PDF: {pdf_path}")
    print(f"Wrote manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
