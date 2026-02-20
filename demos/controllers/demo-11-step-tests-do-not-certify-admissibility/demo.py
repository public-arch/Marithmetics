#!/usr/bin/env python3
"""
FLAGSHIP DEMO (self-contained)

Paper title (strategic):
    STEP TESTS DO NOT CERTIFY ADMISSIBILITY
    A constructive boundary for spectral cutoff operators

What this demo does:
  1) Defines three periodic spectral filters on length-N signals:
        - Fejér (Cesàro) low-pass   [admissible]
        - Sharp cutoff low-pass    [non-admissible]
        - Signed cutoff (+1/-1)    [non-admissible]
  2) Sweeps spectral cutoff K from 1..Nyquist-1 and computes:
        - observed step overshoot (max distance outside [0,1] on a step input)
        - guaranteed worst-case violation lower bound (constructive witness from kernel negativity)
        - kernel minimum coefficient and negative mass
  3) Produces three paper-ready figures (PNG + PDF):
        - HERO: overshoot → 0 near Nyquist while worst-case violation stays > 0
        - MECHANISM: kernel triptych (Fejér vs Sharp vs Signed) with zoomed insets
        - PHASE: observed overshoot vs guaranteed failure (scatter, colored by K)
  4) Emits reproducible hashes:
        - spec_sha256: hashes all parameters that define the experiment
        - determinism_sha256: hashes all computed numeric outputs (arrays)
        - file sha256: hashes each saved figure and JSON output

Dependencies:
    numpy, matplotlib  (no SciPy)

Run:
    python flagship_demo_step_tests_do_not_certify_admissibility.py
    python flagship_demo_step_tests_do_not_certify_admissibility.py --out demo_out

Outputs:
    <out_dir>/
        data/demo_data.json
        figures/Fig1_HERO.png, Fig1_HERO.pdf
        figures/Fig2_KernelTriptych.png, Fig2_KernelTriptych.pdf
        figures/Fig3_ObservedVsGuaranteed.png, Fig3_ObservedVsGuaranteed.pdf
        MANIFEST.json   (file hashes)

Notes:
  - This demo is fully deterministic. No random numbers are used.
  - All operators are mass-preserving (DC gain = 1), so they fix constant signals.
  - Admissibility here means: maps every x in [0,1]^N back into [0,1]^N.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# -----------------------------
# Styling (premium defaults)
# -----------------------------

def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "font.size": 13,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# -----------------------------
# Hash helpers
# -----------------------------

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_json_dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def spec_hash(spec: dict) -> str:
    return sha256_hex(canonical_json_dumps(spec).encode("utf-8"))


def determinism_hash(spec_sha256: str, arrays: Dict[str, np.ndarray], scalars: Dict[str, float | int | str]) -> str:
    """
    Robust determinism hash: hashes spec_sha256 + array byte payloads + scalar payloads.
    """
    h = hashlib.sha256()
    h.update(spec_sha256.encode("ascii"))
    # Scalars (sorted for determinism)
    for k in sorted(scalars.keys()):
        v = scalars[k]
        h.update(k.encode("utf-8"))
        h.update(b"\0")
        h.update(str(v).encode("utf-8"))
        h.update(b"\0")
    # Arrays (sorted key order)
    for k in sorted(arrays.keys()):
        a = np.asarray(arrays[k], dtype=np.float64)
        h.update(k.encode("utf-8"))
        h.update(b"\0")
        h.update(str(a.shape).encode("utf-8"))
        h.update(b"\0")
        h.update(a.tobytes(order="C"))
        h.update(b"\0")
    return h.hexdigest()


# -----------------------------
# Spectral filters (operators)
# -----------------------------

def freq_ints(N: int) -> np.ndarray:
    """
    Map FFT indices 0..N-1 to integer frequencies:
        0,1,2,...,N/2, -(N/2-1),..., -2,-1  (for even N)
    """
    k = np.arange(N)
    return np.where(k <= N // 2, k, k - N)


def H_fejer(N: int, K: int) -> np.ndarray:
    """
    Fejér/Cesàro low-pass: triangular multiplier
        H(k) = 1 - |k|/(K+1)   for |k|<=K
             = 0              otherwise
    """
    f = freq_ints(N)
    H = np.zeros(N, dtype=np.float64)
    m = np.abs(f) <= K
    H[m] = 1.0 - (np.abs(f[m]) / (K + 1.0))
    return H


def H_sharp(N: int, K: int) -> np.ndarray:
    """
    Sharp low-pass:
        H(k)=1 for |k|<=K else 0
    """
    f = freq_ints(N)
    H = np.zeros(N, dtype=np.float64)
    H[np.abs(f) <= K] = 1.0
    return H


def H_signed(N: int, K: int) -> np.ndarray:
    """
    Signed cutoff control:
        H(k)=+1 on passband |k|<=K, and H(k)=-1 on stopband.
    """
    f = freq_ints(N)
    H = -np.ones(N, dtype=np.float64)
    H[np.abs(f) <= K] = 1.0
    return H


def kernel_from_H(H: np.ndarray) -> np.ndarray:
    """
    Impulse response (periodic kernel) from frequency response H via inverse FFT.
    """
    h = np.fft.ifft(H).real
    return h.astype(np.float64, copy=False)


def apply_filter(H: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Periodic convolution via FFT: y = ifft(fft(x)*H).
    """
    return np.fft.ifft(np.fft.fft(x) * H).real.astype(np.float64, copy=False)


def step_signal(N: int) -> np.ndarray:
    """
    Canonical step on a periodic grid:
        x[n]=1 for n < N/2, 0 for n >= N/2
    Discontinuity at n=N/2.
    """
    x = np.zeros(N, dtype=np.float64)
    x[: N // 2] = 1.0
    return x


# -----------------------------
# Admissibility witness
# -----------------------------

@dataclasses.dataclass(frozen=True)
class Metrics:
    K: int
    kmin: float
    neg_sum: float
    pos_sum: float
    neg_count: int
    step_out: float
    worst_lb: float     # guaranteed lower bound on violation
    sum_kernel: float


def row0_coeffs_from_kernel(h: np.ndarray) -> np.ndarray:
    """
    For periodic convolution y[n] = sum_m h[m] x[n-m],
    the coefficients for y[0] are row0[j] = h[-j mod N].
    """
    N = h.size
    idx = (-np.arange(N)) % N
    return h[idx]


def compute_metrics_for_K(N: int, K: int, op: str, X_step: np.ndarray) -> Metrics:
    if op == "fejer":
        H = H_fejer(N, K)
    elif op == "sharp":
        H = H_sharp(N, K)
    elif op == "signed":
        H = H_signed(N, K)
    else:
        raise ValueError(f"Unknown op={op!r}")

    h = kernel_from_H(H)
    row0 = row0_coeffs_from_kernel(h)

    tol = 1e-15
    neg_mask = row0 < -tol
    pos_mask = row0 > tol

    neg_sum = float(row0[neg_mask].sum())
    pos_sum = float(row0[pos_mask].sum())
    kmin = float(row0.min())
    sum_kernel = float(row0.sum())

    # step overshoot
    y_step = np.fft.ifft(X_step * H).real
    step_out = float(max(y_step.max() - 1.0, -y_step.min(), 0.0))

    # constructive worst-case lower bound:
    # - choose x_neg = 1 on negative coefficients → y(0)=neg_sum<0
    # - choose x_pos = 1 on positive coefficients → y(0)=pos_sum>1
    # Because sum(row0)=1, both violations have magnitude -neg_sum (when neg_sum<0).
    worst_lb = float(max(-neg_sum, pos_sum - 1.0, 0.0))

    return Metrics(
        K=int(K),
        kmin=kmin,
        neg_sum=neg_sum,
        pos_sum=pos_sum,
        neg_count=int(neg_mask.sum()),
        step_out=step_out,
        worst_lb=worst_lb,
        sum_kernel=sum_kernel,
    )


def witness_vectors_from_row0(row0: np.ndarray, tol: float = 1e-15) -> Tuple[np.ndarray, np.ndarray]:
    neg = (row0 < -tol).astype(np.float64)
    pos = (row0 > tol).astype(np.float64)
    return neg, pos


# -----------------------------
# Figure builders
# -----------------------------

def fig1_hero(
    out_dir: Path,
    title: str,
    N: int,
    K_primary: int,
    K_ill: int,
    K_sweep: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]],
    illusion: Dict[str, Metrics],
    X_step: np.ndarray,
    hash_tag: str,
) -> Path:
    """
    HERO: Top panel shows worst-case LB vs K (solid) + step overshoot vs K (dashed on right axis).
    Bottom-left: step looks fine at illusion point K=Nyquist-1.
    Bottom-right: constructive witness forces failure on {0,1} input.
    """
    # Colors chosen to match a consistent narrative
    colors = {
        "fejer": "tab:blue",
        "sharp": "tab:orange",
        "signed": "tab:red",
    }
    names = {
        "fejer": "Fejér (admissible)",
        "sharp": "Sharp cutoff",
        "signed": "Signed cutoff",
    }

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.35, wspace=0.28)

    # --- Top: K sweep
    ax = fig.add_subplot(gs[0, :])
    ax2 = ax.twinx()

    # Worst-case LB (solid)
    for op in ["fejer", "sharp", "signed"]:
        ax.plot(
            K_sweep,
            curves[op]["worst_lb"],
            color=colors[op],
            lw=3.0,
            label=f"Worst-case LB ({names[op]})" if op != "fejer" else "Worst-case LB (Fejér, =0)",
        )

    # Step overshoot (dashed) — context only
    for op in ["sharp", "signed"]:
        ax2.plot(
            K_sweep,
            curves[op]["step_out"],
            color=colors[op],
            lw=2.2,
            ls="--",
            alpha=0.85,
            label=f"Observed step overshoot ({names[op]})",
        )

    ax.set_title(f"Hero: Step passes, admissibility fails (constructive boundary)\n{title}", pad=12)
    ax.set_xlabel("Spectral cutoff K")
    ax.set_ylabel("Guaranteed worst-case violation lower bound (LB)")
    ax2.set_ylabel("Observed step overshoot (context)")

    # Vertical markers
    ax.axvline(K_primary, color="k", ls=":", lw=2.0, alpha=0.8)
    ax.text(K_primary, ax.get_ylim()[1] * 0.98, "K_primary", ha="center", va="top", fontsize=12, alpha=0.8)

    ax.axvline(K_ill, color="k", ls="-", lw=2.0, alpha=0.8)
    ax.text(K_ill, ax.get_ylim()[1] * 0.98, "Nyquist−1", ha="right", va="top", fontsize=12, alpha=0.85)

    # Annotate illusion point
    sharp_ill = illusion["sharp"]
    signed_ill = illusion["signed"]
    # Arrow to sharp point at K_ill
    ax.scatter([K_ill, K_ill], [sharp_ill.worst_lb, signed_ill.worst_lb],
               s=120, zorder=5, color=[colors["sharp"], colors["signed"]])
    ax.annotate(
        "Illusion point:\nstep overshoot ≈ 0\nbut admissibility still fails\n(constructive witness)",
        xy=(K_ill, sharp_ill.worst_lb),
        xytext=(int(0.68 * K_ill), float(0.78 * ax.get_ylim()[1])),
        arrowprops=dict(arrowstyle="->", lw=2.2),
        fontsize=13,
        ha="left",
        va="center",
    )

    # Legends (separate so they don't overlap)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    leg1 = ax.legend(h1, l1, loc="upper left", framealpha=0.95)
    ax.add_artist(leg1)
    ax2.legend(h2, l2, loc="upper center", framealpha=0.95)

    ax2.set_ylim(0.0, max(0.20, float(curves["signed"]["step_out"].max()) * 1.05))

    # --- Bottom-left: step near discontinuity at illusion point
    ax_step = fig.add_subplot(gs[1, 0])

    # Build step and outputs at K_ill
    x_step = step_signal(N)
    H_sh = H_sharp(N, K_ill)
    H_si = H_signed(N, K_ill)
    y_sh = np.fft.ifft(X_step * H_sh).real
    y_si = np.fft.ifft(X_step * H_si).real

    # Window around discontinuity
    center = N // 2
    half_window = int(0.08 * N)  # ~8% of domain
    idx = np.arange(center - half_window, center + half_window)
    pos = (idx - center) / N

    ax_step.plot(pos, x_step[idx], color="k", lw=3.0, label="True step")
    ax_step.plot(pos, y_sh[idx], color=colors["sharp"], lw=2.5, label=f"Sharp @ K={K_ill}")
    ax_step.plot(pos, y_si[idx], color=colors["signed"], lw=2.5, label=f"Signed @ K={K_ill}")

    ax_step.set_title("Step test can falsely reassure", pad=10)
    ax_step.set_xlabel("Position relative to discontinuity (fraction of domain)")
    ax_step.set_ylabel("Value")
    ax_step.set_ylim(-0.12, 1.12)
    ax_step.legend(loc="upper right", framealpha=0.95)

    # --- Bottom-right: constructive witness on {0,1} input
    ax_w = fig.add_subplot(gs[1, 1])

    # witness values (exactly from Metrics)
    cats = ["y_neg(0)", "y_pos(0)"]
    xloc = np.arange(len(cats))
    width = 0.34

    yneg_sh = illusion["sharp"].neg_sum
    ypos_sh = illusion["sharp"].pos_sum
    yneg_si = illusion["signed"].neg_sum
    ypos_si = illusion["signed"].pos_sum

    ax_w.bar(xloc - width / 2, [yneg_sh, ypos_sh], width, color=colors["sharp"], alpha=0.9, label="Sharp witness")
    ax_w.bar(xloc + width / 2, [yneg_si, ypos_si], width, color=colors["signed"], alpha=0.9, label="Signed witness")

    ax_w.axhline(0.0, color="k", lw=1.6, alpha=0.75)
    ax_w.axhline(1.0, color="k", lw=1.6, alpha=0.75)

    ax_w.set_xticks(xloc, cats)
    ax_w.set_ylabel("Output value")
    ax_w.set_title("Guaranteed failure (constructive witness on {0,1} input)", pad=10)
    ax_w.legend(loc="upper left", framealpha=0.95)

    # value labels
    def _label_bars(vals, xshift):
        for i, v in enumerate(vals):
            ax_w.text(xloc[i] + xshift, v + (0.03 if v >= 0 else -0.06), f"{v:.3g}",
                      ha="center", va="bottom" if v >= 0 else "top", fontsize=11, alpha=0.95)

    _label_bars([yneg_sh, ypos_sh], -width / 2)
    _label_bars([yneg_si, ypos_si], +width / 2)

    # Footer metadata (paper-ready)
    footer = (
        f"N={N}  |  K_primary={K_primary}  |  Nyquist−1={K_ill}  |  "
        f"step_out(Nyquist−1): sharp={illusion['sharp'].step_out:.3e}, signed={illusion['signed'].step_out:.3e}"
    )
    fig.text(0.01, 0.01, footer, fontsize=11, alpha=0.85)
    fig.text(0.99, 0.01, hash_tag, fontsize=11, alpha=0.75, ha="right")

    fig_path = out_dir / "Fig1_HERO.png"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(out_dir / "Fig1_HERO.pdf", bbox_inches="tight")
    plt.close(fig)
    return fig_path


def fig2_kernel_triptych(
    out_dir: Path,
    title: str,
    N: int,
    K_primary: int,
    kernels: Dict[str, np.ndarray],
    kmins: Dict[str, float],
    hash_tag: str,
) -> Path:
    """
    MECHANISM: show kernels in real space and highlight negativity.
    """
    colors = {"fejer": "tab:blue", "sharp": "tab:orange", "signed": "tab:red"}
    names = {"fejer": "Fejér kernel (admissible)", "sharp": "Sharp cutoff kernel (non-admissible)", "signed": "Signed control kernel (non-admissible)"}

    fig, axes = plt.subplots(3, 1, figsize=(16, 9), sharex=True)
    fig.suptitle(f"Mechanism: admissibility is visible in real space (kernel nonnegativity)\n{title}", y=0.985, fontsize=20)

    # Centered lag axis
    h0 = kernels["fejer"]
    h0_shift = np.fft.fftshift(h0)
    lags = np.arange(-N // 2, N // 2)

    # Plot only a symmetric window for readability
    L = 256
    sel = (lags >= -L) & (lags <= L)

    for ax, op in zip(axes, ["fejer", "sharp", "signed"]):
        h = kernels[op]
        hs = np.fft.fftshift(h)
        ax.plot(lags[sel], hs[sel], color=colors[op], lw=2.5)
        ax.axhline(0.0, color="k", lw=1.4, alpha=0.7)

        kmin = kmins[op]
        ax.set_ylabel("Kernel value")
        ax.set_title(f"{names[op]}   (kmin={kmin:+.3f})", pad=8)

        # inset zoom near 0
        axins = inset_axes(ax, width="28%", height="55%", loc="upper right", borderpad=1.2)
        zoom = 25
        selz = (lags >= -zoom) & (lags <= zoom)
        axins.plot(lags[selz], hs[selz], color=colors[op], lw=2.0)
        axins.axhline(0.0, color="k", lw=1.0, alpha=0.7)
        axins.set_xticks([-20, 0, 20])
        axins.tick_params(labelsize=10)
        # dynamic y-limits with a small pad
        ymin = float(hs[selz].min())
        ymax = float(hs[selz].max())
        pad = 0.08 * (ymax - ymin + 1e-12)
        axins.set_ylim(ymin - pad, ymax + pad)

    axes[-1].set_xlabel("Lag index (relative)")
    fig.text(0.99, 0.01, hash_tag, fontsize=11, alpha=0.75, ha="right")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.92, bottom=0.07, hspace=0.55)

    fig_path = out_dir / "Fig2_KernelTriptych.png"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(out_dir / "Fig2_KernelTriptych.pdf", bbox_inches="tight")
    plt.close(fig)
    return fig_path


def fig3_observed_vs_guaranteed(
    out_dir: Path,
    title: str,
    K_sweep: np.ndarray,
    curves: Dict[str, Dict[str, np.ndarray]],
    K_ill: int,
    hash_tag: str,
) -> Path:
    """
    PHASE: scatter of observed overshoot vs guaranteed failure; colored by K.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    fig.suptitle(f"Step tests do not certify admissibility: observed overshoot vs guaranteed failure\n{title}", fontsize=20, y=1.02)

    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=int(K_sweep.min()), vmax=int(K_sweep.max()))

    # unified settings
    xlim = (0.0, 0.20)

    panels = [("sharp", "Observed vs Guaranteed (sharp)"), ("signed", "Observed vs Guaranteed (signed)")]
    for ax, (op, t) in zip(axes, panels):
        x = curves[op]["step_out"]
        y = curves[op]["worst_lb"]
        sc = ax.scatter(x, y, c=K_sweep, cmap=cmap, norm=norm, s=70, alpha=0.95, edgecolors="none")
        ax.set_title(t, pad=10)
        ax.set_xlabel("Observed step overshoot (max outside [0,1])")
        ax.set_ylabel("Guaranteed worst-case violation lower bound")
        ax.set_xlim(*xlim)
        ax.grid(True, alpha=0.25)

        # annotate Nyquist-1
        idx = int(K_ill - K_sweep[0])
        x_ill = float(x[idx])
        y_ill = float(y[idx])
        ax.scatter([x_ill], [y_ill], s=220, facecolors="none", edgecolors="k", lw=2.2, zorder=5)
        ax.annotate(
            f"K=Nyquist−1\nstep_out≈{x_ill:.2e}\nLB≈{y_ill:.3g}",
            xy=(x_ill, y_ill),
            xytext=(0.11, 0.65 * ax.get_ylim()[1]),
            arrowprops=dict(arrowstyle="->", lw=2.0),
            fontsize=13,
        )

    # colorbar

    # Colorbar shared across both panels (avoid tight_layout warnings)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.046, pad=0.04)
    cb.set_label("Cutoff K", fontsize=14)
    fig.text(0.99, 0.01, hash_tag, fontsize=11, alpha=0.75, ha="right")
    fig.subplots_adjust(top=0.84, wspace=0.28)

    fig_path = out_dir / "Fig3_ObservedVsGuaranteed.png"
    fig.savefig(fig_path, bbox_inches="tight")
    fig.savefig(out_dir / "Fig3_ObservedVsGuaranteed.pdf", bbox_inches="tight")
    plt.close(fig)
    return fig_path


# -----------------------------
# Main demo
# -----------------------------

def build_spec(title: str, N: int, K_primary: int) -> dict:
    K_ill = N // 2 - 1
    spec = {
        "paper_title": title,
        "model": "Periodic spectral cutoff operators (Fejér / Sharp / Signed)",
        "N": N,
        "K_primary": K_primary,
        "K_ill": K_ill,
        "K_sweep": {"start": 1, "stop": K_ill, "step": 1},
        "step_signal": "x[n]=1 for n<N/2 else 0 (discontinuity at n=N/2)",
        "operators": {
            "fejer": "H(k)=1-|k|/(K+1) for |k|<=K else 0",
            "sharp": "H(k)=1 for |k|<=K else 0",
            "signed": "H(k)=+1 for |k|<=K else -1",
        },
        "admissibility": "maps every x in [0,1]^N back into [0,1]^N (entrywise)",
        "constructive_witness": "if any kernel coefficient is negative then choose x_neg=1 on negative entries to force y(0)<0; choose x_pos=1 on positive entries to force y(0)>1",
    }
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship demo: Step tests do not certify admissibility.")
    parser.add_argument("--out", type=str, default="demo_out", help="Output directory (will be created).")
    parser.add_argument("--N", type=int, default=2048, help="Signal length (even). Default 2048.")
    parser.add_argument("--K_primary", type=int, default=511, help="Primary cutoff to highlight in kernel triptych.")
    args = parser.parse_args()

    apply_style()

    # Paper title (strategic; do NOT refer to any internal numbering)
    PAPER_TITLE = "Step Tests Do Not Certify Admissibility"

    N = int(args.N)
    if N % 2 != 0:
        raise ValueError("N must be even.")
    K_primary = int(args.K_primary)
    K_ill = N // 2 - 1
    if not (1 <= K_primary <= K_ill):
        raise ValueError(f"K_primary must satisfy 1 <= K_primary <= Nyquist−1 ({K_ill}).")

    out_root = Path(args.out).resolve()
    fig_dir = out_root / "figures"
    data_dir = out_root / "data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Spec hash
    spec = build_spec(PAPER_TITLE, N, K_primary)
    spec_sha256 = spec_hash(spec)

    # --- Environment header
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print("=" * 100)
    print(f"FLAGSHIP DEMO — {PAPER_TITLE}")
    print("=" * 100)
    print(f"UTC time : {now_utc}")
    print(f"Python   : {sys.version.split()[0]}")
    print(f"Platform : {platform.platform()}")
    print(f"Deps     : numpy={np.__version__}, matplotlib={plt.matplotlib.__version__}")
    print(f"Output   : {str(out_root)}")
    print()
    print("SPEC")
    print("-" * 100)
    print(f"spec_sha256: {spec_sha256}")
    print()

    # --- Stage 1: sweep K
    print("=" * 100)
    print("STAGE 1 — Sweep K and compute: step overshoot + worst-case LB + kernel negativity")
    print("=" * 100)

    K_sweep = np.arange(1, K_ill + 1, dtype=int)

    x_step = step_signal(N)
    X_step = np.fft.fft(x_step)

    curves: Dict[str, Dict[str, np.ndarray]] = {}
    for op in ["fejer", "sharp", "signed"]:
        step_out = np.empty_like(K_sweep, dtype=np.float64)
        worst_lb = np.empty_like(K_sweep, dtype=np.float64)
        kmin = np.empty_like(K_sweep, dtype=np.float64)
        neg_sum = np.empty_like(K_sweep, dtype=np.float64)

        for i, K in enumerate(K_sweep):
            m = compute_metrics_for_K(N, int(K), op, X_step)
            step_out[i] = m.step_out
            worst_lb[i] = m.worst_lb
            kmin[i] = m.kmin
            neg_sum[i] = m.neg_sum

        curves[op] = {
            "step_out": step_out,
            "worst_lb": worst_lb,
            "kmin": kmin,
            "neg_sum": neg_sum,
        }

        # concise one-line summary
        print(f"{op:>6s}:  min(kmin)={kmin.min():+.3e}  max(step_out)={step_out.max():.3e}  "
              f"LB@Nyquist−1={worst_lb[-1]:.6f}")

    # --- Stage 2: key points (K_primary and illusion point)
    print()
    print("=" * 100)
    print("STAGE 2 — Key points and constructive witnesses")
    print("=" * 100)

    illusion: Dict[str, Metrics] = {}
    for op in ["sharp", "signed"]:
        illusion[op] = compute_metrics_for_K(N, K_ill, op, X_step)

    primary_metrics = {op: compute_metrics_for_K(N, K_primary, op, X_step) for op in ["fejer", "sharp", "signed"]}

    # verify witness identity at illusion point (tight numerical)
    # y_neg(0) should equal neg_sum, y_pos(0) should equal pos_sum by construction.
    # We'll compute explicitly via filtering the witness vectors as a proof point.
    witness_checks = {}
    for op in ["sharp", "signed"]:
        if op == "sharp":
            H = H_sharp(N, K_ill)
        else:
            H = H_signed(N, K_ill)
        h = kernel_from_H(H)
        row0 = row0_coeffs_from_kernel(h)
        xneg, xpos = witness_vectors_from_row0(row0)
        yneg0 = float(apply_filter(H, xneg)[0])
        ypos0 = float(apply_filter(H, xpos)[0])
        witness_checks[op] = {
            "yneg0": yneg0,
            "ypos0": ypos0,
            "neg_sum": float(row0[row0 < -1e-15].sum()),
            "pos_sum": float(row0[row0 >  1e-15].sum()),
        }

    # Print key tables
    def fmt(m: Metrics) -> str:
        return (f"K={m.K:4d}  step_out={m.step_out: .3e}  "
                f"worst_LB={m.worst_lb: .6f}  kmin={m.kmin: .6f}  "
                f"neg_sum={m.neg_sum: .6f}  neg_count={m.neg_count}")

    print("Primary kernel snapshots (for mechanism figure):")
    for op in ["fejer", "sharp", "signed"]:
        print(f"  {op:>6s}: {fmt(primary_metrics[op])}")

    print("\nIllusion point (Nyquist−1): step looks perfect but worst-case LB remains > 0")
    for op in ["sharp", "signed"]:
        print(f"  {op:>6s}: {fmt(illusion[op])}")

    # --- Stage 3: gates (certificates)
    print()
    print("=" * 100)
    print("GATES (certificates)")
    print("=" * 100)

    gates = {}

    # Gate A: mass preservation (sum of kernel coefficients = 1)
    gate_mass = (
        abs(primary_metrics["fejer"].sum_kernel - 1.0) < 1e-12 and
        abs(primary_metrics["sharp"].sum_kernel - 1.0) < 1e-12 and
        abs(primary_metrics["signed"].sum_kernel - 1.0) < 1e-12
    )
    gates["mass_preserving"] = gate_mass
    print(f"{'PASS' if gate_mass else 'FAIL'}  Gate A: sum(kernel)=1 at K_primary (all operators)")

    # Gate B: Fejér is admissible (nonnegative kernel for all K)
    gate_fejer = bool(np.all(curves["fejer"]["kmin"] >= -1e-12) and np.all(curves["fejer"]["worst_lb"] <= 1e-12))
    gates["fejer_admissible"] = gate_fejer
    print(f"{'PASS' if gate_fejer else 'FAIL'}  Gate B: Fejér kernel nonnegative for all K (LB≈0)")

    # Gate C: Sharp cutoff is non-admissible for every K>=1 (kernel has negativity)
    gate_sharp = bool(np.all(curves["sharp"]["kmin"] < -1e-6) and np.all(curves["sharp"]["worst_lb"] > 1e-3))
    gates["sharp_nonadmissible"] = gate_sharp
    print(f"{'PASS' if gate_sharp else 'FAIL'}  Gate C: Sharp cutoff always has negativity (LB>0)")

    # Gate D: Signed cutoff is non-admissible for every K>=1 (kernel has negativity)
    gate_signed = bool(np.all(curves["signed"]["kmin"] < -1e-6) and np.all(curves["signed"]["worst_lb"] > 1e-3))
    gates["signed_nonadmissible"] = gate_signed
    print(f"{'PASS' if gate_signed else 'FAIL'}  Gate D: Signed cutoff always has negativity (LB>0)")

    # Gate E: Illusion point — step overshoot ~ 0 but worst-case LB bounded away from 0
    gate_illusion = (
        illusion["sharp"].step_out <= 1e-10 and illusion["signed"].step_out <= 1e-10 and
        illusion["sharp"].worst_lb >= 0.25 and illusion["signed"].worst_lb >= 0.75
    )
    gates["illusion_point"] = gate_illusion
    print(f"{'PASS' if gate_illusion else 'FAIL'}  Gate E: Nyquist−1 illusion (step_out≈0 but LB still large)")

    # Gate F: witness identity checks (explicitly filtered x_neg/x_pos)
    gate_wit = True
    for op in ["sharp", "signed"]:
        d = witness_checks[op]
        if not (abs(d["yneg0"] - d["neg_sum"]) < 1e-12 and abs(d["ypos0"] - d["pos_sum"]) < 1e-12):
            gate_wit = False
    gates["witness_identity"] = gate_wit
    print(f"{'PASS' if gate_wit else 'FAIL'}  Gate F: witness outputs equal signed kernel mass partitions (tight)")

    gates_pass = all(gates.values())
    print()
    print(f"{'PASS' if gates_pass else 'FAIL'}  ALL GATES")

    # --- Determinism hash (computed BEFORE rendering figures so it can be embedded in figure footers)
    arrays_for_hash: Dict[str, np.ndarray] = {}
    for _op in ["fejer", "sharp", "signed"]:
        for _k, _a in curves[_op].items():
            arrays_for_hash[f"{_op}.{_k}"] = _a
    scalars_for_hash = {"N": N, "K_primary": K_primary, "K_ill": K_ill, "paper_title": PAPER_TITLE}
    det_sha256 = determinism_hash(spec_sha256, arrays_for_hash, scalars_for_hash)
    hash_tag = f"spec={spec_sha256[:10]}  det={det_sha256[:10]}"

    # --- Stage 4: figures
    print()
    print("=" * 100)
    print("STAGE 4 — Render paper-ready figures (PNG + PDF)")
    print("=" * 100)

    # kernel snapshots for K_primary
    kernels = {
        "fejer": kernel_from_H(H_fejer(N, K_primary)),
        "sharp": kernel_from_H(H_sharp(N, K_primary)),
        "signed": kernel_from_H(H_signed(N, K_primary)),
    }
    kmins_primary = {op: float(row0_coeffs_from_kernel(kernels[op]).min()) for op in kernels.keys()}

    f1 = fig1_hero(fig_dir, PAPER_TITLE, N, K_primary, K_ill, K_sweep, curves, illusion, X_step, hash_tag)
    f2 = fig2_kernel_triptych(fig_dir, PAPER_TITLE, N, K_primary, kernels, kmins_primary, hash_tag)
    f3 = fig3_observed_vs_guaranteed(fig_dir, PAPER_TITLE, K_sweep, curves, K_ill, hash_tag)

    # --- Stage 5: hashes + JSON
    print()
    print("=" * 100)
    print("STAGE 5 — Emit JSON + manifest hashes")
    print("=" * 100)

    # determinism_sha256 already computed earlier (and embedded in figure footers)

    # Write a one-page markdown report (easy review + paper drafting scaffold)
    report_path = out_root / "REPORT.md"
    report_md = f"""# {PAPER_TITLE}

**Claim demonstrated:** *Observed step tests do not certify admissibility.*  
A spectral operator can make a step look perfect (overshoot → 0) while still failing badly on other {{0,1}} inputs.

## Reproducibility
- spec_sha256: `{spec_sha256}`
- determinism_sha256: `{det_sha256}`

## Key settings
- N = {N}
- K_primary = {K_primary}
- Nyquist−1 = {K_ill}

## Key facts (constructive)
- **Fejér (Cesàro)** kernel is nonnegative ⇒ admissible on [0,1]^N.
- **Sharp** and **Signed** cutoffs have negative kernel coefficients for every K≥1 ⇒ non-admissible.
- At **Nyquist−1**, step overshoot ≈ 0 (looks perfect), but guaranteed worst-case violation stays bounded away from 0.

## Figures
### Fig 1 — Hero (illusion + constructive boundary)
![](figures/Fig1_HERO.png)

### Fig 2 — Mechanism (kernel nonnegativity in real space)
![](figures/Fig2_KernelTriptych.png)

### Fig 3 — Phase plot (observed overshoot vs guaranteed failure)
![](figures/Fig3_ObservedVsGuaranteed.png)

## Data for the paper
- `data/demo_data.json` contains the full K-sweep curves, key-point summaries, and hashes.
- `MANIFEST.json` contains sha256 hashes of every artifact.
"""
    report_path.write_text(report_md, encoding="utf-8")

    # file hashes
    artifacts = {}
    for p in [
        fig_dir / "Fig1_HERO.png",
        fig_dir / "Fig1_HERO.pdf",
        fig_dir / "Fig2_KernelTriptych.png",
        fig_dir / "Fig2_KernelTriptych.pdf",
        fig_dir / "Fig3_ObservedVsGuaranteed.png",
        fig_dir / "Fig3_ObservedVsGuaranteed.pdf",
    ]:
        artifacts[str(p.relative_to(out_root))] = sha256_file(p)

    artifacts[str(report_path.relative_to(out_root))] = sha256_file(report_path)

    # write demo JSON (paper data)
    demo_data = {
        "paper_title": PAPER_TITLE,
        "spec": spec,
        "spec_sha256": spec_sha256,
        "determinism_sha256": det_sha256,
        "gates": gates,
        "gates_pass": gates_pass,
        "key_points": {
            "K_primary": K_primary,
            "K_ill": K_ill,
            "primary_metrics": {op: dataclasses.asdict(primary_metrics[op]) for op in primary_metrics},
            "illusion_metrics": {op: dataclasses.asdict(illusion[op]) for op in illusion},
            "witness_checks": witness_checks,
        },
        "curves": {
            "K": K_sweep.tolist(),
            "fejer": {k: curves["fejer"][k].tolist() for k in curves["fejer"]},
            "sharp": {k: curves["sharp"][k].tolist() for k in curves["sharp"]},
            "signed": {k: curves["signed"][k].tolist() for k in curves["signed"]},
        },
        "artifacts": artifacts,
    }

    data_path = data_dir / "demo_data.json"
    demo_data["artifacts"] = artifacts
    data_path.write_text(canonical_json_dumps(demo_data), encoding="utf-8")
    artifacts[str(data_path.relative_to(out_root))] = sha256_file(data_path)

    manifest = {
        "paper_title": PAPER_TITLE,
        "spec_sha256": spec_sha256,
        "determinism_sha256": det_sha256,
        "files": artifacts,
    }
    manifest_path = out_root / "MANIFEST.json"
    manifest_path.write_text(canonical_json_dumps(manifest), encoding="utf-8")


    print(f"determinism_sha256: {det_sha256}")
    print("Saved artifacts (sha256):")
    for k in sorted(artifacts.keys()):
        print(f"  {k:40s}  {artifacts[k][:16]}…")

    print()
    print("=" * 100)
    print("FINAL VERDICT")
    print("=" * 100)
    print(f"{'PASS' if gates_pass else 'FAIL'}  Flagship demo verified (self-generated figures + reproducible hashes)")
    print()

    # Also print a compact JSON payload to stdout for easy copy/paste into logs
    print("BEGIN_JSON")
    compact = {
        "paper_title": PAPER_TITLE,
        "spec_sha256": spec_sha256,
        "determinism_sha256": det_sha256,
        "gates_pass": gates_pass,
        "key_illusion": {
            "K": K_ill,
            "sharp": {"step_out": illusion["sharp"].step_out, "worst_lb": illusion["sharp"].worst_lb, "kmin": illusion["sharp"].kmin},
            "signed": {"step_out": illusion["signed"].step_out, "worst_lb": illusion["signed"].worst_lb, "kmin": illusion["signed"].kmin},
        },
        "artifacts": {k: v for k, v in artifacts.items() if k.endswith(".png")},
    }
    print(canonical_json_dumps(compact))
    print("END_JSON")


if __name__ == "__main__":
    main()