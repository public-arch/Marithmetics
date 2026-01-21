#!/usr/bin/env python3
import json
from pathlib import Path

def _get_nested(d, *keys, default=None):
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur

def _run_camb_tt(H0, ombh2, omch2, tau, ns, As, lmax=2500):
    import camb
    import numpy as np

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=float(H0), ombh2=float(ombh2), omch2=float(omch2), mnu=0.06, omk=0.0, tau=float(tau))
    pars.InitPower.set_params(As=float(As), ns=float(ns), r=0.0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    tot = powers["total"]  # TT, EE, BB, TE
    ell = np.arange(tot.shape[0])
    tt = tot[:, 0]
    return ell, tt

def main():
    here = Path(__file__).resolve().parent
    results_path = here / "bb36_master_results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}. Run demo.py first.")

    outdir = here / "_artifacts"
    outdir.mkdir(exist_ok=True)

    data = json.loads(results_path.read_text(encoding="utf-8"))

    # GUM/BB36 params (from demo output JSON)
    H0   = _get_nested(data, "cosmo_structural", "H0", default=_get_nested(data, "H0"))
    ombh2= _get_nested(data, "cosmo_structural", "ombh2", default=_get_nested(data, "ombh2"))
    omch2= _get_nested(data, "cosmo_structural", "omch2", default=_get_nested(data, "omch2"))
    tau  = _get_nested(data, "cosmo_structural", "tau", default=_get_nested(data, "tau"))
    ns   = _get_nested(data, "cosmo_structural", "n_s", default=_get_nested(data, "n_s"))
    As   = _get_nested(data, "cosmo_structural", "A_s", default=_get_nested(data, "A_s"))

    missing = [k for k,v in [("H0",H0),("ombh2",ombh2),("omch2",omch2),("tau",tau),("n_s",ns),("A_s",As)] if v is None]
    if missing:
        raise SystemExit(f"bb36_master_results.json missing required keys for CAMB: {missing}")

    # Planck 2018 LCDM baseline (evaluation-only; hardcoded constants)
    # These are standard best-fit-ish values; they must never feed upstream selection.
    planck = {
        "H0": 67.36,
        "ombh2": 0.02237,
        "omch2": 0.1200,
        "tau": 0.0544,
        "ns": 0.9649,
        "As": 2.1005e-9,
        "notes": "Planck 2018 LCDM baseline (evaluation-only, hardcoded). Must not feed upstream selection."
    }
    (outdir / "camb_planck_params.json").write_text(json.dumps(planck, indent=2) + "\n", encoding="utf-8")

    # Run CAMB for both
    ellP, ttP = _run_camb_tt(**{k: planck[k] for k in ("H0","ombh2","omch2","tau","ns","As")})
    ellG, ttG = _run_camb_tt(H0, ombh2, omch2, tau, ns, As)

    # Compare on a safe ell window
    import numpy as np
    ell_min, ell_max = 2, 2000
    mask = (ellP >= ell_min) & (ellP <= ell_max)
    denom = np.maximum(ttP[mask], 1e-12)
    rel = np.abs(ttG[mask] - ttP[mask]) / denom

    rms = float(np.sqrt(np.mean(rel**2)))
    mx  = float(np.max(rel))

    metrics = {
        "ell_range": [ell_min, ell_max],
        "RMS_Delta_TT_over_TT": rms,
        "max_Delta_TT_over_TT": mx,
        "thresholds": {"RMS": 0.05, "max": 0.15},
        "pass": {"RMS": rms < 0.05, "max": mx < 0.15},
    }
    (outdir / "camb_planck_vs_gum_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # Plot overlay
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.plot(ellP[mask], ttP[mask], label="Planck 2018 LCDM")
    plt.plot(ellG[mask], ttG[mask], "--", label="GUM (BB-36)")
    plt.xlabel("ell")
    plt.ylabel("C_ell^TT [muK^2]")
    plt.title("CMB TT Power Spectrum: Planck vs GUM")
    plt.legend()
    out_png = outdir / "camb_planck_vs_gum_overlay.png"
    out_png_canon = outdir / "camb_overlay.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    try:
        import shutil
        shutil.copy2(out_png, out_png_canon)
    except Exception:
        pass
    plt.close(fig)

    (outdir / "camb_overlay_note.txt").write_text(
        "CAMB overlays are evaluation-only.\n"
        "This file compares a hardcoded Planck 2018 LCDM baseline vs BB36-derived params.\n"
        "These overlays must never feed upstream selection.\n",
        encoding="utf-8"
    )

    print("CAMB overlay written:", out_png)
    print("CAMB metrics written:", outdir / "camb_planck_vs_gum_metrics.json")

if __name__ == "__main__":
    main()
