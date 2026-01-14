#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    here = Path(__file__).resolve().parent
    results_path = here / "bb36_master_results.json"
    if not results_path.exists():
        raise SystemExit(f"Missing {results_path}. Run demo.py first.")

    outdir = here / "_artifacts"
    outdir.mkdir(exist_ok=True)

    data = json.loads(results_path.read_text(encoding="utf-8"))

    def get(*keys, default=None):
        cur = data
        for k in keys:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return default
        return cur

    # pull params (adjust if your JSON nesting differs)
    H0   = get("cosmo_structural", "H0", default=get("H0"))
    ombh2= get("cosmo_structural", "ombh2", default=get("ombh2"))
    omch2= get("cosmo_structural", "omch2", default=get("omch2"))
    tau  = get("cosmo_structural", "tau", default=get("tau"))
    ns   = get("cosmo_structural", "n_s", default=get("n_s"))
    As   = get("cosmo_structural", "A_s", default=get("A_s"))

    missing = [k for k,v in [("H0",H0),("ombh2",ombh2),("omch2",omch2),("tau",tau),("n_s",ns),("A_s",As)] if v is None]
    if missing:
        raise SystemExit(f"bb36_master_results.json missing required keys for CAMB: {missing}")

    import camb
    import numpy as np
    import matplotlib.pyplot as plt

    pars = camb.CAMBparams()
    pars.set_cosmology(H0=float(H0), ombh2=float(ombh2), omch2=float(omch2), mnu=0.06, omk=0.0, tau=float(tau))
    pars.InitPower.set_params(As=float(As), ns=float(ns), r=0.0)
    pars.set_for_lmax(2500, lens_potential_accuracy=0)

    results = camb.get_results(pars)
    powers = results.get_cmb_power_spectra(pars, CMB_unit="muK")
    tot = powers["total"]
    ell = np.arange(tot.shape[0])

    # CSV spectrum (audit-friendly)
    spec_csv = outdir / "camb_tt_spectrum.csv"
    with spec_csv.open("w", encoding="utf-8") as f:
        f.write("ell,TT_uK2\n")
        for l, tt in zip(ell.tolist(), tot[:, 0].tolist()):
            f.write(f"{l},{tt}\n")

    # params used (audit-friendly)
    (outdir / "camb_params.json").write_text(json.dumps({
        "H0": float(H0),
        "ombh2": float(ombh2),
        "omch2": float(omch2),
        "tau": float(tau),
        "n_s": float(ns),
        "A_s": float(As),
        "notes": "Evaluation-only CAMB run. Must not feed upstream selection."
    }, indent=2) + "\n", encoding="utf-8")

    # Plot
    fig = plt.figure()
    plt.plot(ell[2:2000], tot[2:2000, 0])
    plt.xlabel("ℓ")
    plt.ylabel("TT [μK²]")
    plt.title("CAMB TT spectrum (evaluation-only) from BB36 parameters")
    out_png = outdir / "camb_overlay.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    (outdir / "camb_overlay_note.txt").write_text(
        "CAMB overlay is evaluation-only.\n"
        "Inputs come from bb36_master_results.json produced by DEMO-36.\n"
        "This overlay must never feed upstream selection.\n",
        encoding="utf-8"
    )

    print("CAMB overlay written:")
    print(" ", out_png)
    print(" ", spec_csv)

if __name__ == "__main__":
    main()
