# CAMB Expected Assets (Evaluation-Only)

These assets are evaluation-only overlays. They must never feed upstream selection.

DEMO-36 produces:
- _artifacts/camb_overlay.png         (generated TT spectrum plot)
- _artifacts/camb_tt_spectrum.csv     (ell,TT_uK2)
- _artifacts/camb_params.json         (H0, ombh2, omch2, tau, n_s, A_s + note)
- _artifacts/camb_overlay_note.txt    (evaluation-only disclaimer)
- bb36_master_plot.png               (from demo)
- bb36_master_results.json           (from demo)

Bundler must ingest nested artifacts so the report can display them.
