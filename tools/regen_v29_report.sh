#!/usr/bin/env bash
set -euo pipefail

# GUM v29 report regeneration helper.
# - Preferred path: consume smoketest ledger (artifacts/smoketest_all/summary.json).
# - Fallback: build the PDF without ledger data (report will mark run data as missing).

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$ROOT"

LEDGER="artifacts/smoketest_all/summary.json"

# Ensure reportlab is present (the report generator hard-requires it)
python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("reportlab") else 1)
PY
if [[ $? -ne 0 ]]; then
  echo "[regen_v29] Installing reportlab..."
  python -m pip install reportlab
fi

if [[ ! -f "$LEDGER" ]]; then
  echo "[regen_v29] Smoketest ledger missing at $LEDGER"
  if [[ -f "tools/run_all_demos_smoketest.py" ]]; then
    echo "[regen_v29] Running smoketest to populate ledger + hashes..."
    python tools/run_all_demos_smoketest.py
  else
    echo "[regen_v29] WARNING: tools/run_all_demos_smoketest.py not found; continuing without ledger."
  fi
fi

echo "[regen_v29] Building GUM v29 report (ledger: $LEDGER)"
python gum/gum_report_generator_v29.py --ledger "$LEDGER"

echo "[regen_v29] Done."
