#!/usr/bin/env bash
set -euo pipefail

echo "== v28 regen v2: legacy links + CAMB deps + report build =="

# Detect archive folder
ARCH="zz_archive"
if [ ! -d "$ARCH" ] && [ -d "archive" ]; then ARCH="archive"; fi
if [ ! -d "$ARCH" ]; then
  echo "ERROR: Could not find zz_archive/ or archive/ at repo root."
  exit 2
fi

# Safety: ensure expected legacy modules exist
if [ ! -d "$ARCH/sm" ]; then
  echo "ERROR: Missing $ARCH/sm (needed for v28 generator)."
  exit 2
fi
if [ ! -d "$ARCH/cosmo" ]; then
  echo "ERROR: Missing $ARCH/cosmo (needed for v28 generator)."
  exit 2
fi

cleanup() {
  echo "Cleaning up temporary symlinks (if created)..."
  [ -L "sm" ] && rm -f sm || true
  [ -L "cosmo" ] && rm -f cosmo || true
}
trap cleanup EXIT

# Create temp symlinks ONLY if the paths do not exist already
if [ ! -e "sm" ]; then
  ln -s "$ARCH/sm" sm
  echo "Created symlink: sm -> $ARCH/sm"
else
  echo "NOTE: sm already exists (not symlinking)."
fi

if [ ! -e "cosmo" ]; then
  ln -s "$ARCH/cosmo" cosmo
  echo "Created symlink: cosmo -> $ARCH/cosmo"
else
  echo "NOTE: cosmo already exists (not symlinking)."
fi

echo ""
echo "== Ensuring CAMB dependencies =="
python - << 'PY'
import importlib, sys
need = []
for pkg in ["numpy", "matplotlib", "camb", "reportlab"]:
    try:
        importlib.import_module(pkg)
    except Exception:
        need.append(pkg)
if need:
    print("MISSING:", need)
    sys.exit(3)
print("OK: numpy/matplotlib/camb/reportlab already installed.")
PY

if [ $? -ne 0 ]; then
  echo "Installing missing packages..."
  python -m pip install --upgrade pip
  python -m pip install numpy matplotlib camb reportlab
fi

echo ""
echo "== Running v28 report generator (module mode so imports work) =="
python -m gum.gum_report_generator_v28

echo ""
echo "== Locate report outputs =="
# Find any report directory or PDF produced
REPORT_DIR="$(find gum -maxdepth 3 -type d -iname 'GUM_Report*' -print | head -n 1 || true)"
PDFS="$(find gum -maxdepth 3 -type f -iname '*.pdf' -print || true)"

echo "Detected report dir: ${REPORT_DIR:-<none>}"
echo "PDF(s) under gum/:"
echo "${PDFS:-<none>}"

echo ""
echo "== Copy key visuals into gum/assets for upgrade planning =="
mkdir -p gum/assets

if [ -n "${REPORT_DIR:-}" ] && [ -d "${REPORT_DIR}" ]; then
  # Copy CAMB plots if present
  shopt -s nullglob
  for f in "${REPORT_DIR}"/*TT_spectrum*png "${REPORT_DIR}"/*Planck*png; do
    cp -f "$f" gum/assets/
  done
  # Copy BB36 plot if present
  for f in "${REPORT_DIR}"/*BB36*png "${REPORT_DIR}"/*big_bang*png; do
    cp -f "$f" gum/assets/
  done
  shopt -u nullglob
fi

# Also copy from gum_camb_output if created
if [ -d "gum_camb_output" ]; then
  shopt -s nullglob
  for f in gum_camb_output/*TT_spectrum*png; do
    cp -f "$f" gum/assets/
  done
  shopt -u nullglob
fi

echo ""
echo "== gum/assets now contains =="
ls -la gum/assets | sed -n '1,220p'

echo ""
echo "DONE."
