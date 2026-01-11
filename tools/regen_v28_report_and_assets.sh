#!/usr/bin/env bash
set -euo pipefail

echo "== v28 regen: legacy links + CAMB deps + report build =="

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
for pkg in ["numpy", "matplotlib", "camb"]:
    try:
        importlib.import_module(pkg)
    except Exception:
        need.append(pkg)
if need:
    print("MISSING:", need)
    sys.exit(3)
print("OK: numpy/matplotlib/camb already installed.")
PY

if [ $? -ne 0 ]; then
  echo "Installing missing packages..."
  python -m pip install --upgrade pip
  python -m pip install numpy matplotlib camb
fi

echo ""
echo "== Running v28 report generator =="
python gum/gum_report_generator_v28.py

echo ""
echo "== Copy key visuals into gum/assets for upgrade planning =="
mkdir -p gum/assets

# v28 writes into gum/GUM_Report/
REPORT_DIR="gum/GUM_Report"
if [ -d "$REPORT_DIR" ]; then
  # Copy CAMB plots if present
  for f in "$REPORT_DIR"/*TT_spectrum*png "$REPORT_DIR"/*Planck*png; do
    [ -f "$f" ] && cp -f "$f" gum/assets/
  done

  # Copy BB36 plot if present
  for f in "$REPORT_DIR"/*BB36*png "$REPORT_DIR"/*big_bang*png; do
    [ -f "$f" ] && cp -f "$f" gum/assets/
  done
fi

# Also copy from gum_camb_output if gum_camb_check creates it
if [ -d "gum_camb_output" ]; then
  for f in gum_camb_output/*TT_spectrum*png; do
    [ -f "$f" ] && cp -f "$f" gum/assets/
  done
fi

echo ""
echo "== What we now have =="
ls -la gum/assets | sed -n '1,200p'
echo ""
echo "Latest report artifacts:"
ls -la gum/GUM_Report | tail -n 20 || true

echo ""
echo "DONE."
