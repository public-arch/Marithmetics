#!/usr/bin/env bash
set -euo pipefail

ts="$(date -u +%Y%m%dT%H%M%SZ)"
out="GUM_UPGRADE_INPUTS_${ts}.zip"

# Detect archive folder name
ARCHIVE_DIR=""
if [ -d "zz_archive" ]; then ARCHIVE_DIR="zz_archive"; fi
if [ -z "${ARCHIVE_DIR}" ] && [ -d "archive" ]; then ARCHIVE_DIR="archive"; fi

# Detect atlas folder name
ATLAS_DIR=""
if [ -d "atlas_substrate_visualization" ]; then ATLAS_DIR="atlas_substrate_visualization"; fi
if [ -z "${ATLAS_DIR}" ] && [ -d "atlas" ]; then ATLAS_DIR="atlas"; fi

echo "Collecting inputs..."
echo "  gum/: $( [ -d gum ] && echo OK || echo MISSING )"
echo "  ${ARCHIVE_DIR:-<none>}/: $( [ -n "${ARCHIVE_DIR}" ] && echo OK || echo MISSING )"
echo "  ${ATLAS_DIR:-<none>}/: $( [ -n "${ATLAS_DIR}" ] && echo OK || echo MISSING )"
echo "  demos/: $( [ -d demos ] && echo OK || echo MISSING )"
echo "  tools/demo_map.yaml: $( [ -f tools/demo_map.yaml ] && echo OK || echo MISSING )"

includes=()

# Core trees
[ -d gum ] && includes+=(gum)
[ -n "${ARCHIVE_DIR}" ] && includes+=("${ARCHIVE_DIR}")
[ -n "${ATLAS_DIR}" ] && includes+=("${ATLAS_DIR}")
[ -d demos ] && includes+=(demos)
[ -d audits ] && includes+=(audits)
[ -d tools ] && includes+=(tools)

# Root files for planning/repro
for f in requirements.txt .gitignore __init__.py ROADMAP.md; do
  [ -f "$f" ] && includes+=("$f")
done

# Optional loose artifacts at root (safe even if nothing matches)
shopt -s nullglob
for f in *.png *_results.json *BUNDLE*.zip *.pdf; do
  [ -f "$f" ] && includes+=("$f")
done
shopt -u nullglob

if [ ${#includes[@]} -eq 0 ]; then
  echo "Nothing found to bundle. Are you in the repo root?"
  exit 2
fi

echo "Creating bundle: $out"
zip -r "$out" "${includes[@]}" \
  -x "**/__pycache__/**" \
  -x "**/.pytest_cache/**" \
  -x "**/.mypy_cache/**" \
  -x "**/.DS_Store" \
  -x "**/.ipynb_checkpoints/**"

echo "DONE: $out"
