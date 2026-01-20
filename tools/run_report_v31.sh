#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"

# Optional: user can provide an explicit bundle dir.
# If omitted, we create a fresh timestamped one under /tmp.
BUNDLE_DIR="${1:-}"
if [[ "${BUNDLE_DIR:-}" == "" || "${BUNDLE_DIR:-}" == "-" ]]; then
  BUNDLE_DIR="/tmp/mari_bundle_v31_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "$BUNDLE_DIR"

echo "==> ROOT      : $ROOT"
echo "==> BUNDLE_DIR: $BUNDLE_DIR"

# ---------------------------------------------------------------------
# 1) BUILD BUNDLE (auto-detect the correct bundler)
# ---------------------------------------------------------------------
if [[ -x "$ROOT/tools/make_report_master.sh" ]]; then
  echo "==> Bundler: tools/make_report_master.sh"

  # If the script supports --bundle-dir, use it. Otherwise run it and copy out the produced bundle.
  if "$ROOT/tools/make_report_master.sh" --help 2>/dev/null | grep -q -- "--bundle-dir"; then
    "$ROOT/tools/make_report_master.sh" --bundle-dir "$BUNDLE_DIR"
  else
    "$ROOT/tools/make_report_master.sh"

    # Try to locate the newest bundle output and copy it into our BUNDLE_DIR
    CAND="$(ls -dt \
      "$ROOT"/audits/bundles/* \
      "$ROOT"/bundle* \
      "$ROOT"/vendored_bundle* \
      2>/dev/null | head -n 1 || true)"

    if [[ -z "${CAND:-}" ]]; then
      echo "ERROR: Bundler ran but its output bundle directory could not be located." >&2
      echo "Searched: audits/bundles/*, bundle*, vendored_bundle*" >&2
      exit 2
    fi

    echo "==> Found bundler output: $CAND"
    rsync -a --delete "$CAND"/ "$BUNDLE_DIR"/
  fi

elif [[ -f "$ROOT/audits/gum_bundle_v31.py" ]]; then
  echo "==> Bundler: python audits/gum_bundle_v31.py"
  python "$ROOT/audits/gum_bundle_v31.py" --bundle-dir "$BUNDLE_DIR"

elif [[ -f "$ROOT/audits/gum_bundle_v30.py" ]]; then
  echo "==> Bundler: python audits/gum_bundle_v30.py"
  python "$ROOT/audits/gum_bundle_v30.py" --bundle-dir "$BUNDLE_DIR"

else
  echo "ERROR: No known bundler found." >&2
  echo "Expected one of:" >&2
  echo "  - tools/make_report_master.sh" >&2
  echo "  - audits/gum_bundle_v31.py" >&2
  echo "  - audits/gum_bundle_v30.py" >&2
  exit 3
fi

# Minimal sanity check that we didn't produce an empty bundle.
echo "==> Bundle contents (top-level):"
ls -la "$BUNDLE_DIR" | head -n 50

# ---------------------------------------------------------------------
# 2) RUN REPORT GENERATOR v31 ON THAT EXACT BUNDLE
# ---------------------------------------------------------------------
echo "==> Running: gum/gum_report_generator_v31.py --bundle-dir $BUNDLE_DIR"
python "$ROOT/gum/gum_report_generator_v31.py" --bundle-dir "$BUNDLE_DIR"

echo "==> DONE (v31 ran against the bundled snapshot)"
echo "==> Bundle dir retained at: $BUNDLE_DIR"
