#!/usr/bin/env bash
set -e

cd "$(git rev-parse --show-toplevel)"

echo "==> Sync main"
git checkout main >/dev/null
git pull --rebase origin main >/dev/null || true

echo "==> Run vendored bundle"
python -u -m audits.gum_bundle_v30 --timeout 600 --vendor-artifacts
BUNDLE_DIR="$(ls -1d audits/bundles/GUM_BUNDLE_v30_* | sort | tail -n 1)"
echo "BUNDLE_DIR=$BUNDLE_DIR"

echo "==> Generate report"
python gum/gum_report_generator_v31.py --bundle-dir "$BUNDLE_DIR"
REPORT_PDF="$(ls -1t gum/reports/GUM_Report_v31_*.pdf | head -n 1)"
REPORT_MANIFEST="${REPORT_PDF}.manifest.json"
echo "REPORT_PDF=$REPORT_PDF"

echo "==> Build master archive"
ARCHIVE_NAME="MASTER_ARCHIVE_${BUNDLE_DIR##*/}.zip"
rm -f "$ARCHIVE_NAME" "$ARCHIVE_NAME.sha256" MASTER_RUN_MANIFEST.json

python - <<PY
import hashlib, json
from pathlib import Path

bundle_dir = Path("$BUNDLE_DIR")
report_pdf = Path("$REPORT_PDF")
report_manifest = Path("$REPORT_MANIFEST")

def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()

run = {
  "bundle_dir": str(bundle_dir),
  "bundle_json_sha256": sha256(bundle_dir/"bundle.json") if (bundle_dir/"bundle.json").exists() else None,
  "runs_json_sha256": sha256(bundle_dir/"runs.json") if (bundle_dir/"runs.json").exists() else None,
  "values_jsonl_sha256": sha256(bundle_dir/"values.jsonl") if (bundle_dir/"values.jsonl").exists() else None,
  "artifacts_index_sha256": sha256(bundle_dir/"artifacts_index.json") if (bundle_dir/"artifacts_index.json").exists() else None,
  "report_pdf": str(report_pdf),
  "report_pdf_sha256": sha256(report_pdf),
  "report_manifest": str(report_manifest),
  "report_manifest_sha256": sha256(report_manifest) if report_manifest.exists() else None,
}
Path("MASTER_RUN_MANIFEST.json").write_text(json.dumps(run, indent=2) + "\\n", encoding="utf-8")
PY

zip -r "$ARCHIVE_NAME" "$BUNDLE_DIR" "$REPORT_PDF" "$REPORT_MANIFEST" MASTER_RUN_MANIFEST.json >/dev/null

python - <<PY
import hashlib, pathlib
p = pathlib.Path("$ARCHIVE_NAME")
h = hashlib.sha256(p.read_bytes()).hexdigest()
pathlib.Path("$ARCHIVE_NAME.sha256").write_text(h+"\\n", encoding="utf-8")
print("SHA256:", h)
PY

mkdir -p gum/master_archives
mv "$ARCHIVE_NAME" "$ARCHIVE_NAME.sha256" gum/master_archives/

echo "==> Archive saved to gum/master_archives/"
ls -lah gum/master_archives | tail -n 5
