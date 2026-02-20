#!/usr/bin/env python3
# stdlib-only • deterministic • AoR-local
# Generates URL_MAP.md + URL_MAP.json inside the AoR root.
from __future__ import annotations

from pathlib import Path
import os, json, hashlib

def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def main():
    # Inputs
    aor_root = Path(os.environ["AOR_ROOT"]).resolve()
    repo_rel = os.environ["AOR_REPO_REL"].strip("/")     # e.g. gum/authority_archive/AOR_...
    repo = os.environ.get("AOR_REPO", "https://github.com/public-arch/Marithmetics").rstrip("/")
    ref = os.environ.get("AOR_REF", "main")

    def blob_url(rel: str) -> str:
        rel = rel.lstrip("/")
        return f"{repo}/blob/{ref}/{repo_rel}/{rel}"

    def tree_url(rel: str) -> str:
        rel = rel.lstrip("/")
        return f"{repo}/tree/{ref}/{repo_rel}/{rel}"

    # Inventory (repo-relative paths under the AoR folder)
    skip = {"URL_MAP.md", "URL_MAP.json"}
    files = []
    for p in sorted(aor_root.rglob("*")):
        if p.is_dir():
            continue
        if p.name in skip:
            continue
        files.append(p.relative_to(aor_root).as_posix())

    # Entry points
    report_pdf = None
    report_dir = aor_root / "report"
    if report_dir.exists():
        pdfs = sorted(report_dir.glob("*.pdf"))
        if pdfs:
            report_pdf = ("report/" + pdfs[0].name)

    bundle_dir = None
    for p in sorted(aor_root.glob("GUM_BUNDLE_*")):
        if p.is_dir():
            bundle_dir = p.name
            break

    master_zip = None
    mzs = sorted(aor_root.glob("MARI_MASTER_RELEASE_*.zip"))
    if mzs:
        master_zip = mzs[0].name

    master_zip_sha = sha256_file(aor_root / master_zip) if master_zip else None

    entry = {
        "repo": repo,
        "ref": ref,
        "aor_repo_rel": repo_rel,
        "summary_md": "SUMMARY.md" if (aor_root / "SUMMARY.md").exists() else None,
        "report_pdf": report_pdf,
        "report_manifest": (report_pdf + ".manifest.json") if report_pdf and (aor_root / (report_pdf + ".manifest.json")).exists() else None,
        "claim_ledger": "claim_ledger.jsonl" if (aor_root / "claim_ledger.jsonl").exists() else None,
        "runner_transcript": "runner_transcript.txt" if (aor_root / "runner_transcript.txt").exists() else None,
        "bundle_dir": bundle_dir,
        "bundle_sha_file": (bundle_dir + "/bundle_sha256.txt") if bundle_dir and (aor_root / bundle_dir / "bundle_sha256.txt").exists() else None,
        "master_zip": master_zip,
        "master_zip_sha256": master_zip_sha,
        "url_map_script": "mari_make_url_map.py",
    }

    # JSON map
    out_json = {
        "repo": repo,
        "ref": ref,
        "aor_repo_rel": repo_rel,
        "entry_points": entry,
        "files": files,
    }
    (aor_root / "URL_MAP.json").write_text(json.dumps(out_json, indent=2, sort_keys=True), encoding="utf-8")

    # MD map
    lines = []
    lines.append("# AoR URL Map")
    lines.append("")
    lines.append(f"- Repo: {repo}")
    lines.append(f"- Ref: `{ref}`")
    lines.append(f"- AoR path: `{repo_rel}/`")
    if master_zip_sha:
        lines.append(f"- Master zip sha256: `{master_zip_sha}`")
    lines.append("")
    lines.append("## Entry points")
    lines.append("")
    if entry["summary_md"]:
        lines.append(f"- SUMMARY.md: {blob_url(entry['summary_md'])}")
    if entry["report_pdf"]:
        lines.append(f"- Report PDF: {blob_url(entry['report_pdf'])}")
    if entry["report_manifest"]:
        lines.append(f"- Report manifest: {blob_url(entry['report_manifest'])}")
    if entry["claim_ledger"]:
        lines.append(f"- Claim ledger: {blob_url(entry['claim_ledger'])}")
    if entry["runner_transcript"]:
        lines.append(f"- Runner transcript: {blob_url(entry['runner_transcript'])}")
    if entry["bundle_dir"]:
        lines.append(f"- GUM bundle dir: {tree_url(entry['bundle_dir'])}")
    if entry["bundle_sha_file"]:
        lines.append(f"- GUM bundle sha: {blob_url(entry['bundle_sha_file'])}")
    if entry["master_zip"]:
        lines.append(f"- Master release zip: {blob_url(entry['master_zip'])}")
    lines.append(f"- URL map script: {blob_url('mari_make_url_map.py')}")
    lines.append("")
    lines.append("## Full file inventory")
    lines.append("")
    lines.append("> Deterministic listing of all files contained in this AoR folder.")
    lines.append("")
    for rel in files:
        # link common artifact types
        if rel.endswith((".md", ".json", ".jsonl", ".csv", ".pdf", ".png", ".txt", ".zip")):
            lines.append(f"- {rel}: {blob_url(rel)}")
        else:
            lines.append(f"- {rel}")
    (aor_root / "URL_MAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("Wrote URL_MAP.md and URL_MAP.json")

if __name__ == "__main__":
    main()
