#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import re, json
from pathlib import Path

GUM = Path("gum")
ARCHIVE_CANDIDATES = [Path("zz_archive"), Path("archive")]
ATLAS_CANDIDATES = [Path("atlas_substrate_visualization"), Path("atlas")]

def find_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

ARCHIVE = find_existing(ARCHIVE_CANDIDATES)
ATLAS = find_existing(ATLAS_CANDIDATES)

TEXT_EXTS = {".py", ".md", ".txt", ".json", ".yaml", ".yml", ".tex"}
NEEDLE_GROUPS = {
    "camb": [r"\bcamb\b", r"CAMB", r"camb_", r"\bcamb\.py\b"],
    "bb36": [r"\bBB-?36\b", r"bb36", r"Big\s*Bang", r"Fej[eé]r", r"Fejer"],
    "legacy_paths": [r"\bsm/", r"\bgr/", r"\bcosmo/", r"\bdemos/", r"\barchive/", r"\bzz_archive/"],
    "plots_results": [r"\.png\b", r"\.pdf\b", r"_results\.json\b", r"results\.json\b"],
}

PATHLIKE = re.compile(r'([A-Za-z0-9_\-./]+?\.(?:png|pdf|json|csv|txt|md|yaml|yml|tex|py))')

def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return p.read_text(encoding="latin-1", errors="replace")

def scan_tree(root: Path):
    hits = {k: [] for k in NEEDLE_GROUPS}
    path_refs = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        txt = read_text(p)

        for group, patterns in NEEDLE_GROUPS.items():
            for pat in patterns:
                if re.search(pat, txt):
                    hits[group].append(str(p))
                    break

        for m in PATHLIKE.finditer(txt):
            ref = m.group(1)
            if ref.startswith("http"):
                continue
            path_refs.append({"file": str(p), "ref": ref})

    return hits, path_refs

def exists_any(ref: str) -> bool:
    candidates = [Path(ref), GUM / ref]
    if ARCHIVE:
        candidates.append(ARCHIVE / ref)
    if ATLAS:
        candidates.append(ATLAS / ref)
    for c in candidates:
        if c.exists():
            return True
    return False

def main():
    if not GUM.exists():
        raise SystemExit("Missing gum/ directory")

    gum_hits, gum_refs = scan_tree(GUM)
    legacy_hits = {}
    if ARCHIVE:
        legacy_hits, _ = scan_tree(ARCHIVE)

    missing = []
    for item in gum_refs:
        ref = item["ref"]
        if not exists_any(ref):
            missing.append(item)

    report = {
        "gum_dir": str(GUM),
        "archive_dir": str(ARCHIVE) if ARCHIVE else None,
        "atlas_dir": str(ATLAS) if ATLAS else None,
        "hits": {"gum": gum_hits, "archive": legacy_hits if ARCHIVE else None},
        "reference_counts": {"gum_path_refs": len(gum_refs), "missing_refs_from_gum": len(missing)},
        "missing_refs_sample": missing[:300],
    }

    out_dir = Path("artifacts") / "gum_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "gum_v28_dependency_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = []
    md.append("# GUM v28 Dependency Audit\n")
    md.append(f"- archive detected: `{report['archive_dir']}`")
    md.append(f"- atlas detected: `{report['atlas_dir']}`\n")

    md.append("## Signal hits in gum/\n")
    for k, v in gum_hits.items():
        md.append(f"- **{k}**: {len(v)} files")

    md.append("\n## Missing referenced files (from gum/)\n")
    md.append(f"Total missing refs: **{len(missing)}**\n")
    for item in missing[:80]:
        md.append(f"- `{item['ref']}`  (referenced by `{item['file']}`)")

    (out_dir / "gum_v28_dependency_audit.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("Wrote:")
    print(" ", out_dir / "gum_v28_dependency_audit.json")
    print(" ", out_dir / "gum_v28_dependency_audit.md")
    print(f"Missing refs (from gum): {len(missing)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
