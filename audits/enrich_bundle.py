#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, List, Any


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_repo_root(start: Path) -> Path:
    # Walk up until we find "demos/" (repo root)
    cur = start.resolve()
    for _ in range(12):
        if (cur / "demos").exists() and (cur / "audits").exists():
            return cur
        cur = cur.parent
    return start.resolve()


def enrich_bundle(bundle_dir: Path, repo_root: Path | None = None) -> None:
    bundle_dir = bundle_dir.resolve()
    if repo_root is None:
        repo_root = _find_repo_root(Path(__file__).resolve())

    runs_path = bundle_dir / "runs.json"
    vend_dir = bundle_dir / "vendored_artifacts"
    logs_dir = bundle_dir / "logs"

    if not runs_path.exists():
        raise RuntimeError(f"Bundle missing runs.json: {runs_path}")
    if not vend_dir.exists():
        # still allow enrichment without vendoring, but capsules will be weaker
        vend_dir.mkdir(exist_ok=True)

    runs_obj = json.loads(runs_path.read_text(encoding="utf-8"))
    runs_list = runs_obj.get("runs", runs_obj)
    if isinstance(runs_list, dict):
        runs_list = list(runs_list.values())

    # ----------------------------
    # 1) CODEPACK: copy demo sources
    # ----------------------------
    codepack = bundle_dir / "codepack"
    codepack.mkdir(exist_ok=True)

    demo_py = list((repo_root / "demos").glob("*/*/demo.py"))
    code_index: List[Dict[str, Any]] = []

    for f in demo_py:
        rel = f.relative_to(repo_root)
        dst = codepack / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dst)
        code_index.append({
            "path": str(rel),
            "sha256": _sha256_file(dst),
            "bytes": dst.stat().st_size,
        })

        # Also include any local *.py siblings (helpers) in the same demo folder
        for g in f.parent.glob("*.py"):
            if g.name == "demo.py":
                continue
            relg = g.relative_to(repo_root)
            dstg = codepack / relg
            dstg.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(g, dstg)
            code_index.append({
                "path": str(relg),
                "sha256": _sha256_file(dstg),
                "bytes": dstg.stat().st_size,
            })

    # de-dup paths
    seen = {}
    for item in code_index:
        seen[item["path"]] = item
    code_index = sorted(seen.values(), key=lambda x: x["path"])

    (codepack / "codepack_index.json").write_text(
        json.dumps({"count": len(code_index), "files": code_index}, indent=2) + "\n",
        encoding="utf-8"
    )

    # ----------------------------
    # 2) CAPSULES: per-demo script + logs + vendored artifacts + manifest
    # ----------------------------
    capsules = bundle_dir / "capsules"
    capsules.mkdir(exist_ok=True)

    for r in runs_list:
        if not isinstance(r, dict):
            continue

        demo_id = str(r.get("demo_id") or r.get("demo") or r.get("slug") or "")
        demo_path = str(r.get("demo_path") or r.get("path") or "")

        # Normalize label: "66b" etc. keep as-is; numeric -> DEMO-##
        label = demo_id
        if demo_id.isdigit():
            label = f"DEMO-{demo_id}"

        cap = capsules / label
        (cap / "artifacts").mkdir(parents=True, exist_ok=True)

        # Copy demo sources (demo folder)
        if demo_path:
            demo_dir = repo_root / demo_path
            if demo_dir.exists():
                for g in demo_dir.glob("*.py"):
                    shutil.copy2(g, cap / g.name)

        # Copy bundle logs for this demo (best-effort match on demo folder name)
        demo_slug = demo_path.split("/")[-1] if demo_path else ""
        if logs_dir.exists() and demo_slug:
            for lp in logs_dir.glob("*"):
                if demo_slug in lp.name:
                    shutil.copy2(lp, cap / lp.name)

        # Copy vendored artifacts for this demo (best-effort match on demo slug)
        if demo_slug and vend_dir.exists():
            for vp in vend_dir.glob("*"):
                if demo_slug in vp.name:
                    shutil.copy2(vp, cap / "artifacts" / vp.name)

        # Manifest
        man = {
            "demo_id": demo_id,
            "label": label,
            "demo_path": demo_path,
            "code_sha256": r.get("code_sha256"),
            "stdout_sha256": r.get("stdout_sha256"),
            "stderr_sha256": r.get("stderr_sha256"),
            "files": [],
        }
        for fp in cap.rglob("*"):
            if fp.is_file():
                man["files"].append({
                    "path": str(fp.relative_to(cap)),
                    "sha256": _sha256_file(fp),
                    "bytes": fp.stat().st_size,
                })
        (cap / "capsule_manifest.json").write_text(
            json.dumps(man, indent=2) + "\n",
            encoding="utf-8"
        )

    # Write an enrichment marker (auditors love this)
    (bundle_dir / "ENRICHMENT_OK.txt").write_text(
        "Bundle enriched with codepack/ and capsules/ for full audit traceability.\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python audits/enrich_bundle.py <bundle_dir>")
    enrich_bundle(Path(sys.argv[1]))
