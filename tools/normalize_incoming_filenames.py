#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
import re
import unicodedata
from pathlib import Path

def slugify(s: str) -> str:
    s = s.replace("—", "-").replace("–", "-").replace("_", "-")
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def parse_demo_num(stem: str) -> int | None:
    # Accept: DEMO-65..., DEMO_65..., demo 65..., etc.
    m = re.search(r"(?:^|[^0-9])demo[^0-9]*0*([0-9]{1,3})(?:[^0-9]|$)", stem, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return None

def clean_title_from_stem(stem: str) -> str:
    # Remove leading demo tokens
    s = stem
    s = re.sub(r"^\s*demo[^0-9]*0*[0-9]{1,3}\s*[-_ ]*", "", s, flags=re.IGNORECASE)
    # Remove common noise
    s = s.replace("master", "").replace("flagship", "")
    s = re.sub(r"\s+", " ", s).strip()
    # Keep something
    return s or "untitled"

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="incoming", help="Directory to normalize (default: incoming)")
    ap.add_argument("--dry-run", action="store_true", help="Print plan only")
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.exists():
        raise SystemExit(f"Missing directory: {d}")

    files = sorted([p for p in d.iterdir() if p.is_file() and p.suffix.lower() == ".py"])
    if not files:
        print("No .py files found.")
        return 0

    # Build plan
    used = set(p.name for p in files)
    plan: list[tuple[Path, Path]] = []

    for p in files:
        stem = p.stem
        n = parse_demo_num(stem)
        if n is None:
            # leave placeholder or unknowns with a safe slug
            new_name = f"{slugify(stem)}.py"
        else:
            title = clean_title_from_stem(stem)
            slug = slugify(title)
            new_name = f"demo-{n:02d}-{slug}.py"

        # Ensure uniqueness (avoid collisions)
        candidate = new_name
        k = 1
        while candidate in used and candidate != p.name:
            candidate = candidate.replace(".py", f"-{k}.py")
            k += 1

        used.add(candidate)
        new_path = p.with_name(candidate)

        if new_path.name != p.name:
            plan.append((p, new_path))

    if not plan:
        print("Nothing to rename; filenames already normalized.")
        return 0

    print("Rename plan:")
    for old, new in plan:
        print(f"  {old.name}  ->  {new.name}")

    if args.dry_run:
        print("\n(dry-run) No changes made.")
        return 0

    for old, new in plan:
        old.rename(new)

    print("\nDone.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
