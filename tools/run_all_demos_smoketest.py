#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run all demos under demos/**/demo.py and write a hash ledger suitable for citation.

Behavior:
- Discovers demo entrypoints at: demos/<category>/<demo-folder>/demo.py
- Runs each demo in its own directory (cwd = demo folder)
- Tries `--cert` first (audit mode); if it fails due to unrecognized args, falls back to plain run.
- Captures stdout/stderr per demo
- Computes sha256 for:
    - demo.py
    - produced artifacts in the demo folder (json/png/pdf/csv/txt) after the run
- Produces:
    artifacts/smoketest_all/summary.json
    artifacts/smoketest_all/summary.md
    artifacts/smoketest_all/summary.csv
    artifacts/smoketest_all/ledger.jsonl (one record per demo)
- Exit code 0 if all PASS (cert or run), else 1.
"""

from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ART_EXTS = {".json", ".png", ".pdf", ".csv", ".txt"}

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def try_git_head() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        return out
    except Exception:
        return None

def list_artifacts(demo_dir: Path) -> List[Path]:
    out = []
    for p in demo_dir.iterdir():
        if p.is_file() and p.suffix.lower() in ART_EXTS:
            out.append(p)
    return sorted(out, key=lambda x: x.name.lower())

@dataclass
class RunResult:
    demo_id: str
    category: str
    folder: str
    demo_path: str
    mode: str               # cert|run
    returncode: Optional[int]
    status: str             # PASS|FAIL|TIMEOUT|SKIP
    seconds: float
    code_sha256: str
    stdout_path: str
    stderr_path: str
    artifacts: List[Dict[str, str]]  # [{name, sha256}]
    notes: str = ""

def run_one(python_exe: str, demo_dir: Path, demo_py: Path, timeout_s: int) -> Tuple[str, subprocess.CompletedProcess, float]:
    """
    Returns (mode, completed_process, seconds).
    mode is 'cert' or 'run'.
    """
    # Try --cert first
    t0 = time.time()
    try:
        proc = subprocess.run(
            [python_exe, str(demo_py.name), "--cert"],
            cwd=str(demo_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        dt = time.time() - t0

        # Heuristic: if arg parsing failed, fall back to plain run.
        # Many scripts exit 2 on argparse error and include "unrecognized arguments" / "usage:".
        msg = (proc.stderr or "") + "\n" + (proc.stdout or "")
        if proc.returncode in (2, 64) and ("unrecognized arguments" in msg.lower() or "usage:" in msg.lower()):
            raise ValueError("cert_not_supported")

        return "cert", proc, dt

    except ValueError:
        # fallback to normal run
        t0 = time.time()
        proc = subprocess.run(
            [python_exe, str(demo_py.name)],
            cwd=str(demo_dir),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=os.environ.copy(),
        )
        dt = time.time() - t0
        return "run", proc, dt

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=int, default=600, help="Per-demo timeout seconds (default: 600)")
    ap.add_argument("--python", default=sys.executable, help="Python executable (default: current)")
    ap.add_argument("--root", default="demos", help="Demos root folder (default: demos)")
    args = ap.parse_args()

    demos_root = Path(args.root)
    if not demos_root.exists():
        print(f"Missing demos root: {demos_root}", file=sys.stderr)
        return 2

    out_dir = Path("artifacts") / "smoketest_all"
    out_dir.mkdir(parents=True, exist_ok=True)

    demo_files = sorted(demos_root.glob("*/*/demo.py"))
    if not demo_files:
        print("No demos found at demos/*/*/demo.py", file=sys.stderr)
        return 2

    git_head = try_git_head()

    results: List[RunResult] = []
    any_fail = False
    start_all = time.time()

    for demo_py in demo_files:
        demo_dir = demo_py.parent
        category = demo_py.parent.parent.name
        folder = demo_dir.name
        demo_id = folder.split("-")[1] if folder.startswith("demo-") and len(folder.split("-")) > 1 else folder

        # Hash code
        code_hash = sha256_file(demo_py)

        # Run
        stdout_path = out_dir / f"{category}__{folder}.out.txt"
        stderr_path = out_dir / f"{category}__{folder}.err.txt"

        try:
            mode, proc, dt = run_one(args.python, demo_dir, demo_py, args.timeout)
            stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="replace")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="replace")

            ok = (proc.returncode == 0)
            status = "PASS" if ok else "FAIL"
            if not ok:
                any_fail = True

            # Collect artifacts created in demo folder
            arts = []
            for a in list_artifacts(demo_dir):
                # Exclude the logs we're writing elsewhere, only local artifacts.
                if a.name.lower().endswith((".out.txt", ".err.txt")):
                    continue
                arts.append({"name": a.name, "sha256": sha256_file(a)})

            results.append(RunResult(
                demo_id=str(demo_id),
                category=category,
                folder=folder,
                demo_path=str(demo_py),
                mode=mode,
                returncode=proc.returncode,
                status=status,
                seconds=round(dt, 3),
                code_sha256=code_hash,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                artifacts=arts,
                notes=""
            ))

            print(f"[{status}] {category}/{folder}  mode={mode} rc={proc.returncode}  {dt:.2f}s  artifacts={len(arts)}")

        except subprocess.TimeoutExpired as e:
            stdout_path.write_text((e.stdout or ""), encoding="utf-8", errors="replace")
            stderr_path.write_text((e.stderr or "") + f"\n\n[TIMEOUT after {args.timeout}s]\n", encoding="utf-8", errors="replace")
            results.append(RunResult(
                demo_id=str(demo_id),
                category=category,
                folder=folder,
                demo_path=str(demo_py),
                mode="cert",
                returncode=None,
                status="TIMEOUT",
                seconds=round(args.timeout, 3),
                code_sha256=code_hash,
                stdout_path=str(stdout_path),
                stderr_path=str(stderr_path),
                artifacts=[],
                notes=f"timeout>{args.timeout}s"
            ))
            any_fail = True
            print(f"[TIMEOUT] {category}/{folder}  {args.timeout}s")

    total_dt = time.time() - start_all

    summary = {
        "git_head": git_head,
        "python": args.python,
        "timeout_seconds": args.timeout,
        "total_seconds": round(total_dt, 3),
        "counts": {
            "pass": sum(1 for r in results if r.status == "PASS"),
            "fail": sum(1 for r in results if r.status == "FAIL"),
            "timeout": sum(1 for r in results if r.status == "TIMEOUT"),
            "total": len(results),
        },
        "results": [r.__dict__ for r in results],
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # CSV for citation convenience
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["demo_id", "category", "folder", "mode", "status", "returncode", "seconds", "code_sha256", "artifact_count"])
        for r in results:
            w.writerow([r.demo_id, r.category, r.folder, r.mode, r.status, r.returncode, r.seconds, r.code_sha256, len(r.artifacts)])

    # JSONL ledger (one record per demo; good for paper supplement)
    with (out_dir / "ledger.jsonl").open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r.__dict__) + "\n")

    # Markdown summary
    md = []
    md.append("# Demo Smoke Test Ledger\n")
    md.append(f"- git head: `{git_head}`")
    md.append(f"- python: `{args.python}`")
    md.append(f"- timeout per demo: `{args.timeout}s`")
    md.append(f"- total wall time: `{total_dt:.2f}s`\n")
    md.append(f"PASS/FAIL/TIMEOUT: **{summary['counts']['pass']} / {summary['counts']['fail']} / {summary['counts']['timeout']}** (total {summary['counts']['total']})\n")
    md.append("| Status | Demo | Mode | Seconds | Code sha256 | Artifacts | stdout | stderr |")
    md.append("|---:|---|---:|---:|---|---:|---|---|")
    for r in results:
        md.append(f"| {r.status} | `{r.category}/{r.folder}` | `{r.mode}` | {r.seconds} | `{r.code_sha256}` | {len(r.artifacts)} | `{Path(r.stdout_path).name}` | `{Path(r.stderr_path).name}` |")
    (out_dir / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print("\nWrote ledger:")
    print(" ", out_dir / "summary.md")
    print(" ", out_dir / "summary.csv")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "ledger.jsonl")

    return 0 if not any_fail else 1

if __name__ == "__main__":
    raise SystemExit(main())
