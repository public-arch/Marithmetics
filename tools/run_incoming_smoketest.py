#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run every Python file in ./incoming as a smoke test.

- Executes each file with the repo's current Python interpreter.
- Captures stdout/stderr to artifacts/smoketest/<file>.out/.err
- Enforces a per-file timeout (default 120s).
- Produces artifacts/smoketest/summary.json and summary.md
- Exit code 0 if all pass, 1 if any fail/time out.
"""

from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--incoming", default="incoming", help="Folder containing demos (default: incoming)")
    ap.add_argument("--timeout", type=int, default=180, help="Per-file timeout seconds (default: 180)")
    ap.add_argument("--python", default=sys.executable, help="Python executable (default: current)")
    args = ap.parse_args()

    incoming = Path(args.incoming)
    if not incoming.exists():
        print(f"Missing folder: {incoming}", file=sys.stderr)
        return 2

    out_dir = Path("artifacts") / "smoketest"
    out_dir.mkdir(parents=True, exist_ok=True)

    py_files = sorted([p for p in incoming.iterdir() if p.is_file() and p.suffix.lower() == ".py"])
    if not py_files:
        print(f"No .py files found in {incoming}")
        return 0

    results = []
    start_all = time.time()

    for p in py_files:
        name = p.name
        out_path = out_dir / (p.stem + ".out.txt")
        err_path = out_dir / (p.stem + ".err.txt")

        cmd = [args.python, str(p)]
        t0 = time.time()

        try:
            proc = subprocess.run(
                cmd,
                cwd=str(Path.cwd()),
                capture_output=True,
                text=True,
                timeout=args.timeout,
                env=os.environ.copy(),
            )
            dt = time.time() - t0
            out_path.write_text(proc.stdout, encoding="utf-8", errors="replace")
            err_path.write_text(proc.stderr, encoding="utf-8", errors="replace")

            ok = (proc.returncode == 0)
            results.append({
                "file": str(p),
                "returncode": proc.returncode,
                "status": "PASS" if ok else "FAIL",
                "seconds": round(dt, 3),
                "stdout_file": str(out_path),
                "stderr_file": str(err_path),
            })
            print(f"[{'PASS' if ok else 'FAIL'}] {name}  rc={proc.returncode}  {dt:.2f}s")

        except subprocess.TimeoutExpired as e:
            dt = time.time() - t0
            out_path.write_text(e.stdout or "", encoding="utf-8", errors="replace")
            err_path.write_text((e.stderr or "") + f"\n\n[TIMEOUT after {args.timeout}s]\n", encoding="utf-8", errors="replace")
            results.append({
                "file": str(p),
                "returncode": None,
                "status": "TIMEOUT",
                "seconds": round(dt, 3),
                "stdout_file": str(out_path),
                "stderr_file": str(err_path),
            })
            print(f"[TIMEOUT] {name}  {dt:.2f}s")

    total_dt = time.time() - start_all
    passed = sum(1 for r in results if r["status"] == "PASS")
    failed = sum(1 for r in results if r["status"] == "FAIL")
    timed  = sum(1 for r in results if r["status"] == "TIMEOUT")

    summary = {
        "incoming": str(incoming),
        "python": args.python,
        "timeout_seconds": args.timeout,
        "total_seconds": round(total_dt, 3),
        "counts": {"pass": passed, "fail": failed, "timeout": timed, "total": len(results)},
        "results": results,
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Markdown summary
    lines = []
    lines.append(f"# Incoming Smoke Test\n")
    lines.append(f"- Python: `{args.python}`")
    lines.append(f"- Timeout per file: `{args.timeout}s`")
    lines.append(f"- Total time: `{total_dt:.2f}s`")
    lines.append(f"- PASS/FAIL/TIMEOUT: **{passed} / {failed} / {timed}** (total {len(results)})\n")

    lines.append("| Status | Seconds | File | stdout | stderr |")
    lines.append("|---:|---:|---|---|---|")
    for r in results:
        status = r["status"]
        sec = r["seconds"]
        f = Path(r["file"]).name
        so = Path(r["stdout_file"]).name
        se = Path(r["stderr_file"]).name
        lines.append(f"| {status} | {sec} | `{f}` | `{so}` | `{se}` |")

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # exit code
    return 0 if (failed == 0 and timed == 0) else 1

if __name__ == "__main__":
    raise SystemExit(main())
