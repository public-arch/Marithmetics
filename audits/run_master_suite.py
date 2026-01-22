#!/usr/bin/env python3
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# -----------------------------
# Minimal ANSI styling (stdlib)
# -----------------------------
class ANSI:
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    gray = "\033[90m"

def _strip_ansi(s: str) -> str:
    return re.sub(r"\x1b\[[0-9;]*m", "", s)

def _term_width(default: int = 100) -> int:
    try:
        return shutil.get_terminal_size((default, 20)).columns
    except Exception:
        return default

def _hr(ch: str = "═", width: Optional[int] = None) -> str:
    w = width or _term_width()
    return ch * max(20, w)

def _now_utc() -> _dt.datetime:
    # timezone-aware UTC time (avoids utcnow() deprecation warnings)
    return _dt.datetime.now(_dt.UTC)

def _ts_compact(dt: _dt.datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%SZ")

def _ts_human(dt: _dt.datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%SZ")

def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)

def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except json.JSONDecodeError:
                continue

@dataclass
class RepoInfo:
    root: Path
    head: str
    head_short: str
    branch: str
    dirty: bool
    python: str
    platform: str

def _run_git(repo_root: Path, args: List[str]) -> str:
    return subprocess.check_output(["git"] + args, cwd=str(repo_root), text=True).strip()

def _get_repo_info(repo_root: Path) -> RepoInfo:
    head = _run_git(repo_root, ["rev-parse", "HEAD"])
    head_short = _run_git(repo_root, ["rev-parse", "--short", "HEAD"])
    try:
        branch = _run_git(repo_root, ["branch", "--show-current"])
    except Exception:
        branch = "?"
    dirty = bool(_run_git(repo_root, ["status", "--porcelain"]))
    python = sys.version.split()[0]
    try:
        plat = f"{os.uname().sysname}-{os.uname().release}-{os.uname().machine}"
    except Exception:
        plat = sys.platform
    return RepoInfo(
        root=repo_root,
        head=head,
        head_short=head_short,
        branch=branch,
        dirty=dirty,
        python=python,
        platform=plat,
    )

class Printer:
    def __init__(self, transcript_path: Path, use_color: bool = True):
        self.use_color = use_color
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        self._f = transcript_path.open("w", encoding="utf-8")

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass

    def _emit(self, s: str) -> None:
        # Console
        if self.use_color:
            print(s)
        else:
            print(_strip_ansi(s))
        # Transcript (no ANSI)
        self._f.write(_strip_ansi(s) + "\n")
        self._f.flush()

    def line(self, s: str = "") -> None:
        self._emit(s)

    def hr(self, ch: str = "═") -> None:
        self._emit(_hr(ch))

    def title(self, s: str) -> None:
        self.hr("═")
        self._emit(f"{ANSI.bold}{s}{ANSI.reset}")
        self.hr("═")

    def section(self, s: str) -> None:
        self.line("")
        self._emit(f"{ANSI.bold}{s}{ANSI.reset}")
        self._emit(_hr("─"))

def _stream_cmd(cmd: List[str], cwd: Path, printer: Printer, label: str) -> int:
    printer.section(f">> {label}")
    printer.line(f"{ANSI.gray}{' '.join(cmd)}{ANSI.reset}")
    p = subprocess.Popen(cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert p.stdout is not None
    for ln in p.stdout:
        printer.line("   " + ln.rstrip("\n"))
    return p.wait()

def _find_latest_bundle(outroot: Path, glob_pat: str) -> Optional[Path]:
    cands = sorted(outroot.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if p.is_dir():
            return p
    return None

def _demo_label_from_text(s: str) -> Optional[str]:
    m = re.search(r"demo-(\d+)", s)
    if not m:
        m = re.search(r"DEMO-(\d+)", s)
    if not m:
        return None
    return f"DEMO-{int(m.group(1))}"

def _parse_verdict(log_text: str) -> str:
    if re.search(r"FINAL VERDICT:\s*VERIFIED", log_text):
        return "VERIFIED"
    if re.search(r"FINAL VERDICT:\s*NOT VERIFIED", log_text):
        return "NOT VERIFIED"
    if re.search(r"\bVERIFIED\b", log_text) and "NOT VERIFIED" not in log_text:
        return "VERIFIED"
    return "UNKNOWN"

def _count_gates(log_text: str) -> Tuple[int, int]:
    pass_n = len(re.findall(r"^\s*✅\s+", log_text, flags=re.M))
    fail_n = len(re.findall(r"^\s*❌\s+", log_text, flags=re.M))
    return pass_n, fail_n

def _extract_fail_gates(log_text: str, max_items: int = 20) -> List[str]:
    fails: List[str] = []
    for m in re.finditer(r"^\s*❌\s+(.+)$", log_text, flags=re.M):
        fails.append(m.group(1).strip())
        if len(fails) >= max_items:
            break
    return fails

def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        try:
            return path.read_text(errors="replace")
        except Exception:
            return ""

def _load_config(repo_root: Path, cfg_path: Optional[str]) -> Dict[str, Any]:
    default_cfg: Dict[str, Any] = {
        "title": "MARI / GUM Master Suite — Authority-of-Record Runner",
        "bundler_module": "audits.gum_bundle_v30",
        "bundler_glob": "GUM_BUNDLE_v30_*",
        "report_generator": "gum/gum_report_generator_v31.py",
        "zip_prefix": "MARI_MASTER_RELEASE",
        "verbosity_default": "flagship",
        "per_demo_timeout_sec": 900,
        "curated_order": [],
        "flagship_full": [],
        "per_demo_intro": {},
        "interludes_after": {},
    }
    if cfg_path:
        p = Path(cfg_path)
        if not p.is_absolute():
            p = repo_root / p
        if p.exists():
            try:
                user_cfg = _load_json(p)
                if isinstance(user_cfg, dict):
                    # Shallow-merge base keys; merge dict sections if present
                    for k, v in user_cfg.items():
                        if isinstance(v, dict) and isinstance(default_cfg.get(k), dict):
                            default_cfg[k].update(v)
                        else:
                            default_cfg[k] = v
            except Exception:
                pass
    return default_cfg

def _build_demo_index_maps(bundle_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    demo_meta_by_label: Dict[str, Dict[str, Any]] = {}
    slug_by_label: Dict[str, str] = {}
    demo_index_json = bundle_dir / "tables" / "demo_index.json"
    if demo_index_json.exists():
        try:
            data = _load_json(demo_index_json)
            items = data.get("demos") if isinstance(data, dict) and "demos" in data else data
            if isinstance(items, list):
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    label = it.get("demo") or it.get("demo_id") or it.get("label")
                    slug = it.get("slug") or it.get("demo_slug") or it.get("run_slug")
                    if not label and slug:
                        label = _demo_label_from_text(str(slug))
                    if isinstance(label, str) and label.startswith("DEMO-"):
                        demo_meta_by_label[label] = it
                        if isinstance(slug, str):
                            slug_by_label[label] = slug
        except Exception:
            pass
    return demo_meta_by_label, slug_by_label

def _find_logs(bundle_dir: Path) -> Dict[str, Dict[str, Path]]:
    logs_dir = bundle_dir / "logs"
    out_map: Dict[str, Dict[str, Path]] = {}

    def ingest(p: Path) -> None:
        name = p.name
        m = re.search(r"demo-(\d+)", name)
        if not m:
            m = re.search(r"DEMO-(\d+)", name)
        if not m:
            return
        label = f"DEMO-{int(m.group(1))}"
        kind = "out" if name.endswith(".out.txt") else "err" if name.endswith(".err.txt") else "txt"
        out_map.setdefault(label, {})[kind] = p

    if logs_dir.exists():
        for p in logs_dir.glob("*.txt"):
            ingest(p)

    cap = bundle_dir / "capsules"
    if cap.exists():
        for p in cap.rglob("*.out.txt"):
            ingest(p)
        for p in cap.rglob("*.err.txt"):
            ingest(p)

    return out_map
    for p in logs_dir.glob("*.txt"):
        label = _demo_label_from_text(p.name)
        if not label:
            continue
        kind = "out" if p.name.endswith(".out.txt") else "err" if p.name.endswith(".err.txt") else "txt"
        out_map.setdefault(label, {})[kind] = p
    return out_map

def _count_structured_exports(bundle_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    values_path = bundle_dir / "values.jsonl"
    if not values_path.exists():
        return counts
    for row in _iter_jsonl(values_path):
        demo_id = row.get("demo_id") or row.get("demo") or row.get("id") or ""
        label = _demo_label_from_text(str(demo_id))
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts

def _count_vendored_artifacts(bundle_dir: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    vdir = bundle_dir / "vendored_artifacts"
    if not vdir.exists():
        return counts
    for p in vdir.iterdir():
        if not p.is_file():
            continue
        label = _demo_label_from_text(p.name)
        if not label:
            continue
        counts[label] = counts.get(label, 0) + 1
    return counts

def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False), encoding="utf-8")

def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")

def _make_zip(zip_path: Path, items: List[Path], root: Path) -> None:
    import zipfile
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(str(zip_path), "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for it in items:
            it = it.resolve()
            if it.is_dir():
                for p in it.rglob("*"):
                    if p.is_file():
                        arc = _safe_relpath(p, root)
                        z.write(str(p), arcname=arc)
            elif it.is_file():
                arc = _safe_relpath(it, root)
                z.write(str(it), arcname=arc)

def _claim_key(name: str) -> str:
    # Stable claim_id component; avoids spaces and punctuation differences.
    return re.sub(r"[^A-Za-z0-9_.:-]+", "_", name).strip("_")

def main() -> int:
    desc_lines = [
        "MARI / GUM Master Suite Runner (Authority-of-Record)",
        "",
        "This command:",
        "  1) Builds a full AoR bundle via audits.gum_bundle_v30 (vendored artifacts + tables + codepack).",
        "  2) Generates the v31 technical report PDF against that bundle.",
        "  3) Produces a human-readable SUMMARY.md + claim_ledger.jsonl for paper rewrites.",
        "  4) Creates a compact master zip for records (bundle + report + manifest + summary).",
        "",
        "Tip: use --verbosity full to print every demo's full stdout (metric-heavy).",
    ]
    ap = argparse.ArgumentParser(
        prog="run_master_suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="\n".join(desc_lines),
    )
    ap.add_argument("--config", default="audits/master_suite_config.json", help="Path to config JSON")
    ap.add_argument("--outroot", default=None, help="AoR output root (default: GUM/authority_archive/AOR_<timestamp>_<gitsha>)")
    ap.add_argument("--timeout", type=int, default=None, help="Per-demo timeout seconds (passed to bundler)")
    ap.add_argument("--python", default=None, help="Python executable (passed to bundler; default: current python)")
    ap.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    ap.add_argument("--strict", action="store_true", help="Nonzero exit if any demo is not VERIFIED")
    ap.add_argument("--verbosity", choices=["compact", "flagship", "full"], default=None, help="CLI verbosity (CI defaults to compact)")
    ap.add_argument("--no-zip", action="store_true", help="Do not create master zip")
    ap.add_argument("--preflight", action="store_true", help="Print planned paths and exit without running")
    args = ap.parse_args()

    try:
        repo_root = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    except Exception:
        repo_root = Path.cwd()

    repo = _get_repo_info(repo_root)
    cfg = _load_config(repo_root, args.config)

    if args.verbosity:
        verbosity = args.verbosity
    else:
        verbosity = "compact" if os.environ.get("GITHUB_ACTIONS", "").lower() == "true" else cfg.get("verbosity_default", "flagship")
        if verbosity not in ("compact", "flagship", "full"):
            verbosity = "flagship"

    ts = _now_utc()
    run_id = f"{_ts_compact(ts)}_{repo.head_short}"
    if args.outroot:
        aor_root = Path(args.outroot)
        if not aor_root.is_absolute():
            aor_root = repo_root / aor_root
    else:
        aor_root = repo_root / "GUM" / "authority_archive" / f"AOR_{run_id}"

    transcript = aor_root / "runner_transcript.txt"
    printer = Printer(transcript_path=transcript, use_color=not args.no_color)

    try:
        printer.title(cfg.get("title", "MARI / GUM Master Suite"))
        printer.line(f"{ANSI.bold}UTC{ANSI.reset}        {_ts_human(ts)}")
        printer.line(f"{ANSI.bold}Repo{ANSI.reset}       {_safe_relpath(repo_root, repo_root)}")
        printer.line(f"{ANSI.bold}Branch{ANSI.reset}     {repo.branch}  {'(dirty)' if repo.dirty else '(clean)'}")
        printer.line(f"{ANSI.bold}HEAD{ANSI.reset}       {repo.head_short}")
        printer.line(f"{ANSI.bold}Python{ANSI.reset}     {repo.python}")
        printer.line(f"{ANSI.bold}Platform{ANSI.reset}   {repo.platform}")
        printer.line(f"{ANSI.bold}AoR root{ANSI.reset}   {_safe_relpath(aor_root, repo_root)}")
        printer.line(f"{ANSI.bold}Mode{ANSI.reset}       verbosity={verbosity}  strict={args.strict}")
        printer.hr("─")

        printer.section("Preface — what this run establishes")
        for b in [
            "A single deterministic kernel evaluated across domains (bridge → SM → GR → QG → cosmology).",
            "Each demo executed as code; stdout/stderr captured as evidence; artifacts vendored and hashed.",
            "Outputs form an Authority-of-Record bundle suitable for rewriting papers against the current demo suite.",
        ]:
            printer.line("  • " + b)
        printer.line("")
        printer.line(f"{ANSI.bold}Narrative flow{ANSI.reset}")
        printer.line("  • Kernel framing (Ω→SM) → residue structure (DRPT cross-base) → admissibility → flagships → breadth sweep.")
        printer.hr("─")

        if args.preflight:
            printer.line("Preflight requested; not running demos.")
            return 0

        aor_root.mkdir(parents=True, exist_ok=True)
        (aor_root / "report").mkdir(parents=True, exist_ok=True)

        bundler_mod = cfg.get("bundler_module", "audits.gum_bundle_v30")
        bundler_glob = cfg.get("bundler_glob", "GUM_BUNDLE_v30_*")
        bundler_cmd = [sys.executable, "-m", bundler_mod, "--outroot", str(aor_root), "--vendor-artifacts", "--python", args.python or sys.executable]
        timeout_val = args.timeout or cfg.get("per_demo_timeout_sec")
        if timeout_val:
            bundler_cmd += ["--timeout", str(timeout_val)]
        demos_root = cfg.get("demos_root")
        if demos_root:
            bundler_cmd += ["--demos-root", str(demos_root)]

        rc = _stream_cmd(bundler_cmd, cwd=repo_root, printer=printer, label="Build AoR bundle (gum_bundle_v30)")
        if rc != 0:
            printer.line(f"{ANSI.red}{ANSI.bold}Bundler failed (exit {rc}).{ANSI.reset}")
            return rc

        bundle_dir = _find_latest_bundle(aor_root, bundler_glob)

        # --- DEBUG PROBE (remove after fix) ---
        try:
            _n_out = len(list((bundle_dir / "logs").glob("*.out.txt")))
            printer.line(f"[debug] bundle/logs out.txt count = {_n_out}")
        except Exception as _e:
            printer.line(f"[debug] bundle/logs probe failed: {_e}")
        if not bundle_dir:
            printer.line(f"{ANSI.red}{ANSI.bold}Could not locate bundle directory under AoR root.{ANSI.reset}")
            return 2

        printer.line("")
        printer.line(f"{ANSI.green}{ANSI.bold}Bundle ready:{ANSI.reset} {_safe_relpath(bundle_dir, repo_root)}")
        bundle_sha_path = bundle_dir / "bundle_sha256.txt"
        bundle_sha = bundle_sha_path.read_text(encoding="utf-8").strip() if bundle_sha_path.exists() else ""
        if bundle_sha:
            printer.line(f"{ANSI.bold}Bundle sha256{ANSI.reset}  {bundle_sha}")
        printer.hr("─")

        report_gen = cfg.get("report_generator", "gum/gum_report_generator_v31.py")
        report_cmd = [sys.executable, str(repo_root / report_gen), "--bundle-dir", str(bundle_dir)]
        rc2 = _stream_cmd(report_cmd, cwd=repo_root, printer=printer, label="Generate v31 technical report (PDF)")
        if rc2 != 0:
            printer.line(f"{ANSI.red}{ANSI.bold}Report generator failed (exit {rc2}).{ANSI.reset}")
            return rc2

        reports_dir = repo_root / "gum" / "reports"
        pdf_latest: Optional[Path] = None
        if reports_dir.exists():
            pdfs = sorted(reports_dir.glob("GUM_Report_v31_*.pdf"), key=lambda p: p.stat().st_mtime, reverse=True)
            if pdfs:
                pdf_latest = pdfs[0]

        pdf_copy: Optional[Path] = None
        manifest_copy: Optional[Path] = None
        if pdf_latest and pdf_latest.exists():
            pdf_copy = aor_root / "report" / pdf_latest.name
            shutil.copy2(pdf_latest, pdf_copy)
            man = pdf_latest.with_suffix(pdf_latest.suffix + ".manifest.json")
            if man.exists():
                manifest_copy = aor_root / "report" / man.name
                shutil.copy2(man, manifest_copy)

        printer.line("")
        if pdf_copy:
            printer.line(f"{ANSI.green}{ANSI.bold}Report copied:{ANSI.reset} {_safe_relpath(pdf_copy, repo_root)}")
        else:
            printer.line(f"{ANSI.yellow}{ANSI.bold}Report PDF not detected in gum/reports; continuing.{ANSI.reset}")
        printer.hr("─")

        demo_meta_by_label, slug_by_label = _build_demo_index_maps(bundle_dir)
        logs_map = {}
        logs_dir = bundle_dir / "logs"
        if logs_dir.exists():
            for fp in logs_dir.glob("*.out.txt"):
                m = re.search(r"demo-(\d+)", fp.name)
                if m: logs_map.setdefault(f"DEMO-{int(m.group(1))}", {})["out"] = fp
            for fp in logs_dir.glob("*.err.txt"):
                m = re.search(r"demo-(\d+)", fp.name)
                if m: logs_map.setdefault(f"DEMO-{int(m.group(1))}", {})["err"] = fp
        cap = bundle_dir / "capsules"
        if cap.exists():
            for fp in cap.rglob("*.out.txt"):
                m = re.search(r"demo-(\d+)", fp.name) or re.search(r"DEMO-(\d+)", fp.name)
                if m: logs_map.setdefault(f"DEMO-{int(m.group(1))}", {})["out"] = fp
            for fp in cap.rglob("*.err.txt"):
                m = re.search(r"demo-(\d+)", fp.name) or re.search(r"DEMO-(\d+)", fp.name)
                if m: logs_map.setdefault(f"DEMO-{int(m.group(1))}", {})["err"] = fp
        structured_counts = _count_structured_exports(bundle_dir)
        artifact_counts = _count_vendored_artifacts(bundle_dir)

        demos_present = sorted(logs_map.keys(), key=lambda d: int(d.split("-")[1]) if "-" in d else 10**9)
        if not demos_present:
            printer.line("")
            printer.line(ANSI.red + ANSI.bold + "ERROR: No demo logs were discovered in the bundle." + ANSI.reset)
            printer.line("Expected logs under bundle/logs/*.out.txt or bundle/capsules/**.out.txt")
            printer.line("Bundle dir: " + _safe_relpath(bundle_dir, repo_root))
            return 3


        order: List[str] = []
        for d in cfg.get("curated_order", []):
            if d in demos_present and d not in order:
                order.append(d)
        for d in demos_present:
            if d not in order:
                order.append(d)

        per_demo_intro: Dict[str, List[str]] = cfg.get("per_demo_intro", {}) or {}
        interludes_after: Dict[str, List[str]] = cfg.get("interludes_after", {}) or {}
        flagship_full = set(cfg.get("flagship_full", []))
        expand_full = (verbosity == "full")

        printer.title("Suite Results — demo-by-demo evidence (logs are canonical)")
        failures: List[str] = []
        rows: List[Dict[str, Any]] = []

        total = len(order)
        for idx, demo in enumerate(order, start=1):
            outp = logs_map.get(demo, {}).get("out")
            errp = logs_map.get(demo, {}).get("err")
            out_text = _read_text(outp) if outp else ""

            verdict = _parse_verdict(out_text)
            pass_n, fail_n = _count_gates(out_text)
            fail_gates = _extract_fail_gates(out_text, max_items=12)
            arts = artifact_counts.get(demo, 0)
            structs = structured_counts.get(demo, 0)

            meta = demo_meta_by_label.get(demo, {})
            title = meta.get("title") or meta.get("name") or meta.get("tests") or ""
            domain = meta.get("domain") or meta.get("cluster") or ""
            slug = slug_by_label.get(demo) or (outp.stem.replace(".out", "") if outp else "")

            ok = (verdict == "VERIFIED") and (fail_n == 0)
            if not ok:
                failures.append(demo)

            badge = f"{ANSI.green}✅{ANSI.reset}" if ok else (f"{ANSI.red}❌{ANSI.reset}" if verdict == "NOT VERIFIED" else f"{ANSI.yellow}⚠{ANSI.reset}")

            printer.hr("─")
            printer.line(f"{ANSI.bold}[{idx:02d}/{total:02d}] {demo}{ANSI.reset}  {badge}  {ANSI.dim}{title}{ANSI.reset}")
            if demo in per_demo_intro:
                printer.line(f"  {ANSI.bold}Context{ANSI.reset}")
                for ln in per_demo_intro[demo]:
                    printer.line("    " + ln)

            if domain:
                printer.line(f"  Domain:     {domain}")
            if slug:
                printer.line(f"  Slug:       {slug}")
            printer.line(f"  Verdict:    {verdict}")
            printer.line(f"  Gates:      PASS {pass_n} / FAIL {fail_n}")
            printer.line(f"  Structured: {structs} rows (values.jsonl)")
            printer.line(f"  Artifacts:  {arts} vendored")
            if outp:
                printer.line(f"  stdout:     {_safe_relpath(outp, repo_root)}")
            if errp and errp.exists():
                printer.line(f"  stderr:     {_safe_relpath(errp, repo_root)}")
            if fail_gates:
                printer.line("  Failing gates:")
                for g in fail_gates:
                    printer.line("    - " + g)

            show_full = expand_full or (verbosity != "compact" and demo in flagship_full)
            if show_full and outp:
                printer.line("")
                printer.line(f"{ANSI.bold}  Output (verbatim; auditable){ANSI.reset}")
                printer.line("  " + _hr("·", width=max(60, _term_width() - 4)))
                for ln in out_text.splitlines():
                    printer.line("  " + ln)
                printer.line("  " + _hr("·", width=max(60, _term_width() - 4)))
            else:
                m = re.search(r"FINAL VERDICT:.*", out_text)
                if m:
                    printer.line(f"  {ANSI.dim}{m.group(0).strip()}{ANSI.reset}")

            if demo in interludes_after:
                printer.line("")
                printer.line(f"  {ANSI.bold}Narrative note{ANSI.reset}")
                for ln in interludes_after[demo]:
                    printer.line("    " + ln)

            rows.append({
                "demo": demo,
                "title": title,
                "domain": domain,
                "slug": slug,
                "verdict": verdict,
                "gates_pass": pass_n,
                "gates_fail": fail_n,
                "structured_rows": structs,
                "vendored_artifacts": arts,
                "stdout": _safe_relpath(outp, repo_root) if outp else "",
                "stderr": _safe_relpath(errp, repo_root) if errp else "",
            })

        printer.hr("─")
        if failures:
            printer.line(f"{ANSI.red}{ANSI.bold}Suite status:{ANSI.reset} NOT CLEAN ({len(failures)} demos not VERIFIED)")
            printer.line("Failures: " + ", ".join(failures))
        else:
            printer.line(f"{ANSI.green}{ANSI.bold}Suite status:{ANSI.reset} VERIFIED (all demos clean)")
        printer.hr("─")

        summary_md = aor_root / "SUMMARY.md"
        summary_lines: List[str] = []
        summary_lines.append(f"# MARI / GUM Master Suite — AoR Summary ({run_id})")
        summary_lines.append("")
        summary_lines.append("## Run identity")
        summary_lines.append(f"- UTC: `{_ts_human(ts)}`")
        summary_lines.append(f"- Repo HEAD: `{repo.head}`")
        summary_lines.append(f"- Branch: `{repo.branch}`  ({'dirty' if repo.dirty else 'clean'})")
        summary_lines.append(f"- Python: `{repo.python}`")
        summary_lines.append(f"- Platform: `{repo.platform}`")
        summary_lines.append("")
        summary_lines.append("## Bundle")
        summary_lines.append(f"- Bundle dir: `{_safe_relpath(bundle_dir, repo_root)}`")
        if bundle_sha:
            summary_lines.append(f"- Bundle sha256: `{bundle_sha}`")
        summary_lines.append("")
        if pdf_copy:
            summary_lines.append("## Report")
            summary_lines.append(f"- PDF: `{_safe_relpath(pdf_copy, repo_root)}`")
            if manifest_copy:
                summary_lines.append(f"- Manifest: `{_safe_relpath(manifest_copy, repo_root)}`")
            summary_lines.append("")
        summary_lines.append("## Demo dashboard")
        summary_lines.append("")
        summary_lines.append("| Demo | Verdict | Gates (P/F) | Structured | Artifacts | Domain |")
        summary_lines.append("|---:|:---:|---:|---:|---:|---|")
        for r in rows:
            verdict = r["verdict"]
            vcell = "✅" if verdict == "VERIFIED" and r["gates_fail"] == 0 else ("❌" if verdict == "NOT VERIFIED" else "⚠")
            summary_lines.append(f"| {r['demo']} | {vcell} | {r['gates_pass']}/{r['gates_fail']} | {r['structured_rows']} | {r['vendored_artifacts']} | {r['domain']} |")
        summary_lines.append("")
        summary_lines.append("## Evidence pointers")
        summary_lines.append("- Canonical per-demo evidence is in `GUM_BUNDLE_v30_*/logs/*.out.txt` and `GUM_BUNDLE_v30_*/vendored_artifacts/` (relative to the AoR root).")
        summary_lines.append("- This AoR directory also includes a full runner transcript: `runner_transcript.txt`.")
        _write_text(summary_md, "\n".join(summary_lines) + "\n")

        claim_path = aor_root / "claim_ledger.jsonl"
        values_path = bundle_dir / "values.jsonl"
        values_sha = _sha256_file(values_path) if values_path.exists() else ""
        claims_written = 0

        with claim_path.open("w", encoding="utf-8") as f:
            runs_json = bundle_dir / "runs.json"
            runs_sha = _sha256_file(runs_json) if runs_json.exists() else ""
            for r in rows:
                demo = r["demo"]
                out_rel = r["stdout"]
                outp2 = (repo_root / out_rel) if out_rel else None
                out_sha = _sha256_file(outp2) if outp2 and outp2.exists() else ""
                base_claim = {
                    "claim_id": f"{demo}:VERDICT",
                    "demo_id": demo,
                    "tier": "A",
                    "claim": f"{demo} verdict = {r['verdict']}; gates PASS {r['gates_pass']} / FAIL {r['gates_fail']}.",
                    "evidence": [
                        {"path": out_rel, "sha256": out_sha},
                        {"path": _safe_relpath(runs_json, repo_root), "sha256": runs_sha},
                    ],
                    "bundle_sha256": bundle_sha,
                    "repo_head": repo.head,
                }
                f.write(json.dumps(base_claim) + "\n")
                claims_written += 1

            if values_path.exists():
                for row in _iter_jsonl(values_path):
                    demo_id = row.get("demo_id") or row.get("demo") or ""
                    label = _demo_label_from_text(str(demo_id))
                    if not label:
                        continue
                    name = (row.get("name") or row.get("value_name") or "").strip()
                    if not name:
                        continue
                    val = row.get("value")
                    units = (row.get("units") or "").strip()
                    claim = {
                        "claim_id": f"{label}:VALUE:{_claim_key(name)}",
                        "demo_id": label,
                        "tier": "B",
                        "claim": f"{label} exports {name} = {val} {units}".strip(),
                        "evidence": [{"path": _safe_relpath(values_path, repo_root), "sha256": values_sha}],
                        "bundle_sha256": bundle_sha,
                        "repo_head": repo.head,
                    }
                    f.write(json.dumps(claim) + "\n")
                    claims_written += 1

        run_meta = {
            "run_id": run_id,
            "generated_utc": _ts_human(ts),
            "repo": {"root": str(repo_root), "branch": repo.branch, "head": repo.head, "dirty": repo.dirty},
            "bundle": {"dir": str(bundle_dir), "sha256": bundle_sha},
            "report": {"pdf": str(pdf_copy) if pdf_copy else "", "manifest": str(manifest_copy) if manifest_copy else ""},
            "counts": {"demos": len(rows), "claims": claims_written, "failures": failures},
            "paths": {"summary_md": str(summary_md), "claim_ledger": str(claim_path), "transcript": str(transcript)},
        }
        _write_json(aor_root / "run_metadata.json", run_meta)

        try:
            shutil.copy2(summary_md, bundle_dir / "SUMMARY.md")
        except Exception:
            pass

        zip_path: Optional[Path] = None
        if not args.no_zip:
            zip_prefix = cfg.get("zip_prefix", "MARI_MASTER_RELEASE")
            zip_path = aor_root / f"{zip_prefix}_{run_id}.zip"
            items: List[Path] = [bundle_dir, summary_md, claim_path, aor_root / "run_metadata.json", transcript]
            if pdf_copy:
                items.append(pdf_copy)
            if manifest_copy:
                items.append(manifest_copy)
            _make_zip(zip_path, items=items, root=repo_root)

        printer.title("AoR Outputs")
        printer.line(f"{ANSI.bold}AoR root{ANSI.reset}      {_safe_relpath(aor_root, repo_root)}")
        printer.line(f"{ANSI.bold}Bundle dir{ANSI.reset}    {_safe_relpath(bundle_dir, repo_root)}")
        if pdf_copy:
            printer.line(f"{ANSI.bold}Report PDF{ANSI.reset}    {_safe_relpath(pdf_copy, repo_root)}")
        printer.line(f"{ANSI.bold}Summary{ANSI.reset}       {_safe_relpath(summary_md, repo_root)}")
        printer.line(f"{ANSI.bold}Claim ledger{ANSI.reset}  {_safe_relpath(claim_path, repo_root)}")
        printer.line(f"{ANSI.bold}Transcript{ANSI.reset}    {_safe_relpath(transcript, repo_root)}")
        if zip_path and zip_path.exists():
            zsize = zip_path.stat().st_size
            printer.line(f"{ANSI.bold}Master zip{ANSI.reset}    {_safe_relpath(zip_path, repo_root)}  ({zsize/1024/1024:.2f} MiB)")
        printer.hr("─")

        if args.strict and failures:
            return 1
        return 0

    finally:
        printer.close()

if __name__ == "__main__":
    raise SystemExit(main())
