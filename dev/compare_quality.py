"""converts one document at each quality (fast / balanced / accurate) on the box and prints the trade-off: wall time, worker
time per stage, LLM work done, text coverage, the box's GPU/CPU/RAM use, and a score when a scorer for that document exists
(a module `suite_checks.py` next to this file with score(md_path) -> ({case: [(check, ok)]}, extra); optional).

usage: python dev/compare_quality.py DOC.pdf [--out test_out/quality] [--qualities fast balanced accurate]
Each quality goes to its own folder (OUT/<quality>), so a run can be repeated or compared later without converting again
(an existing finished folder is reused: delete it to convert again)."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))


def run(doc, out, quality):
    report = out / "report.json"
    if report.is_file():
        return json.loads(report.read_text())["documents"][0], None
    t = time.time()
    proc = subprocess.run([sys.executable, str(ROOT / "md_client.py"), str(doc), "--out", str(out), "--quality", quality],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not report.is_file():
        print(proc.stdout[-2000:], proc.stderr[-2000:])
        return None, None
    return json.loads(report.read_text())["documents"][0], round(time.time() - t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doc")
    ap.add_argument("--out", default=str(ROOT / "test_out" / "quality"))
    ap.add_argument("--qualities", nargs="+", default=["fast", "balanced", "accurate"])
    a = ap.parse_args()
    rows = []
    for q in a.qualities:
        out = Path(a.out) / q
        print(f"--- {q}: converting into {out} ...", flush=True)
        doc, wall = run(a.doc, out, q)
        if doc is None:
            print(f"    {q} failed"); continue
        md = next(out.rglob("*.md"), None)
        score = None
        try:
            import suite_checks as S                          # optional: your own checks for your own test document
        except ImportError:
            S = None
        if md is not None and S is not None and "Comprehensive_Test_Suite" in Path(a.doc).name:
            res, _ = S.score(md)
            score = f"{sum(ok for c in res.values() for _, ok in c)}/{sum(len(c) for c in res.values())}"
        audit = doc.get("audit") or {}
        rows.append({"quality": q, "wall s": wall if wall is not None else "(earlier run)", "box s": doc.get("seconds"),
                     "stages s": doc.get("stage_seconds"), "pages rewritten": doc.get("pages_rewritten"),
                     "kept draft": doc.get("pages_kept_draft"), "figures": doc.get("figures_described"),
                     "audit fixes": {k: v for k, v in audit.items() if k.startswith("confirmed")} or None,
                     "audit refused": audit.get("refused"), "coverage": doc.get("text_coverage"), "suite score": score,
                     "box": doc.get("box_resources"),
                     "md": str(md) if md else None})
    print()
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    if rows and rows[0].get("suite score") and Path(__file__).with_name("suite_checks.py").is_file():
        print("\nper check (python dev/suite_checks.py ...):")
        subprocess.run([sys.executable, str(Path(__file__).with_name("suite_checks.py"))] + [r["md"] for r in rows if r["md"]])


if __name__ == "__main__":
    main()
