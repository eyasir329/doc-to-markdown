"""Runs a benchmark PDF in several quality modes on the box, one after another, then scores them (dev/bench_score.py).
usage: python dev/bench_run.py PDF MANIFEST.json OUT_DIR [--modes express,standard,precise] [--code CODE_ID]
                               [--client "--chunk-rounds 4"]   (more md_client.py options for every mode)
                               [--gt DIR]   (ground truth: <original page>.md files, see dev/bench_score.py)
--code: wait until the box's /health reports this code (the notebook just uploaded) before starting.
Each mode goes to OUT_DIR/<mode> (with --paginate, so pages can be scored one by one) and logs to OUT_DIR/<mode>.log; a mode
whose OUT_DIR/<mode>/report.json exists is skipped (re-run resumes). Scores: OUT_DIR/score (pages.csv, summary.md, history.jsonl)."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def health():
    r = subprocess.run([sys.executable, str(ROOT / "md_client.py"), "--health"], capture_output=True, text=True, timeout=120, cwd=ROOT)
    t = r.stdout
    return json.loads(t[t.index("{"):]) if "{" in t else {}


def main(argv):
    opt = lambda k, d: argv[argv.index(k) + 1] if k in argv else d
    pdf, manifest, out = Path(argv[0]).resolve(), Path(argv[1]).resolve(), Path(argv[2]).resolve()
    modes = opt("--modes", "express,standard,precise").split(",")
    code = opt("--code", "")
    client = opt("--client", "").split()
    gt = ["--gt", opt("--gt", "")] if "--gt" in argv else []
    out.mkdir(parents=True, exist_ok=True)
    while True:                                       # the box must be up (and run the expected code)
        try:
            h = health()
        except Exception as e:
            h = {"error": repr(e)}
        if h.get("ok") and (not code or h.get("code") == code) and not h.get("running") and not h.get("queued"):
            break
        print(time.strftime("%H:%M:%S"), "waiting for the box:", {k: h.get(k) for k in ("ok", "code", "running", "queued", "error")}, flush=True)
        time.sleep(60)
    print("box", h.get("code"), "llm slots", h.get("llm_slots"), "workers", len(h.get("workers") or {}), flush=True)
    for m in modes:
        if (out / m / "report.json").is_file():
            print(m, "already done", flush=True); continue
        t0 = time.time()
        with open(out / f"{m}.log", "w") as log:
            r = subprocess.run([sys.executable, str(ROOT / "md_client.py"), str(pdf), "--quality", m, "--paginate", "--local", "0",
                                "--out", str(out / m), *client], stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        print(f"{m}: exit {r.returncode}, {time.time() - t0:.0f} s", flush=True)
    done = [str(out / m) for m in modes if (out / m / "report.json").is_file()]
    subprocess.run([sys.executable, str(ROOT / "dev" / "bench_score.py"), str(manifest), *done, "--out", str(out / "score"), *gt], cwd=ROOT)


if __name__ == "__main__":
    main(sys.argv[1:])
