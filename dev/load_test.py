"""Throughput test for the box: convert N copies of a document at once and sample how the box is used while it works.

usage: python dev/load_test.py DOC.pdf [--copies 3] [--quality express] [--out test_out/load]
Every 2 s it records, from /health and the Surya load balancer (/vlm/lb-status):
  GPU busy % and memory per GPU, RAM used/free, waiting tasks (pieces no worker has started), Surya requests in flight per GPU server.
At the end it prints pages per hour, the averages, and where the box waited (low GPU busy with few requests in flight =
the workers don't keep Surya fed: more workers or more requests per worker; GPUs near 100% = Surya is the limit).
Writes OUT/samples.csv and OUT/summary.json."""
import argparse, csv, json, os, statistics, subprocess, sys, threading, time, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def box_address():
    out = subprocess.run([sys.executable, str(ROOT / "md_client.py"), "--health"], capture_output=True, text=True, cwd=ROOT).stdout
    for line in out.splitlines():
        if line.startswith("box:"):
            return line.split("box:", 1)[1].strip()
    sys.exit("box not found (is the notebook running?)")


def box_token():
    """the BOX_TOKEN environment variable, else .box_token next to md_client.py (as md_client.py reads it)"""
    tok = os.environ.get("BOX_TOKEN", "").strip()
    if not tok and (ROOT / ".box_token").is_file():
        tok = (ROOT / ".box_token").read_text().strip()
    return tok or sys.exit("no BOX_TOKEN: set it or put it in .box_token next to md_client.py")


def get(url, tok, path):
    req = urllib.request.Request(url + path, headers={"Authorization": f"Bearer {tok}"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc"); ap.add_argument("--copies", type=int, default=3)
    ap.add_argument("--quality", default="express"); ap.add_argument("--out", default="test_out/load")
    a = ap.parse_args()
    out = Path(a.out); docs = out / "docs"; docs.mkdir(parents=True, exist_ok=True)
    src = Path(a.doc).resolve()
    for k in range(a.copies):                                  # separate documents: they share the box's worker queue
        link = docs / f"copy{k + 1}_{src.name}"
        if not link.exists():
            link.symlink_to(src)
    url, tok = box_address(), box_token()
    h = get(url, tok, "/health")
    print(f"box {url} | code {h.get('code')} | workers {len(h.get('workers') or {})} | Surya slots {h.get('vlm_slots')} | "
          f"LLM {h.get('llm')} | default quality {(h.get('defaults') or {}).get('QUALITY')}")

    rows, stop = [], threading.Event()
    def sample():
        t0 = time.time()
        while not stop.is_set():
            try:
                h = get(url, tok, "/health")
                try:
                    lb = get(url, tok, "/vlm/lb-status")
                except Exception:
                    lb = {}
                mem = h.get("memory") if isinstance(h.get("memory"), dict) else {}
                row = {"t": round(time.time() - t0, 1), "waiting_tasks": h.get("waiting_tasks"),   # queued pieces: workers all busy
                       "ram_free_gb": mem.get("free_gb"), "ram_used_gb": mem.get("used_gb")}
                for g in h.get("gpus", []):
                    row[f"gpu{g['index']}_busy"] = g["util_pct"]; row[f"gpu{g['index']}_mem_mb"] = g["mem_used_mb"]
                for k, (b, n) in enumerate(sorted((lb.get("inflight") or {}).items())):
                    row[f"surya{k}_inflight"] = n
                rows.append(row)
            except Exception as e:
                rows.append({"t": round(time.time() - t0, 1), "error": type(e).__name__})
            stop.wait(2)
    th = threading.Thread(target=sample, daemon=True); th.start()

    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "md_client.py"), str(docs), "--out", str(out / "md"), "--quality", a.quality,
                        "--local", "0", "--url", url], cwd=ROOT)
    wall = time.time() - t0
    stop.set(); th.join(5)

    keys = sorted({k for row in rows for k in row})
    with open(out / "samples.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=keys); wr.writeheader(); wr.writerows(rows)
    rep = json.loads((out / "md" / "report.json").read_text()) if (out / "md" / "report.json").exists() else {"documents": []}
    pages = sum(d.get("pages") or 0 for d in rep["documents"] if d.get("ok"))
    def avg(k):
        v = [row[k] for row in rows if isinstance(row.get(k), (int, float))]
        return round(statistics.mean(v), 1) if v else None
    summary = {"exit": r.returncode, "copies": a.copies, "quality": a.quality, "pages": pages, "wall_s": round(wall),
               "pages_per_hour": round(pages / wall * 3600) if wall else None,
               **{f"avg_{k}": avg(k) for k in keys if k not in ("t", "error")},
               "box_resources": [d.get("box_resources") for d in rep["documents"]][:1]}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
