"""Find the fastest Surya concurrency on a running box: for each cap on requests in flight per GPU, set it live
(POST /vlm/lb-config) and run the load test, then report pages per hour and leave the best cap set.

usage: python dev/tune_vlm_cap.py DOC.pdf [--caps 0 4 6 8 12] [--copies 2] [--out test_out/tune]
Put the winner into the notebook (VLM_LB_MAX_INFLIGHT, or EXPRESS_LB_MAX_INFLIGHT for an express-only box) so a new box
starts with it. Each step converts COPIES copies of DOC at once (a few minutes a step for a 30-page document)."""
import argparse, json, subprocess, sys, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import load_test as LT

ROOT = LT.ROOT


def set_cap(url, tok, n):
    req = urllib.request.Request(url + "/vlm/lb-config", data=json.dumps({"max_inflight": n}).encode(), method="POST",
                                 headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc"); ap.add_argument("--caps", nargs="+", type=int, default=[0, 4, 6, 8, 12])
    ap.add_argument("--copies", type=int, default=2); ap.add_argument("--quality", default="express")
    ap.add_argument("--out", default="test_out/tune")
    a = ap.parse_args()
    url, tok = LT.box_address(), LT.box_token()
    h = LT.get(url, tok, "/health")
    print(f"box code {h.get('code')} | workers {len(h.get('workers') or {})} | Surya slots {h.get('vlm_slots')} | LLM {h.get('llm')}")
    rows = []
    for cap in a.caps:
        print(f"--- cap {cap or 'none'} per GPU: {set_cap(url, tok, cap)}", flush=True)
        out = Path(a.out) / f"cap_{cap}"
        subprocess.run([sys.executable, str(ROOT / "dev" / "load_test.py"), a.doc, "--copies", str(a.copies), "--quality", a.quality,
                        "--out", str(out)], cwd=ROOT, stdout=subprocess.DEVNULL)
        s = json.loads((out / "summary.json").read_text())
        st = LT.get(url, tok, "/vlm/lb-status")
        rows.append({"cap": cap, "pages": s["pages"], "wall_s": s["wall_s"], "pages_per_hour": s["pages_per_hour"],
                     "gpu_busy": [s.get("avg_gpu0_busy"), s.get("avg_gpu1_busy")],
                     "in_flight": [s.get("avg_surya0_inflight"), s.get("avg_surya1_inflight")], "waited": st["affinity"].get("waited")})
        print(json.dumps(rows[-1]), flush=True)
    ok = [r for r in rows if r["pages"]]
    best = max(ok, key=lambda r: r["pages_per_hour"]) if ok else None
    if best:
        set_cap(url, tok, best["cap"])
        print(f"\nfastest: cap {best['cap'] or 'none'} -> {best['pages_per_hour']} pages/h (left set on this box)")
        print(f"notebook: EXPRESS_LB_MAX_INFLIGHT = {best['cap']}")
    (Path(a.out) / "tune.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
