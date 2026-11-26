"""End to end on this PC, no GPU: the real server/pipeline/worker/client with fake Marker (dev/fakes), Surya and the vision LLM
on separate fake servers. Converts one 3-page document in every quality and counts the vision-LLM requests each one makes.
usage: python dev/modes_check.py"""
import json, os, subprocess, sys, tempfile, threading, time, urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dev"))
import test_e2e as T


def counting(name):
    class H(T.FakeVLM):
        hits = 0

        def do_POST(self):
            type(self).hits += 1
            return super().do_POST()
    H.__name__ = name
    return H


def main():
    VLM, LLM = counting("VLM"), counting("LLM")
    servers = {}
    for H in (VLM, LLM):
        s = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=s.serve_forever, daemon=True).start(); servers[H] = s
    tmp = Path(tempfile.mkdtemp(prefix="modes_"))
    port = T.free_port()
    env = dict(os.environ)
    env.update(BOX_TOKEN=T.TOKEN, PYTHONPATH=f"{T.DEV / 'fakes'}:{T.DEV / 'mdbox'}", FAKE_SEC_PER_PAGE="0.05", FAKE_HOST="box",
               MD_ROOT=str(tmp / "jobs"), MD_UPLOADS=str(tmp / "up"), MD_POOL=str(tmp / "pool"), BOX_DEADLINE=str(time.time() + 3600),
               MD_DEFAULTS=json.dumps({"WORKERS": 1, "VLM_URL": f"http://127.0.0.1:{servers[VLM].server_address[1]}/v1", "VLM_SLOTS": 4,
                                       "LLM_URL": f"http://127.0.0.1:{servers[LLM].server_address[1]}/v1", "LLM_MODEL": "fake-llm",
                                       "LLM_SLOTS": 2, "USE_LLM": True, "MEM_PAUSE_FREE_GB": 0, "MEM_RESUME_FREE_GB": 0,
                                       "MEM_KILL_FREE_GB": 0}))
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port), "--log-level",
                            "warning"], cwd=T.DEV / "mdbox", env=env, stdout=open(tmp / "server.log", "wb"), stderr=subprocess.STDOUT)
    failed = False
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=1); break
            except Exception:
                time.sleep(0.2)
        doc = tmp / "doc.pdf"; doc.write_text("%PDF fake pages=3")
        for q, extra in (("express", []), ("standard", []), ("precise", []),
                         ("standard", ["--no-review"]),           # no page passes: only the figures' calls are left
                         ("standard", ["--labels"]),              # + one label call per page, the Markdown the same
                         ("balanced", [])):                       # an old name: runs standard
            VLM.hits = LLM.hits = 0
            out = tmp / f"out_{q}{len(extra)}"
            r = subprocess.run([sys.executable, str(ROOT / "md_client.py"), str(doc), "--url", f"http://127.0.0.1:{port}", "--token",
                                T.TOKEN, "--out", str(out), "--quality", q, "--local", "0", *extra], capture_output=True, text=True,
                               timeout=300)
            rep = json.loads((out / "report.json").read_text())["documents"][0]
            ok = r.returncode == 0 and rep.get("ok")
            failed |= not ok
            print(f"{' '.join([q, *extra]):22s} ok {ok} | vision-LLM requests {LLM.hits:3d} | described {rep.get('described_figures') or rep.get('figures_described')}"
                  f" | stages {sorted((rep.get('stage_seconds') or {}).keys())}" + (f" | levels {rep.get('page_levels')}" if rep.get("page_levels") else ""))
            if not ok:
                print(r.stdout[-800:], r.stderr[-800:])
    finally:
        srv.terminate(); srv.wait(10)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
