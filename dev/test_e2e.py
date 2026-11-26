"""End-to-end run on this PC with fake Marker/pypdfium2 (dev/fakes): the real server.py + pipeline.py + worker.py as "the box",
md_client.py with --local workers, a fake VLM server. Checks page chunking, chunk stealing by the PC, merging, the VLM proxy
chain (PC worker -> client proxy -> box /vlm -> VLM) and the combined report. Usage: python dev/test_e2e.py"""
import json, os, shutil, socket, subprocess, sys, tempfile, threading, time, unittest, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEV = Path(__file__).resolve().parent
ROOT = DEV.parent
TOKEN = "t" * 32


def rep_pages_rewritten(out):
    return sum(d.get("pages_rewritten", 0) for d in json.loads((out / "report.json").read_text())["documents"])


class FakeVLM(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):                     # the helper LLM: figure descriptions, and page reviews (it returns the draft + a mark)
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
        text = next((c["text"] for m in req.get("messages", []) for c in m.get("content", []) if isinstance(c, dict) and c.get("text")), "")
        if "DRAFT:\n" in text:
            answer = text.split("DRAFT:\n", 1)[1].split("\n\nTEXT LAYER:\n")[0].replace("page ", "**page** ", 1)   # formatting only
        else:
            answer = json.dumps({"kind": "photo", "summary": "fake figure text 42", "details": ""})
        body = json.dumps({"choices": [{"message": {"content": answer}, "finish_reason": "stop"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        body = json.dumps({"status": "ok"} if self.path.endswith("/health") else {"data": [{"id": "fake-surya"}]}).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


class EndToEnd(unittest.TestCase):
    def test_box_and_pc_share_the_work(self):
        tmp = Path(tempfile.mkdtemp(prefix="mdbox_e2e_"))
        vlm = ThreadingHTTPServer(("127.0.0.1", 0), FakeVLM)
        threading.Thread(target=vlm.serve_forever, daemon=True).start()
        port = free_port()
        env = dict(os.environ, BOX_TOKEN=TOKEN, PYTHONPATH=f"{DEV / 'fakes'}:{DEV / 'mdbox'}", FAKE_SEC_PER_PAGE="0.15",
                   MD_ROOT=str(tmp / "jobs"), MD_UPLOADS=str(tmp / "up"), MD_POOL=str(tmp / "boxpool"), BOX_DEADLINE=str(time.time() + 3600),
                   MD_DEFAULTS=json.dumps({"WORKERS": 1, "VLM_URL": f"http://127.0.0.1:{vlm.server_address[1]}/v1", "VLM_SLOTS": 4,
                                                      "LLM_URL": f"http://127.0.0.1:{vlm.server_address[1]}/v1", "LLM_MODEL": "fake-llm",
                                                      "LLM_SLOTS": 2, "USE_LLM": True,
                                                      "MEM_PAUSE_FREE_GB": 0, "MEM_RESUME_FREE_GB": 0, "MEM_KILL_FREE_GB": 0}))   # this PC's RAM is not under test
        srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "server:app", "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
                               cwd=DEV / "mdbox", env=dict(env, FAKE_HOST="box"), stdout=open(tmp / "server.log", "wb"), stderr=subprocess.STDOUT)
        try:
            for _ in range(100):
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{port}/ping", timeout=1); break
                except Exception:
                    time.sleep(0.2)
            docs = tmp / "docs"; (docs / "sub").mkdir(parents=True)
            (docs / "big.pdf").write_text("%PDF fake pages=120")
            (docs / "small.pdf").write_text("%PDF fake pages=5")
            (docs / "sub" / "inner.pdf").write_text("%PDF fake pages=3")
            (docs / "data.csv").write_text("a,b\n1,2\n")
            (docs / "notes.txt").write_text("hello\n")
            out = tmp / "out"
            # an earlier run already saved pages 24-47 of big.pdf: this run must skip that batch and still join everything
            seed = out / ".big.parts" / "00024-00047" / "big"; seed.mkdir(parents=True)
            (seed / "big.md").write_text("\n\n".join(f"page {p} by seed ![](_page_{p}_Picture_0.jpeg)" for p in range(24, 48)) + "\n")
            (seed / "big_meta.json").write_text(json.dumps({"page_stats": [{"page_id": p} for p in range(24, 48)]}))
            for p in range(24, 48):
                (seed / f"_page_{p}_Picture_0.jpeg").write_bytes(b"x")
            import importlib.util
            spec = importlib.util.spec_from_file_location("md_client", ROOT / "md_client.py"); mc = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mc)
            req = urllib.request.Request(f"http://127.0.0.1:{port}/health", headers={"Authorization": f"Bearer {TOKEN}"})
            sig = mc.batch_signature({"QUALITY": "precise"},                      # the run below asks for precise
                                     mc.box_defaults(json.load(urllib.request.urlopen(req, timeout=10))))
            (seed.parent / "done.json").write_text(json.dumps({"ok": True, "pages": 24, "where": "box", "batch": "24-47", "options": sig}))
            # a batch saved with other settings (here: before the page review existed) is converted again
            old = out / ".big.parts" / "00048-00071"; (old / "big").mkdir(parents=True)
            (old / "done.json").write_text(json.dumps({"ok": True, "pages": 24, "where": "box", "batch": "48-71"}))
            r = subprocess.run([sys.executable, str(ROOT / "md_client.py"), str(docs), "--url", f"http://127.0.0.1:{port}", "--token", TOKEN,
                                "--local", "2", "--out", str(out), "--quality", "precise"],   # precise: the LLM describes figures
                               env=dict(env, FAKE_HOST="pc"), capture_output=True, text=True, timeout=300)
            log = r.stdout + r.stderr
            if os.environ.get("SHOW"):
                print(log)
            self.assertEqual(r.returncode, 0, log + (tmp / "server.log").read_text())

            big = (out / "big" / "big.md").read_text()
            self.assertNotIn("mdbox:", big, log)                                     # no placeholder or failure left
            pages = [int(l.replace("**page**", "page").split()[1]) for l in big.split("\n\n") if l.replace("**page**", "page").startswith("page ")]
            self.assertEqual(pages, list(range(120)))                                 # every page once, in order
            self.assertIn(" by box ", big); self.assertIn(" by pc ", big, log)        # split between the box and this PC
            self.assertEqual([p for p in range(120) if f"page {p} by seed " in big], list(range(24, 48)))   # resumed, not redone
            self.assertIn("pages 24-47: already saved earlier, skipped", log)
            self.assertEqual(log.count("saved ("), 4, log)                             # the other 4 batches, one after another
            self.assertFalse((out / ".big.parts").exists())                            # parts joined and cleaned up
            self.assertNotIn("vlm= ", big)                                           # both sides reached the VLM
            # every figure got its LLM transcription, on the box and on the PC (PC -> client /llm proxy -> box /llm -> LLM)
            self.assertEqual(big.count("> **Figure content:** fake figure text 42"), 96, log)   # 120 pages minus the 24 seeded
            self.assertIn("Figure content", (out / "small" / "small.md").read_text())
            # a finished file's figure descriptions written again through the box's LLM, without reconverting
            small = out / "small" / "small.md"
            small.write_text(small.read_text().replace("fake figure text 42", "9VDC"))
            r2 = subprocess.run([sys.executable, str(ROOT / "md_client.py"), "--describe-again", str(small), "--url", f"http://127.0.0.1:{port}",
                                 "--token", TOKEN], env=env, capture_output=True, text=True, timeout=120)
            self.assertEqual(r2.returncode, 0, r2.stdout + r2.stderr)
            self.assertIn("5 of 5 figures described again", r2.stdout)
            self.assertEqual(small.read_text().count("> **Figure content:** fake figure text 42"), 5)
            self.assertNotIn("9VDC", small.read_text()); self.assertIn("9VDC", (out / "small" / "small.md.bak").read_text())
            # every converted page went through the page review, figures and their descriptions restored after it
            self.assertEqual(big.count("**page** "), 96, log); self.assertNotIn("<<KEEP", big)
            self.assertEqual(rep_pages_rewritten(out), 96 + 5 + 3, log)                      # big (not the seeded batch), small, inner
            self.assertEqual(len(list((out / "big").glob("_page_*_Picture_0.jpeg"))), 120)
            meta = json.loads((out / "big" / "big_meta.json").read_text())
            self.assertEqual([p["page_id"] for p in meta["page_stats"]], list(range(120)))

            self.assertEqual((out / "data" / "data.md").read_text(), "| a | b |\n|---|---|\n| 1 | 2 |\n")
            self.assertTrue((out / "sub" / "inner" / "inner.md").is_file())
            rep = {d["file"]: d for d in json.loads((out / "report.json").read_text())["documents"]}
            self.assertEqual(sorted(rep), ["big.pdf", "data.csv", "notes.txt", "small.pdf", "sub/inner.pdf"])
            self.assertTrue(all(d["ok"] for d in rep.values()), rep)
            self.assertEqual((rep["big.pdf"]["where"], rep["big.pdf"]["batches"], rep["big.pdf"]["pages"]), ("box+pc", 5, 120))
            self.assertIn("pc", {d["where"] for f, d in rep.items() if f != "big.pdf"})
        finally:
            srv.terminate(); srv.wait(10); vlm.shutdown()
        shutil.rmtree(tmp, ignore_errors=True)                                        # kept for inspection if the test failed


if __name__ == "__main__":
    unittest.main(verbosity=2)
