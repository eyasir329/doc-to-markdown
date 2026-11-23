"""Tests for the parts of the box that run without Marker or GPUs. Usage: python dev/test_pipeline.py"""
import json, os, sys, tempfile, unittest, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "mdbox"))
TMP = tempfile.mkdtemp(prefix="mdbox_test_")
os.environ.update(BOX_TOKEN="x" * 32, MD_ROOT=f"{TMP}/jobs", MD_UPLOADS=f"{TMP}/uploads")
import convert as C
import pipeline as P
import review as R
from PIL import Image


class Routing(unittest.TestCase):
    def test_text_and_marker_formats(self):
        for n in ("a.txt", "a.MD", "a.csv", "a.tsv"):
            self.assertEqual(C.route(n, "pdf", False), ("text", None), n)
        for n in ("a.pdf", "a.PNG", "a.epub", "a.html", "a.xlsx"):
            self.assertEqual(C.route(n, "pdf", True), ("marker", None), n)

    def test_office_goes_through_libreoffice_when_available(self):
        self.assertEqual(C.route("a.docx", "pdf", True), ("soffice", "pdf"))       # keeps Word equations / shapes
        self.assertEqual(C.route("a.PPT", "pdf", True), ("soffice", "pdf"))
        self.assertEqual(C.route("a.odt", "native", True), ("soffice", "pdf"))     # Marker cannot read ODT itself
        self.assertEqual(C.route("a.xls", "pdf", True), ("soffice", "xlsx"))
        self.assertEqual(C.route("a.ods", "native", True), ("soffice", "xlsx"))

    def test_djvu(self):
        self.assertEqual(C.route("a.DJVU", "pdf", False, True), ("ddjvu", "pdf"))
        self.assertEqual(C.route("a.djv", "native", True, True), ("ddjvu", "pdf"))
        with self.assertRaisesRegex(C.Unsupported, "djvulibre"):
            C.route("a.djvu", "pdf", True, False)

    def test_ooxml_falls_back_to_marker(self):
        self.assertEqual(C.route("a.docx", "native", True), ("marker", None))
        self.assertEqual(C.route("a.pptx", "pdf", False), ("marker", None))       # no LibreOffice on the box

    def test_legacy_without_libreoffice_is_a_clear_error(self):
        with self.assertRaisesRegex(C.Unsupported, "LibreOffice"):
            C.route("a.doc", "pdf", False)
        with self.assertRaises(C.Unsupported):
            C.route("a.exe", "pdf", True)

    def test_every_extension_routes(self):
        for ext in C.ALL_EXT:
            C.route("x" + ext, "pdf", True, True)
        self.assertEqual(P.DOC_EXT, C.ALL_EXT)

    def test_client_accepts_the_same_extensions(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("md_client", Path(__file__).parent.parent / "md_client.py")
        mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
        self.assertEqual(mc.DOC_EXT, C.ALL_EXT)


class TextFormats(unittest.TestCase):
    def test_csv_table(self):
        md = C.table_to_md('name,note\n"a|b","line1\nline2"\nshort\n\n')
        self.assertEqual(md, "| name | note |\n|---|---|\n| a\\|b | line1<br>line2 |\n| short |  |\n")

    def test_semicolon_sniffed_and_tsv(self):
        self.assertEqual(C.table_to_md("a;b\n1;2\n").splitlines()[2], "| 1 | 2 |")
        self.assertEqual(C.table_to_md("a,b\t1\n", "\t").splitlines()[0], "| a,b | 1 |")
        self.assertEqual(C.table_to_md("  \n"), "")

    def test_encodings(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        (d / "u16.txt").write_bytes("বাংলা\r\nhindi हिन्दी".encode("utf-16"))
        (d / "cp.txt").write_bytes("café".encode("cp1252"))
        self.assertEqual(C.read_text(d / "u16.txt"), "বাংলা\r\nhindi हिन्दी")
        self.assertEqual(C.read_text(d / "cp.txt"), "café")
        r = C.text_to_md(d / "u16.txt", d / "out", "u16")
        self.assertEqual(r["files"], ["u16.md"])
        self.assertEqual((d / "out" / "u16.md").read_text(encoding="utf-8"), "বাংলা\nhindi हिन्दी\n")


class Inputs(unittest.TestCase):
    def test_zip_keeps_new_formats_and_refuses_traversal(self):
        d = Path(tempfile.mkdtemp(dir=TMP)); z = d / "in.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for n in ("a/old.doc", "a/slides.odp", "notes.txt", "data.csv", "skip.exe", "../evil.pdf", "__MACOSX/x.pdf", "a/.hidden.pdf"):
                zf.writestr(n, b"x")
        docs = P.safe_extract(z, d / "out", 100, 10 ** 6)
        self.assertEqual([r for _, r in docs], ["a/old.doc", "a/slides.odp", "data.csv", "evil.pdf", "notes.txt"])
        self.assertTrue(all((d / "out").resolve() in p.parents for p, _ in docs))

    def test_zip_limits(self):
        d = Path(tempfile.mkdtemp(dir=TMP)); z = d / "in.zip"
        with zipfile.ZipFile(z, "w") as zf:
            for i in range(3):
                zf.writestr(f"{i}.txt", b"x" * 100)
        with self.assertRaisesRegex(RuntimeError, "limit"):
            P.safe_extract(z, d / "o1", 2, 10 ** 6)
        with self.assertRaisesRegex(RuntimeError, "too large"):
            P.safe_extract(z, d / "o2", 10, 250)


class Resources(unittest.TestCase):
    def test_pool_shape(self):
        self.assertEqual(P.pool_shape(4, vlm_slots=16), (4, 1, 6))          # Kaggle: a worker per vCPU, 1.5x the GPU slots in flight
        self.assertEqual(P.pool_shape(8, workers=2), (2, 4, 4))            # fewer workers get the spare cores as threads
        self.assertEqual(P.pool_shape(4, vlm_slots=2), (4, 1, 2))

    def test_pages(self):
        self.assertEqual(P.parse_pages(None, 3), [0, 1, 2])
        self.assertEqual(P.parse_pages("5-7, 0,99", 7), [0, 5, 6])
        self.assertEqual(P.format_pages([0, 1, 2, 5, 7, 8]), "0-2,5,7-8")
        self.assertEqual(P.format_pages([4]), "4")

    def test_plan_chunks(self):
        self.assertEqual(len(P.plan_chunks(list(range(15)), 4)), 1)        # short: one piece
        c = P.plan_chunks(list(range(100)), 4)
        self.assertEqual((len(c), sum(map(len, c))), (8, 100))            # ~2 per worker
        c = P.plan_chunks(list(range(1000)), 4)
        self.assertTrue(max(map(len, c)) <= 40 and sum(map(len, c)) == 1000)
        self.assertEqual(sum(P.plan_chunks(list(range(5, 60)), 1), []), list(range(5, 60)))

    def test_merge_and_fill(self):
        d = Path(tempfile.mkdtemp(dir=TMP)); doc = d / "b"

        def chunk(name, pages):
            c = doc / ".chunks" / name; c.mkdir(parents=True)
            (c / "b.md").write_text("\n".join(f"p{p}" for p in pages) + "\n")
            (c / "b_meta.json").write_text(json.dumps({"page_stats": [{"page_id": p} for p in pages], "x": name}))
            for p in pages:
                (c / f"_page_{p}_Picture_0.jpeg").write_bytes(b"j")
            return c
        parts = [("0-1", chunk("0", [0, 1]), "ok"), ("2-3", doc / ".chunks" / "1", "remote"), ("4", doc / ".chunks" / "2", "boom --> x")]
        self.assertEqual(C.merge_chunks(doc, "b", parts), {"failed_pages": ["4"], "remote_pages": ["2-3"]})
        self.assertFalse((doc / ".chunks").exists())
        md = (doc / "b.md").read_text()
        self.assertIn(C.REMOTE_MARK.format("2-3"), md); self.assertIn("pages 4 failed: boom -> x", md)
        local = Path(tempfile.mkdtemp(dir=TMP)) / "c"; local.mkdir()
        (local / "b.md").write_text("p2\np3\n"); (local / "b_meta.json").write_text(json.dumps({"page_stats": [{"page_id": 3}, {"page_id": 2}]}))
        (local / "_page_2_Picture_0.jpeg").write_bytes(b"j")
        self.assertTrue(C.fill_remote(doc, "b", "2-3", local))
        self.assertFalse(C.fill_remote(doc, "b", "2-3", local))            # placeholder already filled
        self.assertEqual((doc / "b.md").read_text().split("\n\n")[:2], ["p0\np1", "p2\np3"])
        meta = json.loads((doc / "b_meta.json").read_text())
        self.assertEqual([e["page_id"] for e in meta["page_stats"]], [0, 1, 2, 3])
        self.assertEqual(sorted(f.name for f in doc.glob("*.jpeg")), ["_page_0_Picture_0.jpeg", "_page_1_Picture_0.jpeg", "_page_2_Picture_0.jpeg"])


class OrphanFigures(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(dir=TMP))
        for n in ("_page_0_Picture_1.jpeg", "_page_1_Diagram_9.jpeg", "_page_1_Diagram_11.jpeg", "_page_3_Figure_2.jpeg", "notes.txt"):
            (self.d / n).write_bytes(b"x")
        sep = lambda p: f"\n\n{{{p}}}" + "-" * 48 + "\n\n"
        (self.d / "a.md").write_text(sep(0) + "intro ![](_page_0_Picture_1.jpeg)" + sep(1) + "page one text" + sep(2) + "page two\n")

    def test_unlinked_figures_go_to_their_page(self):
        self.assertEqual(C.place_orphan_images(self.d / "a.md", self.d), 3)
        self.assertEqual((self.d / "a.md").read_text(),
                         "intro ![](_page_0_Picture_1.jpeg)\n\npage one text\n\n![](_page_1_Diagram_9.jpeg)\n\n![](_page_1_Diagram_11.jpeg)"
                         "\n\npage two\n\n![](_page_3_Figure_2.jpeg)\n")         # page 3 is past the last page: goes to page 2
        self.assertEqual(C.place_orphan_images(self.d / "a.md", self.d), 0)       # idempotent

    def test_keeps_separators_when_asked(self):
        C.place_orphan_images(self.d / "a.md", self.d, keep_pages=True)
        md = (self.d / "a.md").read_text()
        self.assertEqual([int(m.group(1)) for m in C.PAGE_SEP.finditer(md)], [0, 1, 2])
        self.assertLess(md.index("{1}"), md.index("_page_1_Diagram_9"))
        self.assertLess(md.index("_page_1_Diagram_11"), md.index("{2}"))


class FigureRecrop(unittest.TestCase):
    """synthetic 600x400 pt page at 1 px per pt"""
    def page(self):
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (600, 400), "white"); d = ImageDraw.Draw(im)
        d.rectangle([50, 50, 150, 120], outline="black", width=2)          # a chart ...
        d.line([150, 85, 175, 85], fill=(200, 200, 200), width=1)          # ... whose light-grey label sticks out
        d.rectangle([300, 50, 340, 90], fill=(245, 200, 80))               # one figure, detected as two pieces
        d.rectangle([343, 50, 383, 90], fill=(245, 200, 80))
        for y in range(140, 200, 8):                                        # body text right under the chart
            d.line([50, y, 400, y], fill="black", width=2)
        return im

    def test_grow_merge_and_stop_at_body_text(self):
        import figures as F
        figs = [("a.jpeg", (55, 55, 148, 118)), ("b.jpeg", (300, 50, 340, 90)), ("c.jpeg", (343, 50, 383, 90))]
        boxes, merged = F.refine_page(F.ink_mask(self.page()), (600, 400), figs, [((50, 138, 400, 202), 500)])
        a = boxes["a.jpeg"]
        self.assertLessEqual(a[0], 50); self.assertGreaterEqual(a[2], 175)    # grew over the frame and the light label
        self.assertLess(a[3], 138)                                         # but not into the paragraph below
        self.assertEqual(merged, {"c.jpeg": "b.jpeg"})                     # the two pieces became one figure
        self.assertGreaterEqual(boxes["b.jpeg"][2], 383)

    def test_captions_and_text_strips(self):
        import figures as F
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (600, 400), "white"); d = ImageDraw.Draw(im)
        d.rectangle([100, 60, 300, 200], fill=(150, 180, 240))             # the figure
        d.line([100, 208, 290, 208], fill="black", width=2)                # its caption right below, 8 pt away
        for y in (40, 48):
            d.line([100, y, 300, y], fill="black", width=2)                # body text whose last line the layout box reaches
        figs = [("f.jpeg", (100, 45, 300, 200)), ("cap.jpeg", (102, 204, 250, 212))]
        texts = [((98, 204, 292, 212), 40, "caption"), ((98, 36, 302, 51), 300, "text")]
        boxes, gone = F.refine_page(F.ink_mask(im), (600, 400), figs, texts)
        self.assertEqual(gone, {"cap.jpeg": None})                         # the "figure" that is only a caption is dropped
        f = boxes["f.jpeg"]
        self.assertGreaterEqual(f[1], 51 - F.PAD_PT)                        # the body-text strip at the top was trimmed
        self.assertLess(f[3], 207)                                         # and the caption's ink below was not pulled in

    def test_screenshot_box_around_figures_is_dropped(self):
        """Case 01: Marker gave one Diagram box around a whole book-page screenshot (two figures + body text in it);
        the two figures inside are kept on their own, the outer box is not a figure"""
        import figures as F
        from PIL import Image, ImageDraw
        im = Image.new("RGB", (600, 400), "white"); d = ImageDraw.Draw(im)
        d.rectangle([100, 40, 250, 150], fill=(245, 180, 220))             # figure 1
        d.rectangle([300, 40, 450, 150], fill=(150, 180, 240))             # figure 2
        for y in range(200, 260, 8):
            d.line([100, y, 450, y], fill="black", width=2)                # body text of the screenshot
        figs = [("page.jpeg", (80, 20, 470, 280)), ("f1.jpeg", (100, 40, 250, 150)), ("f2.jpeg", (300, 40, 450, 150))]
        texts = [((98, 196, 452, 262), 400, "text")]
        boxes, gone = F.refine_page(F.ink_mask(im), (600, 400), figs, texts)
        self.assertEqual(gone, {"page.jpeg": None})
        self.assertEqual(set(boxes), {"f1.jpeg", "f2.jpeg"})
        self.assertLess(boxes["f1.jpeg"][2], 300)                          # not merged into one
        # a figure with panels (no body text inside) stays one figure
        boxes, gone = F.refine_page(F.ink_mask(im), (600, 400), [("all.jpeg", (95, 35, 455, 155))] + figs[1:], [])
        self.assertNotIn("all.jpeg", gone)

    def test_trim_box(self):
        import figures as F
        self.assertEqual(F.trim_box((0, 0, 100, 100), [(0, -20, 100, 10)]), (0, 10, 100, 100))      # thin strip at the top
        self.assertEqual(F.trim_box((0, 0, 100, 100), [(0, -20, 100, 60)]), (0, 0, 100, 100))      # too big: not a strip, keep
        self.assertEqual(F.trim_box((0, 0, 100, 100), [(80, 0, 130, 100)]), (0, 0, 80, 100))       # right edge

    def test_missing_pdf_images_and_their_captions(self):
        import figures as F
        imgs = [(0, 0, 100, 100), (200, 0, 300, 100), (400, 0, 500, 100)]
        covers = [(0, 0, 90, 95), (390, -5, 505, 105)]                     # a figure, and a table made from the 3rd image
        self.assertEqual(F.missing_images(imgs, covers), [(200, 0, 300, 100)])
        caps = [((200, 106, 300, 116), "Figure 1-2 The cheapest meter"), ((0, 300, 100, 310), "far away")]
        self.assertEqual(F.caption_for((200, 0, 300, 100), caps), "Figure 1-2 The cheapest meter")
        self.assertIsNone(F.caption_for((0, 0, 100, 100), caps))
        self.assertEqual(F.caption_for((200, 30, 300, 100), [((205, 0, 290, 20), "Figure 3 above")]), "Figure 3 above")

    def test_recovered_image_goes_before_its_caption(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        (d / "_page_1_Picture_9000.jpeg").write_bytes(b"x"); (d / "_page_1_Picture_9001.jpeg").write_bytes(b"x")
        sep = lambda p: f"\n\n{{{p}}}" + "-" * 48 + "\n\n"
        (d / "a.md").write_text(sep(0) + "p0" + sep(1) + "text one\n\n*Figure 1-2 The cheapest meter* that I found\n\nmore text\n")
        C.place_orphan_images(d / "a.md", d, anchors={"_page_1_Picture_9000.jpeg": "Figure 1-2 The cheapest meter that I found"})
        self.assertEqual((d / "a.md").read_text(), "p0\n\ntext one\n\n![](_page_1_Picture_9000.jpeg)\n\n*Figure 1-2 The cheapest meter* that I found"
                                                   "\n\nmore text\n\n![](_page_1_Picture_9001.jpeg)\n")       # no caption: end of its page

    def test_coverage_words(self):
        import figures as F
        self.assertEqual(F._words("con\ufffe\r\nnection elec-\ntronic ﬁle of a 9V"), ["connection", "electronic", "file"])

    def test_drop_refs(self):
        import figures as F
        self.assertEqual(F.drop_refs("x\n![](c.jpeg)\ny ![alt](b.jpeg)", ["c.jpeg"]), "x\ny ![alt](b.jpeg)")


class Options(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        P.start_pool = lambda: None            # importing server would otherwise launch Marker workers
        import server
        cls.S = server

    def test_office_option(self):
        self.assertEqual(self.S.clean_options({"office": "native", "IMAGES": None}), {"OFFICE": "native"})
        with self.assertRaises(self.S.HTTPException):
            self.S.clean_options({"OFFICE": "word"})

    def test_page_range(self):
        self.assertEqual(self.S.clean_options({"PAGE_RANGE": "0, 5-10"}), {"PAGE_RANGE": "0,5-10"})
        with self.assertRaises(self.S.HTTPException):
            self.S.clean_options({"PAGE_RANGE": "1-"})


class MemoryGuard(unittest.TestCase):
    """the Kaggle session restarted when RAM filled up during Make Electronics with the LLM passes"""
    def test_worker_count_follows_free_ram(self):
        GB = 2 ** 30
        self.assertEqual(P.workers_for_memory(4, 31 * GB, 6 * GB, 3.5, 3.0), 4)     # plenty: one per CPU
        self.assertEqual(P.workers_for_memory(4, 31 * GB, 18 * GB, 3.5, 3.0), 2)    # (31-18-3)/3.5 = 2.8 -> 2
        self.assertEqual(P.workers_for_memory(4, 16 * GB, 15 * GB, 3.5, 3.0), 1)    # never fewer than one

    def test_cgroup_working_set(self):
        # the stall on Kaggle: GPU servers had just read their model files, so a lot of (freeable) file cache sat in "current"
        files = {"/sys/fs/cgroup/memory.max": "1000", "/sys/fs/cgroup/memory.current": "900",
                 "/sys/fs/cgroup/memory.stat": "anon 500\nfile 400\nshmem 100\nactive_file 250\ninactive_file 150"}
        old = P._read
        P._read = lambda f: files.get(f)
        try:
            self.assertEqual(P.memory(), (1000, 600))                         # file cache is not "used"; tmpfs/shmem is
            self.assertEqual(P.memory_detail()["cache"], 300)
        finally:
            P._read = old

    def test_full_ram_stops_biggest_worker_and_requeues(self):
        import subprocess
        pool = P.Pool(Path(tempfile.mkdtemp(dir=TMP)))
        for d in ("tasks", "results", "running", "logs", "cancel"):
            (pool.dir / d).mkdir(parents=True)
        small = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        big = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(200 << 20); time.sleep(60)"])
        pool.procs = {"w0": small, "w1": big}
        for tid, wid in (("00000001", "w0"), ("00000002", "w1")):
            P.jwrite(pool.dir / "tasks" / f"{tid}.json", {"tid": tid, "job": "j", "doc": 0, "rel": f"book.pdf p{tid}"})
            (pool.dir / "tasks" / f"{tid}.claim").write_text(wid)
            P.jwrite(pool.dir / "running" / f"{tid}.json", {"wid": wid, "t0": 0, "name": "book.pdf"})
        import time; time.sleep(0.5)                                          # let the big one allocate
        old, G = P.memory, 2 ** 30
        try:
            P.memory = lambda: (30 * G, 29 * G)                               # 1 GB free: below MEM_KILL_FREE_GB
            self.assertAlmostEqual(pool.guard_memory(), 1.0)
            self.assertIsNotNone(big.poll()); self.assertIsNone(small.poll())  # only the biggest busy worker stopped
            self.assertFalse((pool.dir / "tasks" / "00000002.claim").exists())  # its piece is free to be taken again
            self.assertFalse((pool.dir / "results" / "00000002.json").exists())  # and not marked failed
            self.assertIn("w1", pool.planned)                                  # relaunched without using the crash budget
            self.assertTrue((pool.dir / "pause").exists())                    # and no new pieces start meanwhile
            pool.guard_memory()
            self.assertIsNone(small.poll())                                   # cooldown: no second kill right away
            P.memory = lambda: (30 * G, 25 * G); pool.guard_memory()
            self.assertTrue((pool.dir / "pause").exists())                    # 5 GB free: between pause and resume, stays paused
            P.memory = lambda: (30 * G, 20 * G); pool.guard_memory()
            self.assertFalse((pool.dir / "pause").exists())                   # 10 GB free: work resumes
        finally:
            P.memory = old; small.kill(); big.kill()

    def test_grown_gpu_server_is_restarted_not_the_workers(self):
        """the Make Electronics run: llama-server's prompt cache (--cache-ram) grew to many GB; the guard stopped 1 GB workers
        again and again until their pages failed. Now the grown server is restarted and the workers keep their pieces."""
        import subprocess, time
        pool = P.Pool(Path(tempfile.mkdtemp(dir=TMP)))
        for d in ("tasks", "results", "running", "logs", "cancel"):
            (pool.dir / d).mkdir(parents=True)
        fake = pool.dir / "llama-server"; fake.symlink_to(sys.executable)      # a process that /proc names "llama-server"
        server = subprocess.Popen([str(fake), "-c", "import time; x = bytearray(300 << 20); time.sleep(60)"])
        worker = subprocess.Popen([sys.executable, "-c", "import time; x = bytearray(100 << 20); time.sleep(60)"])
        pool.procs = {"w0": worker}
        P.jwrite(pool.dir / "tasks" / "00000001.json", {"tid": "00000001", "job": "j", "doc": 0, "rel": "book.pdf p1"})
        (pool.dir / "tasks" / "00000001.claim").write_text("w0")
        P.jwrite(pool.dir / "running" / "00000001.json", {"wid": "w0", "t0": 0, "name": "book.pdf"})
        t0 = time.time()                                                         # wait until both have allocated
        while time.time() - t0 < 20 and (P.rss(server.pid) < 290 << 20 or P.rss(worker.pid) < 95 << 20):
            time.sleep(0.1)
        old, G = P.memory, 2 ** 30
        try:
            P.memory = lambda: (30 * G, 29 * G)
            pool.guard_memory(dict(P.DEFAULTS, MEM_GPU_SERVER_MAX_GB=0.2))
            server.wait(5)
            self.assertIsNotNone(server.poll()); self.assertIsNone(worker.poll())  # the server went, the worker kept working
            self.assertTrue((pool.dir / "tasks" / "00000001.claim").exists())
            self.assertIn("GPU server", pool.mem_events[-1])
            pool.last_kill = 0                                                   # next time nothing is grown: a small worker
            pool.guard_memory(dict(P.DEFAULTS, MEM_GPU_SERVER_MAX_GB=0.2))       # is stopped, but that does not use up a try
            self.assertIsNotNone(worker.poll())
            self.assertEqual(pool.requeued.get("00000001", 0), 0)
        finally:
            P.memory = old; server.kill(); worker.kill()

    def test_low_ram_never_deadlocks(self):
        """the Make Electronics run stalled: RAM stayed above the pause line with nothing running, so no worker ever started.
        Real worker processes (fake Marker) must finish everything one piece at a time while the box stays 'paused'."""
        root = Path(tempfile.mkdtemp(dir=TMP)) / "pool"
        old_env = dict(os.environ)
        os.environ["PYTHONPATH"] = f"{Path(__file__).parent / 'fakes'}:{Path(__file__).parent / 'mdbox'}"
        pool = P.Pool(root)
        try:
            pool.start(2, "", 1, 1)
            (root / "pause").write_text("1.0")                                # RAM low for the whole test
            docs = Path(tempfile.mkdtemp(dir=TMP))
            tids = []
            for k in range(3):
                (docs / f"n{k}.txt").write_text(f"note {k}\n")
                tids.append(pool.enqueue({"job": "j", "doc": k, "src": str(docs / f"n{k}.txt"), "rel": f"n{k}.txt", "stem": f"n{k}",
                                          "out": str(docs / "out" / f"n{k}"), "opts": {"MODE": "fast"}}))
            import time
            t0 = time.time()
            while time.time() - t0 < 60 and not all(pool.result(t) for t in tids):
                pool.maintain(600); time.sleep(0.3)
            self.assertTrue(all((pool.result(t) or {}).get("ok") for t in tids), [pool.log(w, 10) for w in pool.procs])
            t1 = time.time()                                                  # given back right after the result is written
            while (root / "solo").exists() and time.time() - t1 < 5:
                time.sleep(0.1)
            self.assertFalse((root / "solo").exists())                        # the token is given back after each piece
        finally:
            pool.stop(); os.environ.clear(); os.environ.update(old_env)


class PageReview(unittest.TestCase):
    """every page is rewritten by the vision LLM from its image, but a rewrite is only used when it loses nothing"""
    DRAFT = ("## 3. Resistors\n\nA resistor limits the cur-\nrent that flows through a circuit, measured in ohms. Ohm's law says\n"
             "<math>V = I R</math> for every resistor in the circuit, at any voltage.\n\n"
             '![](_page_3_Figure_1.jpeg)\n\n> **Figure content:** a 470 ohm resistor\n> with yellow, violet, brown bands\n\n'
             '<span id="page-3-2"></span>Table 1 lists common values used in the experiments of this chapter.')
    GOOD = ("## 3. Resistors\n\nA resistor limits the current that flows through a circuit, measured in ohms. Ohm's law says\n"
            "$V = I R$ for every resistor in the circuit, at any voltage.\n\n<<KEEP 1>>\n\n"
            "<<KEEP 2>>Table 1 lists common values used in the experiments of this chapter.")

    def run_page(self, answer, finish="stop", layer=""):
        old = R.ask_page
        R.ask_page = lambda *a, **k: (answer, finish)
        try:
            text, why = R.review_page(self.DRAFT, b"jpeg", layer, "http://llm", "m", 8192)
            return text, why.split("|")[0] if why else why             # "reason|what differed": the reason
        finally:
            R.ask_page = old

    def test_protect_restore(self):
        body, kept = R.protect(self.DRAFT)
        self.assertEqual(len(kept), 2); self.assertNotIn("![](", body); self.assertNotIn("Figure content", body)
        self.assertEqual(R.restore(body, kept), self.DRAFT)

    def test_good_rewrite_is_used_with_figures_restored(self):
        text, why = self.run_page("```markdown\n" + self.GOOD + "\n```")        # a fence around the whole answer is removed
        self.assertIsNone(why)
        self.assertIn("$V = I R$", text); self.assertIn("current that flows", text)
        self.assertIn("![](_page_3_Figure_1.jpeg)\n\n> **Figure content:** a 470 ohm resistor\n> with yellow", text)
        self.assertIn('<span id="page-3-2"></span>Table 1', text)

    def test_lossy_rewrites_keep_the_draft(self):
        for answer, finish, why in (
                ("## Resistors\n\nThis page explains resistors and Ohm's law. <<KEEP 1>> <<KEEP 2>>", "stop", "number changed or lost"),
                (self.GOOD.replace("Table 1 lists", "Table 7 lists"), "stop", "number changed or lost"),   # a value changed
                (self.GOOD.replace("in ohms", "in milliohms"), "stop", "words added"),
                (self.GOOD.replace(" at any voltage.", "."), "stop", "draft words lost"),
                (self.GOOD + "\n\nThis chapter is very important for every beginner to read carefully twice.", "stop", "words added"),
                (self.GOOD, "length", "answer cut off"),
                ("", "stop", "empty")):
            text, got = self.run_page(answer, finish)
            if got is None:                                  # the good paragraphs were taken, the bad one kept from the draft
                self.assertNotIn("Table 7", text); self.assertNotIn("milliohms", text); self.assertNotIn("beginner", text)
                self.assertIn("at any voltage.", text); self.assertIn("Table 1 lists", text)
                self.assertIn("![](_page_3_Figure_1.jpeg)", text)
            else:
                self.assertEqual((text, got), (self.DRAFT, why), answer[:40])

    def test_good_paragraphs_of_a_partly_bad_rewrite_are_used(self):
        """Case 03: the rewrite fixed equation 2.13 (ε -> \\hat{e}, λ -> \\lambda) but made up a "clean" version of the
        garbled ground-truth line, and the whole page was thrown away"""
        draft = ("Equation (2.13):\n\n$$T = (λ_1 - λ_2)ε_1ε_1^T + λ_3ε_3ε_3^T.$$\n\nThe following cases.\n\n"
                 "\\$\\$ \\mathbf{T} = (\\lambda\\_1 - \\lambda\\_2)\\mathbf{e}\\_1 + \\lambda\\_3 \\mathbf{I}\\_3 \\$\\$")
        answer = ("$$T = (\\lambda_1 - \\lambda_2)\\hat{e}_1\\hat{e}_1^T + \\lambda_3\\hat{e}_3\\hat{e}_3^T. \\tag{2.13}$$\n\n"
                  "The following cases.\n\n$$\\mathbf{T} = (\\lambda_1 - \\lambda_2)\\mathbf{e}_1$$")
        old = R.ask_page; R.ask_page = lambda *a, **k: (answer, "stop")
        try:
            text, why = R.review_page(draft, b"j", "", "u", "m", 8192)
        finally:
            R.ask_page = old
        self.assertIsNone(why)
        self.assertTrue(text.startswith("$$T = (\\lambda_1 - \\lambda_2)\\hat{e}_1"))         # the equation fix is taken
        self.assertIn("\\tag{2.13}", text)
        self.assertTrue(text.endswith("\\mathbf{I}\\_3 \\$\\$"))                        # the made-up line is not

    def test_dropped_or_repeated_markers_are_put_back(self):
        """27 of 96 Make Electronics pages lost their rewrite because the LLM dropped or repeated a <<KEEP n>>"""
        dropped = self.GOOD.replace("<<KEEP 1>>\n\n", "").replace("<<KEEP 2>>", "")
        text, why = self.run_page(dropped)
        self.assertIsNone(why)
        self.assertIn("at any voltage.\n\n![](_page_3_Figure_1.jpeg)", text)            # figure back between its paragraphs
        self.assertIn('\n\n<span id="page-3-2"></span>Table 1 lists', text)                # anchor back at its paragraph
        text, why = self.run_page(self.GOOD + "\n\n<<KEEP 1>>")                         # repeated: the extra one removed
        self.assertIsNone(why); self.assertEqual(text.count("_page_3_Figure_1.jpeg"), 1)
        text, why = self.run_page(self.GOOD + " <<KEEP 9>>")                              # made up: removed
        self.assertIsNone(why); self.assertNotIn("KEEP", text)

    def test_text_layer_must_not_lose_page_words(self):
        layer = " ".join(f"word{i}" for i in range(30)) + " resistor limits current"
        draft_like = self.DRAFT + " " + " ".join(f"word{i}" for i in range(30))
        old = R.ask_page; R.ask_page = lambda *a, **k: (self.GOOD, "stop")         # misses the 30 words the PDF has
        try:
            text, why = R.review_page(draft_like, b"j", layer, "u", "m", 8192)
        finally:
            R.ask_page = old
        self.assertIn(" ".join(f"word{i}" for i in range(30)), text)         # the paragraph that lost them stays the draft's

    def test_dense_page_is_reviewed_in_parts(self):
        paras = [f"paragraph {k} " + " ".join(f"w{k}x{i}" for i in range(120)) for k in range(12)]   # ~10k tokens of draft
        draft = "\n\n".join(paras[:6] + ["![](_page_1_Figure_1.jpeg)"] + paras[6:])
        calls = []

        def ask(u, m, jpeg, prompt, n):
            calls.append(prompt)
            return prompt.split("DRAFT:\n")[1].upper().replace("<<KEEP", "<<KEEP"), "stop"
        old = R.ask_page; R.ask_page = ask
        try:
            text, why = R.review_page(draft, b"j", "", "u", "m", 8192)
        finally:
            R.ask_page = old
        self.assertGreater(len(calls), 1); self.assertIn("part 1 of", calls[0])
        self.assertIsNone(why)
        self.assertEqual(text.count("![](_page_1_Figure_1.jpeg)"), 1)
        self.assertEqual(text.replace("![](_page_1_Figure_1.jpeg)", "").split(), draft.upper().replace("![](_PAGE_1_FIGURE_1.JPEG)", "").split())
        old = R.ask_page; R.ask_page = lambda *a, **k: self.fail("must not be asked")
        try:                                                                   # one paragraph alone too long: draft kept
            self.assertEqual(R.review_page("word " * 9000, b"j", "", "u", "m", 8192)[1], "page too long for the LLM context")
        finally:
            R.ask_page = old

    def test_review_pages_and_finish(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (100, 140), "white").save(d / "page.png")
        md = d / "doc.md"
        md.write_text(f"\n\n{{0}}{'-' * 48}\n\nfirst page text\n\n{{1}}{'-' * 48}\n\nsecond page text\n")
        old = R.ask_page; R.ask_page = lambda u, m, jpeg, prompt, n: (prompt.split("DRAFT:\n")[1].upper(), "stop")
        try:
            st = R.review_pages(md, d / "page.png", "u", "m", concurrency=2)
        finally:
            R.ask_page = old
        self.assertEqual((st["pages"], st["rewritten"]), (2, 2))
        R.finish_markdown(md)
        self.assertEqual(md.read_text(), "FIRST PAGE TEXT\n\nSECOND PAGE TEXT\n")

    def test_tidy(self):
        md = ("# Title\ntext <math>x^2</math> and\n<math display=\"block\">E = mc^2</math>\nafter\n| a | b |\n|---|---|\n| 1 | 2 |\nnext\n"
              "```python\nprint(1)\n\n\n\nx = 2\n```\n\n\n\nend\n```\nopen code")
        self.assertEqual(R.tidy(md),
                         "# Title\n\ntext $x^2$ and\n\n$$\nE = mc^2\n$$\n\nafter\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nnext\n\n"
                         "```python\nprint(1)\n\n\n\nx = 2\n```\n\nend\n\n```\nopen code\n```\n")

    def test_readability(self):
        md = ("You need a meter because I consider it\n\n<span id=\"page-15-0\"></span>the most essential tool.\n\n"
              "- If you prefer a kit, you have a choice\n\nof kits. See page 311.\n\n"
              "This ends a sentence.\n\nanother lower-case start stays apart\n\n"
              "## Heading\n\nlower text\n\n```\nx = 1 and\n\ny = 2\n```\n\n- • Search eBay\n• Plain bullet\n\nSee [Figure 2-63](#) and [site](http://x.org).\n")
        self.assertEqual(R.tidy(md),
                         "You need a meter because I consider it <span id=\"page-15-0\"></span>the most essential tool.\n\n"
                         "- If you prefer a kit, you have a choice of kits. See page 311.\n\n"
                         "This ends a sentence.\n\nanother lower-case start stays apart\n\n"
                         "## Heading\n\nlower text\n\n```\nx = 1 and\n\ny = 2\n```\n\n- Search eBay\n- Plain bullet\n\nSee Figure 2-63 and [site](http://x.org).\n")

    def test_money_code_and_table_maths(self):
        md = ("US $34.99 CAN $40.99\n\nIt cost $1,000 per year, then $3.\n\nOhm: $1 + 20\\% = 1.2\\text{K}$ and $5$ volts.\n\n"
              "| a | <math display=\"block\">x^2</math> |\n|---|---|\n\n```\nprice = \"$5\"  # • not a bullet <math>y</math>\n```\n\n"
              "> **Figure content:** USB port\n> ```\n")
        self.assertEqual(R.tidy(md),
                         "US \\$34.99 CAN \\$40.99\n\nIt cost \\$1,000 per year, then \\$3.\n\n"
                         "Ohm: $1 + 20\\% = 1.2\\text{K}$ and $5$ volts.\n\n| a | $x^2$ |\n|---|---|\n\n"
                         "```\nprice = \"$5\"  # • not a bullet <math>y</math>\n```\n\n> **Figure content:** USB port\n")

    def test_outline_headings(self):
        """Make Electronics: chapters, experiments and their sections all came out as ##; the PDF outline knows better"""
        marks = [(0, R._key("Chapter 1. The Basics"), 14), (1, R._key("Experiment 3: Your First Circuit"), 31),
                 (2, R._key("What You Will Need"), 31), (2, R._key("Setup"), 31), (1, R._key("Experiment 4: Variable Resistance"), 35),
                 (2, R._key("What You Will Need"), 35)]
        d = Path(tempfile.mkdtemp(dir=TMP)); md = d / "doc.md"
        md.write_text(f"\n\n{{31}}{'-' * 48}\n\n## Experiment 3: Your **First** Circuit\n\n## What You Will Need\n\n- a battery\n\n"
                      f"Setup\n\nText about setup.\n\n## Lighting an LED\n\n```\n## not a heading\n```\n\n"
                      f"{{33}}{'-' * 48}\n\n## Experiment 3: Your First Circuit\n\nMore text.\n\n"
                      f"{{35}}{'-' * 48}\n\n## Experiment 4: Variable Resistance\n\n## What You Will Need\n")
        st = R.apply_outline(md, marks)
        R.finish_markdown(md)
        self.assertEqual(md.read_text(),
                         "## Experiment 3: Your **First** Circuit\n\n### What You Will Need\n\n- a battery\n\n### Setup\n\n"
                         "Text about setup.\n\n#### Lighting an LED\n\n```\n## not a heading\n```\n\nMore text.\n\n"
                         "## Experiment 4: Variable Resistance\n\n### What You Will Need\n")
        self.assertEqual(st, {"matched": 4, "below": 1, "promoted": 1, "repeats_dropped": 1})

    def test_unused_anchors_dropped(self):
        md = '## <span id="page-3-0"></span>Intro\n\nSee [Setup](#page-9-1).\n\n<span id="page-9-1"></span>Setup text and <span id="x"></span>more.\n'
        self.assertEqual(R.drop_unused_anchors(md), '## Intro\n\nSee [Setup](#page-9-1).\n\n<span id="page-9-1"></span>Setup text and more.\n')

    def test_shrink_images(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (3000, 1500), "red").save(d / "_page_1_Figure_1.jpeg")
        Image.new("RGB", (800, 600), "red").save(d / "_page_1_Figure_2.png")
        self.assertEqual(R.shrink_images(d, 2000), 1)
        self.assertEqual(Image.open(d / "_page_1_Figure_1.jpeg").size, (2000, 1000))
        self.assertEqual(Image.open(d / "_page_1_Figure_2.png").size, (800, 600))


class TextCoverage(unittest.TestCase):
    def test_running_headers_do_not_count(self):
        """page 144 of Make Electronics: two captions plus the running header/footer, which the Markdown leaves out on purpose"""
        import types, figures as FIG
        body = "the quick resistor limits current flowing through every circuit you build on the breadboard today"
        words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
        pages = [f"Chapter 3: Getting Somewhat More Serious {100 + i}\n{body} {words[i]}\nExperiment 14: A Wearable Pulsing Glow" for i in range(6)]
        pages.append("Figure 3-73 For conversion between US measurements and metric (millimeters).\n"
                     "Chapter 3: Getting Somewhat More Serious 131\nExperiment 14: A Wearable Pulsing Glow")

        class Page:
            def __init__(self, t): self.t = t
            def get_textpage(self): return types.SimpleNamespace(get_text_range=lambda: self.t, close=lambda: None)
        doc = type("Doc", (), {"__len__": lambda self: len(pages), "__getitem__": lambda self, i: Page(pages[i]), "close": lambda self: None})()
        old = sys.modules.get("pypdfium2")
        sys.modules["pypdfium2"] = types.SimpleNamespace(PdfDocument=lambda path: doc)
        try:
            md = body + " " + " ".join(words) + "\n\n**Figure 3-73** For conversion between US measurements and metric (millimeters).\n"
            cov = {c["page"]: c for c in FIG.text_coverage("x.pdf", md, list(range(len(pages))))}
        finally:
            sys.modules.pop("pypdfium2") if old is None else sys.modules.__setitem__("pypdfium2", old)
        self.assertEqual(cov[6]["found"], 1.0)                                # the header/footer words are not missed
        self.assertEqual(cov[0]["found"], 1.0)


class NoRepeatedWork(unittest.TestCase):
    def test_figures_described_once_icons_skipped(self):
        import figures as FIG
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (60, 60), "red").save(d / "_page_1_Picture_1.jpeg")             # an icon
        Image.new("RGB", (600, 400), "blue").save(d / "_page_1_Picture_2.jpeg")
        (d / "_page_5_Picture_1.jpeg").write_bytes((d / "_page_1_Picture_2.jpeg").read_bytes())   # the same logo again
        Image.new("RGB", (600, 400), "green").save(d / "_page_6_Figure_1.jpeg")
        md = d / "doc.md"
        md.write_text("".join(f"![]({n})\n\ntext {n}\n\n" for n in sorted(f.name for f in d.glob("*.jpeg"))))
        calls = []
        old = FIG._ask_image
        FIG._ask_image = lambda u, m, path, prompt, **k: calls.append(path.name) or {"kind": "photo", "summary": f"content of {path.name}", "details": ""}
        try:
            self.assertEqual(FIG.describe_figures(md, d, "u", "m"), 3)
        finally:
            FIG._ask_image = old
        self.assertEqual(sorted(calls), ["_page_1_Picture_2.jpeg", "_page_6_Figure_1.jpeg"])   # 2 calls for 3 figures
        out = md.read_text()
        self.assertEqual(out.count("content of _page_1_Picture_2.jpeg"), 2); self.assertNotIn("content of _page_1_Picture_1", out)

    def test_logo_named_in_alt_text(self):
        """Case 01: the KEY POINT icon got a whole quote block; a logo is named in its alt text, and not asked again"""
        import figures as FIG
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (214, 160), "pink").save(d / "_page_0_Picture_76.jpeg")
        md = d / "doc.md"; md.write_text("![](_page_0_Picture_76.jpeg)\n\nA state is an externally observable mode.\n")
        calls = []
        old = FIG._ask_image
        FIG._ask_image = lambda u, m, path, prompt, **k: calls.append(1) or {"kind": "logo", "summary": "A 'KEY POINT' [icon] logo.", "details": ""}
        try:
            FIG.describe_figures(md, d, "u", "m"); FIG.describe_figures(md, d, "u", "m")
        finally:
            FIG._ask_image = old
        self.assertEqual(md.read_text(), "![A 'KEY POINT' icon logo.](_page_0_Picture_76.jpeg)\n\nA state is an externally observable mode.\n")
        self.assertEqual(len(calls), 1)

    def test_flow_diagram_graph_to_mermaid(self):
        """Case 01: the prose retold Figure 5.3's three nested activity diagrams as one flow and left out a box; the graph
        is read on its own and written out exactly (arrows in words + Mermaid)"""
        import figures as F
        g = {"diagrams": [
            {"nodes": [{"id": "start", "label": "", "shape": "start"}, {"id": "a", "label": "Elicit requirements", "shape": "activity"},
                       {"id": "end", "label": "", "shape": "end"}],
             "edges": [{"from": "start", "to": "a", "label": ""}, {"from": "a", "to": "end", "label": ""}]},
            {"nodes": [{"id": "start", "label": "", "shape": "start"}, {"id": "f", "label": "", "shape": "fork"},
                       {"id": "u", "label": "Draw use-case diagram", "shape": "activity"}, {"id": "q", "label": 'Say "hi"', "shape": "decision"}],
             "edges": [{"from": "start", "to": "f", "label": ""}, {"from": "f", "to": "u", "label": ""}, {"from": "f", "to": "q", "label": "Yes"},
                       {"from": "u", "to": "ghost", "label": ""}]}],
            "links": ["Elicit requirements -> 2", "Nothing -> 7", "u -> diagram1"]}
        words, mm = F.graph_markdown(g)
        self.assertIn("  - start → fork → Draw use-case diagram", words)               # the main flow first ...
        self.assertIn('  - fork → (Yes) Say "hi"', words)                           # ... then the branch
        self.assertIn("- *Elicit requirements* (Diagram 1) is shown in detail in Diagram 2", words)
        self.assertIn("- *Draw use-case diagram* (Diagram 2) is shown in detail in Diagram 1", words)     # named by its id
        self.assertNotIn("ghost", mm); self.assertNotIn("Nothing", words)
        self.assertIn("flowchart LR", mm); self.assertIn("    direction TB", mm)    # diagrams side by side, each top to bottom
        self.assertIn('n4((" ")):::umlStart', mm)                                  # the 2nd diagram's start is its own node
        self.assertIn('n3(((" "))):::umlEnd', mm)
        self.assertNotIn(":::end", mm); self.assertNotIn("classDef end", mm)         # "end" is a Mermaid keyword: breaks the chart
        self.assertIn('n7{"Say #quot;hi#quot;"}', mm); self.assertIn('n5 -->|"Yes"| n7', mm)
        self.assertIn('d1 -.->|"Elicit requirements in detail"| d2', mm)
        self.assertEqual(F.graph_markdown({"diagrams": [], "links": []}), ("", ""))
        one = {"diagrams": [{"nodes": [{"id": "a", "label": "A", "shape": "activity"}, {"id": "b", "label": "B", "shape": "activity"},
                                       {"id": "c", "label": "C", "shape": "activity"}],
                             "edges": [{"from": "a", "to": "b", "label": ""}, {"from": "b", "to": "c", "label": ""},
                                       {"from": "c", "to": "a", "label": ""}]}], "links": []}
        words, mm = F.graph_markdown(one)
        self.assertEqual(words, "Arrows, as drawn:\n- A → B → C → A")               # one chain, the loop closed
        self.assertNotIn("subgraph", mm); self.assertIn("flowchart TD", mm)

    def test_missing_arrows_added_only_when_every_check_agrees(self):
        """Case 02: answers often lacked the waterfall's 4 feedback arrows; a second look finds them. On a complete graph
        that look invents arrows (fork -> join, fork -> start), never the same labelled-box arrow in every answer"""
        import figures as F
        g = {"diagrams": [{"nodes": [{"id": "a", "label": "Design", "shape": "activity"}, {"id": "b", "label": "Test", "shape": "activity"},
                                     {"id": "f", "label": "", "shape": "fork"}],
                           "edges": [{"from": "a", "to": "b", "label": ""}]}]}
        checks = [[{"from": "Test", "to": "Design", "label": "back"}, {"from": "fork", "to": "Test", "label": ""},
                   {"from": "Design", "to": "Test", "label": ""}],
                  [{"from": "test", "to": "design", "label": ""}, {"from": "Design", "to": "Nowhere", "label": ""}]]
        self.assertEqual(F.add_missing(g, checks), 1)
        self.assertEqual(g["diagrams"][0]["edges"][-1], {"from": "b", "to": "a", "label": ""})
        self.assertEqual(F.add_missing(g, [checks[0], [{"from": "fork", "to": "Test", "label": ""}]]), 0)

    def test_mermaid_kept_outside_quote_and_protected(self):
        import figures as F
        out = F._render({"kind": "diagram", "summary": "S.", "details": "Arrows", "mermaid": "```mermaid\nflowchart TD\n  a --> b\n```"}, "")
        self.assertTrue(out.endswith("\n\n```mermaid\nflowchart TD\n  a --> b\n```"))
        md = "![](f.jpeg)\n\n" + out + "\n\nNext paragraph."
        self.assertEqual(F.DESC_BLOCK.sub("", md), "![](f.jpeg)\n\nNext paragraph.")
        body, kept = R.protect(md)
        self.assertEqual(body, "<<KEEP 1>>\n\nNext paragraph.")                   # the page review can't touch it

    def test_core_crop_leaves_out_caption_header_and_rules(self):
        """Case 01/02: the crop held the FIGURE 5.3 tab + caption, the pink rule, the "30 Chapter 2" header and its line"""
        import figures as F
        from PIL import ImageDraw
        im = Image.new("RGB", (1000, 700), "white"); dr = ImageDraw.Draw(im)
        dr.line([0, 20, 1000, 20], fill=(230, 0, 140), width=4)              # the rule ...
        dr.rectangle([10, 10, 150, 60], fill=(230, 0, 140))                  # ... running into the label tab
        for y in (80, 100, 120):
            dr.line([10, y, 140, y], fill="black", width=3)                 # caption words
        dr.rectangle([300, 150, 600, 300], outline="black", width=4)         # the diagram
        dr.line([200, 650, 950, 650], fill="black", width=3)                # its own long line (a feedback line)
        dr.line([200, 650, 200, 300], fill="black", width=3)
        ink = F.ink_mask(im)
        b = F.core_box(ink, (190, 140, 960, 660), [(5, 5, 155, 130)])
        self.assertIsNotNone(b)
        self.assertGreaterEqual(b[0], 190); self.assertGreaterEqual(b[1], 140); self.assertGreaterEqual(b[3], 650)   # line kept
        # the LLM's box cuts real content (a box at the far right it did not explain): refused
        dr.rectangle([970, 300, 995, 600], fill="black")
        self.assertIsNone(F.core_box(F.ink_mask(im), (190, 140, 960, 660), [(5, 5, 155, 130)]))

    def test_caption_inside_figure_box_blanked(self):
        """Case 01: the crop held the pink FIGURE 5.3 tab and its caption, which the Markdown has as text too"""
        import figures as F
        from PIL import ImageDraw
        im = Image.new("RGB", (400, 300), "white"); dr = ImageDraw.Draw(im)
        dr.rectangle([10, 10, 90, 30], fill=(230, 0, 140))                 # the label tab
        dr.rectangle([150, 50, 380, 280], fill=(245, 180, 220))            # the diagram
        n = F.blank_labels(im, (0, 0, 400, 300), [(10, 10, 90, 30), (500, 0, 600, 50), (150, 50, 380, 280)], (1, 1))
        self.assertEqual(n, 1)                                              # outside / too big: left alone
        self.assertEqual(im.getpixel((50, 20)), (255, 255, 255))
        self.assertEqual(im.getpixel((200, 100)), (245, 180, 220))


class OtherLanguages(unittest.TestCase):
    """Bangla vowel signs split words apart (the coverage check saw no words, a changed vowel went unnoticed); Chinese and
    Japanese clauses were one "word" each (a changed character passed as an OCR fix)"""
    BN = "আমি বাংলায় গান গাই। বিদ্যুৎ প্রবাহ ১০ মিলিঅ্যাম্পিয়ার। এটি একটি সাধারণ বর্তনী।"
    ZH = "电阻器限制电路中的电流，单位是欧姆。电压为9伏。"

    def test_words_of_every_script(self):
        self.assertEqual(R.script_tokens("বাংলায় ১০ মিলি"), ["বাংলায়", "১০", "মিলি"])
        self.assertEqual(R.script_tokens("电阻器9伏"), ["电", "阻", "器", "9", "伏"])
        self.assertEqual(R.script_tokens("电阻器", ngram=2), ["电阻", "阻器"])
        self.assertEqual(R.script_tokens("9V battery"), ["9", "V", "battery"])
        import figures as FIG
        self.assertIn("বাংলায়", FIG._words(self.BN))

    def test_changes_are_caught(self):
        self.assertEqual(R.accept(self.BN, self.BN, {}), (True, ""))
        self.assertEqual(R.accept(self.BN, self.BN.replace("গান", "গুন"), {})[0], False)          # one vowel sign changed
        self.assertEqual(R.accept(self.BN, self.BN.replace("১০", "10"), {}), (False, "number changed or lost"))
        self.assertEqual(R.accept(self.ZH, self.ZH, {}), (True, ""))
        self.assertEqual(R.accept(self.ZH, self.ZH.replace("欧姆", "安培"), {})[0], False)          # characters changed
        self.assertEqual(R.accept(self.ZH, self.ZH.replace("，", ", "), {}), (True, ""))           # punctuation only: fine
        bad = self.ZH.replace("电流", "电留")                                                          # an OCR slip the page text fixes
        self.assertEqual(R.accept(bad, self.ZH, {}, layer=self.ZH * 3), (True, ""))

    def test_budget_by_script(self):
        self.assertLess(R._need("a resistor limits the current " * 10), 120)
        self.assertGreater(R._need(self.ZH), len(self.ZH))                                           # ~1 token per character

    def test_page_break_joins(self):
        sep = lambda n: f"\n\n{{{n}}}{'-' * 48}\n\n"
        d = Path(tempfile.mkdtemp(dir=TMP)); md = d / "doc.md"
        md.write_text(sep(1) + "电阻器限制电路中" + sep(2) + "的电流。\n\n## 第二节" + sep(3) + "বিদ্যুৎ প্রবাহ মাপা হয়" + sep(4) +
                      "মিলিঅ্যাম্পিয়ারে।" + sep(5) + "图3 电路" + sep(6) + "新的段落。")
        R.finish_markdown(md)
        self.assertEqual(md.read_text(), "电阻器限制电路中的电流。\n\n## 第二节\n\nবিদ্যুৎ প্রবাহ মাপা হয় মিলিঅ্যাম্পিয়ারে।\n\n图3 电路\n\n新的段落。\n")

    def test_text_files_in_any_encoding(self):
        import convert as C
        d = Path(tempfile.mkdtemp(dir=TMP))
        cases = {"gbk": "电阻器限制电路中的电流，单位是欧姆。电压为九伏。这是一个简单的电路。" * 3,
                 "shift_jis": "抵抗器は回路を流れる電流を制限します。電圧は九ボルトです。" * 3,
                 "cp1251": "Резистор ограничивает ток в цепи. Напряжение девять вольт. " * 3,
                 "cp1256": "المقاوم يحد من التيار في الدائرة الكهربائية. الجهد تسعة فولت. " * 3,
                 "utf-16": "বিদ্যুৎ প্রবাহ মাপা হয় মিলিঅ্যাম্পিয়ারে। " * 3,
                 "utf-8": "Ελληνικά κείμενο, résumé, 한국어 텍스트. " * 3}
        for enc, text in cases.items():
            f = d / f"{enc}.txt"; f.write_bytes(text.encode(enc))
            self.assertEqual(C.read_text(f), text, enc)
        f = d / "le.txt"; f.write_bytes(cases["utf-16"].encode("utf-16-le"))          # no BOM
        self.assertEqual(C.read_text(f), cases["utf-16"])

    def test_zip_names_from_windows(self):
        import zipfile
        for name, enc in (("中文资料.txt", "gbk"), ("日本語の資料.txt", "shift_jis"), ("Отчёт.txt", "cp1251"), ("تقرير.txt", "cp1256"),
                          ("한국어.txt", "euc_kr"), ("বাংলা.txt", "utf-8")):
            info = zipfile.ZipInfo("x"); info.filename = name.encode(enc).decode("cp437"); info.flag_bits = 0
            self.assertEqual(P.zip_name(info), name, enc)
        info = zipfile.ZipInfo("中文.txt"); info.flag_bits = 0x800                   # flagged UTF-8: kept as is
        self.assertEqual(P.zip_name(info), "中文.txt")

    def test_every_script(self):
        self.assertEqual(R.script_tokens("الْمُقَاوِم ٩ فولت"), ["الْمُقَاوِم", "٩", "فولت"])         # Arabic: vowel marks kept, Arabic digit
        self.assertEqual(R.script_tokens("Резистор 9В"), ["Резистор", "9", "В"])
        self.assertEqual(R.script_tokens("सर्किट में धारा"), ["सर्किट", "में", "धारा"])                   # Devanagari
        self.assertEqual(R.script_tokens("한국어 텍스트"), ["한국어", "텍스트"])
        self.assertEqual(R.script_tokens("ภาษาไทย"), list("ภาษาไทย")[:2] + ["ษ", "า", "ไ", "ท", "ย"][:0] + R.script_tokens("ภาษาไทย")[2:])
        tib = "བོད་ཡིག" * 4                                                               # Tibetan: syllables between tsheg marks
        self.assertTrue(all(len(t) < 20 for t in R.script_tokens(tib)))
        long_run = "ꦗꦮꦺꦱ꧀ꦏꦿꦶꦥ꧀ꦠ꧀" * 5                                                       # Javanese: no spaces, not listed
        self.assertTrue(all(len(t) <= 3 for t in R.script_tokens(long_run)))

    def test_page_break_joins_cased_scripts(self):
        sep = lambda n: f"\n\n{{{n}}}{'-' * 48}\n\n"
        d = Path(tempfile.mkdtemp(dir=TMP)); md = d / "doc.md"
        md.write_text(sep(1) + "Резистор ограничивает" + sep(2) + "ток в цепи." + sep(3) + "Рис. 5 Схема" + sep(4) + "новый абзац.")
        R.finish_markdown(md)
        self.assertEqual(md.read_text(), "Резистор ограничивает ток в цепи.\n\nРис. 5 Схема\n\nновый абзац.\n")

    def test_captions_in_any_language_reach_the_describer(self):
        import figures as FIG
        for cap in ("图3 电阻器的电路", "Рис. 5 Схема цепи", "চিত্র ৪ একটি বর্তনী", "شكل 3 دائرة كهربائية", "**Figure 5-79** An Arduino Uno board"):
            md = f"Some text before.\n\n![](a.jpeg)\n\n{cap}\n\nMore text.\n"
            self.assertIn(cap, FIG._context(md, "a.jpeg"), cap)

    def test_old_bangla_fonts(self):
        import convert as C
        for name in ("SutonnyMJ", "ABCDEF+SutonnyOMJ", "Bijoy-Ekushey", "KongshoMJ-Bold"):
            self.assertTrue(C.LEGACY_FONT.search(name.split("+")[-1]), name)
        for name in ("NotoSansBengali", "SolaimanLipi", "Kalpurush", "Times-Roman", "MinionPro-Regular"):
            self.assertFalse(C.LEGACY_FONT.search(name), name)


class TestSuiteFindings(unittest.TestCase):
    """what the 27-page Doc2MD test suite showed"""
    def test_latex_survives_json(self):
        raw = '{"corrected_equation": "<math>T = (\\boldsymbol{\\rho}_1 - \\rho_2) + \\nabla_w \\tilde{J} \\times \\frac{1}{2} \\text{if} \\beta</math>\\nNext \\u00e9"}'
        self.assertEqual(R.latex_json_loads(raw)["corrected_equation"],
                         "<math>T = (\\boldsymbol{\\rho}_1 - \\rho_2) + \\nabla_w \\tilde{J} \\times \\frac{1}{2} \\text{if} \\beta</math>\nNext é")

    def test_marker_llm_rewrites_are_checked(self):
        prompt = lambda t: f"instructions\n**Input:**\n```html\n{t}\n```\n"
        self.assertEqual(R.check_marker_llm(prompt("Q = a + b"), {"analysis": "", "corrected_equation":
                         "The equation in the image appears to be a sequence of shapes. The minus sign is likely a typo."}),
                         (False, "commentary instead of content"))
        self.assertFalse(R.check_marker_llm(prompt("Powers of (1 + \\sqrt{2})"), {"corrected_html": "Powers of <math>(1 + \\sqrt[2]{2})</math>"})[0])
        self.assertFalse(R.check_marker_llm(prompt("floor brackets (\\lfloor ... \\rfloor)"),
                                            {"corrected_html": "floor brackets (<math>\\lfloor floor ... \\rfloor floor</math>)"})[0])
        self.assertEqual(R.check_marker_llm(prompt("the sum x2 + y2 for n = 5"), {"corrected_html": "the sum <math>x^2 + y^2</math> for <math>n = 5</math>"}), (True, ""))
        self.assertEqual(R.check_marker_llm(prompt("ok"), {"corrected_html": "No corrections needed"}), (True, ""))

    def test_screenshot_is_not_a_missing_figure(self):
        import figures as FIG
        shot = (50, 50, 550, 750)
        text = [(60, 60, 540, 300), (60, 320, 540, 700)]
        self.assertGreater(FIG.text_share(shot, text), 0.25)                 # converted to text already: not added again
        self.assertEqual(FIG.text_share((50, 50, 150, 150), text), FIG.text_share((50, 50, 150, 150), text))
        self.assertLess(FIG.text_share((600, 50, 700, 150), text), 0.01)

    def test_dense_picture_read_in_tiles(self):
        import figures as FIG
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (2400, 3300), "white").save(d / "_page_23_Figure_51.jpeg")
        md = d / "doc.md"; md.write_text("![](_page_23_Figure_51.jpeg)\n")
        calls = {"desc": 0, "tiles": 0}
        old_ask, old_img = FIG._ask, FIG._ask_image

        def ask_image(u, m, path, prompt, max_tokens=2000, **k):
            calls["desc"] += 1
            if 'no details' in prompt:
                return {"kind": "text", "summary": "A newspaper front page.", "details": ""}
            raise ValueError("cut off")
        FIG._ask_image = ask_image
        FIG._ask = lambda u, m, im, prompt, n, schema=None, timeout=900, extra=None: (
            (calls.__setitem__("tiles", calls["tiles"] + 1) or f"tile text {im.size}") if prompt == FIG.TRANSCRIBE_PROMPT
            else ("To Auctioneers, etc.\n" * 12) if prompt == FIG.TRANSCRIBE_ONE_PROMPT    # one answer loops on a newspaper page
            else "{}", "stop")                                                                  # (the core crop asks too)
        try:
            self.assertEqual(FIG.describe_figures(md, d, "u", "m"), 1)
        finally:
            FIG._ask, FIG._ask_image = old_ask, old_img
        out = md.read_text()
        self.assertEqual(calls["desc"], 4); self.assertEqual(calls["tiles"], 6)                    # 2 x 3 tiles of ~1100 px
        self.assertIn("> **Figure content:** A newspaper front page.\n>\n> tile text", out)

    def test_text_layers_that_need_ocr(self):
        import types, convert as C
        pages = {"bn": "আমাদরে মাতৃভাষা বাংলা। ১৯৫২ সালরে ২১শে ফব্রুয়ারি রাষ্টরভাষা বাংলার দাবতিে ঢাকার রাজপথে",
                 "dots": "Section 1: Native Digital Bengali Script (····· ······· ····· ·······)",
                 "toc": "Chapter 1 ............ 5\nChapter 2 ············ 9", "en": "A resistor limits the current."}
        for key, want in (("bn", None), ("dots", "no Unicode"), ("toc", None), ("en", None)):   # Indic: repaired from pdfium, not OCR
            doc = type("Doc", (), {"__len__": lambda self: 1, "close": lambda self: None,
                                   "__getitem__": lambda self, i, t=pages[key]: types.SimpleNamespace(
                                       get_textpage=lambda: types.SimpleNamespace(get_text_range=lambda: t, close=lambda: None))})()
            old = sys.modules.get("pypdfium2"); old_font = C.legacy_font
            sys.modules["pypdfium2"] = types.SimpleNamespace(PdfDocument=lambda path: doc); C.legacy_font = lambda path: None
            try:
                got = C.ocr_reason("x.pdf", [0])
            finally:
                C.legacy_font = old_font
                sys.modules.pop("pypdfium2") if old is None else sys.modules.__setitem__("pypdfium2", old)
            self.assertEqual(bool(got), bool(want), key)
            if want:
                self.assertIn(want, got)
        self.assertEqual(C.page_list("3-5,9"), [3, 4, 5, 9])

    def test_indic_text_repaired_from_the_pdf(self):
        """Case 26: Marker's text had আমাদরে / ফব্রুয়ারি / ইউনস্কেণ্ট where the PDF's own text (pdfium) is right"""
        import types, convert as C
        pdf_text = ("আমাদের মাতৃভাষা বাংলা। ১৯৫২ সালের ২১শে ফেব্রুয়ারি রাষ্ট্রভাষা বাংলার দাবিতে ঢাক\r\nার রাজপথে। "
                    "পরবর্তীতে ১৯৯৯ সালে ইউনেস্কো ২১শে ফেব্রুয়ারিকে আন্তর্জাতিক মাতৃভাষা দিবস হিসেবে স্বীকৃতি দেয়। পরীক্ষা করার জন্য তৈরি।")
        out = ("আমাদরে মাতৃভাষা বাংলা। ১৯৫২ সালরে ২১শে ফব্রুয়ারি রাষ্টরভাষা বাংলার দাবতিে ঢাক ার রাজপথে। "
               "পরবর্তীতে ১৯৯৯ সালে ইউনস্কেণ্ট ২১শে ফব্রুয়ারকিে আন্তর্জাতকি মাতৃভাষা দবিস হসিবেে স্বীকৃতি দেয়। পরীক্ষা করে।")
        want = ("আমাদের মাতৃভাষা বাংলা। ১৯৫২ সালের ২১শে ফেব্রুয়ারি রাষ্ট্রভাষা বাংলার দাবিতে ঢাকার রাজপথে। "
                "পরবর্তীতে ১৯৯৯ সালে ইউনেস্কো ২১শে ফেব্রুয়ারিকে আন্তর্জাতিক মাতৃভাষা দিবস হিসেবে স্বীকৃতি দেয়। পরীক্ষা করার জন্য তৈরি।")
        d = Path(tempfile.mkdtemp(dir=TMP)); md = d / "doc.md"
        md.write_text(f"\n\n{{0}}{'-' * 48}\n\n" + R.tidy(out))
        page = types.SimpleNamespace(get_textpage=lambda: types.SimpleNamespace(get_text_range=lambda: pdf_text, close=lambda: None))
        doc = type("Doc", (), {"__len__": lambda self: 1, "__getitem__": lambda self, i: page, "close": lambda self: None})()
        old = sys.modules.get("pypdfium2")
        sys.modules["pypdfium2"] = types.SimpleNamespace(PdfDocument=lambda path: doc)
        try:
            self.assertGreater(C.repair_indic(md, "x.pdf"), 5)
        finally:
            sys.modules.pop("pypdfium2") if old is None else sys.modules.__setitem__("pypdfium2", old)
        R.finish_markdown(md)
        self.assertEqual(md.read_text().strip(), want)

    def test_scrambled_pdf_text_is_not_trusted(self):
        import convert as C
        self.assertEqual(C._indic_invalid(["েদশ", "িকছু", "বাংলা"]), 2)        # vowel signs first: visual order, not trusted
        self.assertEqual(C._indic_skeleton("আমাদের"), C._indic_skeleton("আমাদরে"))

    def test_footer_from_layer_and_inline_backticks(self):
        """Case 01: the rewrite restored ```mermaid (OCR read ``) and Marker's dropped footer from the text layer"""
        draft = "Flowchart (``mermaid flowchart TD) list of steps to do"
        layer = "Flowchart (```mermaid flowchart TD) list of steps to do\nDocument-to-Markdown Benchmark | Page 2 of 27"
        new = "Flowchart (```mermaid flowchart TD) list of steps to do\n\nDocument-to-Markdown Benchmark | Page 2 of 27"
        self.assertEqual(R.accept(draft, new, {}, layer), (True, ""))
        self.assertEqual(R.accept(draft, new + " of of", {}, layer)[1], "words added")    # more than draft + layer
        self.assertEqual(R.accept(draft, new + " 27", {}, layer)[1], "number changed or lost")

    def test_stale_list_numbers(self):
        """Case 02: Marker wrote "1. 2. *Incremental development*" for the page's item 2; the review's fix was rejected"""
        d = "1. 2. *Incremental development* This approach\n2. 3. *Reuse-oriented* That one"
        new = "2. *Incremental development* This approach\n3. *Reuse-oriented* That one"
        self.assertEqual(R.tidy(d).strip(), new)
        self.assertEqual(R.accept(d, new, {}), (True, ""))
        self.assertEqual(R.tidy("See 1. 2. here").strip(), "See 1. 2. here")          # not at a line start: text

    def test_marker_llm_commentary_never_replaces_an_equation(self):
        """Cases 15/16: two equations became "The LaTeX code is correct, but the equation number (2.55) is not properly
        aligned..." and "The equation is correctly formatted with math tags..." """
        prompt = "**Input:**\n```html\n<math display=\"block\">p_n(y) = \\sum_{m=0}^n y^m</math>\n```"
        for reply in ("The LaTeX code is correct, but the equation number (2.55) is not properly aligned with the equation. "
                      "It should be placed after the equation.",
                      "The equation is correctly formatted with math tags and LaTeX, but the condition should be written as "
                      "n \\geq 0 to ensure KaTeX compatibility.",
                      "This shows how the sum of the terms over every index grows with the size of the polynomial degree."):
            self.assertFalse(R.check_marker_llm(prompt, {"corrected_html": reply})[0], reply[:30])
        self.assertTrue(R.check_marker_llm(prompt, {"corrected_html": "<math display=\"block\">p_n(y) = \\sum_{m=0}^n y^m</math>"})[0])

    def test_equation_numbers_become_tags(self):
        r"""Cases 06/13/16: "(7.3)" on its own line, "\quad (7.5)" outside the maths, "\quad \text{(7.7)}" inside it"""
        self.assertEqual(R.tidy("$$a = b.$$\n\n(7.3)\n\nNext"), "$$a = b. \\tag{7.3}$$\n\nNext\n")
        self.assertEqual(R.tidy("$$a = b.$$\n\\quad (7.5)\n\nIf"), "$$a = b. \\tag{7.5}$$\n\nIf\n")
        self.assertEqual(R.tidy("$$Hw = c \\quad \\text{(7.7)}$$"), "$$Hw = c \\tag{7.7}$$\n")
        self.assertEqual(R.tidy("$$\nx = y\n$$\n\n(2.54)"), "$$\nx = y \\tag{2.54}\n$$\n")
        for same in ("$$f(x) = (x+1)$$\n\nText (3) here\n", "\\$\\$ a = b \\$\\$\n\n(4.1)\n", "```\n$$x$$\n\n(1)\n```\n"):
            self.assertEqual(R.tidy(same), same)                       # a closing bracket, printed $$ text, code: untouched
        self.assertTrue(R.accept("$$a = b.$$\n\n(7.3)", "$$a = b. \\tag{7.3}$$", {})[0])

    def test_unbalanced_display_maths_closed(self):
        self.assertEqual(R.tidy("$$L(\\theta) = x_2^2$"), "$$L(\\theta) = x_2^2$$\n")
        self.assertEqual(R.tidy("costs $$5 $"), "costs $$5 $\n")                # not a line of maths

    def test_table_html_from_marker(self):
        """Case 05: icon cells came out as <img alt="surfel"><br/>surfel</img>, and <b>P</b>"""
        out = R.tidy('| <img alt="surfel"><br/>surfel</img> | <img alt="lone icon"/> | <b>P</b> (see below) |')
        self.assertEqual(out.strip(), "| surfel | lone icon | **P** (see below) |")

    def test_heading_not_flattened_by_review(self):
        draft = "## Case 08: Flight Delay Plot\n\nSome text here."
        self.assertEqual(R.accept(draft, "Case 08: Flight Delay Plot\n\nSome text here.", {})[1], "heading lost")
        self.assertTrue(R.accept(draft, "## Case 08: Flight Delay Plot\n\nSome text here.", {})[0])

    def test_staggered_table_realigned_from_the_image(self):
        """Case 14: the lower rows of a staggered table slid left ("26 49" under columns 5-6, not 6-7)"""
        import figures as F
        from PIL import ImageDraw
        im = Image.new("RGB", (700, 120), "white"); d = ImageDraw.Draw(im)
        xs = [40 + 60 * c for c in range(10)]                               # right edges of 10 columns
        page = [["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"], ["1", "", "3", "", "", "6", "", "", "", "10"], ["1", "", "", "", "", "6", "", "", "", ""]]
        for r, row in enumerate(page):
            for c, v in enumerate(row):
                if v:
                    w = 9 * len(v)
                    d.rectangle([xs[c] - w, 20 + 35 * r, xs[c], 34 + 35 * r], fill="black")   # a value, right-aligned
        written = [page[0], ["1", "3", "6", "10", "", "", "", "", "", ""], ["1", "6", "", "", "", "", "", "", "", ""]]
        self.assertEqual(F.realign_table(written, F.ink_mask(im)), page)
        self.assertIsNone(F.realign_table(written[:2], F.ink_mask(im)))        # rows don't match the image: left alone
        self.assertIsNone(F.realign_table([page[0], ["1", "3", "6"], page[2]], F.ink_mask(im)))   # value count differs

    def test_table_cell_pictures_cropped_from_the_grid(self):
        """Case 05: icon cells came out as <img alt="surfel"> with no file"""
        import figures as F
        from PIL import ImageDraw
        im = Image.new("RGB", (400, 300), "white"); d = ImageDraw.Draw(im)
        for x in (10, 200, 390):
            d.line([x, 10, x, 290], fill="black", width=2)
        for y in (10, 50, 290):
            d.line([10, y, 390, y], fill="black", width=2)
        d.ellipse([60, 80, 140, 160], fill=(90, 90, 90))                     # the picture
        d.rectangle([40, 250, 120, 262], fill="black")                      # its label
        ink = F.ink_mask(im)
        cells = F.grid_cells(ink, 2, 2)
        self.assertIsNotNone(cells)
        x0, y0, x1, y1 = (int(v) for v in cells[1][0])
        pic = F.cell_picture(ink[y0:y1, x0:x1])
        self.assertTrue(abs(x0 + pic[0] - 60) <= 3 and abs(y0 + pic[1] - 80) <= 3 and abs(y0 + pic[3] - 161) <= 3)
        x0, y0, x1, y1 = (int(v) for v in cells[1][1])
        self.assertIsNone(F.cell_picture(ink[y0:y1, x0:x1]))               # an empty cell has no picture

    def test_formula_of_drawings_is_a_picture(self):
        import figures as F
        self.assertTrue(F.drawn_formula(r"Q = \text{---} + \triangle + \square + \triangle + \square + \cdots"))
        self.assertFalse(F.drawn_formula(r"\square x = y"))

    def test_page_text_settles_a_refused_paragraph(self):
        r"""Case 15: the OCR's \sqrt[2]{2} where the page prints \sqrt{2}; the rewrite also unescaped the printed \$"""
        layer = "Case 15: Powers of $(1 + \\sqrt{2})$\nDOC2MD TEST SUITE\nA 9V battery feeds the 470 ohm resistor."
        d = "## Case 15: Powers of \\$(1 + \\sqrt[2]{2})\\$\n\nA 9V battery feeds the 470 ohm resistor."
        n = "## Case 15: Powers of $(1 + \\sqrt{2})$\n\nA V battery feeds the 470 ohm resistor."
        out, taken = R.merge_paragraphs(d, n, {}, layer)
        self.assertEqual(out, "## Case 15: Powers of \\$(1 + \\sqrt{2})\\$\n\nA 9V battery feeds the 470 ohm resistor.")
        self.assertEqual(taken, 1)

    def test_drawn_box_for_ocr_zero(self):
        """Case 15 ex. 51: the domino glyphs were read as O / 0"""
        self.assertTrue(R.accept("the coefficient of 0^j 0^k is", "the coefficient of ▯^j ▭^k is", {})[0])
        self.assertFalse(R.accept("the coefficient of 0^j 0^k is", "the coefficient of x^j y^k is", {})[0])

    def test_equation_rewrite_with_other_letters_is_refused(self):
        """Case 03 (run 3): Marker's equation step turned the OCR's λ / ε (for ê) into \\boldsymbol{\\nu}"""
        p = "**Input:**\n```html\n<math display=\"block\">T = (λ_1 - λ_2)ε_1ε_1^T</math>\n```"
        self.assertEqual(R.check_marker_llm(p, {"corrected_equation": "<math>T = (\\boldsymbol{\\nu_1} - \\boldsymbol{\\nu_2})</math>"}),
                         (False, "symbols swapped"))
        self.assertTrue(R.check_marker_llm(p, {"corrected_equation": "<math>T = (\\lambda_1 - \\lambda_2)\\hat{e}_1\\hat{e}_1^T</math>"})[0])

    def test_checkbox_state_is_content(self):
        draft = "- [x] Step 1: Pre-process\n- [ ] Step 3: Transcribe"
        self.assertEqual(R.accept(draft, draft.replace("[x] Step 1", "[ ] Step 1"), {})[1], "checkbox changed")
        self.assertTrue(R.accept("[X] Step 1: Pre-process", "- [x] Step 1: Pre-process", {})[0])

    def test_audit_defers_to_the_pages_text(self):
        """accurate run 1: the audit "completed" "Figure 2.1 T", which the page itself cuts off; a second look agreed"""
        section = "**CHALLENGE:** figure header/caption binding ('Figure 2.1 T"
        fixes = [{"find": "('Figure 2.1 T", "replace": "('Figure 2.1", "why": "stray T"}]
        old = R.ask_page
        R.ask_page = lambda u, m, j, prompt, n, timeout=900, schema=None: (
            (json.dumps({"fixes": fixes}) if "proofread" in prompt else json.dumps({"answer": "A"})), "stop")
        stats = {}
        try:
            out = R.audit_section(section, b"j", "CHALLENGE: figure header/caption binding ('Figure 2.1 T", "u", "m", stats)
        finally:
            R.ask_page = old
        self.assertEqual(out, section)
        self.assertEqual(stats.get("the page's text has it as written"), 1)

    def test_long_paragraph_is_its_own_section(self):
        """accurate run 1: two pages kept their draft because one paragraph was longer than a 900-token section"""
        long_para = " ".join(f"word{i}" for i in range(600))                 # fits the context, not a 300-token section
        draft = "Intro line.\n\n" + long_para + "\n\nOutro line."
        calls = []
        old = R.ask_page
        R.ask_page = lambda u, m, j, prompt, n, timeout=900, schema=None: (calls.append(1) or prompt.split("DRAFT:\n", 1)[1].split("\n\nTEXT LAYER", 1)[0], "stop")
        try:
            text, why = R.review_page(draft, b"j", "", "u", "m", 8192, part_tokens=300)
        finally:
            R.ask_page = old
        self.assertNotEqual(why, "page too long for the LLM context")
        self.assertGreaterEqual(len(calls), 2)

    def test_grid_from_a_box_inside_the_table_border(self):
        import figures as F
        from PIL import ImageDraw
        im = Image.new("RGB", (400, 300), "white"); d = ImageDraw.Draw(im)
        for x in (10, 200, 390):
            d.line([x, 10, x, 290], fill="black", width=2)
        for y in (10, 50, 290):
            d.line([10, y, 390, y], fill="black", width=2)
        inner = im.crop((14, 14, 386, 286))                                   # Marker's box just inside the outer lines
        self.assertIsNotNone(F.grid_cells(F.ink_mask(inner), 2, 2))

    def test_text_layer_repair(self):
        r"""the OCR's \sqrt[2]{2} where the PDF prints \sqrt{2}; the audit's "Figure 2.1" where the page cuts it off at "Figure 2.1 T".
        Never: list markers, line-break spaces, Markdown syntax, a punctuation swap, a backslash from the layer, maths lines"""
        import convert as C
        layer = ["Case 15: Generating Functions & Powers of $(1 + \\sqrt{2})$",
                 "CHALLENGE: Cascading box layout, figure header/caption binding ('Figure 2.1 T",
                 "· Ponnyo 01: Computer Monitor 27-inch 4K UHD - 38,500 Taka",
                 "[X] Step 1: Pre-process PDF document and detect embedded raster images",
                 "Authentic Scanned Bengali Broad· Newspaper (Anandabazar Patrika)",
                 "The value of x is 5 in this long line of text here"]
        md = ("## Case 15: Generating Functions & Powers of \\$(1 + \\sqrt[2]{2})\\$\\$\n"
              "**CHALLENGE:** Cascading box layout, figure header/caption binding ('Figure 2.1  \n"
              "- Ponnyo 01: Computer Monitor 27-inch 4K UHD - 38,500 Taka\n"
              "- [x] Step 1: Pre-process PDF document and detect embedded raster images\n"
              "## Authentic Scanned Bengali Broad- Newspaper (Anandabazar Patrika)\n"
              "The value of $x$ is 5 in this long line of text here")
        out, n = C.repair_lines(md, layer)
        lines = out.split("\n")
        self.assertEqual(lines[0], "## Case 15: Generating Functions & Powers of \\$(1 + \\sqrt{2})\\$")
        self.assertEqual(lines[1], "**CHALLENGE:** Cascading box layout, figure header/caption binding ('Figure 2.1 T  ")
        self.assertEqual(lines[2:], md.split("\n")[2:])                        # markers, punctuation swap, maths: untouched

    def test_audit_skips_what_the_text_layer_settles(self):
        layer = "The quick brown fox jumps over the lazy dog near 42 river banks."
        self.assertTrue(R.settled("The quick brown fox jumps over the lazy dog near 42 river banks.", layer))
        self.assertFalse(R.settled("The quick brown fox jumps over the lazy cat near 42 river banks.", layer))   # an OCR word
        self.assertFalse(R.settled("The quick brown fox $x^2$ jumps over the lazy dog.", layer))                  # maths
        self.assertFalse(R.settled("The quick brown fox jumps over the lazy dog.", ""))                           # no layer
        calls = []
        old = R.ask_page
        R.ask_page = lambda *a, **k: (calls.append(1) or json.dumps({"fixes": []}), "stop")
        stats = {}
        try:
            out = R.audit_page("The quick brown fox jumps over the lazy dog near 42 river banks.\n\nAn OCR wrod here.",
                               b"j", layer, "u", "m", stats, sizes=(1, 1))
            self.assertIn("lazy dog", out)
        finally:
            R.ask_page = old
        self.assertEqual(len(calls), 2)                                   # 2 passes, only the section the layer can't settle
        self.assertEqual(stats["settled by the text layer"], 2)
        self.assertNotIn("_lock", R._public(stats))

    def test_each_page_audited_right_after_its_review(self):
        d = Path(tempfile.mkdtemp(dir=TMP)); md = d / "doc.md"
        md.write_text(f"\n\n{{0}}{'-' * 48}\n\nPage one text.\n\n{{1}}{'-' * 48}\n\nPage two text.\n")
        old_in, old_ask = R.page_inputs, R.ask_page
        R.page_inputs = lambda src, ids: {i: (b"j", "") for i in ids}
        R.ask_page = lambda *a, **k: (a[3].split("DRAFT:\n", 1)[1].strip(), "stop")
        seen = []
        try:
            R.review_pages(md, "x.pdf", "u", "m", after=lambda text, jpeg, layer: seen.append(text) or text + " (audited)")
        finally:
            R.page_inputs, R.ask_page = old_in, old_ask
        self.assertEqual(sorted(seen), ["Page one text.", "Page two text."])
        self.assertIn("Page two text. (audited)", md.read_text())

    def test_ceiling_and_floor_are_not_swapped_by_a_rewrite(self):
        """Case 15 ex. 49b: the OCR read the page's ceiling right; Marker's LLM step made it a floor"""
        d = "Prove that $\\lceil (1 + \\sqrt{2})^n \\rceil \\equiv n \\pmod{2}$ for all n."
        self.assertEqual(R.accept(d, d.replace("ceil", "floor"), {})[1], "ceiling/floor changed")
        p = "**Input:**\n```html\n" + d + "\n```"
        self.assertEqual(R.check_marker_llm(p, {"corrected_html": d.replace("ceil", "floor")}), (False, "ceiling/floor changed"))
        self.assertTrue(R.check_marker_llm(p, {"corrected_html": d})[0])

    def test_audit_never_strips_maths_or_formatting(self):
        r"""Cases 1/5/15, accurate: the test page's ground-truth line repeats a caption in plain text; the audit "fixed" the
        caption to it: "( $\lambda_i$ are" -> "(lambda_i are", "*stick*" -> "stick" """
        section = "(a) A 3-D generic tensor ( $\\lambda_i$ are its eigenvalues) (b) Decomposition into the *stick* parts"
        fixes = [{"find": "( $\\lambda_i$ are its", "replace": "(lambda_i are its", "why": ""},
                 {"find": "into the *stick* parts", "replace": "into the stick parts", "why": ""}]
        old = R.ask_page
        R.ask_page = lambda u, m, j, prompt, n, timeout=900, schema=None: (
            (json.dumps({"fixes": fixes}) if "proofread" in prompt else json.dumps({"answer": "A"})), "stop")
        stats = {}
        try:
            out = R.audit_section(section, b"j", "Captions: (a) A 3-D generic tensor (lambda_i are its eigenvalues) into the stick parts",
                                  "u", "m", stats)
        finally:
            R.ask_page = old
        self.assertEqual(out, section)
        self.assertEqual(stats.get("would drop maths or formatting"), 2)

    def test_second_look_asked_both_ways(self):
        """a model that always answers "A" confirms nothing (it did confirm ceiling -> floor once)"""
        old = R.ask_page
        R.ask_page = lambda *a, **k: (json.dumps({"answer": "A"}), "stop")
        try:
            self.assertFalse(R._verify(b"j", "u", "m", "ctx", "\\lceil x \\rceil", "\\lfloor x \\rfloor"))
        finally:
            R.ask_page = old
        R.ask_page = lambda u, m, j, prompt, n, timeout=900, schema=None: (
            json.dumps({"answer": "A" if prompt.index("孪") < prompt.index("李") else "B"}), "stop")   # always the right one
        try:
            self.assertTrue(R._verify(b"j", "u", "m", "ctx", "数字李生", "数字孪生"))
        finally:
            R.ask_page = old

    def test_drawn_formula_found_inside_a_sentence(self):
        """Cases 1/5/15 accurate: after Marker's inline-maths rewrite the polygon formula sat in the same paragraph as its
        sentence; only the formula is replaced, the sentence stays"""
        import figures as F
        body = r"**50** Consider polygons: Q = $\_ + \triangle + \square + \triangle + \square + \cdots$. Find $Q$."
        spans = [m.group(0) for m in F.DRAWN_SPAN.finditer(body) if F.drawn_formula(m.group(0))]
        self.assertEqual(spans, [r"Q = $\_ + \triangle + \square + \triangle + \square + \cdots$."])

    def test_inline_drawn_formula_paragraph(self):
        import figures as F
        self.assertTrue(F._drawn_paragraph(r"Q = $\_ + \triangle + \square + \triangle + \square + \cdots$."))
        self.assertTrue(F._drawn_paragraph(r"$$Q = \text{---} + \triangle + \square + \triangle$$"))
        self.assertFalse(F._drawn_paragraph(r"Shapes $\triangle + \square + \square$ appear in this long sentence about tilings."))

    def test_spare_ram_runs_more_workers_and_starts_are_budgeted(self):
        """Kaggle had 8-10 GB of RAM left in every run: up to 1.5 workers per vCPU when the RAM holds them; new pieces start
        only while free RAM covers their growth above the pause level"""
        import pipeline as P
        GB = 2 ** 30
        self.assertEqual(P.workers_for_memory(4, 31 * GB, 6 * GB, 3.5, 3.0, 1.5), 6)       # 22 GB free: 6 workers (cap 1.5 x 4)
        self.assertEqual(P.workers_for_memory(4, 31 * GB, 18 * GB, 3.5, 3.0, 1.5), 2)      # little RAM: fewer, never more than fits
        self.assertEqual(P.workers_for_memory(4, 31 * GB, 6 * GB, 3.5, 3.0), 4)            # per_cpu 1: one per vCPU as before
        self.assertEqual(P.start_budget(9.0, 4.0, 2.5), 2)                                 # 9 GB free: two pieces may start now
        self.assertEqual(P.start_budget(5.0, 4.0, 2.5), 0)                                 # not enough for one more
        self.assertEqual(P.start_budget(2.0, 4.0, 2.5), 0)

    def test_load_balancer_keeps_a_page_on_one_server(self):
        """measured: a page image is 2.8 s of GPU work the first time, 0.06 s when the same server has it; least-busy routing
        reused it for 2 of 48 calls"""
        import importlib, sys as _sys
        old = _sys.argv
        _sys.argv = ["lb.py", "9999", "8181", "8182"]
        try:
            lb = importlib.reload(importlib.import_module("lb"))
        finally:
            _sys.argv = old
        a, b = lb.BACKENDS
        body = b'{"messages":[{"content":[{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,QUJDRA=="}}]}]}'
        key = lb.image_key(body)
        self.assertEqual(key, lb.image_key(body.replace(b"image_url", b"image_url")))
        self.assertIsNone(lb.image_key(b'{"messages":[{"content":"text only"}]}'))
        lb.inflight.update({a: 0, b: 0})
        lb.remember(key, b)
        lb.inflight.update({a: 0, b: 2})
        self.assertEqual(lb.order(key)[0], b)                                  # busier, but holds the image: still b
        lb.inflight.update({a: 0, b: 9})
        self.assertEqual(lb.order(key)[0], a)                                  # much busier: spills over to a
        self.assertEqual(lb.order(None)[0], a)                                 # no image: least busy

    def test_llm_requests_in_flight_are_capped(self):
        import threading, time, urllib.request
        R.set_llm_inflight(2)
        peak, now, lock = [0], [0], threading.Lock()

        class Fake:
            def __enter__(self):
                with lock:
                    now[0] += 1; peak[0] = max(peak[0], now[0])
                time.sleep(0.05)
                return self

            def __exit__(self, *a):
                with lock:
                    now[0] -= 1

            def read(self):
                return b'{"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}'
        old = urllib.request.urlopen
        urllib.request.urlopen = lambda *a, **k: Fake()
        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(8) as ex:
                list(ex.map(lambda i: R.ask_page("http://x", "m", b"j", "p", 4), range(8)))
        finally:
            urllib.request.urlopen = old
            R.set_llm_inflight(0)
        self.assertEqual(peak[0], 2)

    def test_settled_page_is_not_rewritten(self):
        layer = " ".join(f"word{i}" for i in range(40)) + " CHAPTER 3 131"
        page = " ".join(f"word{i}" for i in range(40))
        self.assertTrue(R.page_settled(page, layer))                          # every word from the PDF's text, header aside
        self.assertFalse(R.page_settled(page.replace("word7", "wrod7"), layer))   # an OCR word
        self.assertFalse(R.page_settled(" ".join(f"word{i}" for i in range(20)), layer))   # half the page's text missing
        self.assertFalse(R.page_settled(page + " $x^2$", layer + " x 2"))     # maths to convert
        self.assertFalse(R.page_settled(page, ""))                            # no text layer
        d = Path(tempfile.mkdtemp(dir=TMP)); md = d / "doc.md"
        md.write_text(f"\n\n{{0}}{'-' * 48}\n\n{page}\n")
        old_in, old_ask = R.page_inputs, R.ask_page
        R.page_inputs = lambda src, ids: {i: (b"j", layer) for i in ids}
        calls = []
        R.ask_page = lambda *a, **k: (calls.append(1) or page, "stop")
        try:
            st = R.review_pages(md, "x.pdf", "u", "m", skip_settled=True)
        finally:
            R.page_inputs, R.ask_page = old_in, old_ask
        self.assertEqual(calls, []); self.assertEqual(st["kept_draft"], {"settled by the text layer": 1})

    def test_text_picture_transcribed_tile_by_tile(self):
        """the 1900 newspaper and Tagore's letter are pictures of text (kind "text"): one description can't hold them"""
        import figures as FIG
        d = Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (2200, 2400), "white").save(d / "_page_23_Figure_51.jpeg")
        md = d / "doc.md"; md.write_text("![](_page_23_Figure_51.jpeg)\n\nAfter.\n")
        old_img, old_tiles = FIG._ask_image, FIG.transcribe_tiles
        FIG._ask_image = lambda *a, **k: {"kind": "text", "summary": "A newspaper page.", "details": "Some of it."}
        FIG.transcribe_tiles = lambda *a, **k: "COMMERCIAL HOTEL\n\nEvery column of the page, read tile by tile."
        try:
            FIG.describe_figures(md, d, "u", "m", tighten=False, graph_tries=0, transcribe_text=True)
        finally:
            FIG._ask_image, FIG.transcribe_tiles = old_img, old_tiles
        self.assertIn("COMMERCIAL HOTEL", md.read_text())

    def test_printed_latex_with_plain_dollars_repaired(self):
        """full suite: the OCR left Case 15's title as "$(1 + \\sqrt[2]{2})$$" (plain $); the repair skipped it as maths.
        A "$" line is repaired when the PDF's text has the "$" too (printed LaTeX); real maths never matches"""
        import convert as C
        layer = ["Case 15: Generating Functions & Powers of $(1 + \\sqrt{2})$", "The value x2 + y2 is computed for each pair here."]
        md = "# Case 15: Generating Functions & Powers of $(1 + \\sqrt[2]{2})$$\n\nThe value $x^2 + y^2$ is computed for each pair here."
        out, n = C.repair_lines(md, layer)
        self.assertEqual(out.split("\n")[0], "# Case 15: Generating Functions & Powers of $(1 + \\sqrt{2})$")
        self.assertIn("$x^2 + y^2$", out)

    def test_untrustworthy_text_layer_is_not_copied(self):
        """an old scan's own OCR layer can be worse than Surya's reading: when most matching lines differ, nothing is taken"""
        import convert as C
        good = ["The modern engineer must understand the circuit before building it.",
                "Every component has a rating that limits the current through it.",
                "Resistors convert electrical energy into heat in a predictable way.",
                "Capacitors store charge and release it when the voltage drops."]
        bad_layer = [l.replace("m", "rn").replace("e", "c") for l in good]     # a poor OCR layer: rn for m, c for e
        out, n = C.repair_lines("\n\n".join(good), bad_layer)
        self.assertEqual((out, n), ("\n\n".join(good), 0))
        ok_layer = list(good); ok_layer[1] = ok_layer[1].replace("limits", "limits")
        md = "\n\n".join(good[:3] + [good[3].replace("charge", "charqe")])   # one OCR slip, the layer agrees elsewhere
        out, n = C.repair_lines(md, ok_layer)
        self.assertIn("store charge and", out); self.assertEqual(n, 1)

    def test_pipes_inside_table_cells(self):
        """a "|" in a cell's maths or code split the row into extra cells (a known limit until now)"""
        row = "| $\\|w\\|_2$ | $|x| + \\left|y\\right|$ | `a || b` |"
        out = R.tidy("| A | B | C |\n|---|---|---|\n" + row).strip().split("\n")[-1]
        self.assertEqual(out, "| $\\Vert w\\Vert _2$ | $\\vert x\\vert  + \\left\\vert y\\right\\vert $ | `a \\|\\| b` |")
        self.assertEqual(R.tidy("Text $|x|$ here."), "Text $|x|$ here.\n")          # outside a table: as it was

    def test_text_picture_one_answer_first_and_loops_cleaned(self):
        """the handwritten invoice: one answer had every line and price; tiles split its lines and looped on a dot leader"""
        import figures as FIG
        self.assertTrue(FIG._looped("To Auctioneers, etc.\n" * 5))
        self.assertTrue(FIG._looped("ing mg" + ". " * 200))
        self.assertFalse(FIG._looped("8 Pieces Paper Hanging … 3 3/4\n10 Pieces Elegant Paper … 1/3\n… 15 13 3 …"))
        self.assertEqual(FIG.clean_transcript("8 Pieces Paper Hang . . . . . . . . . . 3 3/4\nA\nA\nA\nA"),
                         "8 Pieces Paper Hang … 3 3/4\nA\nA")
        d = Path(tempfile.mkdtemp(dir=TMP)); Image.new("RGB", (2000, 2300), "white").save(d / "inv.jpeg")
        old = FIG._ask
        FIG._ask = lambda *a, **k: ("Mr. Emlay\n8 Pieces Paper Hanging … 3 3/4\n… 15 13 3 …", "stop")
        try:
            self.assertIn("15 13 3", FIG.transcribe_image("u", "m", d / "inv.jpeg"))
        finally:
            FIG._ask = old

    def test_chunks_come_in_whole_rounds(self):
        """the 27-page suite was cut into 6 chunks for 5 workers: one worker ran a second round while four waited
        (36% idle in balanced). Chunk counts are a multiple of the workers, sizes as even as possible"""
        import pipeline as P
        for n, w in ((27, 5), (27, 6), (120, 5), (120, 6), (349, 6), (96, 4)):
            ch = P.plan_chunks(list(range(n)), w, 4, 24)
            self.assertEqual(len(ch) % w, 0, (n, w))
            self.assertLessEqual(max(map(len, ch)) - min(map(len, ch)), 1)
            self.assertEqual(sum(ch, []), list(range(n)))                      # every page once, in order
            self.assertTrue(all(len(c) >= 4 for c in ch))
        self.assertEqual([len(c) for c in P.plan_chunks(list(range(27)), 5, 4, 24)], [6, 6, 5, 5, 5])

    def test_quality_profiles(self):
        self.assertFalse(R.quality_option({"QUALITY": "fast"}, "PAGE_REVIEW"))
        self.assertTrue(R.quality_option({"QUALITY": "fast", "PAGE_REVIEW": True}, "PAGE_REVIEW"))    # set by the job: kept
        self.assertTrue(R.quality_option({"QUALITY": "accurate"}, "AUDIT"))
        self.assertFalse(R.quality_option({}, "REDO_INLINE_MATH"))                                    # balanced by default
        self.assertEqual(R.quality_option({"QUALITY": "accurate"}, "AUDIT_SECTIONS"), (6, 3))         # two audit passes
        self.assertFalse(R.quality_option({"QUALITY": "fast"}, "MARKER_LLM"))
        self.assertEqual(R.quality_option({"QUALITY": "nonsense"}, "GRAPH_TRIES"), 2)
        for q in ("fast", "balanced", "accurate"):
            self.assertEqual(set(R.QUALITY_PROFILES[q]), set(R.QUALITY_PROFILES["balanced"]))

    def test_planner_leaves_stages_to_the_quality(self):
        import pipeline as P
        old = dict(P.DEFAULTS)
        try:
            P.DEFAULTS.update(LLM_URL="http://x", LLM_SLOTS=4, USE_LLM=True, PAGE_REVIEW=None, DESCRIBE_FIGURES=None)
            plan = P.llm_plan({}, 4, [])
            self.assertNotIn("PAGE_REVIEW", plan); self.assertNotIn("DESCRIBE_FIGURES", plan)
            self.assertFalse(P.llm_plan({"PAGE_REVIEW": False}, 4, [])["PAGE_REVIEW"])
        finally:
            P.DEFAULTS.clear(); P.DEFAULTS.update(old)

    def test_layer_for_a_section(self):
        layer = "Alpha beta gamma delta\nSomething else entirely here\nepsilon zeta eta theta"
        self.assertEqual(R.layer_for("alpha beta gamma delta epsilon", layer), "Alpha beta gamma delta")   # 1 of 4 words: not its line

    def test_audit_applies_only_confirmed_local_fixes(self):
        r"""Cases 15/17: the audit found \sqrt[2]{2} -> \sqrt{2}, 李生 -> 孪生, 音服 -> 客服; it also lists no-ops and may be wrong"""
        section = "在数字李生生态系统中\n\n音服热线: 400-111-9811\n\nThe value is 42 here.\n\nPowers of \\$(1 + \\sqrt[2]{2})\\$"
        fixes = [{"find": "在数字李生生态系统中", "replace": "在数字孪生生态系统中", "why": "OCR"},       # confirmed by a 2nd look
                 {"find": "音服热线", "replace": "客服热线", "why": "OCR"},                             # refused by the 2nd look
                 {"find": "The value is 42 here.", "replace": "The value is 42 here.", "why": "no error"},   # no-op
                 {"find": "nowhere in it", "replace": "x", "why": ""},                                  # not found
                 {"find": "42", "replace": "an entirely different and much longer sentence about nothing", "why": ""},
                 {"find": "\\\\$(1 + \\\\sqrt[2]{2})\\\\$", "replace": "\\\\$(1 + \\\\sqrt{2})\\\\$", "why": ""}]  # doubled \\
        layer = "Powers of $(1 + \\sqrt{2})$"
        asked = []

        def ask(url, model, jpeg, prompt, n, timeout=900, schema=None):
            if "proofread" in prompt:
                return json.dumps({"fixes": fixes}), "stop"
            asked.append(prompt)
            yes = "孪生" in prompt                               # the second look confirms only the first fix
            return json.dumps({"answer": ("A" if "A: «在数字孪生" in prompt else "B") if yes else "neither"}), "stop"
        old = R.ask_page; R.ask_page = ask
        stats = {}
        try:
            out = R.audit_section(section, b"j", layer, "u", "m", stats)
        finally:
            R.ask_page = old
        self.assertIn("在数字孪生生态系统中", out)
        self.assertIn("音服热线", out)                                   # not confirmed: left as it was
        self.assertIn("The value is 42 here.", out)
        self.assertIn("Powers of \\$(1 + \\sqrt{2})\\$", out)            # confirmed by the text layer, escapes kept
        self.assertEqual(stats.get("not found"), 1); self.assertEqual(stats.get("too big"), 1)
        self.assertEqual(stats.get("refused"), 1); self.assertEqual(stats.get("confirmed by a second look"), 1)
        self.assertEqual(stats.get("confirmed by the text layer"), 1)

    def test_code_language_is_not_an_added_word(self):
        draft = "```\nlibrary(descr)\nx_tab <- CrossTable(a, b)\n```"
        self.assertEqual(R.accept(draft, draft.replace("```\nlib", "```r\nlib"), {}), (True, ""))
        detail = []
        self.assertEqual(R.accept("Page 2 of 27 text here", "text here", {}, detail=detail)[1], "number changed or lost")
        self.assertIn("lost", detail[0])

    def test_small_documents_use_every_worker(self):
        self.assertGreaterEqual(len(P.plan_chunks(list(range(27)), 4, int(P.DEFAULTS["CHUNK_MIN_PAGES"]), 24)), 4)

    def test_markdown_clean_ups(self):
        md = ("**SOURCE:** a.png\n**NOTE:** b\n\nthe loan is not expected\n\nAmra vat khai\nDhaka er rajdhani\nMot: 1,250 taka\n\n"
              "[X] Step 1: go\n[ ] Step 2: stop\n\n" + "## loop line repeated by the OCR\n\n" * 14 + "```\n      <|713|> class\n```\n")
        self.assertEqual(R.tidy(md), "**SOURCE:** a.png  \n**NOTE:** b\n\nthe loan is not expected\n\nAmra vat khai  \nDhaka er rajdhani  \n"
                                     "Mot: 1,250 taka\n\n- [x] Step 1: go\n- [ ] Step 2: stop\n\n## loop line repeated by the OCR\n\n"
                                     "```\n       class\n```\n")


class FigureDescriptions(unittest.TestCase):
    """Make Electronics: 47 figures had no description (failed calls were skipped silently) and 348 of 476 never said what
    the figure is (schematics became label lists: "9VDC / 1N4148 / 1N4148...")"""
    def setUp(self):
        import figures as FIG
        self.FIG, self.d = FIG, Path(tempfile.mkdtemp(dir=TMP))
        Image.new("RGB", (600, 400), "blue").save(self.d / "_page_293_Diagram_231.jpeg")
        Image.new("RGB", (600, 400), "green").save(self.d / "_page_316_Picture_9000.jpeg")
        self.md = self.d / "doc.md"
        self.md.write_text("If you think that it's a chip, you are not quite correct.\n\n![](_page_293_Diagram_231.jpeg)\n\n"
                           "**Figure 5-79** An Arduino Uno board, featuring the ATmega 328P-PU microcontroller.\n\n"
                           "```\nint spots = 0;\nint outpin = 0;\nvoid setup() { pinMode(0, INPUT_PULLUP); }\n```\n\n"
                           "![](_page_316_Picture_9000.jpeg)\n\nFigure 5-96 The Nicer Dice program.\n")

    def run_fig(self, answers):
        prompts, old = [], self.FIG._ask_image

        def ask(u, m, path, prompt, **k):
            prompts.append((path.name, prompt))
            a = answers[path.name].pop(0)
            if isinstance(a, Exception):
                raise a
            return a
        self.FIG._ask_image = ask
        try:
            n = self.FIG.describe_figures(self.md, self.d, "u", "m")
        finally:
            self.FIG._ask_image = old
        return n, prompts, self.md.read_text()

    def test_what_it_is_then_what_is_inside(self):
        n, prompts, out = self.run_fig({
            "_page_293_Diagram_231.jpeg": [ValueError("timeout"), {"kind": "photo", "summary": "An Arduino Uno board with its parts labelled.",
                                           "details": "- USB port (top left)\n- Reset button\n- LED linked with pin 13"}],
            "_page_316_Picture_9000.jpeg": [{"kind": "code", "summary": "The Nicer Dice program.",
                                            "details": "```c\nint spots = 0;\nint outpin = 0;\nvoid setup() { pinMode(0, INPUT_PULLUP); }\n```"}]})
        self.assertEqual(n, 2)                                                      # the failed call was asked again
        p293 = dict(prompts)["_page_293_Diagram_231.jpeg"]
        self.assertIn('caption next to it in the document (it may name only one of several figures): "**Figure 5-79** An Arduino', p293)
        self.assertIn("describe every one of them", p293)                                          # several figures in one image
        self.assertIn("not quite correct", p293)
        self.assertIn("![](_page_293_Diagram_231.jpeg)\n\n> **Figure content:** An Arduino Uno board with its parts labelled.\n>\n"
                      "> - USB port (top left)\n> - Reset button\n> - LED linked with pin 13\n\n**Figure 5-79**", out)
        self.assertIn("> **Figure content:** The Nicer Dice program. The same code is printed in the text.\n\nFigure 5-96", out)   # not copied twice

    def test_describe_again_replaces_old_text(self):
        self.md.write_text(self.md.read_text().replace("![](_page_316_Picture_9000.jpeg)\n\n",
                                                       "![](_page_316_Picture_9000.jpeg)\n\n> **Figure content:** 9VDC\n> 1N4148\n\n"))
        old = self.FIG._ask_image
        self.FIG._ask_image = lambda u, m, path, prompt, **k: {"kind": "schematic", "summary": f"new {path.name}", "details": "R1 10K to pin 2"}
        try:
            self.assertEqual(self.FIG.redescribe(self.md, self.d, "u", "m"), 2)
        finally:
            self.FIG._ask_image = old
        out = self.md.read_text()
        self.assertNotIn("1N4148", out); self.assertEqual(out.count("> **Figure content:**"), 2)
        self.assertIn("> **Figure content:** new _page_316_Picture_9000.jpeg\n>\n> R1 10K to pin 2", out)

    def test_quoted_code_survives_tidy(self):
        md = "![](a.jpeg)\n\n> **Figure content:** A listing.\n>\n> ```c\n> x = $5;\n> ```\n\n> **Figure content:** USB port\n> ```\n"
        self.assertEqual(R.tidy(md), "![](a.jpeg)\n\n> **Figure content:** A listing.\n>\n> ```c\n> x = $5;\n> ```\n\n> **Figure content:** USB port\n")


class BatchOverlap(unittest.TestCase):
    """the tail of each batch left box workers idle: the next batch is queued once every piece has started"""
    def test_two_batches_open_saved_in_turn(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("md_client", Path(__file__).parent.parent / "md_client.py")
        mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
        run = object.__new__(mc.Run)
        run.options, run.last, run.args = {}, {}, type("A", (), {"no_overlap": False})()
        jobs, waiting, finished = [], [5], []
        run.box = type("B", (), {"call": lambda self, m, path, *a, **k: {"waiting_tasks": waiting[0]}})()
        run.file_job = lambda u, b: jobs.append(b) or f"j{len(jobs)}"
        run.add_job = lambda jid, item: None
        run.finish_document = lambda u: finished.append(True)
        base = Path(tempfile.mkdtemp(dir=TMP))
        u = {"label": "book", "folder": "book", "base": base, "todo": [(0, 95), (96, 191), (192, 200)], "batch_rows": []}
        run.active = u
        run.next_batch(u)
        self.assertEqual(jobs, [(0, 95)])
        run.prefetch(); self.assertEqual(jobs, [(0, 95)])                  # pieces still waiting: no second batch yet
        waiting[0] = 0; run.last = {}
        run.prefetch(); self.assertEqual(jobs, [(0, 95), (96, 191)])      # all started: the next batch is queued
        run.last = {}; run.prefetch(); self.assertEqual(len(jobs), 2)      # never more than two open
        done = lambda b: run.finish_item({"unit": u, "batch": b, "dest": run.parts_dir(u, b), "rows": [{"ok": True, "pages": 1}]})
        run.parts_dir(u, (0, 95)).mkdir(parents=True); run.parts_dir(u, (96, 191)).mkdir(parents=True)
        run.parts_dir(u, (192, 200)).mkdir(parents=True)
        done((0, 95)); self.assertEqual(len(jobs), 2)                       # the open one moves on when it is saved
        done((96, 191)); self.assertEqual(jobs[-1], (192, 200))
        self.assertFalse(finished)
        done((192, 200)); self.assertEqual(finished, [True])                # joined once all are saved
        self.assertTrue(all((run.parts_dir(u, b) / "done.json").is_file() for b in jobs))


class PreRender(unittest.TestCase):
    def test_office_and_djvu_become_pdf_first(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        calls, old = [], (C.soffice, C.djvu_to_pdf, C.SOFFICE, C.DDJVU)

        def fake(src, outdir):
            calls.append(Path(src).name); Path(outdir).mkdir(parents=True, exist_ok=True)
            pdf = Path(outdir) / (Path(src).stem + ".pdf"); pdf.write_text("%PDF"); return pdf
        C.soffice = lambda src, fmt, outdir, profile, timeout=600: fake(src, outdir)
        C.djvu_to_pdf = lambda src, outdir, timeout=1800: fake(src, outdir)
        C.SOFFICE = C.DDJVU = "/bin/true"
        try:
            for i, name in enumerate(("a.docx", "a.djvu", "b.xls", "c.pdf", "d.txt")):
                (d / name).write_text("x")
                got = P.pre_render(d / name, i, {"OFFICE": "pdf"}, [])
                want = d / ".rendered" / str(i) / f"{Path(name).stem}.pdf" if name in ("a.docx", "a.djvu") else d / name
                self.assertEqual(got, want, name)                              # same stem "a": separate folders, no clash
            self.assertEqual(calls, ["a.docx", "a.djvu"])                      # xls stays a spreadsheet; pdf/txt untouched
            self.assertEqual(P.pre_render(d / "a.docx", 9, {"OFFICE": "native"}, []), d / "a.docx")   # native: Marker reads it
        finally:
            C.soffice, C.djvu_to_pdf, C.SOFFICE, C.DDJVU = old

    def test_failure_falls_back_to_one_piece(self):
        d = Path(tempfile.mkdtemp(dir=TMP)); (d / "a.doc").write_text("x")
        old = (C.soffice, C.SOFFICE)
        C.SOFFICE = "/bin/true"
        C.soffice = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("LibreOffice crashed"))
        try:
            warnings = []
            self.assertEqual(P.pre_render(d / "a.doc", 0, {}, warnings), d / "a.doc")
            self.assertIn("LibreOffice crashed", warnings[0])
        finally:
            C.soffice, C.SOFFICE = old


class Batches(unittest.TestCase):
    def setUp(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("md_client", Path(__file__).parent.parent / "md_client.py")
        self.mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.mc)

    def test_page_batches(self):
        self.assertEqual(self.mc.page_batches(349, 96), [(0, 95), (96, 191), (192, 287), (288, 348)])
        self.assertEqual(self.mc.page_batches(96, 96), [(0, 95)])

    def test_join_in_page_order(self):
        d = Path(tempfile.mkdtemp(dir=TMP)); parts = []
        for a, b in ((10, 19), (0, 9)):                                      # saved out of order
            pd = d / f"{a:05d}-{b:05d}" / "book"; pd.mkdir(parents=True)
            (pd / "book.md").write_text(f"pages {a}-{b}\n")
            (pd / "book_meta.json").write_text(json.dumps({"page_stats": [{"page_id": p} for p in range(a, b + 1)], "toc": a}))
            (pd / f"_page_{a}_Figure_1.jpeg").write_bytes(b"x")
            parts.append((a, pd))
        self.mc.join_batches(parts, d / "final", "book")
        self.assertEqual((d / "final" / "book.md").read_text(), "pages 0-9\n\npages 10-19\n")
        meta = json.loads((d / "final" / "book_meta.json").read_text())
        self.assertEqual([e["page_id"] for e in meta["page_stats"]], list(range(20)))
        self.assertEqual(sorted(f.name for f in (d / "final").glob("*.jpeg")), ["_page_0_Figure_1.jpeg", "_page_10_Figure_1.jpeg"])

    def test_batch_with_failed_pages_is_retried_never_saved(self):
        """batch 0-95 came back with pages 60-95 failed and was saved as done: a rerun would have skipped the hole"""
        run = object.__new__(self.mc.Run); calls = []; run.options = {}
        run.next_batch = lambda u: calls.append(list(u["todo"]))
        dest = Path(tempfile.mkdtemp(dir=TMP)); u = {"label": "book", "todo": [(96, 191)], "batch_rows": []}
        item = lambda rows: {"unit": u, "batch": (0, 95), "dest": dest, "rows": rows}
        holes = [{"ok": True, "failed_pages": ["60-71", "72-95"]}]
        for _ in range(2):                                                    # tries 2 and 3: converted again first
            run.finish_item(item(holes))
            self.assertEqual(u["todo"][0], (0, 95)); u["todo"].pop(0)
        self.assertFalse((dest / "done.json").exists())
        run.finish_item(item(holes))                                          # still failing: left out, not marked done
        self.assertFalse((dest / "done.json").exists()); self.assertEqual(u["todo"], [(96, 191)])
        u["tries"] = {}; run.finish_item(item([{"ok": True, "pages": 96}]))   # a good batch is saved
        self.assertTrue((dest / "done.json").exists())
        self.assertEqual(json.loads((dest / "done.json").read_text())["options"], self.mc.batch_signature({}, {}))   # what it was made with

    def test_combined_report_row(self):
        rows = [{"ok": True, "pages": 96, "seconds": 100, "where": "box", "batch": "0-95", "text_coverage": 0.99, "low_coverage_pages": [7]},
                {"ok": True, "pages": 96, "seconds": 120, "where": "box+pc", "batch": "96-191", "text_coverage": 0.97},
                {"ok": False, "where": "box", "batch": "192-200", "error": "boom"}]
        r = self.mc.combine_rows("book.pdf", rows, "book")
        self.assertEqual((r["ok"], r["pages"], r["batches"], r["where"], r["text_coverage"]), (True, 192, 3, "box+pc", 0.98))
        self.assertEqual((r["failed_pages"], r["low_coverage_pages"]), (["192-200"], [7]))
        self.assertIn("pages 192-200: boom", r["error"])
        rows[0].update(pages_rewritten=90, pages_kept_draft={"page text lost": 6})
        rows[1].update(pages_rewritten=95, pages_kept_draft={"page text lost": 1, "answer cut off": 1})
        r = self.mc.combine_rows("book.pdf", rows, "book")
        self.assertEqual((r["pages_rewritten"], r["pages_kept_draft"]), (185, {"page text lost": 7, "answer cut off": 1}))


class LlamaServerFlags(unittest.TestCase):
    """the notebook passed --no-mmap, which current llama.cpp no longer has: every GPU server failed with 'invalid argument'"""
    def load(self, help_text):
        src = (Path(__file__).parent / "notebook" / "19_start.py").read_text()
        block = src[src.index("LLAMA_HELP = "):src.index("SLOTS, VLM_PORTS = ")]
        ns = {"re": __import__("re"), "subprocess": type("S", (), {"run": staticmethod(
                  lambda *a, **k: type("R", (), {"stdout": help_text})())}), "VLM_ENV": "", "VLM_BIN": "llama-server",
              "log_tail": lambda name, n: "", "print": lambda *a, **k: None}
        exec(block, ns)
        return ns

    def test_current_release(self):
        ns = self.load((Path(__file__).parent / "fixtures" / "llama_server_help_b12627.md").read_text())
        self.assertEqual(ns["NO_MMAP"], "--load-mode none")
        flags = "-fa on -ctk q8_0 -ctv q8_0 --spec-type ngram-mod --threads 1 --image-max-tokens 1536 --reasoning-budget 0"
        self.assertEqual(ns["opt"](f"{ns['NO_MMAP']} {flags}"), f"--load-mode none {flags}")      # all known: all kept
        self.assertEqual(ns["opt"](ns["NO_HOST_CACHE"]), "--cache-ram 0 --ctx-checkpoints 0")      # no prompt cache in RAM

    def test_older_build_and_unknown_flags(self):
        ns = self.load("  -fa, --flash-attn [on|off]   ...\n  --no-mmap   do not memory-map\n  -t, --threads N\n")
        self.assertEqual(ns["NO_MMAP"], "--no-mmap")
        self.assertEqual(ns["opt"]("--no-mmap -fa on -ctk q8_0 -ctv q8_0 --spec-type ngram-mod --threads 1"),
                         "--no-mmap -fa on --threads 1")                       # unknown optional flags dropped with their values
        self.assertEqual(ns["DROPPED"], {"-ctk", "-ctv", "--spec-type"})

    def test_no_help_keeps_everything(self):
        ns = self.load("")
        self.assertEqual(ns["opt"]("--load-mode none -fa on"), "--load-mode none -fa on")


class ClientSurvivesTunnelLoss(unittest.TestCase):
    """the run on Make Electronics died on one Cloudflare 530; the client must wait and follow the box's new address"""
    def setUp(self):
        import importlib.util, threading, time as _t
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        spec = importlib.util.spec_from_file_location("md_client", Path(__file__).parent.parent / "md_client.py")
        self.mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.mc)

        def server(code, body):
            class H(BaseHTTPRequestHandler):
                def log_message(self, *a): pass
                def do_GET(self):
                    self.send_response(code); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            srv = ThreadingHTTPServer(("127.0.0.1", 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
            return srv, f"http://127.0.0.1:{srv.server_address[1]}"
        self.dead, self.dead_url = server(530, b"<html>Cloudflare Tunnel error</html>")
        self.live, self.live_url = server(200, b'{"status": "running"}')
        self._sleep = _t.sleep; _t.sleep = lambda s: None                # no real waiting in the test

    def tearDown(self):
        import time as _t
        _t.sleep = self._sleep; self.dead.shutdown(); self.live.shutdown()

    def test_follows_new_address(self):
        box = self.mc.Box(self.dead_url, "t" * 32, rediscover=lambda: self.live_url, patience=60)
        self.assertEqual(box.call("GET", "/jobs/abc"), {"status": "running"})
        self.assertEqual(box.url, self.live_url)

    def test_gives_up_after_patience(self):
        box = self.mc.Box(self.dead_url, "t" * 32, rediscover=lambda: None, patience=0)
        with self.assertRaises(SystemExit):
            box.call("GET", "/jobs/abc")


@unittest.skipUnless(C.SOFFICE, "LibreOffice not installed here")
class LibreOffice(unittest.TestCase):
    def test_rtf_to_pdf(self):
        d = Path(tempfile.mkdtemp(dir=TMP))
        (d / "a.rtf").write_text(r"{\rtf1\ansi Hello {\b world}.}")
        out = C.soffice(d / "a.rtf", "pdf", d / "o", d / "profile", timeout=180)
        self.assertTrue(out.read_bytes().startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
