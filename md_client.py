#!/usr/bin/env python3
"""md_client.py - send documents to your Kaggle "markdown box" and get Markdown (+ images) back.

  python md_client.py book.pdf                          # -> ./markdown_out/book/book.md (+ images, book_meta.json)
  python md_client.py book.pdf slides.ppt report.doc notes.epub data.csv --out ./md
  python md_client.py ./my_folder --out ./md            # every document in the folder (sub-folders are kept)
  python md_client.py ./my_folder --local 3             # also convert on 3 Marker workers on this PC (default: auto)
  python md_client.py book.pdf --quality express        # fastest: no vision-LLM pass (standard = default, precise = slowest)
  python md_client.py scan.pdf --mode vllm              # Marker's layout/OCR engine: Surya reads every page whole
  python md_client.py paper.pdf --pages 0-9             # only pages 0-9
  python md_client.py big.pdf --no-llm                  # skip the vision-LLM passes (faster, less exact tables/math/figure text)
  python md_client.py notes.docx --office native        # Marker reads the .docx itself (faster, keeps links; Word equations are lost)
  python md_client.py --health                          # is the box alive? GPUs, workers, idle time

Token: put the same BOX_TOKEN you stored in Kaggle Secrets into a file `.box_token` next to this script (or set BOX_TOKEN).
Address: the box announces its temporary Cloudflare URL to a private ntfy.sh topic derived from your token; this script looks
it up. If that fails, copy the `url` from the BOX READY banner in the Kaggle log and pass --url (or set BOX_URL).
Work is shared: the box takes the biggest files (and splits long PDFs over its workers), this PC's own Marker workers take the
smallest ones and page chunks of long PDFs, using the box's GPUs for the VLM through the tunnel (--local; needs Marker installed here).
Without Marker on this PC everything runs on the box, and only the Python standard library is used."""
import argparse, hashlib, hmac, importlib.util, json, os, shutil, signal, sys, tempfile, threading, time, urllib.error, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

__version__ = "2.0.0"
HERE = Path(__file__).resolve().parent
BOX_NAME = "markdown"
DOC_EXT = (".bmp", ".csv", ".djv", ".djvu", ".doc", ".docm", ".docx", ".dot", ".dotx", ".epub", ".fodp", ".fods", ".fodt", ".gif", ".htm", ".html", ".jpeg", ".jpg",
           ".markdown", ".md", ".odg", ".odp", ".ods", ".odt", ".otp", ".ots", ".ott", ".pdf", ".png", ".pot", ".potx", ".pps", ".ppsx", ".ppt",
           ".pptm", ".pptx", ".rtf", ".tif", ".tiff", ".tsv", ".txt", ".webp", ".wpd", ".wps", ".xls", ".xlsb", ".xlsm", ".xlsx")   # = convert.ALL_EXT on the box
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def die(msg, code=1):
    print("error:", msg, file=sys.stderr); sys.exit(code)


def token():
    t = os.environ.get("BOX_TOKEN", "").strip()
    for p in (HERE / ".box_token", Path.cwd() / ".box_token"):
        if not t and p.is_file():
            t = p.read_text().strip()
    if len(t) < 16:
        die("no BOX_TOKEN: put it in a file named .box_token next to md_client.py (or set the BOX_TOKEN environment variable)")
    return t


RETRY_HTTP = (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 525, 526, 527, 530)   # 530 = Cloudflare lost the tunnel
QUALITIES = ("express", "standard", "precise")          # = review.QUALITIES on the box, fastest first
QUALITY_ALIASES = {"superfast": "express", "fast": "express", "balanced": "standard", "auto": "standard", "accurate": "precise"}


class Box:
    """API client. Tunnel errors are retried for up to `patience` seconds; meanwhile it looks for a newer announced address
    (the notebook's watchdog restarts a dead tunnel under a new URL and announces it), so running jobs carry on."""

    def __init__(self, url, tok, rediscover=None, patience=600):
        self.url, self.tok, self.rediscover, self.patience = url.rstrip("/"), tok, rediscover, patience
        self._looked, self._down_since = 0.0, None

    def _trouble(self, what):
        if self._down_since is None:
            self._down_since = time.time()
            print(f"  box unreachable ({what[:80]}); retrying for up to {self.patience // 60} min and watching for a new address")
        if self.rediscover and time.time() - self._looked > 30:
            self._looked = time.time()
            new = self.rediscover()
            if new and new.rstrip("/") != self.url:
                self.url = new.rstrip("/")
                print(f"  box moved to {self.url}")

    def _ok(self):
        if self._down_since is not None:
            print(f"  box reachable again after {time.time() - self._down_since:.0f} s")
            self._down_since = None

    def call(self, method, path, body=None, headers=None, timeout=60, tries=None, raw=False):
        """tries=None: retry tunnel errors for `patience` seconds; tries=N: at most N attempts"""
        h = {"Authorization": f"Bearer {self.tok}", "User-Agent": "md-client"}
        data = None
        if isinstance(body, (dict, list)):
            data, h["Content-Type"] = json.dumps(body).encode(), "application/json"
        elif body is not None:
            data = body
        h.update(headers or {})
        last, k, t0 = None, 0, time.time()
        while True:
            try:
                req = urllib.request.Request(self.url + path, data=data, headers=h, method=method)
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    payload = r.read()
                self._ok()
                return payload if raw else (json.loads(payload) if payload else {})
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", "replace")[:400]
                if e.code not in RETRY_HTTP:
                    try:
                        detail = json.loads(detail).get("detail", detail)
                    except Exception:
                        pass
                    die(f"{method} {path}: HTTP {e.code}: {detail}")
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = repr(e)
            k += 1
            if (tries is not None and k >= tries) or (tries is None and time.time() - t0 > self.patience):
                die(f"{method} {path} failed: {last}. The box may have stopped: check the Kaggle log.")
            if k >= 2:
                self._trouble(last)
            time.sleep(min(2 ** k, 20))

    def download(self, path, dest):
        """streams to disk and resumes with Range after a dropped connection"""
        dest = Path(dest); got, k, t0 = 0, 0, time.time()
        while True:
            h = {"Authorization": f"Bearer {self.tok}", "User-Agent": "md-client"}
            if got:
                h["Range"] = f"bytes={got}-"
            try:
                with urllib.request.urlopen(urllib.request.Request(self.url + path, headers=h), timeout=120) as r:
                    if got and r.status != 206:
                        got = 0                                           # server ignored Range: start over
                    with open(dest, "ab" if got else "wb") as f:
                        while True:
                            chunk = r.read(1 << 20)
                            if not chunk:
                                break
                            f.write(chunk); got += len(chunk)
                self._ok()
                return
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_HTTP:
                    die(f"download failed: HTTP {e.code} {e.read().decode('utf-8', 'replace')[:300]}")
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
                last = repr(e)
            k += 1
            if time.time() - t0 > self.patience:
                die(f"download failed: {last}")
            if k >= 2:
                self._trouble(last)
            time.sleep(min(2 ** k, 15))


def discover(tok):
    """read the box's signed announcement from the private ntfy.sh topic derived from the token"""
    topic = "mbox-" + hashlib.sha256(("announce:" + tok).encode()).hexdigest()[:40]
    try:
        req = urllib.request.Request(f"https://ntfy.sh/{topic}/json?poll=1&since=12h", headers={"User-Agent": "md-client"})
        lines = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "replace").splitlines()
    except Exception as e:
        return None, f"could not reach ntfy.sh ({type(e).__name__})"
    best = None
    for ln in lines:
        try:
            m = json.loads(json.loads(ln).get("message", ""))
            ok = hmac.compare_digest(m["sig"], hmac.new(tok.encode(), f"{m['box']}|{m['url']}|{m['t']}".encode(), "sha256").hexdigest())
        except Exception:
            continue
        if ok and m["box"] == BOX_NAME and m.get("deadline", 0) > time.time() and (best is None or m["t"] > best["t"]):
            best = m
    return (best["url"], None) if best else (None, "no live announcement from a markdown box (is the Kaggle notebook running?)")


def connect(args):
    tok = token()
    url = args.url or os.environ.get("BOX_URL") or None
    if not url:
        url, why = discover(tok)
        if not url:
            die(f"{why}. Pass the url from the BOX READY banner with --url.")
        print("box:", url)
    box = Box(url, tok, rediscover=lambda: discover(tok)[0])
    try:
        urllib.request.urlopen(urllib.request.Request(box.url + "/ping", headers={"User-Agent": "md-client"}), timeout=20).close()
    except Exception as e:
        die(f"the box at {url} does not answer ({type(e).__name__}); it may have stopped or restarted (new address)")
    return box


def supported_in(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXT and not p.name.startswith("."))


def collect_units(inputs, out):
    """every input file becomes one unit of work, biggest first. Folders are expanded (their sub-folders are kept in the output);
    a .zip goes to the box as one job. folder = the output folder name, unique among its siblings like on the box."""
    units = []
    for item in inputs:
        p = Path(item)
        if not p.exists():
            die(f"not found: {item}")
        if p.is_dir():
            files = supported_in(p)
            if not files:
                die(f"no supported documents in {p} (supported: {', '.join(DOC_EXT)})")
            print(f"{p}: {len(files)} documents")
            units += [{"path": f, "rel": f.relative_to(p)} for f in files]
        elif p.suffix.lower() not in DOC_EXT + (".zip",):
            die(f"unsupported file type: {p.name} (supported: {', '.join(DOC_EXT)}, .zip)")
        else:
            units.append({"path": p, "rel": Path(p.name)})
    used = set()
    for u in units:
        parent, stem = u["rel"].parent, u["rel"].stem
        key = (str(parent).lower(), stem.lower())
        u["folder"] = stem if key not in used else f"{stem}_{u['rel'].suffix.lstrip('.').lower()}"
        used |= {key, (str(parent).lower(), u["folder"].lower())}
        u.update(base=out / parent, size=u["path"].stat().st_size, label=str(u["rel"]), zip=u["path"].suffix.lower() == ".zip")
    return sorted(units, key=lambda u: -u["size"])


def send_parts(box, prefix, path, part, parallel=3):
    """uploads a file in parts, `parallel` at a time; each part is retried by Box.call and checked with sha256 on the box"""
    size = Path(path).stat().st_size
    n = (size + part - 1) // part

    def put(i):
        with open(path, "rb") as f:
            f.seek(i * part); chunk = f.read(part)
        box.call("PUT", f"{prefix}/parts/{i}", chunk, {"Content-Type": "application/octet-stream", "X-Sha256": hashlib.sha256(chunk).hexdigest()},
                 timeout=300)
    with ThreadPoolExecutor(parallel) as ex:
        list(ex.map(put, range(n)))
    return n


def upload(box, path, options, name):
    """one job with its own upload (used for .zip batches)"""
    job = box.call("POST", "/jobs", {"name": name, "size": Path(path).stat().st_size, "options": options})
    n = send_parts(box, f"/jobs/{job['job_id']}", path, int(job["part_size"]))
    box.call("POST", f"/jobs/{job['job_id']}/start", {"parts": n})
    return job["job_id"]


def stage_file(box, path, office):
    """stores a document on the box once (office / DjVu are rendered to PDF there) -> {"file_id", "pages", "pdf"}"""
    f = box.call("POST", "/files", {"name": Path(path).name, "size": Path(path).stat().st_size})
    n = send_parts(box, f"/files/{f['file_id']}", path, int(f["part_size"]))
    return box.call("POST", f"/files/{f['file_id']}/finish", {"parts": n, "options": {"OFFICE": office}}, timeout=1800)


def page_batches(pages, size):
    """[(first, last)] page ranges of at most `size` pages (numbered from 0, inclusive)"""
    return [(a, min(a + size, pages) - 1) for a in range(0, pages, size)]


def merge_meta(into, part_meta):
    for k, v in part_meta.items():
        if isinstance(v, list) and k in ("table_of_contents", "page_stats"):
            into.setdefault(k, []).extend(v)
        else:
            into.setdefault(k, v)


def box_defaults(h):
    """the box's job defaults from /health (what a job gets for the options the client leaves out)"""
    return dict(h.get("defaults") or {}, USE_LLM=bool(h.get("llm_default") and h.get("llm")), MODE=h.get("default_mode"))


def batch_signature(options, box_defaults):
    """what a saved batch was made with (job options over the box's defaults): a rerun with other settings, or a box that
    now reviews pages, converts the batch again instead of mixing old and new output"""
    return json.dumps({k: v for k, v in {**box_defaults, **options}.items() if k != "PAGE_RANGE"}, sort_keys=True)


def add_review(row, stats):
    """page-review totals (same as pipeline.add_review on the box): pages_rewritten, and why the others kept the draft"""
    stats = [x for x in stats if x]
    if stats:
        row["pages_rewritten"] = sum(x.get("pages_rewritten", x.get("rewritten", 0)) for x in stats)
        kept = {}
        for x in stats:
            for why, n in (x.get("pages_kept_draft") or x.get("kept_draft") or {}).items():
                kept[why] = kept.get(why, 0) + n
        row["pages_kept_draft"] = kept


def join_batches(parts, final_dir, stem):
    """joins saved batches [(first_page, part_doc_dir)] in page order into final_dir/<stem>.md (+ images, one _meta.json).
    Image names carry the original page number (Marker keeps it for page ranges), so they never collide."""
    final_dir = Path(final_dir); final_dir.mkdir(parents=True, exist_ok=True)
    texts, meta = [], {}
    for _, d in sorted(parts, key=lambda p: p[0]):
        md = next(Path(d).glob("*.md"), None)
        if md is None:
            continue
        texts.append(md.read_text(encoding="utf-8").strip("\n"))
        mp = md.with_name(md.stem + "_meta.json")
        if mp.is_file():
            merge_meta(meta, json.loads(mp.read_text(encoding="utf-8")))
        for f in Path(d).iterdir():
            if f.is_file() and f not in (md, mp):
                os.replace(f, final_dir / f.name)
    for k in ("table_of_contents", "page_stats"):
        if k in meta:
            meta[k].sort(key=lambda e: e.get("page_id", 0) if isinstance(e, dict) else 0)
    (final_dir / f"{stem}.md").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    (final_dir / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def combine_rows(label, rows, output_dir):
    """one report row for a document converted in batches"""
    ok = [r for r in rows if r.get("ok")]
    pages = sum(r.get("pages") or 0 for r in ok)
    row = {"file": label, "ok": bool(ok), "where": "+".join(sorted({p for r in rows for p in str(r.get("where") or "box").split("+")})),
           "pages": pages or None, "seconds": round(sum(r.get("seconds") or 0 for r in rows), 1), "batches": len(rows), "output_dir": output_dir,
           "failed_pages": [p for r in rows for p in (r.get("failed_pages") or [])] + [r["batch"] for r in rows if not r.get("ok")],
           "low_coverage_pages": sorted(p for r in rows for p in (r.get("low_coverage_pages") or []))}
    cov = [(r["text_coverage"], r.get("pages") or 1) for r in ok if r.get("text_coverage") is not None]
    if cov:
        row["text_coverage"] = round(sum(c * w for c, w in cov) / sum(w for _, w in cov), 3)
    if any(r.get("figures_described") for r in rows):
        row["figures_described"] = sum(r.get("figures_described") or 0 for r in rows)
    add_review(row, [r for r in rows if "pages_rewritten" in r])
    stages = {}
    for r in rows:
        for k, v in (r.get("stage_seconds") or {}).items():
            stages[k] = round(stages.get(k, 0) + v, 1)
    if stages:
        row["stage_seconds"] = stages
    if any(r.get("quality") for r in rows):
        row["quality"] = next(r["quality"] for r in rows if r.get("quality"))
    levels = {k: v for r in rows for k, v in (r.get("page_levels") or {}).items()}
    if levels:                                         # standard: the level each page got, over all batches
        row["page_levels"] = dict(sorted(levels.items(), key=lambda kv: int(kv[0])))
    res = [(r["box_resources"], r.get("seconds") or 1) for r in rows if r.get("box_resources")]
    if res:                                            # the box's use over all batches, weighted by their time
        w = sum(t for _, t in res)
        avg = lambda key: [round(sum(x[key][i] * t for x, t in res if len(x.get(key) or []) > i) / w)
                           for i in range(max(len(x.get(key) or []) for x, _ in res))]
        row["box_resources"] = {"gpu_busy_pct": avg("gpu_busy_pct"),
                                "gpu_mem_max_mb": [max(x["gpu_mem_max_mb"][i] for x, _ in res if len(x.get("gpu_mem_max_mb") or []) > i)
                                                   for i in range(max(len(x.get("gpu_mem_max_mb") or []) for x, _ in res))],
                                "ram_free_min_gb": min(x.get("ram_free_min_gb", 99) for x, _ in res)}
        low = min((x for x, _ in res if x.get("ram_at_min")), key=lambda x: x.get("ram_free_min_gb", 99), default=None)
        if low:                                        # the biggest processes when RAM was lowest (out-of-memory tracking)
            row["box_resources"]["ram_at_min"] = low["ram_at_min"]
        cpus = [(x["cpu_busy_pct"], t) for x, t in res if x.get("cpu_busy_pct") is not None]
        if cpus:
            row["box_resources"]["cpu_busy_pct"] = round(sum(c * t for c, t in cpus) / sum(t for _, t in cpus))
    audit = {}
    for r in rows:                                     # the line-by-line audit's counts over all batches
        for k, v in (r.get("audit") or {}).items():
            if isinstance(v, int):
                audit[k] = audit.get(k, 0) + v
    if audit:
        row["audit"] = audit
    for key in ("tables_realigned", "table_pictures", "drawn_formulas", "indic_words_fixed", "layer_fixes", "pictures_read", "edges_peeled", "figures_cropped_by_llm",
                "picture_text_kept", "pictures_ocr_only", "spreads_split", "spreads_reordered", "pages_turned"):
        if any(r.get(key) for r in rows):
            row[key] = sum(r.get(key) or 0 for r in rows)
    trace = {"chunks": [], "pages": {}, "figures": {}, "pictures": {}}   # per chunk / page / figure over all batches (for tuning)
    for r in rows:
        for key in trace:
            if key == "chunks":
                trace[key] += (r.get("trace") or {}).get(key) or []
            else:
                trace[key].update((r.get("trace") or {}).get(key) or {})
    if trace["chunks"]:
        row["trace"] = trace
    errors = [f"pages {r['batch']}: {r.get('error')}" for r in rows if not r.get("ok")]
    if errors:
        row["error"] = "; ".join(errors)[:600]
    return row


def safe_unzip(zpath, dest):
    dest = Path(dest).resolve(); names = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            target = (dest / info.filename).resolve()
            if dest != target and dest not in target.parents:
                continue
            z.extract(info, dest); names.append(info.filename)
    return names


# ----------------------------------------------------------------- your PC's own workers
def mdbox_dir():
    """the box sources (dev/mdbox); local workers run the same worker.py as the box"""
    for d in (HERE / "dev" / "mdbox", HERE / "mdbox"):
        if (d / "worker.py").is_file():
            return d


def polish(md_path):
    """the box's final Markdown tidy-up (review.tidy: syntax, broken paragraphs, bullets) once more on the finished file,
    so documents joined here, or made by an older box, read the same"""
    try:
        sys.path.insert(0, str(mdbox_dir())) if mdbox_dir() and str(mdbox_dir()) not in sys.path else None
        import review
        md_path = Path(md_path)
        md_path.write_text(review.tidy(review.drop_unused_anchors(md_path.read_text(encoding="utf-8"))), encoding="utf-8")
    except Exception as e:                    # never lose a finished document over the tidy-up
        print(f"  note: tidy-up of {md_path.name} skipped: {type(e).__name__}: {e}"[:200])


def describe_again(box, md_path):
    """new figure descriptions for a finished file (what each figure is, then what is in it), through the box's LLM"""
    if not md_path.is_file():
        die(f"{md_path} not found")
    h = box.call("GET", "/health")
    if not (h.get("llm") and h.get("llm_local")):
        die("the box runs no vision LLM: start the notebook with the LLM, then try again")
    sys.path.insert(0, str(mdbox_dir()))
    import figures
    bak = md_path.with_name(md_path.name + ".bak")
    shutil.copy2(md_path, bak)
    proxy = VLMProxy(box, "/llm")
    t0 = time.time()
    n = figures.redescribe(md_path, md_path.parent, proxy.url, h.get("llm_model") or "local", max(2, 2 * int(h.get("llm_slots") or 4)))
    polish(md_path)
    links = len(set(figures.IMG_LINK.findall(md_path.read_text(encoding="utf-8"))))
    print(f"{n} of {links} figures described again in {time.time() - t0:.0f} s -> {md_path} (previous version: {bak.name})")


def local_auto():
    """one worker per 3.5 GB of free memory (what a Marker worker can need on a big book with the LLM passes), leaving
    2 cores and 3 GB for the system, at most 6. The box guards memory the same way while they run (pipeline.Pool.guard_memory)."""
    avail_gb = 0.0
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable:"):
                avail_gb = int(ln.split()[1]) / 2 ** 20
    except OSError:
        avail_gb = 4.0
    return max(0, min((os.cpu_count() or 2) - 2, int((avail_gb - 3) // 3.5), 6))


class VLMProxy:                       # also used for the box's helper LLM: VLMProxy(box, "/llm")
    """127.0.0.1 endpoint for the local workers: forwards to the box's /vlm with the token, answers /health itself
    (Surya probes /health with a 1 s timeout, too short for a round trip through the tunnel)"""

    def __init__(self, box, prefix="/vlm"):
        proxy = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def relay(self):
                if self.path.rstrip("/") == "/health":
                    return self.reply(200, b'{"status":"ok"}', "application/json")
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or None
                h = {"Authorization": f"Bearer {box.tok}", "User-Agent": "md-client",
                     "Content-Type": self.headers.get("Content-Type") or "application/json"}
                try:
                    with urllib.request.urlopen(urllib.request.Request(box.url + prefix + self.path, data=body, headers=h, method=self.command),
                                                timeout=900) as r:
                        self.reply(r.status, r.read(), r.headers.get("Content-Type"))
                except urllib.error.HTTPError as e:
                    self.reply(e.code, e.read(), e.headers.get("Content-Type"))
                except Exception as e:
                    proxy.errors += 1
                    self.reply(502, json.dumps({"error": repr(e)[:200]}).encode(), "application/json")

            def reply(self, code, data, ctype):
                self.send_response(code)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers(); self.wfile.write(data)

            do_GET = do_POST = relay

        self.errors = 0
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.srv.daemon_threads = True
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}/v1"


class Local:
    """warm Marker workers on this PC (pipeline.Pool + worker.py from dev/mdbox), fed by the same queue as the box"""

    def __init__(self, n, vlm_url, root, doc_timeout_s):
        sys.path.insert(0, str(mdbox_dir()))
        import convert, pipeline
        self.C, self.P, self.timeout = convert, pipeline, doc_timeout_s
        self.pool = pipeline.Pool(Path(root) / "pool")
        workers, threads, parallel = pipeline.pool_shape(os.cpu_count() or 2, n, 0, 0, 4 if vlm_url else 0)
        self.pool.start(workers, vlm_url, parallel, threads)
        self.busy, self.n = {}, workers

    def ready(self):
        return sum(v == "ready" for v in self.pool.status().values())

    def free(self):
        return self.ready() - len(self.busy)

    def can_take(self, u, office, split_min):
        """True if this PC converts the file the same way the box would (e.g. not a .docx without LibreOffice here) and it is
        small enough for one piece (long PDFs go to the box, which splits them; the PC then takes some of their page chunks)"""
        if "local_ok" not in u:
            try:
                same = not u["zip"] and self.C.route(str(u["path"]), office) == self.C.route(str(u["path"]), office, True, True)
            except self.C.Unsupported:
                same = False
            u["local_ok"] = same and not (u["path"].suffix.lower() == ".pdf" and (self.P.pdf_pages(u["path"]) or 0) >= split_min)
        return u["local_ok"]

    def submit(self, info, src, stem, out_dir, opts):
        if opts.get("SPLIT_SPREADS", True) and str(src).lower().endswith(".pdf") and opts.get("OUTPUT_FORMAT", "markdown") == "markdown":
            try:                                      # two facing pages in one photo: cut in two, as the box does (pipeline.run_job)
                import spreads
                want = self.P.parse_pages(opts.get("PAGE_RANGE"), self.P.pdf_pages(src) or 0)
                cut = Path(tempfile.mkdtemp(prefix="spreads_", dir=str(self.pool.dir))) / Path(src).name
                new, page_map, turned = spreads.split_spreads(src, cut, set(want))
                if page_map:
                    src, info["spreads_split"], info["pages_turned"] = new, len(page_map) - len(set(page_map)), turned
                    opts = dict(opts, SPREADS=page_map)
                    if opts.get("PAGE_RANGE"):
                        opts["PAGE_RANGE"] = self.P.format_pages(spreads.pages_for(page_map, want))
            except Exception as e:
                print(f"   {info['label']}: looking for two-page photos failed ({type(e).__name__}); pages read whole", flush=True)
        tid = self.pool.enqueue({"job": "local", "doc": 0, "src": str(src), "rel": info["label"], "stem": stem, "out": str(out_dir), "opts": opts})
        self.busy[tid] = info

    def poll(self):
        self.pool.maintain(self.timeout)
        for tid in list(self.busy):
            r = self.pool.result(tid)
            if r:
                info = self.busy.pop(tid)
                if info.get("spreads_split") is not None and r.get("ok"):   # the page count is the original's, as in the box's report
                    r = dict(r, spreads_split=info["spreads_split"], pages_turned=info.get("pages_turned"),
                             pages=(r.get("pages") or 0) - info["spreads_split"] or None)
                yield info, r

    def stop(self):
        self.pool.stop()


# ----------------------------------------------------------------- the run
class Run:
    """The work: a queue of files (biggest first). The box gets the next one when it is about to run dry; this PC's own workers
    take the smallest ones and steal page chunks of the box's current job. A long document is converted in BATCHES of pages:
    one batch at a time, each one downloaded and saved under <name>/.parts/ before the next starts (a restart resumes there),
    then all batches are joined. The batch size fills every box worker with exactly one chunk: workers x CHUNK_MAX_PAGES."""

    def __init__(self, box, args, options, units, out, tmp, local, split_min):
        self.box, self.args, self.options, self.out, self.tmp, self.local, self.split_min = box, args, options, out, tmp, local, split_min
        self.queue, self.lock = list(units), threading.Lock()          # biggest first: the box takes from the front, the PC from the back
        self.inflight, self.uploading, self.upload_error = {}, 0, None   # box jobs: jid -> item {unit, batch, dest, label}
        self.chunks, self.held, self.rows, self.last = {}, {}, [], {}   # chunks: jid -> [pending stolen chunks]; held: jid -> item
        self.box_cap, self.done_chunks, self.local_fails, self.box_waiting = 2, [], 0, 0
        self.converting = set()                                         # box jobs whose page chunks are queued (stealable)
        self.active = None                                              # the document being converted batch by batch
        self.batch_pages = 0

    # -- box side
    def uploader(self):
        while True:
            with self.lock:
                if self.upload_error or not self.queue:
                    return
                # one batched document at a time; with PC workers, the next file only when the box is about to run dry
                # (so the PC keeps the small files); without, keep the box's queue a little ahead of its workers
                if (self.active is not None or len(self.inflight) >= self.box_cap
                        or (self.local and self.local_fails < 3 and self.box_waiting > 0)):
                    u = None
                else:
                    u = self.queue.pop(0); self.uploading += 1
            if u is None:
                time.sleep(1); continue
            try:
                self.start_unit(u)
            except BaseException as e:                                    # die() raises SystemExit in this thread
                self.upload_error = f"upload of {u['label']} failed: {e}"
                return
            finally:
                with self.lock:
                    self.uploading -= 1

    def start_unit(self, u):
        if u["zip"]:
            jid = upload(self.box, u["path"], self.options, u["path"].name)
            print(f"  [{u['label']}] uploaded ({u['size'] / 2 ** 20:.1f} MB) -> box job {jid}")
            self.add_job(jid, {"unit": u, "batch": None, "dest": u["base"], "label": u["label"]})
            return
        info = stage_file(self.box, u["path"], self.options.get("OFFICE", "pdf"))
        u.update(file_id=info["file_id"], pages=info.get("pages"), pdf=info.get("pdf"))
        for w in info.get("warnings") or []:
            print(f"  [{u['label']}] note: {w}")
        pages = u["pages"] or 0
        if self.batch_pages and pages > self.batch_pages and not self.options.get("PAGE_RANGE"):
            u["todo"], u["batch_rows"] = page_batches(pages, self.batch_pages), []
            print(f"  [{u['label']}] uploaded ({u['size'] / 2 ** 20:.1f} MB, {pages} pages) -> {len(u['todo'])} batches of "
                  f"{self.batch_pages} pages, each saved as soon as it is complete")
            self.active = u
            self.next_batch(u)
        else:
            jid = self.file_job(u, None)
            print(f"  [{u['label']}] uploaded ({u['size'] / 2 ** 20:.1f} MB" + (f", {pages} pages" if pages else "") + f") -> box job {jid}")
            self.add_job(jid, {"unit": u, "batch": None, "dest": u["base"], "label": u["label"]})

    def file_job(self, u, batch):
        opts = dict(self.options, PAGE_RANGE=f"{batch[0]}-{batch[1]}") if batch else self.options
        name = u["folder"] + (".pdf" if u.get("pdf") else u["path"].suffix.lower())
        return self.box.call("POST", "/jobs", {"file_id": u["file_id"], "name": name, "options": opts})["job_id"]

    def add_job(self, jid, item):
        with self.lock:
            self.inflight[jid] = item; self.box_waiting = 1               # until the next /health shows the box's real queue

    def signature(self):
        return batch_signature(self.options, getattr(self, "box_defaults", {}))

    def parts_dir(self, u, batch):
        return u["base"] / f".{u['folder']}.parts" / f"{batch[0]:05d}-{batch[1]:05d}"

    def next_batch(self, u):
        """submits the next batch of the active document; batches already saved by an earlier run are skipped"""
        while u["todo"]:
            batch = u["todo"].pop(0)
            dest, key = self.parts_dir(u, batch), f"{batch[0]}-{batch[1]}"
            saved = json.loads((dest / "done.json").read_text(encoding="utf-8")) if (dest / "done.json").is_file() else None
            if saved and not saved.get("failed_pages") and saved.get("options") == self.signature():
                u["batch_rows"].append(json.loads((dest / "done.json").read_text(encoding="utf-8")))
                print(f"  [{u['label']}] pages {key}: already saved earlier, skipped")
                continue
            jid = self.file_job(u, batch)
            print(f"  [{u['label']}] pages {key} -> box job {jid}")
            u.setdefault("open", set()).add(key)
            self.add_job(jid, {"unit": u, "batch": batch, "dest": dest, "label": f"{u['label']} p.{key}"})
            return
        if not u.get("open"):                                             # the last open batch is saved: join them all
            self.finish_document(u)

    def prefetch(self):
        """the tail of a batch: once the box has started every piece of it, some workers would sit idle until the last
        pieces finish. The next batch is queued then, so they keep working; at most two batches are open, each is still
        saved as soon as it is complete, and the box still converts one piece per worker (same RAM)."""
        u = self.active
        if self.args.no_overlap or not u or not u.get("todo") or len(u.get("open") or ()) != 1:
            return
        if time.time() - self.last.get("prefetch", 0) < 10:
            return
        self.last["prefetch"] = time.time()
        if self.box.call("GET", "/health").get("waiting_tasks", 1) == 0:
            print(f"  [{u['label']}] every piece of pages {next(iter(u['open']))} has started: queuing the next batch for the free workers")
            self.next_batch(u)

    def finish_document(self, u):
        parts = sorted(p for p in (u["base"] / f".{u['folder']}.parts").iterdir() if (p / "done.json").is_file())
        docs = [(int(p.name.split("-")[0]), next((d for d in p.iterdir() if d.is_dir() and any(d.glob("*.md"))), p)) for p in parts]
        final = u["base"] / u["folder"]
        stem = next(iter(docs[0][1].glob("*.md"))).stem if docs and any(docs[0][1].glob("*.md")) else u["folder"]
        join_batches(docs, final, stem)
        polish(final / f"{stem}.md")
        self.rows.append(combine_rows(u["label"], u["batch_rows"], final.relative_to(self.out).as_posix()))
        missing = [r["batch"] for r in u["batch_rows"] if r.get("failed_pages") or not r.get("ok")]
        if missing:                                                       # keep the saved batches: a rerun only redoes the missing ones
            print(f"  [{u['label']}] INCOMPLETE: batches {', '.join(missing)} failed; {final} has the rest. Run the same command again to fill them in.")
        else:
            shutil.rmtree(u["base"] / f".{u['folder']}.parts", ignore_errors=True)
            self.drop_file(u)
            print(f"  [{u['label']}] all {len(u['batch_rows'])} batches joined -> {final}")
        self.active = None

    def drop_file(self, u):
        if u.get("file_id") and not self.args.keep_on_box:
            try:
                self.box.call("DELETE", f"/files/{u['file_id']}", tries=2)
            except SystemExit:
                pass

    def poll_box(self):
        for jid, item in list(self.inflight.items()):
            st = self.box.call("GET", f"/jobs/{jid}")
            if st.get("stage") == "converting" and jid not in self.converting:
                self.converting.add(jid); self.last["steal"] = 0          # its page chunks are queued now: the PC may take some at once
            if st["status"] in ("done", "error", "cancelled"):
                self.converting.discard(jid)
            line = f"  [{item['label']}] {st['status']}" + (f" {st.get('pct', 0):.0f}%" if st["status"] == "running" else "") + (f" · {st['msg']}" if st.get("msg") else "")
            if line != self.last.get(jid):
                print(line[:220]); self.last[jid] = line
            if st["status"] == "done":
                self.collect_box(jid, item, st)
            elif st["status"] in ("error", "cancelled"):
                err = st.get("error") or st.get("msg")
                print(f"  [{item['label']}] {st['status'].upper()}: {err}")
                with self.lock:
                    del self.inflight[jid]
                item["rows"] = [{"file": item["unit"]["label"], "ok": False, "where": "box", "error": err}]
                self.held[jid] = item                                     # finished (badly): let the batch logic move on

    def collect_box(self, jid, item, st):
        u = item["unit"]
        for w in st.get("warnings") or []:
            print(f"  [{item['label']}] note: {w}")
        work = Path(tempfile.mkdtemp(prefix="bundle_", dir=self.tmp))
        self.box.download(f"/jobs/{jid}/bundle.zip", work / "bundle.zip")
        safe_unzip(work / "bundle.zip", work / "x")
        rep = json.loads((work / "x" / "report.json").read_text(encoding="utf-8")) if (work / "x" / "report.json").is_file() else {}
        item["dest"].mkdir(parents=True, exist_ok=True)
        for entry in (work / "x").iterdir():
            if entry.name != "report.json":
                dest = item["dest"] / entry.name
                shutil.rmtree(dest, ignore_errors=True) if dest.is_dir() else dest.unlink(missing_ok=True)
                shutil.move(str(entry), str(dest))
        shutil.rmtree(work, ignore_errors=True)
        prefix = "" if str(u["rel"].parent) == "." else u["rel"].parent.as_posix() + "/"
        rows = []
        for d in rep.get("documents") or [{"file": u["label"], "ok": True}]:
            d = dict(d, where="box" + ("+pc" if d.get("remote_pages") else ""))
            d["file"] = u["label"] if not u["zip"] else f"{u['label']}/{d['file']}"
            if d.get("output_dir") and item["batch"] is None:
                d["output_dir"] = prefix + d["output_dir"]
            rows.append(d)
            if not d.get("ok"):
                print(f"  [{d['file']}] FAILED: {d.get('error')}")
        item["rows"] = rows
        if not self.args.keep_on_box:
            self.box.call("DELETE", f"/jobs/{jid}")
        with self.lock:
            del self.inflight[jid]
        self.held[jid] = item
        print(f"  [{item['label']}] done on the box in {st.get('wall_s')} s, {st.get('pages') or '?'} pages")

    def finish_item(self, item):
        """called once the box's output (and any pages this PC took over) is complete on disk"""
        u = item["unit"]
        if item["batch"] is None:
            for r in item.get("rows") or []:
                for md in (self.out / r["output_dir"]).glob("*.md") if r.get("ok") and r.get("output_dir") else []:
                    polish(md)
            self.rows += item.get("rows") or []
            if not u["zip"]:
                self.drop_file(u)
            return
        key = f"{item['batch'][0]}-{item['batch'][1]}"
        u.setdefault("open", set()).discard(key)
        rows = item.get("rows") or [{"ok": False, "error": "no result"}]
        row = dict(rows[0], batch=key)
        failed = row.get("failed_pages") or ([] if row.get("ok") else [key])
        tries = u.setdefault("tries", {})
        if failed and tries.get(key, 0) < 2:                              # missing pages: convert this batch again
            tries[key] = tries.get(key, 0) + 1
            print(f"  [{u['label']}] pages {key}: {', '.join(failed)} failed; converting the batch again (try {tries[key] + 1} of 3)")
            u["todo"].insert(0, item["batch"])
            if not u["open"]:
                self.next_batch(u)
            return
        u["batch_rows"].append(row)
        if failed:                                                        # not marked done: the next run converts it again
            print(f"  [{u['label']}] pages {key}: {', '.join(failed)} still FAILED after 3 tries; left out (run the same command again to retry)")
        else:                                                             # saved: a later run skips this batch
            (item["dest"] / "done.json").write_text(json.dumps(dict(row, options=self.signature()), ensure_ascii=False), encoding="utf-8")
            print(f"  [{u['label']}] pages {key} saved ({len(u['batch_rows'])} batches done, {len(u['todo']) + len(u['open'])} to go)")
        if not u["open"]:                                                 # else the open (prefetched) batch moves on when saved
            self.next_batch(u)

    # -- this PC
    def feed_local(self):
        if self.local_fails >= 3:                     # the PC's workers keep failing: leave everything to the box
            return
        while self.local.free() > 0:
            with self.lock:
                k = next((k for k in range(len(self.queue) - 1, -1, -1)
                          if self.local.can_take(self.queue[k], self.options.get("OFFICE", "pdf"), self.split_min)), None)
                u = self.queue.pop(k) if k is not None else None
            if u is None:
                break
            self.local.submit({"kind": "doc", "unit": u, "label": u["label"]}, u["path"], u["folder"], u["base"] / u["folder"], self.local.opts)
            print(f"  [{u['label']}] converting on this PC")
        free = self.local.free()
        if free > 0 and time.time() - self.last.get("steal", 0) > 3:            # after an empty try, wait 3 s before asking again
            got = 0
            for jid, item in list(self.inflight.items()):
                u = item["unit"]
                if free <= 0 or u["zip"] or u["path"].suffix.lower() != ".pdf":      # page numbers must match the file here
                    continue
                if jid not in self.converting:    # still being prepared (scans set upright first): no chunks yet. Asking then
                    continue                      # started the 3 s wait, and a fast box took every chunk before the next try
                for c in self.box.call("POST", f"/jobs/{jid}/steal", {"n": free}).get("chunks", []):
                    cdir = Path(self.tmp) / "chunks" / jid / c["pages"].replace(",", "_")
                    self.local.submit({"kind": "chunk", "jid": jid, "chunk": c, "label": f"{u['label']} p.{c['pages']}", "dir": cdir},
                                      u["path"], c["stem"], cdir, dict(self.local.opts, PAGE_RANGE=c["pages"]))
                    self.chunks.setdefault(jid, []).append(c["pages"]); free -= 1; got += 1
                    print(f"  [{u['label']}] pages {c['pages']} taken over by this PC")
            if not got:
                self.last["steal"] = time.time()

    def poll_local(self):
        for info, r in self.local.poll():
            self.local_fails = 0 if r.get("ok") else self.local_fails + 1
            if self.local_fails == 3:
                print("  note: 3 failures in a row on this PC; sending the rest to the box. Last worker log lines:\n"
                      + self.local.pool.log(next(iter(self.local.pool.procs)), 12))
            if info["kind"] == "doc" and not r.get("ok") and not info["unit"].get("retried"):
                u = info["unit"]; u.update(retried=True, local_ok=False)
                print(f"  [{u['label']}] failed on this PC ({str(r.get('error'))[:120]}); sending it to the box")
                with self.lock:
                    self.queue.insert(0, u)
                continue
            if info["kind"] == "doc":
                u = info["unit"]
                cov = r.get("coverage") or []
                if cov:
                    r["text_coverage"] = round(sum(c["found"] * c["words"] for c in cov) / max(1, sum(c["words"] for c in cov)), 3)
                    r["low_coverage_pages"] = [c["page"] for c in cov if c["low"]]
                add_review(r, [r.get("page_review")])
                self.rows.append({"file": u["label"], "ok": bool(r.get("ok")), "where": "pc", "pages": r.get("pages"), "seconds": r.get("sec"),
                                  **{k: r[k] for k in ("text_coverage", "low_coverage_pages", "described_figures", "pages_rewritten", "pages_kept_draft", "stage_seconds", "trace", "spreads_split", "spreads_reordered", "pages_turned", "quality", "page_levels") if r.get(k) is not None},
                                  "via": r.get("via"), "error": r.get("error"),
                                  "output_dir": (u["base"] / u["folder"]).relative_to(self.out).as_posix() if r.get("ok") else None})
                print(f"  [{u['label']}] " + (f"done on this PC in {r.get('sec')} s" if r.get("ok") else f"FAILED on this PC: {r.get('error')}"))
            else:
                info["result"] = r
                self.done_chunks.append(info)

    def fill_held(self):
        """box jobs that finished: put in the pages this PC converted (once both sides are done), then save / move on"""
        for jid, item in list(self.held.items()):
            mine = [i for i in self.done_chunks if i["jid"] == jid]
            if len(mine) < len(self.chunks.get(jid, [])):
                continue
            for i in mine:
                c, r = i["chunk"], i["result"]
                ok = self.local.C.fill_remote(item["dest"] / c["folder"], c["stem"], c["pages"], i["dir"], None if r.get("ok") else r.get("error") or "failed")
                if not ok:
                    print(f"  [{item['label']}] note: no placeholder for pages {c['pages']} in the box output")
                elif r.get("ok") and item.get("rows"):                    # the box counted only its own pages
                    row = next((x for x in item["rows"] if x.get("output_dir", "").endswith(c["folder"]) or len(item["rows"]) == 1), None)
                    if row is not None:
                        row["pages"] = (row.get("pages") or 0) + (r.get("pages") or 0)
                        if r.get("page_review"):                             # and its page-review counts
                            add_review(row, [{k: row[k] for k in ("pages_rewritten", "pages_kept_draft") if k in row}, r["page_review"]])
                        if r.get("page_levels"):                             # and the level its pages got (standard)
                            row["page_levels"] = dict(sorted({**(row.get("page_levels") or {}), **r["page_levels"]}.items(),
                                                             key=lambda kv: int(kv[0])))
                        for k, v in (r.get("stage_seconds") or {}).items():
                            row.setdefault("stage_seconds", {})[k] = round(row.get("stage_seconds", {}).get(k, 0) + v, 1)
            del self.held[jid]
            self.finish_item(item)

    # -- the loop
    def finished(self):
        with self.lock:
            return not (self.queue or self.inflight or self.uploading or self.held or self.active is not None
                        or (self.local and self.local.busy))

    def go(self):
        h = self.box.call("GET", "/health")
        self.box_defaults = box_defaults(h)
        self.box_cap = 2 * len(h.get("workers") or {}) + 2
        workers, chunk = max(1, len(h.get("workers") or {})), int(h.get("chunk_pages") or 24)
        self.batch_pages = workers * chunk if self.args.batch_pages == "auto" else int(self.args.batch_pages)
        if self.batch_pages:
            print(f"long documents go in batches of {self.batch_pages} pages ({workers} box workers x {chunk}-page chunks), "
                  "each saved here as soon as it is complete" + ("" if self.args.no_overlap else
                  "; the next is queued when the box's workers would otherwise sit idle"))
        up = threading.Thread(target=self.uploader, daemon=True); up.start()
        while True:
            if self.upload_error:
                die(self.upload_error)
            if self.local:
                self.feed_local(); self.poll_local()
                self.box_waiting = self.box.call("GET", "/health").get("waiting_tasks", 0)
            self.poll_box()
            self.fill_held()
            self.prefetch()
            if self.finished():
                break
            if not up.is_alive() and self.queue:
                up = threading.Thread(target=self.uploader, daemon=True); up.start()
            time.sleep(0.5 if self.local and self.local.free() > 0 and self.inflight else 2)   # idle PC workers: look for chunks often

    def cancel(self):
        for jid in list(self.inflight):
            try:
                self.box.call("DELETE", f"/jobs/{jid}", tries=1)
            except SystemExit:
                pass


def main():
    ap = argparse.ArgumentParser(description="Convert documents to Markdown on your Kaggle GPU box (and your PC's CPUs).",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("inputs", nargs="*", help="documents (pdf, images, Word/PowerPoint/Excel old and new, OpenDocument, rtf, djvu, html, epub, txt/md/csv), a .zip of them, or a folder")
    ap.add_argument("--out", "-o", default="markdown_out", help="output folder (default ./markdown_out)")
    ap.add_argument("--batch-pages", default="auto", help="convert long documents in batches of this many pages, each saved here before "
                                                          "the next starts, and resumed if run again; auto = box workers x chunk size; 0 = one piece")
    ap.add_argument("--url", help="box address (default: looked up automatically)")
    ap.add_argument("--token", help="BOX_TOKEN (default: .box_token file / BOX_TOKEN env)")
    ap.add_argument("--health", action="store_true", help="show the box status and exit")
    ap.add_argument("--version", action="version", version=f"md_client.py {__version__}")
    ap.add_argument("--describe-again", metavar="FILE.md", help="write every figure description of a finished .md again with the "
                    "box's vision LLM (images next to it); the old file is kept as .md.bak. No reconversion")
    ap.add_argument("--local", default="auto", help="Marker workers on this PC next to the box: a number, 'auto' (by free memory and cores) or 0. "
                                                    "Needs Marker installed here (see docs/getting-started.md); they use the box's GPUs for the VLM")
    ap.add_argument("--mode", choices=["auto", "no_ocr", "fast", "balanced", "vllm"], help="Marker's layout/OCR engine (not the "
                    "quality, see --quality): auto = Marker's balanced on the GPU box (default); vllm = full-page VLM on every page; fast/no_ocr = CPU only")
    ap.add_argument("--office", choices=["pdf", "native"], help="pdf = LibreOffice renders office files first, keeping equations and layout (default); native = Marker reads .docx/.pptx itself")
    ap.add_argument("--format", choices=["markdown", "html", "json", "chunks"], dest="fmt", help="output format (default markdown)")
    ap.add_argument("--pages", help="only these pages, numbered from 0, e.g. 0,5-10,20")
    ap.add_argument("--force-ocr", action="store_true", help="OCR every page even if it has a text layer (bad/garbled text)")
    ap.add_argument("--strip-ocr", action="store_true", help="drop an existing bad OCR text layer and OCR again")
    ap.add_argument("--redo-math", action="store_true", help="also run Marker's own LLM steps, which rewrite inline maths as LaTeX (and "
                                                             "redo tables, forms, equations); slower, no quality does it by itself")
    ap.add_argument("--llm", action="store_true", help="force the vision-LLM passes on: tables, equations, forms, headings, figure text "
                                                       "(default: on when the box runs its own LLM; with only a GOOGLE_API_KEY it uses Gemini)")
    ap.add_argument("--no-llm", action="store_true", help="skip the vision-LLM passes (faster)")
    ap.add_argument("--no-describe", action="store_true", help="do not write out the text/data inside figures under each image")
    ap.add_argument("--no-review", action="store_true", help="skip the vision LLM's page-by-page checks against each page image: audit, "
                                                             "inspection, line pass (faster; figures are still described)")
    ap.add_argument("--paginate", action="store_true", help="put a page-number separator between pages")
    ap.add_argument("--keep-headers", action="store_true", help="keep running page headers and footers")
    ap.add_argument("--quality", type=str.lower, choices=QUALITIES + tuple(QUALITY_ALIASES), metavar="{" + ",".join(QUALITIES) + "}",
                    help="express = no vision-LLM pass (Marker + Surya OCR + every fix without a model; fastest); standard (box default) = "
                         "the vision LLM only where it pays: a page the PDF's own text settles gets no call, the others one short look, "
                         "then a whole-page audit (plain text), the element check + audit (tables, maths, code, figures, scans) or also "
                         "a finer audit (handwriting, hard pages); figures with diagrams or charts described by meaning; "
                         "precise = every pass on every page: element check, line-by-line rewrite, fine audit, prose made explicit "
                         "(the original wording kept), every figure described (slowest). Old names still work: "
                         "superfast/fast = express, balanced/auto = standard, accurate = precise")
    ap.add_argument("--no-images", action="store_true", help="do not extract figures")
    ap.add_argument("--chunk-rounds", type=int, help="page chunks per box worker for long PDFs (box default 2): more = smaller chunks, a shorter "
                                                     "tail where a few chunks run alone, more per-chunk overhead (for tuning)")
    ap.add_argument("--labels", action="store_true", help="label every block with the vision LLM too (handwriting, code, "
                                                       "titles, charts...) for the report's trace and --show-layout; the Markdown "
                                                       "is the same; one more short call per page")
    ap.add_argument("--show-layout", action="store_true", help="also write <name>/layout/: every page with the blocks the layout "
                                                            "model found drawn on it (type, score) and their boxes as JSON")
    ap.add_argument("--no-split-spreads", action="store_true", help="read a photo or scan of two facing book pages as one page "
                                                                 "(default: cut in two first; the OCR misses and repeats lines of a spread)")
    ap.add_argument("--raw-figures", action="store_true", help="keep Marker's own figure crops (default: re-cropped from the page so labels are not cut off and split figures are joined)")
    ap.add_argument("--no-overlap", action="store_true", help="start a batch only after the previous one is saved (workers idle at each batch's tail)")
    ap.add_argument("--keep-on-box", action="store_true", help="do not delete the job from the box after downloading")
    args = ap.parse_args()
    if args.token:
        os.environ["BOX_TOKEN"] = args.token
    if args.quality in QUALITY_ALIASES:
        print(f"note: --quality {args.quality} is now called {QUALITY_ALIASES[args.quality]}")
        args.quality = QUALITY_ALIASES[args.quality]
    box = connect(args)
    if args.health:
        print(json.dumps(box.call("GET", "/health"), indent=2)); return
    if args.describe_again:
        describe_again(box, Path(args.describe_again)); return
    if not args.inputs:
        ap.error("give at least one document, folder or .zip (or use --health)")
    options = {k: v for k, v in {"MODE": args.mode, "OFFICE": args.office, "OUTPUT_FORMAT": args.fmt, "PAGE_RANGE": args.pages, "FORCE_OCR": args.force_ocr or None,
                                 "STRIP_EXISTING_OCR": args.strip_ocr or None, "REDO_INLINE_MATH": args.redo_math or None,
                                 "USE_LLM": True if args.llm else (False if args.no_llm else None), "DESCRIBE_FIGURES": False if args.no_describe else None,
                                 "PAGE_REVIEW": False if args.no_review else None, "AUDIT": False if args.no_review else None, "QUALITY": args.quality,
                                 "CHUNK_ROUNDS": args.chunk_rounds, "SPLIT_SPREADS": False if args.no_split_spreads else None, "SHOW_LAYOUT": args.show_layout or None, "LABEL_BLOCKS": True if args.labels else None,
                                 "PAGINATE": args.paginate or None, "KEEP_HEADERS_FOOTERS": args.keep_headers or None,
                                 "IMAGES": False if args.no_images else None, "FIGURE_FIX": False if args.raw_figures else None}.items() if v is not None}
    out = Path(args.out).resolve()
    units = collect_units(args.inputs, out)
    h = box.call("GET", "/health")
    local, tmp = None, tempfile.mkdtemp(prefix="mdbox_")
    try:
        local = start_local(args, box, h, options, units, tmp)
        t0 = time.time()
        run = Run(box, args, options, units, out, tmp, local, 20)
        print(f"converting {len(units)} file(s) on the box ({len(h.get('workers') or {})} workers, {h.get('vlm_slots') or 0} GPU slots)"
              + (f" + {local.n} workers on this PC" if local else "") + " ... (Ctrl+C cancels)")
        def stop(signum, frame):                       # killed or its terminal closed: cancel the box jobs like Ctrl+C,
            raise KeyboardInterrupt                        # or they would keep the box's workers busy for nothing
        for sig in ("SIGTERM", "SIGHUP"):
            if hasattr(signal, sig):
                signal.signal(getattr(signal, sig), stop)
        try:
            run.go()
        except KeyboardInterrupt:
            run.cancel(); print("\ncancelled"); sys.exit(130)
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps({"documents": run.rows, "wall_s": round(time.time() - t0), "box": {k: h.get(k) for k in (
            "code", "llm_model", "llm_slots", "vlm_slots", "workers", "chunk_pages")}, "options": options}, ensure_ascii=False, indent=2), encoding="utf-8")
        ok = sum(1 for r in run.rows if r.get("ok"))
        print(f"done in {time.time() - t0:.0f} s: {ok}/{len(run.rows)} documents ok -> {out}"
              + (f" ({sum(r.get('where') == 'pc' for r in run.rows)} converted on this PC)" if local else ""))
        for r in run.rows:
            if r.get("ok") and r.get("output_dir"):
                print("     ", out / r["output_dir"], f"(text coverage {r['text_coverage']:.1%})" if "text_coverage" in r else "",
                      f"| pages rewritten by the vision LLM: {r['pages_rewritten']}, kept as converted: {r['pages_kept_draft'] or 'none'}"
                      if "pages_rewritten" in r else "")
            if r.get("quality"):
                au = r.get("audit") or {}
                levels = {}
                for v in (r.get("page_levels") or {}).values():
                    levels[v] = levels.get(v, 0) + 1
                print(f"      quality: {r['quality']}" + (f" | line-by-line audit: {au.get('sections', 0)} sections, fixes confirmed by the text "
                      f"layer {au.get('confirmed by the text layer', 0)}, by a second look {au.get('confirmed by a second look', 0)}, "
                      f"refused {au.get('refused', 0)}" if au else "")
                      + (" | pages by level: " + ", ".join(f"{k} {levels[k]}" for k in ("none", "audit", "inspect", "hard") if k in levels)
                         if levels else ""))
            br = r.get("box_resources") or {}
            if br.get("gpu_busy_pct"):
                print(f"      box while converting: GPUs busy {', '.join(f'{u}%' for u in br['gpu_busy_pct'])}"
                      + (f", CPU {br['cpu_busy_pct']}%" if br.get("cpu_busy_pct") is not None else "")
                      + f", RAM free at least {br.get('ram_free_min_gb')} GB")
                if br.get("ram_at_min") and br.get("ram_free_min_gb", 99) < 6:
                    print("      RAM held then by: " + "; ".join(br["ram_at_min"][:6]))
            if r.get("stage_seconds"):
                tot = sum(r["stage_seconds"].values()) or 1
                print("      worker time: " + ", ".join(f"{k} {v / 60:.0f} min ({v / tot:.0%})" for k, v in sorted(r["stage_seconds"].items(), key=lambda kv: -kv[1])))
            if r.get("low_coverage_pages"):
                print(f"  WARNING {r['file']}: pages {r['low_coverage_pages'][:30]} lost text: under 90% of their PDF words are in the "
                      "Markdown; check them, or convert them again with --pages ... --mode vllm")
    finally:
        if local:
            local.stop()
        shutil.rmtree(tmp, ignore_errors=True)


def start_local(args, box, h, options, units, tmp):
    """the PC's workers, or None (not wanted, Marker not installed here, nothing they can do, or no VLM path for a VLM mode)"""
    want = local_auto() if args.local == "auto" else int(args.local)
    if want <= 0 or all(u["zip"] for u in units):
        return None
    if not mdbox_dir() or importlib.util.find_spec("marker") is None:
        if args.local != "auto":
            die("--local needs Marker on this PC (pip install, see docs/getting-started.md) and dev/mdbox next to md_client.py")
        print("tip: install Marker on this PC to convert here too, see docs/getting-started.md (--local)")
        return None
    use_llm = bool(options.get("USE_LLM", h.get("llm_default")) and h.get("llm"))
    local_llm = {}
    if use_llm:                                        # the PC must make the same LLM passes as the box
        if h.get("llm_local"):
            local_llm = {"LLM_URL": VLMProxy(box, "/llm").url, "LLM_MODEL": h.get("llm_model"), "LLM_CONCURRENCY": 2,
                         "LLM_CTX": (h.get("defaults") or {}).get("LLM_CTX", 8192)}
        elif not os.environ.get("GOOGLE_API_KEY"):
            print("note: the box uses Gemini for the LLM passes; set GOOGLE_API_KEY here too to convert on this PC")
            return None
        d = h.get("defaults") or {}
        options.update(USE_LLM=True, QUALITY=options.get("QUALITY") or d.get("QUALITY") or "standard")
        for key in ("REDO_INLINE_MATH", "DESCRIBE_FIGURES", "PAGE_REVIEW"):            # the same stages as the box: set, or as the quality says
            if options.get(key, d.get(key)) is not None:
                options[key] = bool(options.get(key, d.get(key)))
    else:
        options["USE_LLM"] = False
    mode = options.get("MODE") or "auto"
    vlm = mode in ("auto", "balanced", "vllm") and h.get("vlm")
    if mode in ("balanced", "vllm") and not vlm:
        return None
    options["MODE"] = ("balanced" if vlm else "fast") if mode == "auto" else mode     # the PC must use the same mode as the box
    vlm_url = VLMProxy(box).url if vlm else ""
    local = Local(want, vlm_url, tmp, 90 * 60)
    local.opts = dict(options, **local_llm)            # LLM_URL etc. only for the PC's workers, never sent to the box
    print(f"this PC: {local.n} Marker workers" + (" (VLM on the box's GPUs through the tunnel)" if vlm_url else " (CPU only)")
          + (f", LLM passes with {h.get('llm_model')}" if use_llm else ""))
    return local


if __name__ == "__main__":
    main()
