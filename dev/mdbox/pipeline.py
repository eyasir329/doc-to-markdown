"""Document -> Markdown pipeline (called by server.py, one job per runner thread).
A job is one uploaded document, or a .zip of documents. Every document becomes one task in a shared queue - a long PDF
becomes several page-range tasks - and the warm worker processes (worker.py) pull tasks from it, so documents and pages
of the same job and of different jobs run in parallel while the VLM requests of all of them are spread over both GPUs.
Page-range tasks can also be "stolen" by the client's own local workers (steal()), which then fill them in on the PC.
The Pool class is also used by md_client.py to run local workers on the PC."""
import json, math, os, re, shutil, subprocess, sys, threading, time, zipfile
from pathlib import Path

import convert
import review as R

HERE = Path(__file__).resolve().parent
WORKER_PY = Path(os.environ.get("MD_WORKER_PY") or HERE / "worker.py")
TMP_ROOT = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")

DOC_EXT = convert.ALL_EXT
# WORKERS 0 = from the vCPUs and free RAM (WORKERS_PER_CPU, MEM_PER_WORKER_GB); VLM_PARALLEL 0 = from VLM_SLOTS
DEFAULTS = dict(WORKERS=0, CPU_THREADS=0, VLM_URL="", VLM_PARALLEL=0, VLM_SLOTS=0, MODE="auto", OFFICE="pdf", DOC_TIMEOUT_MIN=90,
                LLM_URL="", LLM_MODEL="", LLM_SLOTS=0, USE_LLM=False, REDO_INLINE_MATH=None, DESCRIBE_FIGURES=None, PAGE_REVIEW=None, LLM_CTX=8192,
                QUALITY="balanced",                           # fast / balanced / accurate (review.QUALITY_PROFILES); None above = as the quality sets it
                TIME_BUDGET_H=7.0, MAX_ZIP_FILES=2000, MAX_UNZIPPED_GB=6.0,
                SPLIT_MIN_PAGES=8, CHUNK_MIN_PAGES=4, CHUNK_MAX_PAGES=24,
                MEM_PER_WORKER_GB=3.0, MEM_RESERVE_GB=3.0,     # RAM a Marker worker may need (measured: ~2 GB at its peak in the suite runs) / keep free for the API, tunnel, page cache
                MEM_PAUSE_FREE_GB=4.0, MEM_RESUME_FREE_GB=6.0, MEM_KILL_FREE_GB=2.0,   # free RAM: stop new pieces / start again / stop a worker
                MEM_GPU_SERVER_MAX_GB=5.0,                    # a llama-server bigger than this (healthy: ~2.3 + its prompt cache) is restarted when RAM runs out
                WORKER_RECYCLE_GB=3.0,                        # a worker bigger than this after a piece restarts itself (frees leaks)
                WORKERS_PER_CPU=1.5,                          # workers per vCPU at most (RAM permitting): a worker waiting on the GPUs leaves its CPU free
                MEM_START_GB=2.5, START_WINDOW_S=60)          # RAM a piece grows by in its first minute; how long a start counts against the free RAM
DEFAULTS.update(json.loads(os.environ.get("MD_DEFAULTS") or "{}"))      # set from the notebook's JOB_DEFAULTS
MODES = ("auto", "no_ocr", "fast", "balanced", "vllm")


class Cancelled(Exception):
    pass


def jwrite(path, obj):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False)); os.replace(tmp, path)


def safe_part(s, limit=80):
    """keeps non-ASCII names (Bangla, Hindi, ...) but drops what Windows/zip cannot hold"""
    s = re.sub(r'[\x00-\x1f<>:"/\\|?*]+', "_", s).strip(" .")
    return s[:limit].strip(" .") or "doc"


# ----------------------------------------------------------------- memory (the Kaggle session dies when its RAM is full)
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def memory_detail():
    """bytes: limit, used = what the kernel cannot free (process memory + tmpfs/shared memory), cache = file cache it drops first.
    File cache (e.g. model files the GPU servers just read) is NOT counted as used: counting it paused work for nothing."""
    for lim_f, cur_f, stat_f, file_k, shm_k in (
            ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.stat", "file", "shmem"),
            ("/sys/fs/cgroup/memory/memory.limit_in_bytes", "/sys/fs/cgroup/memory/memory.usage_in_bytes",
             "/sys/fs/cgroup/memory/memory.stat", "total_cache", "total_shmem")):
        lim, cur = _read(lim_f), _read(cur_f)
        if lim and cur and lim != "max" and int(lim) < 1 << 50:
            stat = dict(l.split() for l in (_read(stat_f) or "").splitlines() if len(l.split()) == 2)
            file, shm = int(stat.get(file_k, 0)), int(stat.get(shm_k, 0))
            return {"limit": int(lim), "used": max(0, int(cur) - file + shm), "cache": max(0, file - shm)}
    info = dict((l.split(":")[0], int(l.split()[1]) * 1024) for l in (_read("/proc/meminfo") or "").splitlines() if l.split()[1:2])
    total = info.get("MemTotal", 1)
    return {"limit": total, "used": total - info.get("MemAvailable", total), "cache": 0}


def memory():
    """(limit, used) in bytes, see memory_detail"""
    d = memory_detail()
    return d["limit"], d["used"]


def _scan():
    """[(bytes, pid, name)] of every process"""
    out = []
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        st = _read(f"/proc/{pid}/status") or ""
        kb = next((int(l.split()[1]) for l in st.splitlines() if l.startswith("VmRSS:")), 0)
        if kb:
            cmd = (_read(f"/proc/{pid}/cmdline") or "").replace("\x00", " ").split()
            name = " ".join(Path(c).name for c in cmd[:3])[:60] or next((l.split()[1] for l in st.splitlines() if l.startswith("Name:")), pid)
            out.append((kb * 1024, int(pid), name))
    return out


def processes(n=8):
    """the n biggest processes by resident memory: [(MB, name)], for /health (where does the RAM go?)"""
    return [(b // 2 ** 20, name) for b, _, name in sorted(_scan(), reverse=True)[:n]]


def gpu_servers():
    """[(bytes, pid, name)] of the llama-server processes (the notebook's watchdog restarts one that dies)"""
    return [p for p in _scan() if p[2].startswith("llama-server")]


def rss(pid):
    """resident memory of a process and its children (e.g. LibreOffice under a worker), bytes"""
    total, todo = 0, [str(pid)]
    while todo:
        p = todo.pop()
        st = _read(f"/proc/{p}/status") or ""
        total += next((int(l.split()[1]) * 1024 for l in st.splitlines() if l.startswith("VmRSS:")), 0)
        todo += (_read(f"/proc/{p}/task/{p}/children") or "").split()
    return total


# ----------------------------------------------------------------- the warm worker pool
class Pool:
    def __init__(self, root=None):
        self.dir = Path(root or os.environ.get("MD_POOL") or TMP_ROOT / "md_pool")
        self.procs, self.seq, self.restarts, self.broken = {}, 0, 0, False
        self.requeued, self.mem_events = {}, []           # tid -> times put back after a memory kill; notes for the job report
        self.vlm_url, self.parallel, self.threads = "", 8, 2
        self.mtx, self.seq_lock = threading.Lock(), threading.Lock()
        self.planned, self.last_kill = set(), 0.0           # workers stopped or restarting on purpose

    @property
    def vlm_ok(self):
        return bool(self.vlm_url)

    def start(self, n, vlm_url, parallel, threads):
        shutil.rmtree(self.dir, ignore_errors=True)
        for d in ("tasks", "results", "running", "logs", "cancel"):
            (self.dir / d).mkdir(parents=True, exist_ok=True)
        self.vlm_url, self.parallel, self.threads, self.restarts = vlm_url or "", parallel, threads, 3 * n + 3
        for k in range(n):
            self.launch(f"w{k}")

    def launch(self, wid):
        cmd = [sys.executable, str(WORKER_PY), str(self.dir), wid, self.vlm_url, str(self.parallel), str(self.threads)]
        self.procs[wid] = subprocess.Popen(cmd, stdout=open(self.dir / "logs" / f"{wid}.log", "ab"), stderr=subprocess.STDOUT,
                                           env=dict(os.environ, TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS=str(self.threads),
                                                    MD_WORKER_RECYCLE_GB=str(DEFAULTS["WORKER_RECYCLE_GB"])))

    def stop(self):
        (self.dir / "shutdown").write_text("1")
        for p in self.procs.values():
            if p.poll() is None:
                p.terminate()

    def enqueue(self, task):
        with self.seq_lock:
            self.seq += 1
            tid = f"{self.seq:08d}"
        task["tid"] = tid
        jwrite(self.dir / "tasks" / f"{tid}.json", task)
        return tid

    def result(self, tid):
        p = self.dir / "results" / f"{tid}.json"
        try:
            return json.loads(p.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def waiting(self):
        """tasks no worker (or client) has claimed yet"""
        claimed = {p.stem for p in (self.dir / "tasks").glob("*.claim")}
        return sum(1 for p in (self.dir / "tasks").glob("*.json") if p.stem not in claimed and not (self.dir / "results" / p.name).exists())

    def cancel_job(self, job_id):
        (self.dir / "cancel" / job_id).write_text("1")

    def _fail(self, tid, task, why, wid=None):
        """record a failure for a task unless the worker managed to write its own result first"""
        res = {"ok": False, "error": why, "tid": tid, "job": task["job"], "doc": task["doc"], "rel": task["rel"], "worker": wid, "sec": None}
        tmp = self.dir / "results" / f"{tid}.fail.tmp"
        tmp.write_text(json.dumps(res))
        try:
            os.link(tmp, self.dir / "results" / f"{tid}.json")
        except FileExistsError:
            pass
        tmp.unlink(missing_ok=True)

    def running(self):
        out = {}
        for rp in (self.dir / "running").glob("*.json"):
            if (self.dir / "results" / rp.name).exists():
                continue
            try:
                out[rp.stem] = json.loads(rp.read_text())
            except Exception:
                pass
        return out

    RECYCLED = 75          # exit code of a worker that restarts itself on purpose (worker.py)

    def requeue(self, tid, wid, why, count=True):
        """puts a piece that lost its worker to memory pressure back in the queue (twice at most when its own size was the
        problem, i.e. count=True), else fails it"""
        task = json.loads((self.dir / "tasks" / f"{tid}.json").read_text())
        n = self.requeued[tid] = self.requeued.get(tid, 0) + (1 if count else 0)
        if n > 2:
            self._fail(tid, task, f"{why} (3 times): this piece needs more memory than the box has; try fewer pages per job", wid)
            return
        (self.dir / "running" / f"{tid}.json").unlink(missing_ok=True)
        (self.dir / "tasks" / f"{tid}.claim").unlink(missing_ok=True)
        self.mem_events.append(f"{task['rel']}: {why}; converted again")

    def guard_memory(self, d=None):
        """little free RAM: no new pieces (but see worker.py: with nothing running, one worker may still start, so it can never
        deadlock). Very little: stop the biggest busy worker and requeue its piece. Returns free GB."""
        d = d or DEFAULTS
        if not (self.dir / "tasks").is_dir():                # pool not started (yet)
            return None
        lim, used = memory()
        free_gb, pause = (lim - used) / 2 ** 30, self.dir / "pause"
        try:                                                 # the start budget the workers read (worker.py: may_start)
            starts = self.dir / "starts"
            starts.mkdir(exist_ok=True)
            for f in starts.glob("slot_*"):                  # a start counts for START_WINDOW_S, then the free RAM shows it
                if time.time() - f.stat().st_mtime > float(d["START_WINDOW_S"]):
                    f.unlink(missing_ok=True)
            jwrite(self.dir / "admit.json", {"slots": start_budget(free_gb, float(d["MEM_PAUSE_FREE_GB"]), float(d["MEM_START_GB"])),
                                             "free_gb": round(free_gb, 2)})
        except OSError:
            pass
        if free_gb < float(d["MEM_PAUSE_FREE_GB"]) and not pause.exists():
            pause.write_text(f"{free_gb:.1f}")
        elif free_gb > float(d["MEM_RESUME_FREE_GB"]) and pause.exists():
            pause.unlink(missing_ok=True)
        solo = self.dir / "solo"                             # the one worker allowed while paused; drop it if that worker died
        if solo.exists():
            pid = (_read(str(solo)) or "0").strip()
            if not pid.isdigit() or not Path(f"/proc/{pid}").exists():
                solo.unlink(missing_ok=True)
        if free_gb >= float(d["MEM_KILL_FREE_GB"]) or time.time() - self.last_kill < 20:   # give the kernel time to free the last one
            return free_gb
        why = f"memory nearly full ({free_gb:.1f} GB free of {lim / 2 ** 30:.0f} GB)"
        busy = [(rss(self.procs[i["wid"]].pid), tid, i["wid"]) for tid, i in self.running().items()
                if i["wid"] in self.procs and self.procs[i["wid"]].poll() is None]
        size, tid, wid = max(busy) if busy else (0, None, None)
        grown = [s for s in gpu_servers() if s[0] > float(d.get("MEM_GPU_SERVER_MAX_GB", 4)) * 2 ** 30]
        if grown and max(grown)[0] > size:                   # a GPU server holds the RAM: restart it rather than lose pieces
            b, pid, name = max(grown)
            try:
                os.kill(pid, 9); self.last_kill = time.time()
                self.mem_events.append(f"{why}: GPU server {name} held {b / 2 ** 30:.1f} GB; restarted")
            except ProcessLookupError:
                pass
        elif busy:
            p = self.procs[wid]
            p.kill(); p.wait(10); self.last_kill = time.time()
            # a small worker was not the cause: its piece is redone without counting toward "needs more memory than the box has"
            self.requeue(tid, wid, f"{why}, worker {wid} ({size / 2 ** 30:.1f} GB) stopped",
                         count=size > 0.6 * float(d["MEM_PER_WORKER_GB"]) * 2 ** 30)
            self.planned.add(wid)
        return free_gb

    def maintain(self, doc_timeout_s):
        """kill workers stuck on a cancelled/over-long document, relaunch crashed workers, release their claims"""
        with self.mtx:
            self.guard_memory()
            now = time.time()
            for tid, info in self.running().items():
                try:
                    task = json.loads((self.dir / "tasks" / f"{tid}.json").read_text())
                except Exception:
                    continue
                why = "cancelled" if (self.dir / "cancel" / task["job"]).exists() else (
                    f"timed out after {doc_timeout_s / 60:.0f} min" if now - info["t0"] > doc_timeout_s else None)
                if why:
                    p = self.procs.get(info["wid"])
                    if p and p.poll() is None:
                        p.kill(); p.wait(10)
                    self._fail(tid, task, why, info["wid"])
            for wid, p in list(self.procs.items()):
                if p.poll() is None:
                    continue
                planned = wid in self.planned or p.returncode == self.RECYCLED
                self.planned.discard(wid)
                for tid, info in self.running().items():
                    if info["wid"] == wid:
                        if p.returncode in (-9, 137):          # killed by the kernel: almost always memory
                            self.requeue(tid, wid, "worker killed by the system (out of memory)")
                        else:
                            task = json.loads((self.dir / "tasks" / f"{tid}.json").read_text())
                            self._fail(tid, task, f"the worker process died (exit code {p.returncode}); most likely a damaged file", wid)
                for cl in (self.dir / "tasks").glob("*.claim"):       # claimed but never started: give it to someone else
                    try:
                        if cl.read_text() == wid and not (self.dir / "running" / f"{cl.stem}.json").exists() and not (self.dir / "results" / f"{cl.stem}.json").exists():
                            cl.unlink()
                    except OSError:
                        pass
                if planned:
                    self.launch(wid)                              # restarts on purpose do not use up the crash budget
                elif self.restarts > 0:
                    self.restarts -= 1; self.launch(wid)
                else:
                    self.broken = True

    TQDM = re.compile(r"([A-Za-z][A-Za-z ./()-]{2,40}?):\s+(\d+)%\|[^|]*\|\s*(\d+/\d+)")

    def progress_line(self, wid):
        """the last progress bar Marker printed in this worker's log, e.g. 'Recognizing Layout 12/40'"""
        try:
            p = self.dir / "logs" / f"{wid}.log"
            with open(p, "rb") as f:
                f.seek(max(0, p.stat().st_size - 2000)); tail = f.read().decode("utf-8", "replace")
            m = list(self.TQDM.finditer(tail.replace("\r", "\n")))
            return f"{m[-1].group(1).strip()} {m[-1].group(3)}" if m else ""
        except Exception:
            return ""

    def log(self, wid, n=25):
        p = self.dir / "logs" / f"{wid}.log"
        return "\n".join(p.read_text(errors="replace").replace("\r", "\n").splitlines()[-n:]) if p.exists() else "(no log)"

    def status(self):
        out = {}
        for wid, p in list(self.procs.items()):
            out[wid] = "dead" if p.poll() is not None else ("ready" if (self.dir / f"alive_{wid}.json").exists() else "loading")
        return out


POOL = Pool()


def pool_shape(cpus, workers=0, threads=0, vlm_slots=0, vlm_parallel=0):
    """-> (workers, threads per worker, VLM requests in flight per worker).
    Indra's ONNX layout/OCR runs on the CPU and scales with processes, not threads, so the default is one single-threaded
    worker per CPU; together the workers keep ~1.5x the GPU servers' slots busy (the extra hides CPU gaps between requests)."""
    workers = int(workers) or max(1, cpus)
    threads = int(threads) or max(1, cpus // workers)
    parallel = int(vlm_parallel) or (max(2, math.ceil(1.5 * int(vlm_slots) / workers)) if int(vlm_slots) else 4)
    return workers, threads, parallel


def workers_for_memory(cpus, limit, used, per_worker_gb, reserve_gb, per_cpu=1.0):
    """up to per_cpu workers per CPU (much of a worker's time is waiting for the GPU servers, its CPU idle), but only as many
    as the free RAM can hold (measured after the GPU servers have started)"""
    free_gb = (limit - used) / 2 ** 30 - reserve_gb
    return max(1, min(math.ceil(cpus * per_cpu), int(free_gb // per_worker_gb)))


def start_budget(free_gb, pause_gb, start_gb):
    """how many pieces may start now: each grows by about start_gb in its first minute, and free RAM must stay above the
    pause level (several workers starting at once would otherwise all see the same free RAM)"""
    return max(0, int((free_gb - pause_gb) // start_gb))


def start_pool():
    d = DEFAULTS
    cpus = os.cpu_count() or 2
    if not int(d["WORKERS"]):
        lim, used = memory()
        n = workers_for_memory(cpus, lim, used, float(d["MEM_PER_WORKER_GB"]), float(d["MEM_RESERVE_GB"]), float(d["WORKERS_PER_CPU"]))
        print(f"RAM {used / 2 ** 30:.1f} of {lim / 2 ** 30:.1f} GB in use -> {n} workers", flush=True)
        d = dict(d, WORKERS=n)
    workers, threads, parallel = pool_shape(cpus, d["WORKERS"], d["CPU_THREADS"], d["VLM_SLOTS"], d["VLM_PARALLEL"])
    POOL.start(workers, d["VLM_URL"], parallel, threads)


# ----------------------------------------------------------------- page chunks of long PDFs
def parse_pages(spec, total):
    """'0,5-10' -> sorted page numbers below total (None/'' = every page)"""
    if not spec:
        return list(range(total))
    pages = set()
    for part in str(spec).replace(" ", "").split(","):
        a, _, b = part.partition("-")
        pages.update(range(int(a), int(b or a) + 1))
    return sorted(p for p in pages if 0 <= p < total)


def format_pages(pages):
    """[0,1,2,5,7,8] -> '0-2,5,7-8'"""
    runs, start = [], None
    for k, p in enumerate(pages):
        if start is None:
            start = p
        if k + 1 == len(pages) or pages[k + 1] != p + 1:
            runs.append(str(start) if start == p else f"{start}-{p}"); start = None
    return ",".join(runs)


def plan_chunks(pages, workers, min_pages=10, max_pages=40):
    """contiguous page chunks, min..max pages long, enough for every worker to get about two, in whole rounds: the count is a
    multiple of the workers (6 chunks on 5 workers left one worker a second round while four waited: the 27-page suite took
    ~1/3 longer than its work), rounded up while every chunk keeps min_pages, and the sizes as even as possible"""
    n = len(pages)
    if n < 2 * min_pages:
        return [pages]
    most = n // min_pages
    k = max(math.ceil(n / max_pages), min(2 * workers, most))
    if workers > 1 and k > workers and k % workers:
        up = math.ceil(k / workers) * workers
        k = up if up <= most else max(workers, k // workers * workers)
    k = max(1, min(k, n))
    base, extra = divmod(n, k)
    out, at = [], 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        out.append(pages[at:at + size]); at += size
    return out


def pdf_pages(path):
    try:
        import pypdfium2 as pdfium                       # installed with Marker
        d = pdfium.PdfDocument(str(path))
        try:
            return len(d)
        finally:
            d.close()
    except Exception:
        return None


def chunk_plan(path, opts, workers, o):
    """page lists to convert as separate tasks, or None to convert the document in one piece"""
    # shortcut: only native PDFs in markdown output are split; revisit when long DjVu/office files or html/json output need it
    if Path(path).suffix.lower() != ".pdf" or opts.get("OUTPUT_FORMAT", "markdown") != "markdown" or not int(o["SPLIT_MIN_PAGES"]):
        return None
    total = pdf_pages(path)
    pages = parse_pages(opts.get("PAGE_RANGE"), total or 0)
    if len(pages) < int(o["SPLIT_MIN_PAGES"]):
        return None
    parts = plan_chunks(pages, workers, int(o["CHUNK_MIN_PAGES"]), int(o["CHUNK_MAX_PAGES"]))
    return parts if len(parts) > 1 else None


RENDER_LOCK = threading.Lock()      # one LibreOffice / ddjvu at a time on the box: each can take GBs of RAM


def pre_render(path, i, opts, warnings, note=lambda name: None):
    """office documents and DjVu -> PDF here, before planning, so a long one is split into page chunks like any PDF and no
    worker ever holds LibreOffice and Marker in memory together. Anything else (or a failure) is left to the worker as before."""
    try:
        action, fmt = convert.route(str(path), opts.get("OFFICE", "pdf"))
    except convert.Unsupported:
        return path                                          # the worker reports it
    if action not in ("soffice", "ddjvu") or fmt != "pdf":
        return path
    out = Path(path).parent / ".rendered" / str(i)           # one folder per document: a.doc and a.ppt never collide
    with RENDER_LOCK:
        note(Path(path).name)
        try:
            return (convert.soffice(path, "pdf", out, TMP_ROOT / "lo_profile_api") if action == "soffice"
                    else convert.djvu_to_pdf(path, out))
        except Exception as e:
            warnings.append(f"{Path(path).name}: rendering to PDF first failed ({str(e)[:150]}); converted in one piece")
            return path


STEALABLE = {}          # job id -> chunk task ids, for steal()


def steal(job_id, n):
    """hands up to n not-yet-started page chunks of a running job to the client's local workers (latest pages first, since
    the box's workers take the earliest). The box writes a placeholder for them; the client fills it in."""
    got = []
    for tid in reversed(STEALABLE.get(job_id, [])):
        if len(got) >= n:
            break
        if (POOL.dir / "results" / f"{tid}.json").exists():
            continue
        try:
            fd = os.open(str(POOL.dir / "tasks" / f"{tid}.claim"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)    # same claim as a worker
            os.write(fd, b"remote"); os.close(fd)
        except FileExistsError:
            continue
        t = json.loads((POOL.dir / "tasks" / f"{tid}.json").read_text())
        jwrite(POOL.dir / "results" / f"{tid}.json", {"ok": True, "remote": True, "tid": tid, "job": t["job"], "doc": t["doc"], "rel": t["rel"],
                                                      "worker": "client", "sec": 0, "pages": None})
        got.append({"file": t["rel"], "pages": t["opts"]["PAGE_RANGE"], "folder": t["out_rel"], "stem": t["stem"]})
    return got


# ----------------------------------------------------------------- inputs
def zip_name(info):
    """the real file name: zips made on Windows store non-English names in the local code page (GBK, Shift-JIS...), which
    Python reads as CP437 gibberish; names flagged UTF-8 are kept"""
    if info.flag_bits & 0x800:
        return info.filename
    raw = info.filename.encode("cp437", errors="replace")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return convert.guess_decode(raw) or info.filename


def safe_extract(zpath, dest, max_files, max_bytes):
    """extract only the supported documents of a zip, refusing path tricks and zip bombs; returns [(abs_path, rel_path)]"""
    dest = Path(dest).resolve(); docs = []
    with zipfile.ZipFile(zpath) as z:
        infos = [i for i in z.infolist() if not i.is_dir() and i.filename.lower().endswith(DOC_EXT)
                 and not i.filename.startswith("__MACOSX") and not Path(i.filename).name.startswith(".")]
        if len(infos) > max_files:
            raise RuntimeError(f"the zip holds {len(infos)} documents; the limit is {max_files}")
        if sum(i.file_size for i in infos) > max_bytes:
            raise RuntimeError("the documents in the zip are too large when unpacked")
        for i in infos:
            parts = [safe_part(p, 100) for p in Path(zip_name(i).replace("\\", "/")).parts if p not in ("", ".", "..")]
            target = dest.joinpath(*parts).resolve()
            if dest not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(i) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            docs.append((target, "/".join(parts)))
    return sorted(docs, key=lambda d: d[1].lower())


def llm_plan(opts, workers, warnings):
    """which LLM passes a job gets: the box's own vision LLM (LLM_URL) first, else Gemini, else none"""
    d = DEFAULTS
    have = bool(d["LLM_URL"]) or bool(os.environ.get("GOOGLE_API_KEY"))
    want = opts.get("USE_LLM", bool(d["USE_LLM"]))
    if want and not have:
        if opts.get("USE_LLM"):
            warnings.append("--llm asked for, but this box has no LLM (no local model running, no GOOGLE_API_KEY): converted without it")
        return {"USE_LLM": False}
    if not want:
        return {"USE_LLM": False}
    plan = {"USE_LLM": True, "LLM_CTX": int(d["LLM_CTX"])}
    for key in ("REDO_INLINE_MATH", "DESCRIBE_FIGURES", "PAGE_REVIEW"):   # set by the job or the box, else by the quality
        if opts.get(key, d[key]) is not None:
            plan[key] = bool(opts.get(key, d[key]))
    if d["LLM_URL"]:
        slots = max(1, int(d["LLM_SLOTS"]))
        plan.update(LLM_URL=d["LLM_URL"], LLM_MODEL=d["LLM_MODEL"],
                    LLM_CONCURRENCY=max(2, slots),           # pages / figures a worker works on at once: alone (the last chunk) it fills every slot
                    LLM_INFLIGHT=max(4, slots),              # its requests in flight at most: all workers together can't flood the queues
                    MARKER_LLM_CONCURRENCY=2 * max(1, math.ceil(2.5 * slots / workers)))   # Marker's own LLM steps (not gated): as before
        # 2.5x the slots over all workers: while some workers are in Marker's GPU stage, the others still fill every LLM slot
        # (extra requests wait in llama-server's queue, well within the 900 s timeout)
    return plan


def add_review(row, stats):
    """page-review totals for a report row: pages_rewritten, and why the others kept Marker's draft"""
    stats = [s for s in stats if s]
    if stats:
        row["pages_rewritten"] = sum(s.get("pages_rewritten", s.get("rewritten", 0)) for s in stats)
        kept = {}
        for s in stats:
            for why, n in (s.get("pages_kept_draft") or s.get("kept_draft") or {}).items():
                kept[why] = kept.get(why, 0) + n
        row["pages_kept_draft"] = kept
        row["review_examples"] = [e for s in stats for e in (s.get("examples") or [])][:12]


def add_stages(row, stages):
    """worker seconds per stage (marker, figures, describe_figures, page_review), summed over the pieces"""
    tot = {}
    for st in stages:
        for k, v in (st or {}).items():
            tot[k] = round(tot.get(k, 0) + v, 1)
    if tot:
        row["stage_seconds"] = tot


def resolve_mode(requested, warnings):
    m = requested or DEFAULTS["MODE"]
    if m == "auto":
        return "balanced" if POOL.vlm_ok else "fast"
    if m in ("balanced", "vllm") and not POOL.vlm_ok:
        warnings.append(f"mode '{m}' needs the VLM server, which is not running on this box; used 'fast' instead")
        return "fast"
    return m


# ----------------------------------------------------------------- the job
class ResourceSampler(threading.Thread):
    """GPU busy time and memory, CPU and free RAM every few seconds while a job runs, for its report: is the box used fully?"""
    def __init__(self, every=2.0):
        super().__init__(daemon=True)
        self.every, self.samples, self.done = every, [], threading.Event()

    @staticmethod
    def cpu_seconds():
        for path, scale in (("/sys/fs/cgroup/cpu.stat", 1e-6), ("/sys/fs/cgroup/cpuacct/cpuacct.usage", 1e-9)):
            try:
                text = Path(path).read_text()
                return float(text.split("usage_usec")[1].split()[0] if "usage_usec" in text else text.strip()) * scale
            except (OSError, ValueError, IndexError):
                continue
        try:                                                 # no cgroup: the whole machine
            f = Path("/proc/stat").read_text().split("\n")[0].split()[1:]
            return (sum(map(float, f)) - float(f[3]) - float(f[4])) / os.sysconf("SC_CLK_TCK")
        except (OSError, ValueError, IndexError):
            return None

    def run(self):
        cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        last_t, last_c = time.time(), self.cpu_seconds()
        while not self.done.wait(self.every):
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                gpus = [tuple(int(x) for x in l.split(",")) for l in out.strip().splitlines() if l.strip()]
            except Exception:
                gpus = []
            lim, used = memory()
            now, c = time.time(), self.cpu_seconds()
            cpu = 100 * (c - last_c) / max(1e-6, (now - last_t) * cpus) if c is not None and last_c is not None else None
            last_t, last_c = now, c
            self.samples.append((gpus, (lim - used) / 2 ** 30, cpu))

    def summary(self):
        self.done.set()
        s = self.samples
        if not s:
            return {}
        n = max(len(g) for g, _, _ in s)
        util = [[g[i][0] for g, _, _ in s if len(g) > i] for i in range(n)]
        out = {"seconds_sampled": round(len(s) * self.every),
               "gpu_busy_pct": [round(sum(u) / len(u)) for u in util if u],
               "gpu_mem_max_mb": [max(g[i][1] for g, _, _ in s if len(g) > i) for i in range(n)],
               "ram_free_min_gb": round(min(f for _, f, _ in s), 1)}
        cpu = [c for _, _, c in s if c is not None]
        if cpu:
            out["cpu_busy_pct"] = round(sum(cpu) / len(cpu))
        return out


def run_job(job, opts, session_deadline, upd, cancelled):
    """job: {id, name, src, dir}. upd(stage, pct, msg) reports progress; cancelled() -> bool. Returns a result summary dict."""
    o = {**DEFAULTS, **(opts or {})}
    jdir = Path(job["dir"]); inp, out = jdir / "in", jdir / "out"
    for d in (inp, out):
        shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    t_start = time.time()
    deadline = min(session_deadline - 120, t_start + float(o["TIME_BUDGET_H"]) * 3600)
    warnings, tids = [], {}
    upd("preparing", 1, "unpacking")
    name = job["name"]
    if name.lower().endswith(".zip"):
        docs = safe_extract(job["src"], inp, int(o["MAX_ZIP_FILES"]), int(float(o["MAX_UNZIPPED_GB"]) * 2 ** 30))
        if not docs:
            raise RuntimeError("no supported documents in the zip (supported: " + ", ".join(DOC_EXT) + ")")
        tag = safe_part(Path(name).stem)
    else:
        dst = inp / safe_part(name, 120)
        shutil.move(str(job["src"]), str(dst)); docs = [(dst, dst.name)]
        tag = safe_part(dst.stem)
    mode = resolve_mode((opts or {}).get("MODE"), warnings)
    task_opts = {k: v for k, v in (opts or {}).items() if k != "MODE"}
    task_opts["MODE"] = mode
    task_opts["OFFICE"] = (opts or {}).get("OFFICE") or o["OFFICE"]
    task_opts["QUALITY"] = str((opts or {}).get("QUALITY") or o["QUALITY"]).lower()
    if task_opts["QUALITY"] not in R.QUALITY_PROFILES:
        warnings.append(f"unknown quality {task_opts['QUALITY']!r}: balanced used"); task_opts["QUALITY"] = "balanced"
    llm = llm_plan(opts or {}, max(1, len(POOL.procs)), warnings)
    task_opts.update(llm)
    used, out_rel, doc_dirs, chunks = set(), {}, {}, {}      # chunks: doc index -> [(tid, pages)]
    workers = max(1, len(POOL.procs))
    for i, (path, rel) in enumerate(docs):
        rel_dir, stem = str(Path(rel).parent) if Path(rel).parent != Path(".") else "", Path(rel).stem
        key = (rel_dir.lower(), stem.lower())
        folder = stem if key not in used else f"{stem}_{Path(rel).suffix.lstrip('.').lower()}"
        used.add(key); used.add((rel_dir.lower(), folder.lower()))
        out_dir = out / rel_dir / folder if rel_dir else out / folder
        out_rel[i] = str(out_dir.relative_to(out)).replace(os.sep, "/")
        doc_dirs[i] = (out_dir, folder)
        path = pre_render(path, i, task_opts, warnings, lambda name: upd("preparing", 2, f"rendering {name} to PDF"))
        t = {"job": job["id"], "doc": i, "src": str(path), "rel": rel, "stem": folder, "out": str(out_dir), "out_rel": out_rel[i], "opts": task_opts}
        plan = chunk_plan(path, task_opts, workers, o)
        if not plan:
            tids[POOL.enqueue(t)] = i
            continue
        for k, part in enumerate(plan):
            pages = format_pages(part)
            tid = POOL.enqueue(dict(t, out=str(out_dir / ".chunks" / f"{k:04d}"), opts=dict(task_opts, PAGE_RANGE=pages)))
            tids[tid] = i; chunks.setdefault(i, []).append((tid, pages))
    STEALABLE[job["id"]] = [tid for c in chunks.values() for tid, _ in c]
    n = len(docs)
    n_tasks = len(tids)
    results, last_msg, expired = {}, 0, False
    doc_timeout = float(o["DOC_TIMEOUT_MIN"]) * 60
    sampler = ResourceSampler(); sampler.start()           # GPU / CPU / RAM use while this job runs (its report)
    try:
        while len(results) < n_tasks:
            if cancelled():
                POOL.cancel_job(job["id"]); POOL.maintain(doc_timeout)      # stops the documents of this job that are running right now
                raise Cancelled()
            if not expired and time.time() > deadline:
                expired = True; POOL.cancel_job(job["id"])
                warnings.append("time budget reached: documents that were not finished are listed as failed")
            POOL.maintain(doc_timeout)
            if POOL.broken:
                raise RuntimeError("conversion workers keep crashing:\n" + "\n".join(f"--- {w}\n{POOL.log(w, 15)}" for w in POOL.procs))
            for tid in tids:
                if tid not in results:
                    r = POOL.result(tid)
                    if r:
                        results[tid] = r
            if time.time() - last_msg > 2:
                last_msg = time.time()
                run = [(tid, i) for tid, i in POOL.running().items() if tid in tids and tid not in results]
                now_txt = "; ".join(f"{docs[tids[tid]][1]} ({i['wid']}, {time.time() - i['t0']:.0f}s" + (f", {POOL.progress_line(i['wid'])}" if POOL.progress_line(i['wid']) else "") + ")" for tid, i in run[:3])
                done = f"{len(results)}/{n_tasks} parts of {n} documents" if chunks else f"{len(results)}/{n} documents"
                if (POOL.dir / "pause").exists():
                    done += f" · RAM low ({(_read(str(POOL.dir / 'pause')) or '?')} GB free): one piece at a time"
                upd("converting", 3 + 92 * len(results) / n_tasks, f"[{mode}] {done} done" + (f" · working on {now_txt}" if now_txt else " · waiting for a free worker"))
            time.sleep(0.4)
        # ---- collect + bundle
        upd("bundling", 96, "writing the report and the zip")
        STEALABLE.pop(job["id"], None)
        per_doc = []
        for i in range(n):
            rs = [results[tid] for tid, d in tids.items() if d == i]
            r = next((x for x in rs if x.get("ok") and not x.get("remote")), rs[0])
            row = {"file": docs[i][1], "ok": any(x.get("ok") for x in rs), "pages": sum(x.get("pages") or 0 for x in rs if x.get("ok")) or None,
                   "seconds": round(sum(x.get("sec") or 0 for x in rs), 2), "worker": ",".join(sorted({str(x.get("worker")) for x in rs})), "via": r.get("via")}
            if i in chunks:
                parts = [(pages, Path(doc_dirs[i][0]) / ".chunks" / f"{k:04d}",
                          "remote" if results[tid].get("remote") else ("ok" if results[tid].get("ok") else results[tid].get("error") or "failed"))
                         for k, (tid, pages) in enumerate(chunks[i])]
                row.update(convert.merge_chunks(doc_dirs[i][0], doc_dirs[i][1], parts))
            cov = [c for x in rs for c in (x.get("coverage") or [])]
            if cov:                                            # share of the PDF text layer's words found in the Markdown
                row["text_coverage"] = round(sum(c["found"] * c["words"] for c in cov) / max(1, sum(c["words"] for c in cov)), 3)
                row["low_coverage_pages"] = [c["page"] for c in cov if c["low"]]
            if any(x.get("described_figures") for x in rs):
                row["figures_described"] = sum(x.get("described_figures") or 0 for x in rs)
            add_review(row, [x.get("page_review") for x in rs])
            add_stages(row, [x.get("stage_seconds") for x in rs])
            notes = sorted({x["forced_ocr"] for x in rs if x.get("forced_ocr")})
            if notes:
                row["forced_ocr"] = notes
            rej = {}
            for x in rs:
                for k, v in (x.get("marker_llm_rejected") or {}).items():
                    rej[k] = rej.get(k, 0) + v
            if rej:
                row["marker_llm_rejected"] = rej
            skipped = {}
            for x in rs:
                for k, v in (x.get("tables_skipped") or {}).items():
                    skipped[k] = skipped.get(k, 0) + v
            if skipped:
                row["tables_skipped"] = skipped
            dskip = {}
            for x in rs:
                for k, v in (x.get("drawn_formulas_skipped") or {}).items():
                    dskip[k] = dskip.get(k, 0) + v
            if dskip:
                row["drawn_formulas_skipped"] = dskip
            errs = sorted({x["tables_error"] for x in rs if x.get("tables_error")})
            if errs:
                row["tables_error"] = errs
            audits = [x.get("audit") for x in rs if x.get("audit")]
            if audits:                                         # accurate: what the line-by-line audit proposed and did
                total = {}
                for st in audits:
                    for k, v in st.items():
                        if isinstance(v, int):
                            total[k] = total.get(k, 0) + v
                total["examples"] = [e for st in audits for e in st.get("examples", [])][:12]
                row["audit"] = total
            qs = sorted({x.get("quality") for x in rs if x.get("quality")})
            if qs:
                row["quality"] = qs[0] if len(qs) == 1 else qs
            for key in ("tables_realigned", "table_pictures", "drawn_formulas", "layer_fixes"):
                if any(x.get(key) for x in rs):
                    row[key] = sum(x.get(key) or 0 for x in rs)
            if any(x.get("indic_words_fixed") for x in rs):
                row["indic_words_fixed"] = sum(x.get("indic_words_fixed") or 0 for x in rs)
            if not row["ok"]:
                row["error"] = r.get("error")
            else:
                row["output_dir"] = out_rel[i]
            per_doc.append(row)
        pages = sum(d["pages"] or 0 for d in per_doc if d["ok"])
        warnings += [e for e in POOL.mem_events if any(e.startswith(dd[1] + ":") for dd in docs)]
        resources = sampler.summary()                         # GPU busy %, CPU %, lowest free RAM while this job ran
        for row in per_doc:
            row["box_resources"] = resources
        report = {"mode": mode, "llm": (o["LLM_MODEL"] or "gemini") if llm.get("USE_LLM") else None, "documents": per_doc, "warnings": warnings,
                  "wall_s": round(time.time() - t_start), "pages": pages, "resources": resources}
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        bundle = shutil.make_archive(str(jdir / f"{tag}.markdown"), "zip", out)
        n_ok = sum(d["ok"] for d in per_doc)
        return {"mode": mode, "docs": per_doc, "n_ok": n_ok, "n_failed": n - n_ok, "warnings": warnings, "pages": pages, "bundle": bundle,
                "files": sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".html"))[:200],
                "wall_s": round(time.time() - t_start), "partial": n_ok < n}
    finally:
        sampler.done.set()                                       # a failed or cancelled job stops sampling too
        STEALABLE.pop(job["id"], None)
        shutil.rmtree(inp, ignore_errors=True)
