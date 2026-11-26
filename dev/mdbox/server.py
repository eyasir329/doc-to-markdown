"""Document -> Markdown box API (runs on the Kaggle GPU machine, exposed through a Cloudflare tunnel).

Every request needs `Authorization: Bearer <BOX_TOKEN>` (except GET /ping).
Cloudflare's free tunnel rejects request bodies over ~100 MB and cuts requests after ~100 s, so a document is uploaded in parts:

  POST   /jobs                 {"name", "size", "options"?}         -> {"job_id", "part_size"}
  PUT    /jobs/{id}/parts/{n}  raw bytes (header X-Sha256 optional)  -> {"part", "bytes"}
  POST   /jobs/{id}/start      {"parts": N}                          -> job status (queued)
  GET    /jobs/{id}                                                  -> job status / progress
  GET    /jobs/{id}/bundle.zip   everything in one zip (<name>/<name>.md + images, report.json)
  GET    /jobs/{id}/files/{path} one output file
  DELETE /jobs/{id}            cancel (running) or delete (finished)
  POST   /files {"name","size"} -> {"file_id","part_size"};  PUT /files/{fid}/parts/{n};  POST /files/{fid}/finish {"parts"}
         -> {"pages","pdf"}  store one document once (office/DjVu rendered to PDF);  DELETE /files/{fid}
  POST   /jobs {"file_id", "name", "options": {"PAGE_RANGE": ...}}  a job on a stored file (one batch of pages), queued at once
  POST   /jobs/{id}/steal      {"n"}  -> {"chunks": [...]}  page chunks the client converts on its own workers
  *      /vlm/...  /llm/...    this box's Surya VLM / helper vision LLM (OpenAI API), for the client's own workers
  GET    /jobs      GET /health      GET /ping (no auth)
A job is one document (see pipeline.DOC_EXT: pdf, images, Word/PowerPoint/Excel old and new, OpenDocument, RTF, DjVu, html, epub, txt/md/csv)
or a .zip of documents. Up to RUNNERS jobs run at the same time."""
import asyncio, hashlib, hmac, os, queue, re, shutil, subprocess, threading, time, traceback, uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response

import pipeline

TOKEN = os.environ["BOX_TOKEN"].strip()
if len(TOKEN) < 16:
    raise SystemExit("BOX_TOKEN must be at least 16 characters")
ROOT = Path(os.environ.get("MD_ROOT", "jobs")).resolve()
UPLOADS = Path(os.environ.get("MD_UPLOADS", str(pipeline.TMP_ROOT / "md_uploads"))).resolve()
DEADLINE = float(os.environ.get("BOX_DEADLINE", "0") or 0) or time.time() + 10 * 3600
PART_SIZE = 16 * 1024 * 1024
MAX_UPLOAD = int(float(os.environ.get("MD_MAX_UPLOAD_GB", "3")) * 2 ** 30)
HAS_GEMINI = bool(os.environ.get("GOOGLE_API_KEY"))
RUNNERS = int(os.environ.get("MD_RUNNERS", "8"))       # jobs at once; their documents/pages share the worker queue
ABANDON_S = float(os.environ.get("MD_ABANDON_MIN", "15")) * 60     # a job no client has asked about for this long is cancelled
                                                       # (its client was killed: it would hold the workers for nothing); 0 = never
VLM_BASE = pipeline.DEFAULTS["VLM_URL"].rstrip("/").removesuffix("/v1")
LLM_BASE = pipeline.DEFAULTS["LLM_URL"].rstrip("/").removesuffix("/v1")
HAS_LLM = bool(LLM_BASE) or HAS_GEMINI
ROOT.mkdir(parents=True, exist_ok=True); UPLOADS.mkdir(parents=True, exist_ok=True)

JOBS, ORDER, Q, LOCK = {}, [], queue.Queue(), threading.Lock()
STARTED = time.time()
LAST_ACTIVITY = [STARTED]
app = FastAPI(title="markdown box", docs_url=None, redoc_url=None, openapi_url=None)

OPT_TYPES = {"MODE": str, "OUTPUT_FORMAT": str, "FORCE_OCR": bool, "STRIP_EXISTING_OCR": bool, "REDO_INLINE_MATH": bool, "PAGINATE": bool,
             "KEEP_HEADERS_FOOTERS": bool, "IMAGES": bool, "PAGE_RANGE": str, "USE_LLM": bool, "OFFICE": str,
             "FIGURE_FIX": bool, "DESCRIBE_FIGURES": bool, "PAGE_REVIEW": bool, "QUALITY": str, "AUDIT": bool,
             "CHUNK_ROUNDS": int, "SPLIT_SPREADS": bool, "SHOW_LAYOUT": bool, "LABEL_BLOCKS": bool}


def number(value, what, lo=0):
    """a whole number from a request (JSON or text) -> int, else 400 (a client error, not a box error)"""
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise HTTPException(400, f"{what} must be a whole number")
    if n < lo:
        raise HTTPException(400, f"{what} must be at least {lo}")
    return n


async def body_of(request):
    """the request's JSON object, else 400"""
    try:
        body = await request.json()
    except ValueError:
        raise HTTPException(400, "the request body must be JSON")
    if not isinstance(body, dict):
        raise HTTPException(400, "the request body must be a JSON object")
    return body


def clean_options(raw):
    out = {}
    if raw is not None and not isinstance(raw, dict):
        raise HTTPException(400, "options must be a JSON object")
    for k, v in (raw or {}).items():
        k = k.upper()
        if k not in OPT_TYPES:
            raise HTTPException(400, f"unknown option {k!r}; allowed: {sorted(OPT_TYPES)}")
        if v is None:
            continue
        if OPT_TYPES[k] is bool and not isinstance(v, bool):
            raise HTTPException(400, f"option {k} must be true or false")
        if OPT_TYPES[k] is str and not isinstance(v, str):
            raise HTTPException(400, f"option {k} must be a string")
        if OPT_TYPES[k] is int and (isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 8):
            raise HTTPException(400, f"option {k} must be a whole number from 1 to 8")
        if k == "MODE" and v not in pipeline.MODES:
            raise HTTPException(400, f"MODE must be one of {list(pipeline.MODES)}")
        if k == "OFFICE" and v not in pipeline.convert.OFFICE_MODES:
            raise HTTPException(400, f"OFFICE must be one of {list(pipeline.convert.OFFICE_MODES)}")
        if k == "OUTPUT_FORMAT" and v not in ("markdown", "html", "json", "chunks"):
            raise HTTPException(400, "OUTPUT_FORMAT must be markdown, html, json or chunks")
        if k == "PAGE_RANGE" and not re.fullmatch(r"[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*", v.replace(" ", "")):
            raise HTTPException(400, "PAGE_RANGE looks like 0,5-10,20 (page numbers start at 0)")
        if k == "QUALITY":                                 # an old name (superfast, balanced, ...) becomes the current one
            q = pipeline.R.canonical_quality(v)
            if q is None:
                raise HTTPException(400, f"QUALITY must be one of {list(pipeline.R.QUALITIES)} (old names: "
                                         f"{sorted(pipeline.R.QUALITY_ALIASES)})")
            v = q
        if k == "USE_LLM" and v and not HAS_LLM:
            raise HTTPException(400, "USE_LLM needs a local LLM on this box (LLM_LOCAL in the notebook) or the Kaggle secret GOOGLE_API_KEY")
        out[k] = v.replace(" ", "") if k == "PAGE_RANGE" else v
    return out


@app.middleware("http")
async def auth(request: Request, call_next):
    if request.url.path == "/ping" and request.method == "GET":
        return await call_next(request)
    if not hmac.compare_digest(request.headers.get("authorization", "").encode(), f"Bearer {TOKEN}".encode()):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return await call_next(request)


@app.get("/ping")
async def ping():
    return PlainTextResponse("ok")


def gpus():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=5).stdout
        rows = [[c.strip() for c in l.split(",")] for l in out.strip().splitlines() if l.strip()]
        return [{"index": int(r[0]), "name": r[1], "mem_used_mb": int(r[2]), "mem_total_mb": int(r[3]), "util_pct": int(r[4])} for r in rows]
    except Exception:
        return []


PUBLIC_KEYS = ("id", "name", "size", "status", "stage", "pct", "msg", "created", "started", "finished", "options", "error",
               "mode", "docs", "n_ok", "n_failed", "warnings", "pages", "partial", "files", "wall_s")


def public(j):
    d = {k: j.get(k) for k in PUBLIC_KEYS}
    if j["status"] == "queued":
        d["queue_position"] = sum(1 for i in ORDER if JOBS[i]["status"] in ("queued", "running") and JOBS[i]["created"] < j["created"]) + 1
    return d


def get_job(jid):
    j = JOBS.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j


def mem_status():
    d, G = pipeline.memory_detail(), 2 ** 30
    return {"limit_gb": round(d["limit"] / G, 1), "used_gb": round(d["used"] / G, 1), "free_gb": round((d["limit"] - d["used"]) / G, 1),
            "file_cache_gb": round(d["cache"] / G, 1), "paused": (pipeline.POOL.dir / "pause").exists(),
            "top_processes_mb": pipeline.processes(8)}


def idle_seconds():
    """0 while anything is uploading/queued/running, else the time since the last job activity (the notebook stops the box when this gets long)"""
    if any(j["status"] in ("uploading", "queued", "running") for j in JOBS.values()):
        return 0
    return round(time.time() - LAST_ACTIVITY[0])


CODE_ID = hashlib.sha1(b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()[:10]


@app.get("/health")
async def health():
    now = time.time()
    return {"ok": True, "box": "markdown", "code": CODE_ID, "gpus": gpus(), "uptime_s": round(now - STARTED), "remaining_s": round(DEADLINE - now),
            "workers": pipeline.POOL.status(), "chunk_pages": int(pipeline.DEFAULTS["CHUNK_MAX_PAGES"]), "vlm": pipeline.POOL.vlm_ok, "vlm_slots": int(pipeline.DEFAULTS["VLM_SLOTS"]),
            "waiting_tasks": pipeline.POOL.waiting() + sum(1 for j in JOBS.values() if j["status"] == "queued" or j.get("stage") == "preparing"), "default_mode": "balanced" if pipeline.POOL.vlm_ok else "fast",
            "llm": HAS_LLM, "llm_model": pipeline.DEFAULTS["LLM_MODEL"] or ("gemini" if HAS_GEMINI else ""),
            "llm_local": bool(LLM_BASE), "llm_slots": int(pipeline.DEFAULTS["LLM_SLOTS"]), "llm_default": bool(pipeline.DEFAULTS["USE_LLM"]),
            "defaults": {k: pipeline.DEFAULTS[k] for k in ("REDO_INLINE_MATH", "DESCRIBE_FIGURES", "PAGE_REVIEW", "LLM_CTX", "QUALITY")},
            "qualities": list(pipeline.R.QUALITIES),
            "memory": mem_status(), "libreoffice": bool(pipeline.convert.SOFFICE), "ddjvu": bool(pipeline.convert.DDJVU), "idle_s": idle_seconds(), "queued": sum(1 for j in JOBS.values() if j["status"] == "queued"),
            "running": [j["id"] for j in JOBS.values() if j["status"] == "running"], "max_upload_gb": MAX_UPLOAD / 2 ** 30,
            "extensions": list(pipeline.DOC_EXT) + [".zip"]}


@app.get("/jobs")
async def list_jobs():
    return [public(JOBS[i]) for i in ORDER]


@app.post("/jobs")
async def create_job(request: Request):
    body = await body_of(request)
    if body.get("file_id"):
        return await job_from_file(body)
    name, size = Path(str(body.get("name") or "document.pdf")).name, number(body.get("size") or 0, "size")
    if size <= 0 or size > MAX_UPLOAD:
        raise HTTPException(413, f"size must be 1..{MAX_UPLOAD} bytes")
    if not name.lower().endswith(pipeline.DOC_EXT + (".zip",)):
        raise HTTPException(415, f"unsupported file type {Path(name).suffix!r}; supported: {', '.join(pipeline.DOC_EXT)} and .zip of those")
    if DEADLINE - time.time() < 600:
        raise HTTPException(503, "this box session is about to end; start a new one")
    jid = uuid.uuid4().hex[:12]
    j = {"id": jid, "name": name, "size": size, "status": "uploading", "stage": "uploading", "pct": 0.0, "msg": "", "created": time.time(),
         "options": clean_options(body.get("options")), "dir": str(ROOT / jid), "updir": str(UPLOADS / jid), "cancel": False}
    Path(j["dir"]).mkdir(parents=True); Path(j["updir"]).mkdir(parents=True)
    with LOCK:
        JOBS[jid] = j; ORDER.append(jid)
    LAST_ACTIVITY[0] = time.time()
    return {"job_id": jid, "part_size": PART_SIZE}


@app.put("/jobs/{jid}/parts/{n}")
async def put_part(jid: str, n: int, request: Request):
    j = get_job(jid)
    if j["status"] != "uploading":
        raise HTTPException(409, f"job is {j['status']}")
    j["seen"] = time.time()
    return await save_part(j["updir"], n, request, j["size"])


async def save_part(updir, n, request, limit=MAX_UPLOAD):
    """one part of an upload; never more bytes than the whole file was declared to have (the disk can't be filled)"""
    if not 0 <= n < 100000:
        raise HTTPException(400, "bad part number")
    dest = Path(updir) / f"{n:05d}.part"; tmp = dest.with_suffix(".tmp")
    h, size = hashlib.sha256(), 0
    with open(tmp, "wb") as f:
        async for chunk in request.stream():
            size += len(chunk)
            if size > limit:
                f.close(); tmp.unlink(missing_ok=True)
                raise HTTPException(413, "part larger than the declared file size")
            f.write(chunk); h.update(chunk)
    want = request.headers.get("x-sha256")
    if want and want.lower() != h.hexdigest():
        tmp.unlink(missing_ok=True)
        raise HTTPException(422, "checksum mismatch; resend this part")
    os.replace(tmp, dest)
    LAST_ACTIVITY[0] = time.time()
    return {"part": n, "bytes": size}


def assemble(j, n):
    up = Path(j["updir"]); parts = [up / f"{i:05d}.part" for i in range(n)]
    missing = [i for i, p in enumerate(parts) if not p.exists()]
    if missing:
        return missing, 0
    doc = up / "upload.bin"
    with open(doc, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 8 * 1024 * 1024)
            p.unlink()
    return [], doc.stat().st_size


# ----------------------------------------------------------------- stored files (one upload, many page-range jobs)
FILES = {}


def get_file(fid):
    f = FILES.get(fid)
    if not f:
        raise HTTPException(404, "no such file on this box (it may have restarted): upload it again")
    return f


@app.post("/files")
async def create_file(request: Request):
    body = await body_of(request)
    name, size = Path(str(body.get("name") or "document.pdf")).name, number(body.get("size") or 0, "size")
    if size <= 0 or size > MAX_UPLOAD:
        raise HTTPException(413, f"size must be 1..{MAX_UPLOAD} bytes")
    if not name.lower().endswith(pipeline.DOC_EXT):
        raise HTTPException(415, f"unsupported file type {Path(name).suffix!r}; supported: {', '.join(pipeline.DOC_EXT)}")
    fid = uuid.uuid4().hex[:12]
    d = UPLOADS / f"file_{fid}"; d.mkdir(parents=True)
    FILES[fid] = {"id": fid, "name": name, "size": size, "dir": str(d), "status": "uploading", "seen": time.time()}
    LAST_ACTIVITY[0] = time.time()
    return {"file_id": fid, "part_size": PART_SIZE}


@app.put("/files/{fid}/parts/{n}")
async def put_file_part(fid: str, n: int, request: Request):
    f = get_file(fid)
    if f["status"] != "uploading":
        raise HTTPException(409, f"file is {f['status']}")
    f["seen"] = time.time()
    return await save_part(f["dir"], n, request, f["size"])


def finish_file(f, n, office):
    missing, size = assemble({"updir": f["dir"]}, n)
    if missing:
        return {"error": f"missing parts: {missing[:20]}"}
    if size != f["size"]:
        return {"error": f"assembled {size} bytes but {f['size']} were announced"}
    src = Path(f["dir"]) / pipeline.safe_part(f["name"], 120)
    os.replace(Path(f["dir"]) / "upload.bin", src)
    warnings = []
    path = pipeline.pre_render(src, 0, {"OFFICE": office}, warnings)      # office / DjVu -> PDF once, here
    pdf = path.suffix.lower() == ".pdf"
    f.update(status="ready", path=str(path), pdf=pdf, pages=pipeline.pdf_pages(path) if pdf else None)
    return {"file_id": f["id"], "pages": f["pages"], "pdf": pdf, "warnings": warnings}


@app.post("/files/{fid}/finish")
async def finish(fid: str, request: Request):
    f = get_file(fid)
    if f["status"] != "uploading":
        raise HTTPException(409, f"file is {f['status']}")
    body = await body_of(request)
    office = clean_options({"OFFICE": (body.get("options") or {}).get("OFFICE")}).get("OFFICE") or pipeline.DEFAULTS["OFFICE"]
    n = number(body.get("parts"), "parts", 1)
    f.update(status="rendering", seen=time.time())          # office/DjVu -> PDF can take long: never expired meanwhile
    try:
        res = await asyncio.get_running_loop().run_in_executor(None, finish_file, f, n, office)
    finally:
        f["seen"] = time.time()
        if f["status"] == "rendering":                      # failed: the client may send the missing parts again
            f["status"] = "uploading"
    if res.get("error"):
        raise HTTPException(409, res["error"])
    LAST_ACTIVITY[0] = time.time()
    return res


@app.delete("/files/{fid}")
async def delete_file(fid: str):
    f = get_file(fid)
    shutil.rmtree(f["dir"], ignore_errors=True)
    FILES.pop(fid, None)
    return {"deleted": fid}


async def job_from_file(body):
    f = get_file(str(body["file_id"]))
    if f["status"] != "ready":
        raise HTTPException(409, f"file is {f['status']}")
    f["seen"] = time.time()
    name = Path(str(body.get("name") or Path(f["path"]).name)).name
    if f["pdf"] and not name.lower().endswith(".pdf"):
        name = Path(name).stem + ".pdf"                                   # it was rendered to PDF on upload
    if DEADLINE - time.time() < 600:
        raise HTTPException(503, "this box session is about to end; start a new one")
    jid = uuid.uuid4().hex[:12]
    j = {"id": jid, "name": name, "size": Path(f["path"]).stat().st_size, "status": "queued", "stage": "queued", "pct": 0.0,
         "msg": "waiting for a free slot", "created": time.time(), "options": clean_options(body.get("options")),
         "dir": str(ROOT / jid), "updir": str(UPLOADS / jid), "cancel": False, "file_id": f["id"]}
    Path(j["dir"]).mkdir(parents=True); Path(j["updir"]).mkdir(parents=True)
    os.link(f["path"], Path(j["updir"]) / "upload.bin")                    # the job consumes a link; the stored file stays
    with LOCK:
        JOBS[jid] = j; ORDER.append(jid)
    Q.put(jid)
    LAST_ACTIVITY[0] = time.time()
    return {"job_id": jid, **public(j)}


@app.post("/jobs/{jid}/start")
async def start_job(jid: str, request: Request):
    j = get_job(jid)
    if j["status"] != "uploading":
        raise HTTPException(409, f"job is {j['status']}")
    n = number((await body_of(request)).get("parts"), "parts", 1)
    missing, size = await asyncio.get_running_loop().run_in_executor(None, assemble, j, n)
    if missing:
        raise HTTPException(409, f"missing parts: {missing[:20]}")
    if size != j["size"]:
        raise HTTPException(422, f"assembled {size} bytes but {j['size']} were announced")
    j.update(status="queued", stage="queued", msg="waiting for a free slot")
    Q.put(jid)
    return public(j)


@app.get("/jobs/{jid}")
async def job_status(jid: str):
    j = get_job(jid)
    j["seen"] = time.time()                                # its client is there (polls every 0.5-2 s)
    return public(j)


def abandoned(j):
    """no client has asked about this job for ABANDON_S: its client was stopped or lost (a killed client's full-suite job once
    ran 27 min beside the next one)"""
    return ABANDON_S > 0 and time.time() - (j.get("seen") or j.get("created") or time.time()) > ABANDON_S


@app.get("/jobs/{jid}/bundle.zip")
async def bundle(jid: str):
    j = get_job(jid)
    if j["status"] != "done" or not j.get("bundle"):
        raise HTTPException(409, f"job is {j['status']}")
    return FileResponse(j["bundle"], media_type="application/zip", filename=Path(j["bundle"]).name)


@app.get("/jobs/{jid}/files/{name:path}")
async def one_file(jid: str, name: str):
    j = get_job(jid)
    base = (Path(j["dir"]) / "out").resolve()
    p = (base / name).resolve()
    if j["status"] != "done" or base not in p.parents or not p.is_file():
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=p.name)


@app.delete("/jobs/{jid}")
async def delete_job(jid: str):
    j = get_job(jid)
    if j["status"] in ("running", "queued", "uploading"):
        j["cancel"] = True
        if j["status"] != "running":
            j.update(status="cancelled", stage="cancelled", finished=time.time())
            shutil.rmtree(j["updir"], ignore_errors=True)
        return {"cancelled": jid}
    shutil.rmtree(j["dir"], ignore_errors=True)
    with LOCK:
        JOBS.pop(jid, None)
        if jid in ORDER:
            ORDER.remove(jid)
    return {"deleted": jid}


@app.post("/jobs/{jid}/steal")
async def steal(jid: str, request: Request):
    j = get_job(jid)
    j["seen"] = time.time()
    n = number((await body_of(request)).get("n") or 1, "n")
    if j["status"] != "running" or n < 1:
        return {"chunks": []}
    return {"chunks": pipeline.steal(jid, min(n, 16))}


PROXY_CLIENT = []


async def gpu_proxy(base, what, path, request):
    """the client's own workers use this box's GPU servers through here (the token is checked by the middleware)"""
    if not base:
        raise HTTPException(503, f"no {what} server is running on this box")
    import httpx
    if not PROXY_CLIENT:
        PROXY_CLIENT.append(httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), limits=httpx.Limits(max_connections=128)))
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")}
    try:
        r = await PROXY_CLIENT[0].request(request.method, f"{base}/{path}", params=request.query_params, content=await request.body(), headers=headers)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"{what} server unreachable: {type(e).__name__}")
    LAST_ACTIVITY[0] = time.time()
    return Response(content=r.content, status_code=r.status_code, media_type=r.headers.get("content-type"))


@app.api_route("/vlm/{path:path}", methods=["GET", "POST"])
async def vlm_proxy(path: str, request: Request):
    return await gpu_proxy(VLM_BASE, "VLM", path, request)


@app.api_route("/llm/{path:path}", methods=["GET", "POST"])
async def llm_proxy(path: str, request: Request):
    return await gpu_proxy(LLM_BASE, "LLM", path, request)


def runner():
    while True:
        jid = Q.get()
        j = JOBS.get(jid)
        if not j or j["status"] != "queued":
            continue
        if abandoned(j):
            j.update(status="cancelled", stage="cancelled", msg="cancelled: no client asked about it", finished=time.time())
            shutil.rmtree(j["updir"], ignore_errors=True)
            continue
        j.update(status="running", started=time.time(), stage="starting", pct=0.0)

        def upd(stage, pct=None, msg="", j=j):
            j["stage"], j["msg"] = stage, msg
            if pct is not None:
                j["pct"] = round(float(pct), 1)
        try:
            res = pipeline.run_job({"id": jid, "name": j["name"], "src": str(Path(j["updir"]) / "upload.bin"), "dir": j["dir"]},
                                   j["options"], DEADLINE, upd, lambda j=j: j["cancel"] or abandoned(j))
            j.update(res, status="done", stage="done", pct=100.0, msg=f"{res['n_failed']} of {res['n_ok'] + res['n_failed']} documents failed (see report.json)" if res["n_failed"] else "")
        except pipeline.Cancelled:
            j.update(status="cancelled", stage="cancelled", msg="cancelled" if j["cancel"] else "cancelled: no client asked about it")
        except Exception as e:
            traceback.print_exc()
            j.update(status="error", stage="error", error=f"{type(e).__name__}: {e}"[:1500])
        finally:
            j["finished"] = time.time(); LAST_ACTIVITY[0] = time.time()
            shutil.rmtree(j["updir"], ignore_errors=True)


def expire_abandoned():
    """uploads no client has touched for ABANDON_S (its client was stopped mid-upload, or a POST /jobs retried after a lost
    answer left a twin): cancelled and deleted. An "uploading" job counts as activity, so one of them kept the box from ever
    being idle and the notebook never stopped it (GPU quota). Stored files no job uses any more go the same way (disk)"""
    if ABANDON_S <= 0:
        return
    for j in list(JOBS.values()):
        if j["status"] == "uploading" and abandoned(j):
            j.update(status="cancelled", stage="cancelled", msg="cancelled: the upload was abandoned", finished=time.time())
            shutil.rmtree(j["updir"], ignore_errors=True)
    used = {j.get("file_id") for j in JOBS.values() if j["status"] in ("queued", "running")}
    for fid, f in list(FILES.items()):
        if fid not in used and f["status"] != "rendering" and time.time() - f.get("seen", 0) > ABANDON_S:
            shutil.rmtree(f["dir"], ignore_errors=True)
            FILES.pop(fid, None)


def janitor():
    while True:
        time.sleep(30)
        try:
            expire_abandoned()
        except Exception:
            traceback.print_exc()


def memory_watch():
    """runs all the time, not only while a job polls: RAM can fill between two progress ticks"""
    while True:
        try:
            with pipeline.POOL.mtx:
                pipeline.POOL.guard_memory()
        except Exception:
            traceback.print_exc()
        time.sleep(0.5)


pipeline.start_pool()
threading.Thread(target=memory_watch, daemon=True).start()
threading.Thread(target=janitor, daemon=True).start()
for _ in range(RUNNERS):
    threading.Thread(target=runner, daemon=True).start()
