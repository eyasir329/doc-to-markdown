LOGS = Path("logs"); LOGS.mkdir(exist_ok=True); PROCS = {}
def start(name, cmd, env=None, cwd=None):
    stop(name)
    log = open(LOGS / f"{name}.log", "a"); log.write(f"\n==== start {time.strftime('%H:%M:%S')}: {cmd}\n"); log.flush()
    PROCS[name] = subprocess.Popen(["bash", "-lc", cmd], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=cwd, start_new_session=True)
def alive(name):
    p = PROCS.get(name); return p is not None and p.poll() is None
def stop(name):
    p = PROCS.pop(name, None)
    if p and p.poll() is None:
        try: os.killpg(p.pid, 15); p.wait(15)
        except Exception:
            try: os.killpg(p.pid, 9)
            except Exception: pass
def log_tail(name, n=25):
    f = LOGS / f"{name}.log"; return "\n".join(f.read_text(errors="replace").splitlines()[-n:]) if f.exists() else "(no log)"
def wait_http(url, timeout=60, every=2, ok=(200,)):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "box"}), timeout=5) as r:
                if r.status in ok: return True
        except Exception: pass
        time.sleep(every)
    return False

# llama.cpp releases daily and renames options (e.g. --no-mmap became --load-mode none): ask this build which it knows,
# and leave out optional flags it does not, instead of failing every start with "invalid argument"
LLAMA_HELP = (subprocess.run(f"{VLM_ENV} {VLM_BIN} --help", shell=True, capture_output=True, text=True, timeout=120).stdout
              if VLM_BIN else "")
def has_flag(flag):
    return not LLAMA_HELP or re.search(r"(^|[\s,`|])" + re.escape(flag) + r"([\s,=`|]|$)", LLAMA_HELP, re.M) is not None
NO_MMAP = "--load-mode none" if has_flag("--load-mode") else ("--no-mmap" if has_flag("--no-mmap") else "")   # weights go to the GPU, not RAM
OPTIONAL = ("--load-mode", "--no-mmap", "-fa", "--flash-attn", "-ctk", "-ctv", "--cache-type-k", "--cache-type-v", "--spec-type",
            "--image-max-tokens", "--reasoning-budget", "--threads", "--cache-ram", "--ctx-checkpoints")
NO_HOST_CACHE = "--cache-ram 0 --ctx-checkpoints 0"   # llama.cpp keeps up to 8 GB of old prompts in RAM per server by default: 4 servers filled the box
DROPPED = set()
def opt(flags):
    """the optional flags this llama-server build supports (each with its value); the others are left out and reported once"""
    out, toks, i = [], flags.split(), 0
    while i < len(toks):
        j = i + 1
        while j < len(toks) and not toks[j].startswith("-"):
            j += 1
        if toks[i] in OPTIONAL and not has_flag(toks[i]):
            if toks[i] not in DROPPED:
                DROPPED.add(toks[i]); print(f"   this llama-server has no {toks[i]}: left out")
        else:
            out += toks[i:j]
        i = j
    return " ".join(out)
def bad_args(name):
    """True if the server refused its command line: retrying with fewer slots cannot help"""
    return "invalid argument" in log_tail(name, 4) or "unknown argument" in log_tail(name, 4)
print("llama-server:", "help read," if LLAMA_HELP else "no --help output (all flags kept),", f"no-mmap via [{NO_MMAP or 'not available'}]")

SLOTS, VLM_PORTS = {}, []
def start_vlm(g, slots0):
    port = 8081 + g
    for flags in VLM_FLAG_VARIANTS:               # fastest settings first; plain llama.cpp defaults last
        slots = slots0
        while slots >= 1:
            ctx = slots * 12288                   # Surya's recommended per-slot context
            cmd = (f"CUDA_VISIBLE_DEVICES={g} {VLM_ENV} {VLM_BIN} -m {GGUF[0]} --mmproj {GGUF[1]} -ngl 99 "
                   f"{opt(NO_MMAP + ' --threads 1 ' + NO_HOST_CACHE + ' ' + flags)} "
                   f"--host 127.0.0.1 --port {port} --parallel {slots} --ctx-size {ctx} --alias {VLM_ALIAS} --jinja --special")
            start(f"vlm{g}", cmd)
            t0 = time.time()
            while time.time() - t0 < 600 and alive(f"vlm{g}"):
                if wait_http(f"http://127.0.0.1:{port}/health", timeout=6, every=3):
                    SLOTS[g] = slots; print(f"   GPU {g}: Surya with [{flags or 'defaults'}], {slots} slots"); return port
            print(f"GPU {g}: server did not come up with [{flags or 'defaults'}], {slots} slots:\n" + log_tail(f"vlm{g}", 6))
            stop(f"vlm{g}")
            if bad_args(f"vlm{g}"): break             # a flag problem: fewer slots will not help, try the next settings
            slots //= 2
    return None
def start_vlms():
    ports, th = {}, []
    for g in range(len(gpus)):
        t = threading.Thread(target=lambda g=g: ports.__setitem__(g, start_vlm(g, VLM_SLOTS_PER_GPU))); t.start(); th.append(t)
    for t in th: t.join()
    return [p for _, p in sorted(ports.items()) if p]
def warm_vlm(port):
    """one OCR-like request (a page-sized image): the GPU kernels get compiled now, not during the first job"""
    import base64, io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (1240, 1754), "white"); d = ImageDraw.Draw(im)
    for k in range(30):
        d.text((80, 60 + 50 * k), f"Line {k + 1}: warm-up text", fill="black")
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=85)
    body = {"model": VLM_ALIAS, "temperature": 0, "max_tokens": 32, "messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
        {"type": "text", "text": "OCR this image."}]}]}
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        print(f"   Surya warm-up on port {port} skipped: {str(e)[:120]}")

VLM_URL = ""
if USE_VLM and VLM_BIN and GGUF and gpus:
    t0 = time.time(); VLM_PORTS = start_vlms()
    th = [threading.Thread(target=warm_vlm, args=(p,)) for p in VLM_PORTS]
    for t in th: t.start()
    for t in th: t.join()
    if VLM_PORTS:
        start("lb", f"LB_MAX_INFLIGHT={VLM_LB_MAX_INFLIGHT} {sys.executable} mdbox/lb.py {VLM_LB_PORT} {' '.join(map(str, VLM_PORTS))}")
        if wait_http(f"http://127.0.0.1:{VLM_LB_PORT}/health", timeout=60):
            VLM_URL = f"http://127.0.0.1:{VLM_LB_PORT}/v1"
    print(f"VLM servers: {len(VLM_PORTS)} GPU(s), slots {SLOTS} in {time.time()-t0:.0f}s" if VLM_URL else "!!! no VLM server is running: CPU-only (Marker's fast reading mode)")

LLM_URL, LLM_SERVERS, LLM_RUN = "", {}, {}      # LLM_RUN[gpu] = (model path, mmproj path, flags, slots) that passed the test
def llm_ask(port, content, props, max_tokens):
    body = {"model": LLM_NAME, "temperature": 0, "max_tokens": max_tokens, "messages": [{"role": "user", "content": content}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": {
                "type": "object", "properties": {p: {"type": "string"} for p in props}, "required": list(props), "additionalProperties": False}}}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=900))
    return json.loads(r["choices"][0]["message"]["content"]), (r.get("timings") or {}).get("predicted_per_second") or 0.0
def llm_test(port):
    """like Marker's requests (image + JSON schema). The first call also warms up (the driver JIT-compiles the GPU kernels);
    the answer must be exactly right, and the speed is measured on a warm, longer answer. Returns tokens/s or raises."""
    import base64, io
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (220, 70), "white"); ImageDraw.Draw(im).text((20, 25), "TOTAL = 42", fill="black")
    buf = io.BytesIO(); im.save(buf, "PNG")
    img = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}},
           {"type": "text", "text": "Which number is written in the image? Answer with the digits only."}]
    for _ in range(2):
        ans, _ = llm_ask(port, img, ["number"], 32)
        if ans.get("number", "").strip() != "42":
            raise ValueError(f"wrong answer {ans!r} (expected 42)")
    page = Image.new("RGB", (1240, 1754), "white"); d = ImageDraw.Draw(page)    # a page-sized image too: its GPU kernels are
    for k in range(40):                                                          # compiled now, not during the first job
        d.text((80, 60 + 40 * k), f"Line {k + 1}: warm-up text for the vision model", fill="black")
    buf = io.BytesIO(); page.save(buf, "JPEG", quality=85)
    llm_ask(port, [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                   {"type": "text", "text": "What is written on the first line?"}], ["text"], 24)
    _, tps = llm_ask(port, [{"type": "text", "text": "Write the numbers from 1 to 80, separated by commas."}], ["numbers"], 300)
    if tps < LLM_MIN_TOK_S:
        raise ValueError(f"too slow: {tps:.1f} tok/s < LLM_MIN_TOK_S={LLM_MIN_TOK_S}")
    return tps
def start_llm(g, model, mmproj, flags, slots):
    """one LLM server on GPU g; returns its port if it starts and passes llm_test"""
    port = 8181 + g
    cache = f"--cache-ram {int(LLM_HOST_CACHE_MB)} --ctx-checkpoints 0" if int(LLM_HOST_CACHE_MB or 0) > 0 else NO_HOST_CACHE
    extra = opt(f"{NO_MMAP} {cache} {flags} --threads 1 --image-max-tokens {LLM_IMAGE_MAX_TOKENS} --reasoning-budget 0")   # no thinking: Marker wants JSON
    start(f"llm{g}", f"CUDA_VISIBLE_DEVICES={g} {VLM_ENV} {VLM_BIN} -m {model} --mmproj {mmproj} -ngl 99 {extra} "
                     f"--host 127.0.0.1 --port {port} --parallel {slots} --ctx-size {slots * LLM_CTX_PER_SLOT} --alias {LLM_NAME} --jinja")
    t0 = time.time()
    while time.time() - t0 < 1200 and alive(f"llm{g}"):
        if wait_http(f"http://127.0.0.1:{port}/health", timeout=6, every=3):
            free = int(subprocess.run(["nvidia-smi", f"--id={g}", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                      capture_output=True, text=True).stdout.strip() or 0)
            if free < LLM_MIN_FREE_MB:                # it runs, but Surya on the same GPU could run out of memory under load:
                print(f"   GPU {g}: {slots} slots leave {free} MiB GPU memory (< LLM_MIN_FREE_MB={LLM_MIN_FREE_MB}): trying fewer")
                stop(f"llm{g}"); return None          # (checked before the speed test: no time lost on a variant that can't stay)
            try:
                tps = llm_test(port)
                print(f"   GPU {g}: {LLM_NAME} {os.path.basename(model)} ok | {slots} slots, {tps:.1f} tok/s warm, {free} MiB free | {flags or 'plain'}")
                LLM_RUN[g] = (model, mmproj, flags, slots); return port
            except Exception as e:
                print(f"   GPU {g}: test failed: {str(e)[:200]}"); break
    print(f"   GPU {g}: not usable with [{flags or 'plain'}], {slots} slots:\n" + log_tail(f"llm{g}", 6)); stop(f"llm{g}")
    return None
def setup_llm(g):
    """best file x fastest variant that answers exactly and fast enough, on GPU g"""
    for f in LLM_FILES:
        try:
            model, mmproj = (hf_hub_download(LLM_REPO, x) for x in (f, LLM_MMPROJ))
        except Exception as e:
            print(f"   {f}: download failed:", str(e)[:300]); continue
        for flags, slots in LLM_VARIANTS:
            port = start_llm(g, model, mmproj, flags, slots)
            if port: return port
    return None
if LLM_LOCAL and VLM_BIN and gpus:
    from huggingface_hub import hf_hub_download
    t0 = time.time()
    th = [threading.Thread(target=lambda g=g: LLM_SERVERS.__setitem__(g, setup_llm(g))) for g in range(len(gpus))]
    for t in th: t.start()
    for t in th: t.join()
    LLM_SERVERS = {g: p for g, p in LLM_SERVERS.items() if p}
    if LLM_SERVERS:
        start("lb_llm", f"{sys.executable} mdbox/lb.py {LLM_LB_PORT} {' '.join(map(str, LLM_SERVERS.values()))}")
        if wait_http(f"http://127.0.0.1:{LLM_LB_PORT}/health", timeout=60): LLM_URL = f"http://127.0.0.1:{LLM_LB_PORT}/v1"
    print(f"vision LLM {LLM_NAME}: GPUs {sorted(LLM_SERVERS)} ready in {time.time()-t0:.0f}s" if LLM_URL
          else "!!! vision LLM not running: the LLM passes only work with a GOOGLE_API_KEY secret")

sys.path.insert(0, str(Path("mdbox").resolve()))
def api_env():
    env = os.environ.copy()
    defaults = dict(JOB_DEFAULTS, VLM_URL=VLM_URL, VLM_SLOTS=sum(SLOTS.values()) if VLM_URL else 0,
                    LLM_URL=LLM_URL, LLM_MODEL=LLM_NAME if LLM_URL else "", LLM_SLOTS=sum(r[3] for r in LLM_RUN.values()) if LLM_URL else 0, LLM_CTX=LLM_CTX_PER_SLOT)
    env.update(BOX_TOKEN=BOX_TOKEN, BOX_DEADLINE=str(DEADLINE), MD_ROOT=str(Path(RESULTS_DIR).resolve()), MD_MAX_UPLOAD_GB=str(MAX_UPLOAD_GB),
               MD_DEFAULTS=json.dumps(defaults), PYTHONPATH=str(Path("mdbox").resolve()))
    return env
def start_api():
    start("api", f"{sys.executable} -m uvicorn server:app --host 127.0.0.1 --port {API_PORT} --log-level warning", env=api_env(), cwd=str(Path("mdbox").resolve()))
    if not wait_http(f"http://127.0.0.1:{API_PORT}/ping", timeout=90): raise RuntimeError("API did not start:\n" + log_tail("api"))
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/health", timeout=5); raise RuntimeError("API answered without a token!")
    except urllib.error.HTTPError as e: assert e.code == 401, e.code
def api_get(path):
    req = urllib.request.Request(f"http://127.0.0.1:{API_PORT}{path}", headers={"Authorization": f"Bearer {BOX_TOKEN}"})
    return json.loads(urllib.request.urlopen(req, timeout=10).read())
start_api()
t0 = time.time()
while time.time() - t0 < 180:
    w = api_get("/health")["workers"]
    if w and all(v == "ready" for v in w.values()): break
    time.sleep(3)
print("API up | token required | workers:", w)
