BOX_NAME      = "markdown"
SESSION_HOURS = 8.5         # hard stop (Kaggle kills sessions at 12 h; GPU quota is 30 h/week)
IDLE_STOP_MIN = 60          # stop when no job has been active for this long, to save GPU quota (0 = never)
API_PORT      = 8000
ANNOUNCE      = True        # post the temporary tunnel address (signed, no secrets) to a private ntfy.sh topic so your PC finds it
MAX_UPLOAD_GB = 3
RESULTS_DIR   = "/kaggle/tmp/jobs"   # "jobs" would keep every result in the saved notebook output
EXPRESS_BOX   = False       # True = an express-only box: the vision LLM is not started, its GPU memory goes to Surya
                            # (EXPRESS_VLM_SLOTS per GPU), more workers run (EXPRESS_WORKERS_PER_CPU) and jobs default to express.
                            # Measured: NOT faster for express jobs (Surya's speed is the limit, an idle vision LLM costs it nothing);
                            # it only starts a few minutes sooner and uses less GPU memory. Keep False to have all three qualities
EXPRESS_VLM_SLOTS = 16          # 3-copy load test (81 pages): 16 slots and 8 workers 387 pages/h; 24 slots and 12 workers
EXPRESS_WORKERS_PER_CPU = 2.0   #   334 pages/h (more requests in flight, GPUs no busier: images are encoded one at a time)
EXPRESS_MEM_PER_WORKER_GB = 3.0
EXPRESS_LB_MAX_INFLIGHT = 0     # tuned (dev/tune_vlm_cap.py, 54 pages): no cap 346 pages/h; caps 4/6/8/12: 305/278/318/315

# ---- conversion engine
MARKER_SOURCE = "git+https://github.com/datalab-to/marker.git@b578dc88412984b7531bd4a5b4886eb2d2c58da1"   # your PC must use the same version
OFFICE_VIA_LIBREOFFICE = True   # LibreOffice: .doc/.ppt/.xls/.odt/.rtf... and .docx/.pptx with their equations and layouts

# ---- GPU programs (llama.cpp)
LLAMA_PREBUILT = True       # download llama.cpp's released CUDA build (~30 s) instead of compiling it (~15 min)
LLAMA_CPP_REF = "b12627"    # llama.cpp release the box was measured with (its --help: dev/fixtures). "master" = the newest
                            # release: llama.cpp ships daily and renames flags (--no-mmap -> --load-mode none broke every server once)
USE_VLM       = True        # Surya VLM on the GPUs; False = CPU-only (fast / no_ocr modes)
VLM_SLOTS_PER_GPU = 8       # parallel Surya requests per GPU (halved automatically if a GPU runs out of memory)
VLM_LB_PORT   = 8090        # load balancer in front of the Surya servers
VLM_FLAG_VARIANTS = ["-fa on -ub 1024", ""]   # Surya's llama-server settings, fastest first; the next is tried if a server won't
                            # start. Flash attention is exact (same output, faster with long contexts; the vision LLM already uses
                            # it on these T4s); -ub 1024: a ~1500-token page image is processed in 2 steps, not 3. KV-cache
                            # quantization is not used for Surya (slightly lossy)
VLM_LB_MAX_INFLIGHT = 0     # Surya requests in flight per GPU at most (0 = as many as the slots take); also settable on a running
                            # box: POST /vlm/lb-config {"max_inflight": n} (dev/tune_vlm_cap.py measures which is fastest)

# ---- vision LLM: page triage, checks, audit and rewrites, figure descriptions (standard and precise)
LLM_LOCAL     = True        # False = only Gemini (GOOGLE_API_KEY secret), or no LLM passes
LLM_NAME      = "qwen3-vl-8b"
LLM_REPO      = "unsloth/Qwen3-VL-8B-Instruct-GGUF"   # one server per T4, next to Surya
LLM_FILES     = ["Qwen3-VL-8B-Instruct-Q6_K.gguf", "Qwen3-VL-8B-Instruct-Q4_K_M.gguf"]   # best first; Q4 only if Q6 cannot start
LLM_MMPROJ    = "mmproj-F16.gguf"   # vision encoder (T4s have no bf16)
_FAST  = "-fa on -ctk q8_0 -ctv q8_0"            # flash attention + 8-bit KV cache: half the context memory, more slots
_NGRAM = "--spec-type ngram-mod"                 # speculative decoding from the prompt itself: table/equation fixes mostly copy it
LLM_VARIANTS  = [(f"{_FAST} {_NGRAM}", 5), (f"{_FAST} {_NGRAM}", 4), (_FAST, 4), ("", 2)]   # (llama-server flags, parallel slots), fastest first:
                            # more slots = more requests decoded together (more throughput), if the T4 still has room (LLM_MIN_FREE_MB).
                            # Measured on a T4 next to Surya (8 slots): 4 slots leave 1731 MiB, 6 leave 431 (~650 MiB a slot) -> 5 leave ~1080
LLM_MIN_FREE_MB = 700       # a variant must leave this much GPU memory free (Surya shares the GPU), else the next one is tried
LLM_HOST_CACHE_MB = 2048    # host RAM each vision-LLM server may use to keep processed prompts (page images) a slot gave up: a page's
                            # review, audit and second looks reuse its image (2.8 s of GPU time saved per call). Bounded: the default
                            # (8 GB per server) once filled the box. 0 = off. Surya's servers keep none (their crops never repeat)
LLM_MIN_TOK_S = 5.0         # a variant must answer the test exactly and generate at least this fast once warm
LLM_CTX_PER_SLOT     = 8192
LLM_IMAGE_MAX_TOKENS = 1536 # ~1250x1250 px per image: enough for table/equation crops, keeps requests fast
LLM_LB_PORT   = 8190        # load balancer in front of the two LLM servers

# ---- defaults for every job (the client can override most of them per run)
JOB_DEFAULTS = dict(
    WORKERS=0,              # conversion processes; 0 = up to WORKERS_PER_CPU per vCPU, as many as the free RAM holds (MEM_PER_WORKER_GB each)
    WORKERS_PER_CPU=1.5,    # a worker waiting on the GPU models leaves its CPU free: with spare RAM, 4 vCPUs run 6 workers
    MEM_START_GB=2.5,       # RAM a piece grows by in its first minute: new pieces start only while free RAM covers them above MEM_PAUSE_FREE_GB
    MEM_PER_WORKER_GB=3.0,  # RAM one Marker worker may need (measured ~2 GB at its peak in the suite runs; 3.5 left 1 worker unused)
    MEM_PAUSE_FREE_GB=4.0,  # less free RAM than this: no new pieces, except one if nothing is running (the session restarts if RAM is full)
    MEM_RESUME_FREE_GB=6.0, # ... more than this: all workers again
    MEM_GPU_SERVER_MAX_GB=5.0,  # ... and a GPU server (llama-server) bigger than this is restarted instead (the watchdog brings it back);
                                #     an LLM server is ~2.3 GB + up to LLM_HOST_CACHE_MB of prompt cache
    MEM_KILL_FREE_GB=2.0,   # ... less than this: the biggest busy worker is stopped and its piece converted again (checked every 0.5 s)
    MODE="auto",            # Marker's reading mode, not the quality: auto = Marker's balanced when Surya runs, else fast (--mode)
    OFFICE="pdf",           # pdf = LibreOffice renders office files to PDF first; native = Marker reads .docx/.pptx itself (--office)
    USE_LLM=True,           # the LLM passes on every job when an LLM runs (client: --no-llm = faster)
    REDO_INLINE_MATH=None,  # None = off (no quality uses Marker's own LLM steps); True = those steps with its inline-maths rewrite (--redo-math)
    QUALITY="standard",     # express / standard / precise (client: --quality; old names still work). express: no vision-LLM
                            # pass; standard: the vision LLM only where it pays (a page its own text settles gets no call, the
                            # others a short look, then an audit, the element check, or also a finer audit for hard pages;
                            # figures with diagrams/charts described); precise: every pass on every page (element check,
                            # line-by-line rewrite, fine audit, prose made explicit, every figure described). No step runs twice
    DESCRIBE_FIGURES=None,  # None = as the quality says; True/False forces it (client: --no-describe)
    PAGE_REVIEW=None,       # None = as the quality says (no quality uses the whole-page review); True forces it. --no-review turns every page pass off
    SPLIT_MIN_PAGES=8,      # PDFs with at least this many pages are split into page chunks shared by all workers (and your PC); 0 = never
    CHUNK_MAX_PAGES=24,     # pages per chunk at most: smaller chunks = less RAM per worker
    DOC_TIMEOUT_MIN=90,     # a document (or chunk) that takes longer is stopped and reported as failed
    TIME_BUDGET_H=7.0,
)
if EXPRESS_BOX:             # no Qwen: Surya gets the GPUs, the workers keep more of its requests in flight
    LLM_LOCAL, VLM_SLOTS_PER_GPU, VLM_LB_MAX_INFLIGHT = False, EXPRESS_VLM_SLOTS, EXPRESS_LB_MAX_INFLIGHT
    JOB_DEFAULTS.update(QUALITY="express", USE_LLM=False, WORKERS_PER_CPU=EXPRESS_WORKERS_PER_CPU,
                        MEM_PER_WORKER_GB=EXPRESS_MEM_PER_WORKER_GB)
    # (3-page chunks so all 8 workers get one of a 27-page PDF were measured slower: 375 s vs 283 s, Marker's time per chunk
    #  is mostly fixed; long documents fill all workers anyway)
