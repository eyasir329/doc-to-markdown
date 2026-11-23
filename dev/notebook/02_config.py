BOX_NAME      = "markdown"
SESSION_HOURS = 8.5         # hard stop (Kaggle kills sessions at 12 h; GPU quota is 30 h/week)
IDLE_STOP_MIN = 60          # stop when no job has been active for this long, to save GPU quota (0 = never)
API_PORT      = 8000
ANNOUNCE      = True        # post the temporary tunnel address (signed, no secrets) to a private ntfy.sh topic so your PC finds it
MAX_UPLOAD_GB = 3
RESULTS_DIR   = "/kaggle/tmp/jobs"   # "jobs" would keep every result in the saved notebook output

# ---- conversion engine
MARKER_SOURCE = "git+https://github.com/datalab-to/marker.git@b578dc88412984b7531bd4a5b4886eb2d2c58da1"   # your PC must use the same version
OFFICE_VIA_LIBREOFFICE = True   # LibreOffice: .doc/.ppt/.xls/.odt/.rtf... and .docx/.pptx with their equations and layouts

# ---- GPU programs (llama.cpp)
LLAMA_PREBUILT = True       # download llama.cpp's released CUDA build (~30 s) instead of compiling it (~15 min)
LLAMA_CPP_REF = "master"    # llama.cpp release / git ref (pin one if a new release ever breaks)
USE_VLM       = True        # Surya VLM on the GPUs; False = CPU-only (fast / no_ocr modes)
VLM_SLOTS_PER_GPU = 8       # parallel Surya requests per GPU (halved automatically if a GPU runs out of memory)
VLM_LB_PORT   = 8090        # load balancer in front of the Surya servers

# ---- vision LLM for Marker's quality passes (tables, equations, forms, headings) and figure content
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
    MODE="auto",            # auto = balanced when Surya runs, else fast (client: --mode)
    OFFICE="pdf",           # pdf = LibreOffice renders office files to PDF first; native = Marker reads .docx/.pptx itself (--office)
    USE_LLM=True,           # the LLM passes on every job when an LLM runs (client: --no-llm = faster)
    REDO_INLINE_MATH=None,  # None = as the quality says (accurate: yes); True/False forces Marker's inline-maths rewrite
    QUALITY="balanced",     # fast / balanced / accurate (client: --quality). fast: no page review, light figure text, no Marker LLM steps;
                            # balanced: + Marker's LLM steps, page review, figures cropped to themselves, flow diagrams as Mermaid;
                            # accurate: + review section by section, line-by-line audit with every fix double-checked (slowest)
    DESCRIBE_FIGURES=None,  # None = as the quality says; True/False forces it (client: --no-describe)
    PAGE_REVIEW=None,       # None = as the quality says; True/False forces it (client: --no-review)
    SPLIT_MIN_PAGES=8,      # PDFs with at least this many pages are split into page chunks shared by all workers (and your PC); 0 = never
    CHUNK_MAX_PAGES=24,     # pages per chunk at most: smaller chunks = less RAM per worker
    DOC_TIMEOUT_MIN=90,     # a document (or chunk) that takes longer is stopped and reported as failed
    TIME_BUDGET_H=7.0,
)
