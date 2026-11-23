# Configuration

Settings live in three places:

- **Notebook settings** (section 1 of the notebook, from `dev/notebook/02_config.py`): the box, its models and
  the default for every job.
- **Per-job options** (the client's flags, or the API's `options`): override the box defaults for one job.
- **Environment variables**: low-level knobs of the box's modules, and the client's token and address.

The defaults are tuned and measured on Kaggle's 2×T4 machine (4 vCPUs, ~30 GB RAM). Change them only for a
reason.

## Notebook settings

### Session

| Setting | Default | Meaning |
|---|---|---|
| `BOX_NAME` | `"markdown"` | name shown in the banner and the announcement |
| `SESSION_HOURS` | `8.5` | hard stop. Kaggle kills sessions at 12 h, and the GPU quota is 30 h a week |
| `IDLE_STOP_MIN` | `60` | stop after this many minutes without an active job (`0` = never) |
| `API_PORT` | `8000` | local port of the API behind the tunnel |
| `ANNOUNCE` | `True` | post the signed tunnel address to a private ntfy.sh topic so the client finds it. With `False`, use `--url` |
| `MAX_UPLOAD_GB` | `3` | largest upload |
| `RESULTS_DIR` | `"/kaggle/tmp/jobs"` | where results are written. `/kaggle/tmp` isn't saved with the notebook version, so your documents don't end up in the notebook's output |

### Conversion engine

| Setting | Default | Meaning |
|---|---|---|
| `MARKER_SOURCE` | Marker at a pinned commit | the Marker version. Local PC workers must use the same one (`requirements-local.txt`) |
| `OFFICE_VIA_LIBREOFFICE` | `True` | install LibreOffice so office files can be rendered to PDF |

### GPU programs

| Setting | Default | Meaning |
|---|---|---|
| `LLAMA_PREBUILT` | `True` | download llama.cpp's released CUDA build (~30 s) instead of compiling it (~15 min) |
| `LLAMA_CPP_REF` | `"master"` | `master` takes the newest release that has the CUDA build. Set a release tag (e.g. `"b12627"`) to pin one; useful if a new release ever breaks something |
| `USE_VLM` | `True` | run Surya on the GPUs. `False` = CPU only (`fast` / `no_ocr` Marker modes) |
| `VLM_SLOTS_PER_GPU` | `8` | parallel Surya requests per GPU (halved automatically if a GPU runs out of memory) |
| `VLM_LB_PORT` | `8090` | load balancer in front of the Surya servers |

### Vision LLM

| Setting | Default | Meaning |
|---|---|---|
| `LLM_LOCAL` | `True` | run the vision LLM on the GPUs. `False` = only Gemini (with a `GOOGLE_API_KEY` secret), or no LLM passes |
| `LLM_NAME` | `"qwen3-vl-8b"` | the model's name in the API |
| `LLM_REPO` | `"unsloth/Qwen3-VL-8B-Instruct-GGUF"` | Hugging Face repository of the GGUF files |
| `LLM_FILES` | `[Q6_K, Q4_K_M]` | model files, best first. Q4 is used only if Q6 can't start |
| `LLM_MMPROJ` | `"mmproj-F16.gguf"` | the vision encoder (T4s have no bf16) |
| `LLM_VARIANTS` | 5 slots fast → … → 2 slots plain | (llama-server flags, slots) tried fastest first. A variant must pass the startup test and leave `LLM_MIN_FREE_MB` free |
| `LLM_MIN_FREE_MB` | `700` | GPU memory a variant must leave free for Surya on the same GPU |
| `LLM_HOST_CACHE_MB` | `2048` | host RAM each LLM server may use to keep processed page images (bounded; `0` = off) |
| `LLM_MIN_TOK_S` | `5.0` | a variant must generate at least this fast once warm |
| `LLM_CTX_PER_SLOT` | `8192` | context per slot |
| `LLM_IMAGE_MAX_TOKENS` | `1536` | about 1250 × 1250 px per image |
| `LLM_LB_PORT` | `8190` | load balancer in front of the LLM servers |

### Job defaults (`JOB_DEFAULTS`)

These apply to every job unless the job sets them. You can add any key of `pipeline.DEFAULTS` (listed under
[Advanced defaults](#advanced-defaults)) to `JOB_DEFAULTS` as well.

| Key | Default | Meaning |
|---|---|---|
| `QUALITY` | `"balanced"` | `fast`, `balanced` or `accurate`; see [Quality modes](quality-modes.md) |
| `MODE` | `"auto"` | Marker's mode. `auto` = `balanced` when Surya runs, else `fast` |
| `OFFICE` | `"pdf"` | `pdf` = LibreOffice renders office files first; `native` = Marker reads `.docx`/`.pptx` itself |
| `USE_LLM` | `True` | the LLM passes, when an LLM runs |
| `REDO_INLINE_MATH`, `DESCRIBE_FIGURES`, `PAGE_REVIEW` | `None` | `None` = as the quality says; `True`/`False` forces the stage |
| `WORKERS` | `0` | conversion processes. `0` = up to `WORKERS_PER_CPU` per vCPU, as many as the free RAM holds |
| `WORKERS_PER_CPU` | `1.5` | workers per vCPU at most |
| `MEM_PER_WORKER_GB` | `3.0` | RAM one worker may need (measured: ~2 GB at its peak) |
| `MEM_START_GB` | `2.5` | RAM a piece grows by in its first minute. New pieces start only while free RAM covers them |
| `MEM_PAUSE_FREE_GB` / `MEM_RESUME_FREE_GB` | `4.0` / `6.0` | free RAM below which no new piece starts, and above which all workers resume |
| `MEM_KILL_FREE_GB` | `2.0` | below this, the biggest busy worker is stopped and its piece converted again |
| `MEM_GPU_SERVER_MAX_GB` | `5.0` | a GPU server bigger than this is restarted instead (when RAM runs out) |
| `SPLIT_MIN_PAGES` | `8` | PDFs with at least this many pages are split into chunks. `0` = never split |
| `CHUNK_MAX_PAGES` | `24` | pages per chunk at most. Smaller chunks need less RAM per worker |
| `DOC_TIMEOUT_MIN` | `90` | a document or chunk that takes longer fails |
| `TIME_BUDGET_H` | `7.0` | a job's total time limit (it also always ends 2 minutes before the session) |

### Advanced defaults

Built into `pipeline.DEFAULTS` and not listed in the notebook; add them to `JOB_DEFAULTS` to change them:

| Key | Default | Meaning |
|---|---|---|
| `CHUNK_MIN_PAGES` | `4` | smallest chunk |
| `MEM_RESERVE_GB` | `3.0` | RAM kept free for the API, tunnel and page cache when sizing the pool |
| `WORKER_RECYCLE_GB` | `3.0` | a worker bigger than this after a piece restarts itself |
| `START_WINDOW_S` | `60` | how long a piece's start counts against the free RAM |
| `MAX_ZIP_FILES` / `MAX_UNZIPPED_GB` | `2000` / `6.0` | limits for uploaded zip archives |
| `CPU_THREADS`, `VLM_PARALLEL` | `0` | threads per worker / VLM requests per worker. `0` = derived from the CPUs and slots |
| `LLM_CTX` | `8192` | context the LLM passes plan for |

## Per-job options

The client sets these with flags ([Usage](usage.md#options)); other clients send them in the API's `options`
([HTTP API](api.md)). Unknown keys, values of the wrong type and invalid choices are refused with HTTP 400.

| Key | Type | Client flag |
|---|---|---|
| `QUALITY` | `fast` / `balanced` / `accurate` | `--quality` |
| `MODE` | `auto` / `balanced` / `vllm` / `fast` / `no_ocr` | `--mode` |
| `OFFICE` | `pdf` / `native` | `--office` |
| `OUTPUT_FORMAT` | `markdown` / `html` / `json` / `chunks` | `--format` |
| `PAGE_RANGE` | e.g. `"0,5-10"` | `--pages` |
| `USE_LLM` | bool | `--llm` / `--no-llm` |
| `PAGE_REVIEW` | bool | `--no-review` |
| `DESCRIBE_FIGURES` | bool | `--no-describe` |
| `REDO_INLINE_MATH` | bool | `--redo-math` |
| `AUDIT` | bool | (API only) |
| `FORCE_OCR`, `STRIP_EXISTING_OCR` | bool | `--force-ocr`, `--strip-ocr` |
| `PAGINATE`, `KEEP_HEADERS_FOOTERS`, `IMAGES` | bool | `--paginate`, `--keep-headers`, `--no-images` |
| `FIGURE_FIX` | bool | `--raw-figures` (= `false`) |

## Environment variables

### Box (set by the notebook)

| Variable | Default | Meaning |
|---|---|---|
| `BOX_TOKEN` | required | the token (from the Kaggle secret). At least 16 characters |
| `BOX_DEADLINE` | now + 10 h | Unix time the session ends |
| `MD_ROOT` | `jobs` | where job results are written (`RESULTS_DIR`) |
| `MD_UPLOADS` | temp folder | where uploads are stored while they arrive |
| `MD_MAX_UPLOAD_GB` | `3` | largest upload |
| `MD_RUNNERS` | `8` | jobs that run at the same time (their pieces share one worker queue) |
| `MD_ABANDON_MIN` | `15` | cancel a job no client has asked about for this many minutes (`0` = never) |
| `MD_DEFAULTS` | `{}` | JSON of job defaults (the notebook's `JOB_DEFAULTS` plus the GPU servers' addresses) |
| `MD_POOL` | temp folder | the worker pool's task folder |
| `MD_WORKER_RECYCLE_GB` | `3` | see `WORKER_RECYCLE_GB` |
| `LB_SPILL` | `3` | the load balancer passes over the server that last had an image when it has this many more requests in flight |
| `GOOGLE_API_KEY` | – | Gemini fallback when the local LLM can't start. Sends page content to Google |

### Client

| Variable | Meaning |
|---|---|
| `BOX_TOKEN` | the token, if there's no `.box_token` file and no `--token` |
| `BOX_URL` | the box's address, if the automatic lookup fails (same as `--url`) |
