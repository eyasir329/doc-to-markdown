# Architecture

The system has two halves:

- **The box:** a Kaggle notebook that serves a token-protected HTTP API through a Cloudflare quick tunnel.
- **The client:** `md_client.py` on your PC, which uploads documents, follows progress and downloads results.

```
your PC                                Cloudflare quick tunnel             Kaggle notebook (2 × T4, 4 vCPUs, ~30 GB RAM)
md_client.py ./docs ─────────────────▶ https://xxxx.trycloudflare.com ──▶ server.py  (FastAPI, Bearer token)
  one queue, biggest files first                                            │  jobs: up to 8 at once
  ├─ box: uploads in 16 MB parts                                            ▼
  └─ PC workers (--local, optional):            pipeline.py: scans set upright, two-page photos cut in two
     the smallest files, and page chunks                                    │ documents ─▶ page chunks ─▶ task queue
     stolen from the box's long PDFs ◀── POST /jobs/{id}/steal ──────────────┼────────────────────────────────┘
                                                                            ▼
                                          warm workers (worker.py, up to 6): Marker + repairs + vision-LLM passes
                                                  │ OCR requests                    │ triage / checks / audit / figures
                                                  ▼                                 ▼
                                          lb.py :8090                        lb.py :8190 (image affinity)
                                    Surya 2.5 llama-server ×2           Qwen3-VL-8B llama-server ×2
                                    (8 slots per T4)                    (5 slots per T4)
        ◀── GET /jobs/{id}/bundle.zip  (<name>/<name>.md + images + report.json) ──
```

## Repository layout

| Path | Role |
|---|---|
| `md_client.py` | The client. Python standard library only |
| `docs_to_markdown_cloudflare.ipynb` | The notebook you run on Kaggle. **Generated** by `dev/build_nb.py`; never edit it by hand |
| `dev/notebook/NN_*.md\|py` | The notebook's cells, one file each, in file-name order. `02_config.py` holds every setting |
| `dev/mdbox/*.py` | The box's modules. The notebook writes them to disk with `%%writefile` cells |
| `dev/build_nb.py` | Assembles the notebook from the two folders above |
| `dev/test_pipeline.py`, `dev/test_e2e.py`, `dev/modes_check.py` | Unit tests, an end-to-end test and a per-mode request count; none needs a GPU (see [Development](development.md)) |
| `dev/fakes/` | Fake `marker` and `pypdfium2` packages for the end-to-end tests |
| `dev/fixtures/` | Captured tool output used by the tests (llama-server's `--help`) |
| `dev/compare_quality.py`, `dev/bench_run.py`, `dev/bench_score.py`, `dev/regress_check.py` | Measuring the quality modes on a running box |
| `dev/load_test.py`, `dev/tune_vlm_cap.py` | Throughput tests and tuning on a running box |

## The box's modules

| Module | Responsibility |
|---|---|
| `server.py` | The HTTP API. Bearer-token auth (constant-time comparison), validated options (400 for bad input), uploads in parts with checksums, the job queue (8 runners), status, downloads, `/vlm` and `/llm` proxies for PC workers, chunk stealing. A janitor cancels jobs and uploads no client has asked about, and removes stored files no job uses |
| `pipeline.py` | One job: unpack (with zip-bomb limits), scans set upright and two-page photos cut (one job at a time), one task per document, long PDFs split into page chunks, the worker pool, chunks joined back, `report.json`, the zip. Also the memory guard, the crash budget and the resource sampler |
| `worker.py` | A warm conversion process with Marker's models loaded. It converts one piece (a document or a page chunk) and runs every later stage on it |
| `spreads.py` | Page orientation and two-page photo detection with Indra (CPU): fold finding, cutting, putting the halves in printed order |
| `convert.py` | Chooses the path for each file type. Runs LibreOffice and `ddjvu` and converts text and CSV directly. Also the text-layer and Indic repairs and the joining of page chunks |
| `figures.py` | Figure re-cropping, recovery of missed images, the crop to the figure itself, Surya's reading of pictures, descriptions, flowcharts → Mermaid, table checks against the image, drawn formulas, the coverage check, text lines the layout left out, the layout listing (`--show-layout`) and block labels |
| `review.py` | The quality profiles and page levels, the triage, the element check, the line-by-line rewrite, the audit, the prose rewrite, the loss checks, the Markdown tidy-up, the LLM in-flight gate |
| `lb.py` | The load balancer in front of the two GPU servers of each model: least busy, with image affinity; a server that answers 503 (still loading) is skipped for a few seconds |

## One document's path

1. **Routing** (`convert.route`):
   - `.txt`, `.md` and `.csv` become Markdown directly.
   - Office files are rendered to PDF by LibreOffice. Spreadsheets become one table per sheet.
   - DjVu goes to PDF with `ddjvu`.
   - PDFs, images, HTML and EPUB go to Marker.
2. **Scans first** (PDF pages without a text layer): each page is set upright, and a photo of two facing pages is
   cut into its two pages (`spreads.py`).
3. **Splitting.** A PDF of `SPLIT_MIN_PAGES` (8) pages or more is cut into chunks of 4–24 pages. The number of
   chunks is a multiple of the number of workers, scan-heavy stretches get fewer pages, and the heaviest chunks are
   queued first. A spread's two halves always stay in one chunk.
4. **Marker** (in a worker) finds the layout and reads the text. It uses the PDF's text where that is good;
   otherwise it sends OCR requests to Surya on the GPUs. Text lines its layout left in no block are read too.
5. **Figures:** re-cropped from the page at 300 DPI, missed images recovered, drawn formulas turned into
   pictures.
6. **Repairs from the page's own text:** text layer, Indic scripts.
7. **Vision-LLM passes** (`standard`, `precise`): each page's triage, element check, line-by-line rewrite and
   audit as its mode and level say, many pages at once; then Surya reads every picture and the vision LLM describes
   the figures by their meaning (flowcharts also as Mermaid). In `express`, Surya's reading goes under each picture.
8. **Final repairs:** the text-layer repair again (if a model changed text), tables against the image, heading
   levels from the PDF outline, a spread's halves put in printed order, the Markdown tidy-up.
9. **Join and report:** chunks are joined in page order, the coverage check runs, `report.json` is written and
   everything is zipped for download.

[How accuracy is protected](accuracy.md) explains steps 2 and 4–8 in detail.

## GPU servers

Each T4 runs two [llama.cpp](https://github.com/ggml-org/llama.cpp) servers side by side, from a pinned release
(`LLAMA_CPP_REF`):

| Server | Model | Slots per T4 | Used for |
|---|---|---|---|
| Surya | Surya 2.5 (GGUF, from Marker) | 8 | OCR, layout and tables inside Marker; the text inside pictures |
| Vision LLM | Qwen3-VL-8B-Instruct Q6_K + F16 vision encoder | 5 | triage, element check, rewrite, audit, prose, figures |

- **Startup test.** Each server is used only if it passes a real request: the LLM must read "42" from a test
  image exactly, twice, and generate at least 5 tokens/s once warm. A page-sized warm-up compiles the GPU kernels
  before the first job.
- **Fallbacks.** Settings are tried fastest first (`LLM_VARIANTS`, `VLM_FLAG_VARIANTS`). The vision LLM's first
  try is flash attention, 8-bit KV cache and n-gram speculative decoding with 5 slots; then fewer slots; then plain
  settings. A variant must leave 700 MB of GPU memory for Surya. The Q4_K_M model file is used only if Q6_K can't
  start.
- **No memory-mapping.** Model files go straight to the GPU (`--load-mode none` / `--no-mmap`), so they don't
  sit in system RAM.
- **Optional flags.** The notebook reads `llama-server --help` and leaves out any optional flag the downloaded
  build doesn't know.
- **Image affinity.** A page image costs about 1,500 prompt tokens. `lb.py` sends a request with an image to the
  server that last had that image, unless that server is busier by `LB_SPILL` (3) requests. A page's calls run one
  after the other to reuse it, and the vision-LLM servers keep a bounded prompt cache of `LLM_HOST_CACHE_MB` each.
- **In-flight cap.** Each worker may have at most as many LLM requests in flight as the box has LLM slots (10).
  One worker can still fill every slot, but nested pools can't queue hundreds of requests that would time out.
- **Watchdog.** A GPU server that dies (or that the memory guard stopped) is restarted by the notebook, and the
  load balancer sends requests to the other GPU meanwhile.

## Scheduling and memory safety

Kaggle restarts the whole notebook if its RAM fills, so the box guards memory carefully:

- **Worker count.** Up to `WORKERS_PER_CPU` (1.5) workers per vCPU: a worker waiting for the GPUs leaves its CPU
  free. The count is also limited by the session's real RAM limit (its cgroup), at `MEM_PER_WORKER_GB` (3 GB)
  each with `MEM_RESERVE_GB` (3 GB) kept free. On Kaggle that is 6 workers.
- **Start budget.** A new piece grows by about `MEM_START_GB` (2.5 GB) in its first minute. The pool publishes how
  many pieces may start now: (free RAM − 4 GB) ÷ 2.5 GB. Each worker must claim an exclusive start slot, which
  counts for `START_WINDOW_S` (60 s). So several workers can't all start on the same free RAM.
- **Guard thread** (every 0.5 s):
  - below 4 GB free, no new pieces start (one may still start if nothing is running, so the box can't deadlock);
  - above 6 GB free, all workers resume;
  - below 2 GB free, whatever grew is stopped: an idle worker that kept a past piece's memory is restarted; a GPU
    server over 5 GB is restarted by the watchdog; otherwise the biggest busy worker is stopped and its piece
    converted again.
- **Kernel's first choice.** Workers set `oom_score_adj` to 1000, so if RAM fills anyway the kernel stops a worker
  (its piece is requeued), not the notebook or a GPU server.
- **Workers give memory back** after each piece (`malloc_trim`) and restart themselves if they stay above 3 GB, or
  when they sit idle well above their loaded size.
- **One job at a time prepares scans.** Turning pages upright and cutting two-page photos starts two Indra
  processes per job; only one job does it at a time.
- **Resource sampling.** Every job samples GPU busy %, GPU memory, CPU % and the lowest free RAM every 2 s, with
  the biggest processes at that lowest point. The results go to `report.json` as `box_resources`.

## Network, discovery and failure handling

- **Tunnel.** A Cloudflare quick tunnel gives the box a temporary public HTTPS address. Only the API is exposed,
  and every endpoint except `GET /ping` needs the token. The free tunnel rejects bodies over ~100 MB and cuts
  requests after ~100 s, so uploads go in 16 MB parts and the client polls for status.
- **Discovery.** The box posts its address to an [ntfy.sh](https://ntfy.sh) topic named after a hash of your
  token, signed with an HMAC of your token. The client reads that topic, checks the signature and uses the newest
  address. Without the token, nobody can find the topic, read the address or forge it.
- **Tunnel loss.** If the tunnel drops (HTTP 530), the watchdog restarts it and announces the new address. The
  client retries for up to 10 minutes, follows the new address and carries on with the same job.
- **Failures stay local.** A damaged file, a timeout (`DOC_TIMEOUT_MIN`, 90 min) or a crashed worker fails only
  that document or piece. Crashed workers and GPU servers are restarted.
- **Crash budget.** A worker stopped on purpose (a cancel, a timeout, the memory guard) is relaunched at once.
  Unplanned crashes are counted over 10 minutes (3 per worker + 3): a crash loop parks the workers until the window
  clears, so the box recovers by itself instead of failing every later job.
- **Abandoned work.** A job, or an upload, that no client has asked about for `MD_ABANDON_MIN` (15) minutes is
  cancelled, so a killed client can't hold the workers or keep the box from idling out. Stored files no job uses
  are removed after the same time. The client also cancels its own jobs on Ctrl+C, SIGTERM and SIGHUP.
- **Session limits.** The box stops after `IDLE_STOP_MIN` without jobs, and always at `SESSION_HOURS`. Each job
  gets a deadline that ends before the session does.

## Your PC as a helper

With Marker installed (`--local`), the client runs the same `worker.py` as the box. It uses the box's GPUs through
the tunnel (`/vlm` and `/llm` proxies) and must use the same Marker version and mode, so results don't depend on
where a page was converted. The client keeps one queue, biggest files first:

- **The box** gets the next file from the front when it has no unclaimed tasks.
- **Your PC** takes files from the back, the smallest ones.
- **Stealing.** When your PC runs out of work, it steals not-yet-started chunks from the end of the box's long
  PDFs, as soon as the box has queued them (not from documents whose pages the box turned or cut: the page
  numbers would differ).
- **Failures.** A file that fails on your PC goes to the box. After 3 failures in a row, everything goes to the
  box.
