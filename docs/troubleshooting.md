# Troubleshooting

Start with `python md_client.py --health`. It shows whether the box is reachable, its GPUs, workers, models,
memory, idle time, the qualities it offers and the `code` of its modules. On Kaggle, the notebook's log shows each
section's progress, and a heartbeat line every 5 minutes while it serves.

## Starting the notebook

| Message | Cause and fix |
|---|---|
| `Secret BOX_TOKEN is missing or shorter than 16 chars` | Add the `BOX_TOKEN` secret (Add-ons → Secrets) and tick it for this notebook |
| No GPU / `!!! no VLM server is running: CPU-only (Marker's fast reading mode)` | The accelerator isn't **GPU T4 ×2**, or llama.cpp couldn't be downloaded or built. Check the settings panel; the lines before the message give the reason. The box still works on CPUs only, but slowly |
| `!!! vision LLM not running: …` | The vision LLM failed to start (see the `GPU g:` lines above it). Jobs run with `express`'s steps and say so in their warnings |
| `could not prepare Marker models (internet/HF_TOKEN?)` | Internet is off, or Hugging Face is rate-limiting. Switch **Internet → On**; add an `HF_TOKEN` secret |
| `prebuilt llama.cpp incomplete` / `no ubuntu-cuda-12.8-x64 asset …` | The pinned llama.cpp release (`LLAMA_CPP_REF`) has no prebuilt CUDA build, or its download failed. The notebook falls back to compiling (~15 min) |
| `GPU g: N slots leave X MiB GPU memory (< LLM_MIN_FREE_MB=700): trying fewer` | Normal. A variant with fewer vision-LLM slots is tried next, and the line ending in `ok` shows what was chosen |
| `GPU g: test failed: …` / `not usable with [...]` | The vision LLM failed its startup test with those settings. The next variant is tried. If none passes, the vision LLM is off: check the log tail printed with it |
| `this llama-server has no --flag: left out` | Normal: an optional flag this llama.cpp build doesn't know is skipped |
| `Cloudflare tunnel did not become reachable` | A Cloudflare hiccup. Run the cell again, or restart the notebook |
| `could not announce (…); pass the url below with --url` | ntfy.sh was unreachable from Kaggle. Use `--url` with the banner's address |

The first run takes about 10 minutes (downloads and tests). The first job after start is about a minute slower
while the GPU servers compile their kernels.

## Connecting from your PC

| Message | Cause and fix |
|---|---|
| `no BOX_TOKEN: put it in a file named .box_token …` | Create `.box_token` next to `md_client.py`, or set `BOX_TOKEN`, or pass `--token` |
| `no live announcement from a markdown box (is the Kaggle notebook running?)` | The box isn't running yet (wait for BOX READY), it has stopped, or your token differs from the Kaggle secret. Tokens are compared exactly: check for extra spaces or line breaks |
| `could not reach ntfy.sh (…)` | Your network blocks ntfy.sh. Use `--url` or `BOX_URL` with the banner's address |
| `HTTP 401: unauthorized` | The token doesn't match the box's `BOX_TOKEN` secret |
| `HTTP 400: QUALITY must be one of …` | The box doesn't know that quality name. Use `express`, `standard` or `precise` (a release-1 box only knows `fast`, `balanced`, `accurate`: update its notebook) |
| `the box at … does not answer` | The box stopped (idle or session limit) or restarted with a new address. Check the Kaggle log; run the command again so the client looks up the new address |
| HTTP 530 during a run | Cloudflare lost the tunnel. The client retries for up to 10 minutes and follows the new address, so usually nothing is needed |

## Converting

| Symptom | Cause and fix |
|---|---|
| `note: --quality fast is now called express` | Normal: a release-1 quality name was mapped to its release-2 name (`fast` → `express`: no vision-LLM pass) |
| `quality standard needs the box's vision LLM, which is not running` | The vision LLM didn't start on this box; the job got `express`'s steps. See the notebook's start log |
| Progress says `RAM low (x GB free): one piece at a time` | The memory guard paused new pieces because free RAM fell below 4 GB. It resumes by itself. In `standard` and `precise` this is mostly the vision-LLM servers' prompt caches; see [Performance → RAM](performance.md#where-the-time-goes). Lower `LLM_HOST_CACHE_MB` or `WORKERS_PER_CPU` if it slows your runs |
| `[file] FAILED: … timeout` | The document or chunk took more than `DOC_TIMEOUT_MIN` (90 min). Split the document (`--pages`), or use a faster quality |
| `INCOMPLETE: batches … failed` | Some pages of a long document failed 3 times. Run the same command again to fill them in; saved batches are skipped |
| `conversion workers keep crashing` | The workers crashed many times within 10 minutes (the log lines show why: often a missing model or a damaged file type). They are parked and tried again once the 10 minutes have passed; fix the cause or restart the notebook |
| `cancelled: no client asked about it` | The client stopped polling for 15 minutes (killed, PC asleep). Run the command again: finished batches are kept |
| Low coverage warning for some pages | Fewer than 90% of the PDF's own words were found in the Markdown. Often harmless (units, LaTeX written as text), but check those pages. A garbled text layer is better with `--force-ocr` |
| Garbled text from a digital PDF | The text layer is bad (old fonts, missing Unicode). Use `--force-ocr`, or `--strip-ocr` for a bad OCR layer |
| A photo of two book pages came out as one page, or a single page was cut in two | Check `spreads_split` in `report.json` and look at the pages with `--show-layout`. `--no-split-spreads` reads every photo as it is. Please report such pages (see below) |
| A scanned book has no figures | Marker's `balanced` reading mode (the default) rarely finds figures on scans. Use `--mode vllm` |
| Text missing from a scan | Look at the page with `--show-layout`: text lines left outside every block are drawn in magenta (they are read separately). `--quality precise` asks the vision LLM for anything missing on every page |
| Heading levels differ from the document's styles | Office files go through PDF, so headings come from the layout and the PDF outline. `--office native` keeps Word's structure but loses equations |
| A table split in two | It crosses a page-chunk boundary. Use `SPLIT_MIN_PAGES = 0` in `JOB_DEFAULTS` to never split |
| Word equations missing | You used `--office native`. The default (`pdf`) keeps them |
| `--local needs Marker on this PC …` | Install it with `pip install -r requirements-local.txt`, and keep `dev/` next to `md_client.py` |

## Kaggle restarted the notebook

"Your notebook tried to allocate more memory than is available" means the session's RAM filled up. The box
guards against this (see [Architecture → Scheduling and memory safety](architecture.md#scheduling-and-memory-safety)),
but very unusual documents can still get there. The job's `box_resources.ram_at_min` (in `report.json`) lists the
biggest processes when RAM was lowest. Lower `WORKERS_PER_CPU` to `1.0`, `CHUNK_MAX_PAGES` to 12 or
`LLM_HOST_CACHE_MB` to 1024, then run again. The client resumes long documents where they stopped.

## Reporting a problem

Please open an issue with:
- the command you ran and the client's output;
- the relevant part of the Kaggle log;
- the document's `report.json` row (and the box's `code` from `--health`).

If you can share a document (or one page of it) that converts badly, attach it. Remove anything private first.
