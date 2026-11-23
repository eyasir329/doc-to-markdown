# Performance

## What to expect

On the 27-page benchmark (one hard case per page: diagrams, maths, scans, tables, figures, several scripts), on
Kaggle 2×T4:

| Mode | Wall time | Per page | Pages per hour |
|---|---|---|---|
| `fast` | 520 s | ~19 s | ~190 |
| `balanced` | 1068 s | ~40 s | ~90 |
| `accurate` | 1192 s | ~44 s | ~80 |

Pages of plain digital text need much less LLM work than these: where the PDF's own text settles a page, the
review (in `fast` and `balanced`) and the audit skip it. Scans and pages full of figures and maths need the most.

`report.json` shows where the time went for each document: `stage_seconds` sums the workers' time per stage
(`marker`, `figures`, `describe_figures`, `page_review`, `audit`, …), and `box_resources` shows how busy the box
was. In the latest `fast` run, both GPUs were 85–86% busy and the CPUs 71%, and free RAM never fell below
12 GB. The client prints this under each document.

## What makes it fast

All of these keep accuracy unchanged; each was measured before it was kept (see [Design notes](design-notes.md)).

| Technique | Effect |
|---|---|
| **Image reuse on the GPU.** A page's calls go to the server that already processed its image; the LLM servers keep a bounded prompt cache | an image costs 2.84 s the first time and 0.05 s after. Reuse rose from 2 of 48 calls to 29 of 48 |
| **Nothing waits for nothing.** Figure descriptions run beside the page review. Each page's audit starts as soon as its own review is done | the LLM's slots stay full |
| **Skip what can't be wrong.** Sections and pages that come straight from the PDF's text are settled, not reviewed | most pages of digital PDFs |
| **Whole rounds of chunks.** A document is cut into a multiple of the worker count | before, 27 pages made 6 chunks for 5 workers, so 4 workers waited (17–36% idle) |
| **More workers from spare RAM.** Up to 1.5 workers per vCPU (6 on Kaggle), with a RAM start budget | a worker waiting for the GPUs leaves its CPU free; 8–10 GB of RAM was unused before |
| **5 LLM slots per T4**, n-gram speculative decoding, flash attention, 8-bit KV cache | more requests decoded together; fixes that mostly copy text decode faster |
| **Diagrams read early.** A figure Marker calls a diagram has its graph read while it's still being cropped and described | one figure: 51 s → 26 s |
| **Warm-ups at start.** A page-sized request to every GPU server | the first job is no longer a minute slower |
| **Repeated images described once** | logos and icons on every page |

## Tips

- **Convert many files in one run** (`md_client.py ./folder`). All documents share one queue, and every worker
  stays busy.
- **Use `fast` for bulk text.** It keeps all the text and still checks it. Use `balanced` or `accurate` where
  maths, figures and scans matter.
- **`--no-describe`** saves the figure work when you only need the text. **`--no-llm`** saves all LLM work,
  which is much faster but less exact.
- **Your PC can help** with digital PDFs and office files (`--local auto`), where CPU work dominates. With scans or
  `--mode vllm`, both sides wait for the same two GPUs, so the gain is small.
- **Keep the box warm.** The first job after a start is about a minute slower. When comparing settings, compare
  on a warm box.
- **Mind the quota.** Kaggle gives 30 GPU-hours a week. The box stops itself after `IDLE_STOP_MIN` minutes
  without jobs.

## Tuning

The defaults are measured on Kaggle's 2×T4 machine. On a different machine, or for unusual documents:

| Setting | Raise it when | Lower it when |
|---|---|---|
| `WORKERS_PER_CPU` (1.5) | GPUs are idle while CPUs are busy, and RAM is free | RAM pauses appear in the progress messages |
| `MEM_PER_WORKER_GB` (3.0) | workers are stopped for memory | lots of RAM stays free (more workers fit) |
| `CHUNK_MAX_PAGES` (24) | — | workers run out of RAM on heavy pages |
| `VLM_SLOTS_PER_GPU` (8) | scans dominate and the GPUs have memory to spare | GPU out-of-memory errors |
| `LLM_VARIANTS` | your GPUs have more memory (more slots) | — |
| `LLM_HOST_CACHE_MB` (2048) | RAM is plentiful | RAM is tight |
| `LB_SPILL` (3) | image reuse (`/lb-status` on the box) is low | one GPU is much busier than the other |

## Why an 8B model

Qwen3-VL-8B in Q6_K is the largest vision model that runs at a usable speed next to Surya on a T4. A 27B model
split over both T4s generated 1.9 tokens/s (0.3 with tensor split), far too slow for page-by-page review. The
checks around the 8B model are what make it safe to use (see [How accuracy is protected](accuracy.md)).
