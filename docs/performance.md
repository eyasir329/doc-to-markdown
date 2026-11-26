# Performance

## What to expect

Measured on the 30-page regression set (one hard case per page, see below) on Kaggle 2×T4 with 6 workers,
release 2.0.0:

| Mode | Wall time | Per page | Pages per hour |
|---|---|---|---|
| `express` | 233 s | ~8 s | ~466 |
| `standard` | 842 s | ~28 s | ~128 |
| `precise` | 2107 s | ~70 s | ~51 |

The set is deliberately hard: 29 of its 30 pages are photos or scans, so every page goes through OCR and, in
`standard`, through the vision LLM's checks. Digital documents are faster in every mode, and much faster in
`standard`: a page whose every word is the PDF's own text, with only text blocks on it, gets no vision-LLM call.

`report.json` shows where the time went for each document: `stage_seconds` sums the workers' time per stage
(`marker`, `figures`, `picture_ocr`, `describe_review_audit`, `describe_figures`), `trace` breaks it down per page
chunk, page and figure, and `box_resources` shows how busy the box was. The client prints this under each
document.

## Where the time goes

| Mode | Marker + Surya OCR | Vision-LLM page passes and figures | GPUs busy | CPUs busy |
|---|---|---|---|---|
| `express` | ~85% of the workers' time | – | 69% / 78% | 74% |
| `standard` | ~25% | ~70% | 88% / 90% | 71% |
| `precise` | ~15% | ~80% | 78% / 71% | 49% |

In `express` the box is limited by Surya's speed on the GPUs. In `standard` and `precise` it is limited by the
vision LLM: one worker alone keeps all of its 10 slots busy, and the pages wait their turn.

**RAM.** In `standard` and `precise` the two vision-LLM servers grow from about 1.9 GB to 4.5–5 GB each, because
each keeps up to `LLM_HOST_CACHE_MB` (2 GB) of processed page images so a page's next call doesn't process its
image again. Free RAM then sits near the memory guard's pause line (4 GB): in the regression set's `precise` run
the lowest free RAM was 2.8 GB, and for much of the run the guard let one page chunk start at a time. The GPUs
stayed 71–78% busy, against 88–90% in `standard`: an estimated fifth of `precise`'s speed. The
guard's thresholds are deliberately conservative, because a full session RAM restarts the whole Kaggle
notebook. See [Tuning](#tuning) for the settings involved.

## What makes it fast

| Technique | Effect |
|---|---|
| **Quiet pages.** In `standard`, a page its own text settles, with only text blocks, gets no vision-LLM call | most pages of digital documents cost nothing beyond Marker |
| **One short triage, then only the passes a page needs** | plain text gets one audit; only hard pages get the finer audit |
| **Image reuse on the GPU.** A page's calls go to the server that already processed its image, and the servers keep a bounded prompt cache | a page image is about 1,500 prompt tokens: processed once per server, not once per call |
| **Nothing waits for nothing.** A worker works on up to 10 pages (and figures) at once, and each page's next pass starts as soon as its own previous pass is done | the vision LLM's slots stay full |
| **Skip what can't be wrong.** Sections and pages that come straight from the PDF's text are settled, not audited or rewritten | most pages of digital PDFs |
| **Whole rounds of page chunks, heaviest first.** A document is cut into a multiple of the worker count; scan-heavy chunks get fewer pages and start first | no worker waits for another's second round, and the slowest chunk doesn't end the job alone |
| **More workers from spare RAM.** Up to 1.5 workers per vCPU (6 on Kaggle), with a RAM start budget | a worker waiting for the GPUs leaves its CPU free |
| **5 vision-LLM slots per T4**, n-gram speculative decoding, flash attention, 8-bit KV cache; Surya with flash attention | more requests decoded together |
| **A flowchart's graph is read only after its description says it is one** | no graph reads for circuits, charts and photos |
| **Repeated images described once** | logos and icons on every page |
| **Warm-ups at start.** A page-sized request to every GPU server | the first job isn't slowed by kernel compilation |

## The regression checks

`dev/regress_check.py` checks each mode's output of the 30-page set:

| Check | Modes |
|---|---|
| all 30 pages present, none empty | all |
| no repetition loops (a line or paragraph repeated three times or more) | all |
| exactly the 7 two-page book photos cut in two, and no other page (a curved page, an invoice, a music sheet and a page with an empty middle band stay whole) | all |
| upside-down and sideways photos turned upright | all |
| no Mermaid chart for figures that aren't flowcharts (an op-amp transfer curve, handwritten filter equations) | all |
| the program listing on a book spread keeps its code block | all |
| text coverage of the pages with a text layer ≥ 90% | all |
| the report names the quality that ran | all |
| no vision-LLM work at all | `express` |
| a level for every page; handwritten pages at least `inspect`; a digital page with a table not sent to the finer audit | `standard` |
| the page passes ran, and no vision-LLM call failed | `standard`, `precise` |

How close the three modes' text is, by kind of page (word overlap with the other modes):

| Pages | `express` | `standard` | `precise` |
|---|---|---|---|
| 7 two-page book photos | 0.994 | 0.995 | 0.991 |
| 5 degraded historical scans | 0.977 | 0.979 | 0.978 |
| 5 handwritten exam notes | 0.981 | 0.986 | 0.977 |
| 3 academic pages (LaTeX, code) | 0.990 | 0.989 | 0.981 |
| 10 other pages (receipts, invoices, tables, CJK, Hebrew, Devanagari, rotated photos, …) | 0.98–1.0 | 0.97–1.0 | 0.97–1.0 |

`precise` agrees a little less with the other two modes because it changes more: its line-by-line rewrite and
its prose rewrite (with the original wording kept below each rewritten paragraph).

## Tips

- **Convert many files in one run** (`md_client.py ./folder`). All documents share one queue, and every worker
  stays busy.
- **Use `express` for bulk text** you'll read anyway. It keeps all the text; only the vision LLM's checks are left
  out. Use `standard` for most documents and `precise` where one wrong character matters.
- **`--no-describe`** saves the figure work when you only need the text.
- **Your PC can help** with digital PDFs and office files (`--local auto`), where CPU work dominates. With scans,
  both sides wait for the same two GPUs, so the gain is small.
- **Keep the box warm.** The first job after a start is about a minute slower. Compare settings on a warm box.
- **Mind the quota.** Kaggle gives 30 GPU-hours a week. The box stops itself after `IDLE_STOP_MIN` minutes
  without jobs.

## Tuning

The defaults are measured on Kaggle's 2×T4 machine. On a different machine, or for unusual documents:

| Setting | Raise it when | Lower it when |
|---|---|---|
| `WORKERS_PER_CPU` (1.5) | GPUs are idle while CPUs are busy, and RAM is free | the progress line often says "RAM low" |
| `MEM_PER_WORKER_GB` (3.0) | workers are stopped for memory | lots of RAM stays free (more workers fit) |
| `LLM_HOST_CACHE_MB` (2048) | RAM is plentiful | `standard`/`precise` runs say "RAM low: one piece at a time" for long stretches (each server holds up to this much) |
| `CHUNK_MAX_PAGES` (24) | — | workers run out of RAM on heavy pages |
| `CHUNK_ROUNDS` (2), or `--chunk-rounds` per job | the end of a job runs on few workers | small documents (more, smaller chunks cost more per-chunk work) |
| `VLM_SLOTS_PER_GPU` (8) | scans dominate and the GPUs have memory to spare | GPU out-of-memory errors |
| `LLM_VARIANTS` | your GPUs have more memory (more slots) | — |
| `VLM_LB_MAX_INFLIGHT` (0 = no cap) | — | `dev/tune_vlm_cap.py` measures a cap as faster on your box |
| `LB_SPILL` (3) | image reuse (`/llm/lb-status` on the box) is low | one GPU is much busier than the other |

Change one setting at a time and compare on the same documents (`dev/compare_quality.py`, or the benchmark tools in
[Development](development.md#benchmarks)).

## Why an 8B model

Qwen3-VL-8B in Q6_K is the largest vision model that runs at a usable speed next to Surya on a T4: one server per
GPU, 5 slots each. Larger models split across both T4s are far too slow for page-by-page checking. The checks
around the 8B model are what make it safe to use (see [How accuracy is protected](accuracy.md)).
