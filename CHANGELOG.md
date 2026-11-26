# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-11-26

Three quality modes named by their trade-off, per-page decisions in the default mode, scans and phone photos
handled before OCR, and a box that runs unattended for a whole session. Measured on a 30-page regression set (one
hard case per page) on Kaggle 2×T4: `express` 233 s, `standard` 842 s, `precise` 2107 s, every regression check
passed in every mode.

### Changed (breaking)
- **Quality modes are `express`, `standard` (default) and `precise`.** Release-1 names are mapped: `fast` runs
  `express`, which makes **no** vision-LLM call (release 1's `fast` described figures); `balanced` runs `standard`;
  `accurate` runs `precise`. The client prints a note when it maps a name, and the box refuses unknown names with 400.
- `standard` decides per page: a page whose every word is the PDF's own text, with only text blocks and no picture,
  gets no vision-LLM call; every other page gets a short triage and then one whole-page audit, the element check and
  audit, or (handwriting and hard scans) also a finer audit.
- No quality uses Marker's own LLM steps any more; the element check and the audit replace them. `--redo-math`
  runs them on request (and now really does: it had no effect without them).
- The whole-page review is replaced by the element check in `standard` and `precise`, and in `precise` by a
  line-by-line rewrite in parts of about 600 tokens.
- Mermaid charts are drawn only for flowcharts; other diagrams, circuits and charts are described. Pictures of text
  keep the OCR's full reading under their description instead of a vision-LLM transcription.
- llama.cpp is pinned to the measured release (`LLAMA_CPP_REF = "b12627"`).

### Added
- **Scans first, in every mode:** scanned pages set upright, photos of two facing book pages cut into two pages
  and put in printed order (`--no-split-spreads` turns it off), and text lines the layout left in no block read and
  placed.
- **The element check:** every element of a page checked for its kind and structure, and text or figures missing
  from the Markdown added with evidence.
- **Prose made explicit** in `precise`, with the author's exact wording kept below each rewritten paragraph.
- **The OCR reads the text inside every picture;** the vision LLM uses it as evidence for the figure's description.
- `report.json`: the quality, each page's level in `standard` (`page_levels`), spreads cut and pages turned, a
  per-chunk, per-page and per-figure `trace`, and the biggest processes at the lowest free RAM.
- Client: `--show-layout` (each page drawn with the blocks the layout found), `--labels` (vision-LLM block labels
  for the trace), `--chunk-rounds`, `--version`; a per-level page count in the summary.
- Notebook: an express-only box (`EXPRESS_BOX`); the BOX READY banner shows the default quality.
- Development: `dev/modes_check.py` (vision-LLM requests per mode, in CI), benchmark tools (`bench_run.py`,
  `bench_score.py`, `regress_check.py`) and throughput tools (`load_test.py`, `tune_vlm_cap.py`).

### Fixed
- An upload a client never finished kept the box from ever counting as idle, so it never stopped by itself and
  used up GPU quota. Uploads, jobs and stored files no client has asked about for 15 minutes are now removed.
- Worker restarts were a budget for the box's whole life, and cancelled or timed-out pieces used it up too; after
  that every job failed with "workers keep crashing". Crashes are now counted over 10 minutes, a crash loop parks
  the workers until it clears, and stops on purpose don't count.
- A GPU server that was still loading (just restarted) answered 503 and the page lost its vision-LLM pass; the load
  balancer now sends the request to the other GPU.
- A finished job's queue files are removed (idle workers scanned them several times a second all session); only
  one job at a time prepares scans (each starts two processes the memory guard can't see); scanned pages' text
  lines are detected once instead of twice.
- A prose rewrite answered for "paragraph 0" replaced the last paragraph of its batch.
- The API answered malformed sizes, part counts and bodies with 500; they are 400 now.
- Pictures inside table cells were turned into plain text by the Markdown tidy-up before they could be cropped
  from the page; they are kept again.

## [1.0.0] - 2026-11-23

First public release.

### Conversion
- PDF (digital and scanned), images, Word, PowerPoint, Excel, OpenDocument, RTF, EPUB, HTML, DjVu, text, Markdown,
  CSV/TSV and zip archives to Markdown, with figures, tables, LaTeX equations and code.
- Kaggle notebook ("the box") with Marker and the Surya 2.5 OCR model on both T4 GPUs, and Qwen3-VL-8B as a
  vision LLM next to it.
- Office files rendered to PDF by LibreOffice, which keeps Word equations and layouts. DjVu via `ddjvu`.
- Any Unicode script: per-script text checks, Indic vowel-order repair, detection of untrustworthy text layers
  (including old Bijoy-font Bengali), legacy text encodings.

### Accuracy
- Three quality modes: `fast`, `balanced` (default) and `accurate`.
- Page review against the page image. A rewrite is kept only if no word or number is lost and nothing is
  invented, with paragraph-level merging and the PDF's text layer as the final authority.
- Line-by-line audit. Each fix is confirmed by the text layer or by a two-way second look, and guarded against
  stripping maths or formatting.
- Checks on Marker's own LLM steps: commentary, maths replaced by prose, ceiling/floor swaps, swapped symbols.
- Deterministic repairs: text-layer repair, tables realigned against the image, pictures in table cells, formulas
  drawn with shapes, equation numbers as `\tag{}`, Markdown tidy-up.
- Figures cropped to the figure itself, recovered when the layout model missed them, and described. Flow
  diagrams become Mermaid charts. Pictures of text are transcribed.
- A text coverage check against the PDF's text layer, in every report.

### Throughput and safety
- Up to 6 workers on Kaggle's 4 vCPUs, sized from free RAM, with a RAM start budget and a memory guard (pause,
  resume, stop the largest worker).
- Image-affinity load balancing and a bounded prompt cache: a page image is processed once per server, not once
  per call.
- Long documents split into page chunks in whole rounds. Batches are saved one by one and resumed after an
  interruption.
- Optional local Marker workers on your PC, which steal page chunks from the box.
- Per-job resource report: GPU busy %, GPU memory, CPU %, lowest free RAM.

### Client and API
- `md_client.py`: standard library only. It finds the box through a signed ntfy.sh announcement, uploads in
  parts, retries on tunnel loss, and cancels its jobs on Ctrl+C, SIGTERM and SIGHUP.
- Token-protected HTTP API with constant-time token check, upload size caps and path-traversal protection.
  Abandoned jobs are cancelled after 15 minutes.

[Unreleased]: ../../compare/v2.0.0...HEAD
[2.0.0]: ../../compare/v1.0.0...v2.0.0
[1.0.0]: ../../releases/tag/v1.0.0
