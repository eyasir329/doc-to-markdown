# Changelog

All notable changes to this project are documented in this file. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project uses
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Pictures inside table cells were turned into plain text by the Markdown tidy-up before they could be cropped from the page; they are kept again.

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

[Unreleased]: ../../compare/v1.0.0...HEAD
[1.0.0]: ../../releases/tag/v1.0.0
