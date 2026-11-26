# Usage

Everything is done with one command, `python md_client.py`, on your PC. The box (the Kaggle notebook) must be
running; see [Getting started](getting-started.md).

## Examples

```bash
python md_client.py book.pdf                          # -> markdown_out/book/book.md + images + report.json
python md_client.py report.doc slides.ppt sheet.xls   # several files: they convert in parallel
python md_client.py ./course_material --out ./md      # a whole folder; sub-folders are kept
python md_client.py archive.zip                       # every document inside the zip
python md_client.py big.pdf --quality express         # fastest: no vision-LLM pass, still keeps all the text
python md_client.py paper.pdf --quality precise       # slowest, most careful: every check on every page
python md_client.py paper.pdf --pages 0-9             # only pages 0-9 (numbered from 0)
python md_client.py scans.pdf --show-layout           # also draw each page with the blocks the layout found
python md_client.py notes.docx --office native        # Marker reads the .docx itself (keeps links; loses Word equations)
python md_client.py out/book/book.md --describe-again # rewrite every figure description of a finished file
python md_client.py --health                          # is the box up? GPUs, workers, models, memory, idle time
```

## What comes back

```
<out>/                               # --out, default ./markdown_out
├── report.json                      # one row per document (see below)
└── <name>/
    ├── <name>.md                    # the Markdown (or .html / .json with --format)
    ├── <name>_meta.json             # table of contents and per-page statistics from Marker
    ├── _page_<N>_<Kind>_<M>.jpeg    # figures, diagrams, formula pictures, linked from the .md
    └── layout/                      # with --show-layout: page_<n>.png (blocks drawn) and page_<n>.json
```

Folders keep their sub-folder structure under `<out>`. Each figure is linked where it appears in the text:

- in `standard` and `precise`, its description follows it as a quote block ("**Figure content:**"), with a
  Mermaid chart for a flowchart and, when the OCR read 40 words or more in the picture, the full "**Text in the
  image**";
- in `express`, the text the OCR read in the picture follows it.

In `precise`, a paragraph the vision LLM made more explicit is followed by the author's exact wording in a
collapsed "Original wording" block.

### `report.json`

`report.json` has one row per document. The most useful fields:

| Field | Meaning |
|---|---|
| `ok`, `error` | whether the document converted, and why not |
| `where` | `box`, `pc` or `box+pc`: who converted it |
| `pages`, `seconds`, `stage_seconds` | page count, time, and the workers' time per stage (`marker`, `figures`, `picture_ocr`, `describe_review_audit`, `describe_figures`) |
| `quality`, `page_levels` | the quality mode used; in `standard`, the level each page got (`none`, `audit`, `inspect`, `hard`) |
| `text_coverage`, `low_coverage_pages` | share of the PDF's own words found in the Markdown; pages under 90% are listed |
| `audit` | the vision LLM's page passes: elements inspected and fixed, missing text added, audit fixes confirmed or refused (with the reason), settled sections, rewritten paragraphs |
| `pages_rewritten`, `pages_kept_draft` | `precise`: pages the line-by-line rewrite improved / kept as they were, and why |
| `figures_described`, `pictures_read`, `picture_text_kept` | figures described; pictures the OCR read; long readings kept under a description |
| `spreads_split`, `spreads_reordered`, `pages_turned` | two-page photos cut in two (the page count stays the original's), spreads put in printed order, pages set upright |
| `tables_realigned`, `table_pictures`, `drawn_formulas`, `layer_fixes`, `indic_words_fixed` | what the repair steps did |
| `failed_pages`, `batches` | for long documents: page ranges that failed, and the batches |
| `box_resources` | how busy the box was while converting: GPU busy %, GPU memory, CPU %, lowest free RAM and the biggest processes at that point |
| `trace` | per page chunk, page, figure and picture: steps, seconds and counts, for tuning |

The client prints a summary of these under each document.

## Long documents: batches and resume

A document longer than `--batch-pages` pages (default `auto`, which is box workers × chunk size, e.g. 6 × 24 =
144) is converted one batch at a time:

1. It's uploaded once and stored on the box. Office and DjVu files are rendered to PDF there.
2. Each batch is converted, downloaded and saved on your PC (`<name>/.<name>.parts/<first>-<last>/`) as soon as
   it's complete.
3. The next batch starts while the last pieces of the current one finish, so no worker sits idle.
   `--no-overlap` turns this off.
4. At the end the batches are joined in page order into `<name>/<name>.md`, and the parts folder is removed.

If anything stops the run (the box restarts, your PC sleeps, you press Ctrl+C), **run the same command again**:
every saved batch is skipped. A batch saved with other settings (another quality, for example) is converted again,
so one document never mixes two settings. A batch with a failed page is retried up to 3 times. If a page still
fails, the document is reported `INCOMPLETE` and the same command fills in the gap later.

## Options

### Quality and checking

| Option | Effect |
|---|---|
| `--quality express\|standard\|precise` | how much the vision LLM checks and enriches. Default: the box's `QUALITY` (`standard`). Release-1 names still work (`fast` = express, `balanced` = standard, `accurate` = precise). See [Quality modes](quality-modes.md) |
| `--no-review` | skip the vision LLM's page passes (triage, audit, element check, rewrite); figures are still described |
| `--no-describe` | don't describe figures (the OCR's reading still goes under each picture) |
| `--llm` / `--no-llm` | force the vision-LLM passes on / skip them all |
| `--labels` | also label every layout block with the vision LLM (handwriting, code, titles, charts, …) for the report's trace and `--show-layout`. One more call per page; the Markdown is the same |
| `--redo-math` | also run Marker's own LLM steps, which rewrite inline maths as LaTeX (and redo tables, forms, equations). Slower; no quality does it by itself |

### How pages are read

| Option | Effect |
|---|---|
| `--mode auto\|balanced\|vllm\|fast\|no_ocr` | Marker's reading engine (not the quality). `auto` (default) is `balanced` when the box's GPUs run Surya, otherwise `fast`. `balanced` uses Surya for tables, maths, non-Latin scripts and scans. `vllm` has Surya read every page whole: slowest, for illustrated scans. `fast` is CPU only. `no_ocr` is CPU only and uses only the PDF's own text |
| `--no-split-spreads` | read a photo of two facing book pages as one page (default: set scanned pages upright and cut such photos in two first) |
| `--force-ocr` | OCR every page even if it has a text layer (for a bad or garbled text layer) |
| `--strip-ocr` | drop an existing bad OCR text layer and OCR again |
| `--office pdf\|native` | `pdf` (default): LibreOffice renders office files to PDF first, which keeps Word equations, text boxes and slide layouts. `native`: Marker reads `.docx`/`.pptx` itself, which is faster and keeps hyperlinks but loses Word equations. Old formats (`.doc`, `.ppt`, `.xls`, …) always use LibreOffice |
| `--pages 0,5-10` | convert only these pages, numbered from 0 |

### Output

| Option | Effect |
|---|---|
| `--out DIR`, `-o DIR` | output folder (default `./markdown_out`) |
| `--format markdown\|html\|json\|chunks` | output format (default `markdown`). `.txt`, `.md` and `.csv` inputs always become Markdown |
| `--paginate` | put a page separator between pages |
| `--keep-headers` | keep running page headers and footers |
| `--no-images` | don't extract figures |
| `--raw-figures` | keep Marker's own figure crops. By default figures are re-cropped from the page at 300 DPI, so labels aren't cut off and split figures are joined |
| `--show-layout` | also write `<name>/layout/`: every page with the blocks the layout found (type, score, vision-LLM label) and, in magenta, text lines left outside them; with each page's blocks as JSON |

### Long documents and your PC

| Option | Effect |
|---|---|
| `--batch-pages N\|auto\|0` | batch size in pages for long documents (see above). `0` = the whole document in one piece |
| `--no-overlap` | start a batch only after the previous one is saved |
| `--chunk-rounds N` | page chunks per box worker for long PDFs (1–8, box default 2): more = smaller chunks and a shorter tail, more per-chunk work |
| `--local N\|auto\|0` | Marker workers on your PC next to the box (default `auto` when Marker is installed). They take the smallest files and steal page chunks from the box's long PDFs, using the box's GPUs through the tunnel. Requires `pip install -r requirements-local.txt` |

### Connection and housekeeping

| Option | Effect |
|---|---|
| `--url URL` | the box's address, if the automatic lookup fails (from the BOX READY banner). Also `BOX_URL` |
| `--token TOKEN` | the box token. Default: the `.box_token` file next to `md_client.py` or in the current folder, or `BOX_TOKEN` |
| `--health` | show the box's status and exit |
| `--version` | show the client's version and exit |
| `--keep-on-box` | don't delete the job from the box after downloading |
| `--describe-again FILE.md` | write every figure description of a finished file again with the box's vision LLM, without converting again. The old file is kept as `FILE.md.bak`. Needs `dev/` next to `md_client.py` and `numpy`, `pillow` and `scipy` |

## Supported inputs

| Kind | Extensions |
|---|---|
| PDF | `.pdf`, digital or scanned |
| Images | `.png` `.jpg` `.jpeg` `.webp` `.tif` `.tiff` `.gif` `.bmp` |
| Word processing | `.docx` `.doc` `.docm` `.dot` `.dotx` `.rtf` `.odt` `.ott` `.fodt` `.wpd` `.wps` |
| Presentations | `.pptx` `.ppt` `.pptm` `.pps` `.ppsx` `.pot` `.potx` `.odp` `.otp` `.fodp` `.odg` |
| Spreadsheets | `.xlsx` `.xls` `.xlsm` `.xlsb` `.ods` `.ots` `.fods`: one Markdown table per sheet |
| Other | `.epub` `.html` `.htm` `.djvu` `.djv` `.txt` `.md` `.markdown` `.csv` `.tsv`, and `.zip` archives of any of these |

Not supported: Apple `.pages`, `.key` and `.numbers`, `.mobi`, `.xps`, and password-protected files. Export them
to PDF first. Text files can be in any encoding (UTF-8/16/32, GBK, Big5, Shift-JIS, EUC-KR, Windows code pages,
…).

## Cancelling

Ctrl+C cancels your jobs on the box and exits. So does closing the terminal or stopping the client (`SIGTERM`,
`SIGHUP`). If the client is killed without a chance to clean up, the box cancels any job or upload that no client
has asked about for 15 minutes (`MD_ABANDON_MIN`), so a lost client never holds the box's workers or keeps it from
stopping when idle.

## Exit status

`0` when the run finished: check `report.json` and the printed summary for documents that failed. `1` on a fatal
error (no token, box not found, an option the box refuses). `2` on invalid command-line arguments. `130` when
cancelled.
