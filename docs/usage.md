# Usage

Everything is done with one command, `python md_client.py`, on your PC. The box (the Kaggle notebook) must be
running; see [Getting started](getting-started.md).

## Examples

```bash
python md_client.py book.pdf                          # -> markdown_out/book/book.md + images + report.json
python md_client.py report.doc slides.ppt sheet.xls   # several files: they convert in parallel
python md_client.py ./course_material --out ./md      # a whole folder; sub-folders are kept
python md_client.py archive.zip                       # every document inside the zip
python md_client.py scan.pdf --mode vllm              # the VLM reads every page (handwriting, bad scans, illustrated books)
python md_client.py paper.pdf --quality accurate      # the most careful checking
python md_client.py big.pdf --quality fast            # quickest, still keeps all the text
python md_client.py paper.pdf --pages 0-9             # only pages 0-9 (numbered from 0)
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
    └── _page_<N>_<Kind>_<M>.jpeg    # figures, diagrams, formula pictures, linked from the .md
```

Folders keep their sub-folder structure under `<out>`. Each figure is linked where it appears in the text, and
its description follows it as a quote block. Flow diagrams also get a Mermaid chart.

### `report.json`

`report.json` has one row per document. The most useful fields:

| Field | Meaning |
|---|---|
| `ok`, `error` | whether the document converted, and why not |
| `where` | `box`, `pc` or `box+pc`: who converted it |
| `pages`, `seconds`, `stage_seconds` | page count, time, and time per stage (`marker`, `describe_figures`, `page_review`, `audit`, …) |
| `quality` | the quality mode used |
| `text_coverage`, `low_coverage_pages` | share of the PDF's own words found in the Markdown; pages under 90% are listed |
| `pages_rewritten`, `pages_kept_draft` | pages the review improved / kept as they were |
| `audit` | line-by-line audit: fixes confirmed, refused (with the reason), sections settled by the text layer |
| `figures_described`, `tables_realigned`, `table_pictures`, `drawn_formulas`, `layer_fixes`, `indic_words_fixed` | what the repair steps did |
| `failed_pages`, `batches` | for long documents: page ranges that failed, and the batches |
| `box_resources` | how busy the box was while converting: GPU busy %, GPU memory, CPU %, lowest free RAM |

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
every saved batch is skipped. A batch with a failed page is retried up to 3 times. If a page still fails, the
document is reported `INCOMPLETE` and the same command fills in the gap later.

## Options

### Quality and checking

| Option | Effect |
|---|---|
| `--quality fast\|balanced\|accurate` | how much the vision LLM checks and enriches. Default: the box's `QUALITY` (`balanced`). See [Quality modes](quality-modes.md) |
| `--no-review` | skip the page-by-page review against each page image (faster) |
| `--no-describe` | don't write out the content of figures under each image |
| `--redo-math` | have Marker's LLM step rewrite inline maths as LaTeX (on by default in `accurate`) |
| `--llm` / `--no-llm` | force the vision-LLM passes on / skip them all (much faster, less exact tables, maths and figure text) |

### How pages are read

| Option | Effect |
|---|---|
| `--mode auto\|balanced\|vllm\|fast\|no_ocr` | Marker's engine. `auto` (default) is `balanced` when the box's GPUs run Surya, otherwise `fast`. `balanced` uses the Surya VLM for tables, maths, non-Latin scripts and scans. `vllm` has the VLM read every page: slowest, best for handwriting, poor scans and illustrated books. `fast` is CPU only. `no_ocr` is CPU only and uses only the PDF's own text |
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

### Long documents and your PC

| Option | Effect |
|---|---|
| `--batch-pages N\|auto\|0` | batch size in pages for long documents (see above). `0` = the whole document in one piece |
| `--no-overlap` | start a batch only after the previous one is saved |
| `--local N\|auto\|0` | Marker workers on your PC next to the box (default `auto` when Marker is installed). They take the smallest files and steal page chunks from the box's long PDFs, using the box's GPUs through the tunnel. Requires `pip install -r requirements-local.txt` |

### Connection and housekeeping

| Option | Effect |
|---|---|
| `--url URL` | the box's address, if the automatic lookup fails (from the BOX READY banner). Also `BOX_URL` |
| `--token TOKEN` | the box token. Default: the `.box_token` file next to `md_client.py` or in the current folder, or `BOX_TOKEN` |
| `--health` | show the box's status and exit |
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
`SIGHUP`). If the client is killed without a chance to clean up, the box cancels any job that no client has asked
about for 15 minutes (`MD_ABANDON_MIN`), so a lost client never holds the box's workers.

## Exit status

`0` when the run finished: check `report.json` and the printed summary for documents that failed. `1` on a fatal
error (no token, box not found, bad arguments). `130` when cancelled.
