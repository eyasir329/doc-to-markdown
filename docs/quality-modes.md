# Quality modes

Two separate settings control a conversion:

- **`--quality`** chooses how much the vision LLM (Qwen3-VL-8B) checks and enriches the result: `express`,
  `standard` (default) or `precise`.
- **`--mode`** chooses how Marker *reads* the pages, i.e. which pages go to the Surya OCR model on the GPUs (see
  [Usage](usage.md#how-pages-are-read)). Leave it on `auto`.

Every mode runs the same reading and the same deterministic repairs, and every mode keeps all the text. The modes
differ only in what the vision LLM does on top: how many errors are caught and how much figure content is written
out.

## What each mode does

| | `express` | `standard` (default) | `precise` |
|---|---|---|---|
| Scans set upright, two-page photos cut in two, Marker + Surya OCR, text lines the layout left out read too | ✓ | ✓ | ✓ |
| Fixes without a model: the PDF's own characters, Indic vowel order, tables against the page image, drawn formulas, figures re-cropped, headings from the PDF outline, tidy-up | ✓ | ✓ | ✓ |
| Text inside pictures read by Surya | under each picture | evidence for the description; a long reading (40+ words) kept under it | evidence for the description; a long reading (40+ words) kept under it |
| Page triage (one short look at the page) | – | pages the PDF's own text does not settle | – (every page gets every pass) |
| Whole-page audit: exact corrections, each confirmed | – | plain-text pages | – |
| Element check (kind and structure of every element; missing text and figures added with evidence) + audit | – | tables, maths, code, figures, scans, forms | every page |
| Finer audit (3-paragraph sections) | – | handwriting and hard scans | every page |
| Line-by-line rewrite, each part kept only if nothing is lost or invented | – | – | every page its own text does not settle |
| Prose made explicit, the author's wording kept below each paragraph | – | – | paragraphs the page's own text does not confirm |
| Figures cropped to themselves and described by their meaning | – | figures on pages with diagrams or charts | every figure |
| Flowcharts as Mermaid charts | – | 2 reads + missing-arrow check | 3 reads + missing-arrow check |

Each stage is explained in [How accuracy is protected](accuracy.md).

### How `standard` decides

A page whose every word is the PDF's own text, with only text blocks on it (no table, maths, code, figure or
form) and no picture, gets **no vision-LLM call at all**: nothing on it can be improved. Most pages of a digital
book are such pages. Every other page gets one short triage call, which says what is on the page, and then the
passes of its level:

| Level | When | Passes |
|---|---|---|
| `none` | the triage finds nothing the page's own text doesn't settle | – |
| `audit` | plain, clean text | one whole-page audit |
| `inspect` | tables, maths, code, diagrams or charts; a scan, form or mixed page; a noisy page with a text layer; a failed triage | element check + audit |
| `hard` | handwriting; a degraded or difficult scan (no text layer) | element check + finer audit |

`report.json` lists each page's level (`page_levels`), and the client prints how many pages got each one.

## Measured on the 30-page regression set

The regression set has one page for each case the pipeline has to handle: seven two-page book photos, three
pages that must not be cut, upside-down and sideways photos, five handwritten pages, degraded historical scans,
receipts and invoices, a table grid, maths, a program listing, charts and flowcharts, a digital page with a text
layer, and Chinese, Devanagari and Hebrew text. It contains third-party and personal scans, so it isn't
published. Kaggle 2×T4, 6 workers, release 2.0.0 (box code `6218c2ef29`):

| | `express` | `standard` | `precise` |
|---|---|---|---|
| Wall time | **233 s** | **842 s** | **2107 s** |
| Pages per hour | 466 | 128 | 51 |
| Regression checks | 10 / 10 | 13 / 13 | 11 / 11 |
| Corrections kept | – | 38 | 118 |
| Figures described | – (16 pictures read by Surya) | 16 | 21 |
| Text coverage (pages with a text layer) | 95.2% | 95.2% | 95.2% |
| GPUs busy | 69% / 78% | 88% / 90% | 78% / 71% |
| Lowest free RAM | 11.4 GB | 4.0 GB | 2.8 GB |

Every mode cut all seven book spreads and nothing else, turned 8 pages upright, produced no repetition loops and
drew no Mermaid chart for a figure that isn't a flowchart. The page text agrees about 99% between the three modes.
What the vision LLM did:

- **standard:** the 30 pages got 8 `audit`, 11 `inspect` and 11 `hard` levels. 351 elements were inspected and 18
  of them fixed; 15 audit corrections were confirmed by a second look (for example `I_E = I_D + I_C` → `I_E = I_B +
  I_C`, `অনেক দুত` → `অনেক দ্রুত`); 5 missing passages were added. 62 proposed corrections were refused.
- **precise:** the line-by-line rewrite was kept on 28 pages and refused on 9 (6 added words, 3 changed a number);
  25 elements fixed, 9 audit corrections confirmed (`牡丹准货记卡` → `牡丹准贷记卡`), 8 missing passages added, 48
  paragraphs made more explicit. 52 rewrites of prose were refused, 9 of them because they added a number.

The checks are listed in [Performance](performance.md#the-regression-checks).

## Choosing a mode

| Your documents | Mode |
|---|---|
| Large batches where speed matters most: digital documents with a good text layer, clean scans you will read anyway | `express` |
| Most documents: books, papers, slides, scans, mixed content | `standard` |
| Where one wrong character is costly and you won't proofread: maths, numbers in financial or legal documents, handwriting, difficult scans | `precise` |

You can also switch single stages on or off for one run, whatever the mode:
- `--no-review` skips the page passes (triage, audit, element check, rewrite); figures are still described;
- `--no-describe` skips the figure descriptions (Surya's reading still goes under each picture);
- `--no-llm` skips every vision-LLM call (the same as `express`);
- `--labels` adds vision-LLM block labels to the report's trace (one more call per page; the Markdown is the same);
- through the API, `"AUDIT": false` skips the audit.

A stage you set explicitly always wins over the mode's default.

### Names from release 1

`--quality` still accepts the names of release 1 and maps them: `fast` runs `express`, `balanced` runs `standard`,
`accurate` runs `precise`. The client prints a note when you use one. Note that `fast` now means no vision-LLM pass
at all: release 1's `fast` still described figures.

## Measuring the trade-off on your own documents

With the box running:

```bash
python dev/compare_quality.py my_document.pdf --out test_out/quality
```

The script converts the document once per mode, each into `test_out/quality/<mode>/`. An existing finished
folder is reused, so delete it to convert again. For each mode it prints the wall time, the box's time per stage,
pages rewritten, audit fixes and refusals, text coverage, and the box's GPU, CPU and RAM use.

For page-by-page scores across modes, see [Development → Benchmarks](development.md#benchmarks).

Compare modes on a *warm* box: the first job after the box starts is about a minute slower while the GPU servers
compile their kernels.
