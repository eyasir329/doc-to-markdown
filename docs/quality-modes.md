# Quality modes

Two separate settings control a conversion:

- **`--mode`** chooses how Marker *reads* the pages: which pages get the Surya OCR model on the GPUs (see
  [Usage](usage.md#how-pages-are-read)). Leave it on `auto` unless you have scans with handwriting or
  illustrations (`vllm`).
- **`--quality`** chooses how much the *vision LLM* (Qwen3-VL-8B) checks and enriches the result afterwards. It is
  `fast`, `balanced` (default) or `accurate`.

Every mode keeps all the text: the modes differ in how many errors are caught and how much figure content is
written out, not in what's kept.

## What each mode does

| Stage | `fast` | `balanced` | `accurate` |
|---|---|---|---|
| Marker + Surya OCR, deterministic repairs: text-layer repair, Indic repair, tables checked against the image, drawn formulas, Markdown tidy-up | ✓ | ✓ | ✓ |
| Figures cropped to the figure itself, figure descriptions | ✓ | ✓ | ✓ |
| Flow diagrams read as a graph → Mermaid | 1 read | 2 reads + missing-arrow check | 3 reads + missing-arrow check |
| Marker's own LLM steps (tables, equations, forms, headings), each checked | – | ✓ | ✓ |
| Marker's inline-maths rewrite | – | – (the page review does it) | ✓ |
| Pictures of text (a newspaper page, a letter) transcribed in full, not only described | – | ✓ | ✓ |
| Page review against the page image, rewrite kept only if nothing is lost | – | whole page at once | section by section (≤ 900 tokens each) |
| Skip the review for pages the PDF's own text fully settles | – | ✓ | – (reviews every page) |
| Line-by-line audit: exact corrections, each confirmed by the page's text or by two second looks | 1 pass, 12-paragraph sections | 1 pass, 8-paragraph sections | 2 passes: 6-, then 3-paragraph sections |

The stages are explained in [How accuracy is protected](accuracy.md). The figure descriptions run at the same time
as the page review and audit, so the GPUs stay busy.

## Benchmarks

The benchmark is a 27-page test document with one hard case per page:
- UML activity diagrams and flowcharts;
- dense LaTeX, C and R code, tables with pictures in cells;
- a 1900 newspaper scan and a 1794 handwritten invoice;
- Bengali and Chinese text, charts, multi-panel figures.

59 automatic checks look for specific content, for example:
- a diagram's arrows;
- `\sqrt{2}` instead of `\sqrt[2]{2}`;
- a Bengali phone number;
- a ceiling bracket kept as a ceiling.

The document contains personal data and isn't published. All runs were on Kaggle 2×T4 in November 2026:

| Run | `fast` | `balanced` | `accurate` |
|---|---|---|---|
| 1 (first version of the modes) | 342 s · 54/59 | 875 s · 55/59 | 1042 s · 57/59 |
| 2 | 511 s · 54/59 | 1005 s · 56/59 | 1250 s · 55/59 |
| 3 | 516 s · 56/59 | 1068 s · 57/59 | 1192 s · 56/59 |
| 4 (6 workers, throughput changes) | 520 s · 56/59 | – · 58/59 ¹ | – |

¹ Another job ran on the box at the same time, so its time isn't comparable.

Text coverage (the share of the PDF's own words found in the Markdown) was 98.0–98.4% in every mode and run. In
run 4 the box's two GPUs were 85% and 86% busy, the CPUs 71%, and free RAM never went below 12 GB.

How to read the results:

- **fast** got faster between runs 1 and 2 because it gained figure crops, figure descriptions and a light audit.
  It also became more accurate (54 → 56). Its remaining misses are the things only the later stages do:
  - an equation the page prints as plain-text pseudo-LaTeX stays as text (no page review);
  - the newspaper scan is described, not transcribed;
  - one Chinese OCR look-alike (数字李生 → 数字孪生) is only caught by a finer audit.
- **balanced and accurate are within run-to-run variation of each other on this document.** The vision LLM's
  answers vary a little between runs, so one or two checks flip.
- **accurate does strictly more checking** than balanced:
  - the review works section by section, which matters on long, dense pages;
  - every page is reviewed, even those its text layer settles;
  - two audit passes in smaller sections;
  - Marker's inline-maths rewrite;
  - a third read of each diagram.
- On short documents the modes differ less. On a 3-page sample (a UML diagram, a table with pictures in cells, a
  formula drawn with shapes) all three modes gave the same result: fast 129 s, balanced 165 s, accurate 206 s.

## Choosing a mode

| Your documents | Mode |
|---|---|
| Large batches of digital documents (reports, books with a text layer) where the text matters most | `fast` |
| Most documents: papers, textbooks, slides, mixed content | `balanced` |
| Where one wrong character is costly: maths-heavy papers, numbers in financial or legal documents, scans with difficult OCR, documents you won't proofread | `accurate` |

You can also switch single stages on or off for one run, whatever the mode:
- `--no-review` skips the page review;
- `--no-describe` skips the figure descriptions;
- `--redo-math` adds the inline-maths rewrite;
- `--no-llm` skips every LLM pass;
- through the API, `"AUDIT": false` skips the audit.

A stage you set explicitly always wins over the mode's default.

## Measuring the trade-off on your own documents

With the box running:

```bash
python dev/compare_quality.py my_document.pdf --out test_out/quality --qualities fast balanced accurate
```

The script converts the document once per mode, each into `test_out/quality/<mode>/`. An existing finished
folder is reused, so delete it to convert again. For each mode it prints:
- the wall time and the box's time per stage;
- pages rewritten and kept;
- audit fixes and refusals;
- text coverage;
- the box's GPU, CPU and RAM use.

To add your own pass/fail checks, put a module `dev/suite_checks.py` next to the script. It needs a function
`score(md_path) -> ({case: [(check_name, ok), ...]}, extra)`, and when run as a script it should print a table
for the `.md` files given as arguments. `compare_quality.py` then adds a score column for documents whose name
contains `Comprehensive_Test_Suite` (adjust that test to your document's name). The file is in `.gitignore`,
because test documents often hold private data.

Compare modes on a *warm* box. The first job after the box starts is about a minute slower while the GPU servers
compile their kernels.
