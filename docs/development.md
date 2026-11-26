# Development

## Set up

```bash
git clone https://github.com/eyasir329/doc-to-markdown && cd doc-to-markdown
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Everything in [Tests](#tests) runs on a normal PC without a GPU. Real conversion quality and the GPU servers can
only be tested on Kaggle ([Test on Kaggle](#test-on-kaggle)).

## Layout

| Path | What to edit there |
|---|---|
| `dev/mdbox/*.py` | the box's modules ([Architecture](architecture.md#the-boxs-modules)) |
| `dev/notebook/NN_name.md\|py` | the notebook's cells, one file each, in file-name order. `00_intro.md` is the first page; `02_config.py` is section 1 with every setting |
| `md_client.py` | the client. **Standard library only**: it must run on any PC with Python 3.10+ |
| `docs/` | user documentation |
| `docs_to_markdown_cloudflare.ipynb` | **generated**: never edit it by hand |

## Tests

```bash
python -m pytest -q dev/test_pipeline.py   # ~175 unit tests, ~20 s
python dev/test_e2e.py                     # end to end: the real server, pipeline, worker and client
python dev/modes_check.py                  # every quality end to end, counting the vision-LLM requests each makes
```

`test_pipeline.py` covers, among others:
- file-type routing, text and CSV conversion, encodings, zip safety;
- option validation, quality names (and release-1 names), the API's 400 errors, the upload janitor;
- chunk planning and joining, worker-pool sizing, the memory guard and start budget, the crash budget;
- the quality profiles, `standard`'s page levels and quiet pages, the order of each page's passes;
- the loss checks, paragraph merging, the audit guards, the element check, the prose rewrite's checks;
- text-layer and Indic repairs, the Markdown tidy-up;
- figure cropping, flowchart → Mermaid, table realignment, drawn formulas;
- page orientation, two-page photo detection and printed page order;
- text lines the layout left out, the layout listing and block labels;
- the load balancer's affinity, its cap and its handling of a loading server.

One test converts an RTF file with LibreOffice and is skipped when `soffice` isn't installed.

`test_e2e.py` runs "the box" with uvicorn on your PC, with fake `marker` and `pypdfium2` packages from
`dev/fakes/` and fake GPU servers. It checks that:
- a 120-page PDF is split, the PC's local workers steal chunks, and the result is joined back with every page and
  image in order;
- a long document resumes from saved batches and redoes a batch saved with other settings;
- the VLM chain works (PC worker → client proxy → box `/vlm` → VLM).

`modes_check.py` converts a 3-page document in every quality (and with `--no-review`, `--labels` and a release-1
name) and prints how many vision-LLM requests each made: `express` must make none.

## Lint

```bash
ruff check .
```

The configuration is in `ruff.toml`: syntax errors and pyflakes (undefined names, unused imports and variables).

## Build the notebook

```bash
python dev/build_nb.py      # -> docs_to_markdown_cloudflare.ipynb
```

The build is deterministic, and CI fails if the committed notebook differs from a fresh build. So rebuild and
commit the notebook with every change to `dev/notebook/` or `dev/mdbox/`. A new module in `dev/mdbox/` must also
be added to `MODULES` in `dev/build_nb.py` (a test checks this).

## Test on Kaggle

1. Rebuild the notebook and import it on Kaggle (or replace the notebook's file there).
2. Run it (see [Getting started](getting-started.md)) and wait for BOX READY. `python md_client.py --health`
   shows the box's `code`: a hash of `dev/mdbox/*.py`, so you can check the box runs your change.
3. Convert documents that exercise your change, and compare the quality modes when your change affects them:

   ```bash
   python dev/compare_quality.py my_document.pdf --out test_out/quality
   ```

4. Check the document's `report.json` row: `stage_seconds`, `audit`, `page_levels`, `text_coverage`,
   `box_resources`, `trace`.

`test_out/` and `markdown_out/` are in `.gitignore`. Keep test documents with private data out of the repository.

## Benchmarks

For a benchmark PDF with a manifest (one entry per page: `page`, `category`, `title`, …):

```bash
python dev/bench_run.py BENCH.pdf manifest.json test_out/bench/run1 --modes express,standard,precise --code <box code>
```

`bench_run.py` waits until the box runs the given code, converts the PDF once per mode (with `--paginate`, so pages
can be scored one by one), then runs `bench_score.py`. That writes `pages.csv` (one row per mode and page),
`summary.md` (per mode and per category) and appends to `history.jsonl`, so later runs can be compared. Without
ground truth it scores what needs no reference: words, empty pages, the share of the expected script, repeated
lines, agreement between the modes (a low agreement marks a page to look at) and the box's trace per page. With
ground-truth files (`ground_truth/<page>.md` next to the manifest), it adds character error rate and word F1.

`regress_check.py` turns the release's 30-page regression set into pass/fail checks (see
[Performance → The regression checks](performance.md#the-regression-checks)); the set itself isn't published.

For throughput: `dev/load_test.py` converts several copies of a document at once and samples the box every 2 s
(GPU busy, RAM, waiting tasks, Surya requests in flight); `dev/tune_vlm_cap.py` runs it for several caps on Surya
requests per GPU and leaves the fastest set.

## Conventions

- **Models propose, checks decide.** Any new step that lets a model change text must pass a deterministic check
  that nothing was lost or invented (see [How accuracy is protected](accuracy.md)). When in doubt, keep the draft.
- **Measure, then change.** Comments in the code record why something is the way it is, usually with the
  measurement that decided it. Keep doing that: the next person needs to know what breaks if they change it.
- **Every branch, loop, parser or check you add gets a focused test** in `dev/test_pipeline.py`.
- **No new dependencies for the client.** Box dependencies are installed by the notebook's install cell; add new
  ones there deliberately, with pinned versions where it matters.
- **Small, surgical changes.** No drive-by renames or reformatting in the same pull request.
- **Known limits are marked** in code with `shortcut: <limit>; revisit when <trigger>`.

## Adding a setting to the quality modes

1. Add the key to `_BASE` in `review.py`, and set it in the profiles that differ (`QUALITY_PROFILES`).
2. Read it with `review.quality_option(opts, "KEY")`. A job's own option always wins over the profile.
3. For a per-page pass in `standard`, add it to the levels in `review.PAGE_LEVELS` and to `audit_pages`.
4. To make it settable per job, add it to `OPT_TYPES` in `server.py` and, if useful, a flag in `md_client.py`.
5. Document it in [Quality modes](quality-modes.md) and [Configuration](configuration.md), and add a test.

## Releasing

1. Update `CHANGELOG.md` and the version in `md_client.py` (`__version__`).
2. Make sure CI passes and the notebook is rebuilt.
3. Measure the release on Kaggle (the regression set in every quality) and update the numbers in the docs.
4. Tag the release (`git tag vX.Y.Z`) and create a GitHub release. Attach the notebook so users can import it
   directly.
