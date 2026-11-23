# Development

## Set up

```bash
git clone <this repository> && cd doc-to-markdown
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Everything below runs on a normal PC without a GPU. Real conversion quality and the GPU servers can only be
tested on Kaggle.

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
python -m pytest -q dev/test_pipeline.py   # ~130 unit tests, ~10 s
python dev/test_e2e.py                     # end to end: the real server, pipeline, worker and client
```

`test_pipeline.py` covers, among others:
- file-type routing, text and CSV conversion, encodings;
- zip safety, option validation;
- chunk planning and joining, worker-pool sizing, the memory start budget;
- the loss checks, paragraph merging, the audit guards;
- text-layer and Indic repairs, the Markdown tidy-up;
- figure cropping, graph → Mermaid, table realignment, drawn formulas;
- the load balancer's affinity.

One test converts an RTF file with LibreOffice and is skipped when `soffice` isn't installed.

`test_e2e.py` runs "the box" with uvicorn on your PC, with fake `marker` and `pypdfium2` packages from
`dev/fakes/`. It checks that:
- a 120-page PDF is split, the PC's local workers steal chunks, and the result is joined back with every page and
  image in order;
- the VLM chain works (PC worker → client proxy → box `/vlm` → VLM).

## Lint

```bash
ruff check .
```

The configuration is in `ruff.toml`: syntax errors, pyflakes and bugbear.

## Build the notebook

```bash
python dev/build_nb.py      # -> docs_to_markdown_cloudflare.ipynb
```

The build is deterministic, and CI fails if the committed notebook differs from a fresh build. So rebuild and
commit the notebook with every change to `dev/notebook/` or `dev/mdbox/`.

## Test on Kaggle

1. Rebuild the notebook and import it on Kaggle (or replace the notebook's file there).
2. Run it (see [Getting started](getting-started.md)) and wait for BOX READY.
3. Convert documents that exercise your change, and compare the quality modes when your change affects them:

   ```bash
   python dev/compare_quality.py my_document.pdf --out test_out/quality
   ```

4. Check the document's `report.json` row: `stage_seconds`, `audit`, `text_coverage`, `box_resources`.

`test_out/` and `markdown_out/` are in `.gitignore`. Keep test documents with private data out of the repository.

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

1. Add the key to all three profiles in `review.QUALITY_PROFILES`.
2. Read it with `review.quality_option(opts, "KEY")`. A job's own option always wins over the profile.
3. To make it settable per job, add it to `OPT_TYPES` in `server.py` and, if useful, a flag in `md_client.py`.
4. Document it in [Quality modes](quality-modes.md) and [Configuration](configuration.md), and add a test.

## Releasing

1. Update `CHANGELOG.md`.
2. Make sure CI passes and the notebook is rebuilt.
3. Tag the release (`git tag v1.x.y`) and create a GitHub release. Attach the notebook so users can import it
   directly.
