## What and why

<!-- What does this change, and why? Link the issue it addresses (e.g. "Fixes #12"). -->

## How it was tested

<!-- Unit tests added or changed? For conversion changes: tested on Kaggle, with before/after results
     (e.g. `python dev/compare_quality.py doc.pdf`). -->

## Checklist

- [ ] `python -m pytest -q dev/test_pipeline.py`, `python dev/test_e2e.py` and `python dev/modes_check.py` pass
- [ ] `ruff check .` is clean
- [ ] notebook rebuilt (`python dev/build_nb.py`) if `dev/notebook/` or `dev/mdbox/` changed
- [ ] docs and `CHANGELOG.md` (*Unreleased*) updated for user-visible changes
- [ ] no model output is applied without a check that nothing is lost or invented
