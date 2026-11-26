# Contributing

Thank you for helping! This project turns documents into faithful Markdown, so the most valuable contributions
are often not code. Documents that convert badly, with a clear description of what went wrong, are what make it
better.

By taking part you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to help

- **Report a document that converts badly.** Use the *Conversion quality* issue form and include:
  - what the page shows and what the Markdown has instead;
  - the quality mode (`--quality`) and Marker mode (`--mode`);
  - the document's row from `report.json`.

  If you can, attach the page (or a minimal PDF that reproduces it). **Remove anything private first**: never
  attach documents with personal data.
- **Report a bug** (a crash, a hang, a wrong option) with the *Bug report* form. Include the command, the client's
  output and the relevant part of the Kaggle log.
- **Improve the documentation.** If something was unclear to you, it's unclear to others too.
- **Contribute code.** See below.

## Before you write code

For anything larger than a small fix, please open an issue first and describe the problem and your idea. This
avoids work that can't be merged. For example, a change that makes conversion faster but less accurate won't be
accepted as a default.

## Development setup

See [docs/development.md](docs/development.md). In short:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest -q dev/test_pipeline.py && python dev/test_e2e.py && python dev/modes_check.py && ruff check .
```

## Rules of the codebase

- **Models propose, checks decide.** A model may never change the text unchecked. Every step that applies a
  model's answer must verify that no word or number was lost and nothing was invented. When in doubt, the draft
  stays. See [How accuracy is protected](docs/accuracy.md).
- **Never edit the `.ipynb` by hand.** Edit `dev/notebook/` or `dev/mdbox/`, then run `python dev/build_nb.py`,
  and commit both. CI checks that they match.
- **The client stays standard-library only.** `md_client.py` must run on any PC with Python 3.10+ and nothing
  installed.
- **Test what you add.** Every new check, parser, branch or loop gets a focused test in `dev/test_pipeline.py`.
- **Explain why.** Comments record why the code is the way it is, ideally with the measurement that decided it.
- **Keep changes focused.** One topic per pull request, with no unrelated renames or reformatting.

## Pull request checklist

- [ ] `python -m pytest -q dev/test_pipeline.py`, `python dev/test_e2e.py` and `python dev/modes_check.py` pass
- [ ] `ruff check .` is clean
- [ ] the notebook is rebuilt (`python dev/build_nb.py`) if `dev/notebook/` or `dev/mdbox/` changed
- [ ] the docs are updated for any user-visible change, and `CHANGELOG.md` has an entry under *Unreleased*
- [ ] for changes to conversion: tested on Kaggle, with before/after results (e.g. `dev/compare_quality.py`)

## Commit messages

Use a short summary in the imperative ("Keep ceiling brackets in Marker's equation rewrites"), then a blank line
and the why, if it isn't obvious.

## License

By contributing, you agree that your contributions are licensed under the project's
[Apache License 2.0](LICENSE).
