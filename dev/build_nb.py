"""Builds the Kaggle notebook from its sources. Usage: python dev/build_nb.py [out.ipynb]   (default: docs_to_markdown_cloudflare.ipynb)

dev/notebook/NN_name.md / .py   one notebook cell each, in file-name order
dev/mdbox/*.py                  the box modules, written to disk on Kaggle by %%writefile cells placed after 09_box_files.md"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "docs_to_markdown_cloudflare.ipynb"
MODULES = ["convert.py", "figures.py", "review.py", "lb.py", "worker.py", "pipeline.py", "server.py"]      # dependencies first


def cell(kind, text):
    lines = text.rstrip("\n").split("\n")
    c = {"cell_type": kind, "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}
    if kind == "code":
        c.update(outputs=[], execution_count=None)
    return c


cells = []
for f in sorted((HERE / "notebook").iterdir()):
    if f.suffix not in (".md", ".py"):
        continue
    cells.append(cell("markdown" if f.suffix == ".md" else "code", f.read_text(encoding="utf-8")))
    if f.name.startswith("09_"):
        cells += [cell("code", f"%%writefile mdbox/{m}\n" + (HERE / "mdbox" / m).read_text(encoding="utf-8")) for m in MODULES]

nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
      "metadata": {"accelerator": "GPU", "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}}}
OUT.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT, "with", len(cells), "cells")
