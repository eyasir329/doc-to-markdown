# 📄➡️📝 Document → Markdown **box**: Marker + vision LLMs on Kaggle's 2×T4, driven from your PC through Cloudflare

Converts **PDF (digital and scanned), images, Word (.doc/.docx/.rtf/.odt), PowerPoint (.ppt/.pptx/.odp), Excel (.xls/.xlsx/.ods), HTML, EPUB, DjVu, TXT/MD/CSV**, in any language, into Markdown that keeps **text, headings, tables, equations (LaTeX), code, figures and the information inside figures**.

| Step | What runs | Where |
|---|---|---|
| Layout, OCR, reading order | [Marker](https://github.com/datalab-to/marker) + Surya 2.5 VLM | Surya on both T4s, Marker workers on the 4 CPUs |
| Tables (incl. across pages), equations, inline math, forms, headings, handwriting | Marker's LLM passes with **Qwen3-VL-8B** | one llama-server per T4, next to Surya |
| Figures | re-cropped from the page (complete labels, joined pieces, 300 DPI); images the layout model missed are recovered from the PDF; the LLM writes out each figure's content under it | workers + LLM |
| Office / DjVu | LibreOffice → PDF (keeps Word equations and layouts), ddjvu → PDF | workers |
| Check | every word of each PDF page's text layer is looked up in the Markdown; pages under 90% are reported | workers |

```
your PC                                 Cloudflare quick tunnel             Kaggle (this notebook, 2×T4, 4 vCPUs)
md_client.py ./docs ─────────────────▶  https://xxxx.trycloudflare.com ──▶  API ─▶ task queue (long PDFs split in page chunks)
  biggest files to the box,                                                  │
  smallest + stolen page chunks to                                           ▼
  Marker workers on your PC (optional) ◀── /vlm /llm proxies ──  4 warm Marker workers (CPU) ──┬─▶ Surya VLM  (llama-server GPU0, GPU1)
        ◀──── <name>/<name>.md + images + report.json ─────────────────────────────────────────┴─▶ vision LLM (llama-server GPU0, GPU1)
```

### One-time setup (Kaggle)
1. **Settings → Accelerator → `GPU T4 ×2`**, **Settings → Internet → On**.
2. **Add-ons → Secrets** (tick *attached*): `BOX_TOKEN` (16+ characters; the same one goes into `.box_token` on your PC), optional `HF_TOKEN`, optional `GOOGLE_API_KEY` (Gemini; only used if the local LLM cannot start, and it sends page content to Google).
3. **Save Version → Save & Run All (Commit)** and wait for **BOX READY** (about 10 minutes: llama.cpp's released CUDA build and the models are downloaded, LibreOffice is installed, every GPU server must pass a test request).
4. The box stops by itself after `IDLE_STOP_MIN` minutes without jobs, to save the 30 GPU-hours/week.

### On your PC
Put `md_client.py` (with the `dev/` folder next to it) and a file `.box_token` in one folder, then:
```
python md_client.py book.pdf                  # -> markdown_out/book/book.md + images + report.json
python md_client.py ./folder --out ./md       # every document in the folder (sub-folders kept)
python md_client.py big.pdf                   # long documents: batches of pages, each saved on your PC before the next (re-run resumes)
python md_client.py scan.pdf --mode vllm      # scans / illustrated books: Surya reads every page and finds every figure
python md_client.py big.pdf --no-llm          # much faster, less exact tables/equations, no figure text
python md_client.py --health                  # GPUs, workers, models, LibreOffice, idle time
```
Settings are in section 1. The `docs/` folder of the repository explains every option (`docs/configuration.md`) and what each log line means (`docs/troubleshooting.md`).
