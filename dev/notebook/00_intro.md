# 📄➡️📝 Document → Markdown **box**: Marker + vision LLMs on Kaggle's 2×T4, driven from your PC through Cloudflare

Converts **PDF (digital and scanned), images, Word (.doc/.docx/.rtf/.odt), PowerPoint (.ppt/.pptx/.odp), Excel (.xls/.xlsx/.ods), HTML, EPUB, DjVu, TXT/MD/CSV**, in any language, into Markdown that keeps **text, headings, tables, equations (LaTeX), code, figures and the information inside figures**.

| Step | What runs | Where |
|---|---|---|
| Scans first | photos and scans set upright, two facing book pages in one photo cut in two | workers (Indra, CPU) |
| Layout, OCR, reading order | [Marker](https://github.com/datalab-to/marker) + Surya 2.5; text lines the layout left out are read too | Surya on both T4s, Marker workers on the CPUs |
| Fixes without a model | the PDF's own characters, Indic vowel order, tables against the page image, drawn formulas, figures re-cropped at 300 DPI, headings from the PDF outline | workers |
| Vision-LLM checks (`standard`, `precise`) | **Qwen3-VL-8B** checks each page against its image: audit, element check, missing text and figures, figure meanings, flowcharts as Mermaid | one llama-server per T4, next to Surya |
| Office / DjVu | LibreOffice → PDF (keeps Word equations and layouts), ddjvu → PDF | workers |
| Check | every word of each PDF page's text layer is looked up in the Markdown; pages under 90% are reported | workers |

```
your PC                                 Cloudflare quick tunnel             Kaggle (this notebook, 2×T4, 4 vCPUs)
md_client.py ./docs ─────────────────▶  https://xxxx.trycloudflare.com ──▶  API ─▶ task queue (long PDFs split in page chunks)
  biggest files to the box,                                                  │
  smallest + stolen page chunks to                                           ▼
  Marker workers on your PC (optional) ◀── /vlm /llm proxies ──  up to 6 Marker workers (CPU) ─┬─▶ Surya VLM  (llama-server GPU0, GPU1)
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
python md_client.py big.pdf --quality express    # fastest: no vision-LLM pass
python md_client.py scan.pdf --quality precise    # slowest, most careful: every check on every page
python md_client.py --health                  # GPUs, workers, models, LibreOffice, idle time
```

### Quality modes (`--quality`, box default `QUALITY` in section 1)
| Mode | What the vision LLM (Qwen3-VL-8B) does | 30-page regression set on 2×T4 |
|---|---|---|
| `express` (fastest) | nothing: Marker + Surya OCR, figures re-cropped and their text read by Surya, every fix that needs no model | **233 s** (466 pages/h), all 10 checks |
| `standard` (default) | only where it pays. A page the PDF's own text settles (every word of it, only text blocks, no picture) gets **no call**. Every other page gets one short look (triage), then: plain text → **one whole-page audit**; tables, maths, code, figures, scans, forms → **element check + audit** (missing text and figures added with evidence); handwriting and hard scans → also a **finer audit**. Figures on pages with diagrams or charts are cropped to themselves and **described by meaning** (a Mermaid graph only for a flowchart); other pictures keep Surya's reading | **842 s** (128 pages/h), all 13 checks, 38 corrections |
| `precise` (slowest) | **every pass on every page**: element check, **line-by-line rewrite**, **fine audit** (3-paragraph sections), prose made explicit (the author's wording kept below each paragraph), every figure described. A page whose every word is the PDF's own text skips the rewrite | **2107 s** (51 pages/h), all 11 checks, 118 corrections |

The set has one page for each hard case: two-page book photos, upside-down and sideways photos, handwriting, degraded scans, code, tables, charts and flowcharts, CJK, Devanagari and Hebrew text (`docs/performance.md`). Every fix the vision LLM makes is kept only if it loses and invents nothing; the rest are refused and counted in `report.json`. Old mode names still work: `superfast` and `fast` run express, `balanced` and `auto` run standard, `accurate` runs precise. Vision-LLM block labels (`--labels`, `--show-layout`) cost one more call per page and only change the report's trace, never the Markdown.

**No model work is done twice:** each step runs once per page or figure. The free fixes come first (the PDF's own characters, Indic vowel order), so the vision LLM spends no call on what they fix and skips what they settle; after a model rewrote a page they check it once more, without a model. A figure's graph is read only after its description says it is a flow diagram. Identical pictures are described once. A page chunk is claimed by exactly one worker (on the box or on your PC).

**Scans first:** every page without a text layer is set upright (Indra's orientation check) and a photo of two facing book pages is cut into its two pages, before layout and OCR, in every mode. Read upside down or as one page, the OCR missed and repeated lines. `--show-layout` shows each page with the blocks the layout found and, in magenta, any text line left outside them.

Every mode keeps all the text. All three run on this box as it is: pick one per job (`--quality express`).
Settings are in section 1. The `docs/` folder of the repository explains every option (`docs/configuration.md`) and what each log line means (`docs/troubleshooting.md`).
