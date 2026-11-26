"""Conversion worker. One process per slot, kept alive between jobs so Marker's models stay loaded.
It claims document tasks from pool/tasks/ (atomic claim files, so any number of workers can share the queue) and
runs Marker's PdfConverter on them (after LibreOffice for office formats, see convert.py). The Surya VLM lives in the llama-server processes behind the load balancer;
the worker only holds the small CPU models (Indra ONNX), so a crash here never takes the GPUs down.
Usage: python worker.py pool_dir worker_id vlm_url vlm_parallel cpu_threads"""
import ctypes, gc, json, os, shutil, sys, tempfile, time, traceback
from pathlib import Path

import convert as C
import figures as FIG
import review as R
import spreads as SP

POOL, WID, VLM_URL, PARALLEL = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
CPU_THREADS = int(sys.argv[5]) if len(sys.argv) > 5 else 2
PARENT = os.getppid()
try:                                    # if RAM ever fills anyway, the kernel kills a worker (whose piece is requeued),
    with open("/proc/self/oom_score_adj", "w") as f:      # not the notebook kernel, a GPU server or the API
        f.write("1000")
except OSError:
    pass
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["SURYA_INFERENCE_BACKEND"] = "llamacpp"
if VLM_URL:
    os.environ["SURYA_INFERENCE_URL"] = VLM_URL            # attach to the running servers; never spawn one from a worker
    os.environ["SURYA_INFERENCE_PARALLEL"] = PARALLEL      # concurrent VLM requests from this worker


def log(*a):
    print(f"[{WID} {time.strftime('%H:%M:%S')}]", *a, flush=True)


def jwrite(path, obj):
    tmp = Path(str(path) + f".{WID}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False)); os.replace(tmp, path)


def claim(path):
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, WID.encode()); os.close(fd); return True
    except FileExistsError:
        return False


def shutting_down():
    return (POOL / "shutdown").exists() or os.getppid() != PARENT


RECYCLE_BYTES = float(os.environ.get("MD_WORKER_RECYCLE_GB") or 3) * 2 ** 30
RECYCLED = 75                                   # exit code pipeline.Pool relaunches without counting it as a crash


def own_rss():
    with open("/proc/self/status") as f:
        return next((int(l.split()[1]) * 1024 for l in f if l.startswith("VmRSS:")), 0)


def give_memory_back():
    """Python and onnxruntime free memory but glibc keeps it; malloc_trim returns it to the system"""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import save_output

import marker.services.openai as _marker_openai


class _LatexJson:
    """Marker parses the LLM's JSON with json.loads: LaTeX in it loses its backslashes (\\b, \\r, \\n, \\t, \\f are escapes)"""
    loads = staticmethod(R.latex_json_loads)

    def __getattr__(self, name):
        return getattr(json, name)


_marker_openai.json = _LatexJson()
LLM_REJECTED = {}                     # per piece: Marker LLM rewrites dropped, by reason
_marker_call = _marker_openai.OpenAIService.__call__


def _checked_call(self, prompt, image, block, response_schema, *a, **k):
    """every rewrite Marker's LLM steps make is checked like the page review's: a lossy one is dropped (Marker keeps its text)"""
    out = _marker_call(self, prompt, image, block, response_schema, *a, **k)
    ok, why = R.check_marker_llm(prompt, out)
    if not ok:
        LLM_REJECTED[why] = LLM_REJECTED.get(why, 0) + 1
        return {}
    return out


_marker_openai.OpenAIService.__call__ = _checked_call

t0 = time.time()
MODELS = create_model_dict()          # Indra ONNX (CPU) + a lazy client for the VLM server
MODELS["indra_model"].threads = CPU_THREADS      # onnxruntime threads: the pool's workers share the machine's cores
log(f"models ready in {time.time() - t0:.1f}s | VLM: {VLM_URL or 'none (fast / no_ocr only)'} | LibreOffice: {C.SOFFICE or 'none'}")


PICTURE_MIN_PX = 200       # smaller pictures (icons, logos) hold no text worth reading


def ocr_pictures(md, out, opts, skip, insert=True):
    """express: the text inside each picture, read by Marker with OCR forced (Surya), under the picture
    -> (pictures read, {picture: why not}, {picture: what Surya read}). insert=False (standard, precise): the Markdown is left alone, the
    readings are hints for the vision LLM's descriptions. skip: pictures the layout found text blocks in (read too, but kept
    only when mostly new: Marker drops a diagram's labels)"""
    from PIL import Image
    import re
    text, n, why = md.read_text(encoding="utf-8"), 0, {}
    names = dict.fromkeys(re.findall(r"!\[[^\]]*\]\(([^)\s]+\.(?:jpe?g|png))\)", text))
    todo = []
    for name in names:
        path = out / name
        if re.search(r"_(Formula|Table)_", name) or not path.is_file():
            continue                                  # a formula / an icon in a table
        with Image.open(path) as im:
            if min(im.size) >= PICTURE_MIN_PX:
                todo.append(name)
            else:
                why[name] = "small"
    readings = {}
    if not todo:
        return 0, why, readings

    def read(names, **marker):
        """one Marker call for all of them: one page per picture, read side by side -> {name: text}"""
        with tempfile.TemporaryDirectory() as d:
            pdf = Path(d) / "pictures.pdf"
            pages = []
            for name in names:                        # as Marker reads a lone image: native size, small ones doubled for OCR
                with Image.open(out / name) as im:
                    im = im.convert("RGB")
                    pages.append(im if max(im.size) >= 1000 else im.resize((im.width * 2, im.height * 2), Image.LANCZOS))
            pages[0].save(pdf, save_all=True, append_images=pages[1:], resolution=192)   # Marker OCRs pages at 192 dpi
            for im in pages:
                im.close()
            cli = cli_options(dict(opts, FORCE_OCR=True, IMAGES=False, PAGE_RANGE=None, PAGINATE=True,
                                   OUTPUT_FORMAT="markdown", USE_LLM=False), Path(d))
            cli.update(marker)
            cp = ConfigParser(cli)
            conv = PdfConverter(config=cp.generate_config_dict(), artifact_dict=MODELS, processor_list=cp.get_processors(),
                                renderer=cp.get_renderer(), llm_service=None)
            per_page = FIG.ocr_pages(getattr(conv(str(pdf)), "markdown", ""))
        return {name: per_page.get(k, "") for k, name in enumerate(names)}

    def place(results):
        nonlocal text, n
        for name, ocr in results.items():
            if ocr.strip():
                readings[name] = ocr
            if not insert:
                if ocr.strip():
                    why.pop(name, None)
                else:
                    why[name] = "no text read"
                continue
            text, added = FIG.add_picture_text(text, name, ocr, strict=name in skip)   # text-heavy: stricter
            n += added
            if added:
                why.pop(name, None)
            else:
                why[name] = (FIG.picture_ocr_check(text, ocr, strict=name in skip)[1] or "no text read") if ocr.strip() else "no text read"

    pages_first = [name for name in todo if "_Diagram_" not in name]
    try:
        if pages_first:
            place(read(pages_first))                  # a scan, a letter, a screenshot: read as a page
    except Exception as e:
        log("picture OCR skipped:", repr(e)[:200])
        return 0, dict(why, **{name: f"OCR failed: {type(e).__name__}" for name in pages_first}), readings
    again = [name for name in todo if "_Diagram_" in name or why.get(name) == "no text read"]   # diagrams: full-page OCR read
                                                      # nothing from any of them in the suite runs, so they go straight here
    if again:
        try:                                          # a diagram or chart: full-page OCR sees a picture and reads nothing;
            place(read(again, force_layout_block="Text", ocr_full_page=False))   # read as one block of text, its labels come out
        except Exception as e:
            log("picture OCR, second pass skipped:", repr(e)[:200])
    if n:
        md.write_text(text, encoding="utf-8")
    return n, why, readings


def cli_options(o, out_dir):
    """job options -> the same keys Marker's CLI hands to ConfigParser"""
    k = {"output_format": o.get("OUTPUT_FORMAT", "markdown"), "output_dir": str(out_dir), "mode": o["MODE"],
         "disable_multiprocessing": True,                                  # pdftext single-process: the pool already runs workers in parallel
         "disable_image_extraction": not o.get("IMAGES", True)}
    if k["output_format"] == "markdown":
        k["paginate_output"] = True           # page separators let convert.place_orphan_images put unlinked figures on their page
    for opt, key in (("FORCE_OCR", "force_ocr"), ("STRIP_EXISTING_OCR", "strip_existing_ocr"),
                     ("PAGINATE", "paginate_output"), ("KEEP_HEADERS_FOOTERS", "keep_headers_footers")):
        if o.get(opt):
            k[key] = True
    if o.get("USE_LLM") and R.quality_option(o, "REDO_INLINE_MATH"):      # asked for (--redo-math): inline maths rewritten too
        k["redo_inline_math"] = True
    if o.get("PAGE_RANGE"):
        k["page_range"] = o["PAGE_RANGE"]
    k.update(R.quality_option(o, "MARKER_EXTRA") or {})                  # every mode: Surya re-reads more of the page
    # Marker's own LLM steps: in no mode (the box's passes replace them), only with --redo-math, which needs them: Marker
    # rewrites inline maths only inside those steps (without use_llm the option did nothing)
    if o.get("USE_LLM") and (R.quality_option(o, "MARKER_LLM") or R.quality_option(o, "REDO_INLINE_MATH")):
        k["use_llm"] = True
        if o.get("LLM_URL"):                      # the box's own vision LLM (OpenAI-compatible llama-server), nothing leaves Kaggle
            k.update(llm_service="marker.services.openai.OpenAIService", openai_base_url=o["LLM_URL"], openai_model=o.get("LLM_MODEL") or "local",
                     openai_api_key="local", timeout=900, max_retries=1,
                     max_concurrency=int(o.get("MARKER_LLM_CONCURRENCY") or 2 * int(o.get("LLM_CONCURRENCY") or 2)))
        else:
            k["gemini_api_key"] = os.environ.get("GOOGLE_API_KEY", "")
    return k


def printed_numbers(document):
    """{page: the page number printed in its header or footer} (Marker leaves those out of the Markdown): which half of a
    cut spread comes first"""
    from marker.schema import BlockTypes
    types = tuple(getattr(BlockTypes, k) for k in ("PageHeader", "PageFooter") if hasattr(BlockTypes, k))
    out = {}
    for page in document.pages:
        text = "\n".join(b.raw_text(document) or "" for b in page.contained_blocks(document, types))
        n = SP.page_number(text) if text.strip() else None
        if n is not None:
            out[page.page_id] = n
    return out


def short_document(src, max_pages=2):
    """a picture or a PDF of at most 2 pages (the whole file: chunks and batches are page ranges of it)"""
    s = str(src).lower()
    if s.endswith(C.MARKER_EXT) and not s.endswith((".pdf", ".html", ".htm", ".epub")):
        return True                                       # an image: one page
    if not s.endswith(".pdf"):
        return False
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(src))
        try:
            return len(pdf) <= max_pages
        finally:
            pdf.close()
    except Exception:
        return False


def convert(t):
    out = Path(t["out"]); out.mkdir(parents=True, exist_ok=True)
    action, fmt = C.route(t["src"], t["opts"].get("OFFICE", "pdf"))
    if action == "text":
        return C.text_to_md(t["src"], out, t["stem"])
    src, scratch, via = t["src"], None, "marker"
    try:
        if action in ("soffice", "ddjvu"):
            scratch = tempfile.mkdtemp(prefix=f"conv_{WID}_", dir=str(POOL))
            src = str(C.soffice(src, fmt, scratch, POOL / f"lo_profile_{WID}") if action == "soffice" else C.djvu_to_pdf(src, scratch))
            via = f"{'libreoffice' if action == 'soffice' else 'ddjvu'}->{fmt}+marker"
        clock, last = {}, [time.time()]

        def lap(stage):                                   # seconds per stage, for the report (where does the time go?)
            now = time.time(); clock[stage] = round(clock.get(stage, 0) + now - last[0], 1); last[0] = now
        if scratch:
            lap("render_to_pdf")
        LLM_REJECTED.clear()
        legacy = C.ocr_reason(src, C.page_list(t["opts"].get("PAGE_RANGE"))) \
            if str(src).lower().endswith(".pdf") and not t["opts"].get("FORCE_OCR") else None
        if legacy:                                    # its text layer is not real text: OCR the pages, and don't trust that layer
            t = dict(t, opts=dict(t["opts"], FORCE_OCR=True))
            log(f"{legacy}: pages read with OCR")
        if not t["opts"].get("KEEP_HEADERS_FOOTERS") and short_document(src):
            t = dict(t, opts=dict(t["opts"], KEEP_HEADERS_FOOTERS=True))    # nothing repeats page to page: a header is content
        cp = ConfigParser(cli_options(t["opts"], out))
        conv = PdfConverter(config=cp.generate_config_dict(), artifact_dict=MODELS, processor_list=cp.get_processors(),
                            renderer=cp.get_renderer(), llm_service=cp.get_llm_service())
        document = conv.build_document(src)                  # = conv(src), keeping the document for the figure re-crop
        conv.page_count = len(document.pages)
        rescued, detected = None, {}                   # detected: Indra's text lines per scanned page, found once for
        indra = MODELS.get("indra_model")              # the rescue and reused by the layout listing below
        if indra is not None and VLM_URL:
            try:                                          # text the layout left in no block: read and placed too (scans)
                rescued = FIG.rescue_missed_text(document, MODELS["recognition_model"], indra.engine, seen=detected)
            except Exception as e:
                log("missed-text rescue skipped:", repr(e)[:200])
        save_output(conv.resolve_dependencies(conv.renderer)(document), str(out), t["stem"])
        lap("marker")
        extra, md = {"stage_seconds": clock}, out / f"{t['stem']}.md"
        page_trace, fig_trace, pic_trace = {}, {}, {}     # per page / figure / picture: for tuning by kind of page (report "trace")
        try:                                              # what the layout model found on each page (the page profile)
            indra = MODELS.get("indra_model")
            lo = t["opts"]
            labeler = (lambda img, boxes: FIG.ask_labels(img, boxes, lo["LLM_URL"], lo.get("LLM_MODEL") or "local")) \
                if lo.get("USE_LLM") and lo.get("LLM_URL") and (R.quality_option(lo, "LABEL_BLOCKS") or lo.get("SHOW_LAYOUT")) \
                else None                                 # --labels / --show-layout only: labels never change the Markdown
            for p, c in FIG.layout_of(document, out if lo.get("SHOW_LAYOUT") else None, lo.get("SPREADS"),
                                      detect=indra.engine.detect_all if indra is not None else None, detected=detected,
                                      labeler=labeler, workers=int(lo.get("LLM_CONCURRENCY") or 2)).items():
                page_trace.setdefault(str(p), {})["layout"] = c
        except Exception as e:
            log("layout listing skipped:", repr(e)[:200])
        if rescued:
            extra["text_rescued"] = sum(rescued.values())
            for p, n in rescued.items():
                page_trace.setdefault(str(p), {})["rescued_lines"] = n
        lap("layout_listing")
        if md.is_file() and t["opts"].get("OUTPUT_FORMAT", "markdown") == "markdown":
            if t["opts"].get("IMAGES", True) and t["opts"].get("FIGURE_FIX", True):
                try:
                    extra["figures"] = FIG.recrop_document(document, src, out, t["stem"],   # rules/headers cut without a model
                                                           peel=not R.quality_option(t["opts"], "CORE_CROP")
                                                           or bool(R.quality_option(t["opts"], "OCR_PICTURES")))
                    if extra["figures"].get("edges_peeled"):
                        extra["edges_peeled"] = extra["figures"]["edges_peeled"]   # summed into the report like the others
                except Exception as e:                        # keep Marker's own crops rather than fail the document
                    extra["figures"] = {"error": f"{type(e).__name__}: {e}"[:300]}
                    log("figure re-crop failed, kept Marker's crops:", extra["figures"]["error"])
            anchors = extra.get("figures", {}).pop("anchors", {}) if isinstance(extra.get("figures"), dict) else {}
            text_heavy = extra.get("figures", {}).pop("text_heavy", {}) if isinstance(extra.get("figures"), dict) else {}
            extra["linked_figures"] = C.place_orphan_images(md, out, keep_pages=True, anchors=anchors)   # separators: per-page review
            if t["opts"].get("IMAGES", True):
                try:                                  # a formula drawn with shapes (polygons) LaTeX can't write: a picture of it
                    why = {}
                    n = FIG.picture_formulas(md, document, src, out, skipped=why)
                    if n:
                        extra["drawn_formulas"] = n
                    if why:
                        extra["drawn_formulas_skipped"] = why
                except Exception as e:
                    log("drawn formulas skipped:", repr(e)[:200])
                extra["shrunk_images"] = R.shrink_images(out, int(t["opts"].get("IMAGE_MAX_PX") or R.IMAGE_MAX_PX))
            lap("figures")

            def repair_text():                        # no model: what the PDF's own characters settle
                if str(src).lower().endswith(".pdf") and not legacy:
                    try:                              # a few characters an OCR got wrong where the PDF's own text has the line
                        n = C.repair_from_layer(md, src)
                        if n:
                            extra["layer_fixes"] = extra.get("layer_fixes", 0) + n
                    except Exception as e:
                        log("text-layer repair skipped:", repr(e)[:200])
                if str(src).lower().endswith(".pdf"):
                    try:                              # Indic vowel signs in visual order: fixed from the PDF's own text
                        n = C.repair_indic(md, src)
                        if n:
                            extra["indic_words_fixed"] = extra.get("indic_words_fixed", 0) + n
                    except Exception as e:
                        log("Indic repair skipped:", repr(e)[:200])
            repair_text()                             # before any model pass: no vision-LLM call (or room in its 800-token audit
                                                      # answer) is spent on what the PDF's own text fixes, and the pages and
                                                      # sections it settles are skipped by the audit, line pass and enhancement
            o, llm = t["opts"], t["opts"].get("USE_LLM") and t["opts"].get("LLM_URL")
            q = lambda key: R.quality_option(o, key)
            conc, model = int(o.get("LLM_CONCURRENCY") or 2), o.get("LLM_MODEL") or "local"
            R.set_llm_inflight(int(o.get("LLM_INFLIGHT") or 0))  # one worker may fill every LLM slot, never the queue
            extra["quality"] = R.canonical_quality(o.get("QUALITY")) or R.DEFAULT_QUALITY

            def describe():
                try:                                  # the Markdown is written after the page review (they run at the same time)
                    return FIG.describe_figures(md, out, o["LLM_URL"], model, conc, text_heavy=text_heavy, tighten=bool(q("CORE_CROP")),
                                                graph_tries=int(q("GRAPH_TRIES")), graph_checks=int(q("GRAPH_CHECKS")), defer=True,
                                                transcribe_text=bool(q("TRANSCRIBE_TEXT_IMAGES")))
                except Exception as e:
                    log("figure descriptions failed:", repr(e)[:300])
                    return 0, None

            audit_stats, sizes = {}, tuple(q("AUDIT_SECTIONS") or (R.AUDIT_SECTION_PARAS,))
            audit = (lambda text, jpeg, layer: R.audit_page(text, jpeg, layer, o["LLM_URL"], model, audit_stats, sizes)) \
                if llm and q("AUDIT") else None

            def review():
                if not q("PAGE_REVIEW"):              # fast: no rewrite, only the audit (still side by side with the figures)
                    try:
                        audit_stats.update(R.audit_pages(md, src, o["LLM_URL"], model, conc, use_layer=not legacy, sizes=sizes,
                                                         inspect=bool(q("INSPECT")), add_figure=add_figure,
                                                         line_review=int(q("LINE_REVIEW_TOKENS") or 0),
                                                         ctx=int(o.get("LLM_CTX") or 8192),
                                                         keep_headers=bool(o.get("KEEP_HEADERS_FOOTERS")),
                                                         skip_settled=bool(q("REVIEW_SKIP_SETTLED")),
                                                         enhance=bool(q("ENHANCE")), triage=bool(q("TRIAGE")),
                                                         decisions=decisions, trace=page_trace))
                    except Exception as e:
                        log("audit skipped:", repr(e)[:300])
                    return None
                try:                                  # every page checked against its image and rewritten (only if nothing is lost),
                    return R.review_pages(md, src, o["LLM_URL"], model, conc, int(o.get("LLM_CTX") or 8192),   # then audited
                                          keep_headers=bool(o.get("KEEP_HEADERS_FOOTERS")), use_layer=not legacy,
                                          part_tokens=int(q("REVIEW_PART_TOKENS")), after=audit,
                                          skip_settled=bool(q("REVIEW_SKIP_SETTLED")))
                except Exception as e:
                    log("page review failed, kept Marker's pages:", repr(e)[:300])
            def crop_missing(page_id, bbox, k):       # a figure the inspection saw on the page but the Markdown lacks
                try:
                    return FIG.crop_missing_figure(src, page_id, bbox, out, k)
                except Exception as e:
                    return None, f"crop failed: {type(e).__name__}"
            add_figure = crop_missing if (q("INSPECT") or q("TRIAGE")) and o.get("IMAGES", True) and str(src).lower().endswith(".pdf") else None
            decisions = {}                            # standard: {page: its level, and why}
            describe_later = llm and q("DESCRIBE_FIGURES") and q("OCR_PICTURES")   # after the inspection, with Surya's
                                                      # readings as hints, so figures it added are described too

            def crop_figures():                       # no descriptions (--no-describe), but the vision LLM checks every crop
                import re as _re
                from concurrent.futures import ThreadPoolExecutor as _Pool
                names = [n for n in dict.fromkeys(_re.findall(r"!\[[^\]]*\]\(([^)\s]+\.(?:jpe?g|png))\)", md.read_text(encoding="utf-8")))
                         if not _re.search(r"_(Formula|Table)_", n) and (out / n).is_file()]
                with _Pool(max(1, conc)) as px:
                    return sum(bool(x) for x in px.map(lambda n: FIG.tighten_figure(out / n, o["LLM_URL"], model), names))

            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(3) as ex:         # all only wait for the LLM: side by side, the LLM's slots stay full
                fc = ex.submit(crop_figures) if llm and q("CORE_CROP") and not q("DESCRIBE_FIGURES") else None
                fd = ex.submit(describe) if llm and q("DESCRIBE_FIGURES") and not describe_later else None
                fr = ex.submit(review) if llm and (q("PAGE_REVIEW") or audit) else None
                if fr is not None and fr.result():
                    extra["page_review"] = fr.result()
                if fd is not None:
                    n_desc, put = fd.result()
                    if put:
                        put(md)
                    extra["described_figures"] = n_desc
                if fc is not None:
                    try:
                        extra["figures_cropped_by_llm"] = fc.result()
                    except Exception as e:
                        log("figure crop check skipped:", repr(e)[:200])
            lap("describe_review_audit")
            if audit:
                extra["audit"] = R._public(audit_stats)
            if decisions:                             # standard: the level each page got
                extra["page_levels"] = {str(p): d["level"] for p, d in sorted(decisions.items())}
            lines = {k: v for k, v in audit_stats.items() if k.startswith("line review")}
            if lines and "page_review" not in extra:  # precise's line pass reported like the page review: pages rewritten
                extra["page_review"] = {"pages": sum(lines.values()), "rewritten": lines.get("line review: rewritten", 0),
                                        "kept_draft": {k.split("(", 1)[1].rstrip(")"): v for k, v in lines.items() if "kept draft" in k},
                                        "examples": []}
            if describe_later and o.get("IMAGES", True):
                readings = {}
                try:                                  # Surya reads every picture first (labels, values, a page's text)
                    _, skipped, readings = ocr_pictures(md, out, o, text_heavy, insert=False)
                    pic_trace.update({n: {"words": len(r.split())} for n, r in readings.items()})
                    if skipped:
                        extra["pictures_skipped"] = skipped
                except Exception as e:
                    log("picture OCR skipped:", repr(e)[:200])
                lap("picture_ocr")
                try:                                  # then the vision LLM writes what each figure MEANS, Surya's reading as
                    extra["described_figures"] = FIG.describe_figures(       # evidence; Mermaid only for a real diagram
                        md, out, o["LLM_URL"], model, conc, text_heavy=text_heavy, tighten=bool(q("CORE_CROP")),
                        graph_tries=int(q("GRAPH_TRIES")), graph_checks=int(q("GRAPH_CHECKS")), hints=readings, trace=fig_trace,
                        pages=None if not q("TRIAGE") else {p for p, d in decisions.items()    # standard: only pages with
                                                            if not d.get("page") or d["page"].get("diagrams")   # diagrams or charts
                                                            or d["page"].get("charts")}
                        if decisions else None)
                except Exception as e:
                    log("figure descriptions failed:", repr(e)[:300])
                import re
                text, kept_text, fallback = md.read_text(encoding="utf-8"), 0, 0
                for name, ocr in readings.items():  # nothing Surya read is lost: a page's text under its description, and
                    text, added = FIG.append_ocr_text(text, name, ocr)       # Surya's reading where no description came
                    kept_text += added
                    if not added and not re.search(r"\(" + re.escape(name) + r"\)\s*\n+> \*\*Figure content", text):
                        text, added = FIG.add_picture_text(text, name, ocr, strict=name in text_heavy)
                        fallback += added
                md.write_text(text, encoding="utf-8")
                extra["pictures_read"] = len(readings)
                if kept_text:
                    extra["picture_text_kept"] = kept_text
                if fallback:
                    extra["pictures_ocr_only"] = fallback
                lap("describe_figures")
            elif q("OCR_PICTURES") and o.get("IMAGES", True):
                try:                                  # express: no description, but the text inside pictures, read by Surya
                    extra["pictures_read"], skipped, readings = ocr_pictures(md, out, o, text_heavy)
                    pic_trace.update({n: {"words": len(r.split())} for n, r in readings.items()})
                    if skipped:
                        extra["pictures_skipped"] = skipped   # {picture: why it got no OCR text}
                except Exception as e:
                    log("picture OCR skipped:", repr(e)[:200])
                lap("picture_ocr")
            if fr is not None:                        # a model pass rewrote page text: a near-spelling accept() lets through
                repair_text()                         # ("resistor" -> "resistar") goes back to the PDF's characters
            try:                                      # tables against the page image: columns, pictures in cells
                for key, n in FIG.fix_tables(md, document, src, out, t["stem"]).items():
                    if n:
                        extra[key] = n
            except Exception as e:
                extra["tables_error"] = repr(e)[:200]
                log("table check skipped:", repr(e)[:200])
            try:                                      # heading levels from the PDF's outline (bookmarks), when it has one
                extra["outline_headings"] = R.apply_outline(md, R.outline(src))
            except Exception as e:
                log("outline headings skipped:", repr(e)[:200])
            if t["opts"].get("SPREADS"):              # two-page photos the job cut in two: halves in printed order, original pages
                spread_map = t["opts"]["SPREADS"]
                try:
                    numbers = printed_numbers(document)
                except Exception as e:                # the Markdown's own lines are looked at instead
                    numbers = {}
                    log("page headers not read:", repr(e)[:200])
                try:
                    text, n = SP.finish(md.read_text(encoding="utf-8"), spread_map, numbers)
                    md.write_text(text, encoding="utf-8")
                    if n:
                        extra["spreads_reordered"] = n
                    merged = SP.merge_trace(page_trace, spread_map)
                    page_trace.clear(); page_trace.update(merged)
                    if extra.get("page_levels"):
                        extra["page_levels"] = {str(spread_map[int(k)]): v for k, v in reversed(list(extra["page_levels"].items()))}
                except Exception as e:
                    log("spread pages left as cut:", repr(e)[:200])
            R.finish_markdown(md, keep_pages=bool(t["opts"].get("PAGINATE")))
        if legacy:
            extra["forced_ocr"] = f"{legacy}: read with OCR"
        if md.is_file():
            extra["trace"] = {"chunks": [{"pages": t["opts"].get("PAGE_RANGE") or "all", "worker": WID, "quality": t["opts"].get("QUALITY"),
                                          "forced_ocr": bool(legacy), "stage_seconds": dict(clock)}],
                              "pages": page_trace, "figures": fig_trace, "pictures": pic_trace}
        if LLM_REJECTED:
            extra["marker_llm_rejected"] = dict(LLM_REJECTED)
        if md.is_file() and str(src).lower().endswith(".pdf") and not legacy:
            try:
                extra["coverage"] = FIG.text_coverage(src, md.read_text(encoding="utf-8"), [p.page_id for p in document.pages])
            except Exception as e:
                log("coverage check failed:", repr(e)[:300])
        return {"pages": conv.page_count, "files": sorted(p.name for p in out.iterdir()), "via": via, **extra}
    finally:
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def run_task(t):
    tid = t["tid"]
    t1 = time.time()
    jwrite(POOL / "running" / f"{tid}.json", {"wid": WID, "t0": t1, "name": t["rel"]})
    try:
        if (POOL / "cancel" / t["job"]).exists():
            res = {"ok": False, "error": "cancelled"}
        else:
            res = {"ok": True, **convert(t)}
    except BaseException as e:
        if isinstance(e, (KeyboardInterrupt, SystemExit)):
            raise
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"[:600], "trace": traceback.format_exc()[-1500:]}
        log("task", tid, t["rel"], "FAILED:", res["error"])
    res.update(tid=tid, job=t["job"], doc=t["doc"], rel=t["rel"], worker=WID, sec=round(time.time() - t1, 2))
    jwrite(POOL / "results" / f"{tid}.json", res)
    give_memory_back()
    log("task", tid, t["rel"], "ok" if res["ok"] else "failed", f"{res['sec']}s", f"{res.get('pages')} pages", f"{own_rss() / 2 ** 30:.1f} GB")
    if own_rss() > RECYCLE_BYTES:
        log(f"memory {own_rss() / 2 ** 30:.1f} GB after this piece: restarting to give it back")
        sys.exit(RECYCLED)


def anything_running():
    """a piece some worker is converting right now (running file without a result)"""
    return any(not (POOL / "results" / rp.name).exists() for rp in (POOL / "running").glob("*.json"))


def take_solo():
    """RAM is low but nothing runs: one worker (the one that gets this token) may still convert, so the box never deadlocks"""
    try:
        fd = os.open(str(POOL / "solo"), os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        return False


def free_task_waiting():
    """a piece nobody has claimed or finished (task files stay after their result is written)"""
    for tp in (POOL / "tasks").glob("*.json"):
        if not (POOL / "results" / tp.name).exists() and not (POOL / "tasks" / f"{tp.stem}.claim").exists():
            return True
    return False


SLOT = None                                      # the start slot this worker holds (handed back if no piece was taken)


def may_start():
    """a free start slot (pipeline.start_budget: free RAM above the pause level / what a starting piece grows by). Slots are
    files made exclusively, so two workers can't take the same one; the pool frees each after START_WINDOW_S. True when no
    budget is published (old pool, tests)."""
    try:
        slots = int(json.loads((POOL / "admit.json").read_text())["slots"])
    except Exception:
        return True
    (POOL / "starts").mkdir(exist_ok=True)
    for k in range(slots):
        try:
            fd = os.open(str(POOL / "starts" / f"slot_{k}"), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode()); os.close(fd)
            global SLOT
            SLOT = POOL / "starts" / f"slot_{k}"
            return True
        except FileExistsError:
            continue
    return False


def next_task():
    """claims the oldest free piece; None if there is none"""
    for tp in sorted((POOL / "tasks").glob("*.json")):
        if shutting_down():
            return None
        tid = tp.stem
        if (POOL / "results" / f"{tid}.json").exists() or not claim(POOL / "tasks" / f"{tid}.claim"):
            continue
        return json.loads(tp.read_text())
    return None


BASE_RSS = own_rss()                             # this worker with its models loaded and nothing else
jwrite(POOL / f"alive_{WID}.json", {"pid": os.getpid(), "vlm": bool(VLM_URL)})
idle_since = time.time()


def shrink_if_idle(why):
    """a worker idle for 20 s that grew past its loaded size gives the RAM back, or restarts to (models reload on start):
    idle workers kept their last piece's memory, and 6 of them left too little for the next job (box restarted, run2)"""
    if time.time() - idle_since > 20 and own_rss() > max(1.5 * BASE_RSS, BASE_RSS + 0.7 * 2 ** 30):
        give_memory_back()
        if own_rss() > max(1.5 * BASE_RSS, BASE_RSS + 0.7 * 2 ** 30):
            log(f"{why} at {own_rss() / 2 ** 30:.1f} GB: restarting to give it back")
            sys.exit(RECYCLED)


while not shutting_down():
    solo = False
    if (POOL / "pause").exists():                # RAM is low (pipeline.guard_memory)
        if anything_running() or not take_solo():                  # someone is converting: wait for it to finish and free memory
            shrink_if_idle("RAM low and idle")
            time.sleep(1)
            continue
        solo = True                              # nothing runs: this worker may convert one piece
    elif not free_task_waiting():
        shrink_if_idle("idle")                   # the job is over (or others hold its last pieces): free what this one kept
        time.sleep(0.4)                          # nothing to take: no start slot used up
        continue
    elif anything_running() and not may_start():
        time.sleep(1)                            # RAM for one more piece is not there yet: wait (never when nothing runs)
        continue
    try:
        t = next_task()
        if not t and SLOT is not None:           # another worker took it: the slot goes back at once
            SLOT.unlink(missing_ok=True)
        SLOT = None
        if t:
            run_task(t)
            idle_since = time.time()
    finally:
        if solo:
            (POOL / "solo").unlink(missing_ok=True)
    if not t:
        time.sleep(0.4)
log("shutdown")
