"""Re-crops the figures Marker extracted, from the page itself, so they are complete.

The layout model's boxes are often too tight (axis labels, legends, side text cut off) or split one figure into pieces
(each die of a dice picture, each arrow of a diagram). For every page with figures:
  1. render the page at ANALYSIS_DPI and find "ink" (dark or coloured pixels; light scan noise removed)
  2. drop "figures" that are really a caption; trim strips of body text a layout box reaches into at its edges; blank out
     body-text paragraphs and captions so a figure never grows into the text around it
  3. join ink that is closer than GAP_PT and grow each figure box over the ink connected to it (at most MAX_GROW_PT per side)
  4. merge figure boxes that overlap after growing (pieces of one figure share ink): one figure, one image
  5. crop the final box (+PAD_PT) from a CROP_DPI render and overwrite Marker's file; merged-away pieces are deleted
On PDFs it also adds the figures the layout model missed: every raster image the PDF draws (pdfium knows the exact
position) that no figure covers becomes a figure too (_page_N_Picture_9000+k), placed right before its caption in the
Markdown when it has one (convert.place_orphan_images), else at the end of its page. On one digital book the layout model
had missed photos on 107 of 282 pages that have images.
The core (refine_page) works on plain arrays and boxes in PDF points, so it is tested without Marker (dev/test_pipeline.py)."""
import re
from pathlib import Path

import numpy as np
from PIL import Image

ANALYSIS_DPI = 150
CROP_DPI = 300
GAP_PT = 7.0                 # ink closer than this belongs together (strokes, labels next to a chart)
MAX_GROW_PT = 90.0           # a box grows at most this far on each side (keeps a figure from walking across the page)
PAD_PT = 3.0
LONG_TEXT = 60               # text blocks with at least this many characters are body text: figures never grow into them
JPEG_QUALITY = 95
MIN_IMAGE_PT = 30            # smaller PDF images are bullets/icons, not figures


def ink_mask(img):
    """dark or saturated pixels (pale yellow fills count, light-grey scan noise does not)"""
    a = np.asarray(img.convert("RGB"), dtype=np.int32)
    gray = (a[..., 0] * 299 + a[..., 1] * 587 + a[..., 2] * 114) // 1000
    sat = a.max(axis=2) - a.min(axis=2)
    m = (gray < 225) | ((sat > 45) & (gray < 245))         # light-grey text and pale fills count
    from scipy import ndimage
    lab, n = ndimage.label(m, structure=np.ones((3, 3), bool))
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    return m & (sizes[lab] >= 4)                            # drop isolated scan specks; thin strokes stay


def _union(a, b):
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _near(a, b, gap):
    return a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def trim_box(fb, walls, frac=0.3):
    """cuts off a strip of body text / caption that a too-generous layout box reaches into at one of its edges"""
    x0, y0, x1, y1 = fb
    for tb in walls:
        if _inter((x0, y0, x1, y1), tb) <= 0:
            continue
        w, h = x1 - x0, y1 - y0
        across = min(x1, tb[2]) - max(x0, tb[0]) > 0.3 * w
        down = min(y1, tb[3]) - max(y0, tb[1]) > 0.3 * h
        if across and tb[1] <= y0 < tb[3] < y1 and tb[3] - y0 < frac * h:
            y0 = tb[3]
        elif across and y0 < tb[1] < y1 <= tb[3] and y1 - tb[1] < frac * h:
            y1 = tb[1]
        elif down and tb[0] <= x0 < tb[2] < x1 and tb[2] - x0 < frac * w:
            x0 = tb[2]
        elif down and x0 < tb[0] < x1 <= tb[2] and x1 - tb[0] < frac * w:
            x1 = tb[0]
    return (x0, y0, x1, y1)


def refine_page(ink, page_size, figs, texts):
    """ink: bool array (H, W) of the page; page_size: (w, h) in points; figs: [(name, bbox)] in reading order;
    texts: [(bbox, n_chars[, kind])] of the other blocks, kind "caption" or "text". Boxes are (x0, y0, x1, y1) in points.
    -> ({name: bbox}, {name: kept_name, or None if the "figure" was only a caption and is dropped})"""
    from scipy import ndimage
    h, w = ink.shape
    sx, sy = w / page_size[0], h / page_size[1]
    to_px = lambda b: (max(0, int(b[0] * sx)), max(0, int(b[1] * sy)), min(w, int(np.ceil(b[2] * sx))), min(h, int(np.ceil(b[3] * sy))))
    walls = [t[0] for t in texts if t[1] >= LONG_TEXT or (len(t) > 2 and t[2] == "caption")]     # body text and captions
    captions = [t[0] for t in texts if len(t) > 2 and t[2] == "caption"]
    dropped = {n: None for n, fb in figs if any(_inter(fb, c) >= 0.8 * _area(fb) for c in captions)}
    body = [t[0] for t in texts if t[1] >= LONG_TEXT]
    inside = lambda inner, outer: _area(inner) < _area(outer) and _inter(inner, outer) >= 0.8 * _area(inner)
    for n, fb in figs:            # a box around a whole screenshot/page (other figures and body text inside it) is not a figure:
        if (any(m != n and inside(ob, fb) for m, ob in figs)          # its figures are cropped on their own, its text is text
                and any(inside(tb, fb) for tb in body)):
            dropped[n] = None
    figs = [(n, trim_box(fb, walls)) for n, fb in figs if n not in dropped]
    ink = ink.copy()
    for tb in walls:                                                       # walls stop growth, except inside a figure box
        x0, y0, x1, y1 = to_px(tb)
        keep = ink[y0:y1, x0:x1].copy()
        ink[y0:y1, x0:x1] = False
        for _, fb in figs:
            fx0, fy0, fx1, fy1 = to_px(fb)
            ix0, iy0, ix1, iy1 = max(x0, fx0), max(y0, fy0), min(x1, fx1), min(y1, fy1)
            if ix0 < ix1 and iy0 < iy1:
                ink[iy0:iy1, ix0:ix1] = keep[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0]
    g = max(1, int(round(GAP_PT * sx)))
    labels, _ = ndimage.label(ndimage.binary_dilation(ink, structure=np.ones((g, g), bool)))
    labels = np.where(ink, labels, 0)
    objs = ndimage.find_objects(labels)
    boxes = {}
    for name, fb in figs:
        x0, y0, x1, y1 = to_px(fb)
        limit = (fb[0] - MAX_GROW_PT, fb[1] - MAX_GROW_PT, fb[2] + MAX_GROW_PT, fb[3] + MAX_GROW_PT)
        out = tuple(fb)
        for lab in np.unique(labels[max(0, y0 - g):y1 + g, max(0, x0 - g):x1 + g]):     # ink touching the box counts too
            if lab == 0 or objs[lab - 1] is None:
                continue
            ys, xs = objs[lab - 1]
            cb = (xs.start / sx, ys.start / sy, xs.stop / sx, ys.stop / sy)
            if cb[0] >= limit[0] and cb[1] >= limit[1] and cb[2] <= limit[2] and cb[3] <= limit[3]:
                out = _union(out, cb)
        boxes[name] = out
    merged, order = {}, [n for n, _ in figs]
    changed = True
    while changed:                                                         # pieces of one figure become one figure
        changed = False
        alive = [n for n in order if n not in merged]
        for i, a in enumerate(alive):
            for b in alive[i + 1:]:
                if _near(boxes[a], boxes[b], 0):          # overlap only: pieces sharing ink grow into each other, separate panels do not
                    boxes[a] = _union(boxes[a], boxes[b]); merged[b] = a; changed = True
                    break
            if changed:
                break
    for b, a in list(merged.items()):                                      # follow chains: c -> b -> a
        while a in merged:
            a = merged[a]
        merged[b] = a
    final = {n: (max(0.0, b[0] - PAD_PT), max(0.0, b[1] - PAD_PT), min(page_size[0], b[2] + PAD_PT), min(page_size[1], b[3] + PAD_PT))
             for n, b in boxes.items() if n not in merged}
    return final, {**merged, **dropped}


def pdf_image_boxes(pdf_page):
    """boxes (top-left origin, points) of the raster images a PDF page draws, without tiny icons and full-page backgrounds"""
    import pypdfium2.raw as pc
    w, h = pdf_page.get_size()
    l0, b0 = pdf_page.get_cropbox()[:2]
    out = []
    for o in pdf_page.get_objects(filter=(pc.FPDF_PAGEOBJ_IMAGE,), max_depth=3):
        l, b, r, t = (getattr(o, "get_bounds", None) or o.get_pos)()       # pypdfium2 5 / 4
        box = (max(0.0, l - l0), max(0.0, h - (t - b0)), min(w, r - l0), min(h, h - (b - b0)))
        bw, bh = box[2] - box[0], box[3] - box[1]
        if bw >= MIN_IMAGE_PT and bh >= MIN_IMAGE_PT and bw * bh < 0.85 * w * h:
            out.append(box)
    return out


def missing_images(img_boxes, fig_boxes, cover=0.5):
    """the images no figure covers at least half of"""
    return [b for b in img_boxes if not any(_inter(b, f) >= cover * _area(b) for f in fig_boxes)]


def text_share(box, text_boxes):
    """the share of a box that text blocks cover (each counted once, overlaps between them ignored)"""
    return min(1.0, sum(_inter(box, t) for t in text_boxes) / max(1e-6, _area(box)))


def caption_for(box, captions, max_gap=40.0):
    """text of the caption right below (else right above) a figure box, or None. captions: [(bbox, text)]"""
    best = None
    for cb, text in captions:
        if min(box[2], cb[2]) - max(box[0], cb[0]) < 0.3 * min(box[2] - box[0], cb[2] - cb[0]):
            continue
        gap = cb[1] - box[3] if cb[1] >= box[3] - 5 else (box[1] - cb[3]) + max_gap / 2      # below is preferred
        if -5 <= gap <= max_gap and (best is None or gap < best[0]):
            best = (gap, text)
    return best[1] if best else None


def blank_labels(crop, fb, labels, k, pad=6.0):
    """paints over the caption/figure label that a figure's box holds (it is text in the Markdown already: no second copy
    in the picture). crop: PIL image of box fb (points) at k pixels per point; labels: boxes in points -> how many.
    pad: a label's box is the text's; the coloured tab around it ("FIGURE 5.3" in a pink lozenge) reaches ~5 pt further"""
    from PIL import ImageDraw
    n = 0
    for lb in labels:
        if _area(lb) <= 0 or _inter(lb, fb) < 0.9 * _area(lb) or _area(lb) > 0.25 * _area(fb):
            continue
        x0, y0 = int((lb[0] - pad - fb[0]) * k[0]), int((lb[1] - pad - fb[1]) * k[1])
        x1, y1 = int(np.ceil((lb[2] + pad - fb[0]) * k[0])), int(np.ceil((lb[3] + pad - fb[1]) * k[1]))
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(crop.width, x1), min(crop.height, y1)
        if x1 <= x0 or y1 <= y0:
            continue
        a = np.asarray(crop.convert("RGB"))
        ring = np.concatenate([a[max(0, y0 - 2), x0:x1], a[min(crop.height - 1, y1 + 1), x0:x1],
                               a[y0:y1, max(0, x0 - 2)], a[y0:y1, min(crop.width - 1, x1 + 1)]])
        light = ring[ring.min(axis=1) > 200]                  # the background around it (not the figure's own ink)
        color = tuple(int(c) for c in np.median(light if len(light) else np.array([[255, 255, 255]]), axis=0))
        ImageDraw.Draw(crop).rectangle([x0, y0, x1 - 1, y1 - 1], fill=color)
        n += 1
    return n


def drop_refs(md, names):
    """removes the image links of merged-away pieces and dropped captions"""
    for n in names:
        md = re.sub(r"!\[[^\]]*\]\(" + re.escape(n) + r"\)\n?", "", md)
    return md


# ----------------------------------------------------------------- the Marker side (runs in worker.py)
def recrop_document(document, src, out_dir, stem):
    """returns {"pages", "figures", "merged", "dropped_captions"} counts; files that Marker did not write are left alone"""
    from marker.schema import BlockTypes
    from marker.settings import settings
    ext = settings.OUTPUT_IMAGE_FORMAT.lower()
    fig_types = tuple(getattr(BlockTypes, t) for t in ("Picture", "Figure", "Diagram") if hasattr(BlockTypes, t))
    text_types = tuple(getattr(BlockTypes, t) for t in ("Text", "SectionHeader", "ListItem", "Caption", "Footnote", "PageHeader", "PageFooter",
                                                        "Table", "TableOfContents", "Form", "Code", "Equation", "Handwriting") if hasattr(BlockTypes, t))
    content_types = tuple(getattr(BlockTypes, t) for t in ("Table", "TableOfContents", "Form", "Equation", "Code") if hasattr(BlockTypes, t))
    out_dir = Path(out_dir)
    pdf = None
    if str(src).lower().endswith(".pdf"):
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(src))
    stats = {"pages": 0, "figures": 0, "merged": 0, "dropped_captions": 0, "added_from_pdf": 0, "anchors": {}, "text_heavy": {}}
    all_merged = []
    try:
        for page in document.pages:
            figs = [(f"{b.id.to_path()}.{ext}", tuple(b.polygon.bbox)) for b in page.contained_blocks(document, fig_types)]
            figs = [f for f in figs if (out_dir / f[0]).is_file()]
            blocks = page.contained_blocks(document, text_types)
            added = []
            if pdf is not None and page.page_id < len(pdf):
                # an image Marker already turned into a table / form / equation / code is covered too (no duplicate picture)
                covers = [fb for _, fb in figs] + [tuple(b.polygon.bbox) for b in blocks if b.block_type in content_types]
                text_boxes = [tuple(b.polygon.bbox) for b in blocks if b.block_type != BlockTypes.Caption]
                added = [(f"_page_{page.page_id}_Picture_{9000 + k}.{ext}", b)
                         for k, b in enumerate(missing_images(pdf_image_boxes(pdf[page.page_id]), covers))
                         if text_share(b, text_boxes) < 0.25]       # a screenshot/scan whose content became text: not a figure
            figs += added
            if not figs:
                continue
            texts = [(tuple(b.polygon.bbox), len(b.raw_text(document) or ""), "caption" if b.block_type == BlockTypes.Caption else "text")
                     for b in blocks]
            captions = [(tuple(b.polygon.bbox), b.raw_text(document) or "") for b in blocks if b.block_type == BlockTypes.Caption]
            from review import CAPTION                    # "FIGURE 5.3" tabs and captions printed inside a figure's box
            labels = [cb for cb, _ in captions] + [tuple(b.polygon.bbox) for b in blocks if b.block_type != BlockTypes.Caption
                                                   and len((b.raw_text(document) or "").strip()) <= 30
                                                   and CAPTION.fullmatch((b.raw_text(document) or "").strip())
                                                   and re.search(r"[^\W\d_]", b.raw_text(document) or "")]     # not a bare number (an axis value)
            labels += [tuple(b.polygon.bbox) for b in blocks          # the caption's words printed right next to its label
                       if b.block_type in (BlockTypes.Text, BlockTypes.Caption) and tuple(b.polygon.bbox) not in labels
                       and len(b.raw_text(document) or "") <= 300 and any(_near(tuple(b.polygon.bbox), lb, 8) for lb in labels)]
            pw, ph = page.polygon.width, page.polygon.height
            if pdf is not None:
                pg = pdf[page.page_id]
                render = lambda dpi: pg.render(scale=dpi / 72).to_pil()
            else:
                hi = page.get_image(highres=True)
                render = lambda dpi: hi
            small = render(ANALYSIS_DPI)
            boxes, merged = refine_page(ink_mask(small), (pw, ph), figs, texts)
            big = render(CROP_DPI).convert("RGB")
            kx, ky = big.width / pw, big.height / ph
            long_text = [tb for tb, n, kind in texts if kind == "text" and n >= 20]
            for name, fb in boxes.items():
                share = text_share(fb, long_text)
                if share >= 0.4:
                    stats["text_heavy"][name] = round(share, 2)
            for name, (x0, y0, x1, y1) in boxes.items():
                crop = big.crop((int(x0 * kx), int(y0 * ky), int(np.ceil(x1 * kx)), int(np.ceil(y1 * ky))))
                stats["labels_blanked"] = stats.get("labels_blanked", 0) + blank_labels(crop, (x0, y0, x1, y1), labels, (kx, ky))
                crop.save(out_dir / name, quality=JPEG_QUALITY, subsampling=0)
            for name in merged:
                (out_dir / name).unlink(missing_ok=True)
            for name, fb in added:
                if name in boxes:
                    stats["added_from_pdf"] += 1
                    cap = caption_for(fb, captions)
                    if cap:
                        stats["anchors"][name] = cap
            all_merged += list(merged)
            stats["pages"] += 1; stats["figures"] += len(boxes)
            stats["merged"] += sum(v is not None for v in merged.values()); stats["dropped_captions"] += sum(v is None for v in merged.values())
    finally:
        if pdf is not None:
            pdf.close()
    md = out_dir / f"{stem}.md"
    if all_merged and md.is_file():
        md.write_text(drop_refs(md.read_text(encoding="utf-8"), all_merged), encoding="utf-8")
    return stats


# ----------------------------------------------------------------- text inside figures, written out by the vision LLM
DESCRIBE_PROMPT = """This image is a figure from a document.{context}
Describe it for a reader who cannot see it, so that nothing it shows is lost. The image may hold several figures or panels
(a diagram next to another, (a)/(b) parts): describe every one of them, not only the one a caption names. Return JSON with:
- "parts": first look over the whole image and list every separate figure, diagram, chart, table, photo or panel in it
  (with its label or caption if it has one), one entry each; a single figure gives one entry
- "kind": one of photo, diagram, schematic, chart, table, code, screenshot, map, text, logo, decoration
- "summary": one or two sentences saying what the figure is and what it shows or explains: its subject and purpose.
  Use the caption and nearby text to name things correctly, but describe only what is visible.
- "details": everything inside every part, in Markdown, part by part, without repeating the summary:
  - every label, number, unit and value exactly as written (read carefully; the caption and text help with names and spelling);
  - schematic or wiring diagram: each component with its label and value, and what it connects to, following the circuit;
  - other diagrams and flowcharts: each part with its label, and how the parts are connected or ordered;
  - chart or graph: the axes with their ranges and units, and the values of every series;
  - table: a Markdown table with every row and column;
  - code or a program listing: the exact code in a fenced code block with its language;
  - photo: what is in it and how it is arranged, plus any visible text; skip decorative detail;
  - logo or decoration: leave empty.
Write the summary and details in the language of the document (the caption and nearby text show it; English if there
is no text); labels and values stay exactly as written in the figure. Do not add facts that are not visible. No headings."""
FIGURE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["parts", "kind", "summary", "details"],
                 "properties": {"parts": {"type": "array", "items": {"type": "string"}}, "kind": {"type": "string"},
                                "summary": {"type": "string"}, "details": {"type": "string"}}}
SUMMARY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["kind", "summary"],
                  "properties": {"kind": {"type": "string"}, "summary": {"type": "string"}}}
IMG_LINK = re.compile(r"!\[[^\]]*\]\(([^)\s]+)\)")
DESC_MARK = "> **Figure content:**"
DESC_BLOCK = re.compile(r"\n+> \*\*Figure content:\*\*[^\n]*(?:\n>[^\n]*)*(?:\n\n```mermaid\n[\s\S]*?\n```)?")   # + its Mermaid flowchart
MIN_DESCRIBE_PX = 150        # crops are 300 DPI: smaller than half an inch is an icon, not a figure to write out
TEXT_HEAVY_NOTE = """
Most of the running text in this image is already written out in the document next to it: describe its figures, diagrams,
charts, tables and pictures in full, but mention the running text only in a few words."""


def _ask(llm_url, model, im, prompt, max_tokens, schema=None, timeout=900, extra=None):
    """one vision-LLM call on a PIL image -> (text, finish_reason)"""
    import base64, io, json, urllib.request
    im = im.convert("RGB")
    im.thumbnail((1600, 1600))
    buf = io.BytesIO(); im.save(buf, "JPEG", quality=92)
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}},
                {"type": "text", "text": prompt}]}]}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "figure", "strict": True, "schema": schema}}
    body.update(extra or {})
    req = urllib.request.Request(llm_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    import review                                           # the worker's cap on LLM requests in flight (review.LLM_GATE)
    with review.LLM_GATE:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            c = json.load(r)["choices"][0]
    return c["message"]["content"] or "", c.get("finish_reason")


def _ask_image(llm_url, model, img_path, prompt, timeout=900, max_tokens=2000, schema=None):
    """-> {"kind", "summary", "details"}; raises on failure (the caller retries)"""
    from review import latex_json_loads                     # LaTeX in JSON loses its backslashes otherwise
    text, finish = _ask(llm_url, model, Image.open(img_path), prompt, max_tokens, schema or FIGURE_SCHEMA, timeout)
    if finish == "length":
        raise ValueError("cut off")
    out = latex_json_loads(text)
    if not isinstance(out, dict) or not str(out.get("summary", "")).strip():
        raise ValueError("no summary")
    return {k: str(out.get(k, "")).strip() for k in ("kind", "summary", "details")}


GRAPH_PROMPT = """This image is a diagram from a document.{context}
If it is NOT a graph of shapes joined by arrows or lines (a flowchart, UML activity or state diagram, process, workflow,
decision tree, block diagram), return {{"diagrams": [], "links": []}}.
Otherwise write out its graph exactly, so that it could be redrawn from your answer.
The image may hold SEVERAL separate diagrams side by side, each with its own start: list each one separately in
"diagrams", in order from the leftmost to the rightmost. Dotted or dashed lines that fan out from one box to a whole other diagram mean that box is
expanded (zoomed) into that diagram: say so in "links" as "<box label> -> <diagram number>".
For each diagram:
- "nodes": EVERY shape, none skipped, each with a short id unique in the whole answer:
  start dot = shape "start"; end bull's-eye = "end"; box = "activity" with its label exactly as written (line breaks as
  spaces); diamond = "decision" (label = the question next to it) or "merge" (no label); thick bar = "fork" when one
  arrow goes in and several go out, "join" when several go in and one comes out.
- "edges": EVERY arrow, "from" id -> "to" id, with its label ("Yes", "No", a guard) or "". Follow each line to its
  arrowhead: the edge goes to the shape the arrowhead touches. Each branch that leaves a fork runs in parallel: an edge from
  the fork to its first box and from its last box to the join.
- Feedback or return lines: one line often runs along the bottom or side and branches off with an arrowhead into SEVERAL
  boxes (a waterfall's "go back" arrows). Every arrowhead on it is its own edge, from the shape the line starts at.
Before answering, count the boxes and the arrowheads in each diagram: every box must be a node, and there must be one edge
per arrowhead."""
GRAPH_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["diagrams", "links"], "properties": {
    "diagrams": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["nodes", "edges"], "properties": {
        "nodes": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["id", "label", "shape"],
                  "properties": {"id": {"type": "string"}, "label": {"type": "string"}, "shape": {"type": "string"}}}},
        "edges": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["from", "to", "label"],
                  "properties": {"from": {"type": "string"}, "to": {"type": "string"}, "label": {"type": "string"}}}}}}},
    "links": {"type": "array", "items": {"type": "string"}}}}
GRAPH_KINDS = ("diagram", "flowchart")             # "kind" values whose image may be a graph of boxes and arrows (not circuits)


def _chains(nodes, edges):
    """a diagram's arrows as chains "A → B → C", the main flow first: from the start (or the first box) the first arrow out
    of each box is followed as far as it goes; then every arrow not yet told starts a chain of its own (a branch, a
    feedback arrow), which goes on through boxes whose arrows are not told yet"""
    outs = {}
    for e in edges:
        outs.setdefault(e[0], []).append(e)
    has_in = {e[1] for e in edges}
    roots = [i for i in nodes if i not in has_in] or list(nodes)[:1]
    order, todo = [], list(roots)
    while todo:                                                       # boxes in the order a reader meets them
        i = todo.pop(0)
        if i not in order:
            order.append(i); todo += [e[1] for e in outs.get(i, [])]
    order += [i for i in nodes if i not in order]
    used, chains = set(), []
    for u in order:
        for e in outs.get(u, []):
            if id(e) in used:
                continue
            chain, cur = [u], e
            while cur is not None:
                used.add(id(cur)); chain.append((cur[2], cur[1]))
                seen = chain[:1] + [t for _, t in chain[1:]]
                free = [x for x in outs.get(cur[1], []) if id(x) not in used]
                if cur[1] in seen[:-1]:
                    break                                     # the loop is closed: "A → B → C → A"
                fresh = [x for x in free if x[1] not in seen]
                cur = fresh[0] if fresh else (free[0] if free else None)
            chains.append(chain)
    return chains


def graph_markdown(g):
    """a diagram's graph (GRAPH_SCHEMA) -> (Markdown of its arrows in words, a Mermaid flowchart block), or ("", "").
    Several diagrams sit side by side (left to right), each drawn top to bottom; UML start = solid dot, end = bull's-eye,
    fork/join = solid bar. Edges to unknown ids are dropped."""
    shape_text = {"start": "start", "end": "end", "fork": "fork", "join": "join", "merge": "merge"}
    quote = lambda t: '"' + t.replace('"', "#quot;") + '"'
    diagrams = []
    for k, d in enumerate(g.get("diagrams") or [], 1):
        nodes = {str(n.get("id")): n for n in d.get("nodes") or [] if str(n.get("id", "")).strip()}
        edges = [(str(e.get("from")), str(e.get("to")), (e.get("label") or "").strip()) for e in d.get("edges") or []
                 if str(e.get("from")) in nodes and str(e.get("to")) in nodes]
        if edges:
            diagrams.append((k, nodes, edges))
    if not diagrams:
        return "", ""
    many = len(diagrams) > 1
    ids, words, used_cls = {}, [], set()
    lines = ["```mermaid", "flowchart LR" if many else "flowchart TD"]
    nums = {k: n for n, (k, _, _) in enumerate(diagrams, 1)}                  # diagrams without arrows dropped: renumber
    for k, nodes, edges in diagrams:
        for i in nodes:
            ids[k, i] = f"n{len(ids) + 1}"            # per diagram: the model may call each diagram's start "start"
        name = lambda i, nodes=nodes: (nodes[i].get("label") or "").strip() or shape_text.get(nodes[i].get("shape"), "step")
        ind = "    " if many else "  "
        if many:
            words.append(f"- **Diagram {nums[k]}:**")
            lines += [f'  subgraph d{nums[k]}["Diagram {nums[k]}"]', "    direction TB"]
        for i, n in nodes.items():
            sh, label = n.get("shape"), quote(name(i))
            bar = '["' + "&nbsp;" * 16 + '"]'                              # a wide, thin black bar
            node = {"start": '((" "))', "end": '(((" ")))', "fork": bar, "join": bar,
                    "decision": "{" + label + "}", "merge": '{" "}'}.get(sh, f"({label})")
            cls = {"start": "umlStart", "end": "umlEnd", "fork": "umlBar", "join": "umlBar"}.get(sh)    # "end" is a Mermaid keyword
            if cls:
                used_cls.add(cls)
            lines.append(f"{ind}{ids[k, i]}{node}" + (f":::{cls}" if cls else ""))
        for e in edges:
            lines.append(f"{ind}{ids[k, e[0]]} -->" + (f"|{quote(e[2])}|" if e[2] else "") + f" {ids[k, e[1]]}")
        for ch in _chains(nodes, edges):
            text = name(ch[0]) + "".join(f" →{f' ({lab})' if lab else ''} {name(t)}" for lab, t in ch[1:])
            words.append(("  - " if many else "- ") + text)
        if many:
            lines.append("  end")
    labels = {}                                           # a link names its box by label or by id ("activity1 -> diagram2")
    for k, nodes, _ in diagrams:
        for i, n in nodes.items():
            if (n.get("label") or "").strip():
                for key in (n["label"].strip().lower(), i.strip().lower()):
                    labels.setdefault(key, (k, n["label"].strip()))
    for link in g.get("links") or []:
        m = re.match(r"\s*(.+?)\s*->\s*(?:diagram\s*)?(\d+)\s*$", str(link), re.I)
        if m and m.group(1).strip().lower() in labels and int(m.group(2)) in nums and many:
            k, label = labels[m.group(1).strip().lower()]
            if k != int(m.group(2)):
                words.append(f"- *{label}* (Diagram {nums[k]}) is shown in detail in Diagram {nums[int(m.group(2))]}")
                lines.append(f"  d{nums[k]} -.->|{quote(label + ' in detail')}| d{nums[int(m.group(2))]}")    # subgraph to subgraph
    style = {"umlStart": "fill:#000,stroke:#000", "umlEnd": "fill:#000,stroke:#fff,stroke-width:3px",       # end: bull's-eye
             "umlBar": "fill:#000,stroke:#000,font-size:2px"}
    lines += [f"  classDef {c} {style[c]}" for c in sorted(used_cls)] + ["```"]
    return "\n".join(["Arrows, as drawn:"] + words), "\n".join(lines)


MISSING_PROMPT = """This image is a diagram. Someone listed its arrows as:
{arrows}
The boxes are: {boxes}.
Look at the image again, line by line, and find every arrowhead that this list does NOT have yet: especially arrows that go
back or up to an earlier box, lines along the bottom or the side that branch off into several boxes (each arrowhead is one
arrow), and arrows into or out of bars. Return only the missing arrows, with "from" and "to" being box labels exactly as in
the list above, and "label" only if a word is written on that arrow (else ""), or an empty list if none is missing."""
MISSING_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["missing"], "properties": {"missing": {"type": "array",
    "items": {"type": "object", "additionalProperties": False, "required": ["from", "to", "label"], "properties": {
        "from": {"type": "string"}, "to": {"type": "string"}, "label": {"type": "string"}}}}}}


def add_missing(g, answers):
    """adds to graph g the arrows that every one of the check answers ([[{"from", "to", "label"}]] by box label) proposes,
    when both ends are labelled boxes of one diagram and the arrow is not there yet; -> how many were added. (On a complete
    graph the check invents arrows, but never the same one between two labelled boxes in every answer.)"""
    sets = [{(str(e.get("from", "")).strip().lower(), str(e.get("to", "")).strip().lower()) for e in a} for a in answers]
    agreed = set.intersection(*sets) if sets else set()
    added = 0
    for d in g.get("diagrams") or []:
        by_label = {(n.get("label") or "").strip().lower(): str(n.get("id")) for n in d.get("nodes") or [] if (n.get("label") or "").strip()}
        have = {(str(e.get("from")), str(e.get("to"))) for e in d.get("edges") or []}
        for a, b in sorted(agreed):
            if a in by_label and b in by_label and a != b and (by_label[a], by_label[b]) not in have:
                d.setdefault("edges", []).append({"from": by_label[a], "to": by_label[b], "label": ""})
                have.add((by_label[a], by_label[b])); added += 1
    return added


def ask_graph(llm_url, model, img_path, context="", tries=3, checks=2):
    """the boxes and arrows of a flow diagram: (arrows in words, Mermaid block) as graph_markdown, or ("", "").
    The model sometimes leaves arrows out (a waterfall's feedback arrows: 2 of 3 answers had all 8), never adds any: it is
    asked several times at once and the fullest answer is kept. img_path: a file or the image's bytes"""
    import io
    from concurrent.futures import ThreadPoolExecutor
    from review import latex_json_loads
    raw = img_path if isinstance(img_path, bytes) else Path(img_path).read_bytes()

    def one(_):
        for tokens in (3000, 5000):
            try:
                text, finish = _ask(llm_url, model, Image.open(io.BytesIO(raw)), GRAPH_PROMPT.format(context=context), tokens, GRAPH_SCHEMA)
                if finish != "length":
                    return latex_json_loads(text)
            except Exception:
                continue
        return None
    with ThreadPoolExecutor(tries) as ex:
        answers = [g for g in ex.map(one, range(tries)) if isinstance(g, dict)]
    size = lambda g: (sum(len(d.get("nodes") or []) for d in g.get("diagrams") or []),
                      sum(len(d.get("edges") or []) for d in g.get("diagrams") or []), len(g.get("links") or []))
    if not answers:
        return "", ""
    g = max(answers, key=size)
    lines, boxes = [], set()                # a second look for the arrows it left out (a waterfall's feedback arrows)
    for d in g.get("diagrams") or []:
        name = {str(n.get("id")): (n.get("label") or "").strip() or n.get("shape", "") for n in d.get("nodes") or []}
        boxes |= {(n.get("label") or "").strip() for n in d.get("nodes") or [] if (n.get("label") or "").strip()}
        lines += [f"- {name.get(str(e.get('from')), '?')} -> {name.get(str(e.get('to')), '?')}" + (f" ({e['label']})" if e.get("label") else "")
                  for e in d.get("edges") or []]
    if lines and checks:
        prompt = MISSING_PROMPT.format(arrows="\n".join(lines), boxes=", ".join(sorted(boxes)))

        def check(_):
            try:
                text, finish = _ask(llm_url, model, Image.open(io.BytesIO(raw)), prompt, 1500, MISSING_SCHEMA)
                return latex_json_loads(text).get("missing") or [] if finish != "length" else None
            except Exception:
                return None
        with ThreadPoolExecutor(checks) as ex:
            found = [a for a in ex.map(check, range(checks)) if isinstance(a, list)]
        if len(found) == checks:
            add_missing(g, found)
    return graph_markdown(g)


CORE_PROMPT = """This image was cut from a document page around a figure, and the cut may have taken in things that are NOT
part of the figure itself:
- the figure's number or caption, e.g. "FIGURE 5.3" in a coloured tab and the caption words next to it or under it;
- the page's running header or page number, e.g. "30 Chapter 2 Software processes", "140 PART TWO MODELING";
- horizontal rules (long thin lines) above or below that separate the figure from the page;
- body text paragraphs.
List each of those in "not_figure" with what it is and its bounding box. Then give in "figure" the bounding box of the
figure itself only: the diagram, chart, photo or drawing with all of its own boxes, labels, arrows, legends and axes,
and none of the things in "not_figure". Boxes are [x1, y1, x2, y2] in coordinates from 0 to 1000 relative to the image
width and height."""
_BOX = {"type": "array", "items": {"type": "integer"}}
CORE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["not_figure", "figure"], "properties": {
    "not_figure": {"type": "array", "items": {"type": "object", "additionalProperties": False, "required": ["what", "bbox_2d"],
                                              "properties": {"what": {"type": "string"}, "bbox_2d": _BOX}}},
    "figure": _BOX}}
CORE_LOST_INK = 0.005        # the core crop may leave out at most this share of the figure's ink that nobody explained


def core_box(ink, fig, not_fig):
    """ink: bool (H, W); fig and not_fig boxes in pixels (the vision LLM's). -> the figure's box snapped to whole ink shapes,
    or None when it would lose ink that is neither in a not_figure box nor a thin rule (the LLM cut real content)"""
    from scipy import ndimage
    h, w = ink.shape
    rules = ndimage.binary_opening(ink, structure=np.ones((1, max(3, int(0.3 * w))), bool))    # long horizontal lines ...
    rl, _ = ndimage.label(rules)
    for k, sl in enumerate(ndimage.find_objects(rl), 1):                                        # ... outside the figure box:
        if sl is not None and _inter((sl[1].start, sl[0].start - 3, sl[1].stop, sl[0].stop + 3), fig) > 0:
            rules[sl] &= rl[sl] != k                                                            # its own lines (an axis) stay
    ink = ink & ~ndimage.binary_dilation(rules, structure=np.ones((5, 1), bool))              # page rules (one may run into a tab)
    lab, _ = ndimage.label(ndimage.binary_dilation(ink, structure=np.ones((5, 5), bool)))
    lab = np.where(ink, lab, 0)
    keep, lost = None, 0
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is None:
            continue
        cb = (sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
        a = max(1.0, _area(cb))
        if any(_inter(cb, nb) >= 0.5 * a for nb in not_fig):
            continue                                            # the caption, label, header the LLM pointed out
        if _inter(cb, fig) >= 0.5 * a:
            keep = cb if keep is None else _union(keep, cb)
        elif not (cb[3] - cb[1] <= max(4, 0.012 * h) and cb[2] - cb[0] >= 8 * (cb[3] - cb[1])):    # a thin rule may go
            lost += int((lab[sl] == k).sum())
    if keep is None or lost > CORE_LOST_INK * max(1, int(ink.sum())):
        return None
    return keep


def tighten_figure(img_path, llm_url, model, pad_px=12):
    """crops a figure image to the figure itself (the vision LLM tells what is caption, page header or rule) -> True if cropped"""
    from review import latex_json_loads
    img_path = Path(img_path)
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    try:
        text, finish = _ask(llm_url, model, im, CORE_PROMPT, 800, CORE_SCHEMA)
        j = latex_json_loads(text)
        px = lambda b: (b[0] * w / 1000, b[1] * h / 1000, b[2] * w / 1000, b[3] * h / 1000)
        fig = px(j["figure"])
        nots = [px(x["bbox_2d"]) for x in j.get("not_figure") or [] if len(x.get("bbox_2d") or []) == 4]
    except Exception:
        return False
    if finish == "length" or fig[2] - fig[0] < 0.1 * w or fig[3] - fig[1] < 0.1 * h:
        return False
    b = core_box(ink_mask(im), fig, nots)
    if b is None:
        return False
    b = (max(0, b[0] - pad_px), max(0, b[1] - pad_px), min(w, b[2] + pad_px), min(h, b[3] + pad_px))
    if _area(b) >= 0.97 * w * h:
        return False                                            # already the figure alone
    im.crop(tuple(int(v) for v in b)).save(img_path, quality=JPEG_QUALITY, subsampling=0)
    return True


TILE_PX = 1100
TRANSCRIBE_PROMPT = """This is one part of a larger image (a page, a newspaper, a scanned document). Transcribe all the text
in this part exactly as written, in reading order, keeping its language and script, as Markdown (headings, lists, tables
where the layout shows them). Text cut at the edge of the part: write what is visible. No comments; nothing if there is no text."""


NO_LOOPS = {"dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 3}   # llama.cpp's DRY: a repeated word sequence costs
                              # more each time (a newspaper's "To Auctioneers, etc." x100). Only where nothing is copied from the prompt
TRANSCRIBE_ONE_PROMPT = ("Transcribe all the text in this image exactly as written, in reading order (column by column where the "
                         "text is in columns), keeping its language and script, as Markdown (headings, lists, tables where the "
                         "layout shows them). Write a row of dots (a dot leader) as \" … \". No comments.")


def _looped(text):
    """an answer stuck in a loop: a line 4 times in a row, or one stretch repeated 6+ times"""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        if run >= 4:
            return True
    return bool(re.search(r"(.{6,40}?)\1{5,}", text))


def clean_transcript(text):
    """dot leaders as " … ", and a line repeated more than twice in a row kept twice"""
    text = re.sub(r"(?:\s*\.){6,}\s*", " … ", text)
    out = []
    for l in text.split("\n"):
        if len(out) >= 2 and l.strip() and l.strip() == out[-1].strip() == out[-2].strip():
            continue
        out.append(l)
    return "\n".join(out).strip()


def transcribe_image(llm_url, model, img_path):
    """every word of a picture of text: one answer for the whole image (a letter, an invoice: 2 s, every line and price);
    tile by tile only when one answer is cut off or loops (a newspaper page) -> Markdown text"""
    try:
        text, finish = _ask(llm_url, model, Image.open(img_path), TRANSCRIBE_ONE_PROMPT, 4000, extra=NO_LOOPS)
        if finish != "length" and text.strip() and not _looped(text):
            return clean_transcript(text)
    except Exception:
        pass
    return clean_transcript(transcribe_tiles(llm_url, model, img_path))


def transcribe_tiles(llm_url, model, img_path, concurrency=4):
    """the text of a dense image (a newspaper page...) too long for one answer: read tile by tile at full resolution"""
    from concurrent.futures import ThreadPoolExecutor
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    cols, rows = max(1, round(w / TILE_PX)), max(1, round(h / TILE_PX))
    tw, th, pad = w / cols, h / rows, 40
    tiles = [im.crop((max(0, int(c * tw) - pad), max(0, int(r * th) - pad), min(w, int((c + 1) * tw) + pad), min(h, int((r + 1) * th) + pad)))
             for c in range(cols) for r in range(rows)]                       # column by column: newspapers read down first

    def one(t):
        for _ in range(2):
            try:
                text, _ = _ask(llm_url, model, t, TRANSCRIBE_PROMPT, 3000, extra=NO_LOOPS)
                return text.strip()
            except Exception:
                continue
        return ""
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        return "\n\n".join(x for x in ex.map(one, tiles) if x)


SUB_CAPTION = re.compile(r"\s*(\*\*|_)?\(\s*[a-zA-Z]\s*\)\s+\S")      # "(a) Noisy unoriented data set": one panel of a figure


def _context(md, name):
    """the figure's caption (the paragraph after it), the caption of the whole figure when it is one panel of several, and the
    text before it, for the prompt"""
    at = md.find(f"]({name})")
    after = [p for p in DESC_BLOCK.sub("", md[at:at + 4000]).split("\n\n")[1:] if p.strip()]
    from review import CAPTION                              # "Figure 3-2", "图3", "Рис. 5", "চিত্র ৪", "شكل 3"...
    is_cap = lambda p: len(p) < 400 and (CAPTION.match(p) or re.match(r"\s*(\*\*|_)?(fig(ure)?|table|chart|listing|photo)\b", p, re.I))
    caption = next((p.strip() for p in after[:2] if is_cap(p) or SUB_CAPTION.match(p) and len(p) < 200), "")
    whole = ""
    if SUB_CAPTION.match(caption):                          # one panel: the whole figure's caption follows its last panel
        whole = next((p.strip() for p in after[1:12] if is_cap(p) and not SUB_CAPTION.match(p)), "")
    before = next((p.strip() for p in reversed(md[max(0, at - 1500):at].split("\n\n")[:-1])
                   if p.strip() and not p.lstrip().startswith((">", "!", "|", "#")) and not SUB_CAPTION.match(p)), "")
    clean = lambda t, n: re.sub(r"<[^>]+>|!\[[^\]]*\]\([^)]*\)", "", t).strip()[:n]
    out = ""
    if caption:
        out += f'\nA caption next to it in the document (it may name only one of several figures): "{clean(caption, 400)}"'
    if whole:
        out += f'\nIt is one panel of a larger figure whose caption is: "{clean(whole, 400)}"'
    if before and not SUB_CAPTION.match(caption):
        out += f'\nText just before it: "{clean(before, 500)}"'
    return out


def _render(d, md):
    """the description as a quote under the image: what it is, then what is in it"""
    details = d["details"]
    fence = re.search(r"```[^\n]*\n(.*?)```", details, re.S)
    if d["kind"].lower() == "code" and fence:                 # a listing already printed in the text is not copied again
        words = re.findall(r"\w{2,}", fence.group(1))
        text = set(re.findall(r"\w{2,}", md))
        if words and sum(w in text for w in words) / len(words) >= 0.9:
            details = re.sub(r"```[^\n]*\n.*?```", "", details, flags=re.S).strip()
            d = dict(d, summary=d["summary"].rstrip() + " The same code is printed in the text.")
    lines = [DESC_MARK + " " + " ".join(d["summary"].split())]
    body = [l.rstrip() for l in details.splitlines()]
    while body and not body[0].strip():
        body.pop(0)
    if body:
        lines.append(">")
        lines += [("> " + l) if l.strip() else ">" for l in body]
    if d.get("mermaid"):                                      # outside the quote: many viewers draw no Mermaid inside one
        lines += ["", d["mermaid"]]
    return "\n".join(lines)


TEXT_KINDS = ("text", "screenshot")
TEXT_IMAGE_PX = 1000          # a text picture at least this big (long side) is transcribed tile by tile, not only described


def describe_figures(md_path, img_dir, llm_url, model, concurrency=2, text_heavy=None, tighten=True, graph_tries=3,
                     graph_checks=2, defer=False, transcribe_text=False):
    """adds under each linked figure (once) what it shows and what is in it, written by the vision LLM from the image, its
    caption and the text around it; returns how many it described. tighten: crop each figure to itself first (one call);
    graph_tries/graph_checks: answers and missing-arrow checks for a flow diagram. defer=True: the Markdown is not written
    yet -> (count, apply), apply(md_path) puts the descriptions in later (the page review runs at the same time)"""
    from concurrent.futures import ThreadPoolExecutor
    import hashlib
    md_path, img_dir = Path(md_path), Path(img_dir)
    md = md_path.read_text(encoding="utf-8")
    names = [n for n in dict.fromkeys(IMG_LINK.findall(md)) if (img_dir / n).is_file()
             and not re.search(r"(?m)^\|.*\]\(" + re.escape(n) + r"\)", md)]      # a picture in a table cell: a quote would break the table
    names = [n for n in names if not re.search(re.escape(f"]({n})") + r"\s*\n+" + re.escape(DESC_MARK), md)
             and not re.search(r"!\[[^\]]+\]\(" + re.escape(n) + r"\)", md)]        # a logo already named in its alt text
    same, todo = {}, []                     # identical images (a repeated icon or logo) are described once
    for n in names:
        try:
            with Image.open(img_dir / n) as im:
                if max(im.size) < MIN_DESCRIBE_PX:
                    continue                # an icon or bullet: nothing to write out
        except Exception:
            continue
        h = hashlib.sha1((img_dir / n).read_bytes()).hexdigest()
        if h not in same:
            todo.append(n)
        same.setdefault(h, []).append(n)

    text_heavy = text_heavy or {}

    from concurrent.futures import ThreadPoolExecutor as _Pool
    graphs = _Pool(max(1, concurrency))     # a figure's graph is read while it is cropped and described (it needs neither)

    def one(n):
        wide = (img_dir / n).read_bytes()   # the graph is read better with some room around it (Fig 5.3: right order 4/4, not 0/4)
        ctx = _context(md, n)
        early = graph_tries and "_Diagram_" in n          # Marker's layout called it a diagram: likely boxes and arrows
        graph = graphs.submit(ask_graph, llm_url, model, wide, ctx, graph_tries, graph_checks) if early else None
        if tighten:
            try:                            # the figure alone: no caption, label tab, page header or rule in the picture
                tighten_figure(img_dir / n, llm_url, model)
            except Exception:
                pass
        prompt = DESCRIBE_PROMPT.format(context=_context(md, n)) + (TEXT_HEAVY_NOTE if n in text_heavy else "")
        for tokens in (2000, 2000, 4000):   # a failed or cut-off answer is asked again (never skipped silently)
            try:
                d = _ask_image(llm_url, model, img_dir / n, prompt, max_tokens=tokens)
            except Exception:
                continue
            if d["kind"].lower() in GRAPH_KINDS:
                words, mermaid = graph.result() if graph is not None else \
                    (ask_graph(llm_url, model, wide, ctx, graph_tries, graph_checks) if graph_tries else ("", ""))
                if words:                   # boxes and arrows read on their own: the structure is exact, not retold
                    d["details"], d["mermaid"] = words, mermaid
                    try:                        # and the summary says the same (it was written before the graph was read)
                        s = _ask_image(llm_url, model, img_dir / n, DESCRIBE_PROMPT.format(
                            context=_context(md, n) + f"\nIts boxes and arrows, already read exactly:\n{words}")
                            + '\nOnly "kind" and "summary" this time: no details. The summary must agree with those arrows.',
                            max_tokens=600, schema=SUMMARY_SCHEMA)
                        d["summary"] = s["summary"]
                    except Exception:
                        pass
            elif transcribe_text and d["kind"].lower() in TEXT_KINDS and n not in text_heavy:
                try:                        # a newspaper page, a letter: every word of it, read tile by tile (one answer can't hold it)
                    with Image.open(img_dir / n) as im:
                        big = max(im.size) >= TEXT_IMAGE_PX
                    if big:
                        text = transcribe_image(llm_url, model, img_dir / n)
                        if text and not _looped(text):
                            d["details"] = text
                except Exception:
                    pass
            return n, d
        # still too long (a newspaper page in one picture): a short summary, and its text read tile by tile
        try:
            d = _ask_image(llm_url, model, img_dir / n, prompt + "\nOnly \"kind\" and \"summary\" this time: no details.",
                           max_tokens=600, schema=SUMMARY_SCHEMA)
        except Exception:
            d = {"kind": "text", "summary": "A page of text.", "details": ""}
        if n not in text_heavy:
            d["details"] = transcribe_image(llm_url, model, img_dir / n)
        return n, d
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        first = {n: d for n, d in ex.map(one, todo) if d}
    graphs.shutdown(wait=False, cancel_futures=True)          # graphs of figures that were not diagrams: not needed
    got = {m: first[n] for ns in same.values() for n in ns[:1] if n in first for m in ns}
    for ns in same.values():                # the same picture elsewhere gets the same (tightened) crop
        for m in ns[1:]:
            if ns[0] in first:
                (img_dir / m).write_bytes((img_dir / ns[0]).read_bytes())

    def apply(path):
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        for n, d in got.items():
            if d["kind"].lower() in ("logo", "decoration", "icon") and not d["details"].strip():
                alt = " ".join(re.sub(r"[\[\]\n]", " ", d["summary"]).split())[:200]   # a logo or icon: named in one line, no quote
                text = text.replace(f"![]({n})", f"![{alt}]({n})", 1)
                continue
            quote = _render(d, text)
            text = re.sub(r"(!\[[^\]]*\]\(" + re.escape(n) + r"\))", lambda m: m.group(1) + "\n\n" + quote, text, count=1)
        path.write_text(text, encoding="utf-8")
    if defer:
        return len(got), apply
    apply(md_path)
    return len(got)


def redescribe(md_path, img_dir, llm_url, model, concurrency=8):
    """removes every figure description from a finished file and writes them again (better prompt or model, no reconversion)"""
    md_path = Path(md_path)
    md_path.write_text(DESC_BLOCK.sub("", md_path.read_text(encoding="utf-8")), encoding="utf-8")
    return describe_figures(md_path, img_dir, llm_url, model, concurrency)


# ----------------------------------------------------------------- tables: values in the wrong column, pictures in cells
def _runs(mask, gap):
    """[(start, stop)] of the True runs of a 1-D bool array, runs closer than `gap` merged"""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    out, s, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > gap:
            out.append((int(s), int(p) + 1)); s = i
        p = i
    out.append((int(s), int(p) + 1))
    return out


def realign_table(rows, ink):
    """rows: a table's rows (cell strings, "" = empty) as written; ink: bool array of the table's image (no ruled lines).
    -> the rows with every value moved to the column it stands under in the image (Case 14: the lower rows of a staggered
    table had slid left), or None when the image does not match (row count, values per row, column count). Values and their
    order within a row are never changed."""
    ncols = max(len(r) for r in rows)
    bands = [b for b in _runs(ink.any(axis=1), 2) if b[1] - b[0] >= 4]
    if len(bands) != len(rows):
        return None
    blobs = []
    for (y0, y1), r in zip(bands, rows):
        bl = _runs(ink[y0:y1].any(axis=0), max(2, int(0.5 * (y1 - y0))))     # a value's characters, or "..." dots, stay one
        if len(bl) != len([c for c in r if c.strip()]):
            return None
        blobs.append(bl)
    h = float(np.median([b[1] - b[0] for b in bands]))
    best = None
    for side in (1, 0, None):                      # right-aligned numbers, left-aligned text, centred
        xs = sorted((b[side] if side is not None else (b[0] + b[1]) / 2) for bl in blobs for b in bl)
        groups = [[xs[0]]]
        for x in xs[1:]:
            if x - groups[-1][-1] > 0.6 * h:
                groups.append([x])
            else:
                groups[-1].append(x)
        if len(groups) == ncols:
            spread = max(g[-1] - g[0] for g in groups)
            if best is None or spread < best[0]:
                best = (spread, side, [float(np.mean(g)) for g in groups])
    if best is None:
        return None
    _, side, centres = best
    out = []
    for bl, r in zip(blobs, rows):
        row = [""] * ncols
        for b, v in zip(bl, [c for c in r if c.strip()]):
            x = b[side] if side is not None else (b[0] + b[1]) / 2
            k = int(np.argmin([abs(x - c) for c in centres]))
            if row[k]:
                return None                         # two values under one column: the image was not read right
            row[k] = v
        out.append(row)
    return out


def grid_cells(ink, nrows, ncols, frac=0.6):
    """[[cell box (x0, y0, x1, y1)]] of a ruled table of nrows x ncols cells (pixels), or None when its lines don't make that
    grid. Lines of a frame around the table (a coloured box with its title) are left out: the widest run of consecutive
    lines with the right count is the grid."""
    xs = [(a + b) / 2 for a, b in _runs(ink.mean(axis=0) >= frac, 2)]
    ys = [(a + b) / 2 for a, b in _runs(ink.mean(axis=1) >= frac, 2)]

    def edges(v, n, size):                     # the table's box may sit just inside its border lines: its edges are the outer lines
        if len(v) < n and not any(x < 0.03 * size for x in v):
            v = [0.0] + v
        if len(v) < n and not any(x > 0.97 * size for x in v):
            v = v + [float(size)]
        return v
    xs, ys = edges(xs, ncols + 1, ink.shape[1]), edges(ys, nrows + 1, ink.shape[0])
    if len(xs) < ncols + 1 or len(ys) < nrows + 1:
        return None
    gx = max((xs[i:i + ncols + 1] for i in range(len(xs) - ncols)), key=lambda v: v[-1] - v[0])
    gy = max((ys[i:i + nrows + 1] for i in range(len(ys) - nrows)), key=lambda v: v[-1] - v[0])
    return [[(gx[c], gy[r], gx[c + 1], gy[r + 1]) for c in range(ncols)] for r in range(nrows)]


def cell_picture(ink, pad=4):
    """the picture in a table cell (bool ink of the cell): everything above its last line of text (the label), or None"""
    ink = ink.copy()
    ink[:, ink.mean(axis=0) > 0.6] = False                     # the cell's own ruled lines
    ink[ink.mean(axis=1) > 0.6, :] = False
    bands = [(a + pad, b + pad) for a, b in _runs(ink[pad:-pad, pad:-pad].any(axis=1), 3) if b - a >= 2]
    if len(bands) < 2:
        return None
    label = bands[-1]
    y0, y1 = bands[0][0], bands[-2][1]
    if y1 - y0 < 0.5 * (label[1] - label[0]):
        return None                                            # only text above the label: no picture
    xr = _runs(ink[y0:y1, pad:-pad].any(axis=0), 10)
    return (xr[0][0] + pad, y0, xr[-1][1] + pad, y1) if xr else None


SHAPES = re.compile(r"\\(triangle|square|Box|bigtriangleup|bigtriangledown|diamond|Diamond|lozenge|bigcirc|pentagon|hexagon|"
                    r"blacksquare|blacktriangle|boxdot|boxslash|boxbslash|boxtimes|boxplus|varhexagon|octagon)\b|\\text\{\s*-{2,}\s*\}")
DISPLAY_MATH = re.compile(r"(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$")


def drawn_formula(latex):
    r"""a "formula" made of drawings (Case 15: Q = — + △ + □ + pentagons cut by diagonals): LaTeX has no way to write those
    shapes, so the OCR wrote \triangle, \square... for each one"""
    return len(SHAPES.findall(latex)) >= 3


def _drawn_paragraph(p):
    """a paragraph that is only a formula of drawn shapes: display maths, or inline maths with at most a few characters
    around it ("Q = $\\_ + \\triangle + ...$." after Marker's inline-maths rewrite, accurate quality)"""
    if not drawn_formula(p):
        return False
    rest = re.sub(r"(?<!\\)\$\$[\s\S]+?(?<!\\)\$\$|(?<!\\)\$[^$]+(?<!\\)\$", "", p)
    return len(re.sub(r"\s", "", rest)) <= 12


DRAWN_SPAN = re.compile(r"(?:\b[A-Za-z]{1,3}\s*=\s*)?(?<!\\)\$\$?[^$]+?(?<!\\)\$\$?[.,;]?")   # "Q = $\\_ + \\triangle ...$."


def picture_formulas(md_path, document, src, out_dir, skipped=None):
    """a display formula made of drawings becomes a picture of it, cropped from the page (Marker's Equation block), and is
    described like a figure; its LaTeX was a guess. Found as a maths span (display, or inline after Marker's inline-maths
    rewrite, even inside a sentence: only the formula becomes the picture), paired with the Marker block sharing most of it.
    skipped (a dict) gets why a drawn formula was left as it was. -> how many"""
    from marker.schema import BlockTypes
    from review import PAGE_SPLIT
    md_path, out_dir = Path(md_path), Path(out_dir)
    if not str(src).lower().endswith(".pdf") or not hasattr(BlockTypes, "Equation"):
        return 0
    import pypdfium2 as pdfium
    md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    pages = {p.page_id: p for p in document.pages}
    done = 0
    pdf = pdfium.PdfDocument(str(src))
    try:
        for i in range(1, len(parts), 2):
            pid, body = int(parts[i]), parts[i + 1]
            found = [m for m in DRAWN_SPAN.finditer(body) if drawn_formula(m.group(0))]
            page = pages.get(pid)
            if not found or page is None or pid >= len(pdf):
                continue
            kinds = tuple(getattr(BlockTypes, k) for k in ("Equation", "TextInlineMath", "Text") if hasattr(BlockTypes, k))
            blocks = [b for b in page.contained_blocks(document, kinds) if drawn_formula(getattr(b, "html", "") or "")]
            toks = lambda t: set(re.findall(r"\\[a-zA-Z]+|\w+", re.sub(r"<[^>]+>", " ", t)))
            pairs, used = [], set()
            for m in found:                                    # each formula with the block sharing most of it (after Marker's
                ft = toks(m.group(0))                         # inline-maths rewrite, the same formula can be in two blocks)
                score = [(len(ft & toks(getattr(b, "html", "") or "")) / max(1, len(ft)), k) for k, b in enumerate(blocks) if k not in used]
                best = max(score, default=(0, None))
                if best[1] is not None and best[0] >= 0.5:
                    used.add(best[1]); pairs.append((m, blocks[best[1]]))
            if skipped is not None and len(pairs) < len(found):
                why = "no Marker block with that formula" if blocks else "no Marker block with drawn shapes"
                skipped[why] = skipped.get(why, 0) + len(found) - len(pairs)
            found = [m for m, _ in pairs]; blocks = [b for _, b in pairs]
            if not found:
                continue
            img = pdf[pid].render(scale=CROP_DPI / 72).to_pil().convert("RGB")
            sx, sy = img.width / page.polygon.width, img.height / page.polygon.height
            for k, (m, block) in reversed(list(enumerate(zip(found, blocks)))):
                x0, y0, x1, y1 = block.polygon.bbox
                name = f"_page_{pid}_Formula_{k}.jpeg"
                img.crop((max(0, int((x0 - 4) * sx)), max(0, int((y0 - 4) * sy)), min(img.width, int((x1 + 4) * sx)),
                          min(img.height, int((y1 + 4) * sy)))).save(out_dir / name, quality=JPEG_QUALITY, subsampling=0)
                body = body[:m.start()].rstrip(" ") + f"\n\n![]({name})\n\n" + body[m.end():].lstrip(" ")
                done += 1
            parts[i + 1] = body
    finally:
        pdf.close()
    if done:
        md_path.write_text(parts[0] + "".join(f"\n\n{{{parts[i]}}}{'-' * 48}\n\n{parts[i + 1].strip(chr(10))}"
                                              for i in range(1, len(parts), 2)) + "\n", encoding="utf-8")
    return done


TABLE_MD = re.compile(r"(?m)^(?:\|.*\|[ \t]*\n?){2,}")
SEP_ROW = re.compile(r"^\|[\s:|-]+\|$")
NO_SRC_IMG_CELL = re.compile(r'<img(?![^>]*\bsrc=)[^>]*?\balt="([^"]*)"[^>]*/?>')


def _table_rows(t):
    lines = [l.strip() for l in t.strip().splitlines()]
    return [[c.strip() for c in l.strip("|").split("|")] for l in lines if not SEP_ROW.match(l)]


def _table_md(rows):
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    line = lambda r: "| " + " | ".join(r) + " |"
    return "\n".join([line(rows[0]), "|" + "---|" * n] + [line(r) for r in rows[1:]]) + "\n"


def fix_tables(md_path, document, src, out_dir, stem=""):
    """after the page review: each Markdown table of a page is paired with Marker's Table block (same order on the page)
    and checked against the page image. Values that slid into the wrong column are moved back (realign_table), and a
    picture in a cell that Marker left as <img alt="surfel"> (no file) is cropped from the ruled cell and linked.
    -> {"tables_realigned", "table_pictures"}"""
    from marker.schema import BlockTypes
    from review import PAGE_SPLIT
    md_path, out_dir = Path(md_path), Path(out_dir)
    stats = {"tables_realigned": 0, "table_pictures": 0, "tables_skipped": {}}
    skip = lambda why: stats["tables_skipped"].__setitem__(why, stats["tables_skipped"].get(why, 0) + 1)
    if not str(src).lower().endswith(".pdf"):
        return stats
    import pypdfium2 as pdfium
    md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    pages = {p.page_id: p for p in document.pages}
    pdf = pdfium.PdfDocument(str(src))
    try:
        for i in range(1, len(parts), 2):
            pid, body = int(parts[i]), parts[i + 1]
            tables = list(TABLE_MD.finditer(body))
            page = pages.get(pid)
            if not tables or page is None or pid >= len(pdf):
                continue
            blocks = page.contained_blocks(document, (BlockTypes.Table,))
            words = lambda s: set(re.findall(r"[^\W_]{2,}", re.sub(r"<[^>]+>", " ", s).lower()))
            pairs = []                                         # each Markdown table with the Table block sharing most words
            for m in tables:
                mw = words(m.group(0))
                scored = [(len(mw & words(b.raw_text(document) or "")) / max(1, len(mw)), b) for b in blocks]
                best = max(scored, key=lambda x: x[0], default=(0, None))
                if best[1] is None or best[0] < 0.3:
                    skip("no matching table block"); continue
                pairs.append((m, best[1]))
            k = CROP_DPI / 72
            img = None
            for t, (m, block) in reversed(list(enumerate(pairs))):
                if img is None:
                    img = pdf[pid].render(scale=k).to_pil().convert("RGB")
                sx, sy = img.width / page.polygon.width, img.height / page.polygon.height
                x0, y0, x1, y1 = block.polygon.bbox
                box = (max(0, int((x0 - 3) * sx)), max(0, int((y0 - 3) * sy)), min(img.width, int((x1 + 3) * sx)), min(img.height, int((y1 + 3) * sy)))
                crop = img.crop(box)
                ink = ink_mask(crop)
                rows = _table_rows(m.group(0))
                new = None
                if any(NO_SRC_IMG_CELL.search(c) for r in rows for c in r):
                    cells = grid_cells(ink, len(rows), max(len(r) for r in rows))
                    if not cells:
                        skip("no ruled grid for its pictures")
                    if cells:
                        new = [list(r) for r in rows]
                        for r, row in enumerate(rows):
                            for c, cell in enumerate(row):
                                im_m = NO_SRC_IMG_CELL.search(cell)
                                if not im_m or c >= len(cells[r]):
                                    continue
                                cx0, cy0, cx1, cy1 = (int(v) for v in cells[r][c])
                                pic = cell_picture(ink[cy0:cy1, cx0:cx1])
                                if not pic:
                                    skip("no picture found in a cell"); continue
                                name = f"_page_{pid}_Table_{t}_r{r}c{c}.jpeg"
                                crop.crop((cx0 + pic[0] - 6, cy0 + pic[1] - 6, cx0 + pic[2] + 6, cy0 + pic[3] + 6)).save(
                                    out_dir / name, quality=JPEG_QUALITY, subsampling=0)
                                alt = re.sub(r"[\[\]]", "", im_m.group(1))
                                new[r][c] = cell[:im_m.start()] + f"![{alt}]({name})" + cell[im_m.end():]
                                stats["table_pictures"] += 1
                elif not any(re.search(r"<|!\[", c) for r in rows for c in r):
                    clean = ink.copy()                         # ruled lines would join the rows: left out
                    clean[:, clean.mean(axis=0) > 0.6] = False
                    clean[clean.mean(axis=1) > 0.6, :] = False
                    out = realign_table(rows, clean)
                    if out and out != [r + [""] * (len(out[0]) - len(r)) for r in rows]:
                        new = out
                        stats["tables_realigned"] += 1
                if new:
                    body = body[:m.start()] + _table_md(new) + body[m.end():]
            parts[i + 1] = body
    finally:
        pdf.close()
    if not stats["tables_skipped"]:
        del stats["tables_skipped"]
    if stats["tables_realigned"] or stats["table_pictures"]:
        md_path.write_text(parts[0] + "".join(f"\n\n{{{parts[i]}}}{'-' * 48}\n\n{parts[i + 1].strip(chr(10))}"
                                              for i in range(1, len(parts), 2)) + "\n", encoding="utf-8")
    return stats


# ----------------------------------------------------------------- did any text get lost?
WORD = re.compile(r"[^\W_]{3,}")
LOW_COVERAGE = 0.9


def _words(text):
    """words of any script (review.script_tokens), character pairs for Chinese/Japanese/Thai; short Latin words left out"""
    import unicodedata
    from review import script_tokens
    text = unicodedata.normalize("NFKC", text).lower()
    text = re.sub(r"[-­￾\x02]\s*\n\s*|[­￾\x02]", "", text)   # words hyphenated across lines (pdfium marks them U+FFFE)
    return [t for t in script_tokens(text, ngram=2) if len(t) >= 3 or not t.isascii()]


def text_coverage(pdf_path, md_text, pages):
    """for each page of a PDF that has a text layer: the share of its words that appear in the Markdown.
    -> [{"page", "words", "found", "low"}]; pages with fewer than 20 words are not judged"""
    import pypdfium2 as pdfium
    raw = md_text
    md_text = re.sub(r"\\[a-zA-Z]+|[_^]\{([^{}]*)\}", r"\1", md_text)              # V_{CAP} in LaTeX is the page's "VCAP"
    have = set(_words(md_text)) | set(_words(md_text.replace("-", "")))      # "double-throw" = "double-\nthrow" on the page
    have |= set(_words(raw.replace("\\_", "_")))                                # LaTeX printed as text ("\\mathbf{e}_1") counts too
    out, texts = [], {}
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for p in pages:
            if p >= len(pdf):
                continue
            tp = pdf[p].get_textpage()
            try:
                texts[p] = tp.get_text_range()
            finally:
                tp.close()
    finally:
        pdf.close()
    # running headers/footers ("Chapter 3: Getting Somewhat More Serious 131") repeat on many pages and are left out of the
    # Markdown on purpose: they don't count (page 144 of Make Electronics was flagged for its header alone)
    key = lambda line: re.sub(r"[\d\W_]+", "", line.lower())
    seen = {}
    for t in texts.values():
        for k in {key(l) for l in t.splitlines() if key(l)}:
            seen[k] = seen.get(k, 0) + 1
    running = {k for k, n in seen.items() if n >= 3 and n >= 0.2 * len(texts)}
    for p, t in texts.items():
        words = _words("\n".join(l for l in t.splitlines() if key(l) not in running))
        if not words:
            continue
        found = sum(w in have for w in words) / len(words)
        out.append({"page": p, "words": len(words), "found": round(found, 3), "low": len(words) >= 20 and found < LOW_COVERAGE})
    return out
