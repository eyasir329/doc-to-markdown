"""Two facing book pages photographed (or scanned) together, a spread, are cut into two pages before OCR.
Read as one page, the OCR misses or repeats lines of a spread: in the benchmark's upside-down spread of C listings (subset
p84), whole loops were missing and one listing came out twice; its two pages read on their own came back almost exactly.
No model: the text lines' direction and skew come from the ink's projection profiles, the fold is a clean gap in the text
across the whole height, and the page read first is the one on the side the lines start from (their aligned edge: left
for English or Bengali, right for Hebrew or Arabic, where books are bound the other way). Pages are cut only on raster
pages (no text layer) and only when all of that is clear; anything else is left as it was.
Before that, every such page is set upright (page_turn): Marker lays out and reads scans as they stand (Indra with
turn=False), so an upside-down or sideways photo lost lines (subset p94: two questions). The orientation comes from Indra,
the ONNX OCR Marker already runs here: its orientation model, and when that leans to upside down without being sure, the
longest lines read both ways. A turned page that is no spread keeps its image (the PDF page's rotation is set).
The job (pipeline.run_job) prepares the PDF once, converts the new one, then gives the Markdown back the original page
numbers (renumber): the two halves of a spread share their page's separator."""
import io

import numpy as np
from PIL import Image, ImageFilter

DETECT_PX = 1000           # long side of the image the analysis runs on
OUT_DPI = 200              # the halves are cut from a render at this resolution (the OCR reads pages at 192 dpi)
MAX_OUT_PX = 3400          # ... but no bigger than this on the long side (RAM)
VALLEY = 0.1               # the fold: the ink there is at most this share of the text's on either side (subset: spreads
                           # 0-0.08, a single page's column gap 0.14-0.29)
GUTTER = 0.03              # ... and the low-ink valley (two inner margins) is at least this share of the text's width wide
                           # (spreads 0.034-0.17, column gaps 0.012-0.032)
WEAK_VALLEY = 0.3          # a weaker valley (a tightly bound book: subset p83 depth 0.14, width 0.016) is a fold only when
WEAK_GUTTER = 0.012        # the two sides carry consecutive printed page numbers (numbered_pair): a single page with a clean
                           # empty band down its middle (handwritten notes, a formula page: depth 0, width 0.06-0.08) has
                           # no such pair, and pixels alone could not tell them apart
SIDE_INK = 0.25            # each page holds at least this share of the ink
MIN_LINES = 8              # ... and at least this many text lines (a page of text, not a label or a receipt; a strongly curved
                           # page's lines blur together: subset p81's left page counted 9; single pages with an empty middle
                           # band had at most 7 on one side)
MAX_SHEAR = 8              # degrees the fold may lean from the lines' normal (the two pages of a photo tilt differently)
MAX_LAYER_CHARS = 40       # a page with more text-layer characters than this is digital, not a photo or scan
ORIENT_PX = 1600           # long side of the image the orientation is read on
NUMBER_PX = 2400           # ... and the page numbers of a weaker fold's two sides (small footers on a photo of a spread:
                           # subset p85's left number read 28 at 1600 px, 201 at 2400; it prints 200)
NUMBER_GAP = 2             # the two sides' page numbers at most this far apart (one digit misread on a photo)
ORIENT_MARGIN = 0.15       # a half turn only when the longest lines read this much surer turned (subset: p84 +0.28, p11 +0.30;
                           # an upright Thai sign +0.07, read with the Latin recognizer)


def _box(a, k, axis):
    """moving sum of k along an axis (k odd), same size"""
    c = np.cumsum(np.pad(a.astype(np.int32), [(k // 2 + 1, k // 2) if ax == axis else (0, 0) for ax in range(a.ndim)]), axis=axis)
    return np.take(c, np.arange(k, c.shape[axis]), axis=axis) - np.take(c, np.arange(0, c.shape[axis] - k), axis=axis)


def ink_mask(gray):
    """dark strokes on the paper: darker than their surroundings, inside the bright (paper) part of the photo and away from
    its edge (the paper's outline against the background is no text)"""
    a = np.asarray(gray, dtype=np.float32)
    bg = np.asarray(gray.filter(ImageFilter.BoxBlur(15)), dtype=np.float32)
    soft = gray.filter(ImageFilter.BoxBlur(25))
    sa = np.asarray(soft, dtype=np.float32)
    hist = np.histogram(sa, bins=64, range=(0, 256))[0].astype(np.float64)          # Otsu on the blurred photo: paper vs the rest
    w = np.cumsum(hist); m = np.cumsum(hist * np.arange(64)); tot_w, tot_m = w[-1], m[-1]
    between = (m * tot_w - w * tot_m) ** 2 / np.maximum(w * (tot_w - w), 1e-9)
    paper = sa > (np.argmax(between) + 1) * 4
    edge = max(3, max(gray.size) // 70) | 1         # eroded by edge px (two moving sums: MinFilter took 0.6 s a page)
    paper = _box(_box(paper, edge, 0), edge, 1) >= edge * edge
    return (a < bg - 18) & paper


def drop_long_runs(ink, length):
    """removes near-vertical strokes longer than `length` px (the fold, rules, a table's borders, the edges of the pages
    under the open ones): they are not text. Strokes are widened first so a slanted one stays one run"""
    wide = _box(ink, 5, 1) > 0
    core = _box(wide, length | 1, 0) >= (length | 1)
    return ink & ~(_box(_box(core, length | 1, 0) > 0, 7, 1) > 0)


def _rotate(mask, deg):
    return np.asarray(Image.fromarray(mask.astype(np.uint8) * 255).rotate(deg, resample=Image.NEAREST, expand=True)) > 127


def line_angle(ink):
    """the rotation (degrees, PIL's counter-clockwise) that makes the text lines horizontal: the sharpest row profile"""
    best = (-1.0, 0)
    for base in (0, 90):
        for d in np.arange(-12, 12.1, 1.5):
            prof = _rotate(ink, base + d).sum(axis=1).astype(np.float64)
            score = np.mean(np.diff(prof) ** 2) / max(1.0, prof.sum())
            if score > best[0]:
                best = (score, base + float(d))
    return best[1]


def _sheared_cols(ink, deg):
    """ink per column along lines leaning deg from vertical: column x counts row y at x + (y - H/2) * tan(deg)"""
    H, W = ink.shape
    off = np.round((np.arange(H) - H / 2) * np.tan(np.radians(deg))).astype(int)
    ys, xs = np.nonzero(ink)
    x = xs - off[ys]
    keep = (x >= 0) & (x < W)
    return np.bincount(x[keep], minlength=W).astype(np.float64)


def find_fold(ink):
    """ink with horizontal lines -> {"x", "shear", "span", "strong"} or None: the deepest valley in the middle of the text,
    over the leans the fold may have. strong: deep and wide enough on its own; else (a weaker valley) the page is a spread
    only if its two sides carry consecutive page numbers (prepare_page)"""
    best = _fold_in(ink, ink.shape[1])
    if best is None:
        return None
    depth, width = best[0], best[4]
    strong = depth <= VALLEY and width >= GUTTER
    if not strong and not (depth <= WEAK_VALLEY and width >= WEAK_GUTTER):
        return None
    return {"x": best[1], "shear": best[2], "span": best[3], "strong": strong}


def _fold_in(ink, W):
    """the deepest valley of one band over the leans -> (depth, x, lean, (x0, x1), valley width share) or None"""
    best = None
    for deg in range(-MAX_SHEAR, MAX_SHEAR + 1):
        cols = _sheared_cols(ink, deg)
        if cols.sum() < 200:
            return None
        c = np.cumsum(cols) / cols.sum()
        x0, x1 = int(np.searchsorted(c, 0.01)), int(np.searchsorted(c, 0.99))
        width = x1 - x0
        if width < 100:
            return None
        k = max(3, width // 50) | 1
        sm = _box(cols, k, 0) / k
        lo, hi = x0 + int(0.3 * width), x0 + int(0.7 * width)
        x = lo + int(np.argmin(sm[lo:hi]))
        a, b = sm[x0 + width // 20:x - width // 30], sm[x + width // 30:x1 - width // 20]
        if len(a) < 10 or len(b) < 10:
            continue
        side = min(np.median(a), np.median(b))
        depth = sm[x] / max(1e-9, side)
        if min(c[x], 1 - c[x]) >= SIDE_INK and (best is None or depth < best[0]):
            low = sm < 0.5 * side                  # the valley: both inner margins around the fold
            l, r = x, x
            while l > x0 and low[l - 1]:
                l -= 1
            while r < x1 and low[r]:
                r += 1
            best = (depth, x, deg, (x0, x1), (r - l) / width)
    return best


def _lines(part):
    """text lines of a (horizontal) block: [(start x, end x, height)] of the lines at least a quarter of its width long"""
    rows = part.sum(axis=1)
    on = rows > 0.03 * max(1, rows.max())
    out, y = [], 0
    while y < len(on):
        if on[y]:
            s = y
            while y < len(on) and on[y]:
                y += 1
            if y - s >= 3:
                xs = np.nonzero(part[s:y].any(axis=0))[0]
                if len(xs) and xs[-1] - xs[0] >= 0.25 * part.shape[1]:
                    out.append((xs[0], xs[-1], s, y))
        y += 1
    return out


def two_pages(ink, fold):
    """the two sides each hold a page of text: enough lines, over a similar height, of a similar width -> (lines, lines) or None"""
    x, (x0, x1) = fold["x"], fold["span"]
    sides = [ink[:, x0:x], ink[:, x:x1]]
    lines = [_lines(p) for p in sides]
    if min(len(l) for l in lines) < MIN_LINES:
        return None
    tops = [(l[0][2], l[-1][3]) for l in lines]
    overlap = min(tops[0][1], tops[1][1]) - max(tops[0][0], tops[1][0])
    if overlap < 0.5 * min(t[1] - t[0] for t in tops):
        return None
    if min(x - x0, x1 - x) < 0.5 * max(x - x0, x1 - x):
        return None
    return lines


def start_side(lines, widths):
    """'left' or 'right': the edge the text lines start from (aligned), from how often line ends jump vs line starts.
    None when it is not clear"""
    jumps = {"start": 0, "end": 0}
    for ls, w in zip(lines, widths):
        tol = 0.045 * w
        st, en = np.array([l[0] for l in ls]), np.array([l[1] for l in ls])
        jumps["start"] += int(np.sum(np.abs(np.diff(st)) > tol))
        jumps["end"] += int(np.sum(np.abs(np.diff(en)) > tol))
    total = jumps["start"] + jumps["end"]
    if total < 4 or abs(jumps["start"] - jumps["end"]) < 0.25 * total:
        return None
    return "left" if jumps["start"] < jumps["end"] else "right"


def analyse(gray):
    """a page image (grayscale PIL) -> {"angle", "fold", "shear", "span", "rows", "first", "sure_order"} (shares of the
    rotated image) or None (no spread)"""
    s = DETECT_PX / max(gray.size)
    g = gray.resize((max(1, round(gray.width * s)), max(1, round(gray.height * s))), Image.BILINEAR) if s < 1 else gray
    ink = ink_mask(g)
    if ink.sum() < 0.002 * ink.size:
        return None
    angle = line_angle(ink)
    r = _rotate(ink, angle)
    r = drop_long_runs(r, max(20, r.shape[0] // 12))
    fold = find_fold(r)
    if not fold:
        return None
    lines = two_pages(r, fold)
    if not lines:
        return None
    x, (x0, x1) = fold["x"], fold["span"]
    side = start_side(lines, (x - x0, x1 - x))
    rows = np.nonzero(r[:, x0:x1].any(axis=1))[0]
    H, W = r.shape
    return {"angle": angle, "fold": x / W, "shear": fold["shear"], "span": (x0 / W, x1 / W),
            "rows": (rows[0] / H, rows[-1] / H) if len(rows) else (0.0, 1.0),
            "first": side or "left", "sure_order": side is not None, "strong": fold["strong"]}


def cut(img, info, margin=0.02, overlap=0.01):
    """the two pages of a spread (RGB PIL, any size) in reading order, text lines horizontal; the other page's side of the
    (leaning) fold is painted white"""
    from PIL import ImageDraw
    r = img.rotate(info["angle"], resample=Image.BICUBIC, expand=True, fillcolor=(255, 255, 255))
    f, (x0, x1), (y0, y1), sh = info["fold"], info["span"], info["rows"], info["shear"]
    if info["first"] == "right":                   # lines start on the right: turned over, the first page is on the left
        r = r.rotate(180, expand=True)
        f, x0, x1, y0, y1 = 1 - f, 1 - x1, 1 - x0, 1 - y1, 1 - y0      # the lean stays the same under a half turn
    W, H = r.size
    t = np.tan(np.radians(sh))
    fx = lambda y: f * W + (y - H / 2) * t          # the fold's x at row y
    top, bottom = max(0, int((y0 - margin) * H)), min(H, int((y1 + margin) * H))
    halves = []
    for left in (True, False):
        page = r.copy()
        d, ov = ImageDraw.Draw(page), overlap * W
        if left:                                   # everything right of the fold (plus the overlap) goes
            d.polygon([(fx(0) + ov, 0), (W, 0), (W, H), (fx(H) + ov, H)], fill=(255, 255, 255))
            box = (max(0, int((x0 - margin) * W)), top, min(W, int(max(fx(top), fx(bottom)) + ov)), bottom)
        else:
            d.polygon([(0, 0), (fx(0) - ov, 0), (fx(H) - ov, H), (0, H)], fill=(255, 255, 255))
            box = (max(0, int(min(fx(top), fx(bottom)) - ov)), top, min(W, int((x1 + margin) * W)), bottom)
        halves.append(page.crop(box))
    return halves


_ENGINE = []


def _engine():
    """Indra, the ONNX OCR Marker runs here, for page orientation: once per process; None where it is not installed"""
    if not _ENGINE:
        eng = None
        try:
            from marker.ocr.indra import IndraRecognitionPredictor
            eng = IndraRecognitionPredictor(threads=1, math=False, furniture=False, table_heads=False).engine
        except Exception:
            try:
                from indra_ocr import OrtOCR, model_dir
                eng = OrtOCR(model_dir(None), threads=1)
            except Exception:
                eng = None
        _ENGINE.append(eng)
    return _ENGINE[0]


def page_turn(eng, gray):
    """quarter turns (counter-clockwise) the content of a page image (numpy, gray) is turned by: Indra's own rule
    (OrtOCR.__call__), except that an unsure upside-down call is settled by reading the longest lines both ways and must
    be ORIENT_MARGIN surer turned. Pages set in vertical columns are left as they are (0)"""
    from indra_ocr.pipeline import DESKEW_MIN, SIDEWAYS_LINES, SIDEWAYS_MASS, VERTICAL, estimate_skew, preview, rotate_page
    small = preview(gray)
    a = estimate_skew(small)
    p = eng.orientation_probs(rotate_page(small, a) if abs(a) >= DESKEW_MIN else small)
    if p is None:
        return 0
    k = int(p.argmax())
    if p[k] <= eng.orient_min:
        if float(p[1] + p[3] + (p[VERTICAL] if len(p) > VERTICAL else 0)) >= SIDEWAYS_MASS:
            k = eng.read_turn(gray, a, small, p)
        elif k != 2:
            return 0
    if k != 2:
        return k if k in (1, 3) else 0

    def conf(kk):
        img, _, _ = eng._turn(gray, kk, a, small)
        bx = eng.detect(img)
        bx = bx[np.argsort(-(bx[:, 2] - bx[:, 0]))[:SIDEWAYS_LINES]]
        return float(np.mean([c for _, c, _ in eng.recognize(img, bx)])) if len(bx) else 0.0
    return 2 if conf(2) > conf(0) + ORIENT_MARGIN else 0


def numbered_pair(eng, gray, fold):
    """True when both sides of a page image (numpy, gray, upright) at x share `fold` carry a printed page number in their
    outermost lines (header or footer, e.g. "Page | 118" and "Page | 119"), at most NUMBER_GAP apart; Indra reads them. A
    single page has one page number, on one side (subset: no single page had one on each). Only the two top and two bottom
    lines of each side count, so a contents page full of numbers does not"""
    bx = eng.detect(gray)
    if len(bx) < 4:
        return False
    W = gray.shape[1]
    nums = {"left": [], "right": []}
    for side, keep in (("left", (bx[:, 0] + bx[:, 2]) / 2 < fold * W), ("right", (bx[:, 0] + bx[:, 2]) / 2 >= fold * W)):
        b = bx[keep]
        if len(b) < 2:
            continue
        order = np.argsort(b[:, 1])
        edge = b[np.unique(np.concatenate([order[:2], order[-2:]]))]
        for text, _, _ in eng.recognize(gray, edge):
            n = page_number(text)
            if n is not None:
                nums[side].append(n)
    return any(abs(r - l) <= NUMBER_GAP for l in nums["left"] for r in nums["right"])


def upright(img, k):
    """a PIL image of content turned k quarter turns counter-clockwise, turned back"""
    return img.transpose({1: Image.ROTATE_270, 2: Image.ROTATE_180, 3: Image.ROTATE_90}[k]) if k in (1, 2, 3) else img


def prepare_page(src, i, turn=True):
    """page i of a PDF (a photo or scan) -> {"turn": quarter turns to undo, "pdf": a 2-page PDF (bytes) of a spread's pages}
    or None when nothing is to be done; runs in a process pool"""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(src))
    try:
        page = pdf[i]
        if page.get_textpage().count_chars() > MAX_LAYER_CHARS:
            return None                            # digital text: upright, and its columns are no spread
        w, h = page.get_size()
        view = page.render(scale=ORIENT_PX / max(w, h)).to_pil().convert("L")
        k, eng = 0, _engine() if turn else None
        if eng is not None:
            try:
                k = page_turn(eng, np.asarray(view))
            except Exception:
                k = 0
        small = upright(view, k)
        small = small.resize((max(1, round(small.width * DETECT_PX / max(small.size))),
                              max(1, round(small.height * DETECT_PX / max(small.size)))), Image.BILINEAR)
        info = analyse(small)
        if info and not info["strong"]:            # a weaker fold: a spread only with consecutive page numbers on its sides
            try:
                big = upright(page.render(scale=NUMBER_PX / max(w, h)).to_pil().convert("L"), k)
                ok = eng is not None and numbered_pair(eng, np.asarray(big), info["fold"])
            except Exception:
                ok = False
            info = info if ok else None
        if not info:
            return {"turn": k} if k else None
        scale = min(OUT_DPI / 72, MAX_OUT_PX / max(w, h))
        halves = cut(upright(page.render(scale=scale).to_pil().convert("RGB"), k), info)
        buf = io.BytesIO()
        halves[0].save(buf, "PDF", save_all=True, append_images=halves[1:], resolution=72 * scale, quality=90)
        return {"turn": k, "pdf": buf.getvalue()}
    finally:
        pdf.close()


def split_spreads(src, dst, pages=None, workers=1, turn=True):
    """src PDF -> (dst, page_map, pages turned upright) when some page is a spread or was turned, else (src, None, 0).
    page_map[k] = the original page of the new PDF's page k (a spread's two halves share it, first half first).
    pages: the original pages that may be prepared"""
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(src))
    try:
        todo = [i for i in range(len(pdf)) if (pages is None or i in pages)
                and pdf[i].get_textpage().count_chars() <= MAX_LAYER_CHARS]   # photos and scans only
        if workers > 1 and len(todo) > 2:
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor        # spawn: the API process runs threads, fork could hang
            with ProcessPoolExecutor(min(workers, len(todo)), mp_context=multiprocessing.get_context("spawn")) as ex:
                found = dict(zip(todo, ex.map(prepare_page, [str(src)] * len(todo), todo, [turn] * len(todo))))
        else:
            found = {i: prepare_page(src, i, turn) for i in todo}
        found = {i: v for i, v in found.items() if v}
        if not found:
            return src, None, 0
        out, page_map = pdfium.PdfDocument.new(), []
        for i in range(len(pdf)):
            got = found.get(i) or {}
            if got.get("pdf"):
                two = pdfium.PdfDocument(got["pdf"])
                out.import_pages(two); page_map += [i, i]
                two.close()
                continue
            out.import_pages(pdf, [i]); page_map.append(i)
            if got.get("turn"):                    # its image is kept: the page is shown turned (clockwise, as /Rotate is)
                pg = out[len(out) - 1]
                pg.set_rotation((pg.get_rotation() + 90 * got["turn"]) % 360)
        out.save(str(dst))
        out.close()
        return dst, page_map, sum(1 for v in found.values() if v.get("turn"))
    finally:
        pdf.close()


def keep_pairs(parts, page_map):
    """page chunks (lists of the new PDF's pages) with no spread's halves in two chunks: a chunk ending on a first half
    takes the second one too"""
    out = [list(p) for p in parts]
    for k in range(len(out) - 1):
        while out[k] and out[k + 1] and page_map[out[k][-1]] == page_map[out[k + 1][0]]:
            out[k].append(out[k + 1].pop(0))
    return [p for p in out if p]


ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _roman(t):
    n = 0
    for a, b in zip(t, t[1:] + " "):
        v = ROMAN[a]
        n += -v if ROMAN.get(b, 0) > v else v
    return n


def page_number(text):
    """the printed page number in a page's first or last lines ("Page | 127", "- 12 -", "xi"), or None"""
    import re
    lines = [re.sub(r"[#*_>|`]+", " ", l).strip() for l in text.splitlines() if l.strip() and not l.startswith("![")]
    edge = lines[:4] + lines[-4:]
    for l in edge:
        m = re.search(r"\bpage\s*:?\s*(\d{1,4}|[ivxlc]{1,7})\b", l, re.I)
        if m:
            v = m.group(1).lower()
            return int(v) if v.isdigit() else _roman(v)
    for l in edge:
        m = re.fullmatch(r"[-–—\s]*(\d{1,4})[-–—\s]*", l)
        if m:
            return int(m.group(1))
    return None


def finish(md, page_map, numbers=None):
    """the worker's Markdown of the new PDF (with page separators) -> the halves of each spread in their printed order (the
    second half's page number one below the first's: the photo was read upside down), then the original page numbers.
    numbers: {page: its printed number} from the page headers / footers (Marker leaves those out of the Markdown); a page
    without one is looked up in its Markdown"""
    numbers = numbers or {}
    import re
    sep = re.compile(r"(\n*\{\d+\}-{48}\n*)")
    parts = sep.split(md)                          # text, separator, text, separator, text ...
    num = lambda s: int(re.search(r"\{(\d+)\}", s).group(1))
    swapped = 0
    for j in range(1, len(parts) - 3, 2):
        a, b = num(parts[j]), num(parts[j + 2])
        if b == a + 1 and b < len(page_map) and page_map[a] == page_map[b]:
            na, nb = numbers.get(a, page_number(parts[j + 1])), numbers.get(b, page_number(parts[j + 3]))
            if na is not None and nb is not None and nb == na - 1:
                parts[j + 1], parts[j + 3] = parts[j + 3], parts[j + 1]; swapped += 1
    return renumber("".join(parts), page_map), swapped


def pages_for(page_map, original):
    """the new PDF's pages for a list of original pages (PAGE_RANGE)"""
    want = set(original)
    return [k for k, p in enumerate(page_map) if p in want]


def renumber(md, page_map):
    """Markdown of the new PDF with page separators -> original page numbers; a spread's second half joins its first"""
    import re
    sep = re.compile(r"(\n*)\{(\d+)\}(-{48})(\n*)")
    seen = set()

    def one(m):
        k = int(m.group(2))
        p = page_map[k] if k < len(page_map) else k
        if p in seen:
            return "\n\n"
        seen.add(p)
        return f"{m.group(1)}{{{p}}}{m.group(3)}{m.group(4)}"
    return sep.sub(one, md)


def merge_trace(pages, page_map):
    """per-page trace keyed by the new PDF's pages -> keyed by original pages (a spread's halves added up)"""
    out = {}
    for k, v in pages.items():
        p = str(page_map[int(k)]) if int(k) < len(page_map) else k
        if p not in out:
            out[p] = dict(v, halves=1) if page_map.count(int(p)) > 1 else dict(v)
            continue
        o = out[p]
        o["halves"] = o.get("halves", 1) + 1
        for key, x in v.items():
            if isinstance(x, (int, float)) and not isinstance(x, bool):
                o[key] = round(o.get(key, 0) + x, 1) if isinstance(x, float) else o.get(key, 0) + x
            elif isinstance(x, list):
                o[key] = list(dict.fromkeys((o.get(key) or []) + x))
            elif isinstance(x, dict):
                o[key] = {kk: o.get(key, {}).get(kk, 0) + vv for kk, vv in x.items()} if all(isinstance(vv, (int, float)) for vv in x.values()) else x
            else:
                o.setdefault(key, x)
    return out
