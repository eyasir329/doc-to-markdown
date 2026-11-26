"""Which path each file type takes, and the conversions that need no model (standard library only).
Imported by worker.py (does the work) and pipeline.py (the list of accepted extensions); tested in dev/test_pipeline.py.

  marker  : Marker reads the file itself (PDF, images, HTML, EPUB, and DOCX/PPTX/XLSX in OFFICE="native" mode)
  soffice : LibreOffice converts it first, then Marker reads the result
            - documents and slides -> PDF: Word equations (OMML), slide layouts, text boxes, SmartArt and embedded figures
              survive as rendered pages, which the layout model and the VLM turn into LaTeX, tables and figure images.
              (Marker's own DOCX path goes through mammoth, which drops equations and floating shapes.)
            - spreadsheets -> XLSX: Marker's spreadsheet path keeps every sheet as a table
  ddjvu   : DjVu is rendered to an image PDF by djvulibre's ddjvu, then Marker OCRs it like a scan
  text    : plain text, Markdown, CSV/TSV are turned into Markdown directly"""
import csv, io, json, os, re, shutil, signal, subprocess
from pathlib import Path

MARKER_EXT = (".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".gif", ".bmp", ".html", ".htm", ".epub")
OOXML_NATIVE = (".docx", ".pptx", ".xlsx")                  # Marker can read these without LibreOffice
OFFICE_TO_PDF = (".doc", ".docx", ".docm", ".dot", ".dotx", ".odt", ".ott", ".fodt", ".rtf", ".wpd", ".wps",
                 ".ppt", ".pptx", ".pptm", ".pps", ".ppsx", ".pot", ".potx", ".odp", ".otp", ".fodp", ".odg")
SHEET_TO_XLSX = (".xls", ".xlsm", ".xlsb", ".ods", ".ots", ".fods")
TEXT_EXT = (".txt", ".md", ".markdown", ".csv", ".tsv")
DJVU_EXT = (".djvu", ".djv")
ALL_EXT = tuple(sorted(set(MARKER_EXT + OOXML_NATIVE + OFFICE_TO_PDF + SHEET_TO_XLSX + TEXT_EXT + DJVU_EXT)))
OFFICE_MODES = ("pdf", "native")

SOFFICE = shutil.which("soffice") or shutil.which("libreoffice")
DDJVU = shutil.which("ddjvu")


class Unsupported(Exception):
    pass


def route(name, office_mode="pdf", has_soffice=None, has_ddjvu=None):
    """-> (action, target_format): ("marker", None) | ("soffice", "pdf"|"xlsx") | ("ddjvu", "pdf") | ("text", None)"""
    has_soffice = bool(SOFFICE) if has_soffice is None else has_soffice
    has_ddjvu = bool(DDJVU) if has_ddjvu is None else has_ddjvu
    ext = Path(name).suffix.lower()
    if ext in TEXT_EXT:
        return "text", None
    if ext in DJVU_EXT:
        if not has_ddjvu:
            raise Unsupported(f"{ext} files need ddjvu (apt package djvulibre-bin), which is not installed on this box")
        return "ddjvu", "pdf"
    if ext in MARKER_EXT:
        return "marker", None
    if ext == ".xlsx":
        return "marker", None                                # Marker's own XLSX path is already table-faithful
    if ext in OOXML_NATIVE and (office_mode == "native" or not has_soffice):
        return "marker", None
    if ext in OFFICE_TO_PDF + SHEET_TO_XLSX:
        if not has_soffice:
            raise Unsupported(f"{ext} files need LibreOffice, which is not installed on this box (OFFICE_VIA_LIBREOFFICE in the notebook)")
        return "soffice", "xlsx" if ext in SHEET_TO_XLSX else "pdf"
    raise Unsupported(f"unsupported file type {ext!r}")


def run_tool(cmd, out, what, timeout):
    """runs a converter in its own process group (killed whole on timeout) and checks that it wrote `out`"""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        log = p.communicate(timeout=timeout)[0].decode("utf-8", "replace")
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL); p.wait()            # soffice forks soffice.bin: kill the whole group
        raise RuntimeError(f"{what} took longer than {timeout} s")
    if not Path(out).is_file() or Path(out).stat().st_size == 0:
        raise RuntimeError(f"{what} failed (exit {p.returncode}): {log.strip()[-400:]}")
    return Path(out)


def soffice(src, fmt, outdir, profile_dir, timeout=600):
    """converts src with headless LibreOffice; every worker passes its own profile_dir so several can run at once"""
    if not SOFFICE:
        raise Unsupported("LibreOffice is not installed on this box")
    src, outdir = Path(src), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = [SOFFICE, "--headless", "--norestore", "--nolockcheck", f"-env:UserInstallation={Path(profile_dir).resolve().as_uri()}",
           "--convert-to", fmt, "--outdir", str(outdir), str(src)]
    return run_tool(cmd, outdir / f"{src.stem}.{fmt}", f"LibreOffice converting {src.name} to {fmt}", timeout)


def djvu_to_pdf(src, outdir, timeout=1800):
    """DjVu -> image PDF (ddjvu drops the hidden text layer; Marker OCRs the pages)"""
    if not DDJVU:
        raise Unsupported("ddjvu (djvulibre-bin) is not installed on this box")
    src, outdir = Path(src), Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{src.stem}.pdf"
    return run_tool([DDJVU, "-format=pdf", "-quality=85", str(src), str(out)], out, f"ddjvu converting {src.name}", timeout)


CODEPAGES = ("gbk", "shift_jis", "big5", "euc_kr", "cp1251", "cp1256", "cp1253", "cp1255", "cp874", "cp1250", "cp1252")


def _common(c):
    """a character people write often: full-width kana, Hangul, frequently used hanzi (GB2312 level 1, Big5 A440-C67E),
    other letters; half-width katakana and rare ideographs are what a wrong code page produces"""
    o = ord(c)
    if 0xFF61 <= o <= 0xFF9F:
        return False
    if 0x3040 <= o <= 0x30FF or 0xAC00 <= o <= 0xD7A3 or not (0x3400 <= o <= 0x9FFF or 0xF900 <= o <= 0xFAFF):
        return True
    for enc, lo, hi in (("gb2312", 0xB0A1, 0xD7FE), ("big5", 0xA440, 0xC67E)):
        try:
            b = c.encode(enc)
            if len(b) == 2 and lo <= int.from_bytes(b, "big") <= hi:
                return True
        except UnicodeEncodeError:
            pass
    return False


def _plausible(text):
    """how much a decoded text looks like real writing (higher is better), None if it can't be"""
    import unicodedata
    if any(unicodedata.category(c) in ("Cc", "Co", "Cn") and c not in "\t\n\r\f" for c in text):
        return None
    letters = [c for c in text if c.isalpha() and not c.isascii()]
    if not letters:
        return 0.0
    scripts = {unicodedata.name(c, "?").split()[0] for c in letters}
    scripts = {"CJK" if x in ("CJK", "HIRAGANA", "KATAKANA", "HALFWIDTH") else x for x in scripts}
    if len(scripts) > 2:                                   # real text mixes at most two (Japanese, Korean with Hanja...)
        return None
    flips = sum(1 for x, y in zip(text, text[1:]) if x.islower() and y.isupper() and not x.isascii())   # "КЮСнС"
    phonetic = sum(1 for c in letters if 0x3040 <= ord(c) <= 0x30FF or 0xAC00 <= ord(c) <= 0xD7A3)       # kana, Hangul
    words = re.findall(r"\w+", text)
    mixed = sum(1 for w in words if re.search(r"[a-zA-Z]", w) and any(c.isalpha() and not c.isascii() and
                                                                    not unicodedata.name(c, "").startswith("LATIN") for c in w))   # "cafй"
    bonus = 0.5 * phonetic if len(scripts) == 1 else 0     # kana / Hangul text: a wrong code page rarely gives whole words of it
    return (sum(_common(c) for c in letters) - flips - mixed + bonus) / len(letters) - 0.5 * (len(scripts) - 1)


def guess_decode(raw, sample=20000):
    """bytes in an unknown legacy code page -> text: every candidate that decodes without error is scored on its start"""
    best, score = None, None
    for k, enc in enumerate((("utf-16-le", "utf-16-be") if len(raw) % 2 == 0 else ()) + CODEPAGES):   # UTF-16 without a BOM too
        try:
            text = raw.decode(enc)
        except UnicodeDecodeError:
            continue
        if enc.startswith("utf-16") and not re.search(r"[ \n]", text):
            continue                                       # real UTF-16 text has spaces or line breaks; 4 bytes of "café" don't
        sc = _plausible(text[:sample])
        if sc is not None and (score is None or sc - k * 1e-3 > score):
            best, score = text, sc - k * 1e-3
    return best


def read_text(path):
    """text in any encoding: UTF-8/16/32 (with or without BOM), else the most plausible legacy code page (GBK, Big5,
    Shift-JIS, EUC-KR, Windows-1251/1256/...). Decoding as cp1252 alone turned a Chinese GBK file into "ÖÐÎÄ" """
    raw = Path(path).read_bytes()
    for bom, enc in ((b"\xff\xfe\x00\x00", "utf-32"), (b"\x00\x00\xfe\xff", "utf-32"), (b"\xff\xfe", "utf-16"), (b"\xfe\xff", "utf-16")):
        if raw.startswith(bom):
            return raw.decode(enc, errors="replace")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    return guess_decode(raw) or raw.decode("cp1252", errors="replace")


def md_cell(s):
    return s.replace("\\", "\\\\").replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>").strip()


def table_to_md(text, delimiter=None):
    """CSV/TSV text -> one Markdown table (first row = header). delimiter None = sniff among , ; | TAB"""
    if not text.strip():
        return ""
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:20000], delimiters=",;|\t").delimiter
        except csv.Error:
            delimiter = ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delimiter) if any(c.strip() for c in r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [[md_cell(c) for c in r] + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines) + "\n"


def text_to_md(src, out_dir, stem):
    """writes <out_dir>/<stem>.md for .txt/.md/.csv/.tsv; returns the worker result fields"""
    src, out_dir = Path(src), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext, text = src.suffix.lower(), read_text(src)
    if ext in (".csv", ".tsv"):
        md = table_to_md(text, "\t" if ext == ".tsv" else None)
    else:
        md = text.replace("\r\n", "\n")
        if not md.endswith("\n"):
            md += "\n"
    (out_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    return {"pages": None, "files": [f"{stem}.md"], "via": "text"}


# ----------------------------------------------------------------- old Bangla PDFs (Bijoy / SutonnyMJ fonts)
# Bangla typeset with Bijoy fonts (SutonnyMJ, ...) maps Bangla glyphs onto Latin codes: the text layer reads "Avwg evsjvq"
# for "আমি বাংলায়". Marker would copy that; such PDFs must be read with OCR instead.
LEGACY_FONT = re.compile(r"(?i)(sutonny|bijoy|boishakhi|mj$|mj[\W_])")


def legacy_font(pdf_path, max_pages=12):
    """the name of a legacy (non-Unicode) Bangla font used for text in the first pages, else None"""
    try:
        import ctypes
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw
    except Exception:
        return None
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(min(len(pdf), max_pages)):
            for obj in pdf[i].get_objects(filter=[raw.FPDF_PAGEOBJ_TEXT], max_depth=2):
                font = raw.FPDFTextObj_GetFont(obj.raw)
                if not font:
                    continue
                n = raw.FPDFFont_GetBaseFontName(font, None, 0)
                buf = ctypes.create_string_buffer(max(1, n))
                raw.FPDFFont_GetBaseFontName(font, buf, n)
                name = buf.value.decode("latin-1").split("+")[-1]            # ABCDEF+SutonnyMJ
                if LEGACY_FONT.search(name):
                    return name
    except Exception:
        return None
    finally:
        pdf.close()
    return None


INDIC = re.compile(r"[\u0900-\u0dff]")                 # Devanagari, Bengali, Gurmukhi, Gujarati, Oriya, Tamil, Telugu, Kannada, Malayalam, Sinhala
DOT_WORDS = re.compile(r"(?:[·•]{2,12}[ \u00a0]+){2,}[·•]{1,12}")      # "····· ··· ····": glyphs with no Unicode, shown as dots
BROKEN = re.compile(r"[\ufffd\ue000-\uf8ff]")


def _indic_skeleton(word):
    """an Indic word's consonants and independent letters, without vowel signs, virama and nukta"""
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c)[0] != "M" and c not in "\u200c\u200d")


def _indic_invalid(words):
    """how many words have a vowel sign where none can be (word start, or right after another vowel sign): visual-order text"""
    import unicodedata
    bad = 0
    for w in words:
        prev = None
        for i, c in enumerate(w):
            sign = "VOWEL SIGN" in unicodedata.name(c, "")
            if sign and (i == 0 or prev):
                bad += 1
                break
            prev = sign
    return bad


def _align_indic(text, ref, word):
    """words garbled beyond a reordering (ইউনস্কেণ্ট for ইউনেস্কো): the output's Indic words are aligned with the PDF's; a short
    stretch (up to 3 words) that differs between two stretches that match is replaced by the PDF's words"""
    from difflib import SequenceMatcher
    spans = [(m.start(), m.end(), m.group(0)) for m in word.finditer(text)]
    ops = SequenceMatcher(None, [w for _, _, w in spans], ref, autojunk=False).get_opcodes()
    edits = []
    for k, (tag, i1, i2, j1, j2) in enumerate(ops):
        before = k == 0 or ops[k - 1][0] == "equal"
        after = k == len(ops) - 1 or ops[k + 1][0] == "equal"
        if tag == "replace" and i2 - i1 <= 3 and j2 - j1 <= 3 and before and after and len(ops) > 1:
            edits.append((spans[i1][0], spans[i2 - 1][1], " ".join(ref[j1:j2])))
    for a, b, w in reversed(edits):
        text = text[:a] + w + text[b:]
    return text, len(edits)


def _unescaped(line):
    r"""a Markdown line without its backslash escapes (\$ -> $), and where each character came from in the line"""
    out, at, i = [], [], 0
    while i < len(line):
        if line[i] == "\\" and i + 1 < len(line) and line[i + 1] in "$_*#`|!\\.-+":
            out.append(line[i + 1]); at.append(i); i += 2
        else:
            out.append(line[i]); at.append(i); i += 1
    at.append(len(line))
    return "".join(out), at


def repair_lines(text, layer_lines, need=0.9, most=8):
    r"""one page's Markdown against its text layer (the PDF's exact characters): a line that matches a layer line (or two
    joined) at least `need` but differs in a few characters gets those characters from the layer (Case 15: the OCR's
    \sqrt[2]{2} where the page prints \sqrt{2}). Only small differences (at most `most` characters each) are taken, never
    Markdown syntax (#, *, _, `, |, >, !, links, task boxes), and never in code, maths, tables, quotes or images.
    A layer that disagrees with most lines it matches (3 or more lines differ, under half as many are the same) is not trusted: an old
    scan's own OCR layer can be worse than Surya's reading, and Marker re-reads such pages for that reason.
    -> (text, characters spans fixed)"""
    from difflib import SequenceMatcher
    cands = [l.strip() for l in layer_lines if l.strip()]
    cands += [f"{a.strip()} {b.strip()}" for a, b in zip(layer_lines, layer_lines[1:]) if a.strip() and b.strip()]
    out, fixed, fence, math = [], 0, False, False
    pending, same = [], 0                                       # (line number, its corrected text), lines the layer agrees with
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith(("```", "~~~")):
            fence = not fence
        if s == "$$" or (s.startswith("$$") and not s.endswith("$$")) or (s.endswith("$$") and not s.startswith("$$") and math):
            math = not math
        dollars = bool(re.search(r"(?<!\\)\$", s))             # maths, unless the PDF's text has those "$" too (printed LaTeX)
        skip = (fence or math or s.startswith(("|", ">", "![", "<")) or (s.startswith("$$") and not dollars) or "](" in s
                or len(s) < 20)
        if not skip:
            tail = line[len(line.rstrip()):]                   # "  " at the end is a line break: kept
            line = line.rstrip()
            lead = re.match(r"\s*(?:(?:[-*+]|\d+[.)])\s+|#{1,6}\s+|>\s*|\[[ xX]?\]\s*)*", line).end()   # list / heading markers
            u, at = _unescaped(line)
            best, score = None, 0.0
            for c in (cands if not dollars else [c for c in cands if "$" in c]):
                sm = SequenceMatcher(None, u, c, autojunk=False)
                if sm.real_quick_ratio() >= need and sm.quick_ratio() >= need:
                    r = sm.ratio()
                    if r > score:
                        best, score = c, r
            if best is not None and score >= need and score < 1.0:
                markup = set("*#_`|>!") | (set("[]") if re.search(r"\[[ xX]\]", u) else set())
                edits = []
                for tag, i1, i2, j1, j2 in SequenceMatcher(None, u, best, autojunk=False).get_opcodes():
                    a, b = u[i1:i2], best[j1:j2]
                    if tag == "equal" or not (a + b).strip() or max(len(a), len(b)) > most or set(a + b) & markup:
                        continue
                    if at[i1] < lead or "\\" in b:
                        continue                                # the line's own markers; never a backslash from the layer
                    if a.strip() and b.strip() and not re.search(r"\w", a + b):
                        continue                                # one punctuation mark for another ("-" / "·"): the layer's
                    if (not a.strip() or not b.strip()) and len((a + b).strip()) <= 1 and (a + b).strip() in ".,;:":
                        continue                                # a dot the layout added or left out: not worth it
                    edits.append((at[i1], at[i2], b if dollars else re.sub(r"([$])", r"\\\1", b)))
                if edits:
                    new = line
                    for x0, x1, b in reversed(edits):
                        new = new[:x0] + b + new[x1:]
                    pending.append((len(out), new + tail, len(edits)))
                else:
                    same += 1                                   # differs in Markdown syntax only
            elif best is not None:
                same += 1
            line += tail
        out.append(line)
    if len(pending) >= 3 and same < 0.5 * len(pending):
        return text, 0                                          # the layer disagrees with most lines: it is the one that's wrong
    for k, new, n in pending:
        out[k] = new
        fixed += n
    return "\n".join(out), fixed


def repair_from_layer(md_path, pdf_path):
    """repair_lines on every page that has a text layer (page separators needed) -> character spans fixed"""
    import pypdfium2 as pdfium
    from review import PAGE_SPLIT
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    if len(parts) < 3:
        return 0
    fixed = 0
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(1, len(parts), 2):
            p = int(parts[i])
            if p >= len(pdf):
                continue
            tp = pdf[p].get_textpage()
            try:
                layer = tp.get_text_range().replace("\r", "")
            finally:
                tp.close()
            if len(layer.split()) < 5:
                continue
            from review import tidy                   # the final clean-up escapes printed "$" (\\$): a line still holding "$(1 + ...)$$"
            body, n = repair_lines(tidy(parts[i + 1]), layer.split("\n"))   # looks like maths and was skipped (Case 15 in a chunk)
            if n:                                      # tidy is idempotent: running it here changes nothing at the end
                parts[i + 1] = body
            fixed += n
    finally:
        pdf.close()
    if fixed:
        md_path.write_text(parts[0] + "".join(f"\n\n{{{parts[i]}}}{'-' * 48}\n\n{parts[i + 1].strip(chr(10))}"
                                              for i in range(1, len(parts), 2)) + "\n", encoding="utf-8")
    return fixed


def repair_indic(md_path, pdf_path):
    """Marker's text extraction stores Indic vowel signs in visual order (আমাদের -> আমাদরে, দাবিতে -> দাবতিে) where the PDF's
    own text (pdfium) is right. Per page: when pdfium's Indic words are well-formed, an output word that differs from the
    one pdfium word with the same consonants is replaced by it. Needs the page separators. -> words fixed"""
    import pypdfium2 as pdfium
    from review import PAGE_SPLIT
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    if len(parts) < 3:
        return 0
    word = re.compile(r"[\u0900-\u0dff\u200c\u200d]+")
    fixed = 0
    pdf = pdfium.PdfDocument(str(pdf_path))
    try:
        for i in range(1, len(parts), 2):
            p = int(parts[i])
            if p >= len(pdf) or not INDIC.search(parts[i + 1]):
                continue
            tp = pdf[p].get_textpage()
            try:
                from review import INDIC_SPLIT                 # "ঢাক ার" (a line break inside the word) joined first
                ref = word.findall(INDIC_SPLIT.sub("", tp.get_text_range().replace("\r", " ").replace("\n", " ")))
            finally:
                tp.close()
            if len(ref) < 5 or _indic_invalid(ref) > 0.01 * len(ref):
                continue                                   # pdfium's text is not trustworthy here either
            by_skel = {}
            for w in ref:
                by_skel.setdefault(_indic_skeleton(w), set()).add(w)
            have = set(ref)

            def fix(m):
                nonlocal fixed
                w = m.group(0)
                cands = by_skel.get(_indic_skeleton(w), set())
                if w in have or len(cands) != 1 or len(_indic_skeleton(w)) < 2:
                    return w
                fixed += 1
                return next(iter(cands))
            parts[i + 1] = word.sub(fix, parts[i + 1])
            parts[i + 1], n = _align_indic(parts[i + 1], ref, word)
            fixed += n
    finally:
        pdf.close()
    md_path.write_text(parts[0] + "".join(f"\n\n{{{parts[i]}}}{'-' * 48}\n\n{parts[i + 1].strip(chr(10))}"
                                          for i in range(1, len(parts), 2)) + "\n", encoding="utf-8")
    return fixed


def page_list(page_range, default=40):
    """"3-7,9" -> [3..7, 9]; no range: the first `default` pages"""
    if not page_range:
        return list(range(default))
    return sorted({p for part in str(page_range).split(",") for p in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)})


def ocr_reason(pdf_path, pages):
    """why the text layer of these pages can't be used (they must be read with OCR), or None:
    - an old Bangla (Bijoy) font: Latin codes stand for Bangla glyphs;
    - Indic scripts: PDFs store their vowel signs in visual order (আমাদের comes out as আমাদরে, ফেব্রুয়ারি as ফব্রুয়ারি);
    - glyphs without Unicode: dots ("····· ···"), U+FFFD or private-use characters instead of letters"""
    font = legacy_font(pdf_path)
    if font:
        return f"old Bangla font {font} (its text layer is not Unicode)"
    try:
        import pypdfium2 as pdfium
        pdf = pdfium.PdfDocument(str(pdf_path))
    except Exception:
        return None
    indic = broken = dots = 0
    try:
        for p in pages:
            if p >= len(pdf):
                continue
            tp = pdf[p].get_textpage()
            try:
                t = tp.get_text_range()
            finally:
                tp.close()
            indic += len(INDIC.findall(t)); broken += len(BROKEN.findall(t)); dots += len(DOT_WORDS.findall(t))
    finally:
        pdf.close()
    if dots or broken >= 5:
        return "text layer with glyphs that have no Unicode (dots or replacement characters)"
    return None


# ----------------------------------------------------------------- joining page chunks of one long PDF
# Marker keeps the original page numbers when it converts a page range, so image names (_page_37_Picture_2.jpeg) and
# meta page ids never collide between chunks: joining is concatenation in page order.
REMOTE_MARK = "<!-- mdbox:remote-pages {} -->"


def _take_chunk(chunk_dir, doc_dir, stem, meta):
    """moves a chunk's images into doc_dir, merges its meta into `meta`, returns its markdown"""
    chunk_dir, doc_dir = Path(chunk_dir), Path(doc_dir)
    text = (chunk_dir / f"{stem}.md").read_text(encoding="utf-8")
    mp = chunk_dir / f"{stem}_meta.json"
    if mp.is_file():
        m = json.loads(mp.read_text(encoding="utf-8"))
        for k, v in m.items():
            if isinstance(v, list) and k in ("table_of_contents", "page_stats"):
                meta.setdefault(k, []).extend(v)
            else:
                meta.setdefault(k, v)
    for f in chunk_dir.iterdir():
        if f.is_file() and f.name not in (f"{stem}.md", f"{stem}_meta.json"):
            os.replace(f, doc_dir / f.name)
    if (chunk_dir / "layout").is_dir():               # --show-layout: each page's detected blocks, drawn (worker.layout_of)
        (doc_dir / "layout").mkdir(exist_ok=True)
        for f in (chunk_dir / "layout").iterdir():
            os.replace(f, doc_dir / "layout" / f.name)
    return text.strip("\n")


def _write_meta(doc_dir, stem, meta):
    for k in ("table_of_contents", "page_stats"):
        if k in meta:
            meta[k].sort(key=lambda e: e.get("page_id", 0) if isinstance(e, dict) else 0)
    (Path(doc_dir) / f"{stem}_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def _failed(pages, why):
    return f"<!-- mdbox: pages {pages} failed: {str(why).replace('--', '-')[:300]} -->"


def merge_chunks(doc_dir, stem, parts):
    """joins chunk outputs into <doc_dir>/<stem>.md (+ one _meta.json, images moved up).
    parts: [(pages, chunk_dir, state)] in page order; state "ok", "remote" (a placeholder the client fills in) or an error."""
    doc_dir = Path(doc_dir); doc_dir.mkdir(parents=True, exist_ok=True)
    texts, meta, failed, remote = [], {}, [], []
    for pages, chunk_dir, state in parts:
        if state == "remote":
            texts.append(REMOTE_MARK.format(pages)); remote.append(pages)
        elif state == "ok" and (Path(chunk_dir) / f"{stem}.md").is_file():
            texts.append(_take_chunk(chunk_dir, doc_dir, stem, meta))
        else:
            texts.append(_failed(pages, state)); failed.append(pages)
    (doc_dir / f"{stem}.md").write_text("\n\n".join(texts) + "\n", encoding="utf-8")
    _write_meta(doc_dir, stem, meta)
    shutil.rmtree(doc_dir / ".chunks", ignore_errors=True)
    return {"failed_pages": failed, "remote_pages": remote}


def fill_remote(doc_dir, stem, pages, chunk_dir, error=None):
    """replaces the placeholder of a chunk converted on the client with its markdown (or a failure note); True if it was found"""
    doc_dir = Path(doc_dir); md_path, mp = doc_dir / f"{stem}.md", doc_dir / f"{stem}_meta.json"
    md, mark = md_path.read_text(encoding="utf-8"), REMOTE_MARK.format(pages)
    if mark not in md:
        return False
    if error or not (Path(chunk_dir) / f"{stem}.md").is_file():
        new = _failed(pages, error or "no output")
    else:
        meta = json.loads(mp.read_text(encoding="utf-8")) if mp.is_file() else {}
        new = _take_chunk(chunk_dir, doc_dir, stem, meta)
        _write_meta(doc_dir, stem, meta)
    md_path.write_text(md.replace(mark, new), encoding="utf-8")
    return True


# ----------------------------------------------------------------- figures Marker extracted but did not link
# Marker saves a crop for every Picture/Figure/Diagram block, but when such a block sits inside a block that renders its own
# HTML (a figure group, a complex region...) its <img> tag is lost: on a scanned book, 28 of 124 figures had no link.
# The worker always asks Marker for page separators, puts each unlinked image at the end of its own page, then removes the
# separators again unless the user asked for them (PAGINATE).
PAGE_SEP = re.compile(r"\n*\{(\d+)\}-{48}\n*")
IMG_NAME = re.compile(r"^_page_(\d+)_[A-Za-z]+_(\d+)\.(?:jpe?g|png|webp|gif)$", re.I)


def _norm(t):
    return re.sub(r"[\W_]+", "", t.lower())


def _insert_before_anchor(text, link, anchor):
    """puts link just before the paragraph containing anchor (a caption); None if the anchor is not in text"""
    key = _norm(anchor)[:40]
    if len(key) < 8:
        return None
    paras = text.split("\n\n")
    for i, para in enumerate(paras):
        if key in _norm(para):
            return "\n\n".join(paras[:i] + [link.strip("\n")] + paras[i:])
    return None


def place_orphan_images(md_path, img_dir, keep_pages=False, anchors=None):
    """links every unlinked image: before its caption (anchors: {name: caption text}) if found on its page, else at the
    end of its page. Returns the number of images it linked"""
    anchors = anchors or {}
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    orphans = sorted(((int(m.group(1)), int(m.group(2)), f.name) for f in Path(img_dir).iterdir()
                      if (m := IMG_NAME.match(f.name)) and f.name not in md))
    parts = PAGE_SEP.split(md)
    pre, pages = parts[0], [[int(parts[i]), parts[i + 1]] for i in range(1, len(parts), 2)]
    for page, _, name in orphans:
        link = f"\n\n![]({name})\n"
        target = next((p for p in reversed(pages) if p[0] <= page), pages[0] if pages else None)
        placed = _insert_before_anchor(target[1] if target else pre, link, anchors[name]) if name in anchors else None
        if placed is not None:
            if target is None:
                pre = placed
            else:
                target[1] = placed
        elif target is None:
            pre = pre.rstrip("\n") + link
        else:
            target[1] = target[1].rstrip("\n") + link
    if keep_pages:
        out = pre + "".join(f"\n\n{{{p}}}{'-' * 48}\n\n{t}" for p, t in pages)
    else:
        out = "\n\n".join(s.strip("\n") for s in [pre] + [t for _, t in pages] if s.strip())
    md_path.write_text(out.strip("\n") + "\n", encoding="utf-8")
    return len(orphans)
