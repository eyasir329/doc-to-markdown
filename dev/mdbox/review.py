"""Page review: the vision LLM checks every page's Markdown against the page itself and writes the final version.

For each page it gets the page image, Marker's draft Markdown of that page and (PDFs with text) the page's text layer,
and returns the corrected page: missing or misread words fixed, headings, lists, GitHub tables, LaTeX math ($...$,
$$...$$), fenced code, paragraphs joined for reading. Figures (image link + "Figure content" quote) and link anchors are
swapped for <<KEEP n>> tokens first, so the LLM cannot drop, move or rewrite them.
A rewrite is only used when it loses nothing (accept()): every KEEP token exactly once, at least WORDS_KEPT of the draft's
words, no fewer words overall, and (with a text layer) at least the draft's share of the page's real words. Otherwise
the page keeps Marker's draft. tidy() then fixes Markdown syntax over the whole document, deterministically.
shrink_images() caps image size so the .md stays light to open (crops are made at 300 DPI)."""
import base64
import io
import json
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

PAGE_SPLIT = re.compile(r"\n*\{(\d+)\}-{48}\n*")
KEEP = re.compile(r"!\[[^\]]*\]\([^)\s]+\)(?:\n\n> \*\*Figure content:\*\*[^\n]*(?:\n>[^\n]*)*(?:\n\n```mermaid\n[\s\S]*?\n```)?)?"   # figure + its description (+ Mermaid)
                  r"|<span id=\"[^\"]*\">\s*</span>")                                                # anchor of an internal link
TOKEN = "<<KEEP {}>>"
WORDS_KEPT = 0.9             # share of the draft's distinct words a rewrite must keep (room for fixed OCR errors)
CHARS_PER_TOKEN = 3.0        # conservative, to size requests to the LLM's context
IMAGE_TOKENS = 1536          # the server's --image-max-tokens
PAGE_PX = 1600               # page render, long side (the server scales it to IMAGE_TOKENS)
IMAGE_MAX_PX = 2400          # images in the output are at most this big (long side): a full page at 300 DPI, so crops keep their detail
WORD = re.compile(r"[^\W_]{3,}")

PROMPT = """You check and correct ONE page of a document that was converted to Markdown.
You get: the page image, the DRAFT Markdown of this page{layer_note}.
Return the corrected Markdown of this page and nothing else (no explanations, no ``` around the whole answer).

Rules:
- Keep ALL content of the page: every word, number, label, list item, table cell, footnote, sidebar and caption, in reading
  order, in the page's own language and script (digits too: ১০ stays ১০). Never summarize, shorten, explain, translate or
  add anything that is not on the page.
- Fix words the draft got wrong or left out, by reading the image{layer_rule}.
- Tokens like <<KEEP 3>> stand for figures and link anchors: keep every one exactly once, at the same place.
- Headings: '#' marks that follow the page's visual hierarchy; keep the draft's heading levels unless clearly wrong.
- Paragraphs: join lines broken only by the page layout; join words hyphenated across a line break.
- Lists: '-' or '1.' items. Tables: GitHub pipe tables (HTML <table> only when cells are merged); if the table on the page
  has no header row, its first row is the header: never invent column names.
- Math: inline $...$ and display $$...$$ (on its own lines), valid LaTeX, for formulas, symbols and variables only.
  Check every symbol against the image: ceiling \\lceil \\rceil vs floor \\lfloor \\rfloor, \\leqslant vs \\le, hats, bars,
  primes, sub- and superscripts. A symbol LaTeX has no command for (a drawn box, a domino) stays that Unicode character in
  \\text{{}} (\\text{{▯}}); an equation's number goes inside it as \\tag{{2.13}}.
- Program code and command listings: fenced ``` blocks (with the language if known), exact characters and indentation.
- A dollar sign for money is written \\$ (\\$34.99), so it is not read as maths.\n- Bold and italic as on the page.{header_rule}

DRAFT:
{draft}{layer}"""


# scripts written without spaces: every character is a token (a whole clause would otherwise be one "word")
NO_SPACE = ((0x3040, 0x30FF), (0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0x20000, 0x2FA1F),   # kana, Han
            (0x0E00, 0x0EFF), (0x1000, 0x109F), (0x1780, 0x17FF))                                          # Thai, Lao, Myanmar, Khmer


def _no_space(ch):
    o = ord(ch)
    return any(a <= o <= b for a, b in NO_SPACE)


import unicodedata


def script_tokens(text, ngram=1):
    """words of any script: letters and digits apart (9V -> 9, v), vowel signs and other combining marks kept with their
    letter (Bangla বাংলা is one word, not ব ল), Chinese/Japanese/Thai one token per character (ngram=2: character pairs,
    for coverage, where single characters would match anywhere)"""
    import unicodedata
    out, cur, run = [], "", ""
    def flush_word():
        nonlocal cur
        if cur:
            out.append(cur); cur = ""
    def flush_run():
        nonlocal run
        if run:
            out.extend(run if ngram == 1 or len(run) == 1 else [run[i:i + ngram] for i in range(len(run) - ngram + 1)]); run = ""
    for ch in text:
        cat = unicodedata.category(ch)
        if _no_space(ch) and cat[0] in "LM":
            flush_word()
            if cat[0] == "M" and run:
                run = run[:-1] + run[-1] + ch if ngram > 1 else run + ch
            else:
                run += ch
            continue
        flush_run()
        if cat[0] == "M" and cur:
            cur += ch
        elif cat[0] in "LN":
            if cur and cur[-1].isdigit() != ch.isdigit():
                flush_word()
            cur += ch
        else:
            flush_word()
    flush_word(); flush_run()
    res = []
    for t in out:                     # a 20+ letter run in a script not listed (Tibetan, Javanese...): no spaces, per character
        letters = [c for c in t if unicodedata.category(c)[0] != "M"]
        if len(letters) >= 20 and not t.isascii() and not t.isdigit():
            chars = re.findall(r".[\u0300-\u036f\u0900-\u0dff\u0f00-\u0fff]*", t)
            res += chars if ngram == 1 else [chars[i] + chars[i + 1] for i in range(len(chars) - 1)]
        else:
            res.append(t)
    return res


GREEK_NAMES = dict(zip("αβγδεζηθικλμνξπρστυφχψω", "alpha beta gamma delta e zeta eta theta iota kappa lambda mu nu xi pi rho "
                                                     "sigma tau upsilon phi chi psi omega".split()))
LATEX_GREEK = re.compile(r"\\(var)?(" + "|".join(sorted(set(GREEK_NAMES.values()) | {"epsilon"}, key=len, reverse=True)) + r")(?![a-z])")
LONE_GREEK = re.compile(r"(?<![\u0370-\u03ff\u1f00-\u1fff])[\u03b1-\u03c9](?![\u0370-\u03ff\u1f00-\u1fff])")


def LATEX_GREEK_NAMES(text):
    """the Greek letters of a formula by name ("lambda"), however written (λ, \\lambda, 𝜆); ε / \\epsilon count as "e"."""
    import unicodedata
    t = unicodedata.normalize("NFKC", text).lower()
    names = ["e" if m.group(2) == "epsilon" else m.group(2) for m in LATEX_GREEK.finditer(t)]
    return names + [GREEK_NAMES[c] for c in t if c in GREEK_NAMES]


def _clean(text):
    """for comparing a draft and its rewrite: a maths letter is the same word as λ, \\lambda or 𝜆 ("lambda"); OCR's ε for an
    ê and \\epsilon / \\varepsilon are all "e" (Greek words, letters next to other Greek letters, stay as they are)"""
    import unicodedata
    text = unicodedata.normalize("NFKC", text).lower()
    text = LATEX_GREEK.sub(lambda m: " " + ("e" if m.group(2) == "epsilon" else m.group(2)) + " ", text)
    text = LONE_GREEK.sub(lambda m: " " + GREEK_NAMES[m.group(0)] + " ", text)
    return re.sub(r"[-\xad\ufffe\x02]\s*\n\s*|[\xad\ufffe\x02]", "", text)     # words hyphenated across lines


def _words(text):
    """words for the distinct-word and text-layer checks (short Latin words left out)"""
    return [t for t in script_tokens(_clean(TOKEN_RE.sub(" ", text))) if len(t) >= 3 or not t.isascii()]


TOKEN_RE = re.compile(r"<<KEEP \d+>>", re.I)


def protect(text):
    """-> (text with figures/anchors replaced by <<KEEP n>>, {token: original})"""
    kept = {}

    def sub(m):
        tok = TOKEN.format(len(kept) + 1); kept[tok] = m.group(0); return tok
    return KEEP.sub(sub, text), kept


def restore(text, kept):
    for tok, orig in kept.items():
        text = text.replace(tok, orig)
    return text


def repair_tokens(new, body, kept):
    """the LLM often drops or repeats a <<KEEP n>>: repeats are removed, a dropped one goes back next to the text that
    surrounded it in the draft (anchors at the start of the paragraph they belong to, figures as their own paragraph)"""
    new = TOKEN_RE.sub(lambda m: m.group(0) if m.group(0) in kept else "", new)   # tokens the LLM made up
    for tok in kept:
        at = new.find(tok)
        if at >= 0:
            new = new[:at + len(tok)] + new[at + len(tok):].replace(tok, "")
            continue
        i = body.find(tok)
        after, before = _words(body[i + len(tok):])[:5], _words(body[:i])[-5:]
        paras = new.split("\n\n")
        norm = [" ".join(_words(p)) for p in paras]
        anchor = kept[tok].startswith("<span")
        j = next((k for k, t in enumerate(norm) if after and " ".join(after) in t), None)
        if j is not None:
            if anchor:
                paras[j] = tok + paras[j]
            else:
                paras.insert(j, tok)
        else:
            j = next((k for k in range(len(norm) - 1, -1, -1) if before and " ".join(before) in norm[k]), None)
            if j is None:                                      # same relative place in the page
                j = min(len(paras), round(len(_words(body[:i])) / max(1, len(_words(body))) * len(paras))) - 1
            paras.insert(j + 1, tok)
        new = "\n\n".join(paras)
    return new


def _tokens(text):
    """every word and number (also 1-2 characters: 9V, mA, IC), LaTeX commands left out (\\text, \\frac...)"""
    text = re.sub(r"</?[a-zA-Z][^>]*>", " ", TOKEN_RE.sub(" ", text))             # HTML tags (<math>, <br>) are markup
    text = re.sub(r"(?m)^([ \t>]*)(```|~~~)[\w+#.-]*", r"\1 ", text)
    text = STALE_NUMBER.sub(r"\1", text)                                          # Marker's "1. 2. item" = the page's "2. item"                                # a code block's language is markup too
    return script_tokens(re.sub(r"\\[a-z]+", " ", _clean(text)))           # 1.2K and 1.2\\text{K} give the same tokens


CONFUSABLE = set(zip("εοννρκιτχαμ", "eovyrkitxau"))     # Greek letters an OCR gives for Latin ones that look the same


def _changes(draft_t, new_t, layer_t):
    """-> (lost, added) word Counters. lost: draft words gone from the rewrite (only words that are on the page, when the
    PDF has a text layer: OCR junk may go). added: rewrite words on neither the draft nor the page, except near-spellings
    of a lost word (an OCR fix: "rcsistor" -> "resistor")"""
    from collections import Counter
    from difflib import SequenceMatcher
    D, N, L = Counter(draft_t), Counter(new_t), Counter(layer_t)
    lost = D - N
    added = Counter({w: n for w, n in (N - (D + L)).items() if not w.isdigit()})     # a layer word the draft lacked once more (a footer)
    added.update({w: n for w, n in (N - (D | L)).items() if w.isdigit()})          # numbers: exact
    for w in list(added):                                   # OCR read a Latin letter as Greek (ê -> ε): the fix is no new word
        for m in list(lost):
            if (w, m) in CONFUSABLE or (m, w) in CONFUSABLE:
                k = min(added[w], lost[m]); added[w] -= k; lost[m] -= k
    added, lost = +added, +lost
    if L:
        lost = Counter({w: n for w, n in lost.items() if w in L})
    for w in list(added):
        for m in list(lost):
            if not w.isdigit() and not m.isdigit() and SequenceMatcher(None, w, m).ratio() >= 0.75:
                k = min(added[w], lost[m]); added[w] -= k; lost[m] -= k
    return +lost, +added


def brackets(text):
    """ceiling / floor brackets of a text, by kind: a rewrite may not turn one into the other (a model "normalises" the
    page's ⌈x⌉ to the far more common ⌊x⌋: Case 15 ex. 49b). Only the audit can, after a second look at the image."""
    return (len(re.findall(r"\\[lr]ceil\b|[⌈⌉]", text)), len(re.findall(r"\\[lr]floor\b|[⌊⌋]", text)))


def accept(draft, new, kept, layer="", detail=None):
    """(True, "") if the rewrite loses and invents nothing, else (False, why); detail (a list) gets what differed"""
    if not new.strip():
        return False, "empty"
    for tok in kept:
        if new.count(tok) != 1:
            return False, "figure/anchor lost or repeated"
    if len(TOKEN_RE.findall(new)) != len(kept):
        return False, "invented token"
    if brackets(new) != brackets(draft):
        return False, "ceiling/floor changed"
    boxes = lambda t: (len(re.findall(r"\[[xX✓✔]\]|[☑☒✅]", t)), len(re.findall(r"\[ \]|☐", t)))
    if boxes(new) != boxes(draft):
        return False, "checkbox changed"                    # Case 25: "[X] Step 1" came back unticked
    heads = lambda t: len(re.findall(r"(?m)^#{1,6} \S", t))
    if heads(new) < heads(draft):
        return False, "heading lost"                        # "## Case 08: ..." flattened into a plain line
    if new.count("\\$") < draft.count("\\$"):
        return False, "escaped $ unescaped"                 # "(\$\$ ... \$\$)" in text would become maths
    dt, nt, lt = _tokens(draft), _tokens(new), _tokens(layer)
    lost, added = _changes(dt, nt, lt)
    shapes = sum(0x25A0 <= ord(c) <= 0x25FF for c in new) - sum(0x25A0 <= ord(c) <= 0x25FF for c in draft)
    for w in ("0", "o"):                                    # OCR's 0 / O for a drawn box (▯ ▭ □) the rewrite restored
        k = min(max(0, shapes), lost.get(w, 0))
        if k:
            lost[w] -= k; shapes -= k
    lost = +lost
    if detail is not None and (lost or added):
        detail.append(f"lost {sorted(lost.elements())[:8]} added {sorted(added.elements())[:8]}")
    if any(w.isdigit() for w in lost) or any(w.isdigit() for w in added):
        return False, "number changed or lost"              # 9V -> 6V, 20 mA -> 2 mA
    if added:
        return False, "words added"                         # on neither the draft nor the page: made up
    if sum(lost.values()) > max(2, 0.01 * len(dt)):         # a symbol the LaTeX spells as a command (Ω -> \\Omega)
        return False, "draft words lost"
    dw, nw = _words(draft), _words(new)
    if len(set(dw)) >= 15 and len(set(dw) & set(nw)) < WORDS_KEPT * len(set(dw)):
        return False, "draft words lost"
    lw = _words(layer)
    if len(lw) >= 20:
        cov = lambda ws: sum(w in ws for w in lw) / len(lw)
        if cov(set(nw)) < cov(set(dw)) - 0.01:
            return False, "page text lost"
    return True, ""


def _unwrap(text):
    t = text.strip()
    m = re.fullmatch(r"```(?:markdown|md)?\s*\n(.*)\n```", t, re.S)
    return (m.group(1) if m else t).strip()


import threading as _threading
LLM_GATE = _threading.BoundedSemaphore(10 ** 6)       # LLM requests this worker may have in flight (set_llm_inflight)


def set_llm_inflight(n):
    """at most n LLM requests in flight from this worker (figures, review, audit together; Marker's own steps aside). The
    nested pools (describe x graph answers, review x audit) could otherwise put hundreds in the servers' queues, and a request
    waiting longer than its 900 s timeout makes a page keep its unreviewed draft. 0 = no cap."""
    global LLM_GATE
    LLM_GATE = _threading.BoundedSemaphore(int(n) if n and int(n) > 0 else 10 ** 6)


def ask_page(llm_url, model, jpeg, prompt, max_tokens, timeout=900, schema=None):
    """-> (text, finish_reason); schema: a JSON schema the answer must follow"""
    body = {"model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": prompt}]}]}
    if schema:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "answer", "strict": True, "schema": schema}}
    req = urllib.request.Request(llm_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
    with LLM_GATE:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            c = json.load(r)["choices"][0]
    return c["message"]["content"] or "", c.get("finish_reason")


def page_inputs(src, page_ids):
    """{page id: (jpeg bytes, text layer)} rendered one after another (pdfium is not thread-safe)"""
    out, src = {}, Path(src)
    if src.suffix.lower() != ".pdf":
        try:
            im = Image.open(src).convert("RGB")
        except Exception:
            return out                                     # not a page image (e.g. EPUB/HTML): nothing to compare with
        im.thumbnail((PAGE_PX, PAGE_PX)); buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
        return {p: (buf.getvalue(), "") for p in page_ids}
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(src))
    try:
        for p in page_ids:
            if p >= len(pdf):
                continue
            page = pdf[p]
            w, h = page.get_size()
            im = page.render(scale=PAGE_PX / max(w, h)).to_pil().convert("RGB")
            buf = io.BytesIO(); im.save(buf, "JPEG", quality=90)
            tp = page.get_textpage()
            try:
                layer = tp.get_text_range()
            finally:
                tp.close()
            out[p] = (buf.getvalue(), layer.replace("￾", "-").replace("\r", ""))
    finally:
        pdf.close()
    return out


def _need(s):
    """tokens a text takes: ~3 Latin characters per token, but about one per character in Bangla, Chinese, Japanese..."""
    ascii_n = sum(1 for c in s if c < "\x80")
    return int(ascii_n / CHARS_PER_TOKEN + 1.3 * (len(s) - ascii_n))


def _review_part(body, kept, jpeg, layer, llm_url, model, room, keep_headers, part=""):
    """one LLM call for (a part of) a page: -> (rewritten text with tokens, None) or (None, why)"""
    out_tokens = int(2.0 * _need(body)) + 800              # tables and LaTeX take more tokens than the draft (1.6x + 600 cut 3 suite pages off)
    if _need(body) + _need(layer) + out_tokens > room:
        layer = ""                                          # dense page: the image alone must do
    prompt = PROMPT.format(
        layer_note=", and the TEXT LAYER of the PDF page (exact characters, but its reading order and layout may be wrong)" if layer else "",
        layer_rule=" and the text layer" if layer else "",
        header_rule="",                                    # removing headers/page numbers would fail the number check
        draft=body, layer=f"\n\nTEXT LAYER:\n{layer}" if layer else "")
    if part:
        prompt = prompt.replace("Return the corrected Markdown of this page", f"The DRAFT is {part} of this page only: return the corrected Markdown of that part")
    try:
        text, finish = ask_page(llm_url, model, jpeg, prompt, min(out_tokens, room - _need(body) - _need(layer)))
    except Exception as e:
        return None, f"LLM error: {type(e).__name__}"
    if finish == "length":
        return None, "answer cut off"
    if not _unwrap(text).strip():
        return None, "empty"
    new = repair_tokens(_unwrap(text), body, kept)
    detail = []
    ok, why = accept(body, new, kept, layer, detail)
    if ok:
        return new, None
    merged, taken = merge_paragraphs(body, new, kept, layer)    # the good paragraphs of a rewrite with one bad one
    if taken:
        return merged, None
    return None, f"{why}|{detail[0]}" if detail else why


def layer_prefers(draft, new, layer, need=0.9):
    """the PDF's own text settles a paragraph the strict check refused: every word and number of the rewrite is on the page,
    and the stretch of the page's text it matches best agrees with the rewrite (>= 90%) more than with the draft
    (Case 15: the OCR's \\sqrt[2]{2} where the page prints \\sqrt{2}; a rewrite that drops a real number matches worse)"""
    from collections import Counter
    from difflib import SequenceMatcher
    lt, nt, dt = _tokens(layer), _tokens(new), _tokens(draft)
    if not nt or Counter(nt) - Counter(lt):
        return False

    def best(t):
        n, top = len(t), 0.0
        for i in range(0, max(1, len(lt) - n + 1)):
            top = max(top, SequenceMatcher(None, t, lt[i:i + n], autojunk=False).ratio())     # a stretch as long as it
        return top
    rn = best(nt)
    return rn >= need and rn > best(dt)


def _take(da, nb, kept, layer):
    """-> the rewrite nb of draft paragraph(s) da if it may be used, else None"""
    sub = {t: v for t, v in kept.items() if t in da}
    if not nb.strip() or any(nb.count(t) != 1 for t in sub) or len(TOKEN_RE.findall(nb)) != len(sub):
        return None
    verdict = accept(da, nb, sub, layer)
    if not verdict[0] and verdict[1] == "escaped $ unescaped":
        loose = re.findall(r"(?<!\\)\$", nb)                   # printed "$...$" text the rewrite turned into maths:
        if len(loose) == da.count("\\$"):                     # escaped again, its other fixes (\sqrt[2] -> \sqrt) kept
            nb = re.sub(r"(?<!\\)\$", lambda m: "\\$", nb)
            verdict = accept(da, nb, sub, layer)
    if not verdict[0] and layer and verdict[1] in ("number changed or lost", "words added", "draft words lost"):
        verdict = (layer_prefers(da, nb, layer), "")
    return nb if verdict[0] else None


def merge_paragraphs(body, new, kept, layer=""):
    """paragraph by paragraph: each changed group of the rewrite is used if it passes on its own (accept(), or the PDF's
    text agrees with it), else the draft's stays: one made-up line no longer throws away a whole page of fixes. A group
    with as many paragraphs on both sides is judged pair by pair. -> (text, groups taken)"""
    from difflib import SequenceMatcher
    a, b = body.split("\n\n"), new.split("\n\n")
    norm = lambda p: " ".join(p.split())
    out, taken = [], 0
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, [norm(x) for x in a], [norm(x) for x in b], autojunk=False).get_opcodes():
        if tag == "equal":
            out += b[j1:j2]                                    # same words, the rewrite's spacing
            continue
        pairs = list(zip(a[i1:i2], b[j1:j2])) if i2 - i1 == j2 - j1 else [("\n\n".join(a[i1:i2]), "\n\n".join(b[j1:j2]))]
        for da, nb in pairs:
            got = _take(da, nb, kept, layer)
            if got is not None:
                out.append(got); taken += 1
            elif da.strip():
                out.append(da)                                 # nothing of the draft is ever dropped
    return "\n\n".join(x for x in out if x.strip()), taken


def _split(body, limit):
    """paragraph groups of at most `limit` tokens each (a dense page in parts)"""
    parts, cur = [], []
    for para in body.split("\n\n"):
        if cur and _need("\n\n".join(cur + [para])) > limit:
            parts.append("\n\n".join(cur)); cur = []
        cur.append(para)
    return parts + (["\n\n".join(cur)] if cur else [])


def layer_for(part, layer, share=0.5):
    """the lines of a page's text layer that belong to one part of it (half their words or more are in the part): a part
    reviewed on its own must not be handed the rest of the page, or words of other parts could be copied in"""
    have = set(_words(part))
    keep = [l for l in layer.splitlines() if _words(l) and sum(w in have for w in _words(l)) >= share * len(_words(l))]
    return "\n".join(keep)


def review_page(draft, jpeg, layer, llm_url, model, ctx, keep_headers=False, part_tokens=0):
    """-> (final page text, None if rewritten else why the draft was kept). part_tokens: review the page in sections of at
    most this many tokens (each against the page image and its own lines of the text layer): more careful, more calls"""
    if not re.search(r"\w", draft) and not layer.strip():
        return draft, "blank page"
    body, kept = protect(draft)
    room = ctx - IMAGE_TOKENS - 600                         # prompt, chat template
    hard = int((room - 600) / 2.6)                          # draft + 1.6x draft + 600 must fit
    limit = min(hard, part_tokens) if part_tokens else hard
    parts = [body] if _need(body) <= limit else _split(body, limit)
    if any(_need(x) > hard for x in parts):                 # (a paragraph longer than a section is a section of its own)
        return draft, "page too long for the LLM context"   # one paragraph alone is too long
    out, whys = [], []
    for k, part in enumerate(parts):
        mine = {t: v for t, v in kept.items() if t in part}
        new, why = _review_part(part, mine, jpeg, layer if len(parts) == 1 else layer_for(part, layer), llm_url, model, room, keep_headers,
                                f"part {k + 1} of {len(parts)}" if len(parts) > 1 else "")
        out.append(new if why is None else part); whys.append(why)
    if all(whys):
        return draft, whys[0]
    return restore("\n\n".join(out), kept), next((w for w in whys if w), None)


def page_settled(draft, layer):
    """a page the review can't improve in content: every word and number of it is in the PDF's own text (no OCR), nothing of
    that text is left out (a running header aside), and it has no maths, code or table to convert. Digital books are mostly
    such pages. (Its formatting, e.g. heading levels, is Marker's.)"""
    from collections import Counter
    body = TOKEN_RE.sub(" ", protect(draft)[0])
    if not layer.strip() or re.search(r"(?<!\\)\$|```|^\s*\|", body, re.M):
        return False
    d, l = Counter(_tokens(body)), Counter(_tokens(layer))
    if sum(d.values()) < 20 or d - l:
        return False
    return sum((l - d).values()) <= max(8, 0.03 * sum(l.values()))


def review_pages(md_path, src, llm_url, model, concurrency=2, ctx=8192, keep_headers=False, use_layer=True, part_tokens=0,
                 after=None, skip_settled=False):
    """reviews every page of a Markdown file written with page separators (in place).
    after(text, jpeg, layer) -> text: a next step for each page as soon as its review is done (the audit), so no page waits
    for the slowest page's review. -> {"pages", "rewritten", "kept_draft": {why: count}}"""
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    pre, pages = parts[0], [[int(parts[i]), parts[i + 1]] for i in range(1, len(parts), 2)]
    inputs = page_inputs(src, [p for p, _ in pages])
    todo = [k for k, (p, _) in enumerate(pages) if p in inputs]

    def one(k):
        p, text = pages[k]
        jpeg, layer = inputs[p]
        layer = layer if use_layer else ""
        if skip_settled and page_settled(text, layer):        # balanced: the PDF's own text already is this page
            new, why = text.strip("\n"), "settled by the text layer"
        else:
            new, why = review_page(text.strip("\n"), jpeg, layer, llm_url, model, ctx, keep_headers, part_tokens)
        return k, ((after(new, jpeg, layer) if after else new), why)
    stats = {"pages": len(todo), "rewritten": 0, "kept_draft": {}, "examples": []}
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        for k, (text, why) in ex.map(one, todo):
            pages[k][1] = text
            if why:
                reason, _, what = why.partition("|")
                stats["kept_draft"][reason] = stats["kept_draft"].get(reason, 0) + 1
                if what and len(stats["examples"]) < 10:
                    stats["examples"].append(f"page {pages[k][0]}: {reason}: {what}")
            else:
                stats["rewritten"] += 1
    md_path.write_text(pre + "".join(f"\n\n{{{p}}}{'-' * 48}\n\n{t.strip(chr(10))}" for p, t in pages) + "\n", encoding="utf-8")
    return stats


# ----------------------------------------------------------------- line-by-line audit (accurate quality)
# A whole-page rewrite keeps missing small things: an OCR's 李生 for 孪生, \sqrt[2]{2} for the printed \sqrt{2}, y^n for y^m.
# Each section of a page is proofread against the image; the model lists exact corrections, and each one is applied on its own
# only when it is a small local change that the PDF's own text confirms, or that a second, independent look at the image
# (which of the two versions is printed?) picks.
AUDIT_PROMPT = """You proofread one SECTION of a document page that was converted to Markdown, against the page image (and the
page's text layer when there is one: its characters are exact, its order may not be).
Go through the section line by line and compare every word, number, symbol and formula with the image. Look closely at:
characters an OCR confuses (Chinese/Japanese look-alike characters, l/1, O/0, rn/m), maths symbols (ceiling = \\lceil \\rceil
vs floor = \\lfloor \\rfloor, \\sqrt{2} vs \\sqrt[2]{2}, primes, hats, sub- and superscripts), and drawn symbols LaTeX has no
command for (write them as Unicode, e.g. \\text{▯}).
For every error, give "find": the exact wrong text copied from the SECTION, only the wrong word(s) with a few words around them
so it occurs once (under 60 characters, never a whole paragraph), "replace": the same text corrected, and "why" (a few words). Change only what is wrong; keep the Markdown formatting and escapes (\\$, \\_) as they are. Tokens
like <<KEEP 2>> are figures: never touch them. Do not list text that is already right. No error: an empty list.

SECTION:
<<SECTION>>

TEXT LAYER:
<<LAYER>>"""
AUDIT_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["fixes"], "properties": {"fixes": {"type": "array",
    "items": {"type": "object", "additionalProperties": False, "required": ["find", "replace", "why"],
              "properties": {"find": {"type": "string"}, "replace": {"type": "string"}, "why": {"type": "string"}}}}}}
VERIFY_PROMPT = """Look carefully at the page image. A passage of it reads: «<<CONTEXT>>»
Which of these two is what the page actually shows at that place?
A: «<<A>>»
B: «<<B>>»
Answer "A", "B", or "neither" if the image shows neither or you cannot tell."""
VERIFY_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["answer"],
                 "properties": {"answer": {"type": "string", "enum": ["A", "B", "neither"]}}}
AUDIT_SECTION_PARAS = 6


def _find_once(text, find):
    """where an audit fix applies: its "find" exactly once in the text (also with doubled backslashes undone, or with any
    spacing) -> the matched text, or None"""
    for c in dict.fromkeys((find, find.replace("\\\\", "\\"))):
        if c.strip() and text.count(c) == 1:
            return c
        words = c.split()
        if words:
            ms = list(re.finditer(r"\s+".join(map(re.escape, words)), text))
            if len(ms) == 1:
                return ms[0].group(0)
    return None


def _local_change(a, b):
    """a correction, not a rewrite: similar and about as long"""
    from difflib import SequenceMatcher
    return 0 < len(a) <= 600 and len(b) <= 1.5 * len(a) + 20 and SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.5


def _verify(jpeg, llm_url, model, context, old, new):
    """a second, independent look: is the proposed text what the page shows? Asked twice, the two versions in both orders:
    both answers must pick the new text (a model that leans to "A", or to the more common symbol, can't confirm a fix:
    Cases 1/5/15, accurate: ceiling -> floor was "confirmed" by one question) -> True / False"""
    def ask(new_first):
        a, b = (new, old) if new_first else (old, new)
        prompt = VERIFY_PROMPT.replace("<<CONTEXT>>", context[:600]).replace("<<A>>", a).replace("<<B>>", b)
        try:
            text, finish = ask_page(llm_url, model, jpeg, prompt, 20, schema=VERIFY_SCHEMA)
            return json.loads(text).get("answer") == ("A" if new_first else "B")
        except Exception:
            return False
    return ask(True) and ask(False)                    # in turn: the second reuses the image; a first "no" ends it


def audit_section(section, jpeg, layer, llm_url, model, stats):
    """-> the section with the audit's confirmed corrections applied (stats: counts by outcome, examples)"""
    body, kept = protect(section)
    prompt = AUDIT_PROMPT.replace("<<SECTION>>", body).replace("<<LAYER>>", layer_for(body, layer) or "(none)")
    try:
        text, finish = ask_page(llm_url, model, jpeg, prompt, 800, schema=AUDIT_SCHEMA)     # corrections are short
        fixes = json.loads(text).get("fixes") or [] if finish != "length" else []
    except Exception:
        stats["errors"] = stats.get("errors", 0) + 1
        return section
    import threading
    lock = stats.setdefault("_lock", threading.Lock())

    def count(k):
        with lock:
            stats[k] = stats.get(k, 0) + 1
    for fx in fixes:
        find, new = str(fx.get("find", "")), str(fx.get("replace", ""))
        if not find.strip() or find == new or find.replace("\\\\", "\\") == new.replace("\\\\", "\\"):
            continue                                            # "no error found" listed anyway
        old = _find_once(body, find)
        if old is None:
            count("not found"); continue
        if old != find and "\\\\" in find and old == find.replace("\\\\", "\\"):
            new = new.replace("\\\\", "\\")                     # the answer doubled its backslashes: undo in both
        if not _local_change(old, new) or TOKEN_RE.findall(old) != TOKEN_RE.findall(new):
            count("too big"); continue
        marks = lambda t: (len(re.findall(r"(?<!\\)\$", t)), len(re.findall(r"(?<!\\)[*_`]", t)))
        if marks(new) < marks(old):
            count("would drop maths or formatting"); continue   # "*stick*" -> "stick", "$\\lambda_i$" -> "lambda_i"
        candidate = body.replace(old, new, 1)
        para = next((p for p in body.split("\n\n") if old in p), old)
        plain = lambda x: re.sub(r"\\([$_*#])", r"\1", x).strip()        # Markdown escapes: the text layer has none
        if layer and len(plain(old)) >= 8 and plain(old) in layer:      # the draft has the page's exact characters there
            count("the page's text has it as written"); continue  # run 1-3 suite: "Figure 2.1 T" (cut off on the page) -> "Figure 2.1"
        if layer and ((plain(old) not in layer and plain(new) in layer) or layer_prefers(para, para.replace(old, new, 1), layer)):
            how = "confirmed by the text layer"
        elif _verify(jpeg, llm_url, model, para, old, new):
            how = "confirmed by a second look"
        else:
            count("refused"); continue
        body = candidate
        count(how)
        with lock:
            if len(stats.setdefault("examples", [])) < 12:
                stats["examples"].append(f"{old[:60]} -> {new[:60]} ({how})")
    return restore(body, kept)


def settled(section, layer):
    """a section the audit can't improve: every word and number of it is in the PDF's own text (no OCR in it) and it has
    no maths, code or table, where a converted symbol could still be wrong. Real digital PDFs are mostly such pages."""
    from collections import Counter
    if not layer or re.search(r"(?<!\\)\$|```|^\s*\|", section, re.M):
        return False
    words = _tokens(TOKEN_RE.sub(" ", section))
    return len(words) >= 5 and not (Counter(words) - Counter(_tokens(layer)))


def audit_page(text, jpeg, layer, llm_url, model, stats, sizes=(AUDIT_SECTION_PARAS,), workers=1):
    """one page, pass after pass (sections of `sizes[i]` paragraphs; a second pass in smaller sections finds what the first
    missed). Sections go one after the other: the server keeps the page image from the previous call (2.8 s of GPU time
    saved per call), and many pages are audited at once anyway, so the GPUs stay busy. -> the page with the corrections"""
    import threading
    lock = stats.setdefault("_lock", threading.Lock())
    for size in sizes:
        paras = text.strip("\n").split("\n\n")
        secs = ["\n\n".join(paras[j:j + int(size)]) for j in range(0, len(paras), int(size))]

        def one(sec):
            if not re.search(r"\w", sec):
                return sec
            if settled(sec, layer):
                with lock:
                    stats["settled by the text layer"] = stats.get("settled by the text layer", 0) + 1
                return sec
            with lock:
                stats["sections"] = stats.get("sections", 0) + 1
            return audit_section(sec, jpeg, layer, llm_url, model, stats)
        with ThreadPoolExecutor(max(1, min(workers, len(secs)))) as ex:
            text = "\n\n".join(ex.map(one, secs))
    return text


def _public(stats):
    return {k: v for k, v in stats.items() if not k.startswith("_")}


def audit_pages(md_path, src, llm_url, model, concurrency=2, use_layer=True, sizes=(AUDIT_SECTION_PARAS,)):
    """proofreads every page of a Markdown file with page separators (in place) -> stats. (With the page review on, the
    worker audits each page right after its review instead: review_pages(after=...))"""
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    pre, pages = parts[0], [[int(parts[i]), parts[i + 1]] for i in range(1, len(parts), 2)]
    inputs = page_inputs(src, [p for p, _ in pages])
    stats = {}

    def one(k):
        jpeg, layer = inputs[pages[k][0]]
        return k, audit_page(pages[k][1], jpeg, layer if use_layer else "", llm_url, model, stats, sizes)
    with ThreadPoolExecutor(max(1, concurrency)) as ex:
        for k, text in ex.map(one, [k for k, (p, _) in enumerate(pages) if p in inputs]):
            pages[k][1] = text
    md_path.write_text(pre + "".join(f"\n\n{{{p}}}{'-' * 48}\n\n{t.strip(chr(10))}" for p, t in pages) + "\n", encoding="utf-8")
    return _public(stats)


# ----------------------------------------------------------------- quality: fast / balanced / accurate
QUALITY_PROFILES = {
    # Marker's own LLM steps (+ its inline-maths rewrite), figure descriptions (crop to the figure, flow-diagram graph
    # answers/checks), page review (sections of at most N tokens; 0 = the whole page at once), line-by-line audit (paragraphs
    # per section, one pass per number). Everything without an LLM (tables, drawn formulas, text-layer repair, Indic repair,
    # tidy-up) runs at every quality.
    # fast: no Marker LLM steps and no page rewrite (the two big costs); the cheap checks that fix the most stay: figures
    #       cropped to themselves, one graph read, and a light audit (short answers: only the corrections)
    "fast": dict(MARKER_LLM=False, REDO_INLINE_MATH=False, DESCRIBE_FIGURES=True, CORE_CROP=True, GRAPH_TRIES=1, GRAPH_CHECKS=0,
                 PAGE_REVIEW=False, REVIEW_PART_TOKENS=0, REVIEW_SKIP_SETTLED=True, AUDIT=True, AUDIT_SECTIONS=(12,),
                 TRANSCRIBE_TEXT_IMAGES=False),
    # balanced: a page its PDF text settles completely is not rewritten (only its formatting could change); a picture that is
    #       mostly text (a newspaper page, a letter) is transcribed tile by tile, not only described
    "balanced": dict(MARKER_LLM=True, REDO_INLINE_MATH=False, DESCRIBE_FIGURES=True, CORE_CROP=True, GRAPH_TRIES=2, GRAPH_CHECKS=2,
                     PAGE_REVIEW=True, REVIEW_PART_TOKENS=0, REVIEW_SKIP_SETTLED=True, AUDIT=True, AUDIT_SECTIONS=(8,),
                     TRANSCRIBE_TEXT_IMAGES=True),
    # accurate: nothing left out; the review section by section and two audit passes (the second in smaller sections finds
    #       what the first missed: run 1 of the suite caught 数字李生 but not 音服热线)
    "accurate": dict(MARKER_LLM=True, REDO_INLINE_MATH=True, DESCRIBE_FIGURES=True, CORE_CROP=True, GRAPH_TRIES=3, GRAPH_CHECKS=2,
                     PAGE_REVIEW=True, REVIEW_PART_TOKENS=900, REVIEW_SKIP_SETTLED=False, AUDIT=True, AUDIT_SECTIONS=(6, 3),
                     TRANSCRIBE_TEXT_IMAGES=True),
}


def quality_option(opts, key):
    """an option as the job's quality sets it, unless the job sets it itself"""
    if opts.get(key) is not None:
        return opts[key]
    return QUALITY_PROFILES.get(str(opts.get("QUALITY") or "balanced").lower(), QUALITY_PROFILES["balanced"])[key]


# ----------------------------------------------------------------- LLM answers in JSON that carry LaTeX
# The model writes "\boldsymbol{\rho}" inside a JSON string with one backslash: JSON then reads \b as a backspace and \r
# as a carriage return, and the equation becomes "oldsymbol{<newline>ho}". Before parsing, a lone backslash in front of a
# LaTeX command is doubled. Only command names are touched, so real JSON escapes (\n before a word, \u00e9) stay as they are.
LATEX_CMDS = ("alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa lambda mu nu xi pi varpi rho varrho "
              "sigma varsigma tau upsilon phi varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
              "frac dfrac tfrac sqrt sum prod int iint oint lim log ln exp sin cos tan sec csc cot arcsin arccos arctan sinh cosh "
              "tanh min max sup inf det dim ker deg gcd Pr mod pmod bmod nabla partial infty times div cdot cdots ldots dots vdots "
              "ddots pm mp neq ne leq geq le ge ll gg approx equiv sim simeq cong propto in notin ni subset subseteq supset "
              "supseteq cup cap setminus emptyset varnothing forall exists nexists neg not land lor to rightarrow leftarrow "
              "Rightarrow Leftarrow leftrightarrow Leftrightarrow mapsto implies iff uparrow downarrow left right big Big bigg Bigg "
              "bigl bigr langle rangle lfloor rfloor lceil rceil lvert rvert lVert rVert vert Vert mathbf mathbb mathcal mathrm "
              "mathit mathsf mathtt mathfrak boldsymbol bm text textbf textit textrm texttt textsf emph hat bar tilde vec dot ddot "
              "breve check acute grave widehat widetilde overline underline overbrace underbrace overrightarrow binom tbinom "
              "dbinom choose begin end quad qquad hspace vspace mbox operatorname displaystyle textstyle scriptstyle prime circ "
              "bullet star ast dagger perp parallel mid nmid angle triangle square top bot hbar ell Re Im aleph wp neq nleq ngeq "
              "nsubseteq newline tag label limits nolimits substack stackrel overset underset cases matrix pmatrix bmatrix vmatrix "
              "array hline frown smile because therefore").split()
_LATEX_ESC = re.compile(r"(?<!\\)\\(?=(?:" + "|".join(sorted(set(LATEX_CMDS), key=len, reverse=True)) + r")(?![A-Za-z]))")


def latex_json_loads(raw):
    """json.loads for LLM answers: a lone backslash before a LaTeX command is doubled first"""
    import json
    return json.loads(_LATEX_ESC.sub(r"\\\\", raw))


CHATTER = re.compile(r"(?i)\b(the (image|picture|figure|equation|text|table|block)( provided)? (shows|appears|contains|seems|is (likely|probably|a))|"
                     r"appears to (be|show)|is likely a typo|(likely|probably) (a|an) (typo|error|mistake)|i (cannot|can't|am unable|could not)|"
                     r"as an ai|here is the (corrected|latex)|"
                     r"(is|are) (already )?(correct(ly)?|properly|well)[ -]?(formatted|written|aligned)|the (current )?latex( code)? is|"
                     r"should be (written|placed|enclosed|formatted|aligned|separated)|needs? to be (separated|placed|aligned|enclosed|written)|"
                     r"to ensure (katex|mathjax|proper|correct)|\bmath tags?\b)\b")       # ("No corrections needed" is Marker's own "keep it")
MATH_MARK = re.compile(r"<math|\\[a-zA-Z]+|\$|[=^_]")
MARKER_INPUT = re.compile(r"\*\*Input:\*\*\s*```html\n(.*)\n```\s*$", re.S)


def check_marker_llm(prompt, response):
    """Marker's own LLM steps (inline maths, equations, tables...) rewrite blocks without any check. -> (keep, why).
    A rewrite is dropped (Marker keeps its own text) when it talks about the image instead of transcribing it, and, for a
    text block rewritten for its maths, when it loses or changes words or numbers of the original (accept())."""
    if not isinstance(response, dict):
        return True, ""
    m = MARKER_INPUT.search(prompt or "")
    original = m.group(1) if m else ""
    for key, value in response.items():
        if key == "analysis" or not isinstance(value, str):
            continue
        if CHATTER.search(value) and not CHATTER.search(original):
            return False, "commentary instead of content"
        if (original and re.search(r"<math|\\[a-zA-Z]+", original) and len(value.split()) >= 8
                and not MATH_MARK.search(value) and "no corrections needed" not in value.lower()):
            return False, "maths replaced by prose"           # an equation came back as sentences about it
    for key in ("corrected_equation", "corrected_html", "html"):
        v = response.get(key)
        if original and isinstance(v, str) and "no corrections needed" not in v.lower() and brackets(v) != brackets(original):
            return False, "ceiling/floor changed"
    eq = response.get("corrected_equation")
    if original and isinstance(eq, str) and "no corrections needed" not in eq.lower():
        greek = lambda t: set(LATEX_GREEK_NAMES(t))
        before, after = greek(original), greek(eq)
        if before and after and not (before & after):
            return False, "symbols swapped"                   # Case 03: λ and ê came back as \\boldsymbol{\\nu}
    if original and isinstance(response.get("corrected_html"), str) and "no corrections needed" not in response["corrected_html"].lower():
        ok, why = accept(original, response["corrected_html"], {})
        if not ok:
            return False, why
    return True, ""


# ----------------------------------------------------------------- heading levels from the PDF's own outline
def _key(t):
    import unicodedata
    return "".join(script_tokens(unicodedata.normalize("NFKC", re.sub(r"<[^>]+>", "", t)).lower()))   # keeps vowel signs


def outline(src):
    """the PDF's bookmarks: [(level 0.., title key, page index)]; [] if it has (almost) none"""
    if not str(src).lower().endswith(".pdf"):
        return []
    import pypdfium2 as pdfium
    pdf = pdfium.PdfDocument(str(src))
    try:
        marks = []
        for b in pdf.get_toc(max_depth=8):
            d, title = b.get_dest(), (b.get_title() or "").strip()
            if d is not None and d.get_index() is not None and len(_key(title)) >= 3:
                marks.append((b.level, _key(title), d.get_index()))
    finally:
        pdf.close()
    return marks if len(marks) >= 3 else []


def apply_outline(md_path, marks):
    """page-by-page review cannot see the book's structure (Make Electronics came out with chapters, experiments and their
    sections all as ##). The PDF's outline has it: a heading matching a bookmark on its page (+-1) gets that bookmark's
    level (chapter #, experiment ##, section ###); a heading the outline doesn't list goes one level below the outline
    entry it follows; a short paragraph that is exactly a bookmark title becomes that heading. -> {"matched", "below", "promoted"}"""
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    parts = PAGE_SPLIT.split(md)
    pre, pages = parts[0], [[int(parts[i]), parts[i + 1]] for i in range(1, len(parts), 2)]
    stats = {"matched": 0, "below": 0, "promoted": 0, "repeats_dropped": 0}
    if not marks or not pages:
        return stats
    prior = [m for m in marks if m[2] < pages[0][0]]
    cur = prior[-1][0] + 1 if prior else None              # heading level of the outline entry we are under
    chain = {}                                             # outline level -> title key of the entries we are inside
    for m in prior:
        chain = {l: k for l, k in chain.items() if l < m[0]}; chain[m[0]] = m[1]
    for pg in pages:
        near = [m for m in marks if abs(m[2] - pg[0]) <= 1]
        segs = CODE_BLOCK.split(pg[1])
        for k in range(0, len(segs), 2):
            paras = segs[k].split("\n\n")
            for i, para in enumerate(paras):
                line = para.strip()
                if not line or "\n" in line:
                    continue
                h = re.match(r"(#{1,6})\s+(.*)$", line)
                title = h.group(2) if h else line
                key = _key(title.replace("*", ""))
                if len(key) < 3:
                    continue
                if key in chain.values() and (h or len(line) <= 120):   # the title of a section we're in, again: a running header
                    paras[i] = None; stats["repeats_dropped"] += 1
                    continue
                m = next((m for m in near if m[1] == key), None) or (
                    next((m for m in near if min(len(key), len(m[1])) >= 12 and (m[1].startswith(key) or key.startswith(m[1]))), None) if h else None)
                if m:
                    level = min(6, m[0] + 1); cur = level
                    chain = {l: k for l, k in chain.items() if l < m[0]}; chain[m[0]] = m[1]
                    if not h:
                        if len(line) > 120 or re.match(r"(\||!\[|>|[-*+] |\d+[.)] |\$)", line):
                            continue
                        stats["promoted"] += 1
                    else:
                        stats["matched"] += 1
                elif h and cur:
                    level = min(6, cur + 1); stats["below"] += 1
                else:
                    continue
                paras[i] = para.replace(line, "#" * level + " " + title.strip(), 1)
            segs[k] = "\n\n".join(x for x in paras if x is not None)
        pg[1] = "".join(segs)
    md_path.write_text(pre + "".join(f"\n\n{{{p}}}{'-' * 48}\n\n{t.strip(chr(10))}" for p, t in pages) + "\n", encoding="utf-8")
    return stats


SENTENCE_END = re.compile(r"[.!?:;。！？｡」』）)\]\"”’»।॥؟۔።։;]\s*$")      # Latin, CJK, Indic, Arabic/Urdu, Ethiopic, Armenian, Greek
CAPTION = re.compile(r"\s*(\*\*|_)?\S{1,14}\.?\s*\d+([-.:]\d+)*\b")          # "Figure 3-2", "图3", "Рис. 5", "Abb. 2", "চিত্র ৪", "شكل 3"


def _plain(para):
    p = para.strip()
    return p and not re.match(r"(#|\||!\[|>|\$\$|```|~~~|<(?!span)|[-*+] |\d+[.)] )", p) and not CAPTION.match(p)


def join_pages(texts):
    """joins pages, mending a paragraph cut by the page break in any script: the page ends without closing punctuation
    and the next starts with text (English, which has the lower-case clue, is also handled anywhere by join_broken)"""
    out = ""
    for t in (x.strip("\n") for x in texts if x.strip()):
        if out:
            last, first = out.rsplit("\n\n", 1)[-1], t.split("\n\n", 1)[0]
            body = ANCHOR_START.sub("", first.lstrip())
            if (_plain(last) and _plain(first) and not SENTENCE_END.search(last) and body[:1].isalpha()
                    and not body[:1].isascii()):
                out += ("" if _no_space(last.rstrip()[-1]) and _no_space(body[0]) else " ") + first.lstrip() + t[len(first):]
                continue
            out += "\n\n"
        out += t
    return out


def finish_markdown(md_path, keep_pages=False):
    """removes the page separators (unless asked to keep them) and tidies the Markdown, in place"""
    md_path = Path(md_path); md = md_path.read_text(encoding="utf-8")
    if not keep_pages:
        parts = PAGE_SPLIT.split(md)
        md = join_pages([parts[0]] + parts[2::2])
    md_path.write_text(no_src_images(tidy(md)), encoding="utf-8")      # table icons fix_tables could not crop: their alt text


# ----------------------------------------------------------------- Markdown syntax, deterministic
FENCE = re.compile(r"^\s*(```|~~~)")


PLAIN_END = re.compile(r"[^\W\d_],?$|,$")                    # a paragraph that stops mid-sentence (letter or comma)
NOT_PLAIN = re.compile(r"(#|\||!\[|>|\$\$|```|~~~|<(?!span)|\[|[-*+] |\d+[.)] )")
ANCHOR_START = re.compile(r"^(?:<span id=\"[^\"]*\">\s*</span>)*")


def join_broken(md):
    """joins a paragraph cut by a page or column break ("because I consider it" + "the most essential...") when the first
    stops without punctuation and the next starts in lower case; code blocks and display maths are left alone"""
    out = []
    for k, seg in enumerate(re.split(r"(^(?:```|~~~)[^\n]*\n.*?^(?:```|~~~)[ \t]*$)", md, flags=re.M | re.S)):
        if k % 2:
            out.append(seg); continue
        paras = seg.split("\n\n")
        res = []
        for para in paras:
            prev = res[-1].rstrip() if res else ""
            body = ANCHOR_START.sub("", para.lstrip())
            if (prev and "\n" not in prev and PLAIN_END.search(prev) and not re.match(r"\s*(#|\||!\[|>|\$\$|<(?!span))", prev) and not CAPTION.match(prev)
                    and "\n|" not in prev and body[:1].islower() and not NOT_PLAIN.match(para.lstrip())):
                res[-1] = prev + " " + para.lstrip()
            else:
                res.append(para)
        out.append("\n\n".join(res))
    return "".join(out)


CODE_BLOCK = re.compile(r"(^(?:```|~~~)[^\n]*\n.*?^(?:```|~~~)[ \t]*$)", re.M | re.S)
AMOUNT = re.compile(r"(?<![\\$\w])\$(?=\d[\d,]*(?:\.\d+)?)")


def escape_currency(text):
    """US $34.99 CAN $40.99 -> US \\$34.99 CAN \\$40.99: with math on, the two $ would make a formula of the text between.
    A $ is money when it starts an amount and no formula closes after it (next $ right after a non-space, operators between)."""
    def fix(m):
        rest = text[m.end():]
        nxt = re.search(r"(?<!\\)\$", rest)
        inner = rest[:nxt.start()] if nxt else ""
        if nxt and inner and not inner[-1].isspace() and (re.search(r"[=+*/^_\\{}<>]|\s-\s", inner) or not re.search(r"\s", inner)):
            return "$"                                     # $5 + 3$ and $5$ are maths
        return "\\$"
    return AMOUNT.sub(fix, text)


def _hard_breaks(para):
    """a block of short lines (an address, a receipt, a poem) stays line by line: Markdown would run them together"""
    lines = para.strip("\n").split("\n")
    if (len(lines) >= 2 and all(0 < len(l.strip()) <= 80 for l in lines) and not any(l.endswith("  ") for l in lines[:-1])
            and not any(re.match(r"\s*(#|\||>|[-*+] |\d+[.)] |\$\$|```|~~~|<|!\[)", l) for l in lines)):
        lead = para[:len(para) - len(para.lstrip("\n"))]
        return lead + "  \n".join(l.rstrip() for l in lines) + para[len(para.rstrip("\n")):]
    return para


EQ_NUM = r"(?:\\q?quad|\\hfill|\\[,;!]|~|\s)*(?:\\text\{\s*)?\(\s*(\d{1,3}(?:\.\d{1,3}){0,2}[a-z]?)\s*\)[ \t]*\}?"
DISPLAY_EQ = re.compile(r"(?<!\\)\$\$([\s\S]+?)(?<!\\)\$\$(\n{1,2}[ \t]*" + EQ_NUM + r"[ \t]*(?=\n|$))?")
NUM_AT_END = re.compile(EQ_NUM + r"\s*$")


NO_SRC_IMG = re.compile(r'<img(?![^>]*\bsrc=)[^>]*?\balt="([^"]*)"[^>]*>(?:\s*<br\s*/?>)?\s*')


def no_src_images(text, tables=True):
    """an <img> with no picture (an icon Marker did not save) becomes its alt text unless the same words follow it
    ("<img alt="surfel"><br/>surfel</img>" -> "surfel"). tables=False leaves table rows alone: fix_tables (after the review)
    crops those pictures from the page's grid, and finish_markdown() turns any it could not crop into text"""
    def one(line):
        def img(m):
            rest = line[m.end():m.end() + len(m.group(1)) + 2]
            return "" if m.group(1) and rest.lower().startswith(m.group(1).lower()) else (m.group(1) + " " if m.group(1) else "")
        return re.sub(r"</img>", "", NO_SRC_IMG.sub(img, line))
    return "\n".join(line if not tables and line.lstrip().startswith("|") else one(line) for line in text.split("\n"))


def _html_bits(seg):
    """Marker's HTML: pictureless <img> outside tables (see no_src_images); <b>/<i> are Markdown bold/italic"""
    seg = no_src_images(seg, tables=False)
    seg = re.sub(r"<b>([^<\n]+)</b>", r"**\1**", seg)
    return re.sub(r"<i>([^<\n]+)</i>", r"*\1*", seg)


def _tag_numbers(seg):
    r"""an equation's number goes into it as \tag{7.3} (shown at the right): "(7.3)" on the line after it, or "\quad (2.11)" /
    "\quad \text{(7.7)}" at its end. A bracket that closes the formula itself ("= (x+1)") is not a number."""
    def fix(m):
        body, num = m.group(1), m.group(3)
        if num is None:
            e = NUM_AT_END.search(body)
            if not e or not re.search(r"\\q?quad|\\text\{|\\hfill|\n", e.group(0)):
                return m.group(0)
            body, num = body[:e.start()], e.group(1)
        if "\\tag" in body:
            return m.group(0) if not re.search(r"\\tag\{" + re.escape(num) + r"\}", body) else f"$${body}$$"
        end = "\n" if "\n" in body.strip() or body.startswith("\n") else ""
        return f"$${body.rstrip()} \\tag{{{num}}}{end}$$"
    return DISPLAY_EQ.sub(fix, seg)


STALE_NUMBER = re.compile(r"(?m)^([ \t>]*)\d{1,3}[.)][ \t]+(?=\d{1,3}[.)][ \t])")    # "1. 2. Incremental": the list's own number, then the page's


def _table_pipes(seg):
    r"""a "|" inside a table row's maths ($|x|$, a norm \|w\|) or code (`a || b`) splits the row into extra cells: in maths it
    becomes \vert / \Vert, in code \| (GitHub tables read that as a "|" in the cell)"""
    def math(m):
        t = m.group(1)
        t = re.sub(r"\\(left|right|big|Big|bigg|Bigg)\|", r"\\\1\\vert ", t)
        t = t.replace("\\|", "\\Vert ")
        return "$" + re.sub(r"(?<!\\)\|", r"\\vert ", t) + "$"

    def row(line):
        if not line.lstrip().startswith("|"):
            return line
        line = re.sub(r"(?<![\\$])\$(?!\$)([^$\n]+?)(?<!\\)\$", math, line)
        return re.sub(r"`[^`\n]+`", lambda m: re.sub(r"(?<!\\)\|", r"\\|", m.group(0)), line)
    return "\n".join(row(l) for l in seg.split("\n"))


def _prose(seg):
    """the clean-ups that must never touch code"""
    seg = "\n".join(re.sub(r'<math[^>]*>(.*?)</math>', lambda m: f"${m.group(1).strip()}$", l) if l.lstrip().startswith("|") else l
                    for l in seg.split("\n"))            # in a table row maths stays inline (a newline would break the table)
    seg = re.sub(r'<math[^>]*display="block"[^>]*>(.*?)</math>', lambda m: f"\n\n$$\n{m.group(1).strip()}\n$$\n\n", seg, flags=re.S)
    seg = re.sub(r"<math[^>]*>(.*?)</math>", lambda m: f"${m.group(1).strip()}$", seg, flags=re.S)
    seg = re.sub(r"^(\s*)(?:[-*+]\s+)?[•●▪◦·]\s+", r"\1- ", seg, flags=re.M)    # "- • item" / "• item" -> "- item"
    seg = re.sub(r"(?<!!)\[([^\]\n]+)\]\(#\)", r"\1", seg)                     # links with no target: plain text
    seg = STALE_NUMBER.sub(r"\1", seg)                                          # "1. 2. item" -> "2. item" (the page's number)
    seg = _table_pipes(seg)
    seg = re.sub(r"(?m)^([ \t]*)\$\$([^$\n]+?)(?<![\\$])\$[ \t]*$", r"\1$$\2$$", seg)   # "$$ x = y $": closed as display maths
    seg = _tag_numbers(_html_bits(seg))
    seg = re.sub(r"^(\s*)(?:[-*+]\s+)?(?:\[([ xX✓✔])\]|([☐☑☒✅]))\s+",          # "[X] Step 1" / "☐ item" -> GFM task list
                 lambda m: f"{m.group(1)}- [{'x' if (m.group(2) or '').strip() or (m.group(3) or '') in '☑☒✅' and m.group(3) else ' '}] ", seg, flags=re.M)
    seg = "\n\n".join(_hard_breaks(x) for x in seg.split("\n\n"))
    out, quote, in_qcode = [], [], False
    for l in seg.split("\n") + [""]:
        if l.startswith(">"):
            quote.append(l)
            continue
        if quote:                                         # a quote block: an unmatched fence (the LLM's stray ```) goes
            fences = [k for k, q in enumerate(quote) if re.match(r">[ \t]*(```|~~~)", q)]
            if len(fences) % 2:
                del quote[fences[-1]]
            for q in quote:
                if re.match(r">[ \t]*(```|~~~)", q):
                    in_qcode = not in_qcode; out.append(q)
                else:
                    out.append(q if in_qcode else escape_currency(q))   # code quoted in a figure description stays as is
            quote, in_qcode = [], False
        out.append(escape_currency(l))
    return "\n".join(out[:-1])


INDIC_SIGNS = "".join(chr(c) for c in range(0x0900, 0x0E00)
                      if "VOWEL SIGN" in unicodedata.name(chr(c), "") or "SIGN VIRAMA" in unicodedata.name(chr(c), ""))
INDIC_SPLIT = re.compile(r"(?<=[\u0900-\u0dff]) +(?=[" + INDIC_SIGNS + "])")


def _no_loops(seg):
    """the same paragraph three or more times in a row is a model loop (OCR repeated one line 14 times): kept once"""
    paras, out = seg.split("\n\n"), []
    for para in paras:
        key = para.strip()
        if len(key) >= 20 and len(out) >= 2 and out[-1].strip() == key and out[-2].strip() == key:
            continue
        out.append(para)
    res = []
    for k, para in enumerate(out):                          # the third copy was dropped above: drop the second too
        if len(para.strip()) >= 20 and res and res[-1].strip() == para.strip() and k + 1 < len(out) and out[k + 1].strip() != para.strip() \
                and seg.count(para.strip()) >= 3:
            continue
        res.append(para)
    return "\n\n".join(res)


def tidy(md):
    """valid, readable Markdown: <math> tags as $/$$, money not read as maths, blank lines around headings, tables,
    display math and code, every code fence closed, no runs of blank lines, broken paragraphs joined. Code is left as is."""
    md = re.sub(r"<\|\d{1,6}\|>", "", md)                      # the OCR model's own tokens (<|713|>), even inside code
    md = INDIC_SPLIT.sub("", md)                                # "ঢাক ার": a vowel sign never starts a word
    md = "".join(seg if k % 2 else _prose(seg) for k, seg in enumerate(CODE_BLOCK.split(md)))
    md = join_broken(md)
    md = "".join(seg if k % 2 else _no_loops(seg) for k, seg in enumerate(CODE_BLOCK.split(md)))
    out, in_code, in_math = [], False, False
    for line in md.split("\n"):
        s = line.strip()
        prev = out[-1].strip() if out else ""
        if FENCE.match(line):
            if not in_code and prev:
                out.append("")
            out.append(line); in_code = not in_code
            if not in_code:
                out.append("")
            continue
        if in_code:
            out.append(line); continue
        if s == "$$":
            if not in_math and prev:
                out.append("")
            out.append("$$"); in_math = not in_math
            if not in_math:
                out.append("")
            continue
        if in_math:
            out.append(line); continue
        if not s and (not out or not out[-1].strip()):
            continue                                        # one blank line at most (outside code)
        if re.match(r"#{1,6}\s", s) and prev:
            out.append("")
        elif s.startswith("|") and prev and not prev.startswith("|"):
            out.append("")
        elif prev.startswith("|") and s and not s.startswith("|"):
            out.append("")
        elif re.match(r"#{1,6}\s", prev) and s:
            out.append("")
        out.append(line.rstrip() + ("  " if line.endswith("  ") and line.strip() else ""))   # a hard line break stays
    if in_code:
        out.append("```")
    if in_math:
        out.append("$$")
    return "\n".join(out).strip("\n") + "\n"


def drop_unused_anchors(md):
    """<span id="page-15-0"></span> marks that no link points to (Marker writes one per heading and page): only clutter.
    For a whole document only: a link in one chunk may point into another"""
    linked = set(re.findall(r"\]\(#([^)\s]+)\)", md))
    return re.sub(r'<span id="([^"]+)">\s*</span>', lambda m: m.group(0) if m.group(1) in linked else "", md)


def shrink_images(img_dir, max_px=IMAGE_MAX_PX):
    """scales down images whose long side is above max_px (same file, same format); returns how many"""
    n = 0
    for f in Path(img_dir).iterdir():
        if f.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        try:
            with Image.open(f) as im:
                if max(im.size) <= max_px:
                    continue
                im.load(); small = im.copy()
            small.thumbnail((max_px, max_px), Image.LANCZOS)
            if f.suffix.lower() in (".jpg", ".jpeg"):
                small.convert("RGB").save(f, "JPEG", quality=92)
            else:
                small.save(f)
            n += 1
        except Exception:
            continue
    return n
