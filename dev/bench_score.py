"""Scores benchmark runs page by page, for tuning: python dev/bench_score.py MANIFEST.json RUN_DIR [RUN_DIR ...] [--out DIR]

MANIFEST.json  the benchmark's page list (doc2md_test_manifest.json: page, title, category, challenge, target_metric, ...)
RUN_DIR        one conversion of the benchmark PDF with --paginate (md_client.py ... --paginate --out RUN_DIR); its name is
               the run's label (e.g. test_out/bench/standard). Its report.json "trace" (per page / figure / chunk) is used.
Ground truth (optional): GT_DIR/<page>.md, GT_DIR = --gt DIR or MANIFEST_DIR/ground_truth, <page> = the ORIGINAL benchmark
page number ("orig_page" in a subset's manifest, else "page"), so one set serves the full PDF and any subset. When a page has one, its character
error rate and word F1 are scored; without it, only measures that need no reference:
  words, empty page, share of the script the category expects, tables / maths / code / headings / figures, defects
  (model tokens, repeated lines), agreement with the other runs (word overlap: low = the modes disagree, a page to look at),
  and the box's trace (standard's page level, steps, seconds of vision-LLM work, fixes).
Writes OUT/pages.csv (one row per run and page), OUT/summary.md (per run and per category), and appends one line per run to
OUT/history.jsonl (code id, date, timings, box resources, scores), so later runs can be compared with this one."""
import csv
import json
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

PAGE_SPLIT = re.compile(r"\n*\{(\d+)\}-{48}\n*")
SCRIPTS = {                                   # category (part of its name) -> the script its text must be in
    "Chinese": ("Han",), "Japanese": ("Hiragana", "Katakana", "Han"), "Korean": ("Hangul",), "Thai": ("Thai",),
    "Devanagari": ("Devanagari",), "Hebrew": ("Hebrew",), "Bengali": ("Bengali",), "Arabic": ("Arabic",),
}
SCRIPT_OF = {"CJK": "Han", "HIRAGANA": "Hiragana", "KATAKANA": "Katakana", "HANGUL": "Hangul", "THAI": "Thai",
             "DEVANAGARI": "Devanagari", "HEBREW": "Hebrew", "BENGALI": "Bengali", "ARABIC": "Arabic"}


def script(ch):
    name = unicodedata.name(ch, "")
    for key, s in SCRIPT_OF.items():
        if name.startswith(key) or (key == "CJK" and "IDEOGRAPH" in name):
            return s
    return "Latin" if "LATIN" in name else "other"


def words(text):
    """words, with CJK / Thai characters one word each (no spaces between them)"""
    out = []
    for w in re.findall(r"\w+", text):
        if any(script(c) in ("Han", "Hiragana", "Katakana", "Thai") for c in w):
            out += list(w)
        else:
            out.append(w.lower())
    return out


def plain(md):
    """the Markdown's text without links, figure descriptions' markers and syntax (for word measures)"""
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", md)
    md = re.sub(r"<details>[\s\S]*?</details>", " ", md)          # accurate's original wording: not counted twice
    md = re.sub(r"```mermaid[\s\S]*?```", " ", md)                   # figure text is scored on its own (figure_words)
    md = re.sub(r"(?m)^>.*$", " ", md)
    md = re.sub(r"</?[a-zA-Z][^>]*>", " ", md)                       # HTML tags (<b>, <u>, <br>) are markup, not words
    return re.sub(r"[#>*_`|\\$]+", " ", md)


def pages_of(md_text):
    parts = PAGE_SPLIT.split("\n" + md_text)
    if len(parts) < 3:                                  # no separators (converted without --paginate): one page
        return {1: md_text}
    return {int(parts[i]) + 1: parts[i + 1] for i in range(1, len(parts), 2)}      # 1-based, as the manifest


def looped(text):
    """1 if the page shows a decoding loop: the same line 3+ times in a row, or a long line (40+ characters) 3+ times;
    answers like "Ans: C" repeating down a page, or a sentence printed in several fonts, are not loops"""
    lines = [ln.strip() for ln in text.split("\n") if ln.strip() and "**Figure content:**" not in ln   # our own labels and
             and "**Text in the image**" not in ln and not ln.strip().startswith(("<details", "</details", "<summary"))]   # markup
    run = best = 1
    for x, y in zip(lines, lines[1:]):
        run = run + 1 if x == y and len(x) >= 20 else 1         # short ones repeat for real (an answer key, diagram labels)
        best = max(best, run)
    long = Counter(ln for ln in lines if len(ln) >= 40 and not ln.startswith(("|", "```")))
    return int(best >= 3 or any(n >= 3 for n in long.values()))


def overlap(a, b):
    """word-multiset F1 of two texts (0-1)"""
    ca, cb = Counter(a), Counter(b)
    common = sum((ca & cb).values())
    return round(2 * common / (sum(ca.values()) + sum(cb.values())), 3) if ca or cb else 1.0


def cer(ref, hyp):
    ref, hyp = re.sub(r"\s+", " ", ref).strip(), re.sub(r"\s+", " ", hyp).strip()
    if not ref:
        return None
    sm = SequenceMatcher(None, ref, hyp, autojunk=False)
    edits = sum(max(i2 - i1, j2 - j1) for tag, i1, i2, j1, j2 in sm.get_opcodes() if tag != "equal")
    return round(edits / len(ref), 4)


def page_row(entry, text, trace):
    pl = plain(text)
    w = words(pl)
    letters = [c for c in pl if c.isalpha()]
    want = next((v for k, v in SCRIPTS.items() if k in entry.get("category", "") + " " + entry.get("title", "")), None)
    row = {
        "page": entry["page"], "category": entry.get("category", ""), "title": entry.get("title", ""),
        "words": len(w), "chars": len(pl.strip()), "empty": len(w) < 5,
        "script_share": round(sum(script(c) in want for c in letters) / len(letters), 3) if want and letters else "",
        "non_ascii_letters": sum(ord(c) > 127 for c in letters),
        "tables": len(re.findall(r"(?m)^\|[\s:|-]+\|\s*$", text)), "maths": len(re.findall(r"(?<!\\)\$", text)) // 2,
        "code_blocks": len(re.findall(r"(?m)^```", text)) // 2, "headings": len(re.findall(r"(?m)^#{1,6} ", text)),
        "figures": len(re.findall(r"!\[[^\]]*\]\(", text)), "descriptions": text.count("**Figure content:**"),
        "figure_words": len(words(" ".join(re.findall(r"(?m)^>(.*)$", text)))),
        "mermaid": text.count("```mermaid"), "model_tokens": len(re.findall(r"<\|\d+\|>", text)), "loop": looped(text),
    }
    t = trace.get(str(entry["page"] - 1)) or {}
    row.update(level=t.get("level", ""), llm_s=t.get("s", ""), steps="+".join(t.get("steps") or []),
               fixes=sum(v for k, v in t.items() if isinstance(v, int) and (k.startswith("confirmed") or k in (
                   "inspect fixes", "missing text added", "missing figures added", "paragraphs enhanced", "line review: rewritten"))),
               refused=sum(v for k, v in t.items() if isinstance(v, int) and ("refused" in k or k == "not found")),
               layout=" ".join(f"{k}:{v}" for k, v in sorted((t.get("layout") or {}).items()) if not k.startswith("_")),
               lines_outside=(t.get("layout") or {}).get("_lines_outside", ""),   # text lines the layout put in no block
               labels=" ".join(f"{k}:{v}" for k, v in sorted(((t.get("layout") or {}).get("_labels") or {}).items())),
               disputed=(t.get("layout") or {}).get("_disputed", ""), rescued=t.get("rescued_lines", ""))
    return row, w


def main(argv):
    out = Path(argv[argv.index("--out") + 1]) if "--out" in argv else Path("test_out/bench/score")
    gt_arg = Path(argv[argv.index("--gt") + 1]) if "--gt" in argv else None
    args = [a for i, a in enumerate(argv) if a not in ("--out", "--gt") and (i == 0 or argv[i - 1] not in ("--out", "--gt"))]
    manifest = json.loads(Path(args[0]).read_text(encoding="utf-8"))
    entries = {p["page"]: p for p in manifest["pages"]}
    gt_dir = gt_arg or Path(args[0]).parent / "ground_truth"
    runs = {}
    for d in map(Path, args[1:]):
        mds = sorted(d.glob("*/*.md")) or sorted(d.glob("*.md"))
        rep = json.loads((d / "report.json").read_text(encoding="utf-8")) if (d / "report.json").is_file() else {}
        doc = (rep.get("documents") or [{}])[0]
        runs[d.name] = {"pages": pages_of(mds[0].read_text(encoding="utf-8")) if mds else {}, "report": rep, "doc": doc,
                        "trace": (doc.get("trace") or {}).get("pages") or {}, "figtrace": (doc.get("trace") or {}).get("figures") or {}}
    rows, toks = [], defaultdict(dict)
    for label, r in runs.items():
        for n, e in sorted(entries.items()):
            row, w = page_row(e, r["pages"].get(n, ""), r["trace"])
            row = {"run": label, **row, "missing_page": n not in r["pages"]}
            gt = gt_dir / f"{e.get('orig_page', n)}.md"
            if gt.is_file():
                ref = gt.read_text(encoding="utf-8")
                row.update(cer=cer(plain(ref), plain(r["pages"].get(n, ""))), word_f1=overlap(words(plain(ref)), w))
            toks[n][label] = w
            rows.append(row)
    for row in rows:                                   # agreement: how close each run's page is to the other runs' pages
        others = [overlap(toks[row["page"]][row["run"]], v) for k, v in toks[row["page"]].items() if k != row["run"]]
        row["agreement"] = round(sum(others) / len(others), 3) if others else ""
    out.mkdir(parents=True, exist_ok=True)
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "pages.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, cols); wr.writeheader(); wr.writerows(rows)

    def mean(xs):
        xs = [x for x in xs if x not in ("", None)]
        return round(sum(xs) / len(xs), 3) if xs else ""
    lines = ["# Benchmark scores", "", "| run | pages out | wall s | pages/h | GPU busy % | RAM free min GB | empty | words/page | agreement | "
             "loops | model tokens | LLM s/page | fixes | CER (GT pages) |", "|" + "---|" * 14]
    history = []
    for label, r in runs.items():
        rr = [x for x in rows if x["run"] == label]
        rep, doc = r["report"], r["doc"]
        res = doc.get("box_resources") or {}
        wall = rep.get("wall_s") or 0
        s = {"run": label, "date": time.strftime("%Y-%m-%d %H:%M"), "code": (rep.get("box") or {}).get("code") or doc.get("code"),
             "quality": doc.get("quality"), "pages_out": sum(not x["missing_page"] for x in rr), "wall_s": wall,
             "pages_per_h": round(3600 * len(r["pages"]) / wall) if wall else None, "gpu_busy_pct": res.get("gpu_busy_pct"),
             "ram_free_min_gb": res.get("ram_free_min_gb"), "cpu_busy_pct": res.get("cpu_busy_pct"),
             "empty": sum(x["empty"] for x in rr), "words_per_page": mean([x["words"] for x in rr]),
             "agreement": mean([x["agreement"] for x in rr]), "loops": sum(x["loop"] for x in rr),
             "model_tokens": sum(x["model_tokens"] for x in rr), "llm_s_per_page": mean([x["llm_s"] for x in rr]),
             "fixes": sum(x["fixes"] or 0 for x in rr), "cer": mean([x.get("cer") for x in rr]),
             "stage_seconds": doc.get("stage_seconds"), "audit": {k: v for k, v in (doc.get("audit") or {}).items() if k != "examples"},
             "figure_kinds": dict(Counter(f.get("kind", "?") for f in r["figtrace"].values())),
             "page_levels": dict(Counter(x["level"] for x in rr if x["level"]))}   # standard: none/audit/inspect/hard
                                                                                 # (before 2026-11-26: "auto_levels", mode names)
        history.append(s)
        lines.append(f"| {label} | {s['pages_out']} | {wall} | {s['pages_per_h']} | {s['gpu_busy_pct']} | {s['ram_free_min_gb']} | {s['empty']} | "
                     f"{s['words_per_page']} | {s['agreement']} | {s['loops']} | {s['model_tokens']} | {s['llm_s_per_page']} | {s['fixes']} | {s['cer']} |")
    lines += ["", "## By category (words per page / agreement / empty pages)", "",
              "| category | pages | " + " | ".join(runs) + " |", "|---|---|" + "---|" * len(runs)]
    cats = defaultdict(list)
    for x in rows:
        cats[x["category"]].append(x)
    for cat, xs in sorted(cats.items(), key=lambda kv: -len(kv[1])):
        cells = []
        for label in runs:
            rx = [x for x in xs if x["run"] == label]
            cells.append(f"{mean([x['words'] for x in rx])} / {mean([x['agreement'] for x in rx])} / {sum(x['empty'] for x in rx)}"
                         + (f" / script {mean([x['script_share'] for x in rx])}" if any(x["script_share"] != "" for x in rx) else ""))
        lines.append(f"| {cat} | {len(xs) // max(1, len(runs))} | " + " | ".join(cells) + " |")
    worst = sorted((x for x in rows if x["agreement"] != ""), key=lambda x: x["agreement"])[:25]
    lines += ["", "## Pages the runs disagree on most (look at these first; good ground-truth candidates)", "",
              "| run | page | category | words | agreement | level | steps |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {x['run']} | {x['page']} | {x['category']} | {x['words']} | {x['agreement']} | {x['level']} | {x['steps']} |" for x in worst]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out / "history.jsonl", "a", encoding="utf-8") as f:
        for s in history:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("\n".join(lines[:4 + len(runs)]))
    print(f"\nwrote {out / 'pages.csv'}, {out / 'summary.md'}, appended {len(history)} runs to {out / 'history.jsonl'}")


if __name__ == "__main__":
    main(sys.argv[1:])
