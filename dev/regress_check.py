"""Pass/fail checks for the 30-page regression set, per quality mode.

The set (Regression_30.pdf, not distributed: third-party and personal scans) has one page for each problem the pipeline
had to solve: two-page book photos to cut (pages 23-29) and pages that must stay whole (2, 3, 10, 16), upside-down and
sideways photos, handwriting (12, 13, 20-22), a program listing (25), charts that are not flowcharts (19, 22), a digital
page with a text layer (1), degraded scans, CJK, Devanagari and Hebrew text. See docs/performance.md.

usage: python dev/bench_run.py Regression_30.pdf manifest.json RUN_DIR --modes express,standard,precise
       python dev/regress_check.py RUN_DIR [modes...]
RUN_DIR/<mode>/ is a md_client.py --paginate output of Regression_30.pdf. Page numbers below are the set's (1-30)."""
import json, re, sys
from pathlib import Path

SEP = re.compile(r"\n*\{(\d+)\}-{48}\n*")
SPREADS = {23, 24, 25, 26, 27, 28, 29}          # two-page book photos that must be cut (and no other page)
HANDWRITTEN = {12, 13, 20, 21, 22}
FAKE_MERMAID = {19, 22}                          # op-amp curve, filter equations: no flowchart
CODE = {25}                                      # the book spread with a program listing
LEGIT_REPEATS = {8}                              # Devanagari font samples repeat on purpose


def pages_of(md):
    parts = SEP.split(md)
    return {int(parts[i]) + 1: parts[i + 1] for i in range(1, len(parts), 2)}


def looped(text):
    """a non-trivial line or paragraph repeated 3+ times (OCR loop); precise's "Original wording" blocks left out"""
    text = re.sub(r"<details>[\s\S]*?</details>", " ", text)
    paras = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if len(p.strip()) > 25
             and not p.lstrip("> ").startswith("**")]     # a figure's quote header ("**Figure content:**", "**Text in
                                                                                # the image**"): one per figure, not a loop
    return any(paras.count(p) >= 3 for p in set(paras))


def check(run, mode):
    out = Path(run) / mode
    md_file = next(out.rglob("Regression_30.md"), None)
    rep = json.loads((out / "report.json").read_text())["documents"][0]
    res = []
    ok = lambda name, cond, detail="": res.append((name, bool(cond), detail))
    if md_file is None:
        ok("output exists", False); return res, rep
    pages = pages_of(md_file.read_text(encoding="utf-8"))
    empty = [p for p in range(1, 31) if len(re.sub(r"\W", "", re.sub(r"!\[[^\]]*\]\([^)]*\)", "", pages.get(p, "")))) < 20]
    ok("all 30 pages present, none empty", not empty and len(pages) == 30, f"empty/missing {empty}" if empty else "")
    loops = [p for p, t in pages.items() if p not in LEGIT_REPEATS and looped(t)]
    ok("no repetition loops", not loops, f"pages {loops}" if loops else "")
    ok("7 spreads cut (no false cuts)", rep.get("spreads_split") == len(SPREADS), f"spreads_split={rep.get('spreads_split')}")
    ok("turned pages set upright", (rep.get("pages_turned") or 0) >= 2, f"pages_turned={rep.get('pages_turned')}")
    fake = [p for p in FAKE_MERMAID if "```mermaid" in pages.get(p, "")]
    ok("no fake Mermaid (p19 op-amp, p22 filter)", not fake, f"pages {fake}" if fake else "")
    nocode = [p for p in CODE if "```" not in pages.get(p, "").replace("```mermaid", "")]
    ok("code pages have code fences", not nocode, f"missing on {nocode}" if nocode else "")
    ok("text coverage of text-layer pages >= 90%", (rep.get("text_coverage") or 0) >= 0.9 or not rep.get("text_coverage"),
       f"{rep.get('text_coverage')} low {rep.get('low_coverage_pages')}")
    ok("quality reported", rep.get("quality") == {"express": "express", "standard": "standard", "precise": "precise"}[mode],
       str(rep.get("quality")))
    lv = rep.get("page_levels") or {}
    if mode == "standard":
        ok("standard: page levels reported", len(lv) >= 28, f"{len(lv)} pages")
        ok("standard: digital cover p1 (has a table) not sent to the finer audit", lv.get("0") in ("inspect", "audit", "none"),
           f"p1={lv.get('0')}")
        hw = {p: lv.get(str(p - 1)) for p in HANDWRITTEN}
        ok("standard: handwriting pages at least inspect", all(v in ("inspect", "hard") for v in hw.values()), str(hw))
    else:
        ok(f"{mode}: no page levels (no triage)", not lv, str(lv)[:80])
    au = rep.get("audit") or {}
    if mode == "express":
        ok("express: no vision-LLM work", not au and not rep.get("figures_described"), f"audit {bool(au)}, described {rep.get('figures_described')}")
    else:
        ok(f"{mode}: page passes ran", au.get("sections", 0) + au.get("inspected elements", 0) > 0,
           f"sections {au.get('sections')}, inspected {au.get('inspected elements')}, refused {au.get('refused')}, errors {au.get('errors')}")
        ok(f"{mode}: no failed LLM calls", not any(v for k, v in au.items() if "failed" in k or k == "errors"),
           str({k: v for k, v in au.items() if "failed" in k or k == "errors"}))
    return res, rep


def main(argv):
    run, modes = argv[0], argv[1:] or ["express", "standard", "precise"]
    bad = 0
    for m in modes:
        if not (Path(run) / m / "report.json").is_file():
            print(f"== {m}: no report.json"); bad += 1; continue
        res, rep = check(run, m)
        print(f"== {m}: {sum(r[1] for r in res)}/{len(res)} ok | {rep.get('seconds')} s box time, pages {rep.get('pages')}")
        for name, good, detail in res:
            print(f"   {'ok  ' if good else 'FAIL'} {name}" + (f"  ({detail})" if detail else ""))
            bad += not good
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main(sys.argv[1:])
