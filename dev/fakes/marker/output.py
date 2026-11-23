import json, os


def save_output(r, out_dir, stem):
    lines = [f"page {p} by {r['host']} vlm={r['vlm']} ![](_page_{p}_Picture_0.jpeg)" for p in r["pages"]]
    # like Marker with paginate_output (the worker always asks for it): "{page}" + 48 dashes before each page
    open(os.path.join(out_dir, f"{stem}.md"), "w").write("".join(f"\n\n{{{p}}}{'-' * 48}\n\n{l}" for p, l in zip(r["pages"], lines)) + "\n")
    json.dump({"page_stats": [{"page_id": p} for p in r["pages"]], "table_of_contents": [{"title": f"p{p}", "page_id": p} for p in r["pages"][:1]]},
              open(os.path.join(out_dir, f"{stem}_meta.json"), "w"))
    from PIL import Image
    for p in r["pages"]:
        Image.new("RGB", (400, 300), (p % 256, 90, 160)).save(os.path.join(out_dir, f"_page_{p}_Picture_0.jpeg"))   # figure-sized, one per page
