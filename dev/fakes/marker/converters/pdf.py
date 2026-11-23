import json, os, re, time, urllib.request


class PdfConverter:
    def __init__(self, config, artifact_dict, processor_list, renderer, llm_service):
        self.config, self.page_count = config, None

    renderer = None

    def resolve_dependencies(self, cls):
        return lambda document: document.rendered

    def build_document(self, src):
        r = self(src)
        page = lambda i: type("Page", (), {"page_id": i, "contained_blocks": lambda self, doc, types: []})()
        return type("Doc", (), {"pages": [page(i) for i in r["pages"]], "rendered": r})()

    def __call__(self, src):
        total = int(re.search(rb"pages=(\d+)", open(src, "rb").read()).group(1))
        spec = self.config.get("page_range")
        pages = list(range(total))
        if spec:
            pages = sorted({p for part in spec.split(",") for p in range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1)})
        vlm = ""
        if self.config["mode"] in ("balanced", "vllm"):          # exercises the VLM path (box lb, or PC proxy -> box /vlm)
            vlm = json.load(urllib.request.urlopen(os.environ["SURYA_INFERENCE_URL"] + "/models", timeout=30))["data"][0]["id"]
        time.sleep(float(os.environ.get("FAKE_SEC_PER_PAGE", "0")) * len(pages))
        self.page_count = len(pages)
        return {"pages": pages, "vlm": vlm, "host": os.environ.get("FAKE_HOST", "?")}
