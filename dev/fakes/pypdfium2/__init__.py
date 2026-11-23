import re


class PdfDocument:
    def __init__(self, path):
        self.n = int(re.search(rb"pages=(\d+)", open(path, "rb").read()).group(1))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return _Page()

    def close(self):
        pass


class _Page:
    def get_size(self):
        return (612.0, 792.0)

    def get_cropbox(self):
        return (0.0, 0.0, 612.0, 792.0)

    def get_objects(self, filter=None, max_depth=1):
        return iter(())


class _Bitmap:
    def __init__(self, w, h):
        self.w, self.h = w, h

    def to_pil(self):
        from PIL import Image
        return Image.new("RGB", (self.w, self.h), "white")


class _TextPage:
    def get_text_range(self):
        return ""

    def close(self):
        pass


_Page.render = lambda self, scale=1.0, **k: _Bitmap(max(1, int(612 * scale)), max(1, int(792 * scale)))
_Page.get_textpage = lambda self: _TextPage()
