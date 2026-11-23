import subprocess, sys, os, shutil
def sh(cmd, check=True):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(f"{cmd}\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return r
# weasyprint (HTML / EPUB / DOCX rendering) needs pango
sh("apt-get update -qq", check=False)
sh("apt-get install -y -qq libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz-subset0 fonts-dejavu-core", check=False)
# fonts for every script (Bangla, Devanagari, Arabic, Thai, CJK, Ethiopic, emoji...): rendering HTML/EPUB/DOCX or office
# files to PDF without them turns that text into empty boxes
sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends fonts-noto-core fonts-noto-extra fonts-noto-cjk "
   "fonts-noto-color-emoji", check=False)
if OFFICE_VIA_LIBREOFFICE:
    # Carlito/Caladea are metric clones of Calibri/Cambria (same line breaks as Word)
    r = sh("DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends libreoffice-writer libreoffice-impress "
           "libreoffice-calc fonts-crosextra-carlito fonts-crosextra-caladea", check=False)
sh("apt-get install -y -qq --no-install-recommends djvulibre-bin", check=False)      # ddjvu: DjVu -> PDF
print("ddjvu (DjVu):", shutil.which("ddjvu") or "NOT installed; .djvu files will fail")
if OFFICE_VIA_LIBREOFFICE:
    print("LibreOffice:", shutil.which("soffice") or "NOT installed; only pdf/images/docx/pptx/xlsx/html/epub/txt will work\n" + r.stderr[-800:])
sh(f'{sys.executable} -m pip install -q --no-warn-conflicts "marker-pdf[full] @ {MARKER_SOURCE}" fastapi uvicorn python-multipart httpx' if MARKER_SOURCE.startswith("git+")
   else f'{sys.executable} -m pip install -q --no-warn-conflicts "{MARKER_SOURCE}" fastapi uvicorn python-multipart httpx')
sh("mkdir -p mdbox logs cache")
print(sh(f"{sys.executable} -m pip show marker-pdf | head -2").stdout)
