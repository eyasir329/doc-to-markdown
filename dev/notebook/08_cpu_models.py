r = subprocess.run([sys.executable, "-c", """
from marker.util import download_font; download_font()
from marker.models import create_model_dict
m = create_model_dict(); m['indra_model'].engine; print('Indra ONNX models + font ready')
"""], capture_output=True, text=True)
print(r.stdout[-300:] or r.stderr[-1500:])
if r.returncode: raise RuntimeError("could not prepare Marker models (internet/HF_TOKEN?)")
