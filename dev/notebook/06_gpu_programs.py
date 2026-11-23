import glob, json, re, threading, time
def _prefetch_llm():                 # the vision LLM downloads while LibreOffice installs and Surya starts
    try:
        from huggingface_hub import hf_hub_download
        for x in (LLM_FILES[0], LLM_MMPROJ): hf_hub_download(LLM_REPO, x)
    except Exception as e:
        print("LLM prefetch:", str(e)[:200])
if LLM_LOCAL: threading.Thread(target=_prefetch_llm, daemon=True).start()
from pathlib import Path
VLM_BIN, GGUF = None, None
def cuda_major(text, pat):
    m = re.search(pat, text or "")
    return int(m.group(1)) if m else None
DRIVER_CUDA = cuda_major(subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout, r"CUDA Version:\s*(\d+)")
def nvcc_major(nvcc):
    return cuda_major(subprocess.run([nvcc, "--version"], capture_output=True, text=True).stdout, r"release (\d+)")
def find_nvcc():
    found = [c for c in dict.fromkeys([shutil.which("nvcc")] + sorted(glob.glob("/usr/local/cuda*/bin/nvcc"), reverse=True)) if c and os.path.exists(c)]
    ok = [c for c in found if DRIVER_CUDA is None or (nvcc_major(c) or 0) <= DRIVER_CUDA]   # the driver must be able to run what we build
    print("driver CUDA:", DRIVER_CUDA, "| toolkits:", {c: nvcc_major(c) for c in found})
    return ok[0] if ok else None
def find_libcuda(nvcc):
    root = os.path.dirname(os.path.dirname(os.path.realpath(nvcc)))
    for pat in ("/usr/lib/x86_64-linux-gnu/libcuda.so*", "/usr/lib64-nvidia/libcuda.so*", "/usr/local/nvidia/lib64/libcuda.so*",
                f"{root}/lib64/stubs/libcuda.so", f"{root}/targets/x86_64-linux/lib/stubs/libcuda.so"):
        hits = sorted(glob.glob(pat))
        if hits: return hits[0]
m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", subprocess.run("nvidia-smi", shell=True, capture_output=True, text=True).stdout)
DRIVER_CUDA_V = (int(m.group(1)), int(m.group(2))) if m else None
TMPD = Path("/kaggle/tmp") if Path("/kaggle/tmp").exists() else Path("/tmp")
VLM_ENV = f"CUDA_CACHE_PATH={TMPD}/cuda-cache CUDA_CACHE_MAXSIZE=4294967296 CUDA_MODULE_LOADING=LAZY"
def prebuilt_llama():
    """llama.cpp's released ubuntu CUDA 12.8 x64 build + the CUDA 12 runtime libraries it links (from pip if present)"""
    if not DRIVER_CUDA_V or DRIVER_CUDA_V < (12, 8):
        print("driver CUDA", DRIVER_CUDA_V, "is older than 12.8: cannot run the prebuilt build"); return None
    import urllib.request
    api = "https://api.github.com/repos/ggml-org/llama.cpp/releases" + ("?per_page=15" if LLAMA_CPP_REF == "master" else f"/tags/{LLAMA_CPP_REF}")
    rels = json.load(urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "box"}), timeout=30))
    for r in (rels if isinstance(rels, list) else [rels]):
        assets = {a["name"]: a["browser_download_url"] for a in r.get("assets", [])}
        main = f"llama-{r['tag_name']}-bin-ubuntu-cuda-12.8-x64.tar.gz"
        if main in assets:
            break
    else:
        print("no ubuntu-cuda-12.8-x64 asset in the recent llama.cpp releases"); return None
    d = TMPD / "llama-bin"
    sh(f"rm -rf {d} && mkdir -p {d} && curl -fsSL {assets[main]} | tar xz -C {d} --strip-components=1")
    need = ["libcudart.so.12", "libcublas.so.12", "libcublasLt.so.12"]
    import site
    for lib in need:                                  # torch's pip CUDA 12 libraries, if Kaggle has them
        hits = [h for sp in site.getsitepackages() for h in glob.glob(f"{sp}/nvidia/*/lib/{lib}")]
        if hits and not (d / lib).exists(): os.symlink(hits[0], d / lib)
    if any(not (d / l).exists() for l in need) and "cudart-" + main in assets:
        sh(f"curl -fsSL {assets['cudart-' + main]} | tar xz -C {d} --strip-components=1")
    missing = [l for l in need if not (d / l).exists()]
    if missing or not (d / "llama-server").exists(): raise RuntimeError(f"prebuilt llama.cpp incomplete, missing {missing or ['llama-server']}")
    print("prebuilt llama.cpp", r["tag_name"])
    return str(d / "llama-server")
def build_llama():
    cached = [p for p in glob.glob("/kaggle/input/*/llama-server") + glob.glob("/kaggle/input/*/*/llama-server") + ["cache/llama-server"] if os.path.exists(p)]
    if cached:
        os.chmod(cached[0], 0o755); return cached[0]
    if LLAMA_PREBUILT:
        try:
            p = prebuilt_llama()
            if p: return p
        except Exception as e:
            print("prebuilt llama.cpp failed, compiling instead:", str(e)[:500])
    nvcc = find_nvcc()
    if not nvcc:
        v = f"{DRIVER_CUDA}-{8 if DRIVER_CUDA == 12 else 0}" if DRIVER_CUDA else "12-4"
        print(f"no CUDA compiler the driver can run; installing cuda-nvcc-{v}")
        sh(f"apt-get install -y -qq cuda-nvcc-{v} cuda-cudart-dev-{v} libcublas-dev-{v}", check=False)
        nvcc = find_nvcc()
    if not nvcc: raise RuntimeError("no CUDA compiler available")
    t0 = time.time(); print("building llama-server with", nvcc, "(about 10 minutes the first time)")
    sh("rm -rf /tmp/llama.cpp && git clone -q --depth 1 " + ("" if LLAMA_CPP_REF == "master" else f"--branch {LLAMA_CPP_REF} ") + "https://github.com/ggml-org/llama.cpp /tmp/llama.cpp")
    flags = f"-DGGML_CUDA=ON -DCMAKE_CUDA_COMPILER={nvcc} -DCMAKE_CUDA_ARCHITECTURES=75 -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=Release"
    libcuda = find_libcuda(nvcc)
    if libcuda: flags += f" -DCUDA_cuda_driver_LIBRARY={libcuda}"
    print("libcuda:", libcuda)
    r = sh(f"cd /tmp/llama.cpp && cmake -B build {flags} > /tmp/cmake.log 2>&1 && cmake --build build --config Release -j{os.cpu_count()} --target llama-server >> /tmp/cmake.log 2>&1", check=False)
    if r.returncode or not os.path.exists("/tmp/llama.cpp/build/bin/llama-server"):
        raise RuntimeError("llama.cpp build failed:\n" + Path("/tmp/cmake.log").read_text()[-2500:])
    shutil.copy("/tmp/llama.cpp/build/bin/llama-server", "cache/llama-server"); os.chmod("cache/llama-server", 0o755)
    print(f"built in {time.time()-t0:.0f}s")
    return "cache/llama-server"
if USE_VLM:
    try:
        VLM_BIN = os.path.abspath(build_llama())
        from huggingface_hub import hf_hub_download
        from surya.settings import settings as _ss
        GGUF = [hf_hub_download(_ss.SURYA_GGUF_REPO, f) for f in (_ss.SURYA_GGUF_MODEL_FILE, _ss.SURYA_GGUF_MMPROJ_FILE)]
        VLM_ALIAS = _ss.SURYA_MODEL_CHECKPOINT
        print("llama-server:", VLM_BIN, "| model files:", [os.path.basename(g) for g in GGUF])
    except Exception as e:
        VLM_BIN = GGUF = None
        print("!!! VLM unavailable, continuing CPU-only (fast mode):", str(e)[:2500])
