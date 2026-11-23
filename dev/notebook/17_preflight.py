import re, json, hashlib, hmac, threading, urllib.request, urllib.error
T0 = time.time(); DEADLINE = T0 + SESSION_HOURS * 3600
def secret(name):
    v = os.environ.get(name)
    if not v:
        try:
            from kaggle_secrets import UserSecretsClient; v = UserSecretsClient().get_secret(name)
        except Exception: v = None
    return v
BOX_TOKEN = (secret("BOX_TOKEN") or "").strip()
if len(BOX_TOKEN) < 16: raise RuntimeError("Secret BOX_TOKEN is missing or shorter than 16 chars: Kaggle -> Add-ons -> Secrets, tick it for this notebook.")
for s in ("HF_TOKEN", "GOOGLE_API_KEY"):
    if secret(s): os.environ[s] = secret(s)
out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
gpus = [l for l in out.strip().splitlines() if l.strip()]
print("GPUs:", *gpus, sep="\n   ")
if USE_VLM and VLM_BIN and len(gpus) < 2: print("WARNING: fewer than 2 GPUs (Settings -> Accelerator -> 'GPU T4 x2'). It still works, just slower.")
print(f"session ends in {SESSION_HOURS} h | idle stop {IDLE_STOP_MIN} min | token length {len(BOX_TOKEN)} (never printed) | Gemini key: {'yes' if os.environ.get('GOOGLE_API_KEY') else 'no'}")
