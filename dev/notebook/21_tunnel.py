def cloudflare_up(port=API_PORT, timeout=150):
    if not shutil.which("cloudflared"):
        subprocess.run('curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb" -o /tmp/cf.deb && dpkg -i /tmp/cf.deb', shell=True, check=True, capture_output=True)
    start("cloudflared", f"cloudflared tunnel --no-autoupdate --protocol http2 --edge-ip-version 4 --url http://127.0.0.1:{port}")
    t0, url, log = time.time(), None, ""
    while time.time() - t0 < timeout:
        f = LOGS / "cloudflared.log"
        log = f.read_text(errors="replace").rsplit("==== start", 1)[-1] if f.exists() else ""
        urls = re.findall(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
        if urls:
            url = urls[-1]
            if wait_http(url + "/ping", timeout=40, every=3): return url
        if not alive("cloudflared"): break
        time.sleep(2)
    raise RuntimeError(f"Cloudflare tunnel did not become reachable (url seen: {url}):\n" + log_tail("cloudflared"))
def announce(url):
    topic = "mbox-" + hashlib.sha256(("announce:" + BOX_TOKEN).encode()).hexdigest()[:40]
    t = int(time.time())
    msg = {"box": BOX_NAME, "url": url, "t": t, "deadline": int(DEADLINE), "sig": hmac.new(BOX_TOKEN.encode(), f"{BOX_NAME}|{url}|{t}".encode(), "sha256").hexdigest()}
    try:
        urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{topic}", data=json.dumps(msg).encode()), timeout=15).close()
        print("   address announced: your PC will find it automatically")
    except Exception as e: print(f"   could not announce ({type(e).__name__}); pass the url below with --url")
def network_up():
    url = cloudflare_up()
    if ANNOUNCE: announce(url)
    return url
PUBLIC_URL = network_up()
h = api_get("/health")
print("\n================ BOX READY ================")
print("  box   :", BOX_NAME, "| default quality:", (h.get("defaults") or {}).get("QUALITY"), "| Marker reading mode:", h["default_mode"],
      "| vision LLM:", h["llm_model"] or ("Gemini" if h["llm"] else "off"))
print("  url   :", PUBLIC_URL)
print("  PC    : python md_client.py book.pdf          (add --local auto to also convert on your PC's CPUs)")
print("          (or add  --url", PUBLIC_URL, " if auto-find does not work)")
print("===========================================")
