def api_ok(): return alive("api") and wait_http(f"http://127.0.0.1:{API_PORT}/ping", timeout=15)
def net_ok(): return alive("cloudflared") and wait_http(PUBLIC_URL + "/ping", timeout=20)
fails = {"api": 0, "net": 0}; last_beat = 0
try:
    while time.time() < DEADLINE:
        for name, ok in (("api", api_ok), ("net", net_ok)): fails[name] = 0 if ok() else fails[name] + 1
        for g in range(len(VLM_PORTS)):
            if not alive(f"vlm{g}"):
                print(time.strftime("%H:%M:%S"), f"GPU server {g} died; restarting\n" + log_tail(f"vlm{g}", 6)); start_vlm(g, SLOTS.get(g, 2))
        for g in LLM_SERVERS:
            if not alive(f"llm{g}"):
                print(time.strftime("%H:%M:%S"), f"vision LLM on GPU {g} died; restarting\n" + log_tail(f"llm{g}", 6)); start_llm(g, *LLM_RUN[g])
        if LLM_URL and not alive("lb_llm"): start("lb_llm", f"{sys.executable} mdbox/lb.py {LLM_LB_PORT} {' '.join(map(str, LLM_SERVERS.values()))}")
        if VLM_URL and not alive("lb"): start("lb", f"LB_MAX_INFLIGHT={VLM_LB_MAX_INFLIGHT} {sys.executable} mdbox/lb.py {VLM_LB_PORT} {' '.join(map(str, VLM_PORTS))}")
        if fails["api"] >= 2: print(time.strftime("%H:%M:%S"), "API down; restarting (job history in memory is lost)\n" + log_tail("api", 8)); start_api(); fails["api"] = 0
        if fails["net"] >= 2: print(time.strftime("%H:%M:%S"), "tunnel down; restarting"); PUBLIC_URL = network_up(); print("   new url:", PUBLIC_URL); fails["net"] = 0
        try: h = api_get("/health")
        except Exception: h = {}
        if IDLE_STOP_MIN and h.get("idle_s", 0) > IDLE_STOP_MIN * 60:
            print(f"{time.strftime('%H:%M:%S')} idle for {IDLE_STOP_MIN} min: stopping to save GPU quota (run the notebook again when needed)"); break
        if time.time() - last_beat > 300:
            last_beat = time.time()
            try: print(f"{time.strftime('%H:%M:%S')} alive | {(DEADLINE-time.time())/3600:.1f} h left | idle {h.get('idle_s')} s | RAM {h.get('memory', {}).get('used_gb')}/{h.get('memory', {}).get('limit_gb')} GB | jobs:", [(j['id'], j['status'], f"{j['pct']:.0f}%") for j in api_get('/jobs')[-5:]] or "none")
            except Exception as e: print("heartbeat:", repr(e)[:100])
        time.sleep(30)
finally:
    for n in ["cloudflared", "api", "lb", "lb_llm"] + [f"vlm{g}" for g in range(len(gpus))] + [f"llm{g}" for g in range(len(gpus))]: stop(n)
    print("box stopped after %.1f h" % ((time.time() - T0) / 3600))
