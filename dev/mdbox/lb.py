"""Tiny least-busy load balancer in front of the llama-servers (Surya VLM or the vision LLM), one per GPU.
Marker and the pipeline talk to ONE OpenAI-compatible URL; this fans their concurrent requests out over both GPUs.

Image affinity: a request carrying an image the same server saw before goes back to that server (unless it is clearly
busier), because llama-server reuses an image it already processed: measured on a T4, a page image is 1484 tokens and
2.8 s of prompt processing the first time, 0.06 s when reused. With plain least-busy routing, the several calls a page
gets (review, audit sections, second looks, a diagram's graph answers) were spread over both GPUs: 2 of 48 reused.
Usage: python lb.py PORT BACKEND_PORT [BACKEND_PORT ...]"""
import asyncio, hashlib, os, sys, time
from collections import OrderedDict

import httpx
import uvicorn
from fastapi import FastAPI, Request, Response

PORT, BACKENDS = int(sys.argv[1]), [f"http://127.0.0.1:{p}" for p in sys.argv[2:]]
HOP = {"host", "content-length", "connection", "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization", "proxy-authenticate"}
SPILL = int(os.environ.get("LB_SPILL", "3"))          # a preferred backend this many requests busier than the least busy is passed over (6 left one GPU 15-20 points idler)
AFFINITY_MAX = 4096                                    # images remembered (most recent)
inflight = {b: 0 for b in BACKENDS}
down_until = {b: 0.0 for b in BACKENDS}
affinity = OrderedDict()                               # image digest -> backend that has it
stats = {"with_image": 0, "affinity_kept": 0, "spilled": 0, "waited": 0, "wait_s": 0.0}
CAP = {"max_inflight": int(os.environ.get("LB_MAX_INFLIGHT") or 0)}   # requests in flight per backend at most (0 = no cap):
# llama-server gets slower with too many at once (per-request overhead, KV-cache contention; measured: 12 workers and 24
# slots per GPU gave 334 pages/h, 8 and 16 gave 387). Settable while running: POST /lb-config {"max_inflight": n}
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0), limits=httpx.Limits(max_connections=256, max_keepalive_connections=64))


def image_key(body):
    """digest of the first base64 image in a request body, or None"""
    i = body.find(b";base64,")
    if i < 0:
        return None
    j = body.find(b'"', i)
    return hashlib.blake2b(body[i + 8:j if j > 0 else len(body)], digest_size=12).digest()


def order(key=None):
    now = time.time()
    up = [b for b in BACKENDS if down_until[b] <= now] or list(BACKENDS)       # if everything looks down, still try
    spread = key[0] if key is not None else int(time.time() * 1000)     # equal load: a new image's digest picks, not list order
    ranked = sorted(up, key=lambda b: (inflight[b], (BACKENDS.index(b) + spread) % len(BACKENDS)))
    pref = affinity.get(key) if key is not None else None
    if pref in up:
        if inflight[pref] - inflight[ranked[0]] < SPILL:
            ranked.remove(pref); ranked.insert(0, pref)
            stats["affinity_kept"] += 1
        else:
            stats["spilled"] += 1
    return ranked


def remember(key, backend):
    if key is None:
        return
    affinity[key] = backend
    affinity.move_to_end(key)
    while len(affinity) > AFFINITY_MAX:
        affinity.popitem(last=False)


@app.get("/lb-status")
async def status():
    return {"inflight": inflight, "down": {b: max(0, round(t - time.time())) for b, t in down_until.items()}, "affinity": stats,
            "max_inflight": CAP["max_inflight"]}


@app.post("/lb-config")
async def config(request: Request):
    """{"max_inflight": n}: the cap per backend from now on (0 = none); for tuning a running box (behind the API's token)"""
    body = await request.json()
    n = int(body.get("max_inflight", CAP["max_inflight"]))
    if not 0 <= n <= 1024:
        return Response(content=b"max_inflight must be 0..1024", status_code=400)
    CAP["max_inflight"] = n
    return {"max_inflight": n}


async def pick(key):
    """the backends to try, best first; while every backend is at the cap, wait (checked and taken without an await between)"""
    t0, waited = time.time(), False
    while True:
        cap = CAP["max_inflight"]
        ranked = [b for b in order(key) if not cap or inflight[b] < cap]
        if ranked:
            if waited:
                stats["waited"] += 1; stats["wait_s"] = round(stats["wait_s"] + time.time() - t0, 1)
            return ranked
        waited = True
        await asyncio.sleep(0.01)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD"])
async def proxy(path: str, request: Request):
    body = await request.body()
    headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP}
    key = image_key(body) if request.method == "POST" and body else None
    if key is not None:
        stats["with_image"] += 1
    last = None
    for b in (await pick(key) if request.method == "POST" else order(key)):
        remember(key, b)                               # at once: the page's next calls follow, even while this one runs
        inflight[b] += 1
        try:
            r = await client.request(request.method, f"{b}/{path}", params=request.query_params, content=body, headers=headers)
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError, httpx.ConnectTimeout) as e:
            down_until[b] = time.time() + 10; last = e
            continue
        finally:
            inflight[b] -= 1
        if r.status_code == 503:                       # still loading (the watchdog or the memory guard just restarted it):
            down_until[b] = time.time() + 5            # the other backend takes this request and the next few; before, the
            last = r; continue                         # 503 went back to the worker and the page lost its pass
        out = {k: v for k, v in r.headers.items() if k.lower() not in HOP and k.lower() != "content-encoding"}
        return Response(content=r.content, status_code=r.status_code, headers=out)
    if isinstance(last, httpx.Response):
        return Response(content=last.content, status_code=last.status_code)
    return Response(content=f"no backend reachable: {last!r}".encode(), status_code=502)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
