# HTTP API

`md_client.py` is the reference client, but any program can drive the box. The API is served by `dev/mdbox/server.py`
(FastAPI) at the tunnel address from the BOX READY banner.

## Authentication

Every endpoint except `GET /ping` needs the header

```
Authorization: Bearer <BOX_TOKEN>
```

The token is compared in constant time. Requests without a valid token get `401`.

## Why uploads come in parts

Cloudflare's free quick tunnel rejects request bodies over about 100 MB and cuts requests after about 100 s. So:

- documents are uploaded in parts of `part_size` bytes (16 MB);
- results are polled (`GET /jobs/{id}`) rather than streamed;
- results are downloaded with plain `GET` requests.

## Endpoints

| Method and path | Body | Returns |
|---|---|---|
| `GET /ping` | – | `ok` (no auth: for health checks) |
| `GET /health` | – | box status: `workers`, `gpus`, `vlm`, `vlm_slots`, `llm`, `llm_model`, `llm_slots`, `defaults`, `default_mode`, `memory`, `waiting_tasks`, `running`, `queued`, `idle_s`, `remaining_s`, `extensions`, `max_upload_gb`, `libreoffice`, `ddjvu`, … |
| `POST /jobs` | `{"name", "size", "options"?}` | `{"job_id", "part_size"}`: a new job waiting for its upload |
| `PUT /jobs/{id}/parts/{n}` | raw bytes; optional header `X-Sha256` | `{"part", "bytes"}`. Parts are numbered from 0. A part larger than the declared file size is refused (`413`) |
| `POST /jobs/{id}/start` | `{"parts": N}` | job status. The parts are joined and checked against the declared size, and the job is queued |
| `GET /jobs/{id}` | – | job status (see below). Poll every 1–2 s |
| `GET /jobs/{id}/bundle.zip` | – | everything: `<name>/<name>.md`, images, `<name>_meta.json`, `report.json` |
| `GET /jobs/{id}/files/{path}` | – | one output file |
| `DELETE /jobs/{id}` | – | cancels a queued or running job, or deletes a finished one |
| `GET /jobs` | – | all jobs |

### Stored files and batches

The client uses these for long documents. The document is uploaded once and converted in batches of pages:

| Method and path | Body | Returns |
|---|---|---|
| `POST /files` | `{"name", "size"}` | `{"file_id", "part_size"}` |
| `PUT /files/{fid}/parts/{n}` | raw bytes | `{"part", "bytes"}` |
| `POST /files/{fid}/finish` | `{"parts": N}` | `{"file_id", "pages", "pdf", "warnings"}`. Office and DjVu files are rendered to PDF here |
| `POST /jobs` | `{"file_id", "name", "options": {"PAGE_RANGE": "0-95", …}}` | a job on the stored file, queued at once |
| `DELETE /files/{fid}` | – | removes the stored file |

### Helpers for PC workers

| Method and path | Purpose |
|---|---|
| `POST /jobs/{id}/steal` | `{"n"}` → `{"chunks": [{"file", "pages", "folder", "stem"}]}`: not-yet-started page chunks the caller converts itself. The box leaves a placeholder for them |
| `GET\|POST /vlm/...` | the box's Surya servers (OpenAI-compatible API, e.g. `/vlm/v1/chat/completions`) |
| `GET\|POST /llm/...` | the box's vision-LLM servers (OpenAI-compatible API) |

## Job status

`GET /jobs/{id}` returns:

| Field | Meaning |
|---|---|
| `status` | `uploading` → `queued` → `running` → `done`, or `cancelled` / `error` |
| `stage`, `pct`, `msg` | the current stage, progress in percent and a readable message |
| `docs`, `n_ok`, `n_failed`, `pages`, `partial` | per-document results once done |
| `warnings`, `error` | anything worth knowing (memory pauses, a refused option), or the job's error |
| `created`, `started`, `finished`, `wall_s` | timing |
| `options`, `mode` | what the job ran with |

A job that no client has polled for `MD_ABANDON_MIN` minutes (default 15) is cancelled with the message
"cancelled: no client asked about it". Keep polling while you wait.

## Options

`options` takes the keys listed in [Configuration → Per-job options](configuration.md#per-job-options), for
example:

```json
{"QUALITY": "accurate", "PAGE_RANGE": "0-9", "OUTPUT_FORMAT": "markdown", "PAGE_REVIEW": true}
```

Unknown keys, wrong types and invalid values are refused with `400` and a message listing what's allowed.

## Minimal example (Python, standard library)

```python
import json, os, time, urllib.request

BOX = os.environ["BOX_URL"]
TOKEN = os.environ["BOX_TOKEN"]

def call(method, path, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(BOX + path, data=data, method=method,
                                 headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90) as r:
        out = r.read()
        return json.loads(out) if r.headers.get_content_type() == "application/json" else out

doc = open("paper.pdf", "rb").read()
job = call("POST", "/jobs", {"name": "paper.pdf", "size": len(doc), "options": {"QUALITY": "balanced"}})
step = job["part_size"]
parts = [doc[i:i + step] for i in range(0, len(doc), step)]
for n, part in enumerate(parts):
    call("PUT", f"/jobs/{job['job_id']}/parts/{n}", raw=part)
call("POST", f"/jobs/{job['job_id']}/start", {"parts": len(parts)})

while (st := call("GET", f"/jobs/{job['job_id']}"))["status"] not in ("done", "cancelled", "error"):
    print(st["stage"], st["pct"], st["msg"]); time.sleep(2)

open("paper.zip", "wb").write(call("GET", f"/jobs/{job['job_id']}/bundle.zip"))
call("DELETE", f"/jobs/{job['job_id']}")
```

`md_client.py` adds parallel part uploads, checksums, retries on tunnel loss, batches with resume, and local
workers.
