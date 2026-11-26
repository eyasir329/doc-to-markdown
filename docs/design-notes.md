# Design notes

Why the pipeline is built the way it is: the decisions behind it and what each one protects against. It's meant
for contributors and for anyone curious about the details; users will find what they need in the other guides.
Numbers are from the 30-page regression set on Kaggle 2×T4, release 2.0.0 ([Performance](performance.md)).

## Principles

- **Models propose, checks decide.** The vision LLM is a strong proofreader and a confident one: it can drop a
  footnote, change a number or swap a ceiling bracket for a floor. Every change it proposes passes a deterministic
  check before it's kept, and the page's own text layer outranks both the OCR and the LLM.
- **Free fixes first.** Whatever code can fix without a model (the PDF's own characters, Indic vowel order, tables
  against the image) is fixed before any model pass, so no call is spent on it and the passes can skip what it
  settles.
- **The vision LLM only where it pays.** The default quality decides per page what the vision LLM looks at.
- **No work twice.** Each step runs once per page or figure; a page chunk is claimed by exactly one worker.
- **Fail locally, recover by itself.** A bad file fails one piece, a crashed process is restarted, and the box is
  meant to run unattended for a whole Kaggle session.

## Three qualities

- **Named by the trade-off.** `express`, `standard`, `precise` say what each costs and gives. They don't reuse
  Marker's reading modes (`auto`, `fast`, `balanced`), which `--mode` keeps; the two settings answer different
  questions, and shared names made it easy to set one when meaning the other. Release-1 names are mapped by intent:
  `fast` → `express`, `balanced` → `standard`, `accurate` → `precise`.
- **`express` makes no vision-LLM call.** Marker, Surya and every fix without a model already keep all the text;
  the vision LLM catches errors and explains figures. Without it the box is limited only by Surya (466 pages an hour
  on the regression set's scans).
- **`standard` decides per page.** A page whose every word is the PDF's own text, with only text blocks and no
  picture, can't be improved: it gets no call, not even the triage. Every other page gets one short triage call and
  the passes of its level (`none`, `audit`, `inspect`, `hard`). The triage's image stays in the server's cache, so the
  page's next calls don't process it again.
- **`standard` never runs the line-by-line rewrite or the prose rewrite.** They are the most expensive passes.
  Giving them to every hard page made per-page routing cost about as much as running every pass everywhere on
  scan-heavy documents. A hard page gets the element check and the finer audit instead.
- **A text layer outranks the triage's "difficulty".** A page with its own text goes to the `hard` level only when
  it is handwritten: the regression set's digital cover page (a table, a list) was rated "difficulty 3" and would
  otherwise have had a finer audit its own text already settles.
- **`precise` gives every page every pass,** with no triage: it is for documents where one wrong character costs
  more than the time.
- **Block labels are in no mode.** Asking the vision LLM to label every layout block (handwriting, code, titles,
  charts) costs one call per page and changes only the report's trace and the `--show-layout` pictures, never the
  Markdown. It stays available with `--labels`.

## Scans first

- **Orientation.** Marker reads a page as it stands, so an upside-down or sideways photo lost lines. Indra's
  orientation model decides quarter turns; an upside-down call must also read clearly better turned (by 0.15 in
  Indra's reading confidence), because the model is unsure on dense phone photos. The page's image is kept and the
  page is marked rotated.
- **Two-page photos.** A photo of an open book, read as one page, mixed and repeated lines of both pages. It's cut
  where a fold shows: a deep, wide valley in the ink with text lines on both sides, or a weaker fold confirmed by
  printed page numbers at the top or bottom of both halves. A single page with an empty middle band, a music sheet,
  a curved page or an invoice has no such fold and stays whole (the regression set checks all four). The halves
  are put in printed order, and a page chunk never separates them.
- **Text the layout left out.** On scans Marker reads block by block, so text outside every block was never read.
  Indra's line detector finds those lines; they are grouped, read by Surya and placed in reading order. A region
  counts only if it reads clearly as text and better than mirrored: specks, page edges and the other side of the
  paper showing through (which reads better mirrored) are left out.

## The vision LLM's passes

- **Element check before audit.** A page's elements are checked for their kind and structure (maths as maths, code
  as code, a table's rows, heading levels), which an audit of words can't fix. Fixes that would drop maths, a code
  block, formatting or line breaks are refused.
- **Missing content needs evidence.** Text the model says is missing is added only if the page's text layer has
  90% of its words or, on a scan, if the model says yes to two differently put questions. A missing figure is
  cropped only where there is ink and no known figure.
- **Exact corrections, confirmed.** The audit asks for "old → new" pairs, not a rewritten page. Each fix is applied
  on its own and only when the text layer confirms it or a two-way second look picks it in both orders: one
  question can be answered by a model that leans towards the first option.
- **Small sections find more.** On hard pages and in `precise` the audit works three paragraphs at a time, and the
  rewrite in parts of about 600 tokens: the model compares every word of a short section with the image, where a
  whole page lets errors through.
- **Prose rewrites keep the original.** `precise` makes unclear paragraphs explicit, but a reader must be able to
  see the author's words: they stay in a collapsed block below each rewritten paragraph. A rewrite that adds a
  number is refused (9 on the regression set), since a number the page doesn't print is invented.
- **The loss check is the safety net.** Every rewrite must keep 90% of the draft's distinct words, every number
  exactly, every heading and checkbox state, and invent no word that neither the draft nor the page's text has. A
  failed rewrite is judged paragraph by paragraph, so one bad paragraph doesn't cost the good ones.

## Figures

- **Re-cropped at 300 DPI.** Marker's crops are often too tight and cut labels and legends off; figures are grown
  over their own ink, never into text, and split pieces are joined. Page rules and running headers at a crop's edge
  are peeled off without a model; in `standard` and `precise` the vision LLM also marks what in the crop isn't the
  figure.
- **Surya reads, the vision LLM explains.** Surya reads the text inside each picture; the vision LLM gets that
  reading as evidence and describes what the figure shows and means. A long reading (40 words or more: a newspaper
  page, a letter) is kept in full under the description, so a summary never replaces text.
- **Mermaid only for flowcharts.** A figure's graph is read only after its description calls it a flowchart, with
  several reads that must agree and a check for missing arrows. Circuits, charts and other diagrams are described,
  not drawn: reading them as graphs gave wrong structure (the regression set checks that an op-amp curve and
  handwritten filter equations get no chart).
- **Repeated images are described once,** and logos are named in their alt text.

## Throughput

- **Image affinity.** A page image is about 1,500 prompt tokens. The load balancer sends a page's calls to the
  server that already processed its image, and those calls run one after another, so the image is processed once
  per server.
- **A bounded prompt cache.** llama-server's default host cache (8 GB per server) once filled the session's RAM.
  Each vision-LLM server keeps at most `LLM_HOST_CACHE_MB` (2 GB); Surya's servers keep none (their crops never
  repeat). In `standard` and `precise` the cache is what brings free RAM down to the memory guard's pause line.
- **Whole rounds of chunks, heaviest first.** A document is cut into a multiple of the worker count, scan-heavy
  stretches get fewer pages, and the heaviest chunks start first, so no worker waits for another's second round and
  the slowest chunk doesn't end the job alone.
- **More workers than CPUs.** A worker waiting for the GPUs leaves its CPU free, so up to 1.5 workers per vCPU run,
  as many as the RAM holds.
- **A cap on requests in flight.** Each worker may have at most as many vision-LLM requests in flight as the box
  has slots: one worker alone can fill every slot, but nested pools can't queue requests that would time out.

## Robustness

- **Memory.** Kaggle restarts the whole notebook when its RAM is full, so the box guards it: a start budget, a
  pause line, and below 2 GB free whatever grew is stopped (an idle worker that kept a past piece's memory, an
  oversized GPU server, or the biggest busy worker, whose piece is converted again). Workers are the kernel's first
  choice to kill, and they give memory back or restart after a piece. While nothing runs, one piece may still
  start, so the box can't deadlock.
- **A crash budget per window, not per session.** A box that runs for hours meets damaged files, each crashing a
  worker once; a budget for the whole session ran out and failed every later job. Crashes are counted over 10
  minutes, a crash loop parks the workers until the window clears, and a worker stopped on purpose (a cancel, a
  timeout, the memory guard) isn't a crash.
- **Abandoned work.** A client that was killed left its job running or its upload half-finished. A job or upload no
  client has asked about for 15 minutes is cancelled; an unfinished upload had kept the box from ever counting as
  idle, so it never stopped and used up GPU quota. Stored files no job uses go the same way.
- **A loading GPU server is skipped.** A server the watchdog or the memory guard just restarted answers 503 while
  it loads; the load balancer sends the request to the other GPU instead of failing the page's pass.
- **A pinned llama.cpp.** llama.cpp ships daily and renames flags; a new release once broke every server's command
  line. The release is pinned (`LLAMA_CPP_REF`), and the notebook still reads `llama-server --help` and leaves out
  optional flags the build doesn't know.
- **One job prepares scans at a time.** Turning pages and finding two-page photos runs two Indra processes per job,
  outside the memory guard's view; with eight jobs starting together that was sixteen.
- **Clean queues.** A finished job's task files are removed: every idle worker scans the queue several times a
  second, all session.

## Security

- Every endpoint but `GET /ping` needs the token, compared in constant time; at start the notebook stops if the API
  answers a request without it.
- The box's address is announced on an ntfy.sh topic derived from the token and signed with it: without the token
  nobody can find the topic or forge the address.
- Uploads are capped by declared size (a part can't grow past it), job and file IDs are generated by the server,
  download paths must stay inside the job's folder, and zip archives are checked for path traversal, file count and
  unpacked size.
- Options are validated: unknown keys, wrong types, bad values and malformed numbers are refused with 400.
- Nothing leaves the Kaggle session except the results you download, unless you add a `GOOGLE_API_KEY` (a Gemini
  fallback, used only if the local vision LLM can't start).
