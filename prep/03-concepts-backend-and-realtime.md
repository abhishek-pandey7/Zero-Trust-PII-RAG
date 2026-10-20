# 03 — Backend, APIs & Real-Time, Explained

The transport layer: HTTP, WebSockets, concurrency models, error semantics, and deployment. This is
where a backend interviewer will spend most of their time, and where this project has the most
"I know exactly what's wrong with it" material.

---

## 1. WSGI vs. ASGI, and why it decides everything else

**WSGI** (Web Server Gateway Interface) is the old Python web standard. A request comes in, your
function runs to completion, a response goes out. **One request occupies one worker for its entire
duration.** Flask is WSGI.

**ASGI** is the async successor. A handler can `await` — voluntarily yielding while waiting on I/O —
so one worker interleaves hundreds of in-flight requests. FastAPI, Starlette and Django's async views
are ASGI. WebSockets are natively supported because a long-lived connection isn't a blocked worker.

**Why this is the root decision:** it determines your concurrency model, which determines how many
users you can serve, which determines whether your storage needs to be shared. My whole scaling story
in main doc §4 traces back to this choice.

**My stack:** Flask (WSGI) + Flask-SocketIO, run under `gunicorn --worker-class eventlet -w 1`
(`render.yaml`, `Procfile`). Which brings us to the interesting part.

---

## 2. Concurrency models, and the mismatch in this project

There are four ways to serve concurrent requests, and the distinctions matter here.

### Process-based (multiple workers)

`gunicorn -w 4` forks 4 OS processes. True parallelism across CPU cores — processes bypass Python's
GIL entirely. Cost: 4× memory, and **no shared in-process state**.

### Thread-based

One process, multiple OS threads. Cheaper memory, shared state. But Python's **GIL** (Global
Interpreter Lock) means only one thread executes Python bytecode at a time. Threads help for **I/O
wait** (the GIL is released during I/O) and help almost not at all for **CPU work**.

### Green threads / cooperative multitasking (eventlet, gevent)

Thousands of lightweight "coroutines" in one thread. eventlet **monkeypatches** the standard library
so that `socket`, `time.sleep`, etc. yield control instead of blocking. Extremely cheap concurrency
for I/O-bound work — which is why it's the standard choice for WebSocket servers, where you hold many
mostly-idle connections.

**The critical caveat: cooperative means voluntary.** A green thread yields *only* when it hits
patched I/O. Pure CPU work never yields, so it blocks every other green thread in the process.

### Async/await

Same cooperative idea, but explicit in the language. You can *see* the yield points, which makes
reasoning about blocking far easier than with monkeypatching.

### The mismatch in my deployment — and how to present it

`render.yaml` runs `gunicorn --worker-class eventlet -w 1`. So: **one process, cooperative green
threads, no parallelism at all.**

My workload is the exact opposite of what that's good at:

| Operation | Where | Type | Yields under eventlet? |
|---|---|---|---|
| Presidio `analyze()` | `pii_redactor.py:124` | **CPU** (spaCy inference) | **no** |
| Mean pooling | `pipeline.py:36-40` | **CPU** (pure Python loops) | **no** |
| String replacement loop | `pii_redactor.py:173` | **CPU** | **no** |
| SQLite reads/writes | `vault.py:65` | blocking C ext | **no** (sqlite3 isn't patchable) |
| HF embedding HTTP | `pipeline.py:46` | I/O | yes ✓ |
| Groq LLM HTTP | `pipeline.py:295` | I/O | yes ✓ |

Four of six are non-yielding. **So one large ingest freezes the entire process** — every other user's
request, the WebSocket heartbeats, and Render's health check poll at `/api/health`. If the freeze
outlasts the health check timeout, Render restarts the instance mid-request.

**Why eventlet was chosen anyway:** Flask-SocketIO essentially requires eventlet or gevent for
production WebSocket support. So the concurrency model was picked for the *transport* and is wrong
for the *workload*. That's a genuinely good architectural observation to make about your own code.

**The right shape:**
1. Keep eventlet for the web/WebSocket process — that's what it's good at.
2. Move CPU work (Presidio, embedding, chunking) to a **Celery worker** consuming from Redis.
3. `POST /api/ingest` enqueues and returns `202 Accepted` with a job ID.
4. The worker publishes progress; the web process relays it over the existing Socket.IO events —
   which already model exactly this (`ingestion_start` → `pii_detected` → `ingestion_complete`).

Then the web process is purely I/O-bound, eventlet is correct, and `-w 1` stops being a ceiling.

> Note the `async_mode` switch at `app.py:25`: `'threading'` when `FLASK_ENV == 'development'`, else
> `'eventlet'`. Sensible — the Werkzeug dev server doesn't play well with eventlet, and threading
> gives usable debugger behaviour locally. It also means **dev and prod have different concurrency
> semantics**, so a race that only appears under eventlet won't reproduce locally. Worth knowing.

---

## 3. REST API design in this project

Eight endpoints (`app.py`):

| Method | Path | Purpose | Line |
|---|---|---|---|
| GET | `/` | dashboard HTML | 43 |
| POST | `/api/ingest` | ingest pasted text | 49 |
| POST | `/api/upload` | ingest a file | 143 |
| POST | `/api/ask` | RAG query | 213 |
| GET | `/api/stats` | vault counters | 276 |
| GET | `/api/vault` | **all token→PII mappings** | 294 |
| GET | `/api/audit` | recent audit events | 308 |
| GET | `/api/policies` | active config | 322 |
| GET | `/api/health` | liveness probe | 348 |

### What's right about it

- **Verb semantics.** POST for state-changing operations, GET for reads. Correct.
- **Consistent envelope.** Everything returns `{'success': bool, ...}` — a client can branch on one
  field without inspecting status codes. Slightly redundant with HTTP status, but internally
  consistent.
- **Health check separated from readiness.** `/api/health` at `app.py:348` reports
  `rag_initialized: rag is not None` *without forcing initialization*. That's a real distinction:
  **liveness** ("is the process alive?") vs. **readiness** ("can it serve traffic?"). If the health
  check called `get_rag()` it would trigger a multi-second model load on every poll. Deliberate and
  correct.
- **Errors mapped to meaningful statuses** — see §5.

### What's wrong about it

- **No authentication anywhere.** `/api/vault` returns every plaintext PII value to anyone who asks.
  Main doc §5.1.
- **Not really RESTful.** These are RPC-style action endpoints, not resources. Defensible for this
  kind of pipeline API, but say "RPC over HTTP" rather than claiming REST if pushed.
- **No versioning.** `/api/v1/...` costs nothing now and is painful to add later.
- **No pagination.** `/api/vault` returns the entire table; `/api/audit` hardcodes `limit=50`
  (`app.py:313`) with no way for the client to page.
- **No `MAX_CONTENT_LENGTH`**, so `/api/upload` buffers arbitrarily large files to disk
  (`app.py:155`) before any validation runs.

---

## 4. WebSockets: what they are and when they're right

### The mechanics

HTTP is request/response — the client must ask. A **WebSocket** starts as an HTTP request with
`Upgrade: websocket`, and after the handshake the TCP connection stays open with both sides able to
send frames at any time. Full duplex, low per-message overhead.

**Socket.IO** is a library *on top* of WebSockets adding automatic reconnection, fallback to HTTP
long-polling when WebSockets are blocked, named events, rooms/namespaces, and heartbeats. That's why
`static/js/app.js:19` uses `io()` rather than raw `new WebSocket()`.

### The alternatives, honestly compared

| Approach | Direction | Complexity | Fits my use case? |
|---|---|---|---|
| **Polling** | client pulls | lowest | works, but latency/load trade-off is bad |
| **Long-polling** | client pulls, server holds | medium | wasteful |
| **SSE** | server → client only | low | **best fit** |
| **WebSocket** | bidirectional | highest | over-provisioned |

**Server-Sent Events is the honest answer for this project.** All five of my events go
server→client:

```
ingestion_start   app.py:64     query_processing  app.py:230
pii_detected      app.py:74     query_complete    app.py:247
ingestion_complete app.py:81
```

Plus `connect`/`disconnect` handlers at `app.py:359-371`. I never send a client→server message over
the socket — every client action is a `fetch()` POST (`static/js/app.js:304`, `:341`). **The
bidirectional channel is unused.** SSE is plain HTTP, has built-in reconnect, needs no extra library,
and works through more proxies.

Volunteering "I used WebSockets and SSE would have been the better call, here's why" is a much
stronger answer than defending the choice.

### The scaling constraint WebSockets impose

A WebSocket is **stateful** — bound to one specific process. With multiple workers:

1. Client connects, load balancer routes to worker 2.
2. Client POSTs `/api/ingest`, load balancer routes to worker 5.
3. Worker 5 calls `socketio.emit(...)`.
4. Worker 5 has no connection to that client. **The event goes nowhere.**

**Fixes:** sticky sessions (route a client to the same worker — fragile, and uneven load), or a
**message queue** (`SocketIO(app, message_queue='redis://...')`) so any worker can publish and the
worker holding the connection delivers. The latter is correct, and it's another reason my scaling
path runs through Redis.

This is part of why `render.yaml` pins `-w 1`. **WebSockets are a real reason my app can't scale
horizontally without new infrastructure** — and that's the kind of causal chain interviewers are
listening for.

### The bug: unscoped broadcasts

```python
socketio.emit('pii_detected', {'entities': result['redaction_details'], ...})   # app.py:74
```

No `room=` or `to=` argument, so this **broadcasts to every connected client**. And
`redaction_details` includes `original_value` — the actual plaintext PII (built at
`pii_redactor.py:175-179`).

So: user A ingests a document, and **user B's dashboard receives A's plaintext PII over the
WebSocket.** That's not a theoretical multi-tenancy concern, it's an active leak in the current code.

**The fix:** Socket.IO **rooms** — `join_room(session_id)` on connect, then `emit(..., room=...)`.
Rooms are the standard mechanism for exactly this scoping, and needing them is the natural
consequence of adding the tenancy from main doc §5.2.

---

## 5. Error handling and HTTP status semantics

### The exception hierarchy (`exceptions.py`)

```
ZeroTrustRAGError               ← base; catch-all for "my library failed"
├── ConfigurationError
├── APIError(message, provider, status_code)
├── VaultError
├── ValidationError
└── RedactionError
```

**Why a hierarchy rather than builtins:** it lets callers choose their granularity. `app.py` catches
`ValidationError` specifically to return 400, and `ZeroTrustRAGError` broadly to return 500. With
plain `ValueError` you cannot distinguish "the user sent 2MB of text" from "Groq is down" — and those
need different status codes, different retry advice, and different alerting.

**Why `APIError` carries `provider` and `status_code`** (`exceptions.py:18-21`): with multiple
upstreams (Groq, HuggingFace), "which one failed" is the first question during an incident.

### The status mapping (`app.py:262-273`)

```python
except ValidationError as e:     return ..., 400   # client's fault
except APIError as e:            return ..., 503   # upstream unavailable
except ZeroTrustRAGError as e:   return ..., 500   # my fault
except Exception as e:           return ..., 500   # unknown
```

**The 503 is the sophisticated bit.** Status codes are a contract with the client:

- **4xx** — you made a mistake, don't retry unchanged.
- **500** — I'm broken. Retrying probably won't help.
- **503 Service Unavailable** — a dependency is temporarily down. **Retry later is appropriate**, and
  well-behaved clients and load balancers act on that.

A Groq outage isn't my bug; it's upstream unavailability. Returning 503 rather than 500 tells the
client something actionable and true. That's a deliberate, defensible decision and a good thing to
point at.

### Ordering matters

Exception handlers are checked top to bottom, so **most specific must come first**. `ValidationError`
and `APIError` are both subclasses of `ZeroTrustRAGError` — if the base class were caught first, the
specific handlers would be unreachable and everything would be a 500. The ordering in `app.py` is
correct, and `pipeline.py:313-314` reinforces it by re-raising `ZeroTrustRAGError` untouched so it
doesn't get double-wrapped on the way up.

### The wrapping bug

```python
# pipeline.py:253-255
except Exception as e:
    self.audit.log_operation("ingest", "error", {"error": str(e)})
    raise RedactionError(f"Ingestion failed: {e}") from e
```

This wraps **everything** in `RedactionError` — including a HuggingFace timeout, a Chroma write
failure, or a disk-full error. So the exception *type* actively lies about where the failure was, and
`app.py` maps it to 500 when an embedding-API outage should be a 503.

**The `from e` is correct though** — that's exception chaining, preserving `__cause__` so the original
traceback survives. Good instinct, wrong exception type. The fix is to catch narrowly: let `APIError`
propagate, and only wrap genuine redaction failures.

---

## 6. Lazy initialization and the singleton pattern

```python
# app.py:29-40
rag = None

def get_rag():
    """Get or create RAG instance (lazy init with thread safety)."""
    global rag
    if rag is None:
        # ... validate keys ...
        rag = ZeroTrustRAG(groq_api_key=groq_key, hf_api_key=hf_key)
    return rag
```

**Why lazy at all.** `ZeroTrustRAG.__init__` builds a spaCy pipeline and a Presidio `AnalyzerEngine`
(`pii_redactor.py:96-106`) — hundreds of MB and several seconds. Three options:

- **Per request:** multi-second latency on every call. Non-viable.
- **At import:** the module-load blocks, so Render's health check can time out during a cold deploy
  and the platform kills the container before it ever serves traffic.
- **Lazy on first use:** deploy succeeds fast, first user pays the cost. Chosen.

**The bug: that docstring is false.** `if rag is None` followed by `rag = ...` is a **check-then-act
race**. Two concurrent first requests can both pass the check and both construct a pipeline — two
spaCy models, two Presidio engines, on a free tier with a tight memory budget. One instance is then
orphaned, and (worse) each has its own `PIIVault` with its own in-memory `_counters`, so they can mint
divergent tokens.

**Fixes, in ascending quality:**

```python
# 1. Double-checked locking
_lock = threading.Lock()
def get_rag():
    global rag
    if rag is None:
        with _lock:
            if rag is None:          # re-check inside the lock
                rag = ZeroTrustRAG(...)
    return rag
```

```python
# 2. Better: initialize at fork time, so no user pays the cost
# gunicorn.conf.py
def post_fork(server, worker):
    from app import get_rag
    get_rag()
```

Option 2 is what I'd actually do — combined with a readiness endpoint that reports not-ready until
it completes, which `/api/health` is already 90% of the way to.

### The other singleton: `Config`

```python
# config.py:13-17
def __new__(cls):
    if cls._instance is None:
        cls._instance = super().__new__(cls)
        cls._instance._load()
    return cls._instance
```

`__new__`-based singleton: every `Config()` call returns the same object, and the YAML is parsed once.
`_data` is a **class attribute** (`config.py:11`), so it's shared across all instances by definition.

**Trade-off:** config is frozen at first import, so changes need a restart. Fine for this system, and
arguably a *feature* for security-relevant settings — you don't want the PII threshold mutating at
runtime without a deploy.

**The real problem is elsewhere:** `CONFIG_PATH = Path(__file__).parent / "config.yaml"`
(`config.py:6`) resolves relative to the *module*, so the root `config.yaml` is never read. Two files,
one live, silently drifted. Main doc §5.4.

---

## 7. Configuration and secrets management

### Three-layer separation

| Layer | Contents | Mechanism | In git? |
|---|---|---|---|
| Code | logic | `.py` files | yes |
| Config | models, thresholds, top-k | `config.yaml` | yes |
| Secrets | API keys | env vars / `.env` | **no** |

`.env` is gitignored, loaded by `load_dotenv()` at `app.py:15`, read via `os.getenv` at
`pipeline.py:159-160`. In production, `render.yaml` declares:

```yaml
- key: GROQ_API_KEY
  sync: false          # ← "prompt me in the dashboard, never store in the repo"
- key: SECRET_KEY
  generateValue: true  # ← Render generates a random value
```

That's correct and worth pointing at — `sync: false` and `generateValue: true` are exactly the right
primitives, and using them shows familiarity with the platform rather than copy-paste.

### Fail-fast validation

```python
# utils.py:47-52
def validate_api_key(api_key, provider):
    if not api_key:      raise ValidationError(f"Missing {provider} API key...")
    if len(api_key) < 10: raise ValidationError(f"Invalid {provider} API key...")
```

Called at `pipeline.py:162-163`, i.e. at construction. **Fail fast** — crash at startup with a clear
message rather than at 3am on a user's request with a confusing 401 from a vendor. The length check
is a cheap typo/empty-string guard, not real validation, and the error messages name the env var to
set, which is genuinely good developer experience.

### The weak spot

```python
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
```

`app.py:19` — a **hardcoded fallback secret**. If `SECRET_KEY` is ever unset in production, Flask
silently signs sessions with a value that's public in my GitHub repo. The name *says* "change in
production," which is a comment where it should be an exception. `render.yaml` does set it, so it's
covered *there* — but any other deployment path (a plain `python app.py`, a Docker run, a different
host) inherits the insecure default. Fail-fast would be correct: raise if it's missing and
`FLASK_ENV != 'development'`.

---

## 8. Retry logic and resilience patterns

### What's implemented (`utils.py:18-44`)

```python
for attempt in range(max_retries + 1):
    try:
        return func(*args, **kwargs)
    except APIError as e:
        last_exception = e
        if attempt < max_retries:
            logger.warning(f"API error on attempt {attempt+1}/{max_retries+1}...")
            time.sleep(retry_delay)      # fixed 1.0s
        else:
            logger.error(...)
raise last_exception
```

A decorator (`@wraps` preserves the wrapped function's name and docstring — a small correctness
detail), 3 retries, fixed 1-second delay, applied at `pipeline.py:207` and `pipeline.py:257`.

**Why retry at all.** Transient failures are normal: HF cold-starts models (the fallback even sets
`wait_for_model: True` at `pipeline.py:63`), free tiers rate-limit, networks blip. A user shouldn't
see a failure for a model warm-up.

### Three problems

**1. Fixed delay instead of exponential backoff.** All clients retry at exactly 1s, 2s, 3s. If the
upstream is struggling, synchronized retries create a **thundering herd** that keeps it down. The
standard fix is exponential backoff **with jitter**:

```
delay = min(cap, base * 2**attempt) * random.uniform(0.5, 1.5)
```

The jitter is the important half — it's what decorrelates clients.

**2. `tenacity` is already in `requirements.txt` and unused.** `@retry(wait=wait_exponential_jitter(),
stop=stop_after_attempt(4), retry=retry_if_exception_type(APIError))` is one decorator and does all of
the above, tested. I hand-rolled a worse version of an installed dependency.

**3. It catches the wrong exception.** `except APIError` — but `_embed_via_requests` raises a bare
`RuntimeError` (`pipeline.py:79`). So **embedding failures are never retried**, and the decorator on
`_add_texts_to_vector_db` is effectively dead code. That's worse than no retry, because it *looks*
like resilience that isn't there.

### What's missing: circuit breaker

Retry handles *transient* failure. A **circuit breaker** handles *sustained* failure:

- **Closed** — normal, requests flow.
- **Open** — after N consecutive failures, fail immediately without calling upstream. Stops wasting
  time and stops hammering a dead dependency.
- **Half-open** — after a cooldown, let one request through to test recovery.

Right now, if Groq is down for an hour, every request still burns 3 retries × 1s before failing. A
breaker would fail in microseconds and recover automatically. It needs shared state (Redis) to be
useful across workers — another entry on the "everything routes through Redis" list.

### Also missing: timeouts

`pipeline.py:64` sets `timeout=90` on the raw-requests fallback. Good. But the `InferenceClient` path
(`pipeline.py:46`) and the `ChatGroq` call (`pipeline.py:295`) have **no explicit timeout**, so
they inherit library defaults — which may be none. A hung upstream connection with no timeout holds a
worker forever, and with `-w 1` that's the whole service. **Timeouts are more important than
retries** and are the cheaper fix.

### The error-swallowing anti-pattern

```python
# pipeline.py:85-90
if self._client is not None:
    try:
        return self._embed_via_client(texts)
    except Exception:
        pass                       # ← swallows everything
return self._embed_via_requests(texts)
```

Graceful degradation is the right *intent*. But `except Exception: pass` means an invalid API token,
a network partition, a model outage and a bug in my own pooling code are **indistinguishable** — all
you see is the fallback path running, or a generic `RuntimeError` at `pipeline.py:79` if both fail.
Debugging this in production would be miserable. Minimum fix: `logger.warning("client path failed:
%s", e, exc_info=True)` before falling through. Never silently discard an exception you didn't
specifically anticipate.

---

## 9. The frontend, briefly

`static/js/app.js` — 654 lines, one IIFE (`static/js/app.js:15`), no build step, no framework.

**Patterns actually used:**
- `const $ = (id) => document.getElementById(id)` (line 32) — a minimal DOM helper.
- Event delegation via explicit `addEventListener` per control (lines 136-140).
- **Lazy data loading on view switch** (lines 170-172): `fetchVault()`, `fetchAudit()`,
  `fetchPolicies()` fire only when their tab opens. Correct instinct — don't load four views' data on
  page load.
- Socket handlers append to a scrolling terminal feed (lines 50-130).
- `async/await` over `fetch` for all mutations (lines 304, 341).

**Problems worth naming before they're found:**
- **XSS via string-built HTML.** `fetchVault` (line 185) renders PII values into table rows by string
  concatenation with no escaping. If an ingested document contains a name like
  `<img src=x onerror=...>`, Presidio tokenizes it, the real value goes to the vault, and the vault
  view injects it into the DOM as markup. The input is attacker-controlled (uploaded documents) and it
  renders in an admin view. That's a real stored-XSS path, not a hypothetical. Fix: `textContent`, or
  escape on insert.
- **No error boundaries.** A single malformed response can throw mid-render and leave a view
  half-populated.
- **CDN dependencies** (`templates/index.html:13-15`): Phosphor Icons, motion.js and the Socket.IO
  client all load from third-party CDNs with no SRI hashes. That's both an availability dependency and
  a supply-chain risk — a compromised CDN executes arbitrary JS in a page that displays plaintext PII.
  Subresource Integrity attributes are the cheap mitigation.
- **No tests.**

**Why vanilla was still reasonable:** no build pipeline, no bundle, instant load, and the app is four
views driven by server events. React would add a toolchain for genuinely marginal benefit at this
size — though it *would* have given escaping-by-default, which would have prevented the XSS. That's
the honest framing: the choice was defensible, and one of its costs actually bit.

---

## 10. Deployment: `render.yaml` line by line

```yaml
services:
  - type: web
    runtime: python
    plan: free
    buildCommand: |
      pip install -r requirements.txt
      python -m spacy download en_core_web_sm     # model isn't a pip dep at runtime
    startCommand: gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT app:app
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0                              # pinned — avoids surprise minor upgrades
      - key: GROQ_API_KEY
        sync: false                                # never in the repo
      - key: SECRET_KEY
        generateValue: true                        # platform-generated random
    healthCheckPath: /api/health
    autoDeploy: true
```

**Good decisions here:** infrastructure as code, so the deployment is reviewable and reproducible;
Python version pinned; secrets handled with the right primitives; health check wired to a real
endpoint; `0.0.0.0` bind (not `localhost`) so the container is reachable.

**The `-w 1` chain of consequences** — this is the thing to be able to explain:

```
Flask-SocketIO needs eventlet for prod WebSockets
   → eventlet worker class
      → WebSocket connections are bound to one process
         → multiple workers would break event delivery
            → -w 1
               → no parallelism, and CPU work blocks everything
                  → one big ingest freezes the whole service
```

Every step follows from the one before. Being able to walk that chain — and then say "the way out is
to move CPU work into a Celery worker and add a Redis message queue for Socket.IO, at which point
`-w N` becomes safe" — is a complete architectural answer.

**Free-tier realities:**
- **Ephemeral filesystem.** `chroma_store/`, `vault.db` and `audit.log` are wiped on every deploy.
  The vector index, all token mappings, and the entire compliance trail. Main doc §5.13.
- **Cold starts.** Free instances spin down when idle, and my cold start includes building the
  Presidio engine, so the first request after a sleep is very slow.
- **No persistent disk**, so the fix genuinely requires managed Postgres plus a hosted or
  volume-backed vector store.

**The `Procfile`** duplicates `startCommand` — harmless redundancy that makes the app portable to
Heroku-style platforms that read `Procfile` instead.

---

## Concept checklist for this file

- [ ] I can explain WSGI vs. ASGI and why it determines my concurrency ceiling
- [ ] I can explain why eventlet is wrong for CPU work and right for WebSockets
- [ ] I can walk the full `-w 1` causal chain and give the way out
- [ ] I can explain liveness vs. readiness and point at `/api/health`
- [ ] I can argue SSE over WebSockets for this app and explain why I'd still keep Socket.IO's rooms
- [ ] I can explain the unscoped-broadcast leak and name rooms as the fix
- [ ] I can justify 400 vs. 503 vs. 500 and explain why handler ordering matters
- [ ] I can explain the `get_rag()` race and write double-checked locking on a whiteboard
- [ ] I can explain why the retry decorator is effectively dead code
- [ ] I can explain why timeouts matter more than retries
- [ ] I can name the XSS path in the vault view and the fix
