# 05 — Scaling Concepts, Explained

The generic system-design vocabulary an interviewer expects — caching, sharding, load balancing,
queues, replication, rate limiting, CDN — explained properly, and then applied to *this* codebase
rather than to a hypothetical Twitter clone.

The framing to use throughout: **scaling is never "make it faster." It's "find the bottleneck, name
what breaks, add a mechanism, accept its cost."** The cost is the part interviewers listen for.

---

## 1. Vertical vs. horizontal scaling

- **Vertical (scale up):** a bigger machine. More CPU, more RAM. Trivial — no code changes. Hits a
  hard ceiling, costs superlinearly, and doesn't improve availability (still one box, still one
  failure domain).
- **Horizontal (scale out):** more machines. Effectively unbounded, and it improves availability
  because you can lose a node. Requires your application to be **stateless**, or to externalize its
  state.

### Where my app sits

`gunicorn --worker-class eventlet -w 1` — one worker, one process. Vertical only.

**And I can't scale horizontally without changes**, because three pieces of state are
process-local:

| State | Where | Breaks how |
|---|---|---|
| ChromaDB index | `pipeline.py:195`, local dir | worker 2 can't see worker 1's ingests |
| SQLite vault | `vault.py:65`, local file | separate files → divergent tokens |
| WebSocket connections | `app.py:23` | emit from worker 5 never reaches a client on worker 2 |

**This is the single most important scaling insight about this project:** it isn't slow because of
inefficient code (though there's some of that). It's un-scalable because **state lives in the
process**. Every fix in this file is fundamentally about moving state out.

```
Today:                          Target:
┌─────────────────┐             ┌──────┐ ┌──────┐ ┌──────┐
│ Flask + Chroma  │             │ web  │ │ web  │ │worker│
│ + SQLite + WS   │             └──┬───┘ └──┬───┘ └──┬───┘
└─────────────────┘                └────────┼────────┘
   everything local                ┌────────┴────────┐
                                   │ Postgres+pgvector│
                                   │ Redis (queue/WS) │
                                   └──────────────────┘
```

---

## 2. Load balancing

**What it is.** A component in front of N servers distributing incoming requests.

**Algorithms worth naming:**
- **Round robin** — simplest, ignores actual load.
- **Least connections** — better when request durations vary wildly. **This is the right one for my
  workload**, because a 1M-character ingest takes minutes while `/api/health` takes microseconds.
  Round robin would happily hand a new request to a worker mid-ingest.
- **Consistent hashing** — routes the same key to the same node. Essential for distributed caches
  (and interesting for me: hashing on tenant ID would give per-tenant cache locality).
- **Sticky sessions (session affinity)** — pins a client to one server. Required for
  naive WebSocket setups.

### The WebSocket complication, and why sticky sessions are the wrong fix

Sticky sessions "work" for WebSockets but are a bad answer: they defeat even load distribution, and
when a server dies all its clients drop at once rather than degrading gracefully.

**The right fix is a message queue:**

```python
socketio = SocketIO(app, message_queue='redis://...')
```

Now any worker can `emit()`; the event goes through Redis pub/sub, and whichever worker actually holds
that client's connection delivers it. Workers become interchangeable, which is the property you
actually wanted.

**Trade-off:** Redis becomes a dependency on the critical path for real-time events, and a new failure
domain. If Redis dies, HTTP still works but live updates stop — a graceful degradation, which is the
acceptable kind.

---

## 3. Caching

**What it is.** Store the result of an expensive operation so you don't recompute it.

**The layers, outside-in:**

| Layer | Caches | Applies to me? |
|---|---|---|
| **CDN** | static assets at edge locations | marginally — see below |
| **HTTP cache** | responses, via `Cache-Control`/`ETag` | **yes** — `/api/policies` |
| **Application cache** | computed values in Redis/memcached | **yes** — embeddings, answers |
| **Database cache** | query plans, buffer pool | handled by the DB |

### What I'd cache, in order of value

**1. Query embeddings.** Every `ask()` makes an HTTP call to HuggingFace to embed the question
(`pipeline.py:274` → `embed_query`). Repeated questions are extremely common in practice. Key on
`sha256(masked_query)` → cache the 384-float vector in Redis. Removes a network round trip from the
critical path of a repeated query.

**2. LLM responses.** `temperature: 0` (`src/zero_trust_rag/config.yaml:6`) makes generation
deterministic, so identical input → identical output. Key on
`sha256(masked_question + sorted(retrieved_doc_ids))`. Including the doc IDs is what makes it correct:
if the index changes, the key changes, so you get automatic invalidation on re-ingest rather than
having to track it.

**3. `/api/policies`.** Static config. Re-fetched on every tab switch
(`static/js/app.js:172`) with no caching headers. A `Cache-Control: max-age=300` is a one-line fix.

### The cache invalidation problem

The famously hard part. Strategies:
- **TTL** — expire after N seconds. Simple; serves stale data for up to N.
- **Write-through** — update the cache when you update the source. Correct; more coupling.
- **Event-based** — publish invalidation messages on change. Correct; needs infrastructure.
- **Key-includes-version** — bake the inputs into the key so a change *is* a different key. Cheapest
  correct option, and what the doc-IDs-in-key trick above does.

### The danger specific to *this* system — say this before they ask

A cache in a PII system is a **privacy boundary**, not just a performance optimization.

- A response cache holds masked LLM output — safe in itself.
- But if the key doesn't include **tenant identity**, user B gets user A's cached answer. And since
  `unmask_text` will happily resolve any token (file 04, main doc §5.2), B's cached answer gets
  unmasked with **A's PII**.

So: **authentication and tenancy must land before caching.** Adding a cache to the current
single-tenant design would convert a latent design gap into an active data breach. Sequencing security
before performance — and explaining *why* in mechanism terms — is a strong signal.

Also: never cache the **unmasked** answer. Cache masked output and unmask per request. That keeps
plaintext out of Redis entirely, which matters because Redis persistence and replication are much
harder to reason about than a single Postgres table.

---

## 4. Message queues and asynchronous processing

**What it is.** Instead of doing slow work inside the request, put a job on a queue, return
immediately, and let a separate worker process it.

```
Now:      POST /api/ingest ──────────── 4 minutes ───────────→ 200 OK
Better:   POST /api/ingest → enqueue → 202 Accepted {job_id}
                              ↓
                          Celery worker → redact → embed → store
                              ↓
                          publish progress → Socket.IO → browser
```

### Why this is the single biggest architectural win available here

My ingest path is minutes long: Presidio NER over the whole document
(`pii_redactor.py:124`), then thousands of sequential embedding HTTP calls
(`pipeline.py:44-53`). During that time, with `-w 1`, **the entire service is blocked** — including
the health check, which can get the instance killed mid-request.

**And the client contract already exists.** My Socket.IO events are already phase-shaped:

```
ingestion_start → pii_detected → ingestion_complete      app.py:64, :74, :81
```

That is *exactly* a job-progress protocol. The frontend already renders it as a live feed
(`static/js/app.js:66-100`). Moving to a queue requires almost no client change — which is a nice
thing to be able to point out, because it means the original design accidentally anticipated this.

### Queue concepts worth having ready

- **At-least-once delivery** — the norm. Messages can be redelivered, so **handlers must be
  idempotent**. My content-hash IDs (`pipeline.py:200`) already give me that for the vector write,
  and the vault's deterministic lookup (`vault.py:75`) gives it for tokens. So a redelivered ingest
  job is naturally safe. That's a genuine, non-obvious strength of the existing design.
- **Dead-letter queue** — after N failed attempts, park the message for inspection instead of
  retrying forever.
- **Backpressure** — when the queue grows faster than workers drain it, you must shed load or
  scale workers. Queue depth is the metric to alarm on.
- **Ordering** — most queues don't guarantee it across workers. Doesn't matter for me: ingests are
  independent.

**Trade-offs:** a broker (Redis/RabbitMQ) and a worker process are new failure domains and new ops
burden. The UI must handle pending/failed states, which is real frontend work. Debugging becomes
harder — a failure is now in a worker log, not in the request trace. And "did my upload work?"
becomes a question the UI has to answer explicitly rather than implicitly via the response.

---

## 5. Database scaling: replication, sharding, partitioning

### Replication

Copies of the same data on multiple nodes.

- **Leader/follower (primary/replica):** writes to the leader, reads from any follower. Scales
  **reads**, gives failover. Introduces **replication lag** — a read right after a write may not see
  it (read-after-write inconsistency).
- **Multi-leader:** writes anywhere, conflicts must be resolved. Complex; for multi-region.
- **Quorum-based:** write to W of N, read from R of N, with R + W > N for strong consistency.

**Applied to my vault:** the workload is read-heavy for `unmask` and `resolve`, write-heavy during
ingest. A leader with read replicas fits. **But replication lag would be a correctness bug here:**
`store()` writes a token, then a subsequent `resolve()` hitting a lagging replica returns `None`, and
the token silently fails to unmask. So vault reads on the unmask path must go to the **leader**, or
use a read-your-writes guarantee. Recognizing that lag becomes a *correctness* issue rather than a
freshness annoyance is the sophisticated point.

### Sharding (horizontal partitioning)

Split data across nodes by a key, so each node holds a subset.

- **Shard key choice is everything.** A bad key creates **hot shards**.
- **Range sharding** (A–M, N–Z) — good for range scans, prone to hotspots.
- **Hash sharding** — even distribution, no range queries.
- **Consistent hashing** — minimizes data movement when nodes are added or removed.

**Applied to my vault:** the natural shard key is `tenant_id` — once tenancy exists. It's ideal
because every query is already naturally tenant-scoped (or should be), so no query needs to fan out
across shards. That's the property you want: **the shard key should match your access pattern**, and
here the access pattern is "one tenant's data."

Sharding on `real_value` hash would also distribute evenly but would make "show me this tenant's
vault" a scatter-gather across every shard. Being able to compare two candidate keys on that basis is
the actual skill being tested.

### Partitioning

Splitting one table within one database, usually by time. **My audit log is the textbook case** —
append-only, time-ordered, queried recently-first (`audit_logger.py:61`, `limit=50`). Monthly
partitions mean old data can be dropped with `DROP PARTITION` (instant) rather than
`DELETE WHERE timestamp < ...` (slow, bloats the table), and recent-first queries only touch the
newest partition.

---

## 6. Rate limiting

**What it is.** Cap how many requests a client can make in a window.

**Algorithms:**
- **Fixed window** — N per minute. Simple; allows a 2N burst at a window boundary.
- **Sliding window** — smooths that out; more state.
- **Token bucket** — tokens refill at a steady rate; a request costs one. **Allows controlled bursts**,
  which is usually what you want. The most common production choice.
- **Leaky bucket** — enforces a strictly constant output rate.

### Why this project needs it urgently

`POST /api/ingest` accepts up to 1,000,000 characters (`utils.py:55`) of the most expensive work in
the system, with **no authentication and no rate limit**. One client, one loop: the service is down.
This isn't a scaling nicety, it's an availability vulnerability in the current code.

**Weighted limiting is the right design here**, because my endpoints differ in cost by orders of
magnitude:

| Endpoint | Relative cost | Suggested budget |
|---|---|---|
| `/api/health` | ~0 | unlimited |
| `/api/policies` | trivial | generous |
| `/api/ask` | 1 embed + 1 ANN + 1 LLM | moderate |
| `/api/ingest`, `/api/upload` | minutes of CPU + thousands of HTTP calls | **very tight** |

Plus `MAX_CONTENT_LENGTH` on the Flask app, which is missing entirely and is the cheapest possible
mitigation for the upload path (`app.py:155` writes to disk before any validation).

**Trade-offs:** limits need **shared state** (Redis) to work across workers — in-memory counters are
per-worker and trivially bypassed by reconnecting. Legitimate bulk users will hit limits, so you need
per-key tiers. And rate limiting is a *mitigation*: the real fix is making ingestion asynchronous
(§4) so a burst queues instead of blocking.

---

## 7. CDN and edge caching

**What it is.** Geographically distributed servers caching static assets near users. Reduces latency
and offloads origin traffic.

**Honest applicability here:** small. My static payload is `style.css` (803 lines) and `app.js`
(654 lines), served by Flask (`templates/index.html:9`, `:207`). Two small files. A CDN would help
marginally with global latency and would remove static serving from my single worker — which,
given `-w 1`, is not entirely nothing.

**The more interesting angle — I already depend on CDNs, and that's a risk:**

```html
<script src="https://unpkg.com/@phosphor-icons/web"></script>              <!-- index.html:13 -->
<script src="https://cdn.jsdelivr.net/npm/motion@11.11.13/..."></script>   <!-- index.html:14 -->
<script src="https://cdn.socket.io/4.7.2/socket.io.min.js"></script>       <!-- index.html:15 -->
<link href="https://fonts.googleapis.com/css2?family=..."/>                <!-- index.html:12 -->
```

Two real problems:

1. **Availability.** If unpkg is down, my dashboard breaks. I've added third-party uptime dependencies
   to an app whose whole selling point is not trusting third parties.
2. **Supply chain — and this one actually matters.** No **Subresource Integrity** hashes. A compromised
   CDN serves arbitrary JavaScript into a page that **displays plaintext PII from `/api/vault`**. That's
   a direct exfiltration path.

**The fix:** add `integrity="sha384-..." crossorigin="anonymous"` so the browser refuses a modified
file, or better, vendor the files locally. For a privacy-focused application, self-hosting is the
consistent choice — and noticing that my CDN usage contradicts my own threat model is exactly the kind
of self-critique that lands well.

Note `<link rel="preconnect">` at `templates/index.html:10-11` — a real performance optimization (DNS
+ TLS handshake started early). Small, but it shows frontend performance awareness.

---

## 8. Observability: the three pillars

You can't scale what you can't measure. Interviewers expect this vocabulary.

| Pillar | Question it answers | In my repo? |
|---|---|---|
| **Logs** | what happened? | ✅ `audit_logger.py`, plus stdlib `logging` |
| **Metrics** | how much, how fast? | ❌ **nothing** |
| **Traces** | where did the time go in *this* request? | ❌ **nothing** |

### What exists

Structured JSON Lines audit logging (`audit_logger.py:30-59`) — genuinely good, and SIEM-ready by
design. Plus standard `logging` calls through the pipeline (`pipeline.py:245`, `:309`).

### What's missing, and why it matters for scaling

**Metrics.** I have no idea what my p50/p95/p99 latency is, what my PII detection rate is, or how often
the HF fallback path fires. So every performance claim in this prep is *reasoned from code, not
measured* — which is why the main doc flags the baseline as **[confirm with me]**. What I'd instrument:

- `redaction_duration_seconds` (histogram, by entity count) — is Presidio the bottleneck I think it is?
- `embedding_requests_total` and duration — quantifies the batching win.
- `vault_size` (gauge) — directly predicts when `unmask_text` degrades.
- `entities_detected_total` by type — detects a broken recognizer, e.g. the dead 0.5-score phone
  pattern (file 02 §4).
- `embedding_fallback_total` — right now the fallback fires **silently** (`pipeline.py:88-89`
  swallows the exception), so I'd never know the preferred path was failing.

That last one is the good example: an unmonitored silent fallback is a bug you can run in production
for months without noticing.

**Traces.** A query touches redaction → embedding → ANN → LLM → unmask. With distributed tracing
(OpenTelemetry) you'd see the per-stage breakdown for a single slow request. Without it, "the query was
slow" is unactionable.

**Health checks.** `/api/health` (`app.py:348`) checks liveness only — the process is up. It does
**not** check whether Groq is reachable, HF is reachable, or the vault is writable. A proper readiness
check would verify dependencies, so a load balancer stops routing to a worker whose upstreams are
dead. The distinction (liveness = restart me; readiness = don't route to me) is worth stating
explicitly.

---

## 9. Putting it together: the scaling roadmap

The order matters, and being able to justify the *sequence* is what distinguishes a system-design
answer from a list of technologies.

### Phase 1 — Correctness and security (before any scaling)

Scaling a leaky system multiplies the leak.

1. Authentication on all endpoints; gate or remove `/api/vault`.
2. Tenancy: `tenant_id` in the vault schema, Chroma metadata filters, Socket.IO rooms, scoped
   `unmask_text`.
3. `MAX_CONTENT_LENGTH`; tighten SocketIO CORS off `*`.
4. Fix the dead root `config.yaml` and the `security` vs `presidio` key bug.

### Phase 2 — Cheap performance wins (no new infrastructure)

Order-of-magnitude gains for a day of work:

5. **Batch the embedding calls** — 2,500 HTTP requests → ~50 (`pipeline.py:44-53`).
6. **Invert `unmask_text`** — O(V×L) → O(L) (`vault.py:116`).
7. **Batch vault writes** — one transaction per document instead of per entity.
8. **Fix the O(n²) string building** — accumulate segments, `''.join()` once (`pii_redactor.py:173`).
9. **Enable SQLite WAL mode** — one `PRAGMA`, real concurrency improvement.
10. **Add timeouts** to the `InferenceClient` and `ChatGroq` calls.

### Phase 3 — Externalize state (unlocks horizontal scaling)

11. Vault → **Postgres** with `ON CONFLICT ... RETURNING` (kills the minting race).
12. Vectors → **pgvector** in the same instance (transactional consistency with the vault).
13. Audit logs → stdout JSON → a real log sink.
14. **Redis** for the Socket.IO message queue.
15. Now `-w N` is safe. Add a **load balancer** (least-connections).

### Phase 4 — Asynchronous processing

16. **Celery + Redis**; `/api/ingest` returns `202` with a job ID.
17. Progress published by the worker, relayed through the existing Socket.IO events.
18. Dead-letter queue and queue-depth alerting.

### Phase 5 — Caching and rate limiting (safe only now)

19. **Rate limiting** (Flask-Limiter + Redis), weighted per endpoint.
20. **Query embedding cache** and **response cache**, keyed with tenant + doc IDs.
21. HTTP cache headers on `/api/policies`.

### Phase 6 — Quality and scale-out

22. Local embeddings or a self-hosted TEI container — removes the network from the critical path.
23. **Hybrid retrieval** (BM25 + vector) — exploits the exact-token property (file 01 §8).
24. A **reranker** — retrieve 50, cross-encode, keep 5.
25. Metrics and tracing throughout; shard by `tenant_id` when one Postgres stops being enough.

### The one-sentence version

> "Phase 1 is security, because scaling a leaky system multiplies the leak. Phase 2 is the cheap
> algorithmic wins that need no new infrastructure. Phase 3 externalizes state, which is the actual
> unlock for horizontal scaling. Then queues, then caching — and caching has to come after tenancy,
> or the cache becomes the breach."

---

## Concept checklist for this file

- [ ] I can explain why my app is vertical-only, naming all three pieces of process-local state
- [ ] I can explain why least-connections suits my workload and why sticky sessions are the wrong
      WebSocket fix
- [ ] I can name three things I'd cache and explain why caching must come *after* tenancy
- [ ] I can explain why my Socket.IO events already form a job-progress protocol
- [ ] I can explain why at-least-once delivery is safe here (stable IDs + deterministic tokens)
- [ ] I can explain why replication lag would be a correctness bug for the vault, not just staleness
- [ ] I can justify `tenant_id` as the shard key by reference to the access pattern
- [ ] I can explain token bucket vs. fixed window, and why my limits must be weighted
- [ ] I can explain the SRI supply-chain risk in my own `index.html`
- [ ] I can name the three pillars and identify the silent embedding fallback as the metric I most need
- [ ] I can justify the *ordering* of the roadmap, not just its contents
