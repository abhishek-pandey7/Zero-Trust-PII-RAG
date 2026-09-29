# Zero-Trust PII RAG — Interview Prep

> **Role/level:** left blank on purpose. **[confirm with me]** — the framing of a few answers
> (how much I lean on "I'd own the migration" vs. "I'd raise it with my tech lead") shifts
> between, say, SDE-1 and SDE-3. Everything technical below is role-independent.

Every claim here points at a file and line in this repo. Where the code doesn't reveal my
reasoning, it's marked **[confirm with me]** rather than invented.

**Companion reading** (the "explain the concept to me" files):
- [01 — RAG & Retrieval](./01-concepts-rag-and-retrieval.md)
- [02 — PII, Tokenization & Zero-Trust](./02-concepts-pii-and-zero-trust.md)
- [03 — Backend, APIs & Real-Time](./03-concepts-backend-and-realtime.md)
- [04 — Storage, Concurrency & Idempotency](./04-concepts-storage-and-concurrency.md)
- [05 — Scaling Concepts](./05-concepts-scaling.md)

---

## Table of Contents

1. [Project pitch (60 seconds and 3 minutes)](#1-project-pitch)
2. [System design concepts I actually used](#2-system-design-concepts-i-actually-used)
3. [Framework and tech-stack choices](#3-framework-and-tech-stack-choices)
4. [If this scales — 10x and 100x](#4-if-this-scales--10x-and-100x)
5. [Weak spots an interviewer will poke at](#5-weak-spots-an-interviewer-will-poke-at)
6. [15 likely questions with model answers](#6-15-likely-questions-with-model-answers)

---

## 1. Project pitch

### The 60-second version

> "The problem I was solving is this: everyone wants to point an LLM at their internal documents,
> but the moment you call an external API like OpenAI or Groq, your customers' names, Aadhaar
> numbers and salary data leave your network. For anyone handling regulated data that's a
> non-starter.
>
> So I built a RAG pipeline where the external LLM structurally cannot see PII. Before any text is
> embedded or sent anywhere, it goes through Microsoft Presidio for entity detection, and every hit
> gets swapped for an opaque token — `<<PERSON_04_a1b2>>`. The real value goes into a local SQLite
> vault keyed by that token. So the vector database stores masked text, the LLM reasons over masked
> text, and it answers in masked text. The very last step, on my own server, is a reverse lookup
> that swaps tokens back to plaintext before the user sees it.
>
> The design detail that makes it actually work is that tokenization is *deterministic* — the same
> person always maps to the same token, so semantic search and the LLM's reasoning still hold
> together. And there's a Flask dashboard on top with a live WebSocket feed, a vault explorer and
> the compliance audit log, because for this kind of system 'prove it redacted' matters as much as
> 'it redacted'."

### The 3-minute version

Open the same way, then go deeper along the data flow:

**The trust boundary.** The whole design hangs on one line I can draw: everything inside my
process is trusted, everything past the network call is not. `pipeline.py` is written so redaction
happens *before* that boundary in every path — `ingest()` redacts at `pipeline.py:232` before
chunking or embedding, and `ask()` redacts at `pipeline.py:271` before retrieval or inference.
There is no code path where raw text reaches Groq.

**Ingestion.** Text arrives via `POST /api/ingest` or a file upload (`app.py:49`, `app.py:143` —
`.txt`, `.pdf` via PyMuPDF, `.docx` via python-docx). First it's sanitized of control characters
(`pii_redactor.py:113`) — that's both hygiene and a small prompt-injection hardening step. Then
Presidio runs spaCy NER plus my three custom regex recognizers for Indian identifiers: Aadhaar,
PAN, and Indian phone formats (`pii_redactor.py:42-83`). Detected spans get post-processed — I
truncate any span crossing a newline (`pii_redactor.py:139`), because in structured records spaCy
loves to swallow `"John Doe\nEmail:"` as one PERSON entity, and I trim possessives so `"John's"`
and `"John"` resolve to the same token (`pii_redactor.py:151`). Overlapping spans go through a
greedy sweep (`pii_redactor.py:191`), then replacement runs right-to-left (`pii_redactor.py:165`)
so earlier character offsets stay valid while I mutate the string.

**The vault.** Each unique value hits `PIIVault.store()` (`vault.py:72`), which first checks
`value_mappings` to see if this exact value already has a token. That lookup is the heart of the
design — it's what makes tokens deterministic across documents and across restarts. New values get
`<<{ENTITY_TYPE}_{counter:02d}_{uuid4[:4]}>>`, written to three tables under a `threading.Lock`
(`vault.py:28`, schema at `vault.py:33-56`).

**Vectorization.** The masked text is chunked at 500 chars with 100 overlap by
`RecursiveCharacterTextSplitter` (`pipeline.py:185`), embedded through a custom `Embeddings`
adapter I wrote against the HuggingFace Inference API (`pipeline.py:20-96`), and upserted into
ChromaDB with content-derived SHA-256 IDs (`pipeline.py:200`) so re-ingesting the same document
doesn't duplicate rows.

**Querying.** The question is redacted with the *same vault*, so "What is John Doe's Aadhaar?"
becomes "What is `<<PERSON_01_3f9c>>`'s Aadhaar?" — and because the stored chunks use that identical
token, vector similarity still matches. Top-k chunks (`top_k: 20` in the active config) plus the
masked question go to Groq (`pipeline.py:292`), and the masked answer comes back through
`redactor.unmask()` → `vault.unmask_text()` (`vault.py:116`) for the final substitution.

**Observability.** Every operation writes a JSON line to an audit log (`audit_logger.py:30-59`) —
JSON Lines specifically so a SIEM can ingest it without a custom parser. The dashboard surfaces
that plus the vault contents and the active policy config, and pushes live progress over Socket.IO
so a 1,000-record document doesn't look like a hung spinner.

**If they ask what I'd change first:** two things — the vault is unauthenticated and single-tenant,
and embedding is one sequential HTTP call per chunk. Section 5 has the honest version.

---

## 2. System design concepts I actually used

### 2.1 Tokenization / reversible pseudonymization

**What it is.** Replacing a sensitive value with a surrogate that carries no information about the
original, keeping the mapping in a separate secured store. The data stays *usable* (unlike hashing
or `****`) but the token itself leaks nothing. This is the standard pattern behind PCI card-data
tokenization.

**Where it is in my code.** `PIIVault.store()` at `vault.py:72` mints the token; the format
`<<{entity_type}_{count:02d}_{short_id}>>` is at `vault.py:87`. `pii_redactor.py:172-173` does the
substitution into the text.

**Why I chose it.** Inferred from the code, and I'm confident: masking to `[REDACTED]` would have
destroyed the pipeline, because RAG needs to distinguish two different people in the same chunk.
The token keeps entity *identity* while dropping entity *value*. The fact that `entity_type` is
baked into the token also means the LLM can still reason about types — it knows
`<<EMAIL_ADDRESS_02_...>>` is an email, which helps it answer "what's their contact?"

**Alternatives I could have used.**
- *Format-preserving encryption.* A ciphertext that looks like a real phone number. Better for
  systems that validate formats downstream, but it's reversible with a key rather than a lookup,
  so key compromise is total compromise, and it's much heavier to implement correctly.
- *Irreversible hashing (SHA-256 of the value).* Simpler, no vault to protect. But then I can never
  unmask, so the user gets an answer full of hex digests. Fundamentally incompatible with the
  product.

**Trade-offs I accepted.** The vault is now the crown jewels — a single table that reverses all my
privacy guarantees. And because `value_mappings.real_value` is the primary key (`vault.py:45`), the
same string under two different entity types collapses to one token.

---

### 2.2 Deterministic mapping (the thing that makes RAG still work)

**What it is.** The same input always produces the same token, forever, across processes and
restarts.

**Where it is.** `vault.py:75-81` — the read-before-write against `value_mappings`. If the value
exists, return the existing token and don't mint a new one.

**Why I chose it.** This one I'm certain about, because the system is broken without it. Two things
depend on it: (a) a chunk ingested on Monday and a query asked on Friday must use the same token or
cosine similarity won't match them; (b) `unmask_text` has to resolve tokens the LLM saw in
*retrieved context*, which were minted during a completely different request.

**Alternatives.**
- *Random token per occurrence.* Maximum privacy — even frequency analysis on the vector store
  tells you nothing. But retrieval collapses: "John Doe" in the query becomes a token that appears
  nowhere in the index.
- *Token derived as `HMAC(key, value)`.* Deterministic with no lookup table needed for minting.
  Genuinely attractive, and I'd consider it in v2. But you still need a table for *reverse* lookup,
  so it saves less than it appears to.

**Trade-offs.** Determinism leaks structure. An attacker who reads my ChromaDB store can count how
many times `<<PERSON_01_3f9c>>` appears and which documents co-mention it. That's a real correlation
channel — I traded a slice of privacy for the system working at all.

---

### 2.3 Trust boundary / control-plane vs. data-plane split

**What it is.** Explicitly deciding which components are trusted with plaintext, and designing so
untrusted components structurally cannot receive it.

**Where it is.** The ordering inside `pipeline.py`: `ingest()` calls `redact()` at line 232 before
`split_text` at 236 and before `_add_texts_to_vector_db` at 238. `ask()` calls `redact()` at 271
before `similarity_search` at 274 and before `chain.invoke` at 295. Unmasking is the final step at
302.

**Why I chose it.** [Mild inference, but the docstring at `pipeline.py:128-133` states the model
explicitly, so it was deliberate.] Compliance is about provable properties. "PII is masked before
every egress point" is a property I can audit by reading one file.

**Alternatives.**
- *Self-hosted LLM, no redaction.* Genuinely solves the problem and you keep full fidelity. But
  you're now paying for GPUs and you've inherited an inference-serving ops problem — the whole
  appeal of Groq here is that it's someone else's cluster.
- *Redact at the network egress layer (a proxy scrubbing outbound requests).* Centralizes the
  guarantee, which is nice. But it can't do the reverse mapping coherently, and it has no idea
  which parts of a JSON payload are prompt vs. metadata.

**Trade-offs.** The LLM is now reasoning on degraded input. And my active config masks `DATE_TIME`,
`LOCATION` and `URL` (`src/zero_trust_rag/config.yaml:30-34`), which strips genuinely useful
semantics — "joined in March 2023 in Mumbai" becomes two opaque tokens, so the model can't do date
arithmetic or geographic reasoning at all.

---

### 2.4 RAG (retrieval-augmented generation)

**What it is.** Instead of fine-tuning a model on your data, you retrieve relevant passages at
query time and place them in the prompt as context.

**Where it is.** `ask()` at `pipeline.py:258` — `similarity_search` at 274, context assembly at
275, a `ChatPromptTemplate` at 281, and LCEL composition `prompt | llm | StrOutputParser()` at 292.

**Why I chose it.** Inferred: the dataset is employee records that change, and the tokens change as
the vault grows. Fine-tuning on tokenized text would bake in a snapshot of a mapping that's mutable
by design. RAG keeps the index as the source of truth.

**Alternatives.**
- *Fine-tuning.* Lower per-query latency, no retrieval infra. But retraining per document update
  is absurd here, and it makes deletion (GDPR erasure) essentially impossible.
- *Stuff the whole corpus into a long-context prompt.* Fine for small corpora. My test document is
  over 1,000,000 characters — roughly 250k+ tokens — and I'd pay for all of it on every query.

**Trade-offs.** Answer quality is now capped by retrieval quality. If the right chunk isn't in the
top-k, the model confidently answers from nothing. There's no reranker and no retrieval-quality
evaluation in this repo — see section 5.

---

### 2.5 Chunking with overlap

**What it is.** Splitting documents into pieces small enough to embed meaningfully, with a sliding
overlap so facts spanning a boundary survive intact in at least one chunk.

**Where it is.** `pipeline.py:185-190` — `RecursiveCharacterTextSplitter`, `chunk_size: 500`,
`chunk_overlap: 100`, separators `["\n\n", "\n", " ", ""]` from
`src/zero_trust_rag/config.yaml:15-18`.

**Why I chose these numbers.** **[confirm with me]** — 500/100 is a common default and the code
doesn't record a tuning experiment. What I *can* defend from the code: the separator list is ordered
coarse-to-fine, so the splitter prefers paragraph breaks over mid-word cuts, which matters a lot for
record-structured data where one record is one paragraph.

**Alternatives.**
- *Fixed-size split with no overlap.* Simpler, no duplicated storage. But a record split across two
  chunks loses the name→value association, which is exactly the query pattern here.
- *Semantic chunking (split where embedding similarity drops).* Better boundaries in prose. But it
  requires embedding the document twice, and my embeddings are remote HTTP calls — the cost would
  roughly double for records already cleanly delimited by newlines.

**Trade-offs.** 20% overlap means ~20% more vectors to store and embed. And a 500-char window on
tokenized text holds fewer real facts than on plaintext, because `<<EMAIL_ADDRESS_07_9d1f>>` costs
~26 characters where the original email cost ~20.

---

### 2.6 Vector embeddings and approximate nearest-neighbour search

**What it is.** Map text to a dense vector so semantic closeness becomes geometric closeness, then
use an index (HNSW, in Chroma's case) to find nearest neighbours in sublinear time.

**Where it is.** `_HFApiEmbeddings` at `pipeline.py:20`, `Chroma` init at `pipeline.py:195`,
`similarity_search(masked_q, k=top_k)` at `pipeline.py:274`. The `.bin` files under `chroma_store/`
— `link_lists.bin`, `data_level0.bin` — are HNSW's graph layers on disk.

**Why I chose it.** Inferred: the queries are natural-language questions, not keyword lookups. "Who
works in the Mumbai office?" has to match text that never contains the word "works".

**Alternatives.**
- *BM25 / full-text search.* Cheap, no embedding API, and exact-term recall is actually *better*.
  Here's the interesting wrinkle — after tokenization my documents are full of exact literal tokens,
  so BM25 on `<<PERSON_01_3f9c>>` would be excellent. I'd genuinely argue hybrid retrieval is the
  right answer for this system, and I'd raise that myself.
- *SQL `LIKE` over the raw records.* Works for structured lookups, useless for "summarize the
  engineering team".

**Trade-offs.** ANN is approximate — recall isn't guaranteed. And I'm at the mercy of a remote
embedding API for both indexing and every query, which puts a network round trip on the critical
path of every single search.

---

### 2.7 Content-addressed IDs for idempotent writes

**What it is.** Derive a record's primary key from a hash of its content, so writing the same
content twice is a no-op upsert instead of a duplicate insert.

**Where it is.** `_stable_id()` at `pipeline.py:200-205` — SHA-256 of the chunk text, reformatted
into a UUID shape — used at `pipeline.py:216-217`.

**Why I chose it.** The comment at `pipeline.py:214-215` says it outright: re-ingesting the same
text was producing duplicate ChromaDB entries with fresh UUIDs. So this was a bug fix. The symptom
would have been retrieval returning the same chunk three times and crowding out other context.

**Alternatives.**
- *Check-then-insert (query for existence first).* Doubles round trips and still races under
  concurrency.
- *A `document_id` in metadata plus a delete-then-reinsert cycle.* Handles *updated* documents
  correctly, which content hashing does not. That's a real gap — see the trade-off.

**Trade-offs.** This makes re-ingestion idempotent but makes *editing* invisible. Change one
character and you get a brand new ID, so the old chunk lives on forever. There's no tombstoning, so
the index accumulates stale versions. `clear_vector_db()` (`pipeline.py:326`) and
`reset_vector_db()` (`pipeline.py:343`) are the blunt escape hatches.

---

### 2.8 Singleton + lazy initialization

**What it is.** One shared instance of an expensive object, created on first use rather than at
import.

**Where it is.** Two of them. `Config` uses a `__new__`-based singleton with class-level `_data`
(`config.py:9-17`). And the pipeline itself is a module global with a lazy getter — `rag = None` at
`app.py:29`, `get_rag()` at `app.py:31`.

**Why I chose it.** Inferred and well-supported: `PIIRedactor.__init__` builds a spaCy NLP engine
and a Presidio `AnalyzerEngine` (`pii_redactor.py:96-106`). That's hundreds of MB and several
seconds. Doing it per request would be untenable; doing it at import time would mean Render's health
check times out during a cold deploy. Lazy-on-first-request is the right shape — and note
`/api/health` at `app.py:348` deliberately reports `rag_initialized` without forcing init.

**Alternatives.**
- *Per-request instantiation.* Clean isolation, no shared state. Multi-second latency per request —
  non-viable.
- *A worker pool with pre-warmed instances.* What I'd actually want in production. More memory, more
  complexity, and it needs more than the one free-tier worker I have.

**Trade-offs.** `get_rag()` is a check-then-set with no lock, so two concurrent first requests can
both build a pipeline. I own that as a bug in section 5. Also the `Config` singleton caches at first
import, so config changes need a restart.

---

### 2.9 Externalized configuration

**What it is.** Behaviour-controlling values live in a data file, not in code; secrets live in the
environment.

**Where it is.** `src/zero_trust_rag/config.yaml`, read by `Config._load()` at `config.py:19` with
dotted-path access (`config.get("llm.model")`) at `config.py:26`. Secrets come from `os.getenv` at
`pipeline.py:159-160`, with `.env` loaded in `app.py:15` and gitignored. The deploy contract is in
`render.yaml` with `sync: false` on both API keys so they're never in the repo.

**Why I chose it.** Inferred: model names, thresholds and top-k are exactly the knobs you tune while
demoing, and you want to change them without a code review.

**Alternatives.**
- *Constants in code.* Simplest, and honestly fine for a solo project. But `presidio.score_threshold`
  is a security-relevant dial — I want it visible in a config diff and surfaced in the UI via
  `/api/policies`.
- *A config service / feature-flag system.* Runtime changes without a restart. Massive overkill for
  one process.

**Trade-offs.** This bit me. There are **two** `config.yaml` files in the repo and only one is read
— `config.py:6` resolves relative to the module, so the root `config.yaml` is dead and has drifted
(`8b-instant` vs `70b-versatile`, `top_k` 3 vs 20, threshold 0.3 vs 0.6). Section 5.

---

### 2.10 Bounded retry for transient failures

**What it is.** Wrap a call that can fail transiently and retry a fixed number of times before
giving up.

**Where it is.** The `retry_on_api_error` decorator at `utils.py:18-44`, applied to
`_add_texts_to_vector_db` (`pipeline.py:207`) and `ask` (`pipeline.py:257`). Config:
`max_retries: 3`, `retry_delay: 1.0`.

**Why I chose it.** Inferred, and the code supports it: both HuggingFace's free inference tier and
Groq return 503s under load, and HF cold-starts models — the raw-requests fallback even sets
`wait_for_model: True` (`pipeline.py:63`). A user shouldn't see a failure for a model warm-up.

**Alternatives.**
- *Exponential backoff with jitter.* Strictly better — a fixed 1s delay from many clients creates a
  thundering herd against an already-struggling upstream. `tenacity` is *already in my
  requirements.txt* and I hand-rolled this instead. I'd fix that.
- *A circuit breaker.* The right answer at scale: stop hammering a dead upstream and fail fast.
  Needs shared state to be useful across workers.

**Trade-offs.** Worst case adds ~3 seconds to a failing request with no user feedback in between.
And there's a real bug: the decorator only catches `APIError` (`utils.py:30`), but
`_embed_via_requests` raises a bare `RuntimeError` (`pipeline.py:79`), so embedding failures aren't
retried at all. The decorator on `_add_texts_to_vector_db` is close to dead code.

---

### 2.11 Structured audit logging

**What it is.** Emit machine-parseable events (one JSON object per line) rather than prose log
strings, so downstream tooling can query them.

**Where it is.** `audit_logger.py` — `log_redaction` (line 30), `log_operation` (line 42),
`log_access` (line 52), each writing `json.dumps(event)` through a `logging.FileHandler` whose
formatter is a bare `%(message)s` (line 27) so nothing corrupts the JSON. The docstring at line 17
says it explicitly: "JSON lines for easy parsing by SIEM tools."

**Why I chose it.** Inferred, and it's the standard move for compliance: an auditor's question is
"show me every time PII of this entity type was unmasked," and that's a query, not a grep.

**Alternatives.**
- *Plain text logs.* Human-readable, unparseable at volume.
- *Log to a database table.* Queryable immediately, transactional with the vault write. But it
  couples audit availability to DB availability, and append-only files are harder to tamper with —
  which for an audit trail is a feature.

**Trade-offs.** `text_preview` truncates to 50 chars (`audit_logger.py:37`) — and note I pass
`safe_text` (post-masking) at `pii_redactor.py:185`, so the audit log deliberately doesn't contain
plaintext PII. That was the right call. The weak spot is `get_recent_logs` doing `f.readlines()` on
the whole file (`audit_logger.py:70`) — fine at demo size, a memory problem at GB scale.

---

### 2.12 Custom exception hierarchy mapped to HTTP semantics

**What it is.** A tree of domain exceptions so callers can catch at the granularity they care
about, with each mapped to a meaningful status code at the API edge.

**Where it is.** `exceptions.py` — `ZeroTrustRAGError` base, with `ConfigurationError`, `APIError`
(carrying `provider` and `status_code`), `VaultError`, `ValidationError`, `RedactionError`. The
mapping is in `app.py:262-273`: `ValidationError` → 400, `APIError` → **503**, generic
`ZeroTrustRAGError` → 500.

**Why I chose it.** Inferred, and the 503 is the tell — it's deliberate. A Groq outage isn't my bug,
it's upstream unavailability, and 503 tells the client "retry later" where 500 says "something's
broken here." Also `pipeline.py:313-314` re-raises `ZeroTrustRAGError` untouched so it doesn't get
double-wrapped.

**Alternatives.**
- *Return error dicts instead of raising.* Explicit, no hidden control flow, but every call site has
  to check and errors get silently dropped.
- *Just use builtins.* Less code. But `except ValueError` can't distinguish "user sent 2MB of text"
  from "Groq is down", and those need different status codes and different alerting.

**Trade-offs.** `ingest()` wraps *everything* in `RedactionError` (`pipeline.py:255`), including
network failures from the embedding call — so the exception type actively misleads you about where
the failure was.

---

### 2.13 WebSocket push for long-running work

**What it is.** A persistent bidirectional connection so the server can push progress events
instead of the client polling.

**Where it is.** `Flask-SocketIO` at `app.py:23`; emits at `app.py:64` (`ingestion_start`),
`app.py:74` (`pii_detected`), `app.py:81` (`ingestion_complete`), `app.py:230`
(`query_processing`), `app.py:247` (`query_complete`). Client handlers at
`static/js/app.js:50-130`.

**Why I chose it.** Inferred with high confidence: ingesting a 1M-character document means a
multi-minute Presidio pass plus thousands of embedding calls. A plain request/response would look
like a hung browser tab. The event names map exactly onto pipeline phases, which is the point — the
user sees *which* phase.

**Alternatives.**
- *Polling `/api/stats`.* Simple, stateless, firewall-friendly. But you pick a bad trade-off between
  latency and load, and you can't easily stream per-entity detail.
- *Server-Sent Events.* Actually a better fit — this traffic is entirely server→client, and SSE is
  plain HTTP with automatic reconnect. Socket.IO's bidirectionality is unused here. I'd call this
  mild over-engineering and I'd say so.

**Trade-offs.** WebSockets need sticky sessions or a shared message queue the moment you have more
than one worker — which is part of why `render.yaml` pins `-w 1`. And the emits are broadcast to
*all* connected clients with no room scoping (`app.py:64`), so two users watching the dashboard see
each other's redaction events. That's a genuine privacy bug at multi-user scale.

---

### 2.14 Adapter pattern + graceful degradation

**What it is.** Adapter: write a class conforming to an interface the framework expects, wrapping
something it doesn't natively support. Degradation: try the preferred path, fall back on failure.

**Where it is.** `_HFApiEmbeddings(Embeddings)` at `pipeline.py:20` implements LangChain's
`embed_documents`/`embed_query` contract (lines 92-96) so it drops into `Chroma` unchanged. The
fallback chain is `_embed` at `pipeline.py:84`: try `InferenceClient`, on any exception fall through
to raw HTTP, which itself tries two different URL shapes (`pipeline.py:58`) before raising.

**Why I chose it.** The docstring at `pipeline.py:21-22` gives it away: "no torch, no local models."
This is a deployment-constraint decision — `requirements.txt` comments mention "Render memory
limits" and the spaCy `sm` model was picked for the same reason. Running sentence-transformers
locally pulls in PyTorch, which blows a free-tier memory budget. The two-URL fallback is scar tissue
from HF changing their inference endpoints.

**Alternatives.**
- *`langchain_huggingface.HuggingFaceEndpointEmbeddings`.* Less code, maintained upstream. But I
  lose control of the fallback behaviour and pooling, and it pulls extra dependencies.
- *Local sentence-transformers.* No network on the critical path, no rate limits, better latency —
  and it's what I'd use with real infrastructure. Needs ~500MB+ of RAM and torch.

**Trade-offs.** I'm hand-implementing mean pooling in pure Python (`pipeline.py:36-40`) — a nested
comprehension over `seq_len × 384`, orders of magnitude slower than numpy, and numpy is already a
dependency. Worse, `except Exception: pass` at `pipeline.py:88-89` swallows the real error, so a bad
token looks identical to a model outage.

---

### 2.15 Input sanitization and validation at the boundary

**What it is.** Normalize and bound-check untrusted input at the point of entry, before it reaches
any logic.

**Where it is.** `sanitize_input` at `pii_redactor.py:108-116` strips C0 control characters, DEL,
and the C1 range. `validate_input` at `utils.py:55-63` enforces non-empty, string type, and a
1,000,000-char ceiling. `validate_api_key` at `utils.py:47` fails fast on missing/short keys at
startup rather than on first use.

**Why I chose it.** Partly inferred. Control-char stripping matters concretely: zero-width and
control characters can split a PII string so the regex misses it (`1234​5678` won't match
Aadhaar), and they can smuggle prompt-injection payloads past display layers. The 1M ceiling is a
DoS bound. [**confirm with me** on whether the C1-range strip at line 114 was a specific bug fix or
defensive breadth.]

**Alternatives.**
- *Unicode NFKC normalization.* More thorough — it'd collapse homoglyph and width variants my regex
  strip misses entirely. Worth adding.
- *Validate at the Flask layer only.* Keeps the library clean. But then the library is unsafe if
  called directly, and `demo.py`/`main.py` do exactly that.

**Trade-offs.** Note that `validate_input` runs on the *masked* text too (`pipeline.py:234`), which
is a nice touch — masking can grow text past the limit. But Flask has no `MAX_CONTENT_LENGTH` set,
so the upload endpoint (`app.py:143`) will happily buffer an arbitrarily large file to disk before
any of this runs.

---

### 2.16 Interval overlap resolution (greedy sweep)

**What it is.** Given a set of possibly-overlapping `[start, end)` spans, pick a non-overlapping
subset. The classic greedy approach: sort, then sweep keeping a running `last_end`.

**Where it is.** `_remove_overlaps` at `pii_redactor.py:191-206` — sorts by `(start, -score)`, then
accepts a span only if `result.start >= last_end`.

**Why I chose it.** Necessity: Presidio's recognizers fire independently, so a 10-digit number can
match `PHONE_NUMBER` from my custom recognizer *and* `US_SSN` from a built-in, and you can't
substitute two tokens into the same character range without corrupting offsets.

**Alternatives.**
- *Weighted interval scheduling (DP).* Actually maximizes total confidence rather than greedily
  taking the leftmost. O(n log n) too. Better results, more code.
- *Trust Presidio's own conflict resolution.* Presidio has a decision process for this. I bypassed
  it — **[confirm with me]** whether that was because it wasn't resolving my custom recognizers the
  way I wanted.

**Trade-offs.** The docstring claims it keeps "the highest scoring ones" and it doesn't — sorting by
`(start, -score)` only breaks ties *at the same start offset*. A span starting one character later
with a much higher score gets dropped. So a high-confidence Aadhaar match (0.85) loses to a
lower-confidence `DATE_TIME` that happens to start earlier. That's a real correctness gap in the
detection layer, and it's the fix I'd make first in the redactor.

---

### 2.17 Reverse-order in-place replacement (offset stability)

**What it is.** When replacing multiple spans in a string by index, iterate from the highest offset
down, so replacements never invalidate the offsets of spans you haven't processed yet.

**Where it is.** `pii_redactor.py:165` — `sorted(results, key=lambda r: r.start, reverse=True)`,
then slicing at line 173.

**Why I chose it.** Pure necessity, and it's a small detail I'd actually call out in an interview
because it shows I thought about it. Tokens are longer than most PII values, so every forward
replacement would shift all subsequent offsets and every later substitution would land in the wrong
place.

**Alternatives.**
- *Build the output by walking spans forward and appending segments.* Equivalent, arguably clearer,
  and linear instead of copying the tail on every slice. Better for large inputs.
- *`re.sub` with a callback.* Only works if you can express all PII as one pattern — you can't,
  because NER isn't a regex.

**Trade-offs.** Each replacement at `pii_redactor.py:173` builds a whole new string. On a 1M-char
document with thousands of entities that's quadratic in document size. A genuine performance issue
at my stated scale.

---

### 2.18 Layered architecture / separation of concerns

**What it is.** Each module owns one responsibility and depends only on layers below it.

**Where it is.** The dependency graph is clean and one-directional: `app.py` (transport) →
`pipeline.py` (orchestration) → `pii_redactor.py` (detection) → `vault.py` (persistence), with
`config.py`, `utils.py`, `exceptions.py` and `audit_logger.py` as cross-cutting leaves. The public
surface is curated by `__all__` in `src/zero_trust_rag/__init__.py:19` and re-exported at
`src/__init__.py`.

**Why I chose it.** Inferred from the structure, and from dependency injection being used
deliberately — `PIIRedactor.__init__` takes a vault and an optional audit logger
(`pii_redactor.py:92`) rather than constructing them. That's a testability affordance.

**Alternatives.**
- *Single-file script.* Faster to write, and honestly fine below ~300 lines. Falls apart once you
  want to swap the vault backend.
- *Full hexagonal architecture with ports and adapters.* More swappable. Ceremony that doesn't pay
  for itself at this size.

**Trade-offs.** The seams exist but I never used them — there isn't a single test in the repo, which
is the most obvious hole in the project. And `pipeline.py` defines `_HFApiEmbeddings` inline above
its own imports (lines 20-96, imports resume at 98), which breaks the layering it otherwise
respects.

---

## 3. Framework and tech-stack choices

### Flask + Flask-SocketIO (`app.py`)

**Why this.** I needed a handful of JSON endpoints plus server-push, and Flask-SocketIO gives both
in one process with almost no ceremony. Server-rendered template at `app.py:44`, so no separate
frontend build step.

**Pros.** Tiny surface area; `render_template` + `url_for` means zero frontend tooling; the SocketIO
integration is genuinely first-class; the synchronous programming model matches my code, which is
blocking end to end (sqlite3, Presidio, `requests`).

**Cons.** Flask is WSGI and synchronous, so a long Presidio pass blocks a worker. Under
`gunicorn --worker-class eventlet -w 1` (`render.yaml`, `Procfile`) my blocking CPU work stalls the
*entire* event loop — eventlet can only yield on patched I/O, and spaCy inference is CPU. So one big
ingest freezes every other user. FastAPI with `async def` endpoints and a thread-pool offload for
CPU work would be the more honest architecture, and it'd give me OpenAPI docs and Pydantic
validation for free — Pydantic is already installed and unused.

**Alternatives considered.** FastAPI (better for async + auto docs, but I'd need a separate ASGI
websocket story and my libraries are sync anyway); Django (ORM, admin and auth out of the box —
which would have solved my auth gap for free, but a lot of framework for 8 endpoints).

### Microsoft Presidio + spaCy `en_core_web_sm` (`pii_redactor.py`)

**Why this.** PII detection needs both pattern matching (Aadhaar, PAN, credit cards) and contextual
NER (names, locations). Presidio ships both plus a clean extension point — `registry.add_recognizer`
at `pii_redactor.py:104-106` let me add three Indian-specific recognizers in ~40 lines.

**Pros.** Open source, runs fully locally — which is the whole point, since a cloud PII API would
defeat the design; confidence scores let me tune precision/recall via one threshold; large built-in
recognizer set.

**Cons.** `en_core_web_sm` is the *small* model, chosen for memory (the comment in `requirements.txt`
says so). Small-model NER on structured records is mediocre — it's why I needed the
newline-truncation hack at `pii_redactor.py:139` and the possessive trimming at 151. Both are
workarounds for weak entity boundaries. Recall on non-Western names is also meaningfully worse,
which for a project targeting Indian identifiers is an uncomfortable gap. Presidio is also slow:
single-threaded spaCy on 1M chars is minutes, not seconds.

**Alternatives considered.** Regex-only (fast and predictable, but can't find names at all —
non-starter); a commercial PII API like Google DLP (better accuracy, but sending the data to a third
party to find PII is exactly the thing I built this to avoid); a fine-tuned transformer NER model
(better accuracy, needs torch and a labelled dataset).

### ChromaDB (`pipeline.py:195`)

**Why this.** Embedded, zero-ops, persists to a local directory — which matters because the vector
store holds masked-but-correlatable data and I didn't want it leaving the box.

**Pros.** `pip install` and it works; no server to run; the LangChain integration is one line; HNSW
index included; persistence to `./chroma_store` survives restarts.

**Cons.** Single-node, single-writer. No replication, so the store is a SPOF and a backup problem.
It's process-local, which is another reason I'm stuck at `-w 1`. On Render's free tier the filesystem
is ephemeral, so **the index doesn't survive a redeploy** — a real production gap, not a theoretical
one. Metadata filtering is weak compared to pgvector.

**Alternatives considered.** pgvector (one database for both the vault and the vectors —
transactional consistency between a token and its chunk, real backups, real replication; needs a
Postgres instance and is slower than dedicated ANN at very high dimensions/scale — this is what I'd
pick for v2); Pinecone/Weaviate Cloud (managed and scalable, but it's a third-party service holding
my correlatable masked data, which cuts against the trust model); FAISS (faster ANN, but no
persistence or metadata story — I'd be writing the storage layer myself).

### SQLite (`vault.py`)

**Why this.** The vault needs durability, atomicity and exact-match lookup, and nothing more. SQLite
is in the standard library, gives me ACID transactions, and the whole vault is one file I can back
up with `cp`.

**Pros.** Zero setup; ACID; `PRIMARY KEY` on both `token` and `real_value` (`vault.py:37`, `:45`)
enforces the one-to-one mapping at the schema level rather than in application code; single-file
backup.

**Cons.** Single-writer — concurrent writes serialize, and under load I'd hit `database is locked`.
No network access, so every worker would need its own file, which breaks the determinism guarantee
across processes. And critically, **the PII is stored in plaintext**; SQLite offers no at-rest
encryption without SQLCipher.

**Alternatives considered.** Postgres (concurrent writers, network-accessible, row-level security,
`pgcrypto` for column encryption — the right answer for multi-user; needs infra); Redis (fast
lookups, ideal for the `unmask` hot path as a cache, but durability is configuration-dependent and I
don't want the plaintext mapping in an in-memory store as the system of record); HashiCorp Vault or
AWS KMS+DynamoDB (the actually-correct enterprise answer: managed key hierarchy, envelope
encryption, access audit built in — heavy for a portfolio project, but the right thing to name when
asked "how would you productionize the vault").

### Groq + Llama 3.3 70B (`pipeline.py:170`)

**Why this.** The token-substitution design means the LLM is doing extraction and light reasoning
over provided context, not recalling world knowledge. That's a task where a good open model is
sufficient, and Groq's LPU inference is dramatically faster than typical GPU serving — which matters
because my pipeline already spends seconds on redaction and embedding. `temperature: 0` for
determinism, which is the right call for extraction.

**Pros.** Very low inference latency; generous free tier; open-weight model, so I have a credible
migration path to self-hosting the exact same model if the trust boundary needs to move inward;
LangChain integration via `langchain-groq`.

**Cons.** Still a third-party API — my tokens and masked context go over the network, and a
correlation attack on masked context is not nothing. Weaker instruction-following than frontier
models on adversarial context. And the two config files disagree about which model is live
(`8b-instant` vs `70b-versatile`), which I should fix.

**Alternatives considered.** OpenAI/Anthropic APIs (stronger reasoning, more reliable structured
output — but no self-hosting escape hatch, and higher cost); local Ollama (perfect trust model, zero
egress, arguably makes the whole redaction layer optional — but needs a GPU, and then the project is
about GPU ops instead of privacy architecture); a smaller local model (cheap, but token-heavy masked
context confuses small models badly).

### HuggingFace Inference API + MiniLM-L6-v2 (`pipeline.py:20`)

**Why this.** Deployment constraint, stated plainly in the docstring at `pipeline.py:21`: "no torch,
no local models." The free tier can't hold PyTorch in memory. MiniLM-L6 is 384-dimensional and a
strong quality-per-dimension pick.

**Pros.** Almost no memory footprint; no model download at boot, so fast cold start — which matters
for Render's health check; MiniLM is well-benchmarked.

**Cons.** This is the single worst architectural compromise in the project. Embeddings are on the
network critical path for *every ingest chunk and every query*. `_embed_via_client` loops one HTTP
request per text (`pipeline.py:44-53`) — a 2,000-chunk document is 2,000 sequential round trips. The
free tier rate-limits and cold-starts. And the whole point of the project is data locality, yet
**the masked chunks go to HuggingFace** — defensible because they're masked, but it's an egress
point I should name rather than hide.

**Alternatives considered.** Local sentence-transformers (no network, batched, ~50x faster, fully
self-contained — needs ~500MB RAM, and this is exactly what I'd switch to on a paid tier); OpenAI
`text-embedding-3-small` (better quality, batched API, one vendor fewer — costs money and adds
another egress point); a self-hosted TEI container (best of both — needs real infra).

### LangChain (`pipeline.py:98-102`)

**Why this.** I used a narrow slice deliberately: the text splitter, the Chroma wrapper, the
`Embeddings` interface, and LCEL for `prompt | llm | parser` (`pipeline.py:292`).

**Pros.** The `Embeddings` ABC is what made my custom adapter drop in cleanly — that's real value.
`RecursiveCharacterTextSplitter` is a solid, well-tested implementation I'd otherwise write badly.
Swapping vector stores or LLM providers is a one-line change.

**Cons.** A heavy dependency tree for four classes, and it's the main source of version churn — the
git history shows a `langchain-chroma` bump specifically to support a `chromadb` release. I'm also
importing `Chroma` from the deprecated `langchain_community.vectorstores` (`pipeline.py:99`) while
having `langchain-chroma` installed, which will break on a future upgrade. And the abstraction hides
token counts and retries, which is annoying when debugging cost.

**Alternatives considered.** Direct SDK calls to `groq` and `chromadb` (fewer dependencies, full
visibility, and I'd only be reimplementing the splitter — genuinely tempting for a project this
focused); LlamaIndex (better retrieval abstractions, heavier and more opinionated).

### Vanilla JS + Socket.IO client (`static/js/app.js`, 654 lines)

**Why this.** No build step, no `node_modules`, served directly by Flask. For a 4-view dashboard
driven by server events, an IIFE with `getElementById` helpers (`static/js/app.js:15`, `:32`) is
adequate.

**Pros.** Zero toolchain and zero deploy complexity; instant load; the whole frontend is one
`<script>` tag; the animation layer (`motion.js`) and icons come from CDNs.

**Cons.** Manual DOM updates get fragile past this size. No component reuse — the vault table, audit
list and policy cards each hand-roll their own HTML strings (`fetchVault` at
`static/js/app.js:185`, `fetchAudit` at `:212`, `fetchPolicies` at `:256`), and building HTML from
server data via string concatenation is an XSS path, since PII values are rendered into the vault
table. CDN dependencies are an availability and supply-chain risk. No tests.

**Alternatives considered.** React (component reuse, safe-by-default escaping, real state management
for the event feed — but a build pipeline and a bundle for a dashboard this small); HTMX
(server-rendered fragments would suit the tab views nicely, and it's closer to my existing
architecture — a good middle ground I didn't consider at the time).

### Render + gunicorn/eventlet (`render.yaml`, `Procfile`)

**Why this.** Free tier, git-push deploys (`autoDeploy: true`), managed TLS, and
`healthCheckPath: /api/health` wired to the endpoint at `app.py:348`. Secrets via `sync: false` so
they never enter the repo; `SECRET_KEY` via `generateValue: true`.

**Pros.** Infrastructure-as-code in 24 lines; zero-cost hosting; the build command handles the spaCy
model download, so the container is self-contained.

**Cons.** `-w 1` means no parallelism at all — one slow ingest blocks every user. The ephemeral
filesystem means ChromaDB and the vault are lost on redeploy (they're gitignored, so there's no
fallback). Free instances cold-start, and my cold start includes building the Presidio engine. And
eventlet with blocking CPU work is the wrong concurrency model, as covered above.

**Alternatives considered.** Docker on a VPS (a persistent disk fixes the data loss, full control —
I'd own the ops); AWS ECS/Fargate + RDS + EFS (production-grade, properly solves persistence and
scaling — real cost and real complexity); Fly.io with a persistent volume (closest thing to a
drop-in fix for the persistence problem specifically).

---

## 4. If this scales — 10x and 100x

Baseline assumption, since the repo has no load testing: **[confirm with me]** — I'll say the
current system handles roughly a handful of concurrent users, with a 1M-character ingest taking
minutes. Everything below is reasoned from the code, not measured.

### Bottleneck 1 — Sequential embedding HTTP calls *(breaks first, at ~2x)*

**What breaks and why.** `_embed_via_client` at `pipeline.py:44-53` loops `for text in texts` and
makes one HTTP request per chunk. A 1M-char document is ~2,500 chunks after 500/100 splitting, so
that's 2,500 sequential round trips. At even 100ms each that's four minutes of pure network wait,
and HuggingFace's free tier rate-limits long before that. This is the first thing to fall over, by a
wide margin.

**What I'd add.** Three layers, in order of effort:
1. **Batching.** The HF API accepts a list of inputs — my raw-requests fallback at `pipeline.py:63`
   already passes `{"inputs": texts}` correctly. Batching in groups of 32-64 cuts 2,500 requests to
   ~50. This is the single highest-leverage fix in the whole project and it's a ~10-line change.
2. **Swap to local embeddings.** `sentence-transformers` with MiniLM batches on CPU at thousands of
   chunks per second and removes the network entirely. Needs ~500MB RAM.
3. **An async job queue.** Move ingestion off the request path into Celery or RQ with Redis, so
   `POST /api/ingest` returns `202 Accepted` with a job ID and the existing WebSocket channel
   reports progress. The event names at `app.py:64-85` already model exactly this.

**Trade-off.** Batching is nearly free — the only cost is a partial-failure story, since one bad
text fails the batch and I'd need per-item retry. Local embeddings trade memory for latency and mean
I can't use the free tier. The queue is the big one: it adds Redis and a worker process as new
failure domains, turns ingestion asynchronous (so the UI must handle "pending" states and I need
idempotency for retried jobs — which the stable IDs at `pipeline.py:200` conveniently already give
me), and I now have to monitor queue depth.

### Bottleneck 2 — Presidio/spaCy CPU on the request thread *(at ~5x)*

**What breaks and why.** `self.analyzer.analyze()` at `pii_redactor.py:124` is synchronous
single-threaded CPU work over the entire document in one call. Under
`--worker-class eventlet -w 1` this is catastrophic: eventlet only context-switches on patched I/O,
and NER is pure CPU, so one ingest freezes the event loop and every other user's request —
including WebSocket heartbeats and the health check. Render will eventually mark the instance
unhealthy. There's also a hard limit: spaCy's default `nlp.max_length` is 1,000,000 characters,
exactly my `validate_input` ceiling (`utils.py:55`), so a document at the limit will raise.

**What I'd add.** Horizontal scaling plus parallelism: run N gunicorn workers behind Render's load
balancer, and inside each, offload the analyze call to a `ProcessPoolExecutor` so it doesn't block
the loop. Split documents and analyze pieces in parallel — noting I'd have to redact *before*
splitting or accept that entities can't span chunk boundaries, which my newline truncation at
`pii_redactor.py:139` suggests is already roughly true for this data shape.

**Trade-off.** Multiple workers immediately breaks three things that are currently free: the
process-local ChromaDB (`pipeline.py:195`) becomes inconsistent per worker, the SQLite vault gets
concurrent writers, and WebSocket broadcasts only reach clients on the emitting worker. So this
bottleneck can't be fixed in isolation — it forces the storage migrations in bottlenecks 3 and 5.
Process pools also multiply memory: each process needs its own spaCy model.

### Bottleneck 3 — SQLite vault under concurrent writers *(at ~10x)*

**What breaks and why.** `PIIVault.store()` holds a `threading.Lock` (`vault.py:73`) and opens a
fresh connection per call (`vault.py:65`). Two problems compound. First, the lock is *per-process* —
with multiple workers, two processes can concurrently mint tokens for the same value, and since
`_counters` is in-memory (`vault.py:30`) they'll diverge, producing two tokens for one person. That
silently breaks retrieval, because the query's token won't match the indexed token. Second, SQLite
serializes writers, so under real concurrency I get `database is locked` errors. And `store()` does
three INSERTs plus a SELECT per entity — on a document with 5,000 entities that's 20,000 statements.

**What I'd add.** Migrate the vault to Postgres. The counter becomes a `SEQUENCE`, or I drop
positional counters entirely and use a UUID, since nothing depends on the count being sequential.
`INSERT ... ON CONFLICT (real_value) DO NOTHING RETURNING token` collapses the read-then-write into
one atomic statement and eliminates the race outright. Batch all entities from one document in a
single transaction. Add `pgcrypto` column encryption on `real_value`, which fixes the
plaintext-at-rest problem at the same time.

**Trade-off.** Postgres is now a required external dependency, a SPOF until replicated, and an ops
cost. Every vault call becomes a network round trip instead of a local file read — slower per call,
which is why the batching matters. Column encryption adds CPU per row and, more importantly, makes
`real_value` non-indexable for exact lookup unless I keep a deterministic HMAC of the value as the
lookup key alongside the encrypted payload. That's the standard pattern, but it's real design work.

### Bottleneck 4 — `unmask_text` is O(vault_size × answer_length) *(at ~10x)*

**What breaks and why.** `vault.py:116-124` loads **every row** in `token_mappings` into a dict, then
runs `text.replace()` once per token over the whole answer. With 50,000 vault entries and a 500-char
answer, that's 50,000 full string scans to substitute maybe three tokens. It gets worse linearly as
the vault grows, forever — and it holds the vault lock for the entire load.

**What I'd add.** Invert it: regex-extract the tokens actually present in the text
(`<<[A-Z_]+_\d+_[0-9a-f]{4}>>` matches my format exactly), then do a single indexed
`WHERE token IN (...)` lookup for just those. That's O(answer_length) plus one query. Add a small
LRU or Redis cache in front for hot tokens.

**Trade-off.** Almost none — this is a straight win and I'd call it out as an obvious fix I'd already
identified. The only nuance is that the current implementation happens to be robust to token-format
changes while a regex isn't, so the regex and the minting format at `vault.py:87` become coupled and
need a shared constant.

### Bottleneck 5 — Single-node ChromaDB *(at ~10x, and immediately on redeploy)*

**What breaks and why.** Chroma here is an embedded library writing to a local directory
(`pipeline.py:194`). With multiple workers each process has its own index, so a document ingested by
worker 1 is invisible to a query on worker 2. On Render's ephemeral filesystem the whole index
vanishes on every deploy. And `clear_vector_db` at `pipeline.py:334` calls `self.vector_db.get()`
with no filter, pulling every ID in the collection into memory.

**What I'd add.** Move to a shared vector store: either Chroma in client/server mode, or pgvector in
the same Postgres instance as the vault. pgvector is my preference here specifically because it gives
transactional consistency between a token and the chunk that references it — right now a crash
between the vault write and the Chroma write leaves an orphan.

**Trade-off.** pgvector's ANN performance is below dedicated vector databases at high scale, and I'd
be putting vector search load on the same instance serving vault lookups. It also caps how far I can
scale before needing read replicas or a real vector DB. The upside — one system to back up, one to
secure, one transaction boundary — is worth it at this scale.

### Bottleneck 6 — No caching anywhere *(at ~50x)*

**What breaks and why.** Every query pays full price: one embedding round trip, one ANN search, one
LLM inference. Nothing is memoized, even though `temperature: 0` makes the LLM deterministic for
identical input, and dashboard traffic re-fetches `/api/vault`, `/api/audit` and `/api/policies` on
every tab switch (`static/js/app.js:170-172`) with no `Cache-Control` headers.

**What I'd add.** Layered caching: a Redis-backed query-embedding cache keyed by the masked query
hash, since repeated questions are common; a response cache keyed by
`hash(masked_question + retrieved_doc_ids)`; HTTP caching headers on `/api/policies`, which is static
config and should never hit the server twice.

**Trade-off.** This is where caching gets genuinely dangerous in *this* system. A response cache
holds masked LLM output — which is safe — but the cache key must include tenant identity or one
user's cached answer surfaces for another. Since there's no tenancy today (section 5), I'd have to
add authentication *before* caching, not after. Caching is also the classic source of stale answers
after an ingest, so I'd need invalidation on write.

### Bottleneck 7 — No rate limiting or request bounds *(a liveness risk at any scale)*

**What breaks and why.** No `MAX_CONTENT_LENGTH`, no per-IP limits, and `POST /api/ingest` accepts
up to 1M chars (`utils.py:55`) of the most expensive work in the system. A single adversarial client
can trivially exhaust my one worker. The upload endpoint writes to disk before validating anything
(`app.py:155`). And SocketIO accepts connections from any origin (`cors_allowed_origins="*"`,
`app.py:24`).

**What I'd add.** `Flask-Limiter` with Redis storage for per-IP and per-key limits, weighted so
`/api/ingest` costs far more budget than `/api/health`. `MAX_CONTENT_LENGTH` on the Flask app.
Tighten CORS to the known origin. Put ingestion behind the job queue from bottleneck 1 so bursts
queue instead of blocking.

**Trade-off.** Rate limiting needs shared state (Redis) to work across workers — in-memory limits are
per-worker and trivially bypassed. Limits also create false positives for legitimate bulk users, so
I'd need a per-API-key tier. And rate limiting is a mitigation, not a fix: the real answer is making
ingestion cheap and asynchronous.

### Bottleneck 8 — Audit log as a local file *(at ~10x)*

**What breaks and why.** `audit_logger.py:26` writes to a local file via `FileHandler`, and
`get_recent_logs` does `f.readlines()` (line 70), loading the entire file into memory to show 50
lines. Multiple workers mean multiple interleaved files. Ephemeral disk means the compliance trail —
the whole point — is lost on redeploy. There's no rotation, so the file grows unbounded.

**What I'd add.** Ship structured logs to a real sink: stdout in JSON (which Render already captures)
forwarded to CloudWatch, Loki or an ELK stack, with the dashboard querying that API instead of
reading a file. The JSON Lines format at `audit_logger.py:40` is already exactly right for this,
which is the payoff of that decision.

**Trade-off.** Audit availability now depends on an external service, and there's a window where a
log is buffered but not yet shipped — for a compliance trail that window matters, so I'd want
synchronous writes for redaction events specifically, even if it costs latency. Log storage also
becomes a recurring cost, and retention becomes a policy decision rather than "whatever fits on
disk."

---

## 5. Weak spots an interviewer will poke at

Ordered roughly by how badly they'd hurt in a real deployment. These are the answers I'd actually
give — I'd rather name these first than get caught by them.

### 5.1 `/api/vault` dumps all plaintext PII with no authentication

**The poke.** "So you built a privacy system with an unauthenticated endpoint that returns every
name, Aadhaar number and email in your database?"

**Honest answer.** "Yes, and that's the most serious flaw in the project. `/api/vault` at
`app.py:294` calls `get_all_mappings()` and returns plaintext with no auth check — there's no
authentication anywhere in the application. I built it as a demo dashboard to *show* the vault
contents, which is a legitimate demo need, but I shipped it without gating it. In production that
endpoint needs to either not exist or sit behind authentication plus role-based authorization plus
its own audit event for every access — reading the vault is exactly the action you most want logged,
and right now it's the one operation that *isn't* audited. The irony isn't lost on me: I built a
careful trust boundary against the LLM and left the front door open."

### 5.2 Single global vault with no tenant isolation

**The poke.** "Two users, A and B. A ingests their data, B asks a question. What stops B seeing A's
PII?"

**Honest answer.** "Nothing, and that's a design limitation I should have named in the README.
There's one vault, one Chroma collection, and `unmask_text` at `vault.py:116` will substitute *any*
token it finds in the LLM's output regardless of who asked. So if B's retrieval pulls a chunk from
A's document, B gets A's plaintext. There's actually a second leak on top of that: SocketIO emits are
broadcast to all connected clients with no room scoping (`app.py:64`), so B's dashboard sees A's
redaction events live. The fix is tenancy threaded all the way through — a `tenant_id` column on the
vault tables, a metadata filter on every Chroma query, per-tenant Socket.IO rooms, and `unmask_text`
scoped to the requesting tenant's tokens only. It's not a patch, it's a schema change, which is why
I'd want to do it before adding any caching."

### 5.3 The vault stores PII in plaintext, and the README overstates the crypto

**The poke.** "Your README calls these 'cryptographic, irreversible tokens.' Are they?"

**Honest answer.** "The tokens are opaque, but 'cryptographic' is the wrong word and I'd correct the
README. The token is `entity_type + counter + 4 hex chars of a UUID4` (`vault.py:87`) — unguessable,
but not derived from the value cryptographically. And 'irreversible' is only true of the token in
isolation; `token_mappings` is the reversal, sitting in an unencrypted SQLite file. So the honest
framing is: I've *relocated* the sensitive data from the LLM provider to a single local store, which
is a real security improvement, but that store is now a high-value target with no encryption at rest
and no key management. The production version needs envelope encryption — column encryption via
`pgcrypto` or KMS-managed data keys — plus a deterministic HMAC as the lookup key so exact-match
search still works on encrypted values."

### 5.4 Two config files, one of which is dead

**The poke.** "Which model is actually running?"

**Honest answer.** "`llama-3.3-70b-versatile`, and I had to trace the code to be sure, which is
itself the bug. `config.py:6` resolves `CONFIG_PATH` relative to the module file, so the only config
ever read is `src/zero_trust_rag/config.yaml`. The `config.yaml` at the repo root is dead and has
drifted — it says `8b-instant`, `top_k: 3`, `score_threshold: 0.3` against live values of
`70b-versatile`, `20`, and `0.6`. That's a genuinely dangerous class of bug for a security config:
someone tightens the PII detection threshold in the wrong file, sees no error, and believes they've
hardened the system. The fix is to delete the root file and have `Config` fail loudly if the expected
file is missing, rather than silently falling back to `{}` at `config.py:24` — which right now means
a missing config gives you every default with no warning."

### 5.5 Zero tests

**The poke.** "How do you know the redaction works?"

**Honest answer.** "I don't, rigorously — there are no tests in the repo, and for a security
component that's the gap I'd close first. I verified behaviour by hand through `demo.py` and the
dashboard, which catches obvious breakage but proves nothing about edge cases. What I'd want is a
layered suite: unit tests on `_remove_overlaps` with hand-built overlapping spans; property-based
tests asserting the invariant that *no substring of the original PII appears in the masked output* —
that's the actual security property and it's testable with Hypothesis; a corpus of known-tricky
inputs (PII at chunk boundaries, unicode homoglyphs, possessives, PII spanning newlines); and a
round-trip test that `unmask(redact(x)) == x`. The architecture is already set up for it:
`PIIRedactor` takes its vault and logger by injection (`pii_redactor.py:92`), so I can test it
against an in-memory vault. I built the seams and then didn't use them."

### 5.6 Overlap resolution doesn't do what its docstring claims

**The poke.** "Walk me through `_remove_overlaps`. Does it keep the highest-scoring span?"

**Honest answer.** "No, and the docstring at `pii_redactor.py:192` is wrong. It sorts by
`(start, -score)` and then greedily accepts any span starting at or after `last_end`. The score only
breaks ties *between spans with identical start offsets*. So if a low-confidence `DATE_TIME` match
starts one character before a high-confidence Aadhaar match and overlaps it, the Aadhaar loses — a
0.85 detection dropped in favour of a 0.3. That's a real detection gap, not just a comment bug, and
it's the kind of thing a test would have caught. The correct algorithm is weighted interval
scheduling — sort by end offset and do the O(n log n) DP to maximize total confidence. Same
complexity class, maybe fifteen more lines."

### 5.7 The retry decorator doesn't retry the thing it's on

**The poke.** "You decorated `_add_texts_to_vector_db` with a retry. What does it catch?"

**Honest answer.** "Only `APIError` (`utils.py:30`) — and the embedding path raises a bare
`RuntimeError` (`pipeline.py:79`), so embedding failures aren't retried at all. The decorator on that
method is effectively dead code, which is worse than having no retry because it looks like resilience
I don't have. Two fixes: wrap the embedding failures in `APIError` so they're caught, and switch to
`tenacity` — which is already in `requirements.txt` and which I hand-rolled around for no good
reason — to get exponential backoff with jitter instead of a fixed 1-second delay that would create
a thundering herd against an upstream that's already struggling."

### 5.8 `get_rag()` claims thread safety it doesn't have

**The poke.** "That docstring says 'lazy init with thread safety.'"

**Honest answer.** "It's wrong. `app.py:31-40` is a check-then-set on a module global with no lock,
so two concurrent first requests can both pass the `if rag is None` check and both construct a full
pipeline — two spaCy models, two Presidio engines, on a free tier with a tight memory budget. In
practice `-w 1` with eventlet makes the window narrow, but 'narrow' isn't 'safe.' The fix is a
`threading.Lock` with double-checked locking, or better, initialize eagerly in a gunicorn `post_fork`
hook so the cost is paid at boot instead of on a user's request."

### 5.9 Redaction runs twice on every request

**The poke.** "Trace a call to `/api/ingest`. How many times does Presidio run?"

**Honest answer.** "Twice, and it's wasteful. `app.py:70` calls `bot.ingest(data)`, which redacts
internally at `pipeline.py:232`, and then `app.py:89` redacts the *same text again* just to get the
masked version for the UI response. Same pattern on the ask path at `app.py:236` and `app.py:239`.
Since Presidio is the single most expensive CPU step, I'm doubling the cost of my slowest operation
to populate a display field. It's correct only by accident — the vault's deterministic lookup
(`vault.py:75`) means the second pass produces identical tokens rather than minting new ones. But it
also writes a second round of audit events under the event type `display`, which pollutes the
compliance log. The fix is trivial: `ingest()` already returns `redaction_details`, so it should just
return the masked text too."

### 5.10 `/api/policies` displays a null it should never display

**The poke.** "Your policies dashboard shows the PII confidence threshold. Is it correct?"

**Honest answer.** "No — `app.py:337` reads `security.score_threshold`, but the key in the config is
`presidio.score_threshold`. `Config.get` returns the default on a miss (`config.py:33`), which here
is `None`, so the dashboard renders an empty value for the single most security-relevant setting in
the system. It's a two-character fix, but the lesson is the bigger issue: string-keyed config lookups
fail silently. Pydantic Settings — and Pydantic is already a dependency — would make this a startup
error instead of a wrong number on a compliance screen."

### 5.11 Masking `DATE_TIME`, `LOCATION` and `URL` degrades retrieval badly

**The poke.** "You mask dates and locations. Can the model still answer 'who joined before 2023?'"

**Honest answer.** "No, and that's an accuracy cost I accepted too broadly. My active config masks
`DATE_TIME`, `LOCATION`, `URL` and `NRP` alongside the genuinely sensitive identifiers
(`src/zero_trust_rag/config.yaml:24-40`), so 'joined March 2023 in Mumbai' becomes two opaque tokens.
Date arithmetic, geographic grouping and any temporal filtering are simply impossible — the model has
no signal to reason over. It also hurts embedding quality: masked text is semantically thinner than
plaintext, so similarity search degrades across the board. The better design is tiered redaction —
full tokenization for direct identifiers like Aadhaar and email, but *generalization* for
quasi-identifiers: bucket dates to the year, locations to the city or region. You keep most of the
analytic value and still break direct re-identification. That's a k-anonymity-style trade-off rather
than an all-or-nothing one, and it's the change I'd be most excited to make."

### 5.12 A 10-digit phone pattern that can never fire

**The poke.** "Your Indian phone recognizer has three patterns. Do they all work?"

**Honest answer.** "One doesn't. The bare-10-digit pattern is scored 0.5 (`pii_redactor.py:64`) and
my active `score_threshold` is 0.6, so Presidio filters it out before I ever see it. So a plain
`9876543210` with no separators and no surrounding context isn't caught as a phone number by that
pattern. It might still get picked up by a built-in recognizer, but I shouldn't be relying on luck.
This is exactly the trade-off that threshold represents — I set it at 0.6 to cut false positives,
and I gave up a recall case without noticing. The right fix is either to raise that pattern's score
and lean on Presidio's context enhancer to boost it when words like 'phone' or 'mobile' appear
nearby, or to accept the false positives — because in a PII system a false positive costs you a
slightly degraded answer and a false negative costs you a data breach. I'd bias hard toward recall."

### 5.13 Nothing survives a redeploy

**The poke.** "You push a commit. What happens to your data?"

**Honest answer.** "It's gone. `chroma_store/`, `vault.db` and `audit.log` are all gitignored and
Render's free tier has an ephemeral filesystem, so every deploy wipes the vector index, the token
mappings and the compliance audit trail. Losing the index is annoying; losing the vault is worse,
because any masked text that survived elsewhere becomes permanently unreadable; losing the audit log
defeats the entire compliance story. Gitignoring them is correct — I absolutely don't want plaintext
PII in git — but I never replaced them with real persistence. The fix is a managed Postgres for the
vault, a persistent volume or hosted vector store for the index, and shipping audit logs off-box."

### 5.14 O(n²) string building on large documents

**The poke.** "You say you handle a million characters. What's the cost of the replacement loop?"

**Honest answer.** "Quadratic, and I'd measure it before claiming that number again.
`pii_redactor.py:173` builds a brand new string for every single entity —
`safe_text[:start] + token + safe_text[end:]` copies the entire document per substitution. On a
1M-char document with a few thousand entities that's gigabytes of copying. The reverse-ordered
iteration at line 165 is the right *idea* — it keeps offsets valid — but the implementation should
walk the spans and append segments to a list, then `''.join()` once. Linear instead of quadratic,
and about the same number of lines. The mean-pooling at `pipeline.py:36-40` has the same flavour of
problem: a pure-Python nested comprehension where numpy is already a dependency."

### 5.15 Nothing validates that redaction actually succeeded

**The poke.** "How do you know a document didn't leak? Is there a check before egress?"

**Honest answer.** "There isn't, and for a zero-trust system that's the philosophical hole. The
design assumes Presidio's recall, and ML-based NER has no recall guarantee — a misspelled name or an
unusual format walks straight through to Groq. `validate_input(masked_text)` at `pipeline.py:234`
only checks length, not safety. What I'd add is a verification pass before egress: after redaction,
scan the masked text against the vault's known plaintext values and hard fail if any appear — a cheap
Aho-Corasick sweep. That catches the specific case where a value was tokenized in document A but
missed in document B, which is the most likely leak path in a deterministic-token system. It doesn't
catch never-before-seen PII, but no automated system does — which is why the honest claim for this
project is 'high-recall automated redaction with an audit trail,' not 'guaranteed zero leakage.' I'd
rather say that precisely than oversell it."

---

## 6. 15 likely questions with model answers

### Q1. Why tokenize instead of just masking with `[REDACTED]`?

"Because `[REDACTED]` destroys the thing RAG needs — identity. If a chunk says 'Employee [REDACTED]
reports to [REDACTED]' the model can't tell whether those are two people or one, and it can't connect
that chunk to a question about a specific person. My tokens preserve identity while dropping value:
`<<PERSON_01_3f9c>>` is consistently the same human everywhere in the corpus, because `vault.py:75`
looks up the existing token before minting a new one. That determinism is what lets vector similarity
match a query token against an indexed token. I also encode the entity *type* into the token, so the
model still knows it's looking at an email versus a phone number and can answer 'what's their contact
info?' sensibly. The cost I accepted is that determinism leaks structure — someone with read access
to my vector store can do frequency and co-occurrence analysis on the tokens. That's a real privacy
trade I made knowingly, because the alternative is a system that doesn't work."

### Q2. What happens if Presidio misses some PII?

"It leaks to Groq, and I want to be direct about that rather than hand-wave it. My mitigations are
all probabilistic: I set the confidence threshold at 0.6 to balance precision and recall, I added
three custom recognizers for Indian identifiers that the built-ins handle poorly
(`pii_redactor.py:42-83`), and I strip control characters first (`pii_redactor.py:113`) so zero-width
characters can't be used to split a number past my regexes. But ML-based NER has no recall guarantee,
full stop.

What's missing — and this is the fix I'd prioritize — is a verification pass before egress. After
redaction, I'd sweep the masked text for any plaintext value already in the vault and hard fail if
one appears. That catches the highest-probability real leak: a name tokenized in one document and
missed in another. For genuinely novel PII, the honest answer is defense in depth — I'd add
recall-biased thresholds, because a false positive costs me a slightly worse answer while a false
negative costs me a breach, and those are not symmetric."

### Q3. Why is the vault SQLite and not just a Python dict?

"Durability and atomicity. A dict dies with the process, and every token in my already-persisted
ChromaDB index would become permanently unresolvable — the vector store would be full of tokens
pointing at nothing. SQLite gives me ACID transactions for free, and I leaned on the schema to
enforce invariants rather than doing it in application code: `token` is the primary key of
`token_mappings` and `real_value` is the primary key of `value_mappings` (`vault.py:37`, `:45`), so
the one-to-one mapping is a database constraint, not a hope.

I'd also be upfront that it's the wrong choice past one process. SQLite serializes writers, and my
`threading.Lock` at `vault.py:28` is per-process, so with multiple gunicorn workers two processes can
mint different tokens for the same person — which silently breaks retrieval. So SQLite is right for a
single-node demo and a blocker for horizontal scaling, and I'd migrate to Postgres with
`INSERT ... ON CONFLICT ... RETURNING` to make minting atomic."

### Q4. How does the LLM answer correctly if it only sees tokens?

"Because the task I'm giving it is extraction and reasoning over provided context, not recall of
world knowledge. My prompt at `pipeline.py:281-289` explicitly tells the model the context uses
placeholders to protect privacy and instructs it to answer using them — so when the context says
`<<PERSON_01_3f9c>>` has Aadhaar `<<IN_AADHAAR_01_b8e2>>` and the question asks about
`<<PERSON_01_3f9c>>`, the model just needs to match tokens and copy one across. That's well within a
70B model's capability, and `temperature: 0` keeps it deterministic.

Where it genuinely fails is anything needing the *semantics* of a masked value. I mask `DATE_TIME`
and `LOCATION`, so 'who joined before 2023' or 'who's in the Mumbai office' is unanswerable — the
model has no signal at all. That's the honest cost, and the better design is tiered redaction:
tokenize direct identifiers, but generalize quasi-identifiers to a year or a city instead of blanking
them. You keep the analytic value and still break direct re-identification."

### Q5. Why WebSockets and not just polling?

"Ingesting a 1M-character document means a multi-minute Presidio pass plus thousands of embedding
calls. With plain request/response the browser sits there looking broken, and the user can't tell a
slow document from a crashed server. So I push phase-level events — `ingestion_start`, `pii_detected`
with per-entity detail, `ingestion_complete` (`app.py:64-85`) — and the client renders them into a
live terminal feed (`static/js/app.js:66-100`). The user sees *which* stage is running.

I'd also concede this is mild over-engineering. All my traffic is server→client; I never use the
bidirectional channel. Server-Sent Events would fit better — plain HTTP, automatic reconnect, no
separate protocol. And WebSockets cost me real architectural flexibility: they need sticky sessions
or a shared message queue with multiple workers, which is part of why `render.yaml` is pinned to
`-w 1`. There's also a bug I'd fix first — the emits are global broadcasts with no room scoping, so
every connected dashboard sees every user's redaction events."

### Q6. Walk me through the exact data flow of one query.

"Request hits `POST /api/ask` at `app.py:213`. I pull the question, grab the lazily-initialized
pipeline via `get_rag()`, and emit `query_processing` so the UI lights up.

Inside `ask()` at `pipeline.py:258`: `validate_input` bounds it, then `redactor.redact()` runs the
full detection path — control-char sanitization, Presidio analyze with spaCy plus my custom
recognizers, newline truncation and possessive trimming on the spans, greedy overlap removal, then
right-to-left substitution using vault-minted tokens. Critically, the vault returns *existing* tokens
for values it's seen, so 'John Doe' in the question becomes the same token sitting in my index.

Then `similarity_search(masked_q, k=20)` at `pipeline.py:274` — that embeds the masked query via an
HTTP call to HuggingFace and does an HNSW lookup in Chroma. I join the chunks into a context string,
and if it's empty I bail early with a friendly message rather than letting the model hallucinate.
Context and masked question go into the `ChatPromptTemplate`, through the LCEL chain
`prompt | llm | StrOutputParser()` at line 292, out to Groq.

The masked answer comes back and `redactor.unmask()` → `vault.unmask_text()` (`vault.py:116`)
substitutes plaintext — that's the only place in the whole flow where PII reappears, and it's inside
my process. I log a success event with entity counts, and `app.py` returns the answer plus the masked
question so the UI can show the transformation. Worth noting: `app.py:236` redacts the question a
second time purely for that display field, which is wasteful — `ask()` should return it."

### Q7. Your embedding call is one HTTP request per chunk. Why?

"Because I wrote the straightforward loop and it worked at demo scale, and it's the first thing that
would fall over in production. `_embed_via_client` at `pipeline.py:44` iterates `for text in texts`
calling `feature_extraction` once each. A 1M-char document is about 2,500 chunks, so 2,500 sequential
round trips — minutes of pure network wait, and HuggingFace's free tier rate-limits well before that.

What makes it slightly embarrassing is that my own fallback path already does it right: the raw HTTP
call at `pipeline.py:63` passes `{"inputs": texts}` as a batch. So the fix is batching the client path
in groups of 32-64, which cuts 2,500 requests to about 50 — roughly ten lines for the single biggest
performance win available in this project. The only new complexity is partial failure: one bad input
fails the batch, so I'd need per-item retry on a batch failure. Longer term I'd move to local
sentence-transformers and take the network off the critical path entirely — the only reason I didn't
is that it pulls in PyTorch, which doesn't fit the free tier's memory, and that constraint is
documented right in the class docstring at `pipeline.py:21`."

### Q8. You have `threading.Lock` but you deploy with eventlet. Is that right?

"It's not, and it's a good catch. Under `--worker-class eventlet` (`render.yaml`) I'm on cooperative
green threads, and eventlet only yields on monkeypatched I/O. Two consequences.

First, `threading.Lock` gets monkeypatched into an eventlet lock, so it technically works for mutual
exclusion — but it's protecting blocking `sqlite3` calls (`vault.py:65`), and sqlite3 isn't
patchable, so every vault write blocks the entire event loop. Second and much worse, Presidio's
analyze at `pii_redactor.py:124` is pure CPU with no I/O to yield on, so one ingest freezes every
other request on that worker — including WebSocket heartbeats and the health check Render uses to
decide whether to kill my instance.

So the concurrency model is genuinely mismatched to the workload. The right shape is a sync worker
class with a real thread or process pool, or better, move the CPU-heavy ingestion off the request
path entirely into a Celery worker and let the web process do only I/O. That also fixes the `-w 1`
constraint, since I'd only need eventlet for the WebSocket layer, where it actually fits."

### Q9. Two users ingest data. Can user B see user A's PII?

"Yes, and it's the most serious design limitation in the project. There's one vault, one Chroma
collection, and no notion of a user anywhere in the codebase. Three distinct leak paths: B's
retrieval can pull chunks from A's documents; `unmask_text` at `vault.py:116` substitutes *any* token
it finds regardless of who owns it; and `/api/vault` at `app.py:294` returns every mapping in
plaintext to anyone who asks. There's a fourth, subtler one — SocketIO emits are unscoped broadcasts
(`app.py:64`), so B's dashboard shows A's redaction events in real time.

The fix isn't a patch, it's tenancy threaded through every layer: a `tenant_id` column on the vault
tables with it in the lookup key, a metadata filter on every Chroma query, per-tenant Socket.IO
rooms, `unmask_text` scoped to the caller's tokens, and authentication in front of all of it. I'd
sequence it before any caching work, because a cache keyed without tenant identity would turn this
from a design gap into an active breach."

### Q10. Why not just run the LLM locally and skip all this?

"That's the honest alternative and I'd say so directly — a locally-hosted Llama with no egress makes
the entire redaction layer optional. The reason I didn't is that it changes the project from a
privacy-architecture problem into a GPU-operations problem: you need the hardware, the serving stack,
the scaling story, and for a portfolio project deployed on a free tier it's simply not available.

But I'd also argue the redaction layer still has independent value. First, it's defense in depth —
even with a local model, redaction means a compromised log, a prompt-injection exfiltration, or a
future decision to call an external API doesn't immediately become a PII breach. Second, the
architecture is *portable*: because I mask at the trust boundary rather than trusting the model, I
can swap Groq for a local Ollama, or for OpenAI, without touching the privacy guarantee. That's
exactly the property you want when you can't predict which model you'll be on in a year. And third,
most organizations genuinely want frontier models they can't self-host, so 'use a local model' isn't
available to them either — which is the actual market for this design."

### Q11. How do you handle GDPR's right to erasure?

"Badly right now, and I'd rather walk through why it's interesting than pretend it's solved.

The good news is that tokenization gives me a genuinely elegant story in principle: delete one row
from `token_mappings` and every occurrence of that person across every chunk in the vector store
becomes permanently unresolvable. That's *crypto-shredding* — you destroy the key rather than hunting
down the data. One delete, global effect, no reindexing.

The bad news is what remains. The token still sits in the vector index, so the structural fact that
*someone* appears in those documents persists, along with all the co-occurrence relationships — and
under GDPR that's arguably still personal data, since it's linkable. My content-hash IDs at
`pipeline.py:200` make it worse: there's no `document_id` metadata, so I can't even find which chunks
belong to a given subject to delete them. And the audit log (`audit_logger.py:30`) deliberately
records token-level events forever, which is a genuine tension — compliance wants an immutable trail,
GDPR wants erasure.

So the real answer is: I'd add document-level metadata for targeted deletion, treat vault-row deletion
as the primary mechanism, and get a legal read on whether orphaned tokens count as anonymized. That
last part isn't an engineering decision."

### Q12. What's your biggest single point of failure?

"The vault, and it's a single point of failure in two different senses, which is what makes it the
interesting answer.

Availability: it's one SQLite file (`vault.db`) on an ephemeral filesystem with no replication and no
backup. Lose it and every masked chunk in ChromaDB becomes permanently undecodable — the vector store
survives and is simultaneously useless. And because Render's free tier wipes the disk, this isn't
hypothetical; it happens on every deploy.

Security: it's the one component that reverses all my privacy guarantees, stored in plaintext with no
encryption at rest and, via `/api/vault`, no access control.

So the same component is my most fragile *and* my most sensitive, which is a bad property — the usual
fixes pull against each other, since replicating it for availability multiplies the attack surface.
The production answer is managed Postgres with automated backups and point-in-time recovery for
availability, plus envelope encryption with KMS-managed keys and a deterministic HMAC as the lookup
index so exact-match search still works on encrypted rows. That's the standard pattern for exactly
this problem, and it's the work I'd scope first."

### Q13. Why `temperature: 0`?

"Because this is an extraction task, not a creative one. The model's job is to find a token in the
provided context and copy it into an answer — there's no value in sampling diversity, and there's
real cost: a model that paraphrases might *mangle* a token, and a mangled token doesn't match
anything in `unmask_text`, so it survives into the user's answer as literal gibberish like
`<<PERSON_1_3f9c>>`. Greedy decoding minimizes that.

It also makes the system testable and cacheable. Deterministic output means I can write regression
tests on answer content, and it means a response cache keyed on
`hash(masked_question + retrieved_doc_ids)` is actually sound — which is the caching layer I'd add at
scale.

What I *should* add alongside it is validation: right now nothing checks that the tokens in the
model's output are well-formed or exist in the vault. A malformed token just passes through silently.
A regex validation pass before unmasking, logging any unresolvable token as an anomaly, would catch
model drift and would also catch the more interesting case — a model hallucinating a token that
happens to collide with a real one, which would substitute the wrong person's data."

### Q14. If I gave you one week to make this production-ready, what would you do?

"I'd sequence it by blast radius, not by effort.

Day one, security, because these are breaches rather than bugs: authentication in front of every
endpoint, delete or gate `/api/vault`, tighten the SocketIO CORS off `*`, and set
`MAX_CONTENT_LENGTH`. Day two, tenancy — `tenant_id` through the vault schema, Chroma metadata
filters, scoped `unmask_text`, per-tenant Socket.IO rooms. It has to come before caching, or the
cache becomes a leak.

Day three, persistence: migrate the vault to Postgres with `ON CONFLICT ... RETURNING` to kill the
token-minting race, move vectors to pgvector for transactional consistency with the vault, and ship
audit logs to a real sink instead of a local file that dies on redeploy.

Day four, tests — and I'd start with property-based tests on the actual security invariant, that no
substring of any known PII value appears in masked output. Plus unit tests on `_remove_overlaps`,
which I know is wrong, and round-trip tests on redact/unmask.

Day five, performance: batch the embedding calls, which is the ten-line change that buys the most;
fix the O(n²) string building in the replacement loop; invert `unmask_text` to query only the tokens
actually present; and move ingestion into a job queue so CPU work is off the request path.

Day six, correctness cleanup: delete the dead root `config.yaml`, fix the `security` vs `presidio`
key bug in `/api/policies`, replace hand-rolled config access with Pydantic Settings so misses fail
at startup, and add the pre-egress verification sweep.

What I'd explicitly *not* do in that week is rewrite the frontend or move to FastAPI. Both are real
improvements and neither changes whether the system leaks data."

### Q15. What's the one thing you'd change about how you built this?

"I'd have written the tests first, and specifically the property-based ones.

Not for the usual reasons — for a reason specific to this project. The core value proposition is a
security property: 'PII does not cross the trust boundary.' That's an *invariant*, and invariants are
exactly what property-based testing is for. If I'd written 'for any input containing PII, no
substring of that PII appears in the masked output' as a Hypothesis property on day one, it would
have caught the overlap-resolution bug at `pii_redactor.py:196`, it would have caught the 10-digit
phone pattern scored below my own threshold, and it would have forced me to think about the
pre-egress verification sweep much earlier — because the property is the same thing.

Instead I verified by eye through `demo.py`, which proves the happy path and nothing else. And the
frustrating part is that the architecture was *already* built for testing — `PIIRedactor` takes its
vault and logger by dependency injection at `pii_redactor.py:92` specifically so you can swap them.
I built the seams and never used them. For a security component, 'I looked at the output and it
seemed masked' isn't a standard of evidence I'd defend, and that's the habit I'd change rather than
any specific technical decision."

---

## Quick-reference cheat sheet

| Thing | Value | Where |
|---|---|---|
| LLM | `llama-3.3-70b-versatile` via Groq, `temperature: 0` | `src/zero_trust_rag/config.yaml:3-6` |
| Embeddings | MiniLM-L6-v2 (384-dim) over HF Inference API | `pipeline.py:20` |
| Vector store | ChromaDB, HNSW, local persist | `pipeline.py:195` |
| Chunking | 500 chars / 100 overlap, recursive separators | `pipeline.py:185` |
| Retrieval | `top_k: 20` | `pipeline.py:273` |
| PII detection | Presidio + spaCy `en_core_web_sm`, threshold `0.6` | `pii_redactor.py:124` |
| Custom recognizers | Aadhaar (0.85), PAN (0.8), Indian phone (0.7/0.6/0.5) | `pii_redactor.py:42-83` |
| Token format | `<<{TYPE}_{NN}_{4 hex}>>` | `vault.py:87` |
| Vault | SQLite, 3 tables, `threading.Lock` | `vault.py:33-56` |
| Audit | JSON Lines to local file | `audit_logger.py:26` |
| Transport | Flask + Flask-SocketIO, 8 REST endpoints | `app.py` |
| Deploy | Render, `gunicorn --worker-class eventlet -w 1` | `render.yaml` |
| Input ceiling | 1,000,000 chars | `utils.py:55` |

**Numbers to have ready:** 1M+ character documents validated; ~2,500 chunks at that size;
384-dimension vectors; 16 entity types configured; 3 custom recognizers; 8 REST endpoints;
5 WebSocket event types; 654 lines of frontend JS; 0 tests.
