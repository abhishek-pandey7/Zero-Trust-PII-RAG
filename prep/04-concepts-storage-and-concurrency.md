# 04 — Storage, Concurrency & Idempotency, Explained

The data layer: ACID, transactions, locking, race conditions, idempotency, and content addressing.
This file covers the concepts a database-leaning interviewer will dig into, all grounded in
`vault.py` and `pipeline.py`.

---

## 1. ACID, and which parts I actually rely on

Four guarantees a transactional database gives you:

- **Atomicity** — all statements in a transaction commit, or none do. No half-writes.
- **Consistency** — constraints (primary keys, foreign keys, checks) always hold.
- **Isolation** — concurrent transactions don't observe each other's partial state.
- **Durability** — once committed, it survives a crash or power loss.

### Where each one matters in my vault

`PIIVault.store()` (`vault.py:72-104`) performs **three writes** for one logical operation:

```python
conn.execute("INSERT INTO token_mappings (token, entity_type, real_value) VALUES (?,?,?)", ...)
conn.execute("INSERT INTO value_mappings (real_value, entity_type, token) VALUES (?,?,?)", ...)
conn.execute("INSERT OR REPLACE INTO counters (entity_type, count) VALUES (?,?)", ...)
conn.commit()
```

- **Atomicity** is load-bearing. If the first insert succeeded and the second didn't, I'd have a token
  that resolves *forward* (unmask works) but not *backward* (the next occurrence of that value mints a
  **second** token). Determinism silently breaks, and retrieval degrades in a way that's very hard to
  debug. The single `commit()` after all three is what prevents that.
- **Durability** is the whole reason this isn't a Python dict. Tokens persisted in ChromaDB outlive the
  process; if the mapping doesn't, the vector store becomes a pile of unresolvable tokens.
- **Consistency** is enforced by schema, not code: `token` is `PRIMARY KEY` on `token_mappings`
  (`vault.py:37`) and `real_value` is `PRIMARY KEY` on `value_mappings` (`vault.py:45`). The one-to-one
  mapping is a **database constraint**, so a bug in my application logic gets an `IntegrityError`
  rather than silently corrupting the vault. That's the right place to put an invariant.
- **Isolation** is where SQLite's model becomes the limitation — see §3.

---

## 2. Why two tables that look redundant

`token_mappings` and `value_mappings` store the same three fields with the primary key swapped. That
looks like duplication. It isn't — it's a **deliberate bidirectional index**.

| Table | PK | Query it serves | Called from |
|---|---|---|---|
| `value_mappings` | `real_value` | "does this value already have a token?" | `vault.py:76` during redaction |
| `token_mappings` | `token` | "what value does this token mean?" | `vault.py:110`, `:119` during unmasking |

Both directions are on hot paths, and **both need to be O(log n)**. A primary key in SQLite is a
B-tree index, so each lookup is a tree descent rather than a table scan.

**With one table** you'd index one direction and scan the other. On a vault with 50,000 entries, a
full scan per entity during redaction — with thousands of entities per document — is disastrous.

**The trade-off:** 2× storage, and the two tables must be written together atomically (see §1). This is
the classic **denormalization for read performance** pattern, and the cost is exactly the one
denormalization always has: the write path now has to maintain consistency across copies.

**What I'd say if challenged:** in Postgres I wouldn't need two tables at all — one table with a
`UNIQUE` index on `real_value` alongside the `token` primary key gives both directions from a single
row. SQLite supports `CREATE UNIQUE INDEX` too, so honestly one table plus an index would have been
cleaner here. The two-table version works and isn't wrong, but it's more than was needed. Being able
to say that is better than defending it as optimal.

---

## 3. Locking, and the three concurrency bugs in the vault

```python
# vault.py:28
self._lock = threading.Lock()

# vault.py:72-74
def store(self, entity_type, real_value):
    with self._lock:
        with self._get_connection() as conn:
            ...
```

### Why a lock is needed: the check-then-act race

```python
cursor = conn.execute("SELECT token FROM value_mappings WHERE real_value = ?", ...)
row = cursor.fetchone()
if row:
    return row["token"]     # ← read
# ...
conn.execute("INSERT INTO value_mappings ...")   # ← write
```

Read, decide, write. Classic **race window**. Without the lock, two threads redacting the same name
simultaneously both SELECT (both find nothing), both increment the counter, both INSERT. One gets an
`IntegrityError` on the primary key — or worse, with different `short_id` values they'd produce two
different tokens for one person if the PK collision didn't stop them. The `threading.Lock` serializes
the whole read-modify-write, making it effectively atomic.

### Bug 1: the lock is per-process, not per-system

`threading.Lock` coordinates threads **inside one Python process**. It provides zero protection
between processes. So the instant you run `gunicorn -w 4`:

- Worker 1 and worker 3 both redact "Priya Sharma" concurrently.
- Each holds its own lock. Neither blocks the other.
- Both SELECT, find nothing (or interleave badly), both try to INSERT.
- Best case: one gets `IntegrityError` — which is **uncaught** in `store()` and propagates up as an
  unhandled exception, becoming a 500.

**The correct fix isn't a better lock — it's to make the operation atomic in the database:**

```sql
-- Postgres: one statement, no application-level locking needed
INSERT INTO value_mappings (real_value, entity_type, token)
VALUES ($1, $2, $3)
ON CONFLICT (real_value) DO NOTHING
RETURNING token;
-- then SELECT if nothing was returned (someone else won the race — use their token)
```

This is the general principle worth stating: **push concurrency control into the database rather than
coordinating in application code.** The database already has the machinery; your lock doesn't scale
past one process.

### Bug 2: in-memory counters diverge

```python
# vault.py:30, 59-61
self._counters: dict[str, int] = {}
cursor = conn.execute("SELECT entity_type, count FROM counters")
for entity_type, count in cursor:
    self._counters[entity_type] = count      # loaded once, at init

# vault.py:83-84
count = self._counters.get(entity_type, 0) + 1
self._counters[entity_type] = count          # incremented in memory
```

Counters are read from the DB **once at startup** and thereafter incremented in memory. With two
processes both starting at `PERSON: 40`, both mint `PERSON_41`, and each writes 41 back via
`INSERT OR REPLACE`. The counter is now wrong and both processes think they own 41.

The `short_id` (4 random hex) saves you from an actual primary-key collision most of the time — which
is arguably a lucky accident of the design rather than a plan.

**Fix:** drop the application-side counter entirely. Use a Postgres `SEQUENCE`, or just use a UUID.
Nothing in the system depends on the counter being sequential or gap-free — it's purely cosmetic, for
readable demo output. **A cosmetic feature is creating a correctness bug**, which is exactly the kind
of trade-off worth naming yourself.

### Bug 3: the lock is held across blocking I/O

```python
def unmask_text(self, text):
    with self._lock:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT token, real_value FROM token_mappings")   # ENTIRE TABLE
            mapping = {row["token"]: row["real_value"] for row in cursor}
```

`vault.py:116-120` holds the lock while loading **every row in the vault into a dict**. At 50,000
entries that's a meaningful pause during which no other thread can `store()` or `resolve()`.

Two separate problems compounding: (a) the query itself is wrong (§5), and (b) holding a lock across
I/O is a general anti-pattern — you want the critical section as short as possible, ideally never
containing a network or disk round trip.

### The eventlet interaction (ties to file 03)

Under `--worker-class eventlet`, eventlet monkeypatches `threading`, so `threading.Lock` becomes a
green-thread lock. Mutual exclusion still works. But:

- **`sqlite3` is a C extension and is not monkeypatchable.** Every query inside the lock blocks the
  *entire event loop* — not just this green thread.
- So the vault isn't merely serialized, it **stalls the whole process** on every write.

This is the concrete mechanism behind the "eventlet is the wrong concurrency model here" claim. Being
able to name *why* (unpatchable C extension) rather than just asserting it is the difference between
sounding rehearsed and sounding like you debugged it.

---

## 4. Connection management

```python
# vault.py:63-70
@contextmanager
def _get_connection(self):
    conn = sqlite3.connect(self._db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
```

**What's right:**
- **`@contextmanager`** with `try/finally` guarantees `close()` even on exception. No leaked file
  handles. This is the correct Python idiom for resource lifecycle.
- **`row_factory = sqlite3.Row`** gives dict-like access (`row["token"]`) instead of positional
  tuples (`row[0]`). Much less fragile — adding a column doesn't silently shift indices.
- **Fresh connection per call** sidesteps SQLite's default thread-affinity restriction
  (`check_same_thread`), which would otherwise raise if a connection crossed threads. Given the
  `-w 1` + eventlet deployment, this is a reasonable simplification.

**What's expensive:** opening a connection means a filesystem open, SQLite header read, and page-cache
setup. `store()` is called **once per detected entity**, so a document with 5,000 entities opens and
closes 5,000 connections and runs ~20,000 statements (1 SELECT + 3 INSERTs each).

**The fix, and the concept name:** **connection pooling** — keep N open connections and check them
out. Plus **batching**: one transaction for all entities in a document instead of one per entity.

```python
# sketch
def store_batch(self, entities: list[tuple[str, str]]) -> dict[str, str]:
    with self._lock, self._get_connection() as conn:
        # one BEGIN ... COMMIT for the whole document
```

That turns 5,000 transactions into 1. SQLite commits are `fsync`-bound, so this is typically a 10-100×
improvement on the write path — the biggest easy win in `vault.py`.

---

## 5. The `unmask_text` complexity problem

```python
# vault.py:116-124
def unmask_text(self, text: str) -> str:
    with self._lock:
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT token, real_value FROM token_mappings")
            mapping = {row["token"]: row["real_value"] for row in cursor}

    for token, real_value in mapping.items():
        text = text.replace(token, real_value)
    return text
```

### Why this is bad

Let **V** = vault size, **L** = answer length.

1. Load all V rows: **O(V)** time and memory, holding the lock.
2. For each of V tokens, scan the whole answer: **O(V × L)**.
3. Each `.replace()` that matches **builds a new string**: more allocation.

Concretely: V = 50,000, L = 500 characters, typically **3** tokens actually present. So 50,000 full
string scans to perform 3 substitutions — and it gets linearly worse forever as the vault grows, even
though answers stay the same size.

### The fix — invert the direction of the lookup

Don't ask "which of my tokens are in this text?" Ask "which tokens are in this text, and what do they
mean?"

```python
TOKEN_RE = re.compile(r'<<[A-Z_]+_\d+_[0-9a-f]{4}>>')

def unmask_text(self, text: str) -> str:
    found = set(TOKEN_RE.findall(text))
    if not found:
        return text
    placeholders = ','.join('?' * len(found))
    with self._get_connection() as conn:
        rows = conn.execute(
            f"SELECT token, real_value FROM token_mappings WHERE token IN ({placeholders})",
            tuple(found),
        )
        mapping = {r["token"]: r["real_value"] for r in rows}
    return TOKEN_RE.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)
```

**O(L) + one indexed query.** Independent of vault size. Also a shorter critical section, and
`TOKEN_RE.sub` does a single pass rather than V passes.

**Two bonus properties worth mentioning:**
1. The `mapping.get(m.group(0), m.group(0))` fallback leaves unrecognized tokens intact — so a
   **hallucinated token** from the LLM is visible in the output rather than silently vanishing. And
   you can log it as an anomaly, which is a genuine observability win the current code can't have.
2. The regex couples to the minting format at `vault.py:87`, so both should reference one shared
   constant. Naming that coupling is the kind of detail that shows care.

---

## 6. Idempotency and content-addressed storage

### What idempotency means

An operation is **idempotent** if performing it N times has the same effect as performing it once.
`SET x = 5` is idempotent; `x += 1` is not. It matters because networks retry, users double-click,
and job queues redeliver — and at-least-once delivery is the norm in distributed systems.

### My implementation

```python
# pipeline.py:200-205
@staticmethod
def _stable_id(text: str) -> str:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

# pipeline.py:216-217
ids = [self._stable_id(t) for t in texts]
self.vector_db.add_texts(texts, ids=ids)
```

The ID **is a hash of the content**. Chroma treats `add` with an existing ID as an upsert, so
re-ingesting identical text overwrites the same row instead of inserting a duplicate. That's
**content-addressed storage** — the same idea behind git object hashes, Docker layer digests and IPFS
CIDs.

**The UUID reformatting** (`8-4-4-4-12`) is cosmetic — it makes the ID look like a UUID so it doesn't
look out of place next to Chroma's native IDs. Note it truncates: only 32 of the 64 hex characters are
used, so 128 bits of the 256-bit digest. Still far beyond any realistic collision risk (birthday bound
around 2^64 items), and worth being able to say confidently rather than hand-waving.

**Why it was added:** the comment at `pipeline.py:214-215` says re-ingesting produced duplicate
entries with new UUIDs. So this was a **bug fix**, and the symptom is worth describing: duplicate
chunks crowd the top-k, so a query returns the same passage three times and genuinely relevant chunks
get pushed out of the context window. Retrieval quality degrades in a way that looks like a model
problem but is a storage problem.

### The limitation: updates are invisible

Content addressing makes re-ingestion idempotent and makes **editing** undetectable:

```
Monday:  "Name: John Doe, Salary: 50000"  → id abc123...
Tuesday: "Name: John Doe, Salary: 60000"  → id def456...   (different content, different id)
```

Both rows now exist. The stale one is never removed, so retrieval can return the **outdated salary**,
and with `top_k: 20` both will often be retrieved — the LLM sees contradictory context and picks
arbitrarily.

**The proper fix** is document-level lifecycle management:
1. Store a `document_id` in chunk metadata.
2. On re-ingest of that document, `delete(where={"document_id": doc_id})` first.
3. Then insert the new chunks.

That's the **expand/contract** or delete-then-insert pattern, and it's what makes updates and GDPR
erasure possible — which is why this limitation shows up again in the erasure discussion (file 02 §9).

**What exists instead:** two blunt instruments —
- `clear_vector_db()` (`pipeline.py:326`) deletes all documents but keeps the store. Note it calls
  `self.vector_db.get()["ids"]` with no filter (`pipeline.py:334`), pulling **every ID in the
  collection into memory** — fine at demo scale, an OOM at real scale.
- `reset_vector_db()` (`pipeline.py:343`) `shutil.rmtree`s the directory entirely. The docstring says
  it's for evaluation runs, to prevent stale IDs from previous sessions appearing in results — which is
  direct evidence the staleness problem was encountered in practice.

---

## 7. Transactions, isolation levels, and what I'd hit in Postgres

Useful vocabulary for a migration discussion, since "I'd move to Postgres" invites follow-ups.

### Isolation levels and the anomalies they permit

| Level | Dirty read | Non-repeatable read | Phantom read |
|---|---|---|---|
| Read Uncommitted | ✅ possible | ✅ | ✅ |
| Read Committed *(Postgres default)* | ❌ | ✅ | ✅ |
| Repeatable Read | ❌ | ❌ | ✅ |
| Serializable | ❌ | ❌ | ❌ |

- **Dirty read** — you see another transaction's uncommitted data.
- **Non-repeatable read** — you read the same row twice in one transaction and get different values.
- **Phantom read** — you run the same query twice and new rows appear.

### Why it matters for the vault specifically

My read-then-write in `store()` is exactly the shape that **Read Committed does not protect**. Two
transactions can both SELECT (neither sees the other's uncommitted insert), both decide the value is
new, both INSERT. Under Read Committed, the second gets a unique-violation error at commit.

Three ways to handle it, in increasing elegance:

1. **`SERIALIZABLE`** — Postgres detects the conflict and aborts one transaction with a
   serialization failure, which you retry. Correct, but you must write retry logic and throughput
   suffers under contention.
2. **`SELECT ... FOR UPDATE`** — pessimistic row lock. Doesn't help here, because the row doesn't
   exist yet, which is the whole problem. (Worth knowing *why* it doesn't apply — that's a sharper
   answer than just listing it.)
3. **`INSERT ... ON CONFLICT DO NOTHING RETURNING`** — a single atomic statement. No application
   lock, no retry loop, no isolation-level tuning. **This is the right answer** and it's why I'd
   reach for it in the migration.

### SQLite's model, for contrast

SQLite doesn't have configurable per-transaction isolation in the same way. In the default
**rollback-journal** mode it's effectively serialized: one writer at a time, readers blocked during a
write. With **WAL mode** (`PRAGMA journal_mode=WAL`) readers and one writer proceed concurrently —
a significant improvement and **one line of configuration I never set**. Combined with
`PRAGMA busy_timeout`, WAL would meaningfully improve the current single-node behaviour for almost no
effort. Good concrete "small fix I'd make first" to have in your pocket.

---

## 8. Why the vault is the single point of failure, in two senses

This is the best answer to "what's your biggest SPOF?" because the two senses pull against each other.

### Availability SPOF

One SQLite file, no replication, no backup, on an **ephemeral filesystem**. Lose it and:
- Every masked chunk in ChromaDB becomes permanently undecodable.
- The vector store survives and is simultaneously **worthless** — you have the index and no key.
- Audit history of which tokens existed is gone.

And this isn't hypothetical: Render's free tier wipes the disk on **every deploy** (main doc §5.13).

### Security SPOF

One table reverses every privacy guarantee in the system:
- Plaintext at rest, no encryption.
- Reachable via `/api/vault` with **no authentication**.
- No access logging on reads — the one operation you'd most want audited is the one that isn't
  (`log_access` exists at `audit_logger.py:52` and is never called from the vault path).

### Why that combination is genuinely awkward

The standard fix for an availability SPOF is **replication** — more copies. The standard fix for a
security SPOF is **fewer copies, tighter access**. They pull in opposite directions, and saying so
demonstrates you understand it as an engineering trade-off rather than a checklist item.

**The resolution** is the standard enterprise pattern, and it's worth being able to describe
precisely:

1. **Managed Postgres** — automated backups, point-in-time recovery, replication handled by the
   provider. Availability solved.
2. **Envelope encryption** — a KMS holds the master key and never releases it; it wraps per-record
   or per-tenant **data keys**; the ciphertext and wrapped key are stored together. Replicas hold
   only ciphertext, so replication no longer multiplies the *plaintext* attack surface. That's what
   resolves the tension.
3. **Deterministic HMAC as the lookup index.** Encrypted columns aren't searchable by equality — the
   same plaintext encrypts differently each time (correctly so). So store
   `HMAC(key, real_value)` as an indexed lookup column alongside the encrypted payload. Exact-match
   lookup still works; the index leaks equality only, not values. **This is exactly the
   `value_mappings` use case**, which is a satisfying closing of the loop.
4. **Authentication, authorization, and an audit event on every read**, since vault access is the
   highest-value action in the system.

---

## Concept checklist for this file

- [ ] I can say which ACID property protects which specific failure in `store()`
- [ ] I can explain why two tables exist, and concede one table + a unique index would be cleaner
- [ ] I can draw the check-then-act race and explain why `threading.Lock` doesn't fix it across
      processes
- [ ] I can explain the counter-divergence bug and why the fix is to delete the feature
- [ ] I can explain why `sqlite3` being unpatchable is what makes eventlet block
- [ ] I can state `unmask_text`'s complexity as O(V × L) and write the O(L) version
- [ ] I can define idempotency and explain content addressing with git/Docker as parallels
- [ ] I can explain why content-hash IDs make updates invisible, and name delete-then-insert
- [ ] I can name isolation levels and explain why `ON CONFLICT ... RETURNING` beats `SERIALIZABLE`
      here
- [ ] I can explain why availability and security SPOF fixes conflict, and how envelope encryption
      plus HMAC lookup resolves it
