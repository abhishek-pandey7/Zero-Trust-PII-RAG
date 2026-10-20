# 01 — RAG & Retrieval, Explained

Concepts taught from first principles, then tied back to exactly where they live in this repo.
Read this one first — it's the layer an interviewer will probe hardest, because it's the part
everyone claims to know and few can explain.

---

## 1. What an embedding actually is

Take the sentence "the employee lives in Mumbai." A computer can't compare that to another sentence
directly — strings either match or they don't. An **embedding model** maps text to a list of
numbers, a *vector*, such that texts with similar *meaning* land near each other in that
high-dimensional space.

My model is `sentence-transformers/all-MiniLM-L6-v2`, which produces **384 numbers** per input
(`src/zero_trust_rag/config.yaml:10`). So every chunk of my documents becomes a point in
384-dimensional space.

### Why "similar meaning lands nearby" works at all

The model is a small transformer trained on hundreds of millions of sentence pairs with a
contrastive objective: pull the vectors of related sentences together, push unrelated ones apart.
After enough of that, the geometry encodes semantics. It's not magic and it's not lookup — it's a
learned compression of meaning into coordinates.

**The practical consequence for me:** "Who works in the Mumbai office?" can match a chunk that never
contains the word "works". That's the entire reason I used embeddings rather than keyword search.

### Mean pooling — the bit I implemented by hand

A transformer doesn't emit one vector per sentence. It emits one vector **per token**. If your input
is 40 tokens, you get a 40 × 384 matrix. To get a single sentence vector you have to collapse that
— and the standard way is to average across the token axis. That's **mean pooling**.

That's exactly what `_mean_pool` at `pipeline.py:36-40` does:

```python
def _mean_pool(self, token_vecs: list) -> list:
    n = len(token_vecs)          # number of tokens
    dim = len(token_vecs[0])     # 384
    return [sum(token_vecs[t][d] for t in range(n)) / n for d in range(dim)]
```

For each of the 384 dimensions, average that dimension across all tokens. The guard at
`pipeline.py:51` (`if vec and isinstance(vec[0], list)`) checks whether the API returned a 2D matrix
needing pooling or already gave a single pooled vector — the HF endpoint does both depending on which
route you hit.

> **Interview-worthy detail:** this is a pure-Python nested comprehension doing `n × 384`
> multiply-accumulates. numpy is already in my `requirements.txt` and would do this in a single
> vectorized call, orders of magnitude faster. Naming that yourself is much better than having it
> pointed out.

**Alternatives to mean pooling:** CLS-token pooling (use only the first token's vector — works for
BERT models trained that way, worse for sentence-transformers), or max pooling (take the max per
dimension — sharper on keywords, worse on overall meaning). MiniLM is trained with mean pooling, so
mean pooling is correct here; it's not an arbitrary choice.

---

## 2. Cosine similarity, and why not Euclidean distance

Once text is a vector, "how similar are these?" becomes geometry. The standard measure is **cosine
similarity** — the cosine of the angle between two vectors:

```
cos(A, B) = (A · B) / (‖A‖ ‖B‖)
```

Range is −1 (opposite) to 1 (identical direction). It measures **direction, not magnitude**.

That last part is the point. A long document and a short sentence about the same topic have very
different vector *lengths* but similar *directions*. Euclidean distance would call them far apart;
cosine calls them similar. Since my chunks vary from a few words to 500 characters, cosine is the
right metric — and it's Chroma's default, which is why you won't find me configuring it anywhere.

---

## 3. Approximate nearest neighbour search (ANN) and HNSW

**The naive approach:** to find the 20 chunks closest to a query, compute cosine similarity against
every stored vector and sort. That's O(N) per query. At 2,500 chunks, fine. At 10 million, hopeless.

**ANN** trades a little accuracy for a lot of speed — it finds *almost certainly* the nearest
neighbours in roughly O(log N).

### How HNSW works, in plain terms

HNSW is *Hierarchical Navigable Small World*. Picture a transport network with layers:

- **Top layer:** a few vectors with long-range links. Intercontinental flights.
- **Middle layers:** more vectors, medium-range links. Domestic flights.
- **Bottom layer:** every vector, short local links. Walking.

A search enters at the top, greedily hops to whichever neighbour is closer to the query, and when it
can't improve at that layer it drops down a level and repeats. You cover enormous distance in a few
hops up top, then refine locally. That's the whole trick.

**Where this is in my repo:** I never wrote it — ChromaDB provides it. But you can see it on disk.
Those files under `chroma_store/`:

| File | What it is |
|---|---|
| `data_level0.bin` | the bottom layer — the full vector data |
| `link_lists.bin` | the graph edges of the upper layers |
| `header.bin` / `length.bin` | index metadata |

Being able to point at `link_lists.bin` and say "that's HNSW's upper-layer adjacency" is a strong
signal that you know what your dependency is doing.

**The trade-off to state out loud:** ANN is *approximate*. There is no guarantee the true top-20 is
returned. Tunable via HNSW's `ef_search` — higher means more accurate and slower. I use Chroma's
defaults and never tuned it, which is honest and fine.

---

## 4. RAG: retrieval-augmented generation

### The problem RAG solves

An LLM knows only what was in its training data. It has never seen your employee records. You have
three options:

1. **Fine-tune** — retrain the model's weights on your data.
2. **Long context** — paste the entire corpus into every prompt.
3. **RAG** — at query time, retrieve only the relevant passages and put *those* in the prompt.

### Why RAG wins for this project

| | Fine-tuning | Long context | RAG |
|---|---|---|---|
| New document added | retrain | free | just index it |
| Cost per query | low | scales with corpus | scales with top-k |
| Delete one person's data | ~impossible | easy | delete rows |
| Corpus size limit | n/a | context window | ~unbounded |

My test corpus is over **1,000,000 characters** — roughly 250,000+ tokens. Long-context is out on
cost alone, and I'd pay it on *every single query*. Fine-tuning is out because my tokens are
*mutable by design*: the vault grows, so the same person's token is minted the first time they're
seen. Baking a snapshot of a mutable mapping into model weights makes no sense.

### The RAG loop in my code

```
question
  → redact                                 pipeline.py:271
  → embed the masked question              via similarity_search → _HFApiEmbeddings.embed_query
  → ANN search for top-k chunks            pipeline.py:274
  → join chunks into a "context" string    pipeline.py:275
  → fill prompt template with context+Q    pipeline.py:281
  → call the LLM                           pipeline.py:292 (prompt | llm | StrOutputParser)
  → unmask the answer                      pipeline.py:302
```

The bail-out at `pipeline.py:277-279` matters more than it looks: if retrieval returns nothing, I
return "I don't have enough context to answer that question" rather than calling the LLM with an
empty context. An LLM handed an empty context will confidently invent an answer. That early return
is a hallucination guard.

### LCEL — what `prompt | llm | StrOutputParser()` means

That's LangChain Expression Language. The `|` operator is overloaded to compose *runnables* into a
pipeline, the same idea as a Unix pipe: `prompt` renders the template, passes the message list to
`llm`, which returns a message object, which `StrOutputParser` unwraps to a plain string. It's
function composition with a nicer syntax, and each stage is independently swappable.

---

## 5. Chunking — why you split documents at all

You can't embed a whole 1M-character document as one vector. Two reasons:

1. **Model input limit.** MiniLM truncates at 256 word-pieces. Anything past that is silently
   dropped — a genuinely nasty failure mode, because nothing errors.
2. **Semantic dilution.** Averaging 400 pages into 384 numbers produces a vector that means
   "documents in general". It'd be equally close to every query, i.e. useless.

So you split. My splitter (`pipeline.py:185-190`):

```python
RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", " ", ""],
)
```

### What "recursive" means here

It tries the separators in order, coarse to fine. Split on blank lines first; if a resulting piece
is still over 500 chars, split *that* piece on single newlines; still too big, split on spaces; last
resort, hard-cut mid-word. This is why it's called recursive — it keeps descending the separator list
until pieces fit.

Why the ordering matters for my data: employee records are newline-delimited, so `\n\n` and `\n`
nearly always find a natural boundary, and the character-level fallback almost never fires. On dense
prose it'd behave differently.

### Why overlap

Without overlap, a boundary landing mid-record splits it:

```
chunk 1: "... Name: John Doe"
chunk 2: "Aadhaar: 1234-5678-9012 ..."
```

Neither chunk can answer "what's John Doe's Aadhaar?" — the association is destroyed. With 100
characters of overlap, chunk 2 begins with the tail of chunk 1, so the pairing survives somewhere.

**Cost:** ~20% more vectors to embed and store (500 chunk / 100 overlap = 1 in 5 characters is
duplicated). You're buying recall with storage and embedding calls.

**A subtlety specific to this project:** my chunks are *masked* text, and tokens are longer than the
values they replace — `<<EMAIL_ADDRESS_07_9d1f>>` is ~26 chars where `a.b@corp.com` was ~12. So a
500-character window holds meaningfully fewer real facts post-masking than pre-masking. Nobody asks
this, and it lands well if you volunteer it.

---

## 6. top-k and the precision/recall dial

`top_k: 20` (`src/zero_trust_rag/config.yaml:21`, read at `pipeline.py:273`) means retrieve the 20
nearest chunks and put them all in the prompt.

- **Low k (say 3):** tight, cheap, fast. If the answer isn't in those 3, you get a confident wrong
  answer.
- **High k (say 20):** better odds the answer is in there. But more tokens (cost + latency), and
  you're relying on the model to ignore 19 irrelevant chunks — real models get distracted, and
  there's a well-documented "lost in the middle" effect where facts buried mid-context get ignored.

**Note the git history here:** a commit message says *"increase top_k limit"*, and the two config
files disagree — dead root file says 3, the live one says 20. So k was raised deliberately, almost
certainly because recall was poor on the 1,000-record document. That's an empirical tuning story you
can tell, which is much better than "20 seemed fine."

**What I'd add instead of brute-force k:** a **reranker**. Retrieve 50 cheaply with ANN, then score
those 50 with a cross-encoder — a model that reads the query and a chunk *together* rather than
comparing precomputed vectors — and keep the best 5. Far more accurate per token of context. It's a
second model call, so latency and complexity go up. There's no reranker in this repo, and naming it
as the known next step is the right move.

---

## 7. Why redaction doesn't destroy retrieval (and where it does)

This is the question that separates people who understand this project from people who've memorized
it.

**Why it survives.** Because tokens are **deterministic** (`vault.py:75-81`). "John Doe" → the same
`<<PERSON_01_3f9c>>` in the indexed chunk and in the query, months apart. The embedding model has
never seen that string, but it doesn't need to *understand* it — it needs the query vector and the
chunk vector to be close, and a shared rare literal string is a very strong similarity signal. In
fact exact-token matching is *easier* for retrieval than paraphrase matching.

**Where it genuinely degrades.**

1. **Out-of-vocabulary noise.** `<<IN_AADHAAR_01_b8e2>>` tokenizes into meaningless word-pieces.
   The surrounding real words carry the semantic load; the token contributes near-noise.
2. **Lost semantics on quasi-identifiers.** This is the real cost. My config masks `DATE_TIME`,
   `LOCATION` and `URL` (`src/zero_trust_rag/config.yaml:30-34`). So "joined March 2023 in Mumbai"
   becomes two opaque tokens, and *"who joined before 2023"* becomes unanswerable — not badly
   answered, **unanswerable**. There is no signal left to reason over.
3. **Thinner vectors overall.** Masked text has less semantic content per chunk, so all similarity
   scores compress toward each other and ranking gets noisier.

**The better design, which I'd propose unprompted:** tiered redaction.

| Tier | Example | Treatment |
|---|---|---|
| Direct identifier | Aadhaar, email, PAN, phone | full tokenization |
| Quasi-identifier | date, city | **generalize** — "2023", "Maharashtra" |
| Non-identifying | department, job title | leave alone |

You keep the analytic value and still break direct re-identification. That's k-anonymity thinking —
a graduated trade-off instead of all-or-nothing — and it's the single most interesting improvement
available in this project.

---

## 8. The hybrid-retrieval insight worth volunteering

Here's a genuinely non-obvious observation about this specific system.

**BM25** is classic keyword search: it scores documents on term overlap with the query, weighting
rare terms heavily (IDF) and normalizing for document length. It's the default in Elasticsearch and
Postgres full-text search. Normally it's *worse* than embeddings for natural-language questions,
because it can't match "employee" to "staff member."

But my documents are full of tokens like `<<PERSON_01_3f9c>>` — strings that are:

- **exact** (no paraphrase possible),
- **extremely rare** (maximum IDF weight),
- **present identically** in both query and document after redaction.

That's the ideal case for BM25. A hybrid retriever — BM25 for exact token hits, embeddings for
semantic similarity on the surrounding text, fused with reciprocal rank fusion — would almost
certainly beat either alone here. Tokenization accidentally created a perfect lexical-search signal.

There's no hybrid retrieval in this repo. Saying "here's a property of my own design I noticed after
the fact, and here's the retrieval architecture it implies" is exactly the kind of answer that
separates a good interview from an average one.

---

## Concept checklist for this file

- [ ] I can explain what an embedding is without saying "it's like a fingerprint"
- [ ] I can explain mean pooling and point to `pipeline.py:36`
- [ ] I know why cosine rather than Euclidean
- [ ] I can describe HNSW's layered graph and point at `link_lists.bin`
- [ ] I can justify RAG over fine-tuning and over long context, with my corpus size as the reason
- [ ] I can explain what "recursive" means in `RecursiveCharacterTextSplitter`
- [ ] I can explain why overlap exists and what it costs
- [ ] I can argue both sides of top_k = 20, and name reranking as the better fix
- [ ] I can explain why deterministic tokens preserve retrieval, and exactly where masking still hurts
- [ ] I can make the BM25/hybrid argument about my own token format
