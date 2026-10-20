# 02 — PII, Tokenization & Zero-Trust, Explained

This is the file that makes the project distinctive. Anyone can build a RAG pipeline; the privacy
architecture is what you're actually selling.

---

## 1. What "PII" means, and why the categories matter

**PII** — Personally Identifiable Information — is any data that can identify a specific person.
The important thing for interviews is that it isn't one flat category:

| Category | Examples | Property |
|---|---|---|
| **Direct identifier** | Aadhaar, PAN, email, phone, SSN | identifies a person *on its own* |
| **Quasi-identifier** | date of birth, ZIP code, job title, gender | identifies a person *in combination* |
| **Sensitive attribute** | salary, medical condition, religion | the thing you're protecting, not the identifier |

**Why quasi-identifiers are the subtle part.** The famous result here: ZIP code + date of birth +
gender uniquely identifies ~87% of the US population. None of those three is PII on its own. Together
they're an identifier. This is why "just remove the names" is never sufficient, and it's the
intellectual core of de-identification.

**Where this shows up in my repo:** my configured entity list
(`src/zero_trust_rag/config.yaml:24-40`) mixes all three categories without distinguishing them —
`US_SSN` (direct) sits next to `DATE_TIME` and `LOCATION` (quasi) and `NRP`
(nationality/religion/political — a sensitive attribute). I treat all 16 types identically with full
tokenization. That's *safe* but blunt, and it's the source of the accuracy cost documented in
section 5.11 of the main prep doc.

---

## 2. The de-identification spectrum

There are more options than "keep it" and "delete it," and knowing the vocabulary is worth a lot.

```
                       reversible?   analytics preserved?
redaction   → "[REMOVED]"      no          none
masking     → "****1234"       no          partial (format/last digits)
generalize  → "1990" or "Maharashtra"  no  bucketed
tokenize    → "<<PERSON_01>>"  YES (via vault)  identity only
FPE         → "8823641097"     YES (via key)    format + identity
encryption  → "gAAAAAB..."     YES (via key)    none
synthesis   → a fake but statistically similar record   no   distributional
```

**Where I sit:** tokenization. `pii_redactor.py:172-173` does the substitution;
`vault.py:72` mints the surrogate.

**Why tokenization and not the others, for this system:**

- **Redaction/masking** is irreversible, so I could never show the user a real answer. The product is
  "ask questions about your private data and get real answers." Dead on arrival.
- **Encryption in-place** produces ciphertext that is long, high-entropy, and semantically
  meaningless — it would wreck both chunking (length blowup) and embedding quality, and the LLM
  couldn't treat it as a stable referent.
- **Format-preserving encryption** is the genuinely interesting alternative. It emits a value that
  *looks* like the original type — a fake but well-formed phone number. Better when downstream systems
  validate formats. The downside is that it's reversible *with a key*, so key compromise reverses
  everything everywhere at once, versus a vault where compromise requires exfiltrating a database. It's
  also easy to implement subtly wrong (you want FF1/FF3-1, not something hand-rolled).
- **Synthetic data** is great for training and testing and useless here, because the user wants facts
  about *actual* employees.

---

## 3. Tokenization mechanics in this codebase

### The token format

```python
# vault.py:86-87
short_id = uuid.uuid4().hex[:4]
token = f"<<{entity_type}_{count:02d}_{short_id}>>"
```

Producing e.g. `<<PERSON_04_a1b2>>`. Four design decisions in one line:

1. **`<<` and `>>` delimiters.** A sequence that essentially never occurs in natural text, so the
   unmask pass can't accidentally fire on real content, and a human reading masked output can see
   instantly what was redacted.
2. **`entity_type` embedded.** Deliberate information leak: the LLM learns this is a *person* vs. an
   *email*, which it needs to answer "what's their contact info?" I'm trading a little privacy
   (an attacker learns the type distribution) for a lot of utility.
3. **`count:02d`** — a per-type counter from `_counters` (`vault.py:83-84`), so tokens are
   human-readable and ordered. Useful for demos. Also a scaling liability: see §6.
4. **4 hex chars of a UUID4.** Unguessability. Without it, tokens would be enumerable — an attacker
   who knows there are 50 people could reference `<<PERSON_01>>` through `<<PERSON_50>>`. It's a small
   defense (16 bits) but the right instinct.

### Why deterministic, restated precisely

```python
# vault.py:75-81
cursor = conn.execute("SELECT token FROM value_mappings WHERE real_value = ?", (real_value,))
row = cursor.fetchone()
if row:
    return row["token"]      # ← existing token, do not mint a new one
```

Two independent things break without this:

1. **Retrieval.** A chunk indexed Monday and a query asked Friday must produce identical tokens or
   they don't match.
2. **Unmasking.** The LLM's answer contains tokens it saw in *retrieved context*, minted during
   earlier, unrelated requests. `unmask_text` must be able to resolve them.

### The privacy cost of determinism, stated honestly

Deterministic tokenization leaks **structure**, even though it hides **values**. An attacker with
read access to `chroma_store/` can:

- count occurrences (frequency analysis — the most-mentioned person is probably a manager)
- see co-occurrence (which tokens appear in the same chunk → the org graph)
- see type distribution (how many people, how many Aadhaar numbers)

This is the same reason deterministic encryption is discouraged for database columns: equal
plaintexts produce equal ciphertexts, and equality itself is information. I accepted it because
random-per-occurrence tokens make the system non-functional. **Naming this trade-off before being
asked is the single highest-value thing in this file.**

---

## 4. How PII is actually detected: NER + pattern matching

Detection has two fundamentally different mechanisms, and Presidio runs both.

### Pattern matching (regex) — for structured identifiers

Aadhaar numbers, PAN cards, credit cards have *format*. Regex nails them. My three custom
recognizers (`pii_redactor.py:42-83`):

```python
Pattern("Aadhaar separator", r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b", 0.85)
Pattern("PAN format",        r"\b[A-Z]{5}\d{4}[A-Z]\b",         0.8)
Pattern("India phone +91",   r"\+91[-\s]?\d{5}[-\s]?\d{5}",     0.7)
```

Note the trailing float — that's a **confidence score**, and it's the interesting part.

- Aadhaar at **0.85**: three groups of four digits with separators is a distinctive shape.
- PAN at **0.8**: five letters, four digits, one letter is very distinctive.
- Bare 10 digits at **0.5** (`pii_redactor.py:64`): could be a phone, an order ID, an employee
  number. Low confidence is *correct*.

**And here's the bug worth volunteering:** my active `score_threshold` is **0.6**
(`src/zero_trust_rag/config.yaml:41`), so Presidio filters that 0.5 pattern out before I ever see it.
A bare `9876543210` with no separators isn't caught by that pattern at all. I wrote a recognizer that
can never fire. Catching your own dead code is a strong signal.

**Why add custom recognizers at all?** Presidio's built-ins are US/EU-centric. Aadhaar and PAN
weren't covered in the form I needed, and Indian phone conventions (+91, 5-5 digit splits) differ from
the `PHONE_NUMBER` built-in's expectations. `registry.add_recognizer` (`pii_redactor.py:104-106`) is
the extension point — three classes, ~40 lines.

### Named Entity Recognition (NER) — for unstructured identifiers

Names and places have **no format**. "Priya Sharma" and "Mumbai" are just capitalized words; so is
"Monday" and "Google." You can't regex a name — you need a model that uses *context*.

That's spaCy. `pii_redactor.py:96-100`:

```python
nlp_configuration = {
    "nlp_engine_name": "spacy",
    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
}
```

The model was trained on annotated text and learned that a capitalized bigram following "Name:" or
preceding "reports to" is probably a PERSON. It's statistical inference, which means it has a
**false-negative rate that is not zero and cannot be made zero**.

**`_sm` is the small model**, chosen for memory (the comment in `requirements.txt` says so). Trade-off:
- `en_core_web_sm` — ~12MB, fast, weakest accuracy
- `en_core_web_md` / `_lg` — larger, includes word vectors, better accuracy
- `en_core_web_trf` — transformer-based, best accuracy, needs torch and much more memory

The small model is visibly why I needed two workarounds — newline truncation and possessive trimming
(below). Both are patches for weak entity boundaries.

---

## 5. Precision, recall, and why you should bias toward recall here

Standard classification vocabulary, applied to PII:

|  | Model says PII | Model says not PII |
|---|---|---|
| **Actually PII** | true positive ✓ | **false negative — DATA BREACH** |
| **Not PII** | **false positive — degraded answer** | true negative ✓ |

- **Precision** = of everything I flagged, how much really was PII? (Low precision → over-redaction →
  worse answers.)
- **Recall** = of all real PII, how much did I catch? (Low recall → **leaks**.)

**The asymmetry is the whole point.** A false positive means the LLM sees `<<LOCATION_03_4c1e>>`
instead of "the Mumbai office" and gives a slightly worse answer. A false negative means a real
Aadhaar number is transmitted to a third-party API and appears in their logs. Those costs are not
remotely comparable.

**So `score_threshold` is a dial on that trade-off** (`src/zero_trust_rag/config.yaml:41`, applied at
`pii_redactor.py:128`). Lower threshold → more detections → higher recall, lower precision. I set
**0.6**, which is middling. Given the asymmetry above, the defensible position is that I should be
*lower* — 0.4, maybe 0.3 — and accept more over-redaction. The dead root config file actually says
0.3, which suggests I started there and raised it, probably because over-redaction was hurting demo
answers. That's a real trade-off story, and "I'd re-evaluate and likely lower it" is the right
conclusion to state.

---

## 6. The span post-processing tricks (the underrated part)

Presidio returns `RecognizerResult` objects with `start`, `end`, `entity_type`, `score`. Raw output
isn't directly usable, and `pii_redactor.py:134-162` is where I fix it. These details are very
interview-friendly because they show you debugged real data rather than a tutorial.

### 6.1 Newline truncation (`pii_redactor.py:139-143`)

```python
if '\n' in matched_text:
    result.end = result.start + matched_text.index('\n')
```

**The problem.** On structured records:

```
Name: John Doe
Email: john@corp.com
```

spaCy's small model frequently returns one PERSON span covering `"John Doe\nEmail"` — it doesn't
treat the newline as a hard boundary. Substituting that span gives you
`Name: <<PERSON_01_3f9c>>: john@corp.com` — the field structure is destroyed, and worse, the token
now *means* something different (a name plus a label) so the same person in a differently-formatted
document gets a different token. Determinism breaks.

**The fix.** Hard-truncate any span at the first newline. In record-structured data an entity never
legitimately spans lines, so this is safe here. On free prose it would be wrong — a name split across
a line wrap would be truncated. Worth noting as a domain-specific assumption rather than a universal
fix.

### 6.2 Possessive trimming (`pii_redactor.py:146-156`)

```python
while result.end > result.start and sanitized[result.end-1].isspace():
    result.end -= 1
current_match = sanitized[result.start:result.end].lower()
if current_match.endswith("'s"):
    result.end -= 2
```

**The problem.** The query "What is **John Doe's** Aadhaar?" yields a PERSON span of `"John Doe's"`.
The vault has never seen that string, so it mints a *new* token — which appears nowhere in the index.
Retrieval returns nothing and the user gets "I don't have enough context." A silent, maddening
failure.

**The fix.** Strip trailing whitespace, then trailing `'s` or `'`, so both `"John Doe"` and
`"John Doe's"` normalize to the same vault key.

> This is the clearest example in the codebase of **normalization for determinism** — and it's a
> general principle worth naming: in any deterministic-token system, the lookup key must be
> canonicalized or determinism silently fails. The same class of bug would come from case variation
> ("john doe"), extra internal whitespace, or unicode width variants — none of which I handle. Good
> thing to volunteer as the next hardening step.

### 6.3 Overlap resolution (`pii_redactor.py:191-206`)

**The problem.** Recognizers fire independently, so one 10-digit string can match `PHONE_NUMBER`
(mine) *and* `US_SSN` (built-in). You cannot substitute two tokens into the same character range —
the second substitution corrupts the first.

**The implementation** — a greedy interval sweep:

```python
results = sorted(results, key=lambda r: (r.start, -r.score))
filtered, last_end = [], -1
for result in results:
    if result.start >= last_end:
        filtered.append(result)
        last_end = result.end
```

**The bug** (main doc §5.6). The docstring says it keeps "the highest scoring ones." It doesn't. The
`-score` in the sort key only breaks ties *at the same start offset*. If a low-confidence span starts
one character earlier and overlaps a high-confidence one, the low-confidence span wins on position.
So a 0.3 `DATE_TIME` can beat a 0.85 Aadhaar.

**The correct algorithm** is **weighted interval scheduling**: sort by *end* offset, then dynamic
programming to maximize total score over non-overlapping intervals. Same O(n log n), ~15 more lines,
and it actually does what the docstring claims. Knowing the name of the algorithm and why the greedy
version is wrong is a genuinely good interview moment.

### 6.4 Reverse-order replacement (`pii_redactor.py:165`)

```python
results = sorted(results, key=lambda r: r.start, reverse=True)
```

**Why.** You're mutating a string using character offsets computed against the *original*. Replace
left-to-right and every replacement shifts all later offsets — tokens are longer than most PII values,
so every subsequent substitution lands in the wrong place, progressively worse.

Going **right to left**, every replacement happens *after* all the offsets you haven't used yet, so
they stay valid. Simple, and the kind of detail that shows you reasoned about it rather than
copy-pasted it.

**The performance flaw:** line 173 is `safe_text[:start] + token + safe_text[end:]` — a full string
copy per entity. On 1M chars with thousands of entities that's quadratic. The fix is to collect
segments in a list and `''.join()` once: linear, same line count.

---

## 7. Input sanitization as a security control

```python
# pii_redactor.py:113-114
sanitized = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
sanitized = re.sub(r'[\x80-\x9f]', '', sanitized)
```

Strips C0 control characters (keeping `\t`, `\n`, `\r`), DEL, and the C1 range.

**Why this is a security control and not just tidying:**

1. **Evasion.** Invisible characters inserted inside a PII value break regex matching:
   `1234\x0b5678\x0b9012` doesn't match my Aadhaar pattern, so it sails through undetected. Stripping
   them first means the pattern sees the real digits. This is the defensive-ordering insight — you
   **normalize before you detect**, never after.
2. **Injection.** Control characters can smuggle payloads past display and logging layers, and can be
   used to manipulate how text renders in a terminal or prompt.
3. **Log integrity.** My audit log is JSON Lines. A raw control character in a value can corrupt a
   line and break the parser for everything downstream.

**What's missing:** Unicode normalization. NFKC would collapse homoglyphs (Cyrillic `а` vs. Latin
`a`), full-width digits (`１２３４`), and zero-width joiners — all of which defeat both my regexes and
spaCy, and none of which my two `re.sub` calls touch. One line of `unicodedata.normalize('NFKC', text)`
before the strips. Naming this gap is better than waiting for it.

---

## 8. Zero-trust, precisely

**"Zero trust"** in its original network sense means: never trust based on location. No implicit
"inside the firewall means safe." Verify every request, assume breach, enforce least privilege.

**My application of it:** I don't trust the LLM provider with plaintext, regardless of their
contractual promises or their security posture. The trust boundary is my process; everything past the
network call is hostile territory.

### Threat model, written out

| Threat | Mitigated? | How / why not |
|---|---|---|
| LLM provider logs prompts | ✅ | they only ever receive tokens |
| LLM provider trains on my data | ✅ | tokens carry no values |
| Provider breach exposes prompts | ✅ | attacker gets tokens, not PII |
| Network interception | ✅ | TLS, plus payload is masked anyway |
| Presidio misses an entity | ❌ | **no pre-egress verification** (main doc §5.15) |
| Attacker reads my vault | ❌ | plaintext SQLite, no encryption at rest |
| Attacker hits `/api/vault` | ❌ | **no authentication at all** (§5.1) |
| Another tenant's data leaks | ❌ | **single global vault, no scoping** (§5.2) |
| Correlation on masked vectors | ⚠️ | inherent to deterministic tokens |

Presenting it as a table like this is powerful in an interview: it shows you know security is about
*enumerated threats*, not vibes, and volunteering the ❌ rows is far stronger than being walked into
them.

### The verification gap, as the thing I'd fix first

The design *assumes* Presidio's recall. `validate_input(masked_text)` at `pipeline.py:234` only
checks length — nothing verifies that masking succeeded.

**What I'd add:** after redaction, before egress, sweep the masked text for any plaintext value the
vault already knows and hard-fail if one appears. An Aho-Corasick automaton over the vault's values
does this in a single linear pass over the text, regardless of vault size.

**What it catches:** the most likely real leak in a deterministic-token system — a value tokenized in
document A but *missed* by the detector in document B (different formatting, different context). The
vault already knows it's PII; the detector just failed that time.

**What it doesn't catch:** never-before-seen PII. Nothing automated does. Which is why the honest
claim for this project is **"high-recall automated redaction with a full audit trail"**, not
"guaranteed zero leakage" — and being precise about that is the difference between sounding credible
and sounding oversold.

---

## 9. Compliance concepts worth having ready

### GDPR Article 17 — right to erasure, and crypto-shredding

Tokenization gives an elegant answer *in principle*: delete one row from `token_mappings` and every
occurrence of that person, across every chunk in the vector store, becomes permanently unresolvable.
That's **crypto-shredding** — destroy the key rather than hunting down every copy of the data. One
delete, global effect, no reindexing.

**What remains, honestly:** the token still sits in the index, so the structural facts persist — that
*someone* appears in those documents, and all their co-occurrence relationships. Under GDPR that's
arguably still personal data because it's *linkable*. And my content-hash IDs (`pipeline.py:200`)
carry no `document_id` metadata, so I can't even locate a subject's chunks to delete them.

There's also a real tension you should name: the audit log (`audit_logger.py:30`) deliberately records
token-level events **forever**. Compliance wants an immutable trail; GDPR wants erasure. Those pull
against each other, and resolving it is partly a legal question, not an engineering one.

### Pseudonymization vs. anonymization (GDPR Recital 26)

- **Pseudonymized** data is still personal data — reversal is possible with additional information.
  It gets reduced obligations but not exemption.
- **Anonymized** data is out of scope entirely — reversal must be *impossible*.

**My system is pseudonymization, not anonymization**, because the vault exists. Being precise about
this matters: claiming "anonymized" would be legally wrong, and an interviewer with a privacy
background will notice immediately. The right framing is: *"this is pseudonymization under GDPR —
the vault means reversal is possible, so the data remains in scope, and the control that matters is
who can reach the vault."*

### Data minimization and purpose limitation

The LLM receives the minimum needed to answer: top-k masked chunks, not the corpus, and no plaintext.
That's data minimization by construction rather than by policy. Worth one sentence if compliance
comes up — it's the kind of thing that sounds like you've actually read the regulation.

---

## Concept checklist for this file

- [ ] I can distinguish direct identifiers, quasi-identifiers and sensitive attributes, with the
      ZIP+DOB+gender example
- [ ] I can place tokenization on the de-identification spectrum and say why FPE was the real
      alternative
- [ ] I can justify each of the four design decisions in the token format
- [ ] I can explain why determinism is required, and what structure it leaks
- [ ] I can explain why names need NER and Aadhaar needs regex
- [ ] I can state the precision/recall asymmetry and argue my threshold should probably be lower
- [ ] I can explain all four span post-processing steps and why each exists
- [ ] I can explain why sanitization must happen *before* detection
- [ ] I can present the threat-model table including the unmitigated rows
- [ ] I can explain crypto-shredding and why my system is pseudonymization, not anonymization
