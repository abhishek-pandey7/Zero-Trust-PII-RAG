# Zero-Trust PII RAG Pipeline

A production-ready RAG (Retrieval-Augmented Generation) pipeline that strictly masks PII (Personally Identifiable Information) before sending any data to external Application Programming Interfaces (APIs). External Large Language Models (LLMs) only see masked placeholders such as `<<PERSON_01_A1B2>>`. All unmasking happens exclusively within your secure local environment.

## Table of Contents
1. [Security Model](#security-model)
2. [Architecture Overview](#architecture-overview)
3. [Component Breakdown](#component-breakdown)
4. [Installation and Setup](#installation-and-setup)
5. [Configuration Parameters](#configuration-parameters)
6. [Supported PII Types](#supported-pii-types)
7. [Process Lifecycles](#process-lifecycles)
8. [Usage Instructions](#usage-instructions)
9. [Sample Data and Queries](#sample-data-and-queries)
10. [Audit Logging](#audit-logging)

---

## Security Model

The Zero-Trust framework ensures that sensitive data is completely isolated from third-party services. The core tenets are:

1. **Pre-Transmission Redaction**: All text is analyzed and redacted locally before it ever reaches an external network boundary.
2. **Opaque Tokenization**: Detected PII is mapped to cryptographic, irreversible tokens (e.g., `<<EMAIL_ADDRESS_04_b1c4>>`) rather than being encrypted or hashed directly into the text.
3. **Local Unmasking Protocol**: When the LLM generates a response using the opaque tokens, the system translates these tokens back into plaintext locally just prior to returning the result to the end-user.
4. **Comprehensive Audit Logs**: Every ingestion, redaction, unmasking, and lookup operation is recorded locally in a JSON-formatted format for strict compliance and monitoring.

## Architecture Overview

```text
User Input (Containing Plaintext PII)
       |
       v
+---------------------------------------------------+
|  pii_redactor.py                                  |
|  - NLP processing via Microsoft Presidio          |
|  - Input sanitization (control characters)        |
|  - Possessive normalization (stripping 's)        |
|  - Token replacement                              |
+---------------------------------------------------+
       |
       v
+---------------------------------------------------+
|  vault.py                                         |
|  - SQLite persistence layer                       |
|  - One-to-one Token <-> PII mapping               |
|  - Thread-safe synchronization                    |
+---------------------------------------------------+
       |
       v
[Masked Text - Safe for Transmission]
       |
       v
+---------------------------------------------------+
|  Local Embeddings Function                        |
|  (HuggingFace MiniLM executed locally)            |
+---------------------------------------------------+
       |
       v
+---------------------------------------------------+
|  ChromaDB / Vector Database                       |
|  (Local persistent storage of embedded chunks)    |
+---------------------------------------------------+
       |
       v
User Query (Containing Plaintext PII)
       |
       v
+---------------------------------------------------+
|  Query Redaction -> Context Retrieval -> Groq LLM |
|  (LLM operates purely on semantic tokens)         |
+---------------------------------------------------+
       |
       v
[Masked Output from LLM]
       |
       v
+---------------------------------------------------+
|  vault.unmask_text()                              |
|  (Reverse lookup and restoration)                 |
+---------------------------------------------------+
       |
       v
Final Answer (Unmasked Plaintext)
```

## Component Breakdown

The repository is modularized to ensure separation of concerns:

- `src/zero_trust_rag/pipeline.py`: The orchestrator class (`ZeroTrustRAG`) combining text splitting, vector storage, and the Langchain orchestration logic.
- `src/zero_trust_rag/pii_redactor.py`: Implements the `PIIRedactor` class leveraging `presidio-analyzer`. Contains specific logic to strip newline-span artifacts and normalizes trailing possessives to guarantee static token hashing.
- `src/zero_trust_rag/vault.py`: Implements `PIIVault`, a thread-safe singleton utilizing SQLite for deterministic token mapping and system-restart survival.
- `src/zero_trust_rag/audit_logger.py`: Implements logging facilities to `audit.log` tracking exact system changes locally.
- `src/zero_trust_rag/utils.py`: Contains API key validators and execution decorators like `retry_on_api_error`.
- `app.py`: A Flask and SocketIO web server presenting a real-time reactive user interface to visualize ingestion and unmasking steps live.

## Installation and Setup

### Prerequisites
- Python 3.9 or higher
- Git

### Deployment Steps

1. **Create Virtual Environment**
   ```bash
   python -m venv venv
   ```

2. **Activate the Environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux / macOS:
     ```bash
     source venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Environment Variables**
   Create a `.env` file at the root directory containing the following:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   HUGGINGFACEHUB_API_TOKEN=your_hf_token_here
   FLASK_ENV=development
   ```

## Configuration Parameters

All pipeline settings are centralized within `src/zero_trust_rag/config.yaml`. No code modifications are strictly required to configure the system limits.

### Configuration Sections

- **llm**: Define provider (`groq`), model selection, and generation temperature.
- **embeddings**: Model selection for vector encoding (defaults to `sentence-transformers/all-MiniLM-L6-v2`).
- **chroma**: Set the storage directory for persisted embeddings.
- **splitter**: Tuning parameters for `RecursiveCharacterTextSplitter` including `chunk_size` and `chunk_overlap`.
- **retrieval**: The `top_k` documents to retrieve per query based on cosine similarity.
- **presidio**: Lists the active internal recognizers and threshold scores (defaults to `0.6`).
- **security**: Database paths and retry logic ceilings.

## Supported PII Types

The default configuration intercepts the following entity types via SpaCy NER and regex heuristics:

- PERSON
- EMAIL_ADDRESS
- PHONE_NUMBER
- US_SSN
- CREDIT_CARD
- IBAN_CODE
- IP_ADDRESS
- URL
- LOCATION
- DATE_TIME
- NRP (Nationality, Religion, Political Affiliation)
- MEDICAL_LICENSE
- US_PASSPORT
- US_DRIVER_LICENSE
- IN_PAN (Indian Permanent Account Number via Custom Recognizer)
- IN_AADHAAR (Indian Aadhaar via Custom Recognizer)

## Process Lifecycles

### The Ingestion Lifecycle
1. **Input Validation**: Verifies that input length and types are valid.
2. **Line-Preserving Sanitization**: Removes irregular control characters, keeping carriage returns and newlines intact.
3. **Possessive Normalization**: Drops trailing grammatical possessives from names to prevent diverging token maps.
4. **Vault Storage**: Assigns or retrieves unique cryptographic tokens for PII instances.
5. **Mask Substitution**: Strings replacement applied linearly.
6. **Vector Alignment**: Text split and stored in ChromaDB using local embedding models.

### The Querying Lifecycle
1. **Query Redaction**: Real values in the user query are matched and explicitly tokenized.
2. **Context Searching**: ChromaDB retrieves chunks associated mathematically with the prompt tokens.
3. **LLM Inference**: The strictly masked prompt alongside corresponding masked chunks is fed to the remote Groq endpoint.
4. **Response Interception**: The payload response is fetched securely.
5. **Text Unmasking**: Cryptographic tokens are flipped back to their initial string representations via SQLite indices.
6. **Log Output**: Output is transmitted back to the user interface.

## Usage Instructions

### Direct API via Python

```python
from src import ZeroTrustRAG
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize local pipeline
rag = ZeroTrustRAG(groq_api_key=os.getenv("GROQ_API_KEY"))

# Ingestion Step
data = "Employee John Doe has PAN card ABCDE1234F."
result = rag.ingest(data)
print(f"Chunks: {result['chunks_created']}, PII Masked: {result['pii_redacted']}")

# Actionable Query Step
answer = rag.ask("What is the PAN card for John Doe?")
print("Answer:", answer)
```

### Flask WebSocket UI Interface
To run the interactive web interface incorporating real-time token tracking:

```bash
python app.py
```
After executing, navigate to `http://localhost:5000` via web browser.

## Sample Data and Queries

To effectively test the Zero-Trust implementation, use the variations shown below. Notice the relationships across different datatypes.

### Indian Documentation Context
```text
KYC Verification Document
Name: Rajesh Kumar Sharma
Aadhaar: 2345-6789-0123
PAN: CDEFG5678H
Mobile: +91-98765-43210
Email: rajesh.sharma@mumbai.co.in
Address: Flat 301, Sunrise Apartments, Andheri West, Mumbai, Maharashtra 400058
```

### Potential Queries
- "What is the PAN and Aadhaar number for the employee with email rajesh.sharma@mumbai.co.in?"
- "What is the email address for Rajesh Kumar Sharma's manager?"
- "Show me all mobile numbers tied to the Mumbai branch."

## Audit Logging

Strict compliance requires non-repudiation tracking. Everything resides sequentially within the `audit.log` file formatting JSON rows directly:

```json
{"timestamp": "2026-04-16T10:30:00Z", "event_type": "ingest", "entity_type": "PERSON", "token": "<<PERSON_01_A1B2>>", "action": "mask"}
{"timestamp": "2026-04-16T10:30:01Z", "event_type": "ask", "status": "success", "details": {"pii_in_question": 1, "context_chunks": 3}}
```

## License
MIT License.