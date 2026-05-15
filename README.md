# Zero-Trust PII RAG Pipeline

A production-ready RAG (Retrieval-Augmented Generation) pipeline that strictly masks PII (Personally Identifiable Information) before sending any data to external Application Programming Interfaces (APIs). External Large Language Models (LLMs) only see masked placeholders such as `<<PERSON_01_A1B2>>`. All unmasking happens exclusively within your secure local environment.

## Table of Contents
1. [Security Model](#security-model)
2. [Architecture Overview](#architecture-overview)
3. [Dashboard Features](#dashboard-features)
4. [Component Breakdown](#component-breakdown)
5. [Installation and Setup](#installation-and-setup)
6. [Configuration Parameters](#configuration-parameters)
7. [Supported PII Types](#supported-pii-types)
8. [Process Lifecycles](#process-lifecycles)
9. [Usage Instructions](#usage-instructions)
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
|  - Strict newline truncation for entity bounds    |
|  - Input sanitization (control characters)        |
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

## Dashboard Features

The system includes a modern, responsive web interface built with a custom design system for real-time visualization and management:

- **Pipeline View**: Real-time monitoring of document ingestion, PII detection, and RAG querying. Visualizes masked vs. unmasked text transitions.
- **Vault Explorer**: A secure administrative view of all token-to-PII mappings stored in the local SQLite database.
- **Compliance Audit Log**: Live stream of system operations and redaction events parsed from the JSON audit trail.
- **Security Policies**: Read-only introspection of the active system configuration, including LLM parameters, embedding models, and confidence thresholds.
- **Large-Scale Support**: Validated for documents exceeding 1,000,000 characters (e.g., bulk employee records).

## Component Breakdown

- `src/zero_trust_rag/pipeline.py`: Orchestrator class (`ZeroTrustRAG`) managing text splitting, vector storage, and Langchain integration.
- `src/zero_trust_rag/pii_redactor.py`: Implements `PIIRedactor` using `presidio-analyzer`. Features strict bound detection to prevent PII leakage across newline boundaries.
- `src/zero_trust_rag/vault.py`: Implements `PIIVault`, providing thread-safe SQLite persistence for deterministic token mapping.
- `src/zero_trust_rag/audit_logger.py`: Structured JSON logging for all security-relevant operations.
- `src/zero_trust_rag/utils.py`: Input validation, API key verification, and error handling decorators.
- `app.py`: Flask-SocketIO server presenting the reactive UI and WebSocket stream.

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

All pipeline settings are centralized within `config.yaml`.

### Configuration Sections

- **llm**: Define provider (`groq`), model selection (e.g., `llama-3.3-70b-versatile`), and temperature.
- **embeddings**: Model selection for vector encoding.
- **chroma**: Persistent storage directory for embeddings.
- **splitter**: Parameters for `RecursiveCharacterTextSplitter`.
- **presidio**: Active recognizers and confidence thresholds (default: `0.6`).
- **security**: Database paths and audit log locations.

## Supported PII Types

The default configuration intercepts a wide array of entities via SpaCy NER and custom regex recognizers:

- PERSON, EMAIL_ADDRESS, PHONE_NUMBER
- US_SSN, CREDIT_CARD, IBAN_CODE
- IP_ADDRESS, URL, LOCATION, DATE_TIME
- IN_PAN (Indian PAN Card), IN_AADHAAR (Indian Aadhaar Number)
- US_PASSPORT, US_DRIVER_LICENSE, MEDICAL_LICENSE

## Process Lifecycles

### The Ingestion Lifecycle
1. **Input Validation**: Enforces length constraints (up to 1,000,000 characters).
2. **Strict Bound Redaction**: Analyzes PII bounds and truncates at newlines to prevent cross-line contamination.
3. **Vault Storage**: Assigns unique cryptographic tokens for each unique PII value.
4. **Vector Alignment**: Stores masked chunks in ChromaDB via local embedding.

### The Querying Lifecycle
1. **Query Redaction**: Identifies PII in the user prompt and applies existing or new tokens.
2. **Semantic Retrieval**: Fetches relevant masked chunks from the local vector database.
3. **Remote Inference**: Sends masked context and query to the LLM endpoint (e.g., Groq).
4. **Local Restoration**: Resolves response tokens using the private Vault and returns the unmasked answer.

## Usage Instructions

### Web Interface
To run the interactive dashboard:
```bash
python app.py
```
Navigate to `http://localhost:5000`.

### Direct API Usage
```python
from src import ZeroTrustRAG
import os

rag = ZeroTrustRAG(groq_api_key=os.getenv("GROQ_API_KEY"))
rag.ingest("Employee John Doe (Aadhaar: 1234-5678-9012) joined today.")
answer = rag.ask("What is John Doe's Aadhaar?")
print(answer)
```

## Audit Logging

Operations are recorded in `audit.log` as sequential JSON lines for SIEM integration:
```json
{"timestamp": "2026-05-15T21:00:00Z", "event_type": "redaction", "entity_type": "PERSON", "token": "<<PERSON_01_A1B2>>", "action": "mask"}
```

