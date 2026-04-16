"""
main.py – Entry point for Zero-Trust PII RAG Pipeline.
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=" * 60)
print("Zero-Trust PII RAG Pipeline")
print("=" * 60)

# Check for API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY:
    print("\n[ERROR] GROQ_API_KEY not found!")
    print("Please create a .env file with:")
    print("  GROQ_API_KEY=your_key_here")
    print("Get your key from: https://console.groq.com/")
    exit(1)

if not HF_API_KEY:
    print("\n[ERROR] HUGGINGFACEHUB_API_TOKEN not found!")
    print("Please create a .env file with:")
    print("  HUGGINGFACEHUB_API_TOKEN=your_token_here")
    print("Get your token from: https://huggingface.co/settings/tokens")
    exit(1)

print("[OK] API keys found")

# Import after checking keys
from src import ZeroTrustRAG

print("\n[INIT] Initializing pipeline...")
try:
    llm = ZeroTrustRAG(groq_api_key=GROQ_API_KEY, hf_api_key=HF_API_KEY)
    print("[OK] Pipeline initialized\n")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    exit(1)

# Sample data with PII
data = """
Employee Record #1042
Name: John Doe
Email: john.doe@example.com
Phone: +91-98765-43210
PAN Card: ABCDE1234F
Aadhaar: 1234-5678-9012
Address: 123 Main Street, Mumbai, India
"""

print("[INGEST] Ingesting data:")
print("-" * 60)
print(data)
print("-" * 60)

try:
    ingest_result = llm.ingest(data)
    print(f"[OK] Data ingested successfully!")
    print(f"  - Chunks created: {ingest_result['chunks_created']}")
    print(f"  - PII entities masked: {ingest_result['pii_redacted']}")
    
    if ingest_result['redaction_details']:
        print(f"\n[TOKENS] Tokens created:")
        for item in ingest_result['redaction_details']:
            print(f"  - {item['token']} ({item['entity_type']})")
except Exception as e:
    print(f"[ERROR] Ingestion failed: {e}")
    exit(1)

# Ask a question
print("\n" + "=" * 60)
question = "What is John Doe's PAN card number?"
print(f"[QUERY] {question}")
print("=" * 60)

try:
    result = llm.ask(question)
    print(f"\n[ANSWER] {result}")
except Exception as e:
    print(f"[ERROR] Query failed: {e}")
    exit(1)

# Show stats
print("\n" + "=" * 60)
print("[STATS] Pipeline Statistics")
print("=" * 60)
stats = llm.get_stats()
print(f"  - Vault entries: {stats.get('vault_stats', {})}")
print(f"  - Data stored in: {stats.get('persist_directory', 'N/A')}")

print("\n[DONE] Complete!")
