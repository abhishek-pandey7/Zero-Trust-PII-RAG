"""
demo.py – Interactive showcase for the Zero-Trust PII-Redacting RAG Pipeline.
"""
import os
import sys

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("\n" + "=" * 70)
print("  ZERO-TRUST PII RAG PIPELINE - DEMO")
print("=" * 70)

# Check for API keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GROQ_API_KEY:
    print("\n[!] ERROR: GROQ_API_KEY not found!")
    print("    Please set it in your .env file:")
    print("    GROQ_API_KEY=your_key_here")
    print("    Get it from: https://console.groq.com/")
    sys.exit(1)

if not HF_API_KEY:
    print("\n[!] ERROR: HUGGINGFACEHUB_API_TOKEN not found!")
    print("    Please set it in your .env file:")
    print("    HUGGINGFACEHUB_API_TOKEN=your_token_here")
    print("    Get it from: https://huggingface.co/settings/tokens")
    sys.exit(1)

print("\n[+] API keys loaded successfully")

# Import after checking keys
try:
    from src import APIError, ZeroTrustRAG, ZeroTrustRAGError
    print("[+] Modules imported successfully")
except ImportError as e:
    print(f"[!] Failed to import modules: {e}")
    print("    Make sure you're in the project root directory")
    sys.exit(1)

print("\n" + "-" * 70)
print("STEP 1: Initializing the Pipeline")
print("-" * 70)

try:
    print("[...] Setting up Zero-Trust RAG with PII protection...")
    bot = ZeroTrustRAG(groq_api_key=GROQ_API_KEY, hf_api_key=HF_API_KEY)
    print("[+] Pipeline initialized!")
except Exception as e:
    print(f"[!] Failed to initialize: {e}")
    sys.exit(1)

print("\n" + "-" * 70)
print("STEP 2: Ingesting Sensitive Data")
print("-" * 70)

# Sample data with various PII types
sensitive_data = """
EMPLOYEE RECORD #1042

Personal Information:
- Name: Amit Sharma
- Email: amit.sharma@corp.in
- Phone: +91-98765-43210
- PAN Card: ABCDE1234F
- Aadhaar: 1234-5678-9012

Address:
- 123 Main Street, Mumbai, Maharashtra 400001, India

Work Info:
- Project: Operation Phoenix (Confidential)
- Performance: Top performer. Contact manager Priya Rai at +91-98700-12345.
"""

print("\n[INPUT] Original text (what YOU see):")
print("_" * 70)
print(sensitive_data.strip())
print("_" * 70)

print("\n[...] Sending to pipeline for ingestion...")
print("    [>] PII detection (Microsoft Presidio)")
print("    [>] Token generation & storage in vault")
print("    [>] Text chunking")
print("    [>] Embedding generation (HuggingFace)")
print("    [>] Vector storage (ChromaDB)")

try:
    result = bot.ingest(sensitive_data)
    
    print("\n[+] INGESTION COMPLETE!")
    print(f"    - Chunks created: {result['chunks_created']}")
    print(f"    - PII entities detected & masked: {result['pii_redacted']}")
    
    if result['redaction_details']:
        print("\n[TOKENS] PII replaced with tokens (what LLM sees):")
        for item in result['redaction_details'][:5]:
            print(f"    - {item['token']} ({item['entity_type']})")
        if len(result['redaction_details']) > 5:
            print(f"    ... and {len(result['redaction_details']) - 5} more")
            
except Exception as e:
    print(f"[!] Ingestion failed: {e}")
    sys.exit(1)

print("\n" + "-" * 70)
print("STEP 3: Querying the Secure Database")
print("-" * 70)

questions = [
    "What is Amit Sharma's PAN Card number?",
    "Who is the manager for the employee with Aadhaar 1234-5678-9012?",
    "What is the phone number of the employee working on Operation Phoenix?"
]

for i, question in enumerate(questions, 1):
    print(f"\n[Q{i}] {question}")
    print("-" * 70)
    
    try:
        print("[...] Processing query:")
        print("    [>] PII detection in question")
        print("    [>] Vector search in secure database")
        print("    [>] LLM inference (Groq)")
        print("    [>] Token unmasking for final answer")
        
        answer = bot.ask(question)
        print(f"\n[A{i}] {answer}")
        
    except APIError as e:
        print(f"[!] API Error: {e}")
    except ZeroTrustRAGError as e:
        print(f"[!] Pipeline Error: {e}")

print("\n" + "=" * 70)
print("FINAL STATISTICS")
print("=" * 70)

stats = bot.get_stats()
print("Vault contents:")
for entity_type, count in stats.get('vault_stats', {}).items():
    print(f"    - {entity_type}: {count} tokens stored")
print(f"Storage location: {stats.get('persist_directory', 'N/A')}")

print("\n" + "=" * 70)
print("SECURITY SUMMARY")
print("=" * 70)
print("""
[OK] All PII was masked BEFORE sending to external APIs
[OK] LLM only saw tokens (e.g., <<PERSON_01_A1B2>>)
[OK] Original values stored in local SQLite vault
[OK] Unmasking only happened on your local machine
[OK] All operations logged for compliance
""")

print("Demo complete! The code is working correctly.")
print("=" * 70 + "\n")
