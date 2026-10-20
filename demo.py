"""
demo.py – Command-line walkthrough: ingest a record full of PII, then ask about it.

For each question it prints what the LLM was actually sent and what it replied, next to the
unmasked answer you get back. Uses a throwaway vault and vector store in a temp directory.

    python demo.py          # needs GROQ_API_KEY in .env
"""
import os
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    sys.exit("GROQ_API_KEY not found. Add it to .env (get one at https://console.groq.com/).")

from src import APIError, PIIVault, ZeroTrustRAG  # noqa: E402
from src.zero_trust_rag.audit_logger import AuditLogger  # noqa: E402

RECORD = """EMPLOYEE RECORD #1042
Name: Amit Sharma
Email: amit.sharma@corp.in
Phone: +91-98765-43210
PAN Card: ABCDE1234F
Aadhaar: 5425-2144-4943
Address: 123 Main Street, Mumbai, Maharashtra 400001
Project: Operation Phoenix. Reports to Priya Rai (+91-98700-12345)."""

QUESTIONS = [
    "What is Amit Sharma's PAN Card number?",
    "Who does the employee with Aadhaar 5425-2144-4943 report to?",
    "What is the phone number of the employee working on Operation Phoenix?",
]


def rule(title: str) -> None:
    print(f"\n{'-' * 72}\n{title}\n{'-' * 72}")


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="zt-demo-"))
    rule("Initializing (loads spaCy and the local embedding model)")
    bot = ZeroTrustRAG(
        persist_directory=str(workdir / "chroma"),
        vault=PIIVault(str(workdir / "vault.db")),
        audit=AuditLogger(str(workdir / "audit.log")),
    )

    rule("Ingesting")
    print(RECORD)
    result = bot.ingest(RECORD)
    print(f"\nStored as {result.chunks_created} chunk(s), {result.pii_redacted} values tokenized:\n")
    print(result.masked_text)

    for i, question in enumerate(QUESTIONS, 1):
        rule(f"Q{i}: {question}")
        try:
            answer = bot.ask(question)
        except APIError as e:
            print(f"LLM call failed: {e}")
            continue
        print(f"Sent to the LLM:  {answer.masked_question}")
        print(f"LLM replied:      {answer.masked_answer}")
        print(f"You see:          {answer.answer}")

    rule("Vault contents")
    for entity_type, count in bot.get_stats()["vault_stats"].items():
        print(f"  {entity_type}: {count}")
    print(f"\nEverything for this run is in {workdir}")


if __name__ == "__main__":
    main()
