"""
app.py – Flask web application with WebSocket support for real-time PII protection.
"""
import os
import tempfile
import threading
import time
from datetime import UTC, datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from src import APIError, EgressBlockedError, ValidationError, ZeroTrustRAG, ZeroTrustRAGError
from src.zero_trust_rag.config import Config

load_dotenv()

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# SocketIO setup - for real-time updates
socketio = SocketIO(app,
    cors_allowed_origins="*",
    async_mode='threading' if os.getenv('FLASK_ENV') == 'development' else 'eventlet'
)

PREVIEW_CHARS = 500

_rag: ZeroTrustRAG | None = None
_rag_lock = threading.Lock()


def get_rag() -> ZeroTrustRAG:
    """The shared pipeline, built on first use.

    Double-checked locking: without the lock, two concurrent first requests could each build a
    pipeline (two spaCy models and two embedding sessions in memory).
    """
    global _rag
    if _rag is None:
        with _rag_lock:
            if _rag is None:
                _rag = ZeroTrustRAG(groq_api_key=os.getenv("GROQ_API_KEY"))
    return _rag


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _preview(text: str) -> str:
    return text[:PREVIEW_CHARS] + ('...' if len(text) > PREVIEW_CHARS else '')


def _error(e: Exception, status: int):
    socketio.emit('error', {'message': str(e)})
    return jsonify({'success': False, 'error': str(e)}), status


def _ingest(text: str, source: str | None = None) -> dict:
    """Ingest text and report progress over the socket. Redaction runs exactly once."""
    bot = get_rag()
    socketio.emit('ingestion_start', {'text_length': len(text), 'source': source, 'timestamp': _now()})

    result = bot.ingest(text)

    if result.redaction_details:
        socketio.emit('pii_detected', {
            'count': len(result.redaction_details),
            'entities': result.redaction_details,
            'timestamp': _now(),
        })
    socketio.emit('ingestion_complete', {
        'chunks_created': result.chunks_created,
        'redacted_count': result.pii_redacted,
        'source': source,
        'timestamp': _now(),
    })
    return {
        'success': True,
        'original_text': _preview(text),
        'masked_text': _preview(result.masked_text),
        'chunks_created': result.chunks_created,
        'pii_redacted': result.pii_redacted,
        'redaction_details': result.redaction_details,
    }


@app.route('/')
def index():
    """Main page with the interactive UI."""
    return render_template('index.html')


@app.route('/api/ingest', methods=['POST'])
def api_ingest():
    """API endpoint for data ingestion with real-time updates."""
    data = ((request.get_json(silent=True) or {}).get('data') or '').strip()
    if not data:
        return jsonify({'success': False, 'error': 'No data provided'}), 400
    try:
        return jsonify(_ingest(data))
    except ValidationError as e:
        return _error(e, 400)
    except ZeroTrustRAGError as e:
        return _error(e, 500)


def extract_text_from_file(filepath, filename):
    """Extract text from uploaded files (PDF, DOCX, TXT)."""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''

    if ext == 'txt':
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            return f.read()

    elif ext == 'pdf':
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(filepath)
            text = ''
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except ImportError:
            raise ValueError("PDF support requires PyMuPDF. Install with: pip install PyMuPDF") from None

    elif ext in ('docx', 'doc'):
        try:
            from docx import Document
            doc = Document(filepath)
            return '\n'.join([p.text for p in doc.paragraphs])
        except ImportError:
            raise ValueError("DOCX support requires python-docx. Install with: pip install python-docx") from None

    else:
        raise ValueError(f"Unsupported file type: .{ext}. Supported: .txt, .pdf, .docx")


@app.route('/api/upload', methods=['POST'])
def api_upload():
    """Upload a document file for ingestion."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    with tempfile.NamedTemporaryFile(delete=False, suffix='_' + file.filename) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name
    try:
        text = extract_text_from_file(tmp_path, file.filename)
        if not text or not text.strip():
            return jsonify({'success': False, 'error': 'No text content found in file'}), 400
        response = _ingest(text, source=file.filename)
        response.update({'filename': file.filename, 'char_count': len(text)})
        return jsonify(response)
    except ValueError as e:
        return jsonify({'success': False, 'error': str(e)}), 400
    except ZeroTrustRAGError as e:
        return _error(e, 500)
    finally:
        os.unlink(tmp_path)


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for querying with real-time streaming updates."""
    start_time = time.time()
    question = ((request.get_json(silent=True) or {}).get('question') or '').strip()
    if not question:
        return jsonify({'success': False, 'error': 'No question provided'}), 400

    try:
        bot = get_rag()
        socketio.emit('query_processing', {'timestamp': _now()})
        result = bot.ask(question)
        socketio.emit('query_complete', {'timestamp': _now()})
    except ValidationError as e:
        return _error(e, 400)
    except EgressBlockedError as e:
        return _error(e, 422)
    except APIError as e:
        return _error(e, 503)
    except ZeroTrustRAGError as e:
        return _error(e, 500)

    return jsonify({
        'success': True,
        'original_question': question,
        'masked_question': result.masked_question,
        'answer': result.answer,
        'masked_answer': result.masked_answer,
        'question_entities': result.question_entities,
        'context_chunks': result.context_chunks,
        'processing_time': int((time.time() - start_time) * 1000),
    })


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get current pipeline statistics."""
    return jsonify({'success': True, 'stats': get_rag().get_stats()})


@app.route('/api/vault', methods=['GET'])
def api_vault():
    """Get all token mappings from the vault."""
    return jsonify({'success': True, 'mappings': get_rag().vault.get_all_mappings()})


@app.route('/api/audit', methods=['GET'])
def api_audit():
    """Get recent audit logs."""
    return jsonify({'success': True, 'logs': get_rag().audit.get_recent_logs(limit=50)})


@app.route('/api/policies', methods=['GET'])
def api_policies():
    """Get current security policies/config."""
    cfg = Config()
    return jsonify({
        'success': True,
        'policies': {
            "llm": {"model": cfg.get("llm.model"), "temperature": cfg.get("llm.temperature")},
            "embeddings": {"provider": cfg.get("embeddings.provider"), "model": cfg.get("embeddings.model")},
            "security": {
                # this used to read security.score_threshold, which doesn't exist, and showed null
                "score_threshold": cfg.get("presidio.score_threshold"),
                "entity_types": cfg.get("presidio.entity_types"),
                "vault_path": cfg.get("security.vault_db_path"),
                "audit_path": cfg.get("security.audit_log_path"),
            },
        },
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Liveness check."""
    return jsonify({'status': 'healthy', 'timestamp': _now(), 'rag_initialized': _rag is not None})


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'message': 'Connected to Zero-Trust PII RAG', 'timestamp': _now()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
