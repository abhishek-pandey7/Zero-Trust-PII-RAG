"""
app.py – Flask web application with WebSocket support for real-time PII protection.
"""
import os
import sys
import tempfile
import threading
import time
import traceback
from datetime import UTC, datetime

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from flask_socketio import SocketIO, emit

from src import APIError, EgressBlockedError, ValidationError, ZeroTrustRAG, ZeroTrustRAGError
from src.zero_trust_rag.config import Config
from src.zero_trust_rag.utils import validate_input

load_dotenv()

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")


def _async_mode() -> str:
    """'eventlet' only when gunicorn's eventlet worker has monkey-patched the process, else threads.

    Hard-coding eventlet broke any start command that used plain threads, and vice versa.
    """
    try:
        import eventlet.patcher
        if eventlet.patcher.is_monkey_patched("socket"):
            return "eventlet"
    except ImportError:
        pass
    return "threading"


# SocketIO setup - for real-time updates
socketio = SocketIO(app, cors_allowed_origins="*", async_mode=_async_mode())

PREVIEW_CHARS = 1200

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


_started_at = time.time()
_warm_up = {"state": "pending", "attempts": 0, "last_error": None}
WARM_UP_ATTEMPTS = 6


def _warm_up_pipeline() -> None:
    """Load spaCy and the embedding model at boot, so the first visitor doesn't wait for them.

    Retries with backoff (10 s, 20 s, 40 s, ...): a host's first boot can hit a flaky network while
    fetching the model, and one failure used to leave the pipeline unloaded until someone noticed.
    Progress and the last error are reported by /api/health.
    """
    for attempt in range(1, WARM_UP_ATTEMPTS + 1):
        _warm_up.update(state="loading", attempts=attempt)
        try:
            get_rag().embedder.embed(["warm up"])
            _warm_up.update(state="ready", last_error=None)
            return
        except Exception as e:  # reported by /api/health instead of crashing the worker
            _warm_up["last_error"] = f"{type(e).__name__}: {e}"[:300]
            app.logger.exception("Pipeline warm-up attempt %d failed", attempt)
            time.sleep(10 * 2 ** (attempt - 1))
    _warm_up["state"] = "failed"


def _warm_up_stack() -> list[str]:
    """Where the warm-up thread is right now: file, line and function only, never variable values."""
    for thread in threading.enumerate():
        if thread.name == "warm-up" and thread.ident in sys._current_frames():
            frames = traceback.extract_stack(sys._current_frames()[thread.ident])
            return [f"{f.filename.rsplit('site-packages/', 1)[-1]}:{f.lineno} {f.name}" for f in frames[-12:]]
    return []


def _rss_mb() -> int | None:
    """Resident memory of this worker in MB (Linux only)."""
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1048576
    except (OSError, ValueError, AttributeError):
        return None


if os.getenv("ZT_EAGER_INIT", "1") == "1" and os.getenv("FLASK_ENV") != "testing":
    threading.Thread(target=_warm_up_pipeline, name="warm-up", daemon=True).start()


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _preview(text: str) -> str:
    """The first PREVIEW_CHARS characters, never ending inside a <<TOKEN>>."""
    if len(text) <= PREVIEW_CHARS:
        return text
    cut = PREVIEW_CHARS
    opened = text.rfind('<<', 0, cut)
    if opened != -1 and text.find('>>', opened, cut) == -1:
        cut = opened
    return text[:cut] + '...'


def _error(e: Exception, status: int):
    socketio.emit('error', {'message': str(e)})
    return jsonify({'success': False, 'error': str(e)}), status


def _ingest(text: str, source: str | None = None) -> dict:
    """Ingest text and report progress over the socket. Redaction runs exactly once."""
    validate_input(text)  # before loading anything: an oversized upload is refused, not an OOM kill
    bot = get_rag()
    socketio.emit('ingestion_start', {'text_length': len(text), 'source': source, 'timestamp': _now()})

    result = bot.ingest(text)

    if result.redaction_details:
        # Socket events go to every connected browser, so they carry types and tokens, never values
        socketio.emit('pii_detected', {
            'count': len(result.redaction_details),
            'entities': [{'entity_type': e['entity_type'], 'token': e['token']}
                         for e in result.redaction_details],
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
        validate_input(question)
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
    """Liveness check, plus whether the pipeline has finished loading and which commit is live."""
    return jsonify({
        'status': 'healthy',
        'timestamp': _now(),
        'rag_initialized': _rag is not None,
        'warm_up': dict(_warm_up),
        'uptime_seconds': int(time.time() - _started_at),
        'memory_mb': _rss_mb(),
        'version': (os.getenv('RENDER_GIT_COMMIT') or os.getenv('GIT_COMMIT') or 'dev')[:7],
        'async_mode': socketio.async_mode,
        **({'warm_up_stack': _warm_up_stack()} if request.args.get('trace') else {}),
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {'message': 'Connected to Zero-Trust PII RAG', 'timestamp': _now()})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    socketio.run(app, host='0.0.0.0', port=port, debug=debug, allow_unsafe_werkzeug=True)
