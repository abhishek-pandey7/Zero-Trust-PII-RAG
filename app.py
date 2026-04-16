"""
app.py – Flask web application with WebSocket support for real-time PII protection.
Deployed on Render (requires gunicorn with eventlet/gevent for production).
"""
import os
import time
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

from src import ZeroTrustRAG, ZeroTrustRAGError, ValidationError, APIError

load_dotenv()

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# SocketIO setup - for real-time updates
# Use eventlet for production on Render
socketio = SocketIO(app, 
    cors_allowed_origins="*",
    async_mode='threading' if os.getenv('FLASK_ENV') == 'development' else 'eventlet'
)

# Global RAG instance
rag = None

def get_rag():
    """Get or create RAG instance (lazy init with thread safety)."""
    global rag
    if rag is None:
        groq_key = os.getenv("GROQ_API_KEY")
        hf_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
        if not groq_key or not hf_key:
            raise ValueError("Missing API keys. Set GROQ_API_KEY and HUGGINGFACEHUB_API_TOKEN")
        rag = ZeroTrustRAG(groq_api_key=groq_key, hf_api_key=hf_key)
    return rag


@app.route('/')
def index():
    """Main page with the interactive UI."""
    return render_template('index.html')


@app.route('/api/ingest', methods=['POST'])
def api_ingest():
    """API endpoint for data ingestion with real-time updates."""
    try:
        data = request.json.get('data', '').strip()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        bot = get_rag()
        
        # Emit start event
        socketio.emit('ingestion_start', {
            'text_length': len(data),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
        # Perform ingestion
        result = bot.ingest(data)
        
        # Emit PII detected event
        if result.get('redaction_details'):
            socketio.emit('pii_detected', {
                'count': len(result['redaction_details']),
                'entities': result['redaction_details'],
                'timestamp': datetime.utcnow().isoformat() + 'Z'
            })
        
        # Emit completion event
        socketio.emit('ingestion_complete', {
            'chunks_created': result['chunks_created'],
            'redacted_count': result['pii_redacted'],
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
        # Get masked text from the vault for display
        redactor = bot.redactor
        masked_text, _ = redactor.redact(data, "display")
        
        return jsonify({
            'success': True,
            'original_text': data[:500] + ('...' if len(data) > 500 else ''),
            'masked_text': masked_text[:500] + ('...' if len(masked_text) > 500 else ''),
            'chunks_created': result['chunks_created'],
            'pii_redacted': result['pii_redacted'],
            'redaction_details': result['redaction_details']
        })
        
    except ValidationError as e:
        socketio.emit('error', {'message': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 400
    except ZeroTrustRAGError as e:
        socketio.emit('error', {'message': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        socketio.emit('error', {'message': f'Unexpected error: {e}'})
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/ask', methods=['POST'])
def api_ask():
    """API endpoint for querying with real-time streaming updates."""
    start_time = time.time()
    
    try:
        question = request.json.get('question', '').strip()
        
        if not question:
            return jsonify({
                'success': False,
                'error': 'No question provided'
            }), 400
        
        bot = get_rag()
        
        # Emit processing event
        socketio.emit('query_processing', {
            'question': question,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
        # Get masked question for display
        masked_q, redaction_log = bot.redactor.redact(question, "query")
        
        # Perform query
        answer = bot.ask(question)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Get stats
        stats = bot.get_stats()
        
        # Emit completion event
        socketio.emit('query_complete', {
            'answer': answer[:200] + ('...' if len(answer) > 200 else ''),
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        })
        
        return jsonify({
            'success': True,
            'original_question': question,
            'masked_question': masked_q,
            'answer': answer,
            'question_entities': redaction_log,
            'context_chunks': stats.get('vault_stats', {}),
            'processing_time': processing_time
        })
        
    except ValidationError as e:
        socketio.emit('error', {'message': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 400
    except APIError as e:
        socketio.emit('error', {'message': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 503
    except ZeroTrustRAGError as e:
        socketio.emit('error', {'message': str(e)})
        return jsonify({'success': False, 'error': str(e)}), 500
    except Exception as e:
        socketio.emit('error', {'message': f'Unexpected error: {e}'})
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def api_stats():
    """Get current pipeline statistics."""
    try:
        bot = get_rag()
        stats = bot.get_stats()
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint for Render."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'rag_initialized': rag is not None
    })


# WebSocket events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    emit('connected', {
        'message': 'Connected to Zero-Trust PII RAG',
        'timestamp': datetime.utcnow().isoformat() + 'Z'
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print('Client disconnected')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    # Use socketio.run instead of app.run for WebSocket support
    socketio.run(app, 
        host='0.0.0.0', 
        port=port, 
        debug=debug,
        allow_unsafe_werkzeug=True
    )
