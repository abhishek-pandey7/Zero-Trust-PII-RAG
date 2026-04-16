/**
 * Zero-Trust PII RAG – Frontend
 */

const socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000
});

let stats = {
    ingested: 0,
    redacted: 0,
    queries: 0,
    chunks: 0
};

let isIngesting = false;
let isAsking = false;

const activityFeed = document.getElementById('activity-feed');
const statusDot = document.getElementById('connection-status');
const statusText = document.getElementById('status-text');
const ingestBtn = document.getElementById('ingest-btn');
const queryBtn = document.getElementById('query-btn');
const modal = document.getElementById('result-modal');

socket.on('connect', () => {
    statusDot.className = 'status-dot connected';
    statusText.textContent = 'Connected';
    showToast('Connected', 'success');
});

socket.on('disconnect', () => {
    statusDot.className = 'status-dot';
    statusText.textContent = 'Disconnected';
});

socket.on('connect_error', () => {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Error';
});

socket.on('ingestion_start', (data) => {
    addActivity({
        type: 'warning',
        title: 'Ingestion started',
        text: `Processing ${data.text_length} characters`,
        timestamp: data.timestamp
    });
});

socket.on('pii_detected', (data) => {
    const entityList = data.entities.map(e => 
        `<span class="token">${escapeHtml(e.original_value)}</span>`
    ).join(', ');
    
    addActivity({
        type: 'info',
        title: 'PII detected',
        text: `${data.count} entities: ${entityList}`,
        timestamp: data.timestamp
    });
});

socket.on('ingestion_complete', (data) => {
    stats.ingested++;
    stats.redacted += data.redacted_count;
    stats.chunks += data.chunks_created;
    updateStats();
    
    addActivity({
        type: 'success',
        title: 'Ingestion complete',
        text: `${data.chunks_created} chunks created, ${data.redacted_count} PII masked`,
        timestamp: data.timestamp
    });
    
    showToast('Data ingested', 'success');
});

socket.on('query_processing', (data) => {
    addActivity({
        type: 'warning',
        title: 'Query processing',
        text: `"${data.question}"`,
        timestamp: data.timestamp
    });
});

socket.on('query_complete', (data) => {
    stats.queries++;
    updateStats();
    
    addActivity({
        type: 'success',
        title: 'Answer generated',
        text: data.answer.substring(0, 100) + (data.answer.length > 100 ? '...' : ''),
        timestamp: data.timestamp
    });
});

socket.on('error', (data) => {
    addActivity({
        type: 'error',
        title: 'Error',
        text: data.message,
        timestamp: new Date().toISOString()
    });
    
    showToast(data.message, 'error');
});

async function ingestData() {
    const textarea = document.getElementById('ingest-data');
    const data = textarea.value.trim();
    
    if (!data) {
        showToast('Enter data to ingest', 'warning');
        return;
    }
    
    if (isIngesting) return;
    
    isIngesting = true;
    ingestBtn.disabled = true;
    ingestBtn.innerHTML = '<span class="spinner"></span> Processing';
    
    try {
        const response = await fetch('/api/ingest', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ data })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showComparisonModal('Ingestion Result', result);
            textarea.value = '';
        } else {
            showToast(result.error || 'Ingestion failed', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    } finally {
        isIngesting = false;
        ingestBtn.disabled = false;
        ingestBtn.innerHTML = 'Ingest Data';
    }
}

async function askQuestion() {
    const input = document.getElementById('query-input');
    const question = input.value.trim();
    
    if (!question) {
        showToast('Enter a question', 'warning');
        return;
    }
    
    if (isAsking) return;
    
    isAsking = true;
    queryBtn.disabled = true;
    queryBtn.innerHTML = '<span class="spinner"></span>';
    
    try {
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question })
        });
        
        const result = await response.json();
        
        if (result.success) {
            showResultModal('Answer', result);
            input.value = '';
        } else {
            showToast(result.error || 'Query failed', 'error');
        }
    } catch (error) {
        showToast('Network error: ' + error.message, 'error');
    } finally {
        isAsking = false;
        queryBtn.disabled = false;
        queryBtn.innerHTML = 'Submit';
    }
}

function addActivity({ type, title, text, timestamp }) {
    if (activityFeed.querySelector('.empty-state')) {
        activityFeed.innerHTML = '';
    }
    
    const item = document.createElement('div');
    item.className = `activity-item ${type}`;
    
    const time = new Date(timestamp).toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false
    });
    
    item.innerHTML = `
        <div class="activity-title">${title}</div>
        <div class="activity-text">${text}</div>
        <div class="activity-meta">
            <span class="activity-time">${time}</span>
        </div>
    `;
    
    activityFeed.insertBefore(item, activityFeed.firstChild);
    
    while (activityFeed.children.length > 30) {
        activityFeed.removeChild(activityFeed.lastChild);
    }
}

function updateStats() {
    document.getElementById('stat-ingested').textContent = stats.ingested;
    document.getElementById('stat-redacted').textContent = stats.redacted;
    document.getElementById('stat-queries').textContent = stats.queries;
    document.getElementById('stat-chunks').textContent = stats.chunks;
}

function showComparisonModal(title, data) {
    const body = document.getElementById('modal-body');
    document.getElementById('modal-title').textContent = title;
    
    const entityList = data.redaction_details?.map(e => 
        `<li><span class="token">${e.token}</span> ← ${escapeHtml(e.original_value)} <span style="color: var(--text-muted); font-size: 0.75rem;">(${e.entity_type})</span></li>`
    ).join('') || '<li>No PII detected</li>';
    
    body.innerHTML = `
        <div class="result-section">
            <h4>Original vs Masked</h4>
            <div class="comparison">
                <div class="comparison-column">
                    <h5>Original</h5>
                    <div class="result-box">${escapeHtml(data.original_text)}</div>
                </div>
                <div class="comparison-column masked">
                    <h5>Masked</h5>
                    <div class="result-box">${escapeHtml(data.masked_text)}</div>
                </div>
            </div>
        </div>
        <div class="result-section">
            <h4>PII Tokens</h4>
            <div class="result-box">
                <ul style="list-style: none; padding-left: 0; margin: 0;">
                    ${entityList}
                </ul>
            </div>
        </div>
        <div class="result-section">
            <h4>Stats</h4>
            <div style="display: flex; gap: 24px;">
                <div>
                    <span style="font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">Chunks</span>
                    <div style="font-family: 'DM Mono', monospace; font-size: 1.25rem; font-weight: 600;">${data.chunks_created}</div>
                </div>
                <div>
                    <span style="font-size: 0.6875rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted);">PII Masked</span>
                    <div style="font-family: 'DM Mono', monospace; font-size: 1.25rem; font-weight: 600;">${data.pii_redacted}</div>
                </div>
            </div>
        </div>
    `;
    
    modal.classList.add('active');
}

function showResultModal(title, data) {
    const body = document.getElementById('modal-body');
    document.getElementById('modal-title').textContent = title;
    
    const questionTokens = data.question_entities?.map(e => 
        `<span class="token">${e.token}</span>`
    ).join(', ') || 'None';
    
    body.innerHTML = `
        <div class="result-section">
            <h4>Question</h4>
            <div class="result-box">${escapeHtml(data.original_question)}</div>
        </div>
        <div class="result-section">
            <h4>Answer</h4>
            <div class="result-box answer">${escapeHtml(data.answer)}</div>
        </div>
        <div class="result-section">
            <h4>Details</h4>
            <div class="result-box">
                <p style="margin: 0 0 12px 0;"><strong>PII in question:</strong> ${questionTokens}</p>
                <p style="margin: 0 0 12px 0;"><strong>Context chunks:</strong> ${data.context_chunks}</p>
                <p style="margin: 0;"><strong>Time:</strong> ${data.processing_time}ms</p>
            </div>
        </div>
        <div class="result-section">
            <h4>Masked Question</h4>
            <div class="result-box" style="opacity: 0.8;">${escapeHtml(data.masked_question)}</div>
        </div>
    `;
    
    modal.classList.add('active');
}

function closeModal() {
    modal.classList.remove('active');
}

function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    
    const titles = {
        success: 'Success',
        error: 'Error',
        warning: 'Warning',
        info: 'Info'
    };
    
    toast.innerHTML = `
        <div class="toast-title">${titles[type]}</div>
        <div class="toast-message">${message}</div>
    `;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.opacity = '0';
        toast.style.transform = 'translateX(40px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

modal.addEventListener('click', (e) => {
    if (e.target === modal) closeModal();
});

window.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && e.shiftKey === false) {
        if (document.activeElement.id === 'query-input') {
            e.preventDefault();
            askQuestion();
        }
    }
});
