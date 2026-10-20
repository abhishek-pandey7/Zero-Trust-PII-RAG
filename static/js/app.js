/**
 * Zero-Trust PII RAG — Frontend Logic
 * Skills: taste-design, emildesign (Emil Kowalski), framermotion
 *
 * Key principles applied:
 * - Motion via CDN `window.Motion.animate` for spring physics
 * - Hardware-accelerated transforms only (never top/left/width/height)
 * - Emil: ease-out for entries, never ease-in for UI
 * - Emil: buttons scale(0.97) on :active (CSS), blur during loading
 * - Stagger delays 40-60ms between items
 * - Reduced motion respected
 * - No emojis anywhere
 */

(function () {
  'use strict';

  // ── Socket Setup ──────────────────────────────────────
  const socket = io({
    transports: ['websocket', 'polling'],
    reconnection: true,
    reconnectionAttempts: 5,
    reconnectionDelay: 1000
  });

  let stats = { ingested: 0, redacted: 0, queries: 0, chunks: 0 };
  let isIngesting = false;
  let isAsking = false;
  let isUploading = false;

  // ── DOM References ────────────────────────────────────
  const $ = (id) => document.getElementById(id);
  const activityFeed = $('activity-feed');
  const statusDot = $('connection-status');
  const statusText = $('status-text');
  const ingestBtn = $('ingest-btn');
  const queryBtn = $('query-btn');
  const uploadBtn = $('upload-btn');
  const modal = $('result-modal');
  const modalBackdrop = $('modal-backdrop');
  const modalCloseBtn = $('modal-close-btn');
  const fileInput = $('file-input');
  const uploadZone = $('upload-zone');
  const uploadFilename = $('upload-filename');
  const filenameText = $('filename-text');
  const uploadShimmer = $('upload-shimmer');

  // ── Socket Events ─────────────────────────────────────

  socket.on('connect', () => {
    statusDot.className = 'conn-dot connected';
    statusText.textContent = 'CONNECTED';
    showToast('Secure pipeline established.', 'success');
  });

  socket.on('disconnect', () => {
    statusDot.className = 'conn-dot error';
    statusText.textContent = 'DISCONNECTED';
  });

  socket.on('connect_error', () => {
    statusDot.className = 'conn-dot error';
    statusText.textContent = 'ERROR';
  });

  socket.on('ingestion_start', (data) => {
    const source = data.source ? ` from ${data.source}` : '';
    addActivity({
      type: 'warning',
      title: 'Ingestion Protocol Initiated',
      text: `Processing ${data.text_length.toLocaleString()} characters${source}`,
      timestamp: data.timestamp
    });
  });

  socket.on('pii_detected', (data) => {
    // Values never travel over the socket; summarise by type instead
    const counts = {};
    data.entities.forEach((e) => { counts[e.entity_type] = (counts[e.entity_type] || 0) + 1; });
    const entityList = Object.entries(counts)
      .map(([type, n]) => `<span class="token">${escapeHtml(type)} &times;${n}</span>`)
      .join(' ');
    addActivity({
      type: 'info',
      title: 'PII Intercepted',
      text: `${data.count} entities identified and masked: ${entityList}`,
      timestamp: data.timestamp
    });
  });

  socket.on('ingestion_complete', (data) => {
    stats.ingested++;
    stats.redacted += data.redacted_count;
    stats.chunks += data.chunks_created;
    updateStats();
    const source = data.source ? ` (${data.source})` : '';
    addActivity({
      type: 'success',
      title: 'Redaction Complete',
      text: `${data.chunks_created} secure chunks created, ${data.redacted_count} PII tokens masked${source}`,
      timestamp: data.timestamp
    });
    showToast('Dataset ingested securely.', 'success');
  });

  socket.on('query_processing', (data) => {
    addActivity({
      type: 'warning',
      title: 'Executing LLM Query',
      text: `Transmitting masked prompt to Groq endpoint`,
      timestamp: data.timestamp
    });
  });

  socket.on('query_complete', (data) => {
    stats.queries++;
    updateStats();
    addActivity({
      type: 'success',
      title: 'Query Resolved',
      text: `Response received and unmasked locally`,
      timestamp: data.timestamp
    });
  });

  socket.on('error', (data) => {
    addActivity({
      type: 'error',
      title: 'System Fault',
      text: data.message,
      timestamp: new Date().toISOString()
    });
    showToast(data.message, 'error');
    resetButtons();
  });

  // ── User Actions ──────────────────────────────────────

  ingestBtn.addEventListener('click', ingestData);
  queryBtn.addEventListener('click', askQuestion);
  uploadBtn.addEventListener('click', uploadFile);
  modalBackdrop.addEventListener('click', closeModal);
  modalCloseBtn.addEventListener('click', closeModal);

  // ── Navigation ────────────────────────────────────────

  const navLinks = {
    'nav-pipeline': 'view-pipeline',
    'nav-policies': 'view-policies',
    'nav-vault': 'view-vault',
    'nav-audit': 'view-audit'
  };

  Object.entries(navLinks).forEach(([navId, viewId]) => {
    $(navId).addEventListener('click', (e) => {
      e.preventDefault();
      switchView(navId, viewId);
    });
  });

  function switchView(activeNavId, activeViewId) {
    // Update nav links
    Object.keys(navLinks).forEach(id => {
      $(id).classList.toggle('active', id === activeNavId);
    });

    // Update views
    Object.values(navLinks).forEach(id => {
      $(id).style.display = id === activeViewId ? 'block' : 'none';
    });

    // Fetch data if needed
    if (activeViewId === 'view-vault') fetchVault();
    if (activeViewId === 'view-audit') fetchAudit();
    if (activeViewId === 'view-policies') fetchPolicies();

    // Emil: Subtle entry animation for the card
    const activeView = $(activeViewId);
    if (window.Motion && window.Motion.animate) {
      window.Motion.animate(
        activeView,
        { opacity: [0, 1], y: [10, 0] },
        { duration: 0.4, easing: [0.23, 1, 0.32, 1] }
      );
    }
  }

  async function fetchVault() {
    const tableBody = $('vault-table-body');
    tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 40px; opacity: 0.5;">Accessing secure vault...</td></tr>';

    try {
      const res = await fetch('/api/vault');
      const data = await res.json();
      if (data.success) {
        if (data.mappings.length === 0) {
          tableBody.innerHTML = '<tr><td colspan="4" style="text-align:center; padding: 40px; opacity: 0.5;">Vault is empty. Ingest data to see mappings.</td></tr>';
          return;
        }

        tableBody.innerHTML = data.mappings.map(m => `
          <tr>
            <td><code class="token">${escapeHtml(m.token)}</code></td>
            <td><span class="badge">${escapeHtml(m.entity_type)}</span></td>
            <td>${escapeHtml(m.real_value)}</td>
            <td><span class="time-stamp">${new Date(m.created_at).toLocaleString()}</span></td>
          </tr>
        `).join('');
      }
    } catch (err) {
      showToast('Failed to fetch vault data.', 'error');
    }
  }

  async function fetchAudit() {
    const container = $('audit-log-content');
    container.innerHTML = '<div style="text-align:center; padding: 40px; opacity: 0.5;">Retrieving audit trail...</div>';

    try {
      const res = await fetch('/api/audit');
      const data = await res.json();
      if (data.success) {
        if (data.logs.length === 0) {
          container.innerHTML = '<div style="text-align:center; padding: 40px; opacity: 0.5;">No audit logs found.</div>';
          return;
        }

        container.innerHTML = data.logs.map(log => {
          const type = log.event_type || 'info';
          const statusClass = log.status === 'error' ? 'audit-error' : (log.action === 'mask' ? 'audit-mask' : 'audit-info');
          
          let details = '';
          if (log.details) {
            details = `<div class="audit-details">${escapeHtml(JSON.stringify(log.details))}</div>`;
          } else if (log.token) {
            // Tokens look like <<PERSON_01_ab12>>, which the browser would parse as a tag unless escaped
            details = `<div class="audit-token">${escapeHtml(log.entity_type || '')} &rarr; <code>${escapeHtml(log.token)}</code></div>`;
          }

          return `
            <div class="audit-entry">
              <div class="audit-header">
                <span class="audit-type ${statusClass}">${type.toUpperCase()}</span>
                <span class="audit-time">${new Date(log.timestamp).toLocaleString()}</span>
              </div>
              <div class="audit-body">
                ${log.status === 'error' ? `<div class="audit-msg error">${escapeHtml(log.details.error || 'Unknown error')}</div>` : ''}
                ${details}
              </div>
            </div>
          `;
        }).join('');
      }
    } catch (err) {
      showToast('Failed to fetch audit logs.', 'error');
    }
  }

  async function fetchPolicies() {
    const container = $('policies-content');
    container.innerHTML = '<div style="text-align:center; padding: 40px; opacity: 0.5;">Loading configuration...</div>';

    try {
      const res = await fetch('/api/policies');
      const data = await res.json();
      if (data.success) {
        const p = data.policies;
        container.innerHTML = `
          <div class="policy-group">
            <h5>LLM Provider (Groq)</h5>
            <div class="policy-item"><span>Model</span> <strong>${escapeHtml(String(p.llm.model))}</strong></div>
            <div class="policy-item"><span>Temperature</span> <strong>${escapeHtml(String(p.llm.temperature))}</strong></div>
          </div>
          <div class="policy-group">
            <h5>Embedding Engine (HuggingFace)</h5>
            <div class="policy-item"><span>Model</span> <strong>${escapeHtml(String(p.embeddings.model))}</strong></div>
          </div>
          <div class="policy-group">
            <h5>Redaction Heuristics (Presidio)</h5>
            <div class="policy-item"><span>Confidence Threshold</span> <strong>${escapeHtml(String(p.security.score_threshold))}</strong></div>
          </div>
          <div class="policy-group">
            <h5>Storage Persistence</h5>
            <div class="policy-item"><span>Vault DB</span> <code>${escapeHtml(String(p.security.vault_path))}</code></div>
            <div class="policy-item"><span>Audit Log</span> <code>${escapeHtml(String(p.security.audit_path))}</code></div>
          </div>
        `;
      }
    } catch (err) {
      showToast('Failed to fetch policies.', 'error');
    }
  }

  async function ingestData() {
    if (isIngesting) return;
    const textarea = $('ingest-data');
    const text = textarea.value.trim();
    if (!text) {
      showToast('Payload is empty. Paste text first.', 'warning');
      return;
    }

    isIngesting = true;
    setButtonLoading(ingestBtn, true);

    try {
      const res = await fetch('/api/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ data: text })
      });
      const d = await res.json();
      if (!res.ok) {
        throw new Error(d.error || 'Ingestion failed');
      }
      textarea.value = '';
      showIngestModal(d);
    } catch (err) {
      showToast(err.message, 'error');
      addActivity({
        type: 'error',
        title: 'Ingestion Failed',
        text: err.message,
        timestamp: new Date().toISOString()
      });
    } finally {
      isIngesting = false;
      setButtonLoading(ingestBtn, false);
    }
  }

  async function askQuestion() {
    if (isAsking) return;
    const input = $('query-input');
    const question = input.value.trim();
    if (!question) {
      showToast('Query string is empty.', 'warning');
      return;
    }

    isAsking = true;
    setButtonLoading(queryBtn, true);

    try {
      const res = await fetch('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question })
      });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.error || 'Query failed');
      }
      const data = await res.json();
      if (data.success) {
        showResultModal('Resolution Report', data);
      }
      input.value = '';
    } catch (err) {
      showToast(err.message, 'error');
      addActivity({
        type: 'error',
        title: 'Query Failed',
        text: err.message,
        timestamp: new Date().toISOString()
      });
    } finally {
      isAsking = false;
      setButtonLoading(queryBtn, false);
    }
  }

  // ── File Upload ───────────────────────────────────────

  uploadZone.addEventListener('click', (e) => {
    if (e.target === fileInput) return;
    fileInput.click();
  });

  ['dragenter', 'dragover'].forEach((evt) => {
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.classList.add('drag-over');
    });
  });

  ['dragleave', 'drop'].forEach((evt) => {
    uploadZone.addEventListener(evt, (e) => {
      e.preventDefault();
      uploadZone.classList.remove('drag-over');
    });
  });

  uploadZone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelect();
    }
  });

  fileInput.addEventListener('change', handleFileSelect);

  function handleFileSelect() {
    if (fileInput.files.length > 0) {
      const file = fileInput.files[0];
      filenameText.textContent = file.name;
      uploadFilename.classList.add('active');
      uploadBtn.style.display = 'inline-flex';

      // Motion: spring pop to draw attention
      if (window.Motion && window.Motion.animate) {
        window.Motion.animate(
          uploadBtn,
          { transform: ['scale(0.92)', 'scale(1.04)', 'scale(1)'] },
          { duration: 0.4, easing: [0.23, 1, 0.32, 1] }
        );
      }
    }
  }

  async function uploadFile() {
    if (isUploading || fileInput.files.length === 0) return;
    isUploading = true;
    setButtonLoading(uploadBtn, true);
    uploadShimmer.classList.add('active');

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      if (!res.ok) {
        const d = await res.json();
        throw new Error(d.error || 'Upload failed');
      }
      fileInput.value = '';
      uploadFilename.classList.remove('active');
      uploadBtn.style.display = 'none';
    } catch (err) {
      showToast(err.message, 'error');
      addActivity({
        type: 'error',
        title: 'Upload Failed',
        text: err.message,
        timestamp: new Date().toISOString()
      });
    } finally {
      isUploading = false;
      setButtonLoading(uploadBtn, false);
      uploadShimmer.classList.remove('active');
    }
  }

  // ── Button Loading State ──────────────────────────────
  // Emil: blur(2px) masks imperfect transitions during loading

  function setButtonLoading(btn, loading) {
    if (loading) {
      btn.disabled = true;
      btn.style.filter = 'blur(1.5px)';
      btn.style.opacity = '0.7';
    } else {
      btn.disabled = false;
      btn.style.filter = '';
      btn.style.opacity = '';
    }
  }

  function resetButtons() {
    [ingestBtn, queryBtn, uploadBtn].forEach((b) => setButtonLoading(b, false));
    isIngesting = false;
    isAsking = false;
    isUploading = false;
    uploadShimmer.classList.remove('active');
  }

  // ── Activity Feed ─────────────────────────────────────

  let activityIndex = 0;

  function addActivity(item) {
    const emptyState = activityFeed.querySelector('.empty-state');
    if (emptyState) emptyState.remove();

    const div = document.createElement('div');
    div.className = `activity-item ${item.type}`;
    div.style.setProperty('--index', activityIndex++);

    const time = new Date(item.timestamp).toLocaleTimeString();
    div.innerHTML = `
      <div class="activity-title">${item.title}</div>
      <div class="activity-text">${item.text}</div>
      <span class="activity-time">${time}</span>
    `;

    activityFeed.prepend(div);

    // Keep max 50 items
    while (activityFeed.children.length > 50) {
      activityFeed.lastElementChild.remove();
    }
  }

  // ── Stats ─────────────────────────────────────────────

  function updateStats() {
    animateStat('stat-ingested', stats.ingested);
    animateStat('stat-redacted', stats.redacted);
    animateStat('stat-queries', stats.queries);
    animateStat('stat-chunks', stats.chunks);
  }

  function animateStat(id, targetValue) {
    const el = $(id);
    const current = parseInt(el.textContent) || 0;

    if (window.Motion && window.Motion.animate) {
      // Spring-animated counter
      window.Motion.animate(current, targetValue, {
        type: 'spring',
        stiffness: 100,
        damping: 20,
        onUpdate: (v) => { el.textContent = Math.round(v); }
      });
      // Subtle scale pop — hardware accelerated
      window.Motion.animate(
        el,
        { transform: ['scale(1)', 'scale(1.15)', 'scale(1)'] },
        { duration: 0.35, easing: [0.23, 1, 0.32, 1] }
      );
    } else {
      el.textContent = targetValue;
    }
  }

  // ── Modal ─────────────────────────────────────────────

  function showResultModal(title, data) {
    const body = $('modal-body');
    body.parentElement.classList.remove('wide');
    $('modal-title').textContent = title;

    const questionTokens = data.question_entities
      ?.map((e) => `<span class="token">${escapeHtml(e.token)}</span>`)
      .join(', ') || 'None';

    body.innerHTML = `
      <div class="result-section">
        <h4>You asked</h4>
        <div class="result-box">${escapeHtml(data.original_question)}</div>
      </div>
      <div class="result-section boundary-out">
        <h4>Sent to the LLM</h4>
        <div class="result-box mono">${highlightTokens(data.masked_question)}</div>
      </div>
      <div class="result-section boundary-out">
        <h4>The LLM replied</h4>
        <div class="result-box mono">${highlightTokens(data.masked_answer || '')}</div>
      </div>
      <div class="result-section">
        <h4>You see (unmasked locally)</h4>
        <div class="result-box answer">${escapeHtml(data.answer)}</div>
      </div>
      <div class="result-section">
        <h4>Telemetry</h4>
        <div class="result-box">
          <p style="margin:0 0 8px"><strong>Intercepted PII:</strong> ${questionTokens}</p>
          <p style="margin:0 0 8px"><strong>Chunks retrieved:</strong> ${escapeHtml(String(data.context_chunks))}</p>
          <p style="margin:0"><strong>Latency:</strong> ${data.processing_time}ms</p>
        </div>
      </div>
    `;

    modal.classList.add('active');
  }

  // Wraps <<TOKEN>> placeholders in chips; everything else is escaped first
  function highlightTokens(text) {
    return escapeHtml(text).replace(/&lt;&lt;[A-Z_]+_\d{2,}_[0-9a-f]{4}&gt;&gt;/g,
      (t) => `<span class="token">${t}</span>`);
  }

  function showIngestModal(data) {
    const body = $('modal-body');
    $('modal-title').textContent = 'Ingestion Report';
    body.parentElement.classList.add('wide');
    body.innerHTML = `
      <div class="side-by-side">
        <div class="result-section">
          <h4>Original (stays in this app)</h4>
          <div class="result-box mono pre">${escapeHtml(data.original_text)}</div>
        </div>
        <div class="result-section boundary-out">
          <h4>Stored and searchable (tokens only)</h4>
          <div class="result-box mono pre">${highlightTokens(data.masked_text)}</div>
        </div>
      </div>
      <div class="result-section">
        <div class="result-box">
          <strong>${data.pii_redacted}</strong> values tokenized &middot; <strong>${data.chunks_created}</strong> chunks stored
        </div>
      </div>
    `;
    modal.classList.add('active');
  }

  function closeModal() {
    modal.classList.remove('active');
  }

  // Close modal on Escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });

  // ── Toasts ────────────────────────────────────────────
  // Emil: CSS @starting-style handles entry animation.
  // JS only handles lifecycle/removal.

  function showToast(message, type) {
    type = type || 'info';
    const container = $('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    const titles = {
      success: 'Operation Successful',
      error: 'System Fault',
      warning: 'Attention',
      info: 'Information'
    };

    toast.innerHTML = `
      <div class="toast-title">${titles[type]}</div>
      <div class="toast-msg">${escapeHtml(message)}</div>
    `;

    container.appendChild(toast);

    // Emil: exit is faster than enter (180ms vs 350ms CSS entry)
    setTimeout(() => {
      toast.classList.add('leaving');
      setTimeout(() => toast.remove(), 180);
    }, 4000);
  }

  // ── Utility ───────────────────────────────────────────

  function escapeHtml(text) {
    if (!text) return '';
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
  }

  // ── Perpetual Micro-Interactions (Motion) ─────────────
  // Isolated from main render. Only transform + opacity.

  document.addEventListener('DOMContentLoaded', () => {
    if (!window.Motion || !window.Motion.animate) return;

    // Live dot breathing pulse
    const liveDot = $('live-dot-animate');
    if (liveDot) {
      window.Motion.animate(
        liveDot,
        { transform: ['scale(1)', 'scale(1.5)', 'scale(1)'], opacity: [0.5, 1, 0.5] },
        { duration: 2.2, repeat: Infinity, easing: 'ease-in-out' }
      );
    }

    // Upload icon gentle float
    const uploadIcon = $('upload-icon-animate');
    if (uploadIcon) {
      window.Motion.animate(
        uploadIcon,
        { transform: ['translateY(0px)', 'translateY(-5px)', 'translateY(0px)'] },
        { duration: 2.8, repeat: Infinity, easing: 'ease-in-out' }
      );
    }
  });

  // ── Keyboard Shortcuts ────────────────────────────────
  // Emil: no animation on keyboard-initiated actions (they're high-frequency)

  window.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      if (document.activeElement === $('query-input')) {
        e.preventDefault();
        askQuestion();
      }
    }
  });

})();
