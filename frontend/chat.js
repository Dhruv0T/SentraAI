(function () {
  const API = '';
  const params = new URLSearchParams(window.location.search);
  const reportParam = params.get('report');

  const SUGGESTED = [
    'Summarize the suspicious activity',
    'What times were people flagged?',
    'How many alerts were there?',
    'Who were the main suspects?'
  ];

  async function loadReports() {
    try {
      const res = await fetch(`${API}/reports`);
      if (!res.ok) return [];
      const data = await res.json();
      return data.reports || [];
    } catch (_) {
      return [];
    }
  }

  async function loadReportItems() {
    try {
      const res = await fetch(`${API}/reports`);
      if (!res.ok) return [];
      const data = await res.json();
      return data.items || data.reports?.map(r => ({ name: r, display_name: r.replace(/_report\.txt$/, ''), has_thumbnail: false })) || [];
    } catch (_) {
      return [];
    }
  }

    function updateEmptyState() {
    const empty = document.getElementById('chatEmpty');
    const msgs = document.querySelectorAll('#messages .msg-row');
    if (empty) empty.style.display = msgs.length ? 'none' : 'flex';
  }

  function appendMsg(role, text) {
    const empty = document.getElementById('chatEmpty');
    if (empty) empty.style.display = 'none';
    const container = document.getElementById('messages');
    const wrap = document.createElement('div');
    wrap.className = 'msg-row ' + role;
    const avatar = role === 'user' ? '<span class="msg-avatar user">You</span>' : '<span class="msg-avatar assistant">AI</span>';
    wrap.innerHTML = `${avatar}<div class="msg-bubble ${role}"><span class="msg-text">${escapeHtml(text)}</span></div>`;
    container.appendChild(wrap);
    wrap.scrollIntoView({ behavior: 'smooth' });
  }

  function escapeHtml(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
  }

  document.addEventListener('DOMContentLoaded', async () => {
    const select = document.getElementById('reportSelect');
    const cardsEl = document.getElementById('reportCards');
    const promptsEl = document.getElementById('suggestedPrompts');

    const reports = await loadReports();
    const items = await loadReportItems();

    reports.forEach(r => {
      const opt = document.createElement('option');
      opt.value = r;
      const item = items.find(i => i.name === r);
      opt.textContent = item?.display_name || r.replace(/_report\.txt$/, '');
      if (reportParam && r === reportParam) opt.selected = true;
      select.appendChild(opt);
    });

    if (promptsEl) {
      promptsEl.innerHTML = SUGGESTED.map(p => `<button type="button" class="prompt-chip">${p}</button>`).join('');
      promptsEl.querySelectorAll('.prompt-chip').forEach(btn => {
        btn.addEventListener('click', () => {
          document.getElementById('input').value = btn.textContent;
          document.getElementById('input').focus();
        });
      });
    }

    if (cardsEl && items.length) {
      cardsEl.innerHTML = items.slice(0, 4).map(item => {
        return `<a href="/chat?report=${encodeURIComponent(item.name)}" class="report-card" title="${item.display_name}">
          <div class="report-card-thumb"></div>
          <span class="report-card-title">${item.display_name}</span>
        </a>`;
      }).join('');
    }

    document.getElementById('btnSummary').addEventListener('click', async () => {
      const report = select.value;
      if (!report) return;
      appendMsg('user', 'Summarize this report.');
      appendMsg('assistant', 'Loading summary…');
      try {
        const res = await fetch(`${API}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: 'Summarize the report concisely, highlighting suspicious activity.', report_path: report })
        });
        if (!res.ok) throw new Error('Request failed');
        const data = await res.json();
        const last = document.querySelector('#messages .msg-row.assistant:last-of-type .msg-text');
        if (last) last.textContent = data.answer || 'No reply.';
      } catch (e) {
        const last = document.querySelector('#messages .msg-bubble.assistant:last-child .msg-text');
        if (last) last.textContent = 'Error: ' + (e.message || 'Could not get summary.');
      }
    });

    document.getElementById('chatForm').addEventListener('submit', async (e) => {
      e.preventDefault();
      const inp = document.getElementById('input');
      const text = inp.value.trim();
      if (!text) return;
      const report = select.value;
      appendMsg('user', text);
      inp.value = '';
      appendMsg('assistant', 'Thinking…');
      try {
        const res = await fetch(`${API}/chat`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question: text, report_path: report })
        });
        if (!res.ok) throw new Error('Request failed');
        const data = await res.json();
        const last = document.querySelector('#messages .msg-row.assistant:last-of-type .msg-text');
        if (last) last.textContent = data.answer || 'No reply.';
      } catch (err) {
        const last = document.querySelector('#messages .msg-bubble.assistant:last-child .msg-text');
        if (last) last.textContent = 'Error: ' + (err.message || 'Could not reach server.');
      }
    });
  });
})();
