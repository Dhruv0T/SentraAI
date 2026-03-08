(function () {
  const API = '';

  async function loadReports() {
    try {
      const res = await fetch(`${API}/reports`);
      if (!res.ok) return;
      const data = await res.json();
      const list = document.getElementById('videoList');
      if (!list) return;
      const items = data.items || data.reports?.map(r => ({ name: r, display_name: r.replace(/_report\.txt$/, ''), has_thumbnail: false })) || [];
      const reports = data.reports || [];
      if (items.length === 0 && reports.length > 0) {
        reports.forEach(r => items.push({ name: r, display_name: r.replace(/_report\.txt$/, ''), has_thumbnail: false }));
      }
      const toShow = items.length ? items : reports.map(r => ({ name: r, display_name: r }));
      list.innerHTML = toShow.slice(0, 8).map(item => {
        const name = typeof item === 'string' ? item : item.name;
        const display = typeof item === 'string' ? name.replace(/_report\.txt$/, '') : (item.display_name || name.replace(/_report\.txt$/, ''));
        return `
          <div class="video-card">
            <div class="thumb">
              <span class="badge">ALERT</span>
            </div>
            <div class="meta">
              <span class="video-title" title="${display}">${display}</span>
              <div class="card-buttons">
                <button class="btn-view" onclick="window.location='/chat?report=${encodeURIComponent(name)}'">View</button>
                <button class="btn-analyze">Analyze</button>
              </div>
            </div>
          </div>
        `;
      }).join('');
    } catch (_) {}
  }

  document.addEventListener('DOMContentLoaded', loadReports);
})();
