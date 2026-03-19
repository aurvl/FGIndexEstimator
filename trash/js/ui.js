export function setActiveTab(tabId) {
  document.querySelectorAll(".tab").forEach(b => b.classList.remove("tab--active"));
  document.querySelectorAll(".tabpane").forEach(p => p.classList.remove("tabpane--active"));

  document.querySelector(`.tab[data-tab="${tabId}"]`).classList.add("tab--active");
  document.getElementById(tabId).classList.add("tabpane--active");
}

export function setActiveRange(groupEl, range) {
  groupEl.querySelectorAll(".seg").forEach(b => b.classList.toggle("seg--active", b.dataset.range === range));
}

export function getCheckedMarkets() {
  const res = [];
  document.querySelectorAll('[data-market]').forEach(inp => {
    if (inp.checked) res.push(inp.dataset.market);
  });
  return res;
}

export function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  const label = document.getElementById("themeLabel");
  if (label) label.textContent = theme === "dark" ? "Dark" : "Light";
}

// Initialize a simple market dropdown built from radio inputs inside the container
export function initMarketDropdown(containerOrId) {
  const container = typeof containerOrId === 'string' ? document.getElementById(containerOrId) : containerOrId;
  if (!container) return;
  const btn = container.querySelector('#marketDropdownBtn');
  const menu = container.querySelector('.market-dropdown-menu');
  if (!btn || !menu) return;

  function closeMenu() { menu.setAttribute('aria-hidden', 'true'); menu.style.display = 'none'; }
  function openMenu() { menu.setAttribute('aria-hidden', 'false'); menu.style.display = 'flex'; }

  btn.addEventListener('click', (e) => {
    const hidden = menu.getAttribute('aria-hidden') === 'true';
    if (hidden) openMenu(); else closeMenu();
  });

  menu.querySelectorAll('input[name="market"]').forEach(inp => {
    inp.addEventListener('change', () => {
      const sel = getSelectedMarkets()[0] || '';
      btn.textContent = sel === '' ? 'No market' : sel;
      closeMenu();
      // emit a DOM event so callers can listen
      container.dispatchEvent(new CustomEvent('market-change', { detail: { market: sel } }));
    });
  });

  // close on outside click / ESC
  document.addEventListener('click', (e) => { if (!container.contains(e.target)) closeMenu(); });
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });

  // initialize label from current selection
  const cur = getSelectedMarkets()[0] || '';
  btn.textContent = cur === '' ? 'No market' : cur;
}

export function getSelectedMarkets() {
  const res = [];
  document.querySelectorAll('[data-market]').forEach(inp => {
    if (inp.checked) res.push(inp.dataset.market);
  });
  return res;
}

export function toggleDebugPanel() {
  const p = document.getElementById('debugPanel');
  if (!p) return;
  p.style.display = p.style.display === 'none' ? 'block' : 'none';
}

export function showDebugPayload(obj) {
  const el = document.getElementById('debugContent');
  if (!el) return;
  try { el.textContent = JSON.stringify(obj, null, 2); } catch (e) { el.textContent = String(obj); }
}
