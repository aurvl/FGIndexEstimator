/*
  Theme toggle logic for the "new" HTML pages (line/gauge/point).
  - Uses localStorage key: "theme" (shared with the other pages).
  - Defaults to system preference if nothing was saved yet.
  - Keeps pages consistent by writing to localStorage.
*/

(function () {
  const STORAGE_KEY = 'theme';

  function getSystemTheme() {
    try {
      return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches
        ? 'dark'
        : 'light';
    } catch (e) {
      return 'light';
    }
  }

  function getSavedTheme() {
    try {
      const t = localStorage.getItem(STORAGE_KEY);
      return (t === 'dark' || t === 'light') ? t : null;
    } catch (e) {
      return null;
    }
  }

  function setSavedTheme(theme) {
    try {
      localStorage.setItem(STORAGE_KEY, theme);
    } catch (e) {}
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
  }

  function notifyThemeChange(theme) {
    try {
      // Pages can listen to this to re-render charts (ECharts does not auto-adapt).
      window.dispatchEvent(new CustomEvent('dashboard:themechange', {
        detail: { theme: theme }
      }));
    } catch (e) {
      // Ignore environments without CustomEvent support.
    }
  }

  function setLucideIconForTheme(theme) {
    const iconName = theme === 'dark' ? 'moon' : 'sun';
    document.querySelectorAll('.toogle-theme').forEach((btn) => {
      // Lucide replaces <i> by <svg>, so we always re-inject an <i data-lucide>
      // to make the icon name switch deterministic.
      btn.innerHTML = '<i data-lucide="' + iconName + '"></i>';
    });

    if (window.lucide && typeof window.lucide.createIcons === 'function') {
      window.lucide.createIcons();
    }
  }

  function initTheme() {
    const saved = getSavedTheme();
    const theme = saved || getSystemTheme();

    // Persist the first resolved theme so all pages stay consistent.
    if (!saved) setSavedTheme(theme);

    applyTheme(theme);
    setLucideIconForTheme(theme);
    notifyThemeChange(theme);
  }

  function toggleTheme() {
    const cur = document.documentElement.getAttribute('data-theme') || getSystemTheme();
    const next = cur === 'dark' ? 'light' : 'dark';
    setSavedTheme(next);
    applyTheme(next);
    setLucideIconForTheme(next);
    notifyThemeChange(next);
  }

  function bindToggleButtons() {
    document.querySelectorAll('.toogle-theme').forEach((btn) => {
      if (btn.__themeBound) return;
      btn.__themeBound = true;
      btn.addEventListener('click', toggleTheme);
      btn.setAttribute('role', 'button');
      btn.setAttribute('tabindex', '0');
      btn.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          toggleTheme();
        }
      });
    });
  }

  // Run as early as possible (but after DOM is parsed because we use querySelectorAll)
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      initTheme();
      bindToggleButtons();
    });
  } else {
    initTheme();
    bindToggleButtons();
  }
})();
