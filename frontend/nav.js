/**
 * Initialize a configurable navigation header.
 *
 * @param {Object}  opts
 * @param {string}  opts.appName        - Brand text shown on the left
 * @param {Array}   opts.links          - [{label, href}, ...] rendered in center
 * @param {boolean} opts.showUserPicker - Accepted for backward compat; no-op
 * @param {Element} [opts.mountTo]      - Element to prepend nav into (default: document.body)
 */
export function initNav({ appName = 'Voice Coach', links = [], showUserPicker = false, mountTo } = {}) {
  const path = window.location.pathname;
  const hash = window.location.hash;

  // -- Inject nav styles once --
  if (!document.getElementById('vc-nav-style')) {
    const style = document.createElement('style');
    style.id = 'vc-nav-style';
    style.textContent = '.vc-nav-switch-link { font-size: 0.8rem; color: var(--text-dim, #888); text-decoration: none; margin-left: auto; }';
    document.head.appendChild(style);
  }

  // -- Nav container --
  const nav = document.createElement('nav');
  nav.className = 'vc-nav';

  // -- Brand --
  const brand = document.createElement('span');
  brand.className = 'vc-nav-brand';
  brand.textContent = appName;
  nav.appendChild(brand);

  // -- Links --
  const linksDiv = document.createElement('div');
  linksDiv.className = 'vc-nav-links';

  for (const link of links) {
    const a = document.createElement('a');
    a.href = link.href;
    a.textContent = link.label;
    // Active state: match path, or for hash links match path+hash
    if (link.href.includes('#')) {
      if (path + hash === link.href) a.classList.add('active');
    } else {
      if (path === link.href) a.classList.add('active');
    }
    linksDiv.appendChild(a);
  }

  nav.appendChild(linksDiv);

  // -- Switch user link (pushes to right via margin-left: auto in CSS) --
  const switchLink = document.createElement('a');
  switchLink.href = '/?switch=1';
  switchLink.className = 'vc-nav-switch-link';
  switchLink.textContent = '⇄ Switch user';
  nav.appendChild(switchLink);

  // -- Mount --
  const target = mountTo || document.body;
  target.insertBefore(nav, target.firstChild);
}
