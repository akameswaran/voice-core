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
    style.textContent = `
      .vc-nav-user {
        margin-left: auto;
        display: flex;
        align-items: center;
        gap: 8px;
        text-decoration: none;
        padding: 6px 12px;
        border-radius: 6px;
        transition: background 0.15s;
      }
      .vc-nav-user:hover { background: var(--surface-2, #1e1e21); text-decoration: none; }
      .vc-nav-user-initial {
        width: 28px;
        height: 28px;
        border-radius: 50%;
        background: var(--accent, #e87d3e);
        color: #111;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.8rem;
        font-weight: 700;
        flex-shrink: 0;
      }
      .vc-nav-user-name {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text, #e8e6e1);
      }
    `;
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

  // -- User display (click to switch) --
  const userLink = document.createElement('a');
  userLink.href = '/?switch=1';
  userLink.className = 'vc-nav-user';

  const initial = document.createElement('span');
  initial.className = 'vc-nav-user-initial';
  initial.textContent = '?';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'vc-nav-user-name';
  nameSpan.textContent = '…';

  userLink.appendChild(initial);
  userLink.appendChild(nameSpan);
  nav.appendChild(userLink);

  // -- Mount --
  const target = mountTo || document.body;
  target.insertBefore(nav, target.firstChild);

  // -- Async: fill in user name --
  const userId = localStorage.getItem('activeUserId');
  if (userId) {
    fetch('/api/users')
      .then(r => r.json())
      .then(users => {
        const user = users.find(u => u.id === userId);
        if (user) {
          initial.textContent = user.name[0].toUpperCase();
          nameSpan.textContent = user.name;
        }
      })
      .catch(() => {
        nameSpan.textContent = 'Switch user';
        initial.textContent = '?';
      });
  } else {
    nameSpan.textContent = 'Select user';
  }
}
