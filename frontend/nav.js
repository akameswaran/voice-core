import { listUsers, createUser } from './user_api.js';

/**
 * Initialize a configurable navigation header.
 *
 * @param {Object}  opts
 * @param {string}  opts.appName        - Brand text shown on the left
 * @param {Array}   opts.links          - [{label, href}, ...] rendered in center
 * @param {boolean} opts.showUserPicker - If true, render user dropdown on right
 * @param {Element} [opts.mountTo]      - Element to prepend nav into (default: document.body)
 */
export function initNav({ appName = 'Voice Coach', links = [], showUserPicker = false, mountTo } = {}) {
  const path = window.location.pathname;
  const hash = window.location.hash;

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

  // -- Spacer (pushes user picker to the right) --
  const spacer = document.createElement('div');
  spacer.style.flex = '1';
  nav.appendChild(spacer);

  // -- User picker --
  if (showUserPicker) {
    const userWrap = document.createElement('div');
    userWrap.className = 'vc-user-picker';

    const select = document.createElement('select');
    select.id = 'vc-user-select';
    select.innerHTML = '<option value="">Loading...</option>';
    userWrap.appendChild(select);

    nav.appendChild(userWrap);

    // Load users and populate dropdown
    async function loadUsers() {
      try {
        const users = await listUsers();
        select.innerHTML = '';
        for (const u of users) {
          const opt = document.createElement('option');
          opt.value = u.id;
          opt.textContent = u.name;
          select.appendChild(opt);
        }
        // "New User..." option at the end
        const newOpt = document.createElement('option');
        newOpt.value = '__new__';
        newOpt.textContent = 'New User...';
        select.appendChild(newOpt);

        // Restore saved selection
        const saved = localStorage.getItem('activeUserId');
        if (saved && users.some(u => u.id === saved)) {
          select.value = saved;
        } else if (users.length) {
          // Auto-select first user if none stored
          select.value = users[0].id;
          localStorage.setItem('activeUserId', users[0].id);
          document.dispatchEvent(new CustomEvent('userchange', { detail: { userId: users[0].id } }));
        }
      } catch {
        select.innerHTML = '<option value="">-- No users --</option>';
      }
    }

    select.addEventListener('change', async () => {
      if (select.value === '__new__') {
        const name = prompt('Enter user name:');
        if (!name || !name.trim()) {
          // Revert to previously stored user
          const prev = localStorage.getItem('activeUserId');
          if (prev) select.value = prev;
          return;
        }
        try {
          const user = await createUser(name.trim());
          if (user.id) {
            await loadUsers();
            select.value = user.id;
            localStorage.setItem('activeUserId', user.id);
            document.dispatchEvent(new CustomEvent('userchange', { detail: { userId: user.id } }));
          }
        } catch (err) {
          console.error('Failed to create user:', err);
        }
        return;
      }

      if (select.value) {
        localStorage.setItem('activeUserId', select.value);
      } else {
        localStorage.removeItem('activeUserId');
      }
      document.dispatchEvent(new CustomEvent('userchange', { detail: { userId: select.value } }));
    });

    loadUsers();
  }

  // -- Mount --
  const target = mountTo || document.body;
  target.insertBefore(nav, target.firstChild);
}
