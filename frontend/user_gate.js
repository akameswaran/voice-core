/**
 * user_gate.js — Cookie-based user recognition gate.
 *
 * Usage:
 *   import { initGate } from '/static/core/user_gate.js';
 *   initGate({ redirectTo: '/practice', appName: 'Femme Voice Coach' });
 *
 * Flow:
 *   Known user (valid cookie) → "Continuing as [Name]" → auto-redirect
 *   Unknown / ?switch=1      → user picker or create form
 *   No users exist           → create form directly
 */

import { getCookie, setCookie, listUsers, createUser } from './user_api.js';

const COOKIE_NAME = 'vc_user';

export async function initGate({ redirectTo = '/practice', appName = 'Voice Coach' } = {}) {
  _injectStyles();

  const mount = document.getElementById('gate-mount') || document.body;
  const forceSwitch = new URLSearchParams(location.search).has('switch');
  const cookieId = getCookie(COOKIE_NAME);

  let users = [];
  try {
    users = await listUsers();
  } catch (e) {
    _renderError(mount, 'Could not reach server. Is it running?');
    return;
  }

  const knownUser = cookieId ? users.find(u => u.id === cookieId) : null;

  if (knownUser && !forceSwitch) {
    _renderContinue(mount, knownUser, redirectTo);
  } else if (users.length === 0) {
    _renderCreateForm(mount, redirectTo);
  } else {
    _renderPicker(mount, users, redirectTo);
  }
}

function _renderContinue(mount, user, redirectTo) {
  mount.innerHTML = `
    <div class="gate-screen">
      <div class="gate-card gate-continue">
        <div class="gate-label">Welcome back</div>
        <div class="gate-username">${_esc(user.name)}</div>
        <div class="gate-hint">Entering automatically…</div>
        <a href="?switch=1" class="gate-switch-link">Switch user</a>
      </div>
    </div>`;
  setTimeout(() => _confirm(user.id, redirectTo), 1500);
}

function _renderPicker(mount, users, redirectTo) {
  const cards = users.map(u => `
    <button class="gate-user-card" data-id="${_esc(u.id)}">
      <span class="gate-user-initial">${_esc(u.name[0].toUpperCase())}</span>
      <span class="gate-user-name">${_esc(u.name)}</span>
    </button>`).join('');

  mount.innerHTML = `
    <div class="gate-screen">
      <div class="gate-card">
        <div class="gate-label">Who are you?</div>
        <div class="gate-user-grid">${cards}</div>
        <button class="gate-new-btn" id="gate-new-btn">＋ New user</button>
      </div>
    </div>`;

  mount.querySelectorAll('.gate-user-card').forEach(btn => {
    btn.addEventListener('click', () => _confirm(btn.dataset.id, redirectTo));
  });

  document.getElementById('gate-new-btn').addEventListener('click', () => {
    _renderCreateForm(mount, redirectTo);
  });
}

function _renderCreateForm(mount, redirectTo) {
  mount.innerHTML = `
    <div class="gate-screen">
      <div class="gate-card">
        <div class="gate-label">Create a user</div>
        <input class="gate-name-input" id="gate-name-input" type="text"
               placeholder="Your name" autocomplete="off" maxlength="50">
        <button class="gate-submit-btn" id="gate-submit-btn">Continue</button>
        <div class="gate-error" id="gate-error" style="display:none;"></div>
      </div>
    </div>`;

  const input = document.getElementById('gate-name-input');
  const btn = document.getElementById('gate-submit-btn');
  const err = document.getElementById('gate-error');

  input.focus();

  async function submit() {
    const name = input.value.trim();
    if (!name) { _showErr(err, 'Please enter a name.'); return; }
    btn.disabled = true;
    try {
      const user = await createUser(name);
      _confirm(user.id, redirectTo);
    } catch (e) {
      _showErr(err, 'Could not create user. Try again.');
      btn.disabled = false;
    }
  }

  btn.addEventListener('click', submit);
  input.addEventListener('keydown', e => { if (e.key === 'Enter') submit(); });
}

function _renderError(mount, msg) {
  mount.innerHTML = `
    <div class="gate-screen">
      <div class="gate-card">
        <div class="gate-label" style="color:var(--danger,#e55);">Error</div>
        <div class="gate-hint">${_esc(msg)}</div>
        <button onclick="location.reload()" class="gate-submit-btn">Retry</button>
      </div>
    </div>`;
}

function _confirm(userId, redirectTo) {
  setCookie(COOKIE_NAME, userId);
  localStorage.setItem('activeUserId', userId);
  window.location.replace(redirectTo);
}

function _esc(str) {
  return String(str).replace(/[&<>"']/g, c =>
    ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function _showErr(el, msg) {
  el.textContent = msg;
  el.style.display = '';
}

function _injectStyles() {
  if (document.getElementById('gate-styles')) return;
  const s = document.createElement('style');
  s.id = 'gate-styles';
  s.textContent = `
    .gate-screen {
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      background: var(--bg, #111);
    }
    .gate-card {
      background: var(--surface, #1e1e1e);
      border: 1px solid var(--border, #333);
      border-radius: 12px;
      padding: 40px 48px;
      min-width: 320px;
      max-width: 480px;
      width: 100%;
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 16px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .gate-label {
      font-size: 0.85rem;
      color: var(--text-dim, #888);
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .gate-username {
      font-size: 2.2rem;
      font-weight: 700;
      color: var(--accent, #7eb8f7);
      text-align: center;
    }
    .gate-hint {
      font-size: 0.9rem;
      color: var(--text-dim, #888);
    }
    .gate-switch-link {
      font-size: 0.85rem;
      color: var(--text-dim, #888);
      text-decoration: underline;
      margin-top: 8px;
    }
    .gate-user-grid {
      display: flex;
      flex-direction: column;
      gap: 10px;
      width: 100%;
    }
    .gate-user-card {
      display: flex;
      align-items: center;
      gap: 14px;
      padding: 14px 18px;
      background: var(--bg, #111);
      border: 1px solid var(--border, #333);
      border-radius: 8px;
      cursor: pointer;
      width: 100%;
      text-align: left;
      transition: border-color 0.15s;
    }
    .gate-user-card:hover { border-color: var(--accent, #7eb8f7); }
    .gate-user-initial {
      width: 36px;
      height: 36px;
      border-radius: 50%;
      background: var(--accent, #7eb8f7);
      color: #111;
      display: flex;
      align-items: center;
      justify-content: center;
      font-weight: 700;
      font-size: 1.1rem;
      flex-shrink: 0;
    }
    .gate-user-name {
      font-size: 1.05rem;
      color: var(--text, #eee);
    }
    .gate-new-btn {
      margin-top: 4px;
      background: none;
      border: 1px dashed var(--border, #444);
      border-radius: 8px;
      padding: 10px 20px;
      color: var(--text-dim, #888);
      cursor: pointer;
      width: 100%;
      font-size: 0.95rem;
    }
    .gate-new-btn:hover { border-color: var(--accent, #7eb8f7); color: var(--accent, #7eb8f7); }
    .gate-name-input {
      width: 100%;
      padding: 12px 14px;
      background: var(--bg, #111);
      border: 1px solid var(--border, #444);
      border-radius: 8px;
      color: var(--text, #eee);
      font-size: 1.1rem;
      box-sizing: border-box;
    }
    .gate-name-input:focus { outline: none; border-color: var(--accent, #7eb8f7); }
    .gate-submit-btn {
      padding: 12px 32px;
      background: var(--accent, #7eb8f7);
      color: #111;
      border: none;
      border-radius: 8px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      width: 100%;
    }
    .gate-submit-btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .gate-error { color: var(--danger, #e55); font-size: 0.9rem; }
  `;
  document.head.appendChild(s);
}
