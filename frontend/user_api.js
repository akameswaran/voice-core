// Wraps fetch to always include X-User-Id header
export function apiFetch(url, options = {}) {
  const userId = localStorage.getItem('activeUserId');
  const headers = { ...(options.headers || {}) };
  if (userId) headers['X-User-Id'] = userId;
  return fetch(url, { ...options, headers });
}

// Build WebSocket URL with user_id query param
export function wsUrl(path) {
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const userId = localStorage.getItem('activeUserId');
  return `${proto}//${location.host}${path}?user_id=${userId}`;
}

// User CRUD (used by nav.js user picker)
export async function listUsers() {
  const res = await fetch('/api/users');
  return res.json();
}

export async function createUser(name) {
  const res = await fetch('/api/users', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name }),
  });
  return res.json();
}

export function getActiveUserId() {
  return localStorage.getItem('activeUserId');
}
