// Cookie helpers
export function getCookie(name) {
  const match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
  return match ? decodeURIComponent(match[1]) : null;
}

export function setCookie(name, value, days = 365) {
  const expires = new Date(Date.now() + days * 864e5).toUTCString();
  document.cookie = `${name}=${encodeURIComponent(value)}; expires=${expires}; path=/; SameSite=Lax`;
}

export function clearCookie(name) {
  document.cookie = `${name}=; expires=Thu, 01 Jan 1970 00:00:00 GMT; path=/; SameSite=Lax`;
}

// Sync cookie → localStorage on load so existing code keeps working
const _cookieUser = getCookie('vc_user');
if (_cookieUser && !localStorage.getItem('activeUserId')) {
  localStorage.setItem('activeUserId', _cookieUser);
}

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

// User CRUD
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
