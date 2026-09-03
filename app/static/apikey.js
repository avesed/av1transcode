// Shared X-API-Key plumbing for the two UI pages.
//
// web.api_key gates every mutating endpoint, plus /api/browse (which walks the
// host filesystem, not this app's own state). The key is server-side config so
// the browser cannot be told it: the operator pastes it once on the settings
// page and it lives in this origin's localStorage from then on.
//
// Before this existed neither page sent the header at all, which made
// web.api_key unusable with the bundled UI - setting it did not secure the app
// so much as lock its own frontend out of every write.
const AV1TC_KEY_ITEM = "av1tc_api_key";

function apiKey() {
  try { return localStorage.getItem(AV1TC_KEY_ITEM) || ""; } catch (_) { return ""; }
}

function setApiKey(value) {
  try {
    if (value) localStorage.setItem(AV1TC_KEY_ITEM, value);
    else localStorage.removeItem(AV1TC_KEY_ITEM);
  } catch (_) { /* private mode: the key just does not persist */ }
}

// fetch() with the key attached when one is stored. Sent on every call rather
// than only the writes: which endpoints are gated is the server's business, and
// a GET that starts requiring it (as /api/browse just did) should not also need
// a change here.
function afetch(url, opts) {
  const o = Object.assign({}, opts);
  const key = apiKey();
  if (key) o.headers = Object.assign({}, o.headers || {}, { "X-API-Key": key });
  return fetch(url, o);
}

// Shared wording, so a 401 does not read as a generic failure. The two things
// an operator can actually do about it are the two things it names.
function authHint(status) {
  if (status !== 401) return "";
  return apiKey()
    ? "API 密钥被拒绝（401）。请在「设置 → API 密钥」中确认它与 config.yaml 的 web.api_key 一致。"
    : "此操作需要 API 密钥（401）。请在「设置 → API 密钥」中填入 config.yaml 里的 web.api_key。";
}
