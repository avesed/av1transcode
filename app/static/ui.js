// Shared runtime for the three UI pages: formatting, the Chinese word tables,
// the one fetch wrapper, feedback (.notice / <dialog>) and the poll loop.
//
// Loaded with a plain <script> after apikey.js and before the page's own
// script. Everything here is a top-level declaration - no DOM work happens
// until a page calls something, so load order beyond "apikey.js first" does
// not matter.
//
// Two rules this file exists to enforce:
//   1. Every /api call goes through api(), which goes through afetch(), which
//      attaches X-API-Key. A literal fetch("/api...) anywhere is a bug the
//      test suite fails on (tests/test_queue_api.py:521).
//   2. One implementation of every format. index.html used to carry two byte
//      formatters that disagreed (one stopped at GB and returned "" for null,
//      the other went to TB and returned an en dash), so the same 4.7 GB file
//      read differently in the table and in the directory browser.

/* global afetch, authHint */

// The one "no value" glyph. U+2013, matching what the old table printed.
var DASH = "–";

// ---------------------------------------------------------------- escaping

function esc(s) {
  return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
  });
}

// Mark a string as already-safe HTML. Only h`` and raw() produce these, so an
// interpolated value can never be trusted by accident.
function raw(s) {
  var o = new String(s == null ? "" : s);
  o.__html = true;
  return o;
}

function isRaw(v) { return v instanceof String && v.__html === true; }

function renderValue(v) {
  if (v == null || v === false) return "";
  if (isRaw(v)) return String(v);
  if (Array.isArray(v)) return v.map(renderValue).join("");
  return esc(v);
}

// Auto-escaping template tag. Escapes every interpolation except the ones that
// came out of h`` or raw(), so nested templates compose without re-escaping:
//
//   el.innerHTML = h`<td>${job.source}</td>${rowsHtml}`;
//
// The return value is a String object (truthy even when empty) - test its
// length, not the value itself.
function h(strings) {
  var out = strings[0];
  for (var i = 1; i < arguments.length; i++) out += renderValue(arguments[i]) + strings[i];
  return raw(out);
}

// -------------------------------------------------------------- formatting

// B..TB. `fallback` lets the directory browser print nothing for a directory
// while the job table prints an en dash for "not produced yet".
function fmtBytes(n, fallback) {
  var d = fallback === undefined ? DASH : fallback;
  if (n == null || n === "" || !isFinite(Number(n))) return d;
  var u = ["B", "KB", "MB", "GB", "TB"], v = Number(n), i = 0;
  while (v >= 1024 && i < u.length - 1) { v /= 1024; i++; }
  return (i === 0 ? Math.round(v) : v.toFixed(1)) + " " + u[i];
}

// H:MM:SS once past an hour. The old ETA never carried, so a four-hour job
// showed "剩余 183:07" - a number no operator can read as three hours.
function fmtDuration(sec, fallback) {
  var d = fallback === undefined ? DASH : fallback;
  if (sec == null || !isFinite(Number(sec))) return d;
  var s = Math.max(0, Math.round(Number(sec)));
  var hh = Math.floor(s / 3600); s -= hh * 3600;
  var mm = Math.floor(s / 60); s -= mm * 60;
  var ss = String(s).padStart(2, "0");
  return hh ? hh + ":" + String(mm).padStart(2, "0") + ":" + ss : mm + ":" + ss;
}

// One decimal below 1.5 fps, integer above. A 4K DV shot probing at 0.4 fps
// used to round to "0 fps", which reads as stalled.
function fmtFps(fps, fallback) {
  var d = fallback === undefined ? DASH : fallback;
  if (fps == null || !isFinite(Number(fps))) return d;
  var v = Number(fps);
  return v < 1.5 ? v.toFixed(1) : String(Math.round(v));
}

// The backend writes epoch seconds (time.time()); tolerate milliseconds too.
function toDate(ts) {
  if (ts == null || ts === "" || !isFinite(Number(ts))) return null;
  var n = Number(ts);
  var d = new Date(n > 1e11 ? n : n * 1000);
  return isFinite(d.getTime()) ? d : null;
}

function fmtClock(ts, withDate) {
  var d = toDate(ts);
  if (!d) return DASH;
  var p = function (n) { return String(n).padStart(2, "0"); };
  var t = p(d.getHours()) + ":" + p(d.getMinutes()) + ":" + p(d.getSeconds());
  return withDate ? (d.getMonth() + 1) + "-" + d.getDate() + " " + t : t;
}

function fmtAgo(ts) {
  var d = toDate(ts);
  if (!d) return DASH;
  var s = (Date.now() - d.getTime()) / 1000;
  if (s < 0) s = 0;
  if (s < 60) return "刚刚";
  if (s < 3600) return Math.floor(s / 60) + " 分钟前";
  if (s < 86400) return Math.floor(s / 3600) + " 小时前";
  return Math.floor(s / 86400) + " 天前";
}

// The ratio this archiver exists for. Empty string when it is not known yet,
// so a caller can concatenate it without printing a fake 0%.
function fmtRatio(before, after) {
  var a = Number(before), b = Number(after);
  if (!isFinite(a) || !isFinite(b) || a <= 0 || b <= 0) return "";
  var pct = Math.round((1 - b / a) * 100);
  if (pct > 0) return "↘" + pct + "%";
  if (pct < 0) return "↗" + -pct + "%";
  return "±0%";
}

// Thin-space grouping: frame counters run to six digits and 8420/20340 is
// unreadable at 12px.
function fmtNum(n, fallback) {
  var d = fallback === undefined ? DASH : fallback;
  if (n == null || n === "" || !isFinite(Number(n))) return d;
  return String(Math.round(Number(n))).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
}

// ------------------------------------------------------------ word tables

// The seven job statuses (app/db.py:47-53). `state` is what goes in
// data-state on .mark and .row; the English key stays visible in the expanded
// detail panel, because that is the word the API and the logs use.
var STATUS = {
  pending:   { cn: "排队",   state: "pending" },
  analyzing: { cn: "分析中", state: "analyzing" },
  running:   { cn: "转码中", state: "running" },
  done:      { cn: "已完成", state: "done" },
  failed:    { cn: "失败",   state: "failed" },
  cancelled: { cn: "已取消", state: "cancelled" },
  skipped:   { cn: "已跳过", state: "skipped" }
};

// All eleven stage values the backend writes (app/queue.py + transcoder.py),
// plus "". The old stageLabel() mapped six and let done/failed/retry/skipped/
// cancelled fall through as raw English into a Chinese table.
//   cn    = the full word, for the 阶段 column and the glossary
//   short = the ladder segment name, which has to fit a 36px segment at 390px
var STAGE = {
  "":          { cn: "",             short: "" },
  analyzing:   { cn: "分析中",       short: "分析" },
  scenedetect: { cn: "场景检测",     short: "分镜" },
  probing:     { cn: "质量探测中",   short: "探测" },
  encoding:    { cn: "转码中",       short: "编码" },
  verifying:   { cn: "成品质量校验中", short: "校验" },
  cancelling:  { cn: "取消中…",      short: "取消" },
  retry:       { cn: "准备重试",     short: "重试" },
  done:        { cn: "已完成",       short: "完成" },
  failed:      { cn: "失败",         short: "失败" },
  skipped:     { cn: "已跳过",       short: "跳过" },
  cancelled:   { cn: "已取消",       short: "取消" }
};

// Unknown keys fall through as themselves: a stage added to the backend
// should show up as English, not disappear.
function statusLabel(s) { return (STATUS[s] && STATUS[s].cn) || s || ""; }
function stageLabel(s) { return STAGE[s] ? STAGE[s].cn : (s || ""); }
function stageShort(s) { return STAGE[s] ? STAGE[s].short : (s || ""); }

// ------------------------------------------------------------------- api

// The only afetch() wrapper. Throws an Error whose message is already the
// sentence to show: authHint() first (a 401 on a fresh browser is the most
// common failure and used to surface as the English "invalid api key"), then
// the server's own detail, then the bare status.
//
// opts.json serialises a body and sets the content type, so no caller has to
// remember both.
function api(path, opts) {
  var o = Object.assign({}, opts);
  if (o.json !== undefined) {
    o.body = JSON.stringify(o.json);
    o.headers = Object.assign({ "Content-Type": "application/json" }, o.headers || {});
    if (!o.method) o.method = "POST";
    delete o.json;
  }
  return afetch(path, o).catch(function (e) {
    var err = new Error("连不上服务端（" + ((e && e.message) || "network error") + "）");
    err.offline = true;
    throw err;
  }).then(function (r) {
    if (r.ok) {
      if (r.status === 204) return null;
      var ct = r.headers.get("content-type") || "";
      return ct.indexOf("json") >= 0 ? r.json() : r.text();
    }
    return r.json().catch(function () { return null; }).then(function (body) {
      var detail = "";
      if (body && body.detail) {
        detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
      }
      var err = new Error(authHint(r.status) || detail || ("HTTP " + r.status));
      err.status = r.status;
      err.detail = detail;
      throw err;
    });
  });
}

// -------------------------------------------------------------- feedback

function noticeHost() {
  var host = document.getElementById("notices") || document.querySelector(".notice-host");
  if (!host) {
    host = document.createElement("div");
    host.className = "notice-host";
    host.id = "notices";
    document.body.appendChild(host);
  }
  return host;
}

// Global receipt for things that happened outside a form (cancel, prune).
// Success clears itself after 4s; an error stays until it is dismissed -
// cancel failing has to stay visible even if the operator has scrolled
// somewhere else by then (commit ca1b68c). Three at a time, oldest dropped.
function notice(text, tone) {
  var host = noticeHost();
  while (host.children.length >= 3) host.removeChild(host.firstElementChild);

  var el = document.createElement("div");
  el.className = "notice";
  el.dataset.tone = tone || "ok";
  el.setAttribute("role", tone === "err" ? "alert" : "status");
  el.setAttribute("aria-live", tone === "err" ? "assertive" : "polite");
  // Insert the live region first and fill it after: a region that arrives
  // already populated is not reliably announced.
  host.appendChild(el);
  el.innerHTML = h`<p>${text}</p><button type="button" class="notice__x" aria-label="关闭">✕</button>`;

  var timer = null;
  function dismiss() {
    if (timer) clearTimeout(timer);
    if (el.parentNode) el.parentNode.removeChild(el);
  }
  el.querySelector(".notice__x").addEventListener("click", dismiss);
  if (tone !== "err") timer = setTimeout(dismiss, 4000);
  return { el: el, dismiss: dismiss };
}

// Writes a form's own inline receipt (<p class="formstate">). It never
// disappears on its own; the page clears it on the next edit.
function formState(el, text, tone) {
  if (typeof el === "string") el = document.getElementById(el);
  if (!el) return;
  el.textContent = text == null ? "" : text;
  if (tone) el.dataset.tone = tone; else el.removeAttribute("data-tone");
}

// --------------------------------------------------------------- dialogs

var dlgLocks = 0;
var dlgPrevOverflow = "";

function lockScroll(on) {
  if (on) {
    if (dlgLocks === 0) {
      dlgPrevOverflow = document.body.style.overflow;
      document.body.style.overflow = "hidden";
    }
    dlgLocks++;
  } else if (dlgLocks > 0) {
    dlgLocks--;
    if (dlgLocks === 0) document.body.style.overflow = dlgPrevOverflow;
  }
}

// showModal() gives us the focus trap, Esc, ::backdrop and background inert
// for free. What it does not give us is a scroll lock on the page behind, or
// focus going back where it came from - both of which the old div-based
// browser modal also lacked, which is why a phone back-gesture inside it left
// the page entirely.
function openDlg(dlg) {
  if (typeof dlg === "string") dlg = document.getElementById(dlg);
  if (!dlg || dlg.open) return dlg;
  var opener = document.activeElement;
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
  lockScroll(true);
  dlg.addEventListener("close", function () {
    lockScroll(false);
    if (opener && typeof opener.focus === "function" && document.contains(opener)) {
      opener.focus();
    }
  }, { once: true });
  if (!dlg.__backdropWired) {
    dlg.__backdropWired = true;
    // A click that lands on the dialog element itself landed on the backdrop:
    // the children fill it edge to edge.
    dlg.addEventListener("click", function (e) {
      if (e.target !== dlg) return;
      var r = dlg.getBoundingClientRect();
      var inside = e.clientX >= r.left && e.clientX <= r.right &&
                   e.clientY >= r.top && e.clientY <= r.bottom;
      if (!inside) closeDlg(dlg);
    });
  }
  return dlg;
}

function closeDlg(dlg, value) {
  if (typeof dlg === "string") dlg = document.getElementById(dlg);
  if (!dlg || !dlg.open) return;
  if (typeof dlg.close === "function") dlg.close(value === undefined ? "" : String(value));
  else { dlg.removeAttribute("open"); lockScroll(false); }
}

function ensureDlg(id, cls) {
  var dlg = document.getElementById(id);
  if (!dlg) {
    dlg = document.createElement("dialog");
    dlg.id = id;
    document.body.appendChild(dlg);
  }
  if (cls) dlg.className = cls;
  return dlg;
}

// Replaces confirm(). Resolves true only if the operator pressed the
// destructive button; Esc, the backdrop and the cancel button all resolve
// false, and the cancel button is what holds focus when it opens.
//   confirmDlg({ title, body: "…" | ["…","…"], confirm: "清除历史记录",
//                cancel: "取消", danger: true })
function confirmDlg(opts) {
  opts = opts || {};
  var dlg = ensureDlg("confirmDlg", "dlg dlg--confirm");
  var body = opts.body == null ? [] : (Array.isArray(opts.body) ? opts.body : [opts.body]);
  var danger = opts.danger === false ? "btn" : "btn btn--danger";
  dlg.innerHTML = h`<div class="dlg__head"><h2>${opts.title || "确认"}</h2></div>
<div class="dlg__body">${body.map(function (p) { return h`<p>${p}</p>`; })}</div>
<div class="dlg__foot">
  <button type="button" class="btn" data-act="cancel" autofocus>${opts.cancel || "取消"}</button>
  <button type="button" class="${danger}" data-act="ok">${opts.confirm || "确定"}</button>
</div>`;
  return new Promise(function (resolve) {
    var settled = false;
    function finish(v) {
      if (settled) return;
      settled = true;
      closeDlg(dlg);
      resolve(v);
    }
    dlg.querySelector('[data-act="ok"]').addEventListener("click", function () { finish(true); });
    dlg.querySelector('[data-act="cancel"]').addEventListener("click", function () { finish(false); });
    dlg.addEventListener("close", function () { finish(false); }, { once: true });
    openDlg(dlg);
  });
}

// ------------------------------------------------------------------ poll

// refresh() used to be a bare setInterval with no try/catch: restart the
// backend and the page kept showing a screenful of confident stale numbers,
// forever. A backgrounded phone tab is the normal case, not the exception.
//
//   poll(refresh, { interval: 4000,
//                   onError: (err, waitMs) => …,   // show the disconnected state
//                   onOk: () => … })               // clear it
//
// Returns { stop(), now(), delay } - now() forces an immediate run.
function poll(fn, opts) {
  opts = opts || {};
  var base = opts.interval || 4000;
  var max = opts.max || 32000;
  var delay = base, timer = null, stopped = false, inflight = false, again = false;

  function schedule() {
    if (timer) { clearTimeout(timer); timer = null; }
    // A hidden tab schedules nothing at all; visibilitychange restarts it.
    if (stopped || document.hidden) return;
    timer = setTimeout(run, delay);
  }

  function run() {
    if (stopped) return;
    // A refresh asked for mid-request (after a cancel or a submit) runs as
    // soon as this one lands; dropping it left the old row up for a cycle.
    if (inflight) { again = true; return; }
    if (timer) { clearTimeout(timer); timer = null; }
    inflight = true;
    var failed = false;
    // The last step runs whatever the callbacks do: if onOk or onError
    // threw, inflight stayed true and every later run() returned early -
    // the page stopped polling for good, showing the last good data.
    function fin(e) {
      if (e) console.error(e);
      inflight = false;
      if (again && !failed) { again = false; run(); return; }
      again = false;
      schedule();                                    // waits the delay onError reported
      if (failed) delay = Math.min(delay * 2, max);  // and the next one doubles
    }
    Promise.resolve().then(fn).then(function () {
      delay = base;
      if (opts.onOk) opts.onOk();
    }, function (e) {
      failed = true;
      // Report the wait we are about to take, then double it: 4 -> 8 -> 16 -> 32.
      if (opts.onError) opts.onError(e, delay);
    }).then(function () { fin(); }, fin);
  }

  function onVisible() {
    if (document.hidden || stopped) return;
    run();   // back in the foreground: refresh now, do not wait out the interval
  }
  document.addEventListener("visibilitychange", onVisible);

  var ctl = {
    now: run,
    stop: function () {
      stopped = true;
      if (timer) clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    }
  };
  Object.defineProperty(ctl, "delay", { get: function () { return delay; } });
  if (opts.start !== false) run();
  return ctl;
}

// ----------------------------------------------------------------- misc

// navigator.clipboard only exists in a secure context, and this UI is served
// over plain http on the LAN (web: 0.0.0.0:8080), so half the time it is
// undefined. The textarea fallback is the only reason the 复制 buttons on
// paths and job ids work at all here.
function copyText(text) {
  var s = String(text == null ? "" : text);
  if (navigator.clipboard && window.isSecureContext) {
    return navigator.clipboard.writeText(s).then(function () { return true; },
                                                 function () { return legacyCopy(s); });
  }
  return Promise.resolve(legacyCopy(s));
}

function legacyCopy(s) {
  try {
    var ta = document.createElement("textarea");
    ta.value = s;
    ta.setAttribute("readonly", "");
    ta.style.position = "fixed";
    ta.style.top = "-1000px";
    document.body.appendChild(ta);
    ta.select();
    var ok = document.execCommand("copy");
    document.body.removeChild(ta);
    return ok;
  } catch (_) {
    return false;
  }
}

// Classic scripts: function declarations and var are already on window, but
// naming the surface here is what tells the next reader (and the page files)
// what this module promises.
Object.assign(window, {
  DASH: DASH, esc: esc, raw: raw, h: h,
  fmtBytes: fmtBytes, fmtDuration: fmtDuration, fmtFps: fmtFps, fmtClock: fmtClock,
  fmtAgo: fmtAgo, fmtRatio: fmtRatio, fmtNum: fmtNum,
  STATUS: STATUS, STAGE: STAGE,
  statusLabel: statusLabel, stageLabel: stageLabel, stageShort: stageShort,
  api: api, notice: notice, formState: formState,
  openDlg: openDlg, closeDlg: closeDlg, ensureDlg: ensureDlg, confirmDlg: confirmDlg,
  poll: poll, copyText: copyText
});
