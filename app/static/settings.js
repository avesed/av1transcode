// Settings page: renders every knob out of fields.js, loads and saves the five
// forms, and owns the GPU readout and the preset editor.
//
// Loaded after apikey.js, ui.js and fields.js, and wrapped in one function so
// nothing here can collide with the globals those three declare: index.html
// once re-declared `const esc` at page scope and the whole script died with
// "Identifier 'esc' has already been declared" - a blank page, not an error.
//
// Every /api call goes through api() (ui.js -> afetch -> X-API-Key).
/* global F, OPT_GROUPS, GPU_FIELDS, PRESET_FIELDS, PRESET_INT, SECTIONS,
          api, h, raw, esc, notice, formState, openDlg, closeDlg, confirmDlg,
          fmtClock, apiKey, setApiKey */
(function () {
"use strict";

const $ = id => document.getElementById(id);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

// ------------------------------------------------------------------ state

const OPT_FIELDS = OPT_GROUPS.flatMap(g => g.fields);
const SPEC = new Map([...OPT_FIELDS, ...GPU_FIELDS].map(f => [f.key, f]));
const OPT_KEYS = new Set(OPT_FIELDS.map(f => f.key));
const GPU_KEYS = new Set(GPU_FIELDS.map(f => f.key));
const ALL_KEYS = new Set(SPEC.keys());
const fieldEl = new Map();   // key -> .field, optimizer + GPU

// /api/settings/optimizer/defaults is config.yaml's optimizer block, and
// vulkan_device is not in it - it lives in transcode.dovi. Its config.yaml
// default is empty (let ffmpeg pick); without this entry the field could never
// be marked, counted, or put back.
const EXTRA_DEFAULTS = { vulkan_device: "" };

let optDefaults = null;      // null until loaded: nothing reads as "changed" before we know
const saved = {};            // what the server holds now, per key: the unsaved-edit baseline
let savedWorkers = null, savedSafety = null, activeWorkers = null;
let presets = {};
const loadErrors = new Map();   // where -> message, cleared by that loader's next success
let booted = false;

// ---------------------------------------------------------------- controls

// Numeric keyboards on phones: "numeric" has no decimal point and iOS's
// "decimal" pad has no minus sign, so a field that can go negative
// (probe_crf_offset) keeps the full keyboard.
function inputMode(f) {
  if (f.min === undefined || f.min < 0) return "";
  return f.step && String(f.step).includes(".") ? "decimal" : "numeric";
}

function control(f, id, attr) {
  const a = h`id="${id}" ${raw(attr)}="${f.key}"`;
  if (f.type === "select" || f.type === "bool") {
    const opts = f.type === "bool" ? [["false", "关"], ["true", "开"]] : f.options;
    return h`<select ${a}>${opts.map(o => h`<option value="${o[0]}">${o[1]}</option>`)}</select>`;
  }
  if (f.type === "sycl") return h`<select ${a}><option value="-1">关闭：用 CPU 打分</option></select>`;
  if (f.type === "vulkan") {
    return h`<input type="text" ${a} list="vkList" placeholder="空 = 自动" autocomplete="off" spellcheck="false"><datalist id="vkList"></datalist>`;
  }
  if (f.type === "text" || f.type === "csv") {
    return h`<input type="text" ${a}${f.placeholder ? h` placeholder="${f.placeholder}"` : ""} autocomplete="off" spellcheck="false">`;
  }
  const mode = inputMode(f);
  return h`<input type="number" ${a}${f.min !== undefined ? h` min="${f.min}"` : ""}${f.max !== undefined ? h` max="${f.max}"` : ""}${f.step !== undefined ? h` step="${f.step}"` : ""}${mode ? h` inputmode="${mode}"` : ""}>`;
}

// Every optimizer and GPU field has an entry on the docs page. The three keys a
// preset also carries (min_scene_len, vmaf_threads, probing_rate) have two
// entries there; the optimizer's is p-opt-<key>. tests/test_core.py checks
// that every anchor this produces exists.
const PRESET_KEYS = new Set(PRESET_FIELDS.map(f => f.key));
const docAnchor = key => (PRESET_KEYS.has(key) ? "p-opt-" : "p-") + key;

// Layer 1 is label, key, control, unit, the one-line help and the 默认 X ⟲
// chip; layer 2 is exactly one <details> whose summary names what it holds.
// The note is the original measured text, whole - see fields.js.
function renderField(f, o) {
  const id = `${o.ctl}-${f.key}`;
  const div = document.createElement("div");
  div.className = "field" + (f.wide ? " field--wide" : "");
  div.id = o.id + f.key;
  div.innerHTML = h`<label class="field__label" for="${id}">${f.label} <code class="field__key">${f.key}</code></label>
<div class="field__ctl">${control(f, id, o.attr)}${f.unit ? h`<span class="field__unit">${f.unit}</span>` : ""}</div>
${f.help ? h`<p class="field__help">${f.help}</p>` : ""}
${f.note ? h`<details class="disclose field__note"><summary>${f.summary}</summary><p>${f.note}</p></details>` : ""}
${o.foot ? h`<div class="field__foot"><button type="button" class="field__dflt" data-reset="${f.key}" title="恢复 config.yaml 的值" hidden></button><a class="field__doc" href="/static/docs.html#${docAnchor(f.key)}">说明 ›</a></div>` : ""}`;
  return div;
}

const ctlOf = key => fieldEl.get(key).querySelector("[data-key]");
const isFloat = f => !!f.step && String(f.step).includes(".");

// A CRF / QSV grid. Splitting on "," and parseInt-ing each piece read
// "20，26，32" (the IME's comma) as [20], and a one-point probe_crfs pins
// every shot of every later job to that CRF; a one-point gpu_probe_qs fails
// every qsv job. So the box takes any separator, and anything that is not at
// least two distinct integers is refused (setCustomValidity), never trimmed.
function parseGrid(v) {
  const tokens = String(v).split(/[\s,，;；、]+/).filter(Boolean);
  const bad = tokens.find(t => !/^\d+$/.test(t));
  if (bad !== undefined) return { list: [], err: `「${bad}」不是整数` };
  const list = tokens.map(Number);
  return { list, err: new Set(list).size < 2 ? "至少要两个不同的整数" : "" };
}

function readValue(f, el) {
  const v = el.value;
  if (f.type === "bool") return v === "true";
  if (f.type === "csv") return parseGrid(v).list;
  if (f.type === "text" || f.type === "vulkan" || f.type === "select") return String(v).trim();
  if (f.type === "sycl") return parseInt(v, 10);
  const n = isFloat(f) ? parseFloat(v) : parseInt(v, 10);
  // An empty box is bounced back to the default on change (wire());
  // this only covers a submit that lands before the box ever lost focus.
  return Number.isFinite(n) ? n : (optDefaults ? optDefaults[f.key] : saved[f.key]);
}
const readKey = key => readValue(SPEC.get(key), ctlOf(key));

// A select whose value is not among its options silently reads back as "".
// That is how the GPU card blanked max_crf; never again for any select.
function ensureOption(el, value, label) {
  const v = String(value);
  if (![...el.options].some(o => o.value === v)) el.insertAdjacentHTML("beforeend", h`<option value="${v}">${label || v}</option>`);
}

function writeValue(f, el, v) {
  if (f.type === "bool") el.value = String(!!v);
  else if (f.type === "csv") { el.value = (v || []).join(","); el.setCustomValidity(parseGrid(el.value).err); }
  else if (f.type === "sycl") { ensureOption(el, v, `设备 ${v}`); el.value = String(v); }
  else if (f.type === "select") { if (v != null) ensureOption(el, v); el.value = v == null ? "" : String(v); }
  else el.value = v == null ? "" : String(v);
}

function fmtDefault(v, f) {
  if (f && f.type === "sycl") return v < 0 ? "关闭" : `设备 ${v}`;
  if (Array.isArray(v)) return v.join(",");
  if (v === true) return "开";
  if (v === false) return "关";
  return v === "" || v == null ? "空" : String(v);
}

// Re-triggerable: the class has to come off and the element reflow before the
// animation will run a second time.
function flash(el) {
  if (!el) return;
  el.classList.remove("is-flash");
  void el.offsetWidth;
  el.classList.add("is-flash");
}

// --------------------------------------------------------------- building

function buildForms() {
  const host = $("optGroups");
  for (const g of OPT_GROUPS) {
    const head = document.createElement("h3");
    head.className = "group"; head.id = g.id;
    head.innerHTML = h`${g.name} <span>${g.note}</span>`;
    const grid = document.createElement("div");
    grid.className = "grid"; grid.dataset.group = g.id;
    for (const f of g.fields) {
      const div = renderField(f, { ctl: "o", id: "f-", attr: "data-key", foot: true });
      fieldEl.set(f.key, div); grid.appendChild(div);
    }
    const back = document.createElement("p");
    back.className = "toindex"; back.dataset.group = g.id;
    back.innerHTML = '<a href="#rail">↑ 回索引</a>';
    host.append(head, grid, back);
  }
  for (const f of GPU_FIELDS) {
    const div = renderField(f, { ctl: "g", id: "f-", attr: "data-key", foot: true });
    fieldEl.set(f.key, div); $("gpuGrid").appendChild(div);
  }
  // The preset editor's ids are prefixed differently: vmaf_threads,
  // min_scene_len and probing_rate exist in both tables, and a second
  // id="f-vmaf_threads" would steal the docs page's deep link.
  for (const f of PRESET_FIELDS) {
    $("presetGrid").appendChild(renderField(f, { ctl: "pe", id: "pf-", attr: "data-pkey", foot: false }));
  }
  for (const s of document.querySelectorAll("main > .sheet:not(#sec-opt)")) {
    s.insertAdjacentHTML("beforeend", '<p class="toindex"><a href="#rail">↑ 回索引</a></p>');
  }
}

// The rail (>=900px) and the phone index grid are the same element, generated
// from SECTIONS + OPT_GROUPS. The hand-written rail it replaces drifted from
// the form, and hid every sub-entry on phones - the 字幕 group was unreachable.
function buildRail() {
  const links = SECTIONS.map(s => h`<a href="#${s.id}">${s.name}${s.count ? raw('<span class="n"></span>') : ""}</a>${
    (s.subs || []).map(sub => h`<a class="sub" href="#${sub.id}">${sub.name}<span class="n"></span></a>`)}`);
  $("rail").innerHTML = h`${links}<p class="rail__note">改动按区块分别保存；没保存就离开，浏览器会先问一句。</p>`;

  const as = [...$("rail").querySelectorAll("a")];
  const byId = new Map(as.map(a => [a.getAttribute("href").slice(1), a]));
  const io = new IntersectionObserver(entries => {
    for (const en of entries) {
      if (!en.isIntersecting) continue;
      const on = byId.get(en.target.id);
      for (const a of as) {
        a.classList.toggle("on", a === on);
        if (a === on) a.setAttribute("aria-current", "location"); else a.removeAttribute("aria-current");
      }
    }
  }, { rootMargin: "-20% 0px -70% 0px" });
  for (const id of byId.keys()) { const el = $(id); if (el) io.observe(el); }
}

// Where each rail badge counts from: the rendered block, never a second
// hard-coded key list. The old GPU_KEYS set made editing max_crf bump the
// 优化器 badge while the field sat on the GPU card.
function badgeScope(id) {
  if (id === "sec-gpu") return $("gpuGrid");
  if (id === "sec-opt") return $("optGroups");
  return document.querySelector(`.grid[data-group="${id}"]`);
}

// ----------------------------------------------------- receipts + dirtiness

// Each form's .formstate says what the form is holding until something
// happens, then says what happened - and keeps saying it until the next edit.
// The old toast vanished after 3.5s, usually while the operator was scrolled
// somewhere else on a 49-field page.
const sticky = new Set();

function stick(formId, text, tone) {
  sticky.add(formId);
  formState(RECEIPT[formId].el, text, tone);
}
function unstick(formId) {
  if (sticky.delete(formId)) idleReceipt(formId);
}

const unsavedIn = keys => [...keys].filter(k => k in saved && fieldEl.has(k) && !same(readKey(k), saved[k])).length;
const changedIn = scope => scope ? scope.querySelectorAll(".field.is-changed").length : 0;

function unsavedWorkers() {
  if (!savedWorkers) return 0;
  return ["concurrency", "av1an_workers"].filter(k => $(`w-${k}`).value !== String(savedWorkers[k])).length;
}
const unsavedSafety = () => savedSafety !== null && $("s-delete_source").value !== String(savedSafety) ? 1 : 0;
const unsavedKey = () => $("apiKeyInput").value.trim() !== apiKey() ? 1 : 0;

function unsavedTotal(except) {
  let n = 0;
  if (except !== "optimizerForm") n += unsavedIn(OPT_KEYS);
  if (except !== "gpuForm") n += unsavedIn(GPU_KEYS);
  if (except !== "workersForm") n += unsavedWorkers();
  if (except !== "safetyForm") n += unsavedSafety();
  if (except !== "keyForm") n += unsavedKey();
  return n;
}

const pending = n => `${n} 项待保存`;
const vsConfig = n => n ? `${n} 项与 config.yaml 不同` : "与 config.yaml 一致";

const RECEIPT = {
  keyForm: { el: "keyState", idle() {
    if (unsavedKey()) return [pending(1), "warn"];
    return apiKey() ? ["已保存在本浏览器"] : ["未设置：服务端若启用了密钥，写操作会返回 401", "warn"];
  } },
  workersForm: { el: "workersActive", idle() {
    const n = unsavedWorkers();
    if (n) return [pending(n), "warn"];
    return activeWorkers === null ? [""] : [`运行中的 worker：${activeWorkers}`];
  } },
  gpuForm: { el: "gpuState", idle() {
    const n = unsavedIn(GPU_KEYS);
    return n ? [pending(n), "warn"] : [optDefaults ? vsConfig(changedIn($("gpuGrid"))) : ""];
  } },
  optimizerForm: { el: "optState", idle() {
    const n = unsavedIn(OPT_KEYS);
    return n ? [pending(n), "warn"] : [optDefaults ? vsConfig(changedIn($("optGroups"))) : ""];
  } },
  safetyForm: { el: "safetyActive", idle() {
    if (unsavedSafety()) return [pending(1), "warn"];
    if (savedSafety === null) return [""];
    return savedSafety ? ["已开启：转码成功后源文件会被删除", "warn"] : ["关闭：保留源文件"];
  } },
};

function idleReceipt(formId) {
  if (sticky.has(formId)) return;
  const [text, tone] = RECEIPT[formId].idle();
  formState(RECEIPT[formId].el, text, tone);
}

function refreshMarks() {
  if (optDefaults) {
    for (const [key, div] of fieldEl) {
      const btn = div.querySelector(".field__dflt");
      if (!(key in optDefaults)) { btn.hidden = true; div.classList.remove("is-changed"); continue; }
      div.classList.toggle("is-changed", !same(readKey(key), optDefaults[key]));
      btn.textContent = `默认 ${fmtDefault(optDefaults[key], SPEC.get(key))} ⟲`;
      btn.hidden = false;
    }
  }
  for (const a of $("rail").querySelectorAll("a")) {
    const n = a.querySelector(".n");
    if (n) n.textContent = changedIn(badgeScope(a.getAttribute("href").slice(1))) || "";
  }
  for (const id of Object.keys(RECEIPT)) idleReceipt(id);

  $("optimizerForm").classList.toggle("is-dirty", unsavedIn(OPT_KEYS) > 0);
  $("gpuForm").classList.toggle("is-dirty", unsavedIn(GPU_KEYS) > 0);
  renderPulse();
}

// The page's one-line state, in the header because on a phone the header is
// the only thing still on screen 9 000px down the optimizer form: a failed
// load first (a dead backend must not look like a page of settled values),
// then unsaved edits, then how far the saved settings sit from config.yaml.
function renderPulse() {
  const p = $("globalDirty");
  if (loadErrors.size) {
    const all = [...loadErrors];
    // A dead backend fails all five loads with the same sentence; say it once.
    const one = all.length > 1 && all.every(([, m]) => m === all[0][1]);
    p.dataset.state = "down";
    p.textContent = one ? `读取失败：${all[0][1]}` : `${all[0][0]}读取失败：${all[0][1]}`;
    p.title = all.map(([w, m]) => `${w}读取失败：${m}`).join("\n");
    return;
  }
  p.removeAttribute("title");
  if (!booted) { p.dataset.state = "idle"; p.textContent = "正在读取设置…"; return; }
  const unsaved = unsavedTotal();
  p.dataset.state = unsaved ? "run" : "idle";
  p.textContent = unsaved ? pending(unsaved) : vsConfig(changedIn($("optGroups")) + changedIn($("gpuGrid")));
}

// Discarding edits is never silent: restoring defaults, closing the preset
// editor, and saving the key (which re-runs every loader) all ask first.
function confirmDiscard(n) {
  return confirmDlg({
    title: "有改动没保存",
    body: `还有 ${n} 项没保存，继续会丢掉这些改动。`,
    confirm: "丢掉改动并继续",
    cancel: "取消",
  });
}

window.addEventListener("beforeunload", e => {
  if (unsavedTotal() || editorDirty()) { e.preventDefault(); e.returnValue = ""; }
});

// -------------------------------------------------------------- write keys

function updateNeedKey() {
  const has = !!apiKey();
  for (const a of document.querySelectorAll(".needkey")) a.hidden = has;
}

async function busy(btn, fn) {
  btn.disabled = true;
  try { return await fn(); } finally { btn.disabled = false; }
}

// Apply what the server says. A key is overwritten unless it holds an unsaved
// edit that this reload is not about: saving the optimizer form re-reads the
// whole block, and that must not wipe an edit sitting unsaved on the GPU card.
function applyServer(values, scope) {
  for (const [key, v] of Object.entries(values)) {
    if (!fieldEl.has(key)) continue;
    const dirty = key in saved && !same(readKey(key), saved[key]);
    if (!dirty || scope.has(key)) writeValue(SPEC.get(key), ctlOf(key), v);
    saved[key] = v;
  }
}

// ---------------------------------------------------------------- loaders

// A form whose load failed keeps saying so until a later load succeeds; the
// header pulse carries the first failure so a dead backend is not a page of
// confident blanks.
const readErr = new Set();
function readFailed(formId, where, err) {
  readErr.add(formId);
  stick(formId, `读取失败：${err.message}`, "err");
  loadFailed(where, err);
}
function readOk(formId, where) {
  if (readErr.delete(formId)) unstick(formId);
  loadOk(where);
}
function loadFailed(where, err) {
  loadErrors.set(where, err.message);
  renderPulse();
}
function loadOk(where) {
  if (loadErrors.delete(where)) renderPulse();
}

async function loadWorkers() {
  try {
    const w = await api("/api/workers");
    savedWorkers = { concurrency: w.concurrency, av1an_workers: w.av1an_workers };
    activeWorkers = w.active_worker_count;
    $("w-concurrency").value = w.concurrency;
    $("w-av1an_workers").value = w.av1an_workers;
    readOk("workersForm", "并行处理");
    idleReceipt("workersForm");
  } catch (err) {
    readFailed("workersForm", "并行处理", err);
  }
}

// Field values have exactly one source: /api/settings/optimizer. /api/gpu
// used to write the GPU card back as well, from a settings object that does
// not carry max_crf or probe_dataset - so whichever of the two loads resolved
// last decided whether those fields were blank.
// Two saves inside one round trip start two loads; if the older reply landed
// last it rewrote the form and `saved` with the old values, and the next save
// sent them back. Only the newest load applies.
let optLoadSeq = 0;
async function loadOptimizer(scope) {
  const seq = ++optLoadSeq;
  const submits = [$("optimizerForm"), $("gpuForm")].map(f => f.querySelector('[type="submit"]'));
  try {
    const [values, d] = await Promise.all([api("/api/settings/optimizer"), api("/api/settings/optimizer/defaults")]);
    if (seq !== optLoadSeq) return;
    optDefaults = Object.assign({}, EXTRA_DEFAULTS, d.defaults);
    const own = {};
    for (const key of SPEC.keys()) if (key !== "vulkan_device" && key in values) own[key] = values[key];
    applyServer(own, scope);
    for (const b of submits) b.disabled = false;
    readOk("optimizerForm", "优化器"); readOk("gpuForm", "GPU 设置");
    refreshMarks();
  } catch (err) {
    if (seq !== optLoadSeq) return;
    // Posting a form that never loaded would send its blanks as values.
    for (const b of submits) b.disabled = true;
    readFailed("optimizerForm", "优化器", err);
    readFailed("gpuForm", "GPU 设置", err);
  }
}

async function loadSafety() {
  try {
    const s = await api("/api/settings/safety");
    savedSafety = !!s.delete_source;
    $("s-delete_source").value = String(savedSafety);
    readOk("safetyForm", "危险操作");
    idleReceipt("safetyForm");
  } catch (err) {
    readFailed("safetyForm", "危险操作", err);
  }
}

// ------------------------------------------------------------------- GPU

// Sentences this page writes read in the UI face; what the hardware reported
// (node paths, device names, ffmpeg's own error text) stays monospace.
const prose = t => h`<span class="devcard__prose">${t}</span>`;
function devcard(k, v, state) {
  return h`<div class="devcard"><p class="devcard__k"><span class="dot"${state ? h` data-state="${state}"` : ""}></span>${k}</p><p class="devcard__v">${v}</p></div>`;
}
const lines = arr => arr.map((x, i) => h`${i ? raw("<br>") : ""}${x}`);

function renderGpu(data, scope) {
  const s = data.status, g = data.settings;
  const sy = s.sycl || {}, vkStatus = s.vulkan || {};
  const nodes = s.render_nodes || [];
  const vk = vkStatus.devices || [];
  const cards = [];
  cards.push(devcard("渲染节点", nodes.length ? lines(nodes) : prose("没有 /dev/dri/renderD*：容器里看不到 GPU"), nodes.length ? "ok" : "err"));
  const count = sy.count;
  const countLine = count !== null && count !== undefined ? h`<span class="devcard__sub">${count} 个设备</span>` : "";
  if (!s.sycl_built) cards.push(devcard("SYCL 打分", prose("这个镜像的 libvmaf 没有 SYCL 后端"), "err"));
  else if (sy.device) cards.push(devcard("SYCL 打分", h`${sy.device}${countLine}`, "ok"));
  else cards.push(devcard("SYCL 打分", h`${sy.error || prose("后端在，但没有可用设备")}${countLine}`, count ? "warn" : "err"));
  cards.push(devcard("Vulkan（P5 转换）", vk.length ? lines(vk.map((d, i) => `${i}: ${d}`)) : (vkStatus.error || prose("没有 Vulkan 设备")),
                     vk.some(d => !/llvmpipe|\(cpu\)/i.test(d)) ? "ok" : vk.length ? "warn" : "err"));
  cards.push(devcard("硬件解码", prose(`QSV ${s.qsv ? "可用" : "不可用"} · VA-API ${s.vaapi ? "可用" : "不可用"}`), s.qsv || s.vaapi ? "ok" : "warn"));
  // /api/gpu has always returned this and the page never drew it: a container
  // without ffmpeg showed up only as four 不可用 cards and no reason why.
  if (s.ffmpeg === true) cards.push(devcard("ffmpeg", prose("找到了"), "ok"));
  else if (s.ffmpeg === false) cards.push(devcard("ffmpeg", prose("找不到 ffmpeg：上面几项都是靠它探测的，所以全是不可用"), "err"));
  else cards.push(devcard("ffmpeg", prose("服务端没有报告")));
  $("devices").innerHTML = h`${cards}`;

  // What the SAVED settings make of the hardware. The two fallbacks are the
  // only place a job's silent drop to CPU / soft decode is ever shown.
  const rows = [];
  const scoringGpu = g.vmaf_sycl_device >= 0 && s.sycl_built && sy.device;
  if (g.vmaf_sycl_device < 0) rows.push(["VMAF 打分", "CPU（关闭）"]);
  else if (scoringGpu) rows.push(["VMAF 打分", h`GPU · <span class="mono">${sy.device}</span>，源宽 ≥ ${g.vmaf_sycl_min_width}px 时`]);
  else rows.push(["VMAF 打分", `设置为设备 ${g.vmaf_sycl_device}，但现在不可用 → 作业会退回 CPU`, "warn"]);
  if (g.vulkan_device) rows.push(["P5 转换", h`按选择器 <span class="mono">"${g.vulkan_device}"</span>`]);
  else if (vk.length) rows.push(["P5 转换", h`自动 → <span class="mono">${vk[0]}</span>`]);
  else rows.push(["P5 转换", "自动，但没有设备", "warn"]);
  if (g.scenedetect_hwaccel === "off") rows.push(["分镜解码", "软解"]);
  else if (s.qsv) rows.push(["分镜解码", "QSV"]);
  else rows.push(["分镜解码", "auto，但 QSV 不可用 → 软解", "warn"]);
  if (g.reference_hwaccel === "off") rows.push(["打分参考解码", "软解"]);
  else if (s.vaapi) rows.push(["打分参考解码", "VA-API（探测文件和成品仍软解）"]);
  else rows.push(["打分参考解码", "auto，但 VA-API 不可用 → 软解", "warn"]);
  $("paths").innerHTML = h`${rows.map(([k, v, tone]) => h`<div><dt>${k}</dt><dd${tone ? h` data-tone="${tone}"` : ""}>${v}</dd></div>`)}`;
  formState("gpuNote", `检测于 ${fmtClock(s.checked_at)}`);

  // SYCL device picker. sycl.device names the CONFIGURED index (gpu.probe),
  // so that is the option that gets the name. Rebuilding the options must
  // not lose what the select currently holds - saved or being edited.
  const sel = ctlOf("vmaf_sycl_device");
  const keep = "vmaf_sycl_device" in saved ? readKey("vmaf_sycl_device") : g.vmaf_sycl_device;
  const named = Math.max(0, g.vmaf_sycl_device);
  const n = Math.max(count || (sy.device ? 1 : 0), keep >= 0 ? keep + 1 : 0);
  const opts = [h`<option value="-1">关闭：用 CPU 打分</option>`];
  for (let i = 0; i < n; i++) opts.push(h`<option value="${i}">设备 ${i}${i === named && sy.device ? `：${sy.device}` : ""}</option>`);
  sel.innerHTML = h`${opts}`;
  if (Number.isInteger(keep)) sel.value = String(keep);

  $("vkList").innerHTML = h`${vk.map((d, i) => h`<option value="${i}">${d}</option>`)}<option value="llvmpipe">软件渲染</option>`;

  // The only field value /api/gpu supplies: vulkan_device is not part of the
  // optimizer block, so /api/settings/optimizer cannot carry it.
  applyServer({ vulkan_device: g.vulkan_device || "" }, scope);
  refreshMarks();
}

async function loadGpu(refresh, scope) {
  try {
    renderGpu(await api(`/api/gpu${refresh ? "?refresh=true" : ""}`), scope);
    loadOk("GPU");
  } catch (err) {
    $("devices").innerHTML = devcard("检测失败", prose(err.message), "err");
    formState("gpuNote", "");
    loadFailed("GPU", err);
  }
}

async function selfCheck() {
  // The device picked in the form right now, not the saved one: this is how
  // an operator tries a device before committing to it.
  const dev = Math.max(0, parseInt(ctlOf("vmaf_sycl_device").value, 10) || 0);
  const out = $("checkOut");
  out.removeAttribute("data-tone");
  out.textContent = `正在设备 ${dev} 和 CPU 上各打一遍分…`;
  try {
    const r = await api("/api/gpu/selfcheck", { method: "POST", json: { device: dev } });
    if (r.ok) {
      out.dataset.tone = "ok";
      out.textContent = `GPU ${r.gpu.toFixed(4)} · CPU ${r.cpu.toFixed(4)} · 差 ${r.delta.toExponential(1)} · ${r.device} · ${r.elapsed}s\n和优化器开工前的自检完全一样：后端自报了设备，两边分数一致。`;
    } else {
      out.dataset.tone = "err";
      out.textContent = `未通过：${r.error || (!r.announced ? "后端没有自报 SYCL，可能是在 CPU 上算的" : `GPU ${r.gpu} 与 CPU ${r.cpu} 相差 ${r.delta}`)}\n作业会自动退回 CPU 打分。`;
    }
  } catch (err) {
    out.dataset.tone = "err";
    out.textContent = `自检失败：${err.message}`;
  }
}

// ---------------------------------------------------------------- presets

function presetSummary(p) {
  const out = [`crf=${p.crf} · preset=${p.preset} · passes=${p.passes} · ${p.pixel_format}`];
  if (p.engine === "optimizer") out.push(`engine=optimizer · metric=${p.target_metric}${p.target_quality ? ` · target=${p.target_quality}` : ""}`);
  else if (p.target_quality) out.push(`target_quality=${p.target_quality}`);
  if (p.film_grain) out.push(`film_grain=${p.film_grain}${p.film_grain_denoise ? "+denoise" : ""}`);
  if (p.luminance_qp_bias) out.push(`luminance_qp_bias=${p.luminance_qp_bias}`);
  if (p.vmaf_threads) out.push(`vmaf_threads=${p.vmaf_threads}`);
  if (p.additional_video_params) out.push(p.additional_video_params);
  return out;
}

// Layer 2 of a card: every field the editor writes, read-only, as the
// key=value it would be typed as in the queue page's custom parameters.
function presetAll(p) {
  return h`<details class="disclose"><summary>全部 ${PRESET_FIELDS.length} 项</summary><dl class="kv">${
    PRESET_FIELDS.map(f => {
      const v = p[f.key];
      const val = v === "" || v == null
        ? h`<span class="mono">${f.key}=</span><span class="pcard__empty">空</span>`
        : h`<span class="mono">${f.key}=${String(v)}</span>`;
      return h`<div><dt>${f.label}</dt><dd>${val}</dd></div>`;
    })}</dl></details>`;
}

function renderPresets() {
  const names = Object.keys(presets).sort();
  if (!names.length) {
    $("list").innerHTML = h`<div class="empty"><p>还没有预设</p><button type="button" class="btn" data-new-preset>＋ 新建预设</button></div>`;
    return;
  }
  $("list").innerHTML = h`${names.map(name => {
    const p = presets[name];
    return h`<article class="pcard">
  <div class="pcard__top"><h3 class="pcard__name">${name}</h3><span class="tag">${p._builtin ? "内置" : "自定义"}</span></div>
  <p class="pcard__sum">${lines(presetSummary(p))}</p>
  ${presetAll(p)}
  <div class="btnrow">
    <button type="button" class="btn btn--sm" data-edit="${name}">编辑</button>
    <button type="button" class="btn btn--sm btn--danger" data-del="${name}">删除</button>
  </div>
</article>`;
  })}`;
}

async function loadPresets() {
  try {
    presets = await api("/api/presets") || {};
    renderPresets();
    loadOk("编码预设");
  } catch (err) {
    $("list").innerHTML = h`<p class="formstate" data-tone="err">读取失败：${err.message}</p>`;
    loadFailed("编码预设", err);
  }
}

const pctl = key => $(`pe-${key}`);
let editorBaseline = null;

function writePreset(f, el, v) {
  if (f.type === "bool") el.value = String(!!v);
  else if (f.type === "select") {
    if (v == null || v === "") el.selectedIndex = 0;
    else { ensureOption(el, v); el.value = String(v); }
  } else el.value = v == null ? "" : String(v);
}

const snapshotEditor = () => [$("pe-name").value, ...PRESET_FIELDS.map(f => pctl(f.key).value)];
function editorChanges() {
  if (!editorBaseline || !$("presetDlg").open) return 0;
  const now = snapshotEditor();
  return editorBaseline.filter((v, i) => v !== now[i]).length;
}
const editorDirty = () => editorChanges() > 0;

// A new preset starts from the field defaults. The old editor started from
// whatever the previous edit had left in the form, because "keep the html
// default" only ever held on the very first open.
function openEditor(name) {
  const p = name ? presets[name] : null;
  $("pe-name").value = name || "";
  $("pe-name").disabled = !!name;
  for (const f of PRESET_FIELDS) {
    const v = p ? p[f.key] : (f.dflt !== undefined ? f.dflt : (f.type === "bool" ? false : ""));
    writePreset(f, pctl(f.key), v);
  }
  $("editorTitle").textContent = name ? `编辑预设 ${name}` : "新建预设";
  formState("editingHint", name ? "保存会覆盖这个预设。" : "");
  editorBaseline = snapshotEditor();
  openDlg("presetDlg");
  if (!name) $("pe-name").focus();
}

async function closeEditor() {
  const n = editorChanges();
  if (n && !await confirmDiscard(n)) return;
  editorBaseline = null;
  closeDlg("presetDlg");
}

function presetNumber(f, el) {
  const n = parseInt(el.value, 10);
  if (Number.isFinite(n)) return n;
  // `parseInt(...) || 0` turned an emptied box into 0 without a word -
  // luminance_qp_bias happens to mean off at 0, keyint does not.
  const d = f.dflt !== undefined ? f.dflt : 0;
  el.value = String(d); flash(el);
  return d;
}

function collectPreset() {
  const out = {};
  for (const f of PRESET_FIELDS) {
    const el = pctl(f.key);
    if (f.type === "bool") out[f.key] = el.value === "true";
    else if (PRESET_INT.has(f.key)) out[f.key] = presetNumber(f, el);
    else out[f.key] = String(el.value).trim();
  }
  return out;
}

// ----------------------------------------------------------------- search

function haystack(el) {
  return [".field__label", ".field__help", ".field__note"]
    .map(sel => { const x = el.querySelector(sel); return x ? x.textContent : ""; })
    .join(" ").toLowerCase();
}

// Filters the page down to matching fields. Every token has to match, so
// "probe max" finds probe_max_frames; label, key, help and the whole note are
// all searched, which is what makes a measurement findable by its number.
function applyFilter() {
  const query = $("q").value.trim();
  const tokens = query ? query.toLowerCase().split(/\s+/) : [];
  const only = $("onlyChanged").checked;
  const active = tokens.length > 0 || only;
  let hits = 0;
  for (const el of document.querySelectorAll("main .field")) {
    if (!el.__hay) el.__hay = haystack(el);
    const ok = tokens.every(t => el.__hay.includes(t)) && (!only || el.classList.contains("is-changed"));
    el.hidden = active && !ok;
    if (active && ok) hits++;
  }
  for (const grid of document.querySelectorAll("#optGroups .grid")) {
    const any = !!grid.querySelector(".field:not([hidden])");
    grid.hidden = !any;
    $(grid.dataset.group).hidden = !any;
    document.querySelector(`.toindex[data-group="${grid.dataset.group}"]`).hidden = !any;
  }
  for (const el of document.querySelectorAll("[data-hide-on-filter]")) el.hidden = active;
  for (const s of document.querySelectorAll("main > .sheet")) {
    s.hidden = active && !s.querySelector(".field:not([hidden])");
  }
  formState("hits", active ? `命中 ${hits} 项` : "", hits ? "dim" : "warn");
  $("noHits").hidden = !active || hits > 0;
  if (active && !hits) {
    $("noHitsText").textContent = query
      ? `没有匹配「${query}」的设置。试试 key 名，比如 probe_max_frames。`
      : "没有改过的设置：全部与 config.yaml 一致。";
  }
}

const NOTES_KEY = "av1tc_notes_open";
function setAllNotes(open) {
  for (const d of document.querySelectorAll("details.field__note")) d.open = open;
}

// ----------------------------------------------------------------- events

function wire() {
  document.addEventListener("input", e => {
    const t = e.target;
    if (t.id === "q") return applyFilter();
    if (t.closest("#presetDlg")) return;
    if (t.dataset.key && SPEC.get(t.dataset.key).type === "csv") t.setCustomValidity(parseGrid(t.value).err);
    const form = t.closest("form");
    if (form && RECEIPT[form.id]) unstick(form.id);
    refreshMarks();
  });

  // An emptied number box goes back to its default and the chip flashes:
  // the old page quietly read it as the default (or 0) and said nothing.
  document.addEventListener("change", e => {
    const t = e.target;
    if (t.type !== "number" || t.value.trim() !== "") return;
    if (t.dataset.key) {
      const d = optDefaults ? optDefaults[t.dataset.key] : saved[t.dataset.key];
      if (d === undefined) return;
      writeValue(SPEC.get(t.dataset.key), t, d);
      flash(fieldEl.get(t.dataset.key).querySelector(".field__dflt"));
      refreshMarks();
    } else if (t.dataset.pkey) {
      presetNumber(PRESET_FIELDS.find(f => f.key === t.dataset.pkey), t);
    } else if (savedWorkers && t.id.startsWith("w-")) {
      t.value = String(savedWorkers[t.id.slice(2)]);
      flash(t);
      refreshMarks();
    }
  });

  document.addEventListener("click", e => {
    const reset = e.target.closest("button[data-reset]");
    if (reset && optDefaults && reset.dataset.reset in optDefaults) {
      const key = reset.dataset.reset;
      writeValue(SPEC.get(key), ctlOf(key), optDefaults[key]);
      unstick(reset.closest("form").id);
      refreshMarks();
      return;
    }
    if (e.target.closest("[data-new-preset]")) { openEditor(""); return; }
    const edit = e.target.closest("button[data-edit]");
    if (edit) { openEditor(edit.dataset.edit); return; }
    const del = e.target.closest("button[data-del]");
    if (del) deletePreset(del.dataset.del);
  });

  // Native min/max validation stays on: it scrolls to the offending box and
  // says why. What it cannot do is focus a field the search has hidden -
  // the submit then silently does nothing - and its bubble is gone the moment
  // the operator scrolls. So: unhide, focus, and say it in the receipt too.
  for (const form of document.querySelectorAll("form")) {
    form.addEventListener("invalid", e => {
      if (form.__invalidSeen) return;          // one receipt per submit: the first bad field
      form.__invalidSeen = true;
      setTimeout(() => { form.__invalidSeen = false; });
      const el = e.target, field = el.closest(".field");
      if (field && field.closest("main") && field.offsetParent === null) {
        $("q").value = ""; $("onlyChanged").checked = false; applyFilter();
        setTimeout(() => el.focus());
      }
      const label = field ? field.querySelector(".field__label").firstChild.textContent.trim() : "";
      const msg = `${label}：${el.validationMessage}`;
      if (RECEIPT[form.id]) stick(form.id, msg, "err"); else formState("editingHint", msg, "err");
    }, true);
  }

  $("onlyChanged").addEventListener("change", applyFilter);
  $("clearFilter").addEventListener("click", () => {
    $("q").value = ""; $("onlyChanged").checked = false; applyFilter(); $("q").focus();
  });
  $("allNotes").addEventListener("change", e => {
    try { localStorage.setItem(NOTES_KEY, e.target.checked ? "1" : ""); } catch (_) { /* private mode */ }
    setAllNotes(e.target.checked);
  });

  // ---- API key
  $("keyForm").addEventListener("submit", async e => {
    e.preventDefault();
    const n = unsavedTotal("keyForm");
    if (n && !await confirmDiscard(n)) return;
    setApiKey($("apiKeyInput").value.trim());
    updateNeedKey();
    stick("keyForm", "已保存到本浏览器", "ok");
    loadAll(ALL_KEYS);
  });

  // ---- workers
  $("workersForm").addEventListener("submit", e => {
    e.preventDefault();
    const btn = e.submitter || $("workersForm").querySelector('[type="submit"]');
    const concurrency = parseInt($("w-concurrency").value, 10), av1an_workers = parseInt($("w-av1an_workers").value, 10);
    if (!concurrency || concurrency < 1) { stick("workersForm", "并行任务数至少为 1", "err"); return; }
    // NaN goes out as JSON null and api.py's int(None) answers 500; an empty
    // box is only possible when the workers load failed and nothing refilled it.
    if (!Number.isInteger(av1an_workers) || av1an_workers < 0) { stick("workersForm", "av1an 切片并行要填 0 或正整数", "err"); return; }
    busy(btn, async () => {
      try {
        const r = await api("/api/settings/workers", { method: "PUT", json: { concurrency, av1an_workers } });
        await loadWorkers();
        stick("workersForm", `已保存并行设置：${r.workers.concurrency} 个任务，av1an 切片 ${r.workers.av1an_workers || "自动"}`, "ok");
      } catch (err) { stick("workersForm", `保存失败：${err.message}`, "err"); }
      refreshMarks();
    });
  });

  // ---- GPU
  $("gpuRefresh").addEventListener("click", e => busy(e.currentTarget, () => loadGpu(true, new Set())));
  $("gpuCheck").addEventListener("click", e => busy(e.currentTarget, selfCheck));
  $("gpuForm").addEventListener("submit", e => {
    e.preventDefault();
    const body = {};
    for (const f of GPU_FIELDS) body[f.key] = readKey(f.key);
    // put_gpu writes vulkan_device into settings.json whenever the body has
    // it, which pinned config.yaml's value on every GPU save (and the 恢复
    // 默认值 DELETE never clears it). Send it only when it was changed.
    if ("vulkan_device" in saved && same(body.vulkan_device, saved.vulkan_device)) delete body.vulkan_device;
    busy(e.submitter || $("gpuForm").querySelector('[type="submit"]'), async () => {
      try {
        await api("/api/settings/gpu", { method: "PUT", json: body });
        await Promise.all([loadGpu(false, GPU_KEYS), loadOptimizer(GPU_KEYS)]);
        stick("gpuForm", body.vmaf_sycl_device >= 0 ? "已保存 GPU 设置：VMAF 将在 GPU 上打分，每个作业开工前自检一次" : "已保存 GPU 设置", "ok");
      } catch (err) { stick("gpuForm", `保存失败：${err.message}`, "err"); }
      refreshMarks();
    });
  });

  // ---- optimizer
  $("optimizerForm").addEventListener("submit", e => {
    e.preventDefault();
    const body = {};
    for (const f of OPT_FIELDS) body[f.key] = readKey(f.key);
    if (!body.probe_crfs.length) { stick("optimizerForm", "CRF 网格至少要一个整数", "err"); return; }
    busy(e.submitter || $("optimizerForm").querySelector('[type="submit"]'), async () => {
      try {
        const r = await api("/api/settings/optimizer", { method: "PUT", json: body });
        const n = Object.keys(r.overrides || {}).length;
        await loadOptimizer(OPT_KEYS);
        stick("optimizerForm", n ? `已保存，${n} 项写入 settings.json` : "已保存，与 config.yaml 一致，settings.json 里不再有 optimizer 块", "ok");
      } catch (err) { stick("optimizerForm", `保存失败：${err.message}`, "err"); }
      refreshMarks();
    });
  });

  $("optReset").addEventListener("click", async e => {
    const btn = e.currentTarget;
    // vulkan_device is not in the block DELETE drops, so it is not counted
    // as something this button puts back.
    const n = [...fieldEl].filter(([k, div]) => k !== "vulkan_device" && div.classList.contains("is-changed")).length;
    const unsaved = unsavedIn(OPT_KEYS) + unsavedIn(GPU_KEYS);
    const body = [`把优化器和 GPU 的 ${n} 项改动全部恢复为 config.yaml 的值？`];
    if (unsaved) body.push(`还有 ${unsaved} 项没保存，继续会丢掉这些改动。`);
    if (!await confirmDlg({ title: "恢复 config.yaml 默认值", body, confirm: "恢复默认值", danger: true })) return;
    busy(btn, async () => {
      try {
        await api("/api/settings/optimizer", { method: "DELETE" });
        await Promise.all([loadOptimizer(ALL_KEYS), loadGpu(false, ALL_KEYS)]);
        sticky.delete("gpuForm");
        stick("optimizerForm", "已恢复 config.yaml 默认值", "ok");
      } catch (err) { stick("optimizerForm", `恢复失败：${err.message}`, "err"); }
      refreshMarks();
    });
  });

  // ---- safety
  $("safetyForm").addEventListener("submit", async e => {
    e.preventDefault();
    const btn = e.submitter || $("safetySaveBtn");
    const enabled = $("s-delete_source").value === "true";
    if (enabled && !await confirmDlg({
      title: "开启删除源文件",
      body: ["开启后，每个转码成功的任务都会删除原始源文件。",
             "这一步不可逆：即使输出有问题，源文件也已经没了。",
             "确定开启？"],
      confirm: "开启删除源文件",
      danger: true,
    })) {
      $("s-delete_source").value = "false";
      refreshMarks();
      return;
    }
    busy(btn, async () => {
      try {
        // confirm:true is the server's own guard (api.py put_delete_source):
        // enabling without it is a 422, so the dialog above is not the only lock.
        await api("/api/settings/delete_source", { method: "PUT", json: { enabled, confirm: enabled } });
        await loadSafety();
        stick("safetyForm", enabled ? "已开启删除源文件" : "已关闭删除源文件", enabled ? "warn" : "ok");
      } catch (err) { stick("safetyForm", `保存失败：${err.message}`, "err"); }
      refreshMarks();
    });
  });

  // ---- preset editor
  $("presetForm").addEventListener("submit", async e => {
    e.preventDefault();
    const name = $("pe-name").value.trim();
    if (!name) { formState("editingHint", "请填写预设名称", "err"); $("pe-name").focus(); return; }
    // decisions.py reads preset "custom" as "no preset, overrides only", so a
    // preset by that name would be saved and never used.
    if (name === "custom") { formState("editingHint", "custom 是提交时「自定义」的保留名，换一个名字", "err"); $("pe-name").focus(); return; }
    // A new preset starts from the defaults now, so reusing a name would
    // silently replace that preset - the default one included, which is
    // also where custom jobs start from.
    if (!$("pe-name").disabled && presets[name] && !await confirmDlg({
      title: "覆盖预设",
      body: `已经有名为 ${name} 的预设，保存会用这次的参数覆盖它。`,
      confirm: "覆盖",
      danger: true,
    })) return;
    const body = collectPreset();
    await busy($("saveBtn"), async () => {
      try {
        await api(`/api/presets/${encodeURIComponent(name)}`, { method: "PUT", json: body });
        editorBaseline = null;
        closeDlg("presetDlg");
        notice(`已保存预设 ${name}`);
        loadPresets();
      } catch (err) { formState("editingHint", `保存失败：${err.message}`, "err"); }
    });
  });
  $("presetClose").addEventListener("click", closeEditor);
  $("presetCancel").addEventListener("click", closeEditor);
  // Esc and the backdrop would otherwise close straight past the unsaved
  // check: ui.js wires the backdrop at first open, so this capture-phase
  // listener gets to decide first.
  $("presetDlg").addEventListener("cancel", e => { e.preventDefault(); closeEditor(); });
  $("presetDlg").addEventListener("click", e => {
    const dlg = e.currentTarget;
    if (e.target !== dlg || !editorDirty()) return;
    const r = dlg.getBoundingClientRect();
    const inside = e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    if (inside) return;
    e.stopImmediatePropagation();
    closeEditor();
  }, true);
  $("presetDlg").addEventListener("input", () => {
    if ($("editingHint").dataset.tone === "err") formState("editingHint", $("pe-name").disabled ? "保存会覆盖这个预设。" : "");
  });
}

async function deletePreset(name) {
  const p = presets[name];
  if (!await confirmDlg({
    title: "删除预设",
    body: `删除预设 ${name}？${p && p._builtin ? "内置预设会恢复为 config.yaml 的值。" : ""}`,
    confirm: "删除预设",
    danger: true,
  })) return;
  try {
    await api(`/api/presets/${encodeURIComponent(name)}`, { method: "DELETE" });
    notice(p && p._builtin ? `已恢复 ${name} 为 config.yaml 的值` : `已删除 ${name}`);
    loadPresets();
  } catch (err) {
    notice(`删除失败：${err.message}`, "err");
  }
}

// ------------------------------------------------------------------- boot

async function loadAll(scope) {
  await Promise.all([loadWorkers(), loadOptimizer(scope), loadGpu(false, scope), loadSafety(), loadPresets()]);
  booted = true;
  refreshMarks();
  applyFilter();
}

buildForms();
buildRail();
wire();
new ResizeObserver(() => {
  document.documentElement.style.setProperty("--tools-h", `${$("tools").offsetHeight}px`);
}).observe($("tools"));
$("apiKeyInput").value = apiKey();
updateNeedKey();
$("q").placeholder = `搜索 ${document.querySelectorAll("main .field").length} 项设置`;
let notesOpen = false;
try { notesOpen = localStorage.getItem(NOTES_KEY) === "1"; } catch (_) { /* private mode */ }
$("allNotes").checked = notesOpen;
setAllNotes(notesOpen);
refreshMarks();
loadAll(ALL_KEYS);

})();
