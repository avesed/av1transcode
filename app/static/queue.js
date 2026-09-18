// Queue page (index.html). A job here runs for hours and the operator checks
// on it, often from a phone at night: the running job gets the top of the
// screen as a transport block, and everything else stays quiet.
//
// Every /api call goes through api() in ui.js; nothing here calls fetch().
// The IIFE is not style: ui.js and fields.js already own a dozen top-level
// names (esc, h, F, PRESET_FIELDS...), and one clashing top-level `const` is a
// SyntaxError that blanks the whole script - the old inline `const esc` would
// have done exactly that the day ui.js was loaded beside it.
(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const ACTIVE = new Set(["running", "analyzing"]);
  const CANCELLABLE = new Set(["running", "pending", "analyzing"]);
  const LOGGABLE = new Set(["running", "analyzing", "done", "failed", "cancelled"]);
  const FINISHED = ["done", "failed", "cancelled", "skipped"];

  // Fixed order, all eight always drawn, zero shown as a dash. The old stat
  // tiles existed only for non-zero counts and came in SQL GROUP BY order, so
  // the row re-shuffled itself under the operator's thumb as the queue
  // drained - and "analyzing" was not in the filter at all.
  const CHIP_KEYS = ["", "pending", "analyzing", "running", "done", "failed", "cancelled", "skipped"];

  // api.py caps /api/jobs at limit<=500. The list is newest-first on the
  // server, so without 载入更多 anything past row 100 simply never existed.
  const LIMITS = [100, 300, 500];

  // Ladder segments are the stages each engine actually reports, weighted by
  // their rough share of the wall clock (these are proportions of a bar, not
  // predictions - nothing reads them as time). The backend zeroes `progress`
  // at every stage change, so one optimizer job climbs to ~90% three times;
  // keeping finished segments filled is the only thing on screen that says
  // why.
  //
  // av1an is not the two-step 分析/编码 it looks like from outside:
  // transcoder.py drives stage_cb through scenedetect -> probing -> encoding
  // off av1an's own output and resets the bar each time, and it only probes
  // when there is a target_quality to probe for. The queue-level "analyzing"
  // (ffprobe + DV detection) is not a segment for either engine; the status
  // mark already says 分析中 while it runs.
  const LADDERS = {
    optimizer:  [["scenedetect", 1], ["probing", 3], ["encoding", 5], ["verifying", 1]],
    av1anProbe: [["scenedetect", 1], ["probing", 3], ["encoding", 6]],
    av1an:      [["scenedetect", 1], ["encoding", 9]],
  };

  const S = {
    filter: "",
    limitIdx: 0,
    jobs: [],             // the loaded page of /api/jobs, in display order
    active: [],           // running + analyzing, fetched on their own (see refresh)
    status: null,         // last /api/status
    counts: null,         // last /api/status .jobs
    loaded: false,
    presets: {},
    presetsOk: false,
    presetsBusy: false,
    open: new Set(),      // expanded job ids - shared by table and cards, so rotating a phone keeps a row open
    nowOpen: new Set(),   // transport blocks showing 详情
    seen: new Map(),      // job id -> furthest ladder segment observed this session
    cancelErr: new Map(), // job id -> the failure sentence, until a later cancel succeeds
    lastOk: 0,
    downErr: null,
    downNotice: null,
    retryAt: 0,
    ticker: 0,
    ctl: null,
  };

  // ------------------------------------------------------------ job facts

  const nowSec = () => Date.now() / 1000;
  const basename = (p) => String(p || "").split("/").pop();
  const jobName = (j) => basename(j.source) || j.id;
  const pct = (j) => Math.max(0, Math.min(100, Number(j.progress) || 0));
  const sum = (o) => Object.values(o || {}).reduce((a, b) => a + (Number(b) || 0), 0);
  const count = (k) => (S.counts && Number(S.counts[k])) || 0;

  const params = (j) => j.params || {};
  const overrides = (j) => params(j).overrides || {};
  const presetOf = (j) => S.presets[params(j).preset] || {};
  const engineOf = (j) => overrides(j).engine || presetOf(j).engine || "av1an";
  const targetQuality = (j) => overrides(j).target_quality || presetOf(j).target_quality || "";

  function segmentsFor(j) {
    if (engineOf(j) === "optimizer") return LADDERS.optimizer;
    return targetQuality(j) ? LADDERS.av1anProbe : LADDERS.av1an;
  }

  function ladderState(j) {
    const segs = segmentsFor(j);
    let live = segs.findIndex((s) => s[0] === j.stage);
    const seen = S.seen.has(j.id) ? S.seen.get(j.id) : -1;
    // queue.py writes stage="encoding" the moment a job turns running, before
    // the engine starts: RPU extraction reads the whole source in between
    // (minutes on a 60 GB DV P8 remux - every Better Call Saul episode). An
    // "encoding" with every counter at zero that this page never saw pass
    // through an earlier stage is that preamble; lighting 编码 would paint
    // 分镜 and 探测 as done and then snap back to 分镜.
    const zero = !Number(j.progress) && !j.progress_done && !j.progress_total;
    const preamble = live > 0 && j.stage === "encoding" && zero && seen < 0;
    if (preamble) live = -1;
    if (live >= 0 && live > seen) S.seen.set(j.id, live);
    // Not on the ladder at all (cancelling, a preamble): keep what this page
    // saw finish, light nothing.
    return { segs, live, doneBefore: live >= 0 ? live : Math.max(seen, 0), preamble };
  }

  // The wording is the old liveStats() verbatim; only the number formats
  // moved to ui.js (grouped frame counts, one decimal under 1.5 fps).
  function liveRead(j) {
    const total = j.progress_total || 0, done = j.progress_done || 0, fps = j.progress_fps || 0;
    if (j.stage === "scenedetect") {
      return total ? `场景检测 ${fmtNum(done)}/${fmtNum(total)} 帧` : (done ? `场景检测 ${fmtNum(done)} 处` : "");
    }
    if (j.stage === "probing") return total ? `探测 ${done}/${total} 段` : `探测 ${done} 段`;
    if (j.stage === "verifying") return total ? `校验 ${done}/${total} 段` : "";
    if (!total || (j.stage !== "encoding" && !fps)) return "";
    const frames = `${fmtNum(done)}/${fmtNum(total)} 帧`;
    return fps ? `${frames} · ${fmtFps(fps)} fps` : frames;
  }

  // Remaining time of the current stage. It used to print 剩余 183:07 for a
  // three-hour encode because the minutes never carried into hours.
  function etaText(j) {
    const total = j.progress_total || 0, done = j.progress_done || 0, fps = j.progress_fps || 0;
    if (!total || !fps || j.stage === "scenedetect" || j.stage === "probing" || j.stage === "verifying") return "";
    return `剩余 ${fmtDuration(Math.max(0, total - done) / fps)}`;
  }

  function stageWord(j) {
    if (ACTIVE.has(j.status) && ladderState(j).preamble) return "准备中";
    return stageLabel(j.stage) || statusLabel(j.status);
  }

  function runSeconds(j) {
    if (!j.started_at) return null;
    const end = j.finished_at || (ACTIVE.has(j.status) ? nowSec() : null);
    return end == null ? null : Math.max(0, end - j.started_at);
  }

  // size_before is written when the job turns running; before that the
  // analyzer's meta.size is the only size there is.
  const srcBytes = (j) => j.size_before || (j.meta && j.meta.size) || 0;

  function sizeText(j) {
    const a = srcBytes(j), b = j.size_after || 0;
    if (b) {
      if (!a) return `${DASH} → ${fmtBytes(b)}`;
      const A = fmtBytes(a).split(" "), B = fmtBytes(b).split(" ");
      return A[1] === B[1] ? `${A[0]} → ${B.join(" ")}` : `${A.join(" ")} → ${B.join(" ")}`;
    }
    if (!a) return DASH;
    return ACTIVE.has(j.status) ? `${fmtBytes(a)} → ${DASH}` : fmtBytes(a);
  }

  function ratioHtml(j) {
    const r = fmtRatio(srcBytes(j), j.size_after);
    return r ? h`<span class="ratio" data-dir="${r.charAt(0) === "↗" ? "up" : "down"}">${r}</span>` : "";
  }

  function overridesText(j, sep) {
    const ov = overrides(j);
    return Object.keys(ov).map((k) => {
      const v = ov[k];
      return `${k}=${v !== null && typeof v === "object" ? JSON.stringify(v) : v}`;
    }).join(sep);
  }

  // Built from the structured fields. meta.display is a different thing: the
  // analyzer's own one-line string (analyzer.py MediaInfo.display), the only
  // place trc= appears - the old table printed it where the file name belongs.
  function mediaFacts(m) {
    if (!m) return "";
    const out = [];
    if (m.codec) out.push(m.codec);
    if (m.width && m.height) out.push(`${m.width}×${m.height}`);
    if (m.fps) out.push(`${+Number(m.fps).toFixed(3)} fps`);
    if (m.duration) out.push(fmtDuration(m.duration));
    if (m.audio_count != null) out.push(`音轨 ${m.audio_count}`);
    out.push(m.is_hlg ? "HLG" : m.is_hdr ? "HDR" : "SDR");
    if (m.dovi && m.dovi.present) out.push(`DV-P${m.dovi.profile}`);
    return out.join(" · ");
  }

  function errLine(j) {
    const ce = S.cancelErr.get(j.id);
    if (ce) return h`<p class="row__err">${ce}</p>`;
    if (!j.error) return "";
    const first = String(j.error).split("\n")[0];
    // A skip reason or a cancel's exception text is not a failure; a pending
    // job that carries an error is on a retry, and that error is why.
    const red = j.status === "failed" || (j.status === "pending" && j.retries > 0);
    return red ? h`<p class="row__err">${first}</p>` : h`<p class="row__sub">${first}</p>`;
  }

  // ------------------------------------------------------------- markup

  function markHtml(j) {
    const st = STATUS[j.status] ? STATUS[j.status].state : "";
    return h`<span class="mark" data-state="${st}"><i></i>${statusLabel(j.status)}</span>`;
  }

  function ladderHtml(j, mini) {
    const L = ladderState(j);
    const p = pct(j);
    const read = mini ? "" : liveRead(j);
    const segs = L.segs.map(([st, w], i) => {
      const state = i === L.live ? "live" : i < L.doneBefore ? "done" : "todo";
      const name = stageShort(st);
      const fill = state === "live" ? raw(` style="width:${p}%"`) : "";
      if (mini) return h`<li class="ladder__seg" data-state="${state}" style="--w:${w}"><b class="ladder__fill"${fill}></b></li>`;
      const label = state === "live" ? `${name} 进行中 ${Math.floor(p)}%${read ? "，" + read : ""}`
        : state === "done" ? `${name} 已完成` : `${name} 未开始`;
      return h`<li class="ladder__seg" data-state="${state}" style="--w:${w}" aria-label="${label}"${state === "live" ? raw(' aria-current="step"') : ""}><span class="ladder__name" aria-hidden="true">${name}</span><b class="ladder__fill"${fill}></b>${state === "live" && read ? h`<span class="ladder__read" aria-hidden="true">${read}</span>` : ""}</li>`;
    });
    return mini ? h`<ol class="ladder" aria-hidden="true">${segs}</ol>` : h`<ol class="ladder" aria-label="作业阶段">${segs}</ol>`;
  }

  function copyBtn(value, k) {
    return h`<button type="button" class="btn btn--quiet btn--sm" data-copy="${value}" data-k="${k}">复制</button>`;
  }

  function kv(dt, dd) {
    return h`<div><dt>${dt}</dt><dd>${dd}</dd></div>`;
  }

  function pathDd(p, aux, k) {
    return h`<span class="mono">${p}</span>${aux ? h`<span class="aux">${aux}</span>` : ""}${copyBtn(p, k)}`;
  }

  function actionsHtml(j, where) {
    const k = `${where}:${j.id}`;
    // A pending job with retries has a log from the attempt that failed.
    const log = LOGGABLE.has(j.status) || j.retries > 0;
    const can = CANCELLABLE.has(j.status) && j.stage !== "cancelling";
    if (!log && !can) return "";
    return h`<div class="btnrow">${log ? h`<button type="button" class="btn" data-log="${j.id}" data-k="${k}:log">日志</button>` : ""}${can ? h`<button type="button" class="btn" data-cancel="${j.id}" data-k="${k}:cancel">取消</button>` : ""}</div>`;
  }

  // Everything /api/jobs returns that a row cannot show. Rendered only while
  // open: a collapsed pending row would otherwise re-render every poll for
  // the sake of a hidden, ticking wait time.
  function detailHtml(j, where, withActions) {
    const m = j.meta || null;
    const k = `${where}:${j.id}`;
    const pending = j.status === "pending";
    const waitEnd = j.started_at || (pending ? nowSec() : j.finished_at);
    const waited = j.created_at && waitEnd ? waitEnd - j.created_at : null;
    const run = runSeconds(j);
    const ov = overridesText(j, " ");
    const left = [
      kv("作业 id", h`<span class="mono">${j.id}</span>${copyBtn(j.id, k + ":id")}`),
      kv("源", j.source ? pathDd(j.source, fmtBytes(srcBytes(j), ""), k + ":src") : DASH),
      kv("输出", j.output_path ? pathDd(j.output_path, j.size_after ? fmtBytes(j.size_after) : "", k + ":out") : DASH),
      j.rpu_path ? kv("RPU", pathDd(j.rpu_path, "", k + ":rpu")) : "",
      kv("预设", params(j).preset || DASH),
      kv("参数", ov ? h`<span class="mono">${ov}</span>` : DASH),
      kv("原始键", h`<span class="mono">status=${j.status} stage=${j.stage || '""'}</span>`),
    ];
    const right = [
      kv("源信息", mediaFacts(m) || DASH),
      kv("媒体串", m && m.display ? h`<span class="mono">${m.display}</span>` : DASH),
      kv("提交", fmtClock(j.created_at, true)),
      kv(pending ? "已等待" : "等待", fmtDuration(waited)),
      kv("开始", j.started_at ? fmtClock(j.started_at, true) : DASH),
      kv(ACTIVE.has(j.status) ? "已跑" : "用时", fmtDuration(run)),
      j.finished_at ? kv("结束", fmtClock(j.finished_at, true)) : "",
      // workers.max_retries is not in any API response, so no "/2": a
      // denominator copied from the config.yaml default would lie the day
      // someone changes it.
      kv("重试", `${j.retries || 0} 次`),
    ];
    const err = j.error
      ? h`<dl class="kv detail__full">${kv(j.status === "skipped" ? "跳过原因" : "错误", h`<pre class="errtext">${j.error}</pre>`)}</dl>`
      : "";
    return h`<div class="detail"><dl class="kv">${left}</dl><dl class="kv">${right}</dl>${err}</div>${withActions ? actionsHtml(j, where) : ""}`;
  }

  function progHtml(j, pos) {
    if (ACTIVE.has(j.status)) {
      return h`<p class="prog">${stageWord(j)} <b>${Math.floor(pct(j))}%</b></p>${ladderHtml(j, true)}`;
    }
    if (j.status === "pending") {
      return h`<p class="prog">${pos ? `队列第 ${pos} 位` : DASH}</p>${j.stage === "retry" ? h`<p class="prog">${stageLabel("retry")}（第 ${j.retries} 次）</p>` : ""}`;
    }
    return h`<p class="prog">${j.finished_at ? `结束 ${fmtClock(j.finished_at, true)}` : DASH}</p>`;
  }

  function presetHtml(j) {
    const ov = overridesText(j, ", ");
    return h`${params(j).preset || DASH}${ov ? h`<span class="mono">${ov}</span>` : ""}`;
  }

  function fileHtml(j, ctrl, open, withPath) {
    const m = j.meta || {};
    return h`<button type="button" class="row__toggle" aria-expanded="${String(!!open)}" aria-controls="${ctrl}" data-toggle="${j.id}" data-k="${ctrl}">${jobName(j)}</button>${m.display ? h`<p class="row__sub mono">${m.display}</p>` : ""}${withPath && j.source ? h`<p class="row__sub mono">${j.source}</p>` : ""}${errLine(j)}`;
  }

  function trHtml(j, pos) {
    const open = S.open.has(j.id);
    const ctrl = "d-" + j.id;
    const run = runSeconds(j);
    return h`<tbody data-key="${j.id}"><tr class="row" data-state="${j.status}"><td class="row__mark">${markHtml(j)}</td><td class="row__file">${fileHtml(j, ctrl, open, true)}</td><td class="row__prog">${progHtml(j, pos)}</td><td class="preset">${presetHtml(j)}</td><td class="row__num">${sizeText(j)}${ratioHtml(j)}</td><td class="row__num">${run == null ? DASH : fmtDuration(run)}</td></tr><tr class="row__detail" id="${ctrl}"${open ? "" : raw(" hidden")}><td colspan="6">${open ? detailHtml(j, "t", true) : ""}</td></tr></tbody>`;
  }

  // Phone card: 状态 + one number / file name / 阶段 · 预设 · 大小.
  function cardNum(j, pos) {
    if (ACTIVE.has(j.status)) return `${Math.floor(pct(j))}%`;
    if (j.status === "pending") return pos ? `第 ${pos} 位` : "";
    if (j.status === "done") return ratioHtml(j);
    return j.finished_at ? fmtAgo(j.finished_at) : "";
  }

  function cardLine(j) {
    const bits = [];
    // The ladder's short name, not STAGE.cn: for an encode that is 转码中,
    // the same word the status mark two lines up already says.
    if (ACTIVE.has(j.status)) bits.push(ladderState(j).preamble ? "准备中" : (stageShort(j.stage) || statusLabel(j.status)));
    else if (j.stage === "retry") bits.push(stageLabel("retry"));
    if (params(j).preset) bits.push(params(j).preset);
    bits.push(sizeText(j));
    const run = runSeconds(j);
    if (!ACTIVE.has(j.status) && run != null) bits.push(`用时 ${fmtDuration(run)}`);
    return h`<p class="card">${bits.join(" · ")}</p>`;
  }

  function liHtml(j, pos) {
    const open = S.open.has(j.id);
    const ctrl = "m-" + j.id;
    return h`<li class="row" data-state="${j.status}" data-key="${j.id}"><div class="row__mark">${markHtml(j)}</div><div class="row__num">${cardNum(j, pos)}</div><div class="row__file">${fileHtml(j, ctrl, open, false)}</div><div class="row__prog">${cardLine(j)}${ACTIVE.has(j.status) ? ladderHtml(j, true) : ""}</div><div class="row__detail" id="${ctrl}"${open ? "" : raw(" hidden")}>${open ? detailHtml(j, "m", true) : ""}</div></li>`;
  }

  function nowHtml(j) {
    const m = j.meta || {};
    const open = S.nowOpen.has(j.id);
    const eta = etaText(j);
    const run = runSeconds(j);
    const ov = overridesText(j, " ");
    const ce = S.cancelErr.get(j.id);
    const k = "n:" + j.id;
    const can = j.stage !== "cancelling";
    const facts = [
      params(j).preset ? h`<span>${params(j).preset}</span>` : "",
      ov ? h`<span class="mono">${ov}</span>` : "",
      srcBytes(j) ? h`<span>源 <b>${fmtBytes(srcBytes(j))}</b></span>` : "",
      run != null ? h`<span>已跑 <b>${fmtDuration(run)}</b></span>` : "",
      j.retries ? h`<span>重试 <b>${j.retries}</b> 次</span>` : "",
    ];
    return h`<article class="now" data-key="${j.id}" aria-labelledby="nt-${j.id}">
<h3 class="now__name" id="nt-${j.id}">${jobName(j)}</h3>
<div class="now__top"><p class="figure">${Math.floor(pct(j))}<span class="figure__u">%</span></p><p class="now__stage"><span>${stageWord(j)}</span>${eta ? h`<b>${eta}</b>` : ""}</p></div>
${ladderHtml(j, false)}
${m.display ? h`<p class="now__line mono">${m.display}</p>` : ""}
${j.source ? h`<p class="now__line mono">${j.source}</p>` : ""}
<p class="now__facts">${facts}</p>
${ce ? h`<p class="row__err">${ce}</p>` : ""}
<div class="btnrow"><button type="button" class="btn" data-log="${j.id}" data-k="${k}:log">日志</button>${can ? h`<button type="button" class="btn" data-cancel="${j.id}" data-k="${k}:cancel">取消</button>` : ""}<button type="button" class="btn" data-nowdetail="${j.id}" aria-expanded="${String(!!open)}" aria-controls="nd-${j.id}" data-k="${k}:det">${open ? "收起" : "详情"}</button></div>
<div class="now__detail" id="nd-${j.id}"${open ? "" : raw(" hidden")}>${open ? detailHtml(j, "n", false) : ""}</div>
</article>`;
  }

  // ------------------------------------------------------------ rendering

  // Keyed replace-if-changed. The list is redrawn every 4 s; replacing all
  // of it would drop keyboard focus and wipe a half-made text selection of a
  // path every time, for rows whose data did not move.
  const tpl = document.createElement("template");
  function patch(host, items) {
    const old = new Map();
    for (const el of host.querySelectorAll(":scope > [data-key]")) old.set(el.dataset.key, el);
    let prev = null;
    for (const it of items) {
      let el = old.get(it.key);
      old.delete(it.key);
      if (!el || el.__html !== it.html) {
        tpl.innerHTML = it.html;
        const fresh = tpl.content.firstElementChild;
        fresh.__html = it.html;
        if (el) el.replaceWith(fresh);
        el = fresh;
      }
      const next = prev ? prev.nextElementSibling : host.querySelector(":scope > [data-key]");
      if (el !== next) {
        if (prev) prev.after(el);
        else if (next) next.before(el);
        else host.appendChild(el);
      }
      prev = el;
    }
    for (const el of old.values()) el.remove();
  }

  function selecting(host) {
    const sel = window.getSelection ? window.getSelection() : null;
    return !!(sel && !sel.isCollapsed && sel.rangeCount && host.contains(sel.getRangeAt(0).commonAncestorContainer));
  }

  function redraw(host, items) {
    if (selecting(host)) return;   // caught up on the first poll after the selection ends
    const a = document.activeElement;
    const k = a && host.contains(a) && a.dataset ? a.dataset.k : null;
    patch(host, items);
    if (k && !host.contains(document.activeElement)) {
      const el = host.querySelector(`[data-k="${CSS.escape(k)}"]`);
      if (el) el.focus({ preventScroll: true });
    }
  }

  function positions(jobs) {
    const pend = jobs.filter((j) => j.status === "pending");
    const map = new Map();
    // The server hands out pending jobs oldest-first (db.next_pending), so the
    // sorted order IS the run order - but only if every pending job is loaded.
    // With some cut off by the limit, a number would be a confident lie.
    if (!S.counts || !pend.length || pend.length !== count("pending")) return map;
    pend.forEach((j, i) => map.set(j.id, i + 1));
    return map;
  }

  function renderNow() {
    const host = $("nowList");
    for (const s of host.querySelectorAll(".skel")) s.remove();
    const list = S.active;
    $("nowCount").textContent = list.length > 1 ? `${list.length} 个` : "";
    $("nowEmpty").hidden = list.length > 0;
    $("nowEmpty").textContent = S.status && S.status.workers_running === false
      ? "没有作业在跑。worker 停止，排队的任务不会开始。" : "没有作业在跑。";
    redraw(host, list.map((j) => ({ key: j.id, html: String(nowHtml(j)) })));
  }

  function renderChips() {
    const known = !!S.counts;
    for (const b of $("chips").children) {
      const k = b.dataset.status;
      const n = k ? count(k) : sum(S.counts);
      b.setAttribute("aria-pressed", String(S.filter === k));
      b.querySelector(".chip__n").textContent = known && n ? String(n) : DASH;
    }
  }

  function renderJobs() {
    $("jobsSkel").hidden = S.loaded;
    if (!S.loaded) return;
    const jobs = S.jobs;
    const pos = positions(jobs);
    redraw($("jobsTable"), jobs.map((j) => ({ key: j.id, html: String(trHtml(j, pos.get(j.id))) })));
    redraw($("jobsList"), jobs.map((j) => ({ key: j.id, html: String(liHtml(j, pos.get(j.id))) })));
    $("tableWrap").hidden = !jobs.length;

    const empty = $("jobsEmpty");
    empty.hidden = jobs.length > 0;
    const want = S.filter ? "f:" + S.filter : "all";
    if (!jobs.length && empty.dataset.for !== want) {
      empty.dataset.for = want;
      empty.innerHTML = S.filter
        ? h`<p>没有「${statusLabel(S.filter)}」的任务。</p><button type="button" class="btn" data-status="">看全部</button>`
        : h`<p>暂无任务。填一个绝对路径，或点「浏览…」从 /media 开始挑。</p><button type="button" class="btn" data-open="submit">提交转码</button>`;
    }

    const total = S.filter ? count(S.filter) : sum(S.counts);
    const n = jobs.length;
    let t = "";
    if (n) {
      t = S.counts ? `共 ${total} 条，已载入 ${n}` : `已载入 ${n}`;
      if (n >= LIMITS[LIMITS.length - 1] && total > n) t += "（API 一次最多返回 500 条）";
    }
    $("tally").textContent = t;
    $("moreBtn").hidden = !(n >= LIMITS[S.limitIdx] && S.limitIdx < LIMITS.length - 1);
  }

  function renderPulse() {
    const p = $("pulse"), main = $("pulseMain"), at = $("pulseAt");
    if (S.downErr) {
      p.dataset.state = "down";
      main.textContent = "与服务端失去联系";
      const s = Math.max(0, Math.ceil((S.retryAt - Date.now()) / 1000));
      at.textContent = s ? `，${s} 秒后重试` : "，正在重试…";
      return;
    }
    if (!S.status) {
      p.dataset.state = "idle";
      main.textContent = "正在连接…";
      at.textContent = "";
      return;
    }
    const r = count("running"), a = count("analyzing"), q = count("pending");
    const stopped = S.status.workers_running === false;
    const parts = [stopped ? "worker 停止" : r ? `转码中 ${r}` : "worker 运行中"];
    if (a) parts.push(`分析中 ${a}`);
    parts.push(`排队 ${q}`);
    // Stopped workers with a queue is broken, not idle: nothing will move.
    p.dataset.state = stopped ? "down" : (r || a) ? "run" : "idle";
    main.textContent = parts.join(" · ");
    at.textContent = S.lastOk ? ` · ${fmtClock(S.lastOk)} 更新` : "";
  }

  function render() {
    renderPulse();
    renderChips();
    if (S.loaded) renderNow();
    renderJobs();
  }

  // ------------------------------------------------------------ polling

  function rank(j) { return ACTIVE.has(j.status) ? 0 : j.status === "pending" ? 1 : 2; }
  const asc = (a, b) => (a == null ? Infinity : a) - (b == null ? Infinity : b) || 0;

  // /api/jobs is created_at DESC. Submit a 40-file directory and the one job
  // actually running sinks to the bottom - past row 100 once the queue is big
  // enough. Order here is the real execution order instead: running by start,
  // pending by the order db.next_pending will hand them out, finished newest
  // first.
  function sortJobs(list) {
    return list.slice().sort((a, b) => {
      const ra = rank(a), rb = rank(b);
      if (ra !== rb) return ra - rb;
      if (ra === 0) return asc(a.started_at, b.started_at);
      if (ra === 1) return asc(a.created_at, b.created_at);
      return (b.finished_at || b.created_at || 0) - (a.finished_at || a.created_at || 0);
    });
  }

  async function refresh() {
    const filter = S.filter, limitIdx = S.limitIdx;
    const qs = `limit=${LIMITS[limitIdx]}` + (filter ? `&status=${encodeURIComponent(filter)}` : "");
    // The transport block fetches its jobs on its own: under a status filter,
    // or with more than a page of newer submissions, the running job is not
    // in the list at all.
    const res = await Promise.allSettled([
      api("/api/status"),
      api("/api/jobs?" + qs),
      api("/api/jobs?status=running&limit=50"),
      api("/api/jobs?status=analyzing&limit=50"),
    ]);
    const [st, jb, run, ana] = res;
    if (st.status === "fulfilled" && st.value && typeof st.value === "object") {
      S.status = st.value;
      S.counts = st.value.jobs || {};
    }
    // A chip tapped while this request was out: poll().now() drops a call
    // made mid-flight, so the answer to the old filter is thrown away and the
    // new one asked for straight after, instead of 4 s later.
    const stale = filter !== S.filter || limitIdx !== S.limitIdx;
    if (stale) setTimeout(() => S.ctl.now(), 0);
    if (!stale && jb.status === "fulfilled" && Array.isArray(jb.value)) {
      S.jobs = sortJobs(jb.value);
      S.loaded = true;
    }
    if (run.status === "fulfilled" && ana.status === "fulfilled") {
      // The two lists are fetched in parallel, so a job that moves from
      // analyzing to running between them is in both: keep one copy, or it
      // is drawn twice with the same element ids.
      const byId = new Map([].concat(ana.value || [], run.value || []).map((x) => [x.id, x]));
      S.active = sortJobs([...byId.values()]);
    }
    for (const id of S.cancelErr.keys()) {
      const j = S.jobs.find((x) => x.id === id) || S.active.find((x) => x.id === id);
      if (j && !CANCELLABLE.has(j.status)) S.cancelErr.delete(id);
    }
    if (!S.presetsOk) loadPresets();
    render();
    const bad = res.find((r) => r.status === "rejected");
    if (bad) throw bad.reason;
  }

  function onPollOk() {
    S.lastOk = Date.now();
    if (S.downErr) {
      S.downErr = null;
      clearInterval(S.ticker);
      $("main").classList.remove("is-stale");
      if (S.downNotice) { S.downNotice.dismiss(); S.downNotice = null; }
      notice("已恢复连接");
    }
    renderPulse();
  }

  // refresh() used to have no try/catch: restart the backend and the page
  // kept a screenful of confident numbers from before, forever. Now the bar
  // turns red with a countdown, the page desaturates, and one persistent
  // notice says when the numbers are from.
  function onPollError(err, wait) {
    S.downErr = err || new Error("");
    S.retryAt = Date.now() + wait;
    $("main").classList.add("is-stale");
    const why = err && err.status ? `HTTP ${err.status}` : "网络不通";
    const since = S.lastOk ? `，数据停在 ${fmtClock(S.lastOk)}` : "";
    const text = `与服务端失去联系（${why}）${since}。`;
    if (S.downNotice && S.downNotice.el.isConnected) S.downNotice.el.querySelector("p").textContent = text;
    else S.downNotice = notice(text, "err");
    clearInterval(S.ticker);
    S.ticker = setInterval(renderPulse, 1000);
    renderPulse();
  }

  // ------------------------------------------------------------ actions

  function findJob(id) {
    return S.active.find((j) => j.id === id) || S.jobs.find((j) => j.id === id) || null;
  }

  // api() turns a bodiless failure into the bare "HTTP 404"; keep the old
  // parenthesised form for that case, the sentence itself for the rest.
  // Callers pass the whole word (取消失败, not 取消) so each receipt can be
  // found by grepping for exactly what the operator saw.
  function failText(what, err) {
    const m = (err && err.message) || "";
    return err && err.status && m === `HTTP ${err.status}` ? `${what}（HTTP ${err.status}）` : `${what}：${m}`;
  }

  async function cancelJob(id) {
    const j = findJob(id);
    const name = j ? jobName(j) : id;
    // One 48px tap next to 日志 used to throw away hours of encode with no
    // question asked. Pending and analyzing jobs have nothing to lose yet.
    if (j && j.status === "running") {
      const run = runSeconds(j);
      const ok = await confirmDlg({
        title: "取消转码",
        body: [
          `取消「${name}」？${run != null ? `它已经跑了 ${fmtDuration(run)}。` : ""}`,
          "正在编码的任务会被中断，它已经写出的输出文件会被删除。",
        ],
        confirm: "取消转码",
        cancel: "继续转码",
      });
      if (!ok) return;
      // The dialog can sit open while the job finishes, and the server's
      // cancel does not look at finished statuses: it would turn 已完成
      // into 已取消. Re-read the job before sending.
      try {
        const now = await api(`/api/jobs/${encodeURIComponent(id)}`);
        if (now && !["pending", "analyzing", "running"].includes(now.status)) {
          notice(`${name} 已经结束（${statusLabel(now.status)}），没有取消`);
          S.ctl.now();
          return;
        }
      } catch (_) { /* the cancel below reports its own failure */ }
    }
    try {
      await api(`/api/jobs/${encodeURIComponent(id)}/cancel`, { method: "POST" });
      S.cancelErr.delete(id);
      // What the server did, not what was asked: a running job only gets
      // stage=cancelling and stops at the engine's next check.
      if (j && j.status === "running") notice(`已请求取消 ${name}，引擎停下后变为已取消`);
      else notice(`已取消 ${name}`);
    } catch (err) {
      // 409: it finished between the last poll and the click. Nothing failed.
      if (err && err.status === 409) {
        notice(`${name} 已经结束，没有取消`);
        render(); S.ctl.now();
        return;
      }
      // Kept on the row as well as in the notice: a cancel that failed must
      // not look like a job that is ignoring the button (commit ca1b68c).
      const t = failText("取消失败", err);
      S.cancelErr.set(id, t);
      notice(t, "err");
    }
    render();
    S.ctl.now();
  }

  // POST /api/cancel is manager.cancel() with no id, which is
  // db.cancel_pending(): pending and analyzing rows only, running jobs are
  // not touched. And a worker already analyzing a row writes status=running
  // over the "cancelled" when it finishes. So this promises pending only.
  async function cancelQueued() {
    const n = count("pending"), a = count("analyzing");
    const ok = await confirmDlg({
      title: "取消全部排队",
      body: [
        "取消队列里所有排队中的任务？正在转码的任务不受影响，要停它请用它自己的「取消」。",
        `将取消 ${n + a} 个任务。`,
      ],
      confirm: "取消全部排队",
      cancel: "返回",
    });
    if (!ok) return;
    try {
      const r = await api("/api/cancel", { method: "POST" });
      notice(`已取消 ${r && r.cancelled != null ? r.cancelled : n} 个任务`);
    } catch (err) {
      notice(failText("取消失败", err), "err");
    }
    S.ctl.now();
  }

  async function prune() {
    const n = FINISHED.reduce((a, k) => a + count(k), 0);
    const ok = await confirmDlg({
      title: "清除历史记录",
      body: [
        "删除所有已结束（完成/失败/取消/跳过）的任务记录？此操作不可恢复。",
        `将删除 ${n} 条记录及其日志文件。`,
      ],
      confirm: "清除历史记录",
    });
    if (!ok) return;
    try {
      const r = await api("/api/jobs/prune", { json: {} });
      const p = r && r.pruned, l = r && r.logs_removed;
      notice(p != null && l != null ? `已清除 ${p} 条记录、${l} 份日志` : "已清除历史记录");
    } catch (err) {
      notice(failText("清除失败", err), "err");
    }
    S.ctl.now();
  }

  function setFilter(k) {
    S.filter = k;
    S.limitIdx = 0;
    renderChips();
    S.ctl.now();
  }

  function flashCopy(btn, ok) {
    const was = btn.textContent;
    btn.textContent = ok ? "已复制" : "复制失败";
    setTimeout(() => { if (btn.isConnected) btn.textContent = was; }, 1500);
  }

  function onClick(e) {
    const t = e.target.closest("button");
    if (!t) return;
    const d = t.dataset;
    if (d.toggle) {
      if (S.open.has(d.toggle)) S.open.delete(d.toggle); else S.open.add(d.toggle);
      renderJobs();
    } else if (d.nowdetail) {
      if (S.nowOpen.has(d.nowdetail)) S.nowOpen.delete(d.nowdetail); else S.nowOpen.add(d.nowdetail);
      renderNow();
    } else if (d.cancel) {
      cancelJob(d.cancel);
    } else if (d.log) {
      // A navigation, not an api() call: GET /api/jobs/{id}/log is unkeyed
      // (api.py get_log has no _auth), which is the only reason a plain new
      // tab can read it.
      window.open(`/api/jobs/${encodeURIComponent(d.log)}/log`, "_blank", "noopener");
    } else if (d.copy !== undefined) {
      copyText(d.copy).then((ok) => flashCopy(t, ok));
    } else if (d.status !== undefined) {
      setFilter(t.classList.contains("chip") && d.status === S.filter ? "" : d.status);
    } else if (d.open === "submit") {
      openSubmit();
    } else if (d.browse) {
      openBrowse(d.browse);
    } else if (d.close !== undefined) {
      closeDlg(t.closest("dialog"));
    }
  }

  // ------------------------------------------------------------ submit sheet

  const PRESET_KEYS = typeof PRESET_FIELDS !== "undefined" ? PRESET_FIELDS : [];

  // decisions.py drops any override key VideoParams does not have, without a
  // word: `preest=3` submitted fine and encoded at the preset's value. This
  // warns and never blocks - a key the backend grows before fields.js does
  // must still go through. codec is a real field with one legal value, which
  // is why fields.js leaves it out of the editor.
  const KNOWN = new Set(["codec"].concat(PRESET_KEYS.map((f) => f.key)));

  function optionsOf(key, fallback) {
    const f = PRESET_KEYS.find((x) => x.key === key);
    return (f && f.options) || fallback;
  }

  // Split on the FIRST "=": probe_video_params=preset=10 and
  // additional_video_params=--enable-tf 0 are real overrides, and the old
  // split("=") rejected both as malformed. Empty tokens (a trailing "," or
  // ";") are skipped. Values stay strings - decisions.py hands them to
  // model_copy(update=...) as typed, which is all it has ever received.
  // model_copy does not validate, so a value that is really two overrides
  // ("crf=24，preset=3" typed with the IME's comma, or "target_quality=95
  // crf=24") would reach the encoder whole; the full-width separators split
  // too, and an "=" inside a value is refused except where it is the syntax.
  const EQ_IN_VALUE = new Set(["probe_video_params", "additional_video_params"]);
  function parseParams(text) {
    const out = {};
    for (const tok of String(text || "").split(/[,;，；]/)) {
      const t = tok.trim();
      if (!t) continue;
      const eq = t.indexOf("=");
      if (eq <= 0 || !t.slice(0, eq).trim()) return { bad: t };
      const key = t.slice(0, eq).trim(), value = t.slice(eq + 1).trim();
      if (value.includes("=") && !EQ_IN_VALUE.has(key)) return { bad: t };
      out[key] = value;
    }
    return { overrides: out };
  }

  function buildSubmitStatics() {
    const opt = (pairs) => pairs.map(([v, l]) => h`<option value="${v}">${l}</option>`).join("");
    $("engine").innerHTML = opt(optionsOf("engine", [["av1an", "av1an"], ["optimizer", "optimizer"]]));
    $("target_metric").innerHTML = opt(optionsOf("target_metric",
      [["vmaf", "VMAF"], ["ssimulacra2", "SSIMULACRA2"], ["xpsnr", "XPSNR"]]));
    $("paramKeys").innerHTML = PRESET_KEYS.map((f) => h`<option value="${f.key}=" label="${f.label}"></option>`).join("");
    $("keysSummary").textContent = `可用字段 ${PRESET_KEYS.length} 个`;
    $("keysList").innerHTML = PRESET_KEYS.map((f) =>
      h`<li><code>${f.key}</code><span>${f.label}${f.help ? "：" + f.help : ""}</span></li>`).join("");
  }

  function fillPresets() {
    const sel = $("preset");
    const keep = sel.value;
    const names = Object.keys(S.presets);
    sel.innerHTML = names.map((n) => h`<option value="${n}">${n}</option>`).join("") +
      h`<option value="custom">custom（自定义）</option>`;
    if (keep && (names.includes(keep) || keep === "custom")) sel.value = keep;
    else sel.value = names.includes("balanced") ? "balanced" : (names[0] || "custom");
    syncSubmit(false);
  }

  async function loadPresets() {
    if (S.presetsBusy) return;
    S.presetsBusy = true;
    try {
      const r = await api("/api/presets");
      S.presets = r && typeof r === "object" ? r : {};
      S.presetsOk = true;
      fillPresets();
      render();
    } catch (err) {
      // Best-effort, as it always was: with no list the server's
      // default_preset still applies, and the next poll tries again.
      if (!$("preset").options.length) {
        $("preset").innerHTML = h`<option value="">（服务端默认预设）</option><option value="custom">custom（自定义）</option>`;
        syncSubmit(false);
      }
    } finally {
      S.presetsBusy = false;
    }
  }

  function syncSubmit(enteringCustom) {
    const name = $("preset").value;
    const isCustom = name === "custom";
    const p = S.presets[name] || {};
    if (enteringCustom) $("engine").value = "av1an";   // a clean slate, not the last preset's engine
    const eng = isCustom ? $("engine").value : (p.engine || "av1an");
    $("customCtl").hidden = !isCustom;
    $("target_metric").disabled = !(isCustom && eng === "optimizer");

    const echo = $("presetEcho");
    const bits = [];
    if (!isCustom && name && S.presets[name]) {
      bits.push(`engine=${eng}`);
      if (p.crf != null) bits.push(`crf=${p.crf}`);
      if (p.preset != null) bits.push(`preset=${p.preset}`);
      if (p.target_quality) bits.push(`target_quality=${p.target_quality}`);
      if (eng === "optimizer" && p.target_metric) bits.push(`target_metric=${p.target_metric}`);
    }
    echo.textContent = bits.join(" ");
    echo.hidden = !bits.length;

    // The old page kept the optimizer sentence after custom went back to av1an.
    $("customHint").textContent = !isCustom
      ? `所选预设的引擎/探测指标以设置页为准（当前引擎：${eng}）；自定义参数框可覆盖其余字段。`
      : eng === "optimizer"
        ? "optimizer：Shot-based 并行探测，需填 target_quality（如 target_quality=80 或 75-85）。探测指标取上面的下拉。"
        : "custom 模式：以默认参数为基准，自定义参数框 + 引擎/指标下拉全部生效（留空等于默认 preset）。";
  }

  function syncWarn() {
    const r = parseParams($("custom").value);
    const unknown = r.overrides ? Object.keys(r.overrides).filter((k) => !KNOWN.has(k)) : [];
    const w = $("paramWarn");
    w.hidden = !unknown.length || !PRESET_KEYS.length;
    w.textContent = unknown.length === 1
      ? `${unknown[0]} 不是已知参数，服务端会忽略它。`
      : `${unknown.join("、")} 不是已知参数，服务端会忽略它们。`;
  }

  function renderDirs() {
    const d = (S.status && S.status.dirs) || {};
    const seen = new Set();
    const items = [["input", "输入目录"], ["output", "输出目录"], ["rpu", "RPU 目录"]]
      .filter(([k]) => d[k] && !seen.has(d[k]) && seen.add(d[k]));
    $("dirs").innerHTML = items.length
      ? h`常用：${items.map(([k, l]) => h`<button type="button" class="btn btn--quiet btn--sm" data-browse="${d[k]}" title="${d[k]}">${l}</button>`)}`
      : "";
  }

  function openSubmit() {
    renderDirs();
    syncSubmit(false);
    syncWarn();
    openDlg("submitDlg");
    // Only with a mouse: on a phone focusing the field throws up a keyboard
    // over half the sheet before the operator has chosen 浏览… or typing.
    if (window.matchMedia("(pointer: fine)").matches) $("path").focus();
  }

  async function submit(e) {
    e.preventDefault();
    const fs = $("submitState");
    const path = $("path").value.trim();
    const preset = $("preset").value || "";
    formState(fs, "");
    if (!path) {
      formState(fs, "请输入要转码的文件或目录路径", "err");
      $("path").focus();
      return;
    }
    const parsed = parseParams($("custom").value);
    if (parsed.bad != null) {
      formState(fs, `自定义参数格式错误: "${parsed.bad}"（应为 key=value）`, "err");
      $("custom").focus();
      return;
    }
    const overridesOut = parsed.overrides;
    // The two selects only mean something in custom mode; otherwise the
    // chosen preset (edited on the settings page) is authoritative.
    if (preset === "custom") {
      const engine = $("engine").value, metric = $("target_metric").value;
      if (engine && engine !== "av1an" && !("engine" in overridesOut)) overridesOut.engine = engine;
      if (engine === "optimizer" && metric && !("target_metric" in overridesOut)) overridesOut.target_metric = metric;
      if (engine === "optimizer" && !overridesOut.target_quality) {
        formState(fs, "optimizer 引擎需要 target_quality（如 crf=24,preset=3,target_quality=80 或 75-85）", "err");
        return;
      }
    }
    const btn = $("submitBtn");
    btn.disabled = true;
    formState(fs, "正在提交…", "dim");
    try {
      // A directory enqueues one job per file but the response carries only
      // the first id (queue.py enqueue_file), so the count is a before/after
      // of the status totals.
      const before = await api("/api/status").then((s) => sum(s && s.jobs), () => null);
      await api("/api/jobs", { json: { path, preset, overrides: overridesOut } });
      const after = await api("/api/status").then((s) => sum(s && s.jobs), () => null);
      const n = before != null && after != null ? after - before : 0;
      const msg = n > 0 ? `已提交 ${n} 个任务` : "已提交";
      formState(fs, msg, "ok");
      notice(msg);
      $("path").value = "";
      closeDlg("submitDlg");
    } catch (err) {
      formState(fs, failText("提交失败", err), "err");
    } finally {
      btn.disabled = false;
      S.ctl.now();
    }
  }

  // ------------------------------------------------------------ browser

  const B = { path: "/", parent: null, seq: 0 };

  function parentOf(p) {
    const s = String(p || "/").replace(/\/+$/, "");
    if (!s) return null;
    const i = s.lastIndexOf("/");
    return i <= 0 ? "/" : s.slice(0, i);
  }

  // The old join was `${d.path}/${e.name}`, which at the root is "//media".
  const joinPath = (dir, name) => (dir.endsWith("/") ? dir : dir + "/") + name;

  async function browse(path) {
    const seq = ++B.seq;
    B.path = path || "/";
    $("bPath").value = B.path;
    const tree = $("btree");
    tree.classList.add("is-loading");
    formState("bState", "正在读取…", "dim");
    try {
      const d = await api("/api/browse?path=" + encodeURIComponent(B.path));
      if (seq !== B.seq) return;
      B.path = d.path;
      B.parent = d.parent || null;
      $("bPath").value = d.path;
      const entries = d.entries || [];
      tree.innerHTML = entries.map((e) => e.dir
        ? h`<li><button type="button" class="brow" data-dir="${joinPath(d.path, e.name)}"><span class="brow__ico" aria-hidden="true">▸</span><span class="brow__name">${e.name}</span></button></li>`
        : h`<li><button type="button" class="brow" data-file="${joinPath(d.path, e.name)}"><span class="brow__ico" aria-hidden="true">·</span><span class="brow__name">${e.name}</span><span class="brow__sz">${fmtBytes(e.size, "")}</span></button></li>`
      ).join("");
      formState("bState", entries.length ? "" : "（空目录）", "dim");
    } catch (err) {
      if (seq !== B.seq) return;
      // The old error branch replaced the whole tree, ↑ .. included: step
      // into one unreadable directory and 关闭 was the only way out.
      B.parent = parentOf(B.path);
      tree.innerHTML = "";
      const why = err && err.status && err.message === `HTTP ${err.status}` ? `读取失败（HTTP ${err.status}）` : ((err && err.message) || "读取失败");
      formState("bState", `无法读取 ${B.path}：${why}`, "err");
    } finally {
      if (seq === B.seq) {
        tree.classList.remove("is-loading");
        $("bUp").disabled = !B.parent;
      }
    }
  }

  function openBrowse(start) {
    const d = (S.status && S.status.dirs) || {};
    $("btree").innerHTML = "";
    openDlg("browseDlg");
    browse(start || $("path").value.trim() || d.input || "/media");
  }

  function pickPath(p) {
    $("path").value = p;
    formState("submitState", "");
    closeDlg("browseDlg");
  }

  // ------------------------------------------------------------ wiring

  function init() {
    $("chips").innerHTML = CHIP_KEYS.map((k) =>
      h`<button type="button" class="chip" aria-pressed="${String(S.filter === k)}" data-status="${k}">${k ? statusLabel(k) : "全部"} <b class="chip__n">${DASH}</b></button>`
    ).join("");
    buildSubmitStatics();

    document.addEventListener("click", onClick);
    $("moreBtn").addEventListener("click", () => {
      S.limitIdx = Math.min(S.limitIdx + 1, LIMITS.length - 1);
      $("moreBtn").hidden = true;
      S.ctl.now();
    });
    $("pruneBtn").addEventListener("click", prune);
    $("cancelAllBtn").addEventListener("click", cancelQueued);

    $("submitForm").addEventListener("submit", submit);
    $("submitForm").addEventListener("input", (e) => {
      if ($("submitState").dataset.tone === "err") formState("submitState", "");
      if (e.target.id === "custom") syncWarn();
    });
    $("preset").addEventListener("change", () => syncSubmit($("preset").value === "custom"));
    $("engine").addEventListener("change", () => syncSubmit(false));
    $("browseBtn").addEventListener("click", () => openBrowse());

    $("browseNav").addEventListener("submit", (e) => { e.preventDefault(); browse($("bPath").value.trim() || "/"); });
    $("bUp").addEventListener("click", () => { const p = B.parent || parentOf(B.path); if (p) browse(p); });
    $("bUse").addEventListener("click", () => pickPath(B.path));
    $("btree").addEventListener("click", (e) => {
      const b = e.target.closest(".brow");
      if (!b) return;
      if (b.dataset.dir) browse(b.dataset.dir);
      else if (b.dataset.file) pickPath(b.dataset.file);
    });

    loadPresets();
    S.ctl = poll(refresh, { interval: 4000, onOk: onPollOk, onError: onPollError });
  }

  init();
})();
