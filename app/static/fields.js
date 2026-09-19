// The single table of every knob this UI can edit: the optimizer form, the GPU
// form, and the preset editor all render from here, and tests/test_core.py
// greps this file for F("<key>" to prove no optimizer field has quietly fallen
// out of the frontend.
//
// It lived inside settings.html until this file existed, which is how max_crf
// and probe_dataset came to be drawn on the GPU card that cannot save them:
// PUT /api/settings/gpu accepts _GPU_KEYS only, the optimizer form posted only
// its own groups, and nothing on either side said the two keys fell between.
// They are in the optimizer groups now (api.py did not have to change).

// F(key, label, help, opts) -> the field.
//
//   help     one line, layer 1: the operating conclusion - what to do, or why
//            the default is the default. Never a definition.
//   note     the measured writeup, verbatim, behind a <details> in layer 2.
//            Splitting a long help is the one edit that can lose a measurement,
//            so note holds the ORIGINAL TEXT WHOLE rather than a rewrite of it.
//   summary  the <details> label, which names what is inside it - the sample,
//            the incident - so it can be skipped without being opened.
// Every field in OPT_GROUPS and GPU_FIELDS gets a 说明 › link to its entry on
// docs.html (settings.js docAnchor); adding a field here means adding that
// entry too, or tests/test_core.py fails.
const F = (key, label, help, o = {}) => ({ key, label, help, type: "number", ...o });

const OPT_GROUPS = [
  { id: "grp-detect", name: "分镜", note: "先把源切成镜头，后面所有探测和编码都按镜头进行", fields: [
    F("scenedetect_engine", "分镜引擎", "scdet 是 ffmpeg 内建滤镜，一趟读完不落盘；pyscenedetect 是旧路径。", { type: "select", options: [["scdet", "scdet"], ["pyscenedetect", "pyscenedetect"]] }),
    F("scdet_threshold", "切点门限", "scdet 的场景变化分数，0-100，越低越敏感；ffmpeg 自己的默认是 10。", { step: 0.1, min: 0, max: 100 }),
    F("scenedetect_threshold", "灵敏度", "pyscenedetect 的内容差分阈值，越高分镜越少。", { step: 0.5, min: 0 }),
    F("scenedetect_scale", "检测前缩放", "只为提速；空 = 原生分辨率。实测 540p 和原生选出的切点一样。", { type: "text" }),
    F("min_scene_len", "最小镜头", "短于此帧数不切。", { min: 0, unit: "帧" }),
    F("min_shot_frames", "合并短镜头", "短于此帧数并入相邻镜头；0 = 不合并。", { min: 0, unit: "帧" }),
    F("max_shots", "镜头数上限", "超出时合并最短的镜头。", { min: 1 }),
  ]},
  { id: "grp-probe", name: "探测", note: "每个镜头在源分辨率下用快速 preset 编几个 CRF，量出它自己的质量曲线", fields: [
    F("probe_crfs", "CRF 网格", "逗号分隔的整数，从这里出发二分。", { type: "csv", wide: true }),
    F("probe_encoder", "探测编码器", "默认 svt。qsv+svt 探测阶段快 27%，交付的 CRF 仍由 SVT 量出来。", {
      type: "select", options: [["svt", "svt：原有路径"], ["qsv", "qsv：GPU 探测 + 映射"], ["qsv+svt", "qsv+svt：GPU 预测 + SVT 核验"]],
      summary: "展开实测数据（三段 4K DV，912 个镜头）",
      note: "svt = 原有路径。qsv = 在显卡上探测（VA-API 解码进硬件 AV1 编码器），再用逐作业标定把质量索引映射回 SVT 的 CRF。实测一次 300 帧 4K 探测 16.4 CPU 秒对 88.3 秒，省下的预算可以放开探测窗口——而窗口截断正是今天预测误差的最大来源。映射不可靠时会整个作业或逐镜头退回 SVT。qsv+svt = 同样的显卡工作换一种用法：映射出来的 CRF 只决定从哪里开始探，交付的 CRF 仍然来自 SVT 探测，所以映射错了只多花一次探测而不会交付没量过的值；那条线还会拿每个核验过的镜头继续修正。实测（生产作业形态、零拷贝打分开启，三段 4K 杜比视界片段共 912 个镜头）：探测阶段比 svt 快 23%、23%、33%，合计 27%；每镜头 SVT 探测 3.8–3.9 次降到 2.3–2.6 次，另加 2 次显卡探测；选出的 CRF 与 svt 相差不超过 1 的镜头占 96–100%，没有相差超过 2 的；整卡显存峰值高 1–2 GB。零拷贝上线前同样的设计只打平。" }),
    F("gpu_probe_qs", "QSV 质量网格", "QSV 侧扫描的质量索引，作用同 CRF 网格。av1_qsv 在 4K 上约 50 以上饱和。", { type: "csv", wide: true }),
    F("gpu_probe_anchors", "标定锚点镜头数", "同时用两条路探测、用来拟合映射的镜头数。它们用的是自己的 SVT 结果，不是白跑的。", { min: 4, max: 200 }),
    F("gpu_probe_max_residual", "放弃映射的残差上限", "标定的留一残差超过这么多 CRF 就整个作业退回 SVT。按 0.24 VMAF/CRF，默认约合 0.5 VMAF。", { step: 0.1, min: 0.1, max: 20 }),
    F("gpu_probe_max_q_margin", "逐镜头弃权余量", "质量索引超出锚点范围这么多的镜头改用 SVT 探测，用来兜住外推失效的容易镜头。", { step: 0.1, min: 0, max: 20 }),
    F("gpu_probe_preset", "QSV 编码 preset", "0-7，越小越慢越好。映射就是在 4 上标定的。", { min: 0, max: 7 }),
    F("probe_bracket_width", "二分停止宽度", "目标所在区间窄于此就停；0 = 跑完整网格。", { min: 0 }),
    F("probe_preset", "探测速度", "SVT-AV1 preset，比正式编码快；4K 下 SVT 自动降到 9。", { min: 0, max: 13 }),
    F("probe_max_frames", "探测窗口", "每镜头最多探测这么多连续帧，超长镜头取中间一段。", { min: 24, unit: "帧" }),
    F("probing_rate", "抽帧", "每隔 n 帧探测一帧。大于 1 会低估质量，建议 1。", { min: 1, max: 10 }),
    F("probe_scale", "探测分辨率", "留空 = 源分辨率。填了等于在另一个分辨率上量曲线，CRF 会失真。", { type: "text" }),
    F("probe_crf_offset", "CRF 补偿", "探测 preset 快、会低估成品质量，正值把这部分换回体积。校验阶段会报出实际偏差。", { step: 0.5 }),
    F("min_crf", "CRF 下限", "目标达不到时兜底，防文件变大；0 = 网格最小值。", { min: 0, max: 63 }),
    F("max_crf", "CRF 上限", "0 = probe_crfs 的顶端。它管能交付什么，probe_crfs 管去哪里看。", {
      min: 0, max: 63,
      summary: "展开：为什么探测范围和交付范围要分开",
      note: "交付 CRF 的硬上限，0 = probe_crfs 的顶端；min_crf 的镜像。它让\u201c探测范围\u201d和\u201c交付范围\u201d成为两件事：probe_crfs 管去哪看，min_crf/max_crf 管能发什么。分开之后放宽探测范围几乎不花钱，而探到极端值不等于会用极端值。" }),
    F("max_crf_delta", "相邻镜头 CRF 差上限", "平滑相邻镜头的 CRF；0 = 不平滑。", { step: 0.5, min: 0 }),
    F("fractional_crf", "小数 CRF", "把插值出来的小数 CRF 传给 SVT-AV1。", { type: "bool" }),
  ]},
  { id: "grp-score", name: "打分", note: "用什么模型、在什么分辨率上比较探测编码和源", fields: [
    F("vmaf_model", "VMAF 模型", "1080p 模型的路径。", { type: "text", wide: true }),
    F("vmaf_model_4k", "4K 模型", "源宽度达到下面的阈值时改用它，并按原生分辨率打分。", { type: "text", wide: true }),
    F("vmaf_4k_min_width", "用 4K 模型的宽度", "低于此宽度用 1080p 模型。", { min: 0, max: 7680, unit: "px" }),
    F("vmaf_width", "送入 VMAF 的宽度", "只缩不放；0 = 原分辨率。只在用 1080p 模型时生效。", { min: 0, max: 7680, unit: "px" }),
    F("vmaf_threads", "VMAF 线程", "0 = 自动。写死 40 实测比自动慢 30%、内存翻倍。", {
      min: 0,
      summary: "展开：探测与校验的两条自动规则",
      note: "0 = 自动：探测按每个 worker 分到的核，校验按核数一半。显式值会两边都用，实测 4K 上 40 比自动慢 30%、内存翻倍。" }),
    F("ssimulacra2_model", "SSIMULACRA2 模型", "SSIMULACRA2 现在经 vszip 插件打分，不读这个键；只为兼容旧配置保留。", { type: "text", wide: true }),
    F("ssimulacra2_frame_step", "SSIMULACRA2 抽帧", "打分侧每 n 帧取 1；逐帧度量抽帧成立，和探测抽帧不同。", { min: 1, max: 60 }),
  ]},
  { id: "grp-res", name: "并发与内存", note: "0 都是按容器的 cgroup 预算自动算", fields: [
    F("probe_workers", "并行探测数", "0 = 按内存自动。", { min: 0, max: 256 }),
    F("encode_workers", "并行正式编码数", "0 = 按内存自动；每个 4K 实例持有几 GB 帧池。", { min: 0, max: 128 }),
    F("encode_threads", "每个编码实例的核数", "0（推荐）= 不钉核；只有容器不许碰没分给它的核时才设。", { min: 0, max: 512 }),
    F("probe_cpu_charge", "探测的 CPU 计费", "探测按 lp 的几成计入 CPU 预算，1.0 = 满额；调低放更多并发，代价是余量。", { step: 0.05, min: 0.1, max: 1 }),
  ]},
  { id: "grp-subs", name: "字幕", note: "最后一步把源的音轨/字幕复用进输出时，对字幕做的三件事", fields: [
    F("drop_empty_subtitles", "丢掉空字幕轨", "默认开。只丢一条字幕都没有的轨；读不准一律保留。", {
      type: "bool",
      summary: "展开：Plex 烧空轨那次事故与判定方法",
      note: "源里一条字幕都没有的轨不再复制进输出。起因：Plex 自动选中了一条空的 PGS 轨并烧进画面——整集重编码、0.2-0.9 倍速、转码进程 660% CPU，而那条轨里什么都没有；成品库里 42/45 个输出的字幕轨全是空的。判定只读文件头：mkv 看 NUMBER_OF_FRAMES 统计 tag，没有 tag 就用 mkvmerge -J 的 num_index_entries（v74 不报这个属性，遇到就全保留并打日志），mp4 看轨道自己的时长 duration_ts（nb_frames 用不了：ffprobe 只在非零时才打印它，mov 还会给空轨补一个 padding sample，实测空轨读出来是 1）；读不到、读不懂、两个工具对不上一律保留——丢错一条就是毁掉那份字幕的唯一副本。" }),
    F("ass_srt_companion", "ASS 旁挂 srt", "默认开。ASS 轨原样保留，旁边多挂一条约 35KB 的纯文本 srt。", {
      type: "bool",
      summary: "展开：为什么特效字幕转出来一定是坏的",
      note: "每条保留下来的 ASS/SSA 字幕旁边再加一条纯文本 srt，ASS 轨本身原样保留、绝不转换。它搭在本来就要跑的那次 demux 上，每条约 35KB，给不渲染 ASS、只会烧字幕的播放器一个软字幕选择。ffmpeg 转出的 <font> 标签和字面的 {\\anN} 会被清掉；\\pos/\\move 是编码器自己丢的，所以特效/排版字幕转出来一定是坏的——这正是 ASS 必须留着的原因。" }),
    F("pgs_ocr_srt", "图形字幕 OCR 成 srt", "默认开。只做英文，图形轨原样保留，失败只降级不报错。", {
      type: "bool",
      summary: "展开实测数据（5005 条轨，字符错误率 0.1276%）",
      note: "把源里每条英文 PGS 图形字幕 OCR 成一条纯文本 srt 挂在旁边，图形轨本身原样保留、绝不删除。这是「丢掉空字幕轨」那个毛病的另一半：Plex 对图形字幕没有软字幕目标格式，选中了就只能烧进画面、整集重编码(0.2-0.9 倍速、转码进程约 660% CPU)；丢空轨只是让它不再选中空的，这一条是给真有内容的轨准备一份它能直接发出去的文本。生成的 srt 会接过 default 标记、原图形轨改写成非 default，否则 Plex 照样自动选图形轨，等于白做。只做英文，而且这是能力上限不是偏好：镜像里只装了 eng 一个 tesseract 模型，其它文字喂进去不会报错、只会吐出像模像样的乱码然后被当成字幕写进成品。成品库实测 5005 条图形轨里 1034 条是英文(语言 tag 全部写作 eng)，其中 929 条是 PGS、105 条是 DVD VobSub；VobSub 不是这里能读的格式，同样原样不动。准确度实测(用文件自带的 SDH 文本轨当基准，S.H.I.E.L.D. S05E05 共 711 条)：忽略大小写的字符错误率 0.1276%，689/711 条完全一致；真实库里跑的 71 个文件中 70 个通过结构门限。提取是挂在本来就要跑的那次 demux 上多一个输出，不额外读一遍源；没装 tesseract、解析不了、某条卡住、门限没过，一律降级成「这条轨没有 srt」并打告警，绝不会让编码失败。" }),
  ]},
  { id: "grp-verify", name: "校验与调试", note: "编码完成后抽几个镜头重新打分，对账探测预测和实际交付", fields: [
    F("verify_shots", "抽验镜头数", "0 = 关闭。这是全流程唯一的对账机制。", { min: 0 }),
    F("keep_probes", "保留探测文件", "调试用。", { type: "bool" }),
    F("probe_dataset", "记录探测数据集", "默认开。一集约 160 行，不记录就永久损失。", {
      type: "bool",
      summary: "展开：这些数据用来改进哪条映射线",
      note: "每探测完一个镜头往 <logs>/probe_dataset.jsonl 追加一行：镜头几何、探到的 (crf, 分数) 点、(q, 分数) 点、两个交点、最终 CRF。一集约 160 行。QSV→SVT 那条映射线的残差是关系本身而不是估计误差（104 对样本时已在渐近值 0.5% 以内），要改善只能靠更丰富的模型，而那需要跨集累积数据——不记录就永久损失。" }),
  ]},
];

// Exactly what PUT /api/settings/gpu writes (_GPU_KEYS in app/api.py), plus
// vulkan_device, which that endpoint takes separately because it lives in
// transcode.dovi rather than in the optimizer block. Anything else rendered
// here would be a control that reports success and saves nothing.
const GPU_FIELDS = [
  F("vmaf_sycl_device", "VMAF 打分设备", "用 libvmaf 的 SYCL 后端在 GPU 上打分。关闭 = CPU。作业开始时会先自检一次，设备不可用则整个作业退回 CPU。", { type: "sycl" }),
  F("vmaf_sycl_min_width", "GPU 打分的最小宽度", "源窄于此仍用 CPU。1080p 实测：零拷贝打分让探测阶段快一半、CPU 省七成；1920 以下没测过。", { min: 0, max: 7680, unit: "px" }),
  F("scenedetect_hwaccel", "分镜解码", "auto = 有 QSV 就用它解码分镜那一趟；off = 一律软解。", { type: "select", options: [["auto", "auto：有 QSV 就用"], ["off", "off：一律软解"]] }),
  F("gpu_vram_budget_mb", "显存预算 (MB)", "0 = 自动（6000）：给 12GB 的卡留一半，因为看不见 Plex 占的那份。", {
    min: 0, max: 65536,
    summary: "展开：DRM fdinfo 看不到 Plex 那份",
    note: "本作业最多占用显卡多少显存，0 = 自动（6000）。worker 数管并发个数，这个管字节数——显卡耗尽的是后者。占用量用 DRM fdinfo 实测，但只看得见本容器的进程，Plex 的那份看不到，所以默认给 12GB 的卡留了一半。" }),
  F("vmaf_sycl_workers", "GPU 打分并发上限", "0 = 自动（6）。再加也没用，实测吞吐 4 个就到顶。", {
    min: 0, max: 32,
    summary: "展开：10 个 SYCL 上下文曾弄坏 B580",
    note: "同时最多几个探测在 GPU 上打分，0 = 自动（6）。显卡只有一块显存：各 10 个 SYCL 上下文 + VA-API 解码会话曾把 B580 撑爆并弄坏设备。加宽也没用，实测吞吐 4 个就到顶。" }),
  F("vmaf_zero_copy", "零拷贝打分", "默认 off。开了每次打分 23 → 1.4 CPU 秒，代价是约 1.55GB 显存。", {
    type: "select", options: [["auto", "auto：预检通过就用"], ["off", "off：常规读取"]],
    summary: "展开实测数据（B580 4K，247 次打分）",
    note: "探测打分时源窗口和探测文件两路都在 GPU（VA-API）上解码，帧直接交给 libvmaf 的 SYCL 后端，不回传内存。B580 4K 实测：常规每次打分 5.4s / 23 CPU 秒，零拷贝 1.18s / 1.36 CPU 秒，247 次打分逐帧分数完全一致；代价是每次打分约 1.55GB 显存，并发上限由显存决定。需要 SYCL 打分、4K 模型、源位深与探测编码一致；每个作业先两种方式各打一个窗口，不一致整个作业走常规读取，单个窗口失败退回常规打分。镜像需带 libva 的 libvmaf 和配套 ffmpeg 补丁。" }),
  F("reference_hwaccel", "打分参考解码", "默认 off：40 核上它反而更慢，只有 CPU 是瓶颈的机器才值得开。", {
    type: "select", options: [["auto", "auto：有 QSV 就用"], ["off", "off：一律软解"]],
    summary: "展开实测数据与 QSV 定位陷阱",
    note: "探测打分和验证时读源片的那一路是否用 GPU（VA-API）解码。默认 off：40 核实测它反而更慢（探测 401s→448s），只有 CPU 是瓶颈的机器才值得开（20 核上 826s→592s）。auto = 作业开始先各解 8 帧（带定位）做软解/硬解逐帧对比，不一致整个作业软解。用 VA-API 不用 QSV：QSV 解码器定位后保留的帧在部分源上和软解不同（像素一致、帧全错），VA-API 只给 ffmpeg 自带解码器加速，选帧和软解一样。只放这一路上 GPU：4K 帧回传 CPU 约 130 帧/秒封顶，探测池本来就吃到这个量。实测每次打分省 44% CPU、墙钟 -12%，分数不变。" }),
  F("vulkan_device", "Dolby Vision P5 转换设备", "libplacebo 应用 RPU 用的 Vulkan 设备，ffmpeg 的 -init_hw_device 选择器：序号或名字片段。空 = 让 ffmpeg 挑，有 GPU 就用 GPU；写 llvmpipe 强制软件渲染。", { type: "vulkan", wide: true }),
];

// The encode preset editor: one entry per VideoParams field the form writes,
// in the order the editor shows them. The preset name is not here - it is the
// record's identity, not one of its parameters.
const PRESET_FIELDS = [
  F("engine", "引擎", "", { type: "select", options: [["av1an", "av1an（串行探测）"], ["optimizer", "optimizer（分镜并行探测）"]] }),
  F("target_metric", "质量指标", "只对 optimizer 引擎生效。", { type: "select", options: [["vmaf", "VMAF"], ["ssimulacra2", "SSIMULACRA2（慢约 9 倍，目标值需重定标）"], ["xpsnr", "XPSNR（dB，约 42 ≈ 视觉无损）"]] }),
  F("target_quality", "目标质量", "分场景目标，如 75-85；留空 = 固定 CRF。", { type: "text", placeholder: "94-95" }),
  F("crf", "CRF", "越低越清晰、越慢；有目标质量时只作兜底。", { min: 0, max: 63, dflt: 28 }),
  F("preset", "SVT-AV1 速度", "0 最慢最好，13 最快。", { min: 0, max: 13, dflt: 4 }),
  F("film_grain", "胶片颗粒合成", "0 关；颗粒电影 8-10。", { min: 0, max: 50, dflt: 0 }),
  F("film_grain_denoise", "先去噪再合成颗粒", "", { type: "bool" }),
  F("luminance_qp_bias", "暗帧 QP 偏置", "0 关。实测 50：体积 +9~10%，够不到目标的镜头 6/23 → 2/23。", {
    min: 0, max: 100, dflt: 0,
    summary: "展开实测数据（偏置 50，23 个镜头）",
    note: "0 关。按帧平均亮度降 QP，给暗帧更多比特。实测 50：体积 +9~10%，交付中位 +0.06~0.37 分，够不到目标的镜头 6/23 → 2/23。亮场也会涨，不只作用于暗帧。" }),
  F("passes", "编码遍数", "", { min: 1, max: 2, dflt: 1 }),
  F("keyint", "关键帧间隔", "", { min: 0, dflt: 240, unit: "帧" }),
  F("extra_split_sec", "长场景再切分", "", { min: 0, dflt: 60, unit: "秒" }),
  F("min_scene_len", "最小场景长度", "", { min: 0, dflt: 24, unit: "帧" }),
  F("tune", "tune", "0 = VQ，1 = PSNR，2 = SSIM。", { min: 0, max: 2, dflt: 0 }),
  F("pixel_format", "像素格式", "", { type: "select", options: [["yuv420p10le", "yuv420p10le"], ["yuv420p8le", "yuv420p8le"], ["yuv420p12le", "yuv420p12le"]] }),
  F("probes", "探测次数上限", "av1an 引擎；0 = 默认 4。", { min: 0, max: 10, dflt: 0 }),
  F("probing_rate", "探测抽帧", "每隔 n 帧探测一帧；0 = 用优化器设置。", { min: 0, max: 4, dflt: 0 }),
  F("probe_res", "探测分辨率", "av1an 引擎。", { type: "text", placeholder: "960x540" }),
  F("probing_vmaf_features", "VMAF 特征", "只对 av1an 引擎生效，optimizer 忽略。", { type: "text", placeholder: "default motionless" }),
  F("vmaf_threads", "VMAF 线程", "0 = 自动。写了会盖过优化器设置里的值，包括校验阶段。", { min: 0, dflt: 0 }),
  F("probe_video_params", "探测编码参数", "copy = 与正式编码相同。", { type: "text", placeholder: "preset=10", wide: true }),
  F("additional_video_params", "SVT-AV1 额外参数", "", { type: "text", placeholder: "--sharpness 1 --enable-qm 1", wide: true }),
];

// Which preset keys go to the server as integers. film_grain_denoise is the
// one boolean; everything else is sent as the trimmed string it was typed as.
const PRESET_INT = new Set(["crf", "preset", "film_grain", "luminance_qp_bias", "passes", "keyint", "extra_split_sec", "min_scene_len", "tune", "probes", "probing_rate", "vmaf_threads"]);

// The settings page's sections in page order: the desktop rail and the phone
// index grid are both generated from this, so neither can drift from the page.
// The optimizer's sub-entries are read out of OPT_GROUPS rather than repeated
// here - a hand-written rail beside a generated form is what let grp-subs be
// unreachable on phones for months. count marks the two sections that show a
// "how many differ from config.yaml" badge; the number itself is counted off
// the rendered fields, never off a second hard-coded key list.
const SECTIONS = [
  { id: "sec-key", name: "API 密钥" },
  { id: "sec-workers", name: "并行处理" },
  { id: "sec-gpu", name: "GPU", count: true },
  { id: "sec-opt", name: "优化器引擎", count: true,
    subs: OPT_GROUPS.map(g => ({ id: g.id, name: g.name })) },
  { id: "sec-safety", name: "危险操作" },
  { id: "sec-presets", name: "编码预设" },
];
