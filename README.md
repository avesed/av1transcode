# AV1 Transcode Archive

自动化影视存档系统：基于 **ffmpeg v9**（原生编译，含内置 Dolby Vision 支持）+ **SVT-AV1 v4.2** + **av1an**，尽可能在保证质量的前提下最大化储存效率。内置 HDR 直通与 **Dolby Vision P5/P7/P8 → HDR10 + RPU 另存** 处理。调度与前端全部为 **Python**（FastAPI + 并发 worker + Watchdog + SQLite）。单容器 all-in-one 部署。

## 特性

- **自动发现**：监控输入目录，文件稳定（默认 30s 未变化）后自动入队
- **AV1 高压缩存档**：av1an 场景切分 + SVT-AV1 并行编码，CRF/preset/film-grain 全可配
- **Shot-based 优化器引擎（Netflix 风格，可选）**：ffmpeg `scdet` 分镜 → 所有镜头<b>并行</b>质量探测（VMAF/SSIMULACRA2）→ 二分出每镜头精细 CRF → 并行编码 → 抽验实际交付质量。预设里 `engine=optimizer` + `target_quality` 启用，替代 av1an 串行 `--target-quality`（`scenedetect_engine: pyscenedetect` 可切回旧的 PySceneDetect 路径）
- **GPU 打分（Intel Arc）**：VMAF 可通过 libvmaf 的 SYCL 后端在 Arc 上计算，`optimizer.vmaf_sycl_device` 设为设备号启用（默认 -1 关闭）。4K 上实测每次探测1.7s/1 核/0.25GB，对比 CPU 的 5.6s/10.7 核/7.6GB，分数不变。每个任务开头会自证设备可用且与 CPU 后端一致，不一致就整个任务退回 CPU
- **HDR 直通**：BT.2020 + PQ/HLG 色彩与母版元数据（mastering display / MaxCLL）完整保留
- **Dolby Vision 存档**：
  - P5：ICtCp → HDR10（`libplacebo`，需 Vulkan）
  - P7：取 BL（自带 HDR10）转码，RPU/EL 丢弃
  - P8：剥 RPU 后以 HDR10 编码
  - 三种均通过 `dovi_tool extract-rpu` **单独保存 `.rpu`** 到 rpu 目录
- **输出命名**：沿用源文件名，只改 Sonarr/Radarr 会改的两处——编码（h265/x265/HEVC/AVC/H.264… → `AV1`）和动态范围（成品不带 DV 和 HDR10+：`DV.HDR10`、`HDR10+` → `HDR10`，`DV.HLG` → `HLG`，`DV.SDR` 删去），所以 `…Remux.h265.DTS-HD.MA.DV.HDR10.mkv` 输出为 `…Remux.AV1.DTS-HD.MA.HDR10.mkv`，正是 Sonarr 重命名后的样子。文件名里找不到编码标记时保留旧的 `<片名>.av1.mkv`。RPU 与成品同名，扩展名 `.rpu`（`.bin` 会被 *arr/Plex 当成视频）
- **音频/字幕直通**：av1an 合并时全部 `copy`，不重编码
- **Web 界面 + CLI**：任务状态、进度、日志、提交/取消

## 技术栈

| 层 | 技术 |
|---|---|
| 调度/前端 | Python 3.12, FastAPI, Typer, pydantic, watchdog, loguru |
| 队列 | 内置多 worker 线程池 + SQLite（WAL），无需 Redis |
| 转码 | av1an + SVT-AV1 v4.2 (SvtAv1EncApp) + ffmpeg v9 |
| DV | dovi_tool (RPU提取) + ffmpeg 内建 dovi_split/dovi_rpu BSF + libplacebo |
| 容器 | Ubuntu 24.04 + Docker multi-stage 编译（10 个构建阶段；dovi_tool 那一阶段用 rust:1-bookworm）|

## 运行方式

### Docker（推荐，all-in-one）

要求较大的构建耐心（需从源码编译 ffmpeg/SVT-AV1/dovi_tool，约 10-30 分钟）。

```bash
docker compose up -d --build
# 界面: http://localhost:8080
# 日志: docker compose logs -f
```

挂载目录（默认相对本仓库 `media/`，请在 docker-compose.yml 调整为你真实的媒体库路径）：

```
media/input/      # 放入待转码文件
media/output/     # AV1 输出（按 Sonarr/Radarr 的命名改写源文件名，见下）
media/rpu/        # Dolby Vision 提取的 RPU (.rpu)
media/work/       # av1an 临时文件
```

### 本机开发

```bash
pip install -r requirements.txt
export AV1TC_DIRS_INPUT=/path/in \
       AV1TC_DIRS_OUTPUT=/path/out \
       AV1TC_DIRS_RPU=/path/rpu \
       AV1TC_DIRS_WORK=/tmp/work \
       AV1TC_DIRS_DB=/tmp/av1.db \
       AV1TC_DIRS_LOGS=/tmp/logs
python -m app.cli run            # 完整调度器 + Web
python -m app.cli process x.mkv # 投递单文件
python -m app.cli status        # 查看进度
python -m app.cli check         # 检查工具齐全度
python -m app.cli presets
```

### 测试

```bash
python -m pytest tests/
```

## 转码预设

预设位于 `config.yaml` → `transcode.presets`，可选 `quality / balanced / compact`（或自定义）：

| 预设 | CRF | SVT preset | 定位 |
|---|---|---|---|
| quality | 22 | 2 | 高保真，慢 |
| balanced | 28 | 4 | 存档平衡（默认） |
| compact | 34 | 6 | 文件更小（略损） |

其他常用配置：`transcode.video.film_grain`（胶片类内容 8-10）、`workers.concurrency`、`workers.av1an_workers`（0=auto）。

Web 界面的写操作（提交/取消任务、改预设与设置）和目录浏览受 `web.api_key` 保护；设了之后需在「设置 → API 密钥」里填一次，密钥只存在浏览器本地。空值（默认）= 全部开放。

## 已核实的技术要点

- ffmpeg v9（master）原生支持 DV：`dovi_split` BSF（P7 分层拆分）、`dovi_rpu` BSF（剥/压缩 RPU，支持 HEVC **与 AV1**）、分层流 demux/mux、SMPTE 2094-50 动态元数据、`-mastering_display`/`-content_light` 输入修正
- av1an 场景切分使用 `av-scenechange`；DV RPU 场景边界与视觉场景不同，故本项目的 DV 存档策略为：「转 HDR10 存档 + RPU 单独保存」，不做 DV 场景对齐

## 质量保障

- 输出时长/流校验（av1 失败即重试，`max_retries`）
- mkv 上自动写回母版显示/MaxCLL 元数据
- 作业状态 SQLite 持久化，重启后自动继续 pending 任务