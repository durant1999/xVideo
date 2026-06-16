# Video Understanding

面向中文短视频的本地视频理解流水线。当前目标是处理 10-15 分钟左右的抖音/短视频内容，同时拿到三类信息：

- 画面内容：人物、场景、动作、商品、界面元素等。
- 烧录字幕/OCR：视频画面里的字幕、价格、品牌、账号、页面文字。
- 音轨语音：口播、旁白、对话的带时间戳转写。

默认部署策略是 `Qwen3-VL-32B-Instruct-AWQ` 负责视觉/OCR，`faster-whisper` 负责 ASR，然后按时间戳融合成统一上下文，用于总结、打标、事实提取和 QA。

## Repository Layout

- `configs/pipeline.yaml`：默认下载器、抽帧、VL 服务、ASR、融合、总结配置。
- `video_understanding/`：CLI 和流水线实现。
- `video_understanding/downloaders/`：URL 下载适配器，当前包含 `yt-dlp`、`twitter-video-downloader`、`ideaflow`。
- `scripts/manage_vl_server.sh`：后台管理 Qwen3-VL vLLM 服务，支持 `start/stop/restart/status/tail`。
- `scripts/launch_vllm_qwen3_vl_32b_awq.sh`：32B AWQ vLLM 前台启动器。
- `scripts/launch_vllm_qwen3_vl_8b_bf16.sh`：8B bf16 回退或吞吐副本启动器。
- `scripts/run_single_video.sh`：单条视频闭环脚本。
- `docs/deployment.md`：3xA100 PCIe 部署细节和验证关口。

## Execution Logic

`python -m video_understanding run <video-or-url>` 的完整流程如下：

1. 输入解析
   - 如果输入是本地视频路径，直接使用该文件。
   - 如果输入是 URL 或分享文案，先提取第一个 URL，再进入下载层。

2. 视频下载
   - 下载器按 `configs/pipeline.yaml` 的 `download.order` 顺序尝试。
   - 默认顺序是 `yt-dlp` -> `twitter-video-downloader` -> `ideaflow`。
   - 下载成功后，原视频保存在当前 `workdir/source/` 下，并写入 `download_metadata.json`。

3. 视频探测
   - 用 `ffprobe` 读取视频总时长。
   - 默认拒绝分析超过 `video.max_duration_seconds=1200` 秒的视频，避免长任务占满 GPU 队列。
   - 按 `video.segment_seconds` 切分时间窗，默认每 45 秒一段。

4. 分段抽帧
   - 用 `ffmpeg` 对每个时间窗抽帧。
   - 默认 `fps=1.0`，也就是每秒 1 张图。
   - 默认每段 45 秒，所以每段大约 45 张 JPG；最后一段按剩余时长决定。
   - 默认最大边 `960`，减少视觉 token 和显存压力。

5. 视觉/OCR 理解
   - 每个时间窗的帧按 `image_url` data URL 形式发给 vLLM OpenAI-compatible `/v1/chat/completions`。
   - 模型只看画面，不听音频。
   - 输出每段的画面描述、OCR 字幕、关键实体、不确定项。
   - 结果写入 `visual.jsonl`。

6. 音频抽取和 ASR
   - 用 `ffmpeg` 抽取 `audio.wav`。
   - 用 `faster-whisper` 转写语音，默认模型是 `large-v3`，默认使用 GPU 1。
   - 第一次运行 `large-v3` 时会下载 ASR 模型，后续走本地缓存。
   - 结果写入 `asr.jsonl`。

7. 时间戳融合
   - 按时间窗对齐 `visual.jsonl` 和 `asr.jsonl`。
   - 输出结构化 JSONL 和可读 Markdown。
   - 结果写入 `fused.jsonl` 和 `context.md`。

8. 总结或 QA
   - 默认把 `context.md` 再发给同一个 OpenAI-compatible 服务做结构化总结。
   - 如果提供 `--question`，则输出针对问题的回答。
   - 结果写入 `summary.md` 或指定的输出文件。

注意：流水线不会把完整 MP4 直接发给大模型。VL 模型收到的是按时间戳抽出的图片帧；ASR 模型收到的是抽取后的音频。

## Environment

推荐 Python 3.11。当前已验证的服务端组合：

- `vllm==0.11.0`
- `transformers>=4.57.1,<5`
- `qwen-vl-utils==0.0.14`
- `torch==2.8.x` CUDA 12.8 wheel

`transformers` 不要升到 5.x；vLLM 0.11 的 tokenizer 缓存逻辑依赖 4.x 里的属性。

示例安装：

```bash
conda create -n video_understand python=3.11 -y
conda activate video_understand

pip install -U pip
pip install -e ".[client,asr,server,mcp]"
pip install "vllm==0.11.0" "transformers>=4.57.1,<5" "qwen-vl-utils==0.0.14"
```

系统依赖：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

如果使用 conda 环境运行后台服务，有两种方式：

```bash
conda activate video_understand
scripts/manage_vl_server.sh start
```

或者不激活环境，显式传环境名：

```bash
CONDA_ENV=video_understand scripts/manage_vl_server.sh start
```

## Models

当前默认使用社区 AWQ 量化权重：

```bash
hf download QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
  --local-dir models/Qwen3-VL-32B-Instruct-AWQ \
  --max-workers 3
```

模型目录 `models/` 已被 `.gitignore` 忽略，不应提交到 GitHub。

ASR 默认使用 `faster-whisper` 的 `large-v3`。首次运行 ASR 会自动下载模型。也可以提前触发下载：

```bash
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cuda', device_index=1, compute_type='float16')"
```

## Start VL Server

后台启动 GPU 0 的 32B AWQ 服务：

```bash
scripts/manage_vl_server.sh start
```

查看状态：

```bash
scripts/manage_vl_server.sh status
curl http://127.0.0.1:8000/v1/models
```

查看日志：

```bash
scripts/manage_vl_server.sh tail
```

停止或重启：

```bash
scripts/manage_vl_server.sh stop
scripts/manage_vl_server.sh restart
```

常用覆盖项：

```bash
CONDA_ENV=video_understand \
CUDA_VISIBLE_DEVICES=0 \
PORT=8000 \
MAX_MODEL_LEN=131072 \
LIMIT_MM_IMAGES=80 \
scripts/manage_vl_server.sh start
```

如果要前台调试完整 vLLM 日志：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=models/Qwen3-VL-32B-Instruct-AWQ \
SERVED_MODEL_NAME=Qwen3-VL-32B-Instruct-AWQ \
PORT=8000 \
scripts/launch_vllm_qwen3_vl_32b_awq.sh
```

空闲时 `nvidia-smi` 的 `GPU-Util` 显示 0% 是正常的。判断服务是否加载成功主要看 GPU 显存占用、`VLLM::EngineCore` 进程和 `/v1/models` 健康检查。

## Run Pipeline

处理本地视频：

```bash
python -m video_understanding run /path/to/video.mp4 --workdir runs/demo
```

处理 URL 或分享文案：

```bash
python -m video_understanding run "https://v.douyin.com/xxxx/" --workdir runs/demo
```

默认超过 20 分钟的视频会在抽帧、ASR、VL 之前拒绝分析。临时调整：

```bash
python -m video_understanding run /path/to/video.mp4 \
  --workdir runs/demo \
  --max-duration-seconds 1800
```

`--max-duration-seconds 0` 可关闭本次 CLI 运行的限制；线上 BFF/MCP 默认不暴露放宽入口，仍按配置保护。

只下载，不分析：

```bash
python -m video_understanding fetch "https://v.douyin.com/xxxx/" --output-dir runs/downloads
```

只跑视觉/OCR：

```bash
python -m video_understanding vl /path/to/video.mp4 --workdir runs/demo
```

只跑 ASR：

```bash
python -m video_understanding asr /path/to/video.mp4 --workdir runs/demo
```

只融合已有结果：

```bash
python -m video_understanding fuse \
  --visual runs/demo/visual.jsonl \
  --asr runs/demo/asr.jsonl \
  --output-jsonl runs/demo/fused.jsonl \
  --output-markdown runs/demo/context.md
```

对融合上下文做 QA：

```bash
python -m video_understanding summarize \
  --context runs/demo/context.md \
  --output runs/demo/qa.md \
  --question "视频里提到的商品卖点和价格分别是什么？"
```

完整运行后默认输出：

- `source/`：下载或复制过来的原视频。
- `frames/`：按时间窗抽出的 JPG 帧。
- `audio.wav`：从视频抽取的 16kHz 单声道音频。
- `visual.jsonl`：每个视频窗口的画面/OCR 输出。
- `asr.jsonl`：语音转写片段。
- `fused.jsonl`：按时间窗融合后的结构化上下文。
- `context.md`：给总结、QA、RAG 使用的文本上下文。
- `summary.md`：默认结构化总结。

## Tuning

主要参数在 `configs/pipeline.yaml`：

- `video.fps`：默认 1。漏短动作时升到 2；成本会近似翻倍。
- `video.segment_seconds`：默认 45。单段图片数约为 `fps * segment_seconds`。
- `video.max_side`：默认 960。OCR 不清楚时可升高；显存/延迟压力大时降低。
- `video.max_duration_seconds`：默认 1200，即 20 分钟。超过上限的视频会在抽帧/ASR/VL 前拒绝分析；设置为 0 可关闭。
- `vl.max_tokens`：每段视觉/OCR 输出长度。
- `asr.device_index`：默认 1，即第二张 GPU。
- `summary.max_tokens`：最终总结或 QA 的输出长度。

`LIMIT_MM_IMAGES` 要覆盖单段图片数。默认 `45s * 1fps = 45`，服务脚本默认 `LIMIT_MM_IMAGES=80`。

## MCP Server For Codex

本仓库可以作为 MCP server 暴露给 Mac 本地 Codex。推荐方式是 GPU 服务器本地启动 MCP HTTP server，Mac 通过 SSH tunnel 访问，不把服务裸露到公网。

GPU 服务器启动：

```bash
conda activate video_understand
scripts/launch_mcp_server.sh
```

默认监听：

```text
http://127.0.0.1:9000/mcp
```

如果不激活 conda 环境，也可以显式指定：

```bash
CONDA_ENV=video_understand scripts/launch_mcp_server.sh
```

Mac 上建立 SSH tunnel：

```bash
ssh -L 9000:127.0.0.1:9000 gpu-server
```

Mac 的 `~/.codex/config.toml` 添加：

```toml
[mcp_servers.video_understanding]
url = "http://127.0.0.1:9000/mcp"
tool_timeout_sec = 120
```

Codex 里可用的 MCP 工具：

- `get_server_info`：查看 MCP server 配置和工作流。
- `submit_video_job`：提交 URL、分享文案或服务器本地视频路径，立即返回 `job_id`。
- `get_job_status`：查询任务状态。
- `list_jobs`：列出最近任务。
- `get_job_artifact`：读取 `summary/context/visual/asr/fused/log/download_metadata` 等允许的 job artifact。
- `ask_video`：基于已有 `context.md` 对视频做 QA。
- `cancel_job`：取消排队或运行中的任务。

MCP 采用异步任务模型。不要让工具调用同步等待完整 10-15 分钟视频处理；正确流程是：

```text
submit_video_job -> get_job_status -> get_job_artifact(summary/context) -> ask_video
```

安全边界：

- MCP server 默认只绑定 `127.0.0.1`。
- job 输出固定在 `runs/mcp_jobs/<job_id>/`。
- artifact 读取使用 allow list，不能读任意服务器路径。
- 如需跨机器访问，优先用 SSH tunnel、VPN 或带鉴权的反向代理。
- 公开 GitHub 仓库不会让运行中的 MCP 服务自动暴露。真正决定可访问性的是运行时监听地址和网络转发配置。
- 不要把 `HOST` 改成 `0.0.0.0` 或 VPN 网卡 IP，除非前面有鉴权和防火墙；否则 VPN 内其他机器可能访问这个 MCP 服务。
- 推荐保持服务器端 `127.0.0.1:9000`，Mac 通过 `LocalForward 127.0.0.1:9000 127.0.0.1:9000` 接入。

## App BFF Server

`server/` 是 Android/Web 前端使用的 FastAPI BFF。它和 MCP server 复用同一个 `MCPJobManager`，提供 REST/SSE 接口给手机 App 调用。

安全默认：

- BFF 默认只监听 `127.0.0.1:8788`。
- `/healthz` 开放用于链路探测，其余接口强制 `Authorization: Bearer <token>`。
- job 数据仍写入 `runs/mcp_jobs/`，与 MCP server 共用同一份作业历史。
- 默认拒绝分析超过 20 分钟的视频，保护 GPU 队列和手机端等待时间。
- Mac/手机访问应通过 SSH LocalForward、VPN 或带鉴权的反向代理，不要直接公网暴露。

启动方式见 [server/README.md](server/README.md)。

## A/B Evaluation

把当前 VL+ASR 输出和 Omni 端到端输出做对比：

```bash
python -m video_understanding ab-eval \
  --vl-asr-context runs/demo/context.md \
  --omni-context runs/demo/omni.md \
  --output runs/demo/ab_report.md
```

建议用 5-10 条真实样本覆盖：

- 纯口播/旁白。
- 大量烧录字幕。
- 商品、价格、优惠、品牌信息。
- 音乐或情绪驱动内容。
- 需要精确片段定位的内容。

决策规则：

- 主要差距来自 OCR、画面细节、商品/界面文字：继续优化 VL+ASR。
- 主要差距来自非语音音频、音乐情绪、音画联合事件：评估引入 Omni。
- 主要差距来自定位：优先做视频 RAG/embedding，而不是盲目加大 VL 模型。

## GitHub Hygiene

这些目录不应提交：

- `models/`
- `runs/`
- `downloads/`
- `logs/`
- `.run/`
- `__pycache__/`

提交前建议检查：

```bash
git status --short
```

同时用你们团队的 secret scanning 工具检查 token、私钥、绝对 home 路径和本机用户名。

## TODO

- 增加 `--skip-asr`、`--skip-vl`、`--skip-fusion`，让长视频失败后可断点续跑。
- 增加本地 ASR 模型路径配置，避免生产环境首次运行时联网下载 `large-v3`。
- 增加 Qwen3-ASR backend，与 faster-whisper 做中文口播质量对比。
- 为 MCP HTTP 增加内建 Bearer token 校验，减少对反向代理鉴权的依赖。
- 增加视频下载器的真实站点集成测试和失败样例记录。
- 增加批量任务队列和多 VL 副本 router，支持 GPU 0/2 并发刷量。
- 增加 embedding/RAG：按 `context.md` 时间窗切 chunk，支持精确片段检索。
- 增加 A/B 评测模板，固化 VL+ASR vs Omni 的人工复核字段。
- 增加 Dockerfile 或 systemd unit，规范生产部署和开机恢复。
- 增加 CI：unit tests、shellcheck、README 命令 smoke test。
