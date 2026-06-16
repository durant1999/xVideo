# Deployment Runbook

## 固定决策

- 质量优先：默认 `Qwen3-VL-32B-Instruct`，AWQ-INT4，单卡 TP=1。
- A100 PCIe 没有 NVLink：避免跨卡 TP/EP，不使用 TP=3；最多 TP=2，默认不用。
- 长视频瓶颈主要是 KV cache 和视觉 token：`max-model-len` 卡到 128K，客户端分段抽帧并压缩分辨率。
- 卡 0：32B-INT4 主 VL/OCR。
- 卡 1：ASR。
- 卡 2：按吞吐需求起第二个 VL 副本，或给 embedding/RAG。

## 1. 环境

推荐 Python 3.10/3.11、CUDA 驱动匹配 vLLM wheel。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[client,asr,server,mcp]"
sudo apt-get install -y ffmpeg
```

当前已验证组合是 `vllm==0.11.0` + `transformers==4.57.1`。不要让 `transformers` 升到 5.x；vLLM 0.11 的 tokenizer 缓存逻辑仍依赖 4.x 的 `all_special_tokens_extended` 属性。

如果用 URL 输入，需要 `yt-dlp`；如果只处理本地 mp4，可以不走下载命令。

## 2. 权重

当前采用 `QuantTrio/Qwen3-VL-32B-Instruct-AWQ` 社区量化权重：

```bash
hf download QuantTrio/Qwen3-VL-32B-Instruct-AWQ \
  --local-dir models/Qwen3-VL-32B-Instruct-AWQ \
  --max-workers 3
```

如果这版后续 A/B 质量不过关，再回到官方 `Qwen/Qwen3-VL-32B-Instruct` 自量化：

```bash
pip install -e ".[quant]"
MODEL_PATH=models/Qwen3-VL-32B-Instruct \
OUTPUT_PATH=models/Qwen3-VL-32B-Instruct-AWQ \
scripts/quantize_qwen3_vl_awq.sh
```

自量化前先用 5-10 条真实样本做小批校准和人工检查，避免 OCR 细节劣化。

## 3. 卡 0 启动 VL

推荐用后台管理脚本启动。脚本会写入 PID/日志，并等待 OpenAI-compatible `/v1/models` 健康检查。先激活你的 Python/conda 环境，或用 `CONDA_ENV=<env-name>` 显式指定：

```bash
conda activate video_understand
scripts/manage_vl_server.sh start
scripts/manage_vl_server.sh status
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
START_TIMEOUT=1200 \
scripts/manage_vl_server.sh start
```

如果要前台调试 vLLM，再直接运行底层启动器：

```bash
CUDA_VISIBLE_DEVICES=0 \
MODEL=models/Qwen3-VL-32B-Instruct-AWQ \
SERVED_MODEL_NAME=Qwen3-VL-32B-Instruct-AWQ \
PORT=8000 \
MAX_MODEL_LEN=131072 \
LIMIT_MM_IMAGES=80 \
scripts/launch_vllm_qwen3_vl_32b_awq.sh
```

`LIMIT_MM_IMAGES` 需要和客户端分段长度、fps 匹配。默认 45 秒 × 1 fps 约 45 张，留出余量。

## 4. 卡 1 跑 ASR

默认 CLI 用 faster-whisper 本地加载 `large-v3`：

```bash
CUDA_VISIBLE_DEVICES=1 python -m video_understanding asr /path/to/video.mp4 --workdir runs/demo
```

如果换 Qwen3-ASR，把 `video_understanding/asr.py` 增加对应 backend，保持输出字段 `start/end/text` 不变即可。

## 5. 单条闭环

```bash
python -m video_understanding run /path/to/video.mp4 --workdir runs/demo
```

长视频建议先保持：

- `fps: 1.0`
- `segment_seconds: 45`
- `max_side: 960`
- `max_duration_seconds: 1200`，默认拒绝超过 20 分钟的视频，避免长任务占满队列。

如果漏短动作，把 fps 调到 2；如果 KV/显存压力明显，先降 `segment_seconds`，再降 `max_side`。

## 6. A/B 验证关口

取 5-10 个真实抖音片段，覆盖：

- 纯口播/旁白；
- 大量烧录字幕；
- 商品/价格/优惠信息；
- 音乐或情绪驱动内容；
- 需要精确片段定位的内容。

每条视频都产出：

- `context.md`：VL+ASR 融合输出。
- `omni.md`：Qwen3-Omni-Thinking 端到端输出。
- `ab_report.md`：用 `ab-eval` 生成的比较报告，再人工复核。

决策规则：

- 主要差距来自 OCR、画面细节、商品/界面文字：继续优化 VL+ASR。
- 主要差距来自非语音音频、音乐情绪、音画联合事件：保留 Omni 路线。
- 主要差距来自定位：卡 2 优先上 embedding/RAG，而不是盲目加大 VL。

## 7. 卡 2

吞吐优先：

```bash
CUDA_VISIBLE_DEVICES=2 PORT=8002 scripts/launch_vllm_qwen3_vl_8b_bf16.sh
```

质量优先且预算允许：卡 2 起第二个 32B-AWQ 副本，把上游任务按视频维度分发到 `:8000` 和 `:8002`。

精确定位优先：卡 2 放 embedding，按 `context.md` 的时间窗切 chunk，建立视频 RAG。

## 8. MCP 给 Mac Codex 调用

GPU 服务器启动 MCP server：

```bash
conda activate video_understand
scripts/launch_mcp_server.sh
```

默认只监听 `127.0.0.1:9000/mcp`。Mac 通过 SSH tunnel 访问：

```bash
ssh -L 9000:127.0.0.1:9000 gpu-server
```

Mac 的 `~/.codex/config.toml`：

```toml
[mcp_servers.video_understanding]
url = "http://127.0.0.1:9000/mcp"
tool_timeout_sec = 120
```

MCP 工具是异步任务模型：先 `submit_video_job` 返回 `job_id`，再 `get_job_status` 轮询，最后 `get_job_artifact` 读取 `summary` 或 `context`。

安全说明：

- 仓库公开不等于服务公开。服务是否能被别人访问，取决于运行时监听地址、SSH/VPN/反向代理和防火墙。
- `scripts/launch_mcp_server.sh` 默认只绑定 `127.0.0.1`，外部机器无法直接连接。
- 有 VPN 时也不要把 MCP server 绑定到 `0.0.0.0` 或 VPN 网卡 IP，除非已经加了鉴权和防火墙；否则 VPN 内其他机器可能访问。
- 推荐保持服务器端 `127.0.0.1:9000`，Mac 使用 SSH `LocalForward` 转发。

## 9. App BFF 给手机前端调用

`server/` 是 FastAPI BFF，部署在 GPU 服务器的 `vedio_understand` 环境里，给 Android/Web 前端提供 REST/SSE 接口。

BFF 复用 CLI 的视频时长保护：默认超过 20 分钟的视频会在抽帧、ASR、VL 之前失败，不会烧 GPU。

Phase 0 先只验证链路：

```bash
cd server
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"
conda run -n vedio_understand pip install -r requirements.txt
./run.sh
```

默认监听 `127.0.0.1:8788`。Mac 先用本机 LocalForward 验证：

```sshconfig
LocalForward 127.0.0.1:8788 127.0.0.1:8788
```

手机阶段再按需改为绑定 Mac 的局域网或 Tailscale 入口。BFF 除 `/healthz` 外都需要 Bearer token；不要把服务器端 BFF 直接绑定到公网地址。
