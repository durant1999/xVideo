# BFF — FastAPI gateway for the xVideo pipeline

A thin REST/SSE adapter over `video_understanding.mcp_jobs.MCPJobManager` (the same
job manager the repo's MCP server uses). Deploys **on the GPU server**, inside the
existing `vedio_understand` conda env.

## Layout

```
app/
  main.py      # app factory · /healthz (open) · /ping (authed) · conditional jobs router
  settings.py  # env-driven config
  auth.py      # Bearer-token dependency (fails closed)
  jobs.py      # Phase 1 routes wrapping MCPJobManager
```

## Phase 0 — prove the link (do this first)

On the **GPU server**:

```bash
# 1. get this folder onto the server (rsync/scp/git), then:
cd server
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # paste into XVIDEO_API_TOKEN
conda run -n vedio_understand pip install -r requirements.txt
./run.sh            # serves on 127.0.0.1:8788
```

On the **Mac**: add the `LocalForward 127.0.0.1:8788 127.0.0.1:8788` line from
[../infra/ssh_config.snippet](../infra/ssh_config.snippet) to your SSH config,
reconnect `ssh BJ-10.10.150.55`, then:

```bash
curl http://127.0.0.1:8788/healthz                                   # {"status":"ok"}
curl -H "Authorization: Bearer <token>" http://127.0.0.1:8788/ping   # {"authenticated":true}
curl http://127.0.0.1:8788/ping                                      # 401 (no token) — expected
```

Once that works on the Mac, switch the SSH line to `0.0.0.0:8788` (or a Tailscale
IP) and hit `http://<mac-ip>:8788/healthz` from the phone.

## Phase 1 — enable the job routes

Set `XVIDEO_ENABLE_JOBS=1` in `.env`. If this `server/` directory lives inside the
xVideo checkout, `XVIDEO_REPO_ROOT` can stay unset because it is detected
automatically. If deployed elsewhere, set `XVIDEO_REPO_ROOT` to the real xVideo
checkout path, restart `./run.sh`, then:

```bash
TOKEN=<token>; BASE=http://127.0.0.1:8788
curl -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"source":"<video-url>","question":"这个视频在讲什么？"}' $BASE/jobs
curl -H "Authorization: Bearer $TOKEN" "$BASE/jobs?limit=5"
curl -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>"
curl -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>/events"          # SSE progress
curl -H "Authorization: Bearer $TOKEN" "$BASE/jobs/<job_id>/artifact/summary"
```

Interactive docs: `http://127.0.0.1:8788/docs`.

## Notes

- The BFF binds **loopback only** — never expose it publicly. Auth is required on
  everything except `/healthz`.
- Job execution, queuing (`max_workers=1`), persistence, cancellation and Q&A are
  all handled by `MCPJobManager`; this service does not duplicate any of it.
- The underlying CLI refuses videos longer than `video.max_duration_seconds`
  (30 minutes by default) before frame extraction, ASR, or VL inference.
- `MCPJobManager` is initialized once during FastAPI lifespan startup and shut down
  on application shutdown.
- SSE emits job state changes plus heartbeat events so mobile clients and tunnels
  can keep long-running streams alive.
- Default `XVIDEO_JOB_ROOT=runs/mcp_jobs` is shared with the MCP server so phone and
  Claude see one job history (see ARCHITECTURE.md §3 for the concurrency caveat).
