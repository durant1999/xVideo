from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from video_understanding.mcp_jobs import MCPJobManager
from video_understanding.utils import PipelineError


class MCPJobManagerTests(unittest.TestCase):
    def make_manager(self, temp_dir: str) -> MCPJobManager:
        repo_root = Path(temp_dir)
        (repo_root / "configs").mkdir()
        (repo_root / "configs" / "pipeline.yaml").write_text("workdir: runs\n", encoding="utf-8")
        return MCPJobManager(
            repo_root=repo_root,
            job_root="runs/mcp_jobs",
            config_path="configs/pipeline.yaml",
            python_executable="/usr/bin/python",
            execute_jobs=False,
        )

    def test_submit_video_job_persists_state_without_user_workdir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            state = manager.submit_video_job(
                source="https://v.douyin.com/example/",
                question="商品价格是什么？",
                fps=1.0,
                segment_seconds=45,
                max_side=960,
            )

            self.assertEqual(state["status"], "queued")
            self.assertIn("job_id", state)
            self.assertTrue(state["workdir"].endswith("/work"))
            self.assertIn("summary", state["artifacts"])
            self.assertTrue(state["cleanup_media_on_success"])

            persisted = manager.read_state(state["job_id"])
            command = persisted["command"]
            self.assertEqual(command[:4], ["/usr/bin/python", "-m", "video_understanding", "run"])
            self.assertIn("--workdir", command)
            self.assertIn("--config", command)
            self.assertIn("--question", command)
            self.assertIn("--fps", command)
            self.assertIn("--segment-seconds", command)
            self.assertIn("--max-side", command)

    def test_submit_rejects_out_of_bounds_overrides(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)

            with self.assertRaises(PipelineError):
                manager.submit_video_job(source="https://example.com/video", fps=20)

            with self.assertRaises(PipelineError):
                manager.submit_video_job(source="https://example.com/video", segment_seconds=1)

            with self.assertRaises(PipelineError):
                manager.submit_video_job(source="https://example.com/video", max_side=10000)

    def test_read_artifact_is_allow_listed_and_truncated(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            state = manager.submit_video_job(source="https://example.com/video")
            job_id = state["job_id"]
            context_path = manager.artifact_path(job_id, "context")
            context_path.parent.mkdir(parents=True, exist_ok=True)
            context_path.write_text("abcdef", encoding="utf-8")

            artifact = manager.read_artifact(job_id, "context", max_chars=3)

            self.assertTrue(artifact["exists"])
            self.assertEqual(artifact["content"], "abc")
            self.assertTrue(artifact["truncated"])

            with self.assertRaises(PipelineError):
                manager.read_artifact(job_id, "../../etc/passwd")

    def test_list_jobs_returns_newest_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            first = manager.submit_video_job(source="https://example.com/1")
            second = manager.submit_video_job(source="https://example.com/2")

            rows = manager.list_jobs(limit=10)

            self.assertEqual(rows[0]["job_id"], second["job_id"])
            self.assertEqual(rows[1]["job_id"], first["job_id"])

    def test_cleanup_media_assets_preserves_text_artifacts_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = self.make_manager(temp_dir)
            state = manager.submit_video_job(source="https://example.com/video")
            job_id = state["job_id"]
            workdir = Path(state["workdir"])

            preserved_files = [
                workdir / "summary.md",
                workdir / "context.md",
                workdir / "visual.jsonl",
                workdir / "asr.jsonl",
                workdir / "fused.jsonl",
                workdir / "source" / "download_metadata.json",
            ]
            heavy_files = [
                workdir / "source" / "video.mp4",
                workdir / "source" / "cover.jpg",
                workdir / "audio.wav",
                workdir / "frames" / "0000" / "frame_000001.jpg",
            ]
            for path in preserved_files + heavy_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"x" * 8)

            result = manager.cleanup_media_assets(job_id)

            self.assertTrue(result["enabled"])
            self.assertGreaterEqual(result["bytes_freed"], 32)
            for path in heavy_files:
                self.assertFalse(path.exists(), str(path))
            for path in preserved_files:
                self.assertTrue(path.exists(), str(path))
            self.assertFalse((workdir / "frames").exists())


if __name__ == "__main__":
    unittest.main()
