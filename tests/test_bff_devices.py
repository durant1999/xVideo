from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from server.app import devices
from server.app.main import _watch_completions, app


class DeviceRegistryTests(unittest.TestCase):
    def test_device_route_is_mounted(self):
        self.assertTrue(
            any("/devices" in getattr(route, "path", "") for route in app.routes)
        )

    def test_add_device_is_idempotent_and_updates_platform(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Path(temp_dir) / "devices.json"
            with mock.patch("server.app.devices.store_path", return_value=store):
                devices.add_device(" push-token-abc ", "ios")
                devices.add_device("push-token-abc", "android")

                rows = json.loads(store.read_text(encoding="utf-8"))

        self.assertEqual(rows, [{"push_token": "push-token-abc", "platform": "android"}])

    def test_notify_job_done_sends_expo_payload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = Path(temp_dir) / "devices.json"
            with mock.patch("server.app.devices.store_path", return_value=store):
                devices.add_device("push-token-abc", "android")
                with mock.patch("server.app.devices._send_expo") as send:
                    devices.notify_job_done(
                        {
                            "job_id": "job-1",
                            "status": "succeeded",
                            "source": "https://example.com/video",
                        }
                    )

        send.assert_called_once()
        payload = send.call_args.args[0]
        self.assertEqual(payload[0]["to"], "push-token-abc")
        self.assertEqual(payload[0]["title"], "分析完成")
        self.assertEqual(payload[0]["data"], {"jobId": "job-1"})


class CompletionWatcherTests(unittest.TestCase):
    def test_watcher_seeds_existing_terminal_jobs_and_notifies_new_ones(self):
        notified = []

        class FakeManager:
            def __init__(self) -> None:
                self.calls = 0

            def list_jobs(self, limit=100):
                self.calls += 1
                if self.calls == 1:
                    return [{"job_id": "old", "status": "succeeded"}]
                return [
                    {"job_id": "old", "status": "succeeded"},
                    {"job_id": "new", "status": "failed"},
                ]

        async def run_test() -> None:
            async def fake_to_thread(func, job):
                notified.append(job["job_id"])

            with mock.patch("asyncio.to_thread", side_effect=fake_to_thread):
                task = asyncio.create_task(_watch_completions(FakeManager(), poll_seconds=0.01))
                try:
                    for _ in range(20):
                        if notified:
                            break
                        await asyncio.sleep(0.01)
                finally:
                    task.cancel()
                    with self.assertRaises(asyncio.CancelledError):
                        await task

        asyncio.run(run_test())

        self.assertEqual(notified, ["new"])


if __name__ == "__main__":
    unittest.main()
