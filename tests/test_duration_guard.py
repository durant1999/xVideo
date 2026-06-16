from __future__ import annotations

import argparse
import unittest

from video_understanding.cli import max_duration_seconds_for
from video_understanding.config import DEFAULT_CONFIG
from video_understanding.media import enforce_duration_limit, format_duration
from video_understanding.utils import PipelineError


class DurationGuardTests(unittest.TestCase):
    def test_default_max_duration_is_thirty_minutes(self):
        self.assertEqual(DEFAULT_CONFIG["video"]["max_duration_seconds"], 1800)

    def test_enforce_duration_limit_rejects_over_limit_video(self):
        with self.assertRaises(PipelineError) as ctx:
            enforce_duration_limit(1801.0, 1800, video_path="/tmp/long.mp4")

        message = str(ctx.exception)
        self.assertIn("30m 1s", message)
        self.assertIn("30m", message)
        self.assertIn("Refusing to analyze", message)

    def test_enforce_duration_limit_allows_equal_and_disabled_limits(self):
        enforce_duration_limit(1800.0, 1800)
        enforce_duration_limit(999999.0, 0)
        enforce_duration_limit(999999.0, None)

    def test_cli_override_takes_precedence_over_config(self):
        args = argparse.Namespace(max_duration_seconds=60)

        self.assertEqual(max_duration_seconds_for(args, {"max_duration_seconds": 1800}), 60)

    def test_negative_cli_limit_is_rejected(self):
        args = argparse.Namespace(max_duration_seconds=-1)

        with self.assertRaises(PipelineError):
            max_duration_seconds_for(args, {"max_duration_seconds": 1800})

    def test_format_duration(self):
        self.assertEqual(format_duration(59.4), "59s")
        self.assertEqual(format_duration(1800), "30m 0s")
        self.assertEqual(format_duration(3661), "1h 1m 1s")


if __name__ == "__main__":
    unittest.main()
