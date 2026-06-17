from __future__ import annotations

import asyncio
import os
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware

from server.app import settings as settings_module
from server.app.auth import require_auth
from server.app.jobs import get_manager
from server.app.main import app


class BFFServerTests(unittest.TestCase):
    def tearDown(self) -> None:
        settings_module.get_settings.cache_clear()

    def test_default_repo_root_is_current_checkout(self):
        settings_module.get_settings.cache_clear()
        repo_root = Path(__file__).resolve().parents[1]

        self.assertEqual(Path(settings_module.DEFAULT_REPO_ROOT), repo_root)
        self.assertEqual(Path(settings_module.get_settings().repo_root), repo_root)

    def test_settings_allow_repo_root_override(self):
        with patch.dict(os.environ, {"XVIDEO_REPO_ROOT": "/tmp/xvideo"}, clear=False):
            settings_module.get_settings.cache_clear()
            self.assertEqual(settings_module.get_settings().repo_root, "/tmp/xvideo")

    def test_settings_parse_keep_media_flag(self):
        with patch.dict(os.environ, {"XVIDEO_KEEP_MEDIA": "1"}, clear=False):
            settings_module.get_settings.cache_clear()
            self.assertTrue(settings_module.get_settings().keep_media)

        with patch.dict(os.environ, {"XVIDEO_KEEP_MEDIA": "0"}, clear=False):
            settings_module.get_settings.cache_clear()
            self.assertFalse(settings_module.get_settings().keep_media)

    def test_auth_fails_closed_without_configured_token(self):
        with patch.dict(os.environ, {"XVIDEO_API_TOKEN": ""}, clear=False):
            settings_module.get_settings.cache_clear()
            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(require_auth(None))

        self.assertEqual(ctx.exception.status_code, 503)

    def test_auth_accepts_matching_bearer_token(self):
        with patch.dict(os.environ, {"XVIDEO_API_TOKEN": "secret-token"}, clear=False):
            settings_module.get_settings.cache_clear()
            asyncio.run(require_auth("Bearer secret-token"))

            with self.assertRaises(HTTPException) as ctx:
                asyncio.run(require_auth("Bearer wrong-token"))

        self.assertEqual(ctx.exception.status_code, 401)

    def test_get_manager_requires_lifespan_initialized_manager(self):
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(job_manager=None)))

        with self.assertRaises(HTTPException) as ctx:
            get_manager(request)  # type: ignore[arg-type]

        self.assertEqual(ctx.exception.status_code, 503)

        manager = object()
        request.app.state.job_manager = manager
        self.assertIs(get_manager(request), manager)  # type: ignore[arg-type]

    def test_cors_middleware_allows_app_clients(self):
        middleware = next(item for item in app.user_middleware if item.cls is CORSMiddleware)

        self.assertEqual(middleware.kwargs["allow_origins"], ["*"])
        self.assertEqual(middleware.kwargs["allow_methods"], ["*"])
        self.assertEqual(middleware.kwargs["allow_headers"], ["*"])


if __name__ == "__main__":
    unittest.main()
