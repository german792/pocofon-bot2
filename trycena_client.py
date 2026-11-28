import asyncio
import logging
import os
import time
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)

TRYCENA_BASE_URL = os.getenv("TRYCENA_BASE_URL", "https://trycena.studio").rstrip("/")
TRYCENA_CLERK_URL = os.getenv("TRYCENA_CLERK_URL", "https://present-piglet-62.clerk.accounts.dev").rstrip("/")
TRYCENA_DB_JWT = (os.getenv("TRYCENA_DB_JWT") or "").strip()
TRYCENA_SESSION_ID = (os.getenv("TRYCENA_SESSION_ID") or "").strip()


def _map_trycena_resolution(model_id: str, resolution: str, ratio: str) -> str:
    """Map bot (resolution, ratio) to the exact resolutionOptions string expected by trycena.studio."""
    res = str(resolution or "720p").strip().lower()
    rat = str(ratio or "16:9").strip()

    if model_id in ("seedance-2.0",):
        return res if res in ("480p", "720p", "1080p", "4k") else "1080p"
    if model_id in ("seedance-2.0-fast",):
        return res if res in ("480p", "720p") else "720p"
    if model_id in ("wan-2.7", "pixverse-v6"):
        return "1080p" if res in ("1080p", "4k") else "720p"
    if model_id in ("grok-imagine-video",):
        return "720p"
    if model_id in ("google-veo-3.1", "google-veo-3.1-fast", "google-veo-3.1-lite"):
        r_part = "720p" if res in ("480p", "720p") else "1080p"
        rat_part = "9:16" if rat == "9:16" else "16:9"
        return f"{r_part} ({rat_part})"
    if model_id == "klingai-video-3.0-omni-pro":
        if rat == "9:16":
            return "1080x1920 (9:16)"
        if rat == "1:1":
            return "1440x1440 (1:1)"
        return "1920x1080 (16:9)"
    if model_id == "klingai-video-3.0-4k":
        if rat == "9:16":
            return "2160x3840 (9:16)"
        if rat == "1:1":
            return "2880x2880 (1:1)"
        return "3840x2160 (16:9)"
    if model_id == "ltx-2-pro":
        if res == "4k":
            return "3840x2160 (4K)"
        if res == "1440p":
            return "2560x1440 (1440p)"
        return "1920x1080 (1080p)"
    if model_id == "minimax-hailuo-2.3":
        return "1366x768 (16:9 Landscape)" if res in ("480p", "720p") else "1920x1080 (16:9 Landscape)"
    if model_id == "runway-gen-4.5":
        if rat == "9:16":
            return "720x1280 (9:16)"
        if rat == "4:3":
            return "1104x832 (4:3)"
        if rat == "3:4":
            return "832x1104 (3:4)"
        return "1280x720 (16:9)"
    return res or "720p"


class TrycenaClient:
    """Async client for video generation with automatic Clerk short-lived JWT minting."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        clerk_url: Optional[str] = None,
        db_jwt: Optional[str] = None,
        session_id: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("TRYCENA_BASE_URL") or TRYCENA_BASE_URL).rstrip("/")
        self.clerk_url = (clerk_url or os.getenv("TRYCENA_CLERK_URL") or TRYCENA_CLERK_URL).rstrip("/")
        self.db_jwt = (db_jwt if db_jwt is not None else (os.getenv("TRYCENA_DB_JWT") or TRYCENA_DB_JWT)).strip()
        self.session_id = (
            session_id if session_id is not None else (os.getenv("TRYCENA_SESSION_ID") or TRYCENA_SESSION_ID)
        ).strip()
        self._jwt: str = ""
        self._jwt_expires_at: float = 0.0
        self._lock = asyncio.Lock()

    def get_full_url(self, url_or_path: str) -> str:
        raw = str(url_or_path or "").strip()
        if not raw:
            return ""
        if raw.startswith(("http://", "https://")):
            return raw
        return f"{self.base_url}/{raw.lstrip('/')}"

    async def _mint_jwt(self, force: bool = False) -> str:
        async with self._lock:
            now = time.time()
            if not force and self._jwt and now < self._jwt_expires_at - 10.0:
                return self._jwt
            if not self.db_jwt or not self.session_id:
                raise RuntimeError("Не настроена сессия генерации видео (TRYCENA_DB_JWT / TRYCENA_SESSION_ID).")

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{self.clerk_url}/v1/client/sessions/{self.session_id}/tokens",
                    params={"__clerk_db_jwt": self.db_jwt, "_clerk_js_version": "5.0.0"},
                    headers={
                        "Origin": self.base_url,
                        "Referer": f"{self.base_url}/",
                        "User-Agent": (
                            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                        ),
                    },
                )
                if resp.status_code >= 400:
                    raise RuntimeError(f"Сессия истекла ({resp.status_code}). Обновите токен сессии.")
                data = resp.json()
                jwt = str(data.get("jwt") or "").strip()
                if not jwt:
                    raise RuntimeError("Не удалось обновить токен сессии.")
                self._jwt = jwt
                self._jwt_expires_at = now + 50.0
                return jwt

    async def _auth_headers(self, force_refresh: bool = False, extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        jwt = await self._mint_jwt(force=force_refresh)
        h: Dict[str, str] = {
            "Accept": "application/json",
            "Authorization": f"Bearer {jwt}",
            "Cookie": f"__session={jwt}; __clerk_db_jwt={self.db_jwt}",
            "Origin": self.base_url,
            "Referer": f"{self.base_url}/",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
            ),
        }
        if extra:
            h.update(extra)
        return h

    async def create_video_task(
        self,
        model_info: Dict[str, Any],
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "16:9",
        generate_audio: bool = True,
    ) -> str:
        """Submit POST /api/generate-video and return taskId."""
        model_id = str(model_info.get("api_id") or "seedance-2.0-fast")
        allowed_durs = model_info.get("durations") or [5]
        dur = int(duration or allowed_durs[0])
        if dur not in allowed_durs:
            dur = min(allowed_durs, key=lambda d: abs(d - dur))

        mapped_res = _map_trycena_resolution(model_id, resolution, ratio)
        payload: Dict[str, Any] = {
            "prompt": (prompt or "").strip(),
            "model": model_id,
            "resolution": mapped_res,
            "duration": dur,
            "generate_audio": bool(generate_audio) if model_info.get("has_audio") else False,
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            headers = await self._auth_headers(extra={"Content-Type": "application/json"})
            resp = await client.post(f"{self.base_url}/api/generate-video", headers=headers, json=payload)
            if resp.status_code == 401:
                headers = await self._auth_headers(force_refresh=True, extra={"Content-Type": "application/json"})
                resp = await client.post(f"{self.base_url}/api/generate-video", headers=headers, json=payload)
            if resp.status_code >= 400:
                msg = ""
                try:
                    msg = str(resp.json().get("error") or resp.json().get("message") or "")
                except Exception:
                    msg = resp.text[:250]
                raise RuntimeError(f"Ошибка запуска генерации ({resp.status_code}): {msg}")

            data = resp.json()
            task_id = str(data.get("taskId") or data.get("task_id") or data.get("id") or "")
            if not task_id:
                raise RuntimeError("Сервер не вернул ID задачи генерации.")
            return task_id

    async def get_task(self, task_id: str) -> Dict[str, Any]:
        """Poll GET /api/video-status/{task_id}."""
        async with httpx.AsyncClient(timeout=20.0) as client:
            headers = await self._auth_headers()
            resp = await client.get(f"{self.base_url}/api/video-status/{task_id}", headers=headers)
            if resp.status_code == 401:
                headers = await self._auth_headers(force_refresh=True)
                resp = await client.get(f"{self.base_url}/api/video-status/{task_id}", headers=headers)
            if resp.status_code >= 400:
                return {"status": "PROCESSING", "progress": 0.25}
            data = resp.json()
            task = data.get("task") if isinstance(data, dict) else None
            if not isinstance(task, dict):
                return {"status": "PROCESSING", "progress": 0.25}

            st = str(task.get("status") or "processing").lower()
            if st in ("completed", "done", "succeeded"):
                vurl = str(task.get("video_url") or task.get("videoUrl") or "")
                return {"status": "COMPLETED", "video_url": self.get_full_url(vurl), "progress": 1.0}
            if st in ("failed", "error", "cancelled"):
                err = str(task.get("error_message") or task.get("error") or "Генерация завершилась с ошибкой.")
                return {"status": "FAILED", "error": err}
            return {"status": "PROCESSING", "progress": 0.45}

    async def download_video(self, url_or_path: str) -> bytes:
        full_url = self.get_full_url(url_or_path)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            headers = await self._auth_headers() if full_url.startswith(self.base_url) else {}
            resp = await client.get(full_url, headers=headers)
            resp.raise_for_status()
            return resp.content
