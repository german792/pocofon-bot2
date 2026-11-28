import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

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
        self._org_id: str = (os.getenv("TRYCENA_ORG_ID") or "").strip()
        self._workspace_id: Optional[int] = None
        self._scene_id: Optional[int] = None
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
                common_headers = {
                    "Origin": self.base_url,
                    "Referer": f"{self.base_url}/",
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
                    ),
                }
                if not self._org_id:
                    try:
                        rc = await client.get(
                            f"{self.clerk_url}/v1/client",
                            params={"__clerk_db_jwt": self.db_jwt, "_clerk_js_version": "5.0.0"},
                            headers=common_headers,
                        )
                        if rc.status_code == 200:
                            cd = rc.json().get("response") or rc.json()
                            sessions = cd.get("sessions") or []
                            if sessions and isinstance(sessions[0], dict):
                                orgs = (sessions[0].get("user") or {}).get("organization_memberships") or []
                                if orgs and isinstance(orgs[0], dict):
                                    oid = (orgs[0].get("organization") or {}).get("id")
                                    if oid:
                                        self._org_id = str(oid)
                    except Exception as org_err:
                        logger.debug(f"[trycena] Org lookup note: {org_err}")

                token_data = {"organization_id": self._org_id} if self._org_id else None
                resp = await client.post(
                    f"{self.clerk_url}/v1/client/sessions/{self.session_id}/tokens",
                    params={"__clerk_db_jwt": self.db_jwt, "_clerk_js_version": "5.0.0"},
                    headers=common_headers,
                    data=token_data,
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

    async def _ensure_workspace(self) -> int:
        """Return an active workspace ID (and cache its scene_id) for uploading photo references."""
        if self._workspace_id is not None and self._scene_id is not None:
            return self._workspace_id
        async with httpx.AsyncClient(timeout=20.0) as client:
            headers = await self._auth_headers(extra={"Content-Type": "application/json"})
            if self._workspace_id is None:
                r = await client.get(f"{self.base_url}/api/workspaces", headers=headers)
                if r.status_code == 200:
                    ws_list = r.json().get("data") or []
                    if ws_list and isinstance(ws_list[0], dict) and ws_list[0].get("id"):
                        self._workspace_id = int(ws_list[0]["id"])
                if self._workspace_id is None:
                    rc = await client.post(
                        f"{self.base_url}/api/workspaces",
                        headers=headers,
                        json={"name": "PocoAI Refs"},
                    )
                    if rc.status_code in (200, 201):
                        ws = rc.json().get("data") or rc.json()
                        if isinstance(ws, dict) and ws.get("id"):
                            self._workspace_id = int(ws["id"])
                    else:
                        raise RuntimeError(
                            f"Не удалось подготовить рабочее пространство для референсов ({rc.status_code})."
                        )

            # Fetch workspace detail to obtain an active scene_id (prefer non-special Scene 1)
            if self._scene_id is None and self._workspace_id is not None:
                rd = await client.get(f"{self.base_url}/api/workspaces/{self._workspace_id}", headers=headers)
                if rd.status_code == 200:
                    scenes = (rd.json().get("data") or {}).get("scenes") or []
                    non_special = [s for s in scenes if isinstance(s, dict) and not s.get("is_special")]
                    chosen = non_special[0] if non_special else (scenes[0] if scenes else None)
                    if isinstance(chosen, dict) and chosen.get("id"):
                        self._scene_id = int(chosen["id"])

            if self._workspace_id is None:
                raise RuntimeError("Не удалось получить рабочее пространство для референсов.")
            return self._workspace_id

    async def upload_reference_image(self, image_bytes: bytes, idx: int = 1) -> int:
        """Upload a photo reference to the active workspace and return its imageId."""
        ws_id = await self._ensure_workspace()
        fname = f"ref_{int(time.time())}_{idx}.jpg"
        async with httpx.AsyncClient(timeout=45.0) as client:
            headers = await self._auth_headers()
            resp = await client.post(
                f"{self.base_url}/api/workspaces/{ws_id}/images",
                headers=headers,
                files={"image": (fname, image_bytes, "image/jpeg")},
                data={"name": fname},
            )
            if resp.status_code == 401:
                headers = await self._auth_headers(force_refresh=True)
                resp = await client.post(
                    f"{self.base_url}/api/workspaces/{ws_id}/images",
                    headers=headers,
                    files={"image": (fname, image_bytes, "image/jpeg")},
                    data={"name": fname},
                )
            if resp.status_code >= 400:
                raise RuntimeError(f"Ошибка загрузки фото-референса ({resp.status_code}): {resp.text[:180]}")
            data = resp.json().get("data") or resp.json()
            img_id = data.get("id") if isinstance(data, dict) else None
            if not img_id:
                raise RuntimeError("Сервер не вернул ID загруженного фото-референса.")
            return int(img_id)

    async def create_video_task(
        self,
        model_info: Dict[str, Any],
        prompt: str,
        duration: int = 5,
        resolution: str = "720p",
        ratio: str = "16:9",
        generate_audio: bool = True,
        ref_images: Optional[List[bytes]] = None,
        ref_mode: str = "omni",
    ) -> str:
        """Submit POST /api/generate-video (with optional photo references) and return taskId."""
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

        imgs = [b for b in (ref_images or []) if b]
        max_refs = int(model_info.get("ref_limit") or 0)
        if imgs and max_refs > 0:
            ws_id = await self._ensure_workspace()
            use_multi = bool(model_info.get("multi_ref"))
            uploaded_ids: List[int] = []
            for idx, img_bytes in enumerate(imgs[:max_refs], start=1):
                uploaded_ids.append(await self.upload_reference_image(img_bytes, idx=idx))

            image_inputs: List[Dict[str, Any]] = []
            if len(uploaded_ids) == 1:
                image_inputs.append({"imageId": uploaded_ids[0], "purpose": "first-frame"})
            elif ref_mode == "frames" or not use_multi:
                image_inputs.append({"imageId": uploaded_ids[0], "purpose": "first-frame"})
                image_inputs.append({"imageId": uploaded_ids[1], "purpose": "last-frame"})
            else:
                image_inputs.append({"imageId": uploaded_ids[0], "purpose": "first-frame"})
                for iid in uploaded_ids[1:]:
                    image_inputs.append({"imageId": iid, "purpose": "reference-asset"})

            payload["workspace_id"] = ws_id
            if self._scene_id is not None:
                payload["scene_id"] = self._scene_id
            payload["image_inputs"] = image_inputs

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

            data = resp.json() if resp.content else {}
            nested_task = data.get("task") if isinstance(data, dict) and isinstance(data.get("task"), dict) else {}
            nested_data = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), dict) else {}
            task_ids_list = data.get("taskIds") if isinstance(data, dict) and isinstance(data.get("taskIds"), list) else []
            task_id = str(
                (data.get("taskId") if isinstance(data, dict) else None)
                or (task_ids_list[0] if task_ids_list else None)
                or (data.get("task_id") if isinstance(data, dict) else None)
                or (data.get("id") if isinstance(data, dict) else None)
                or nested_task.get("id")
                or nested_task.get("taskId")
                or nested_data.get("taskId")
                or nested_data.get("task_id")
                or nested_data.get("id")
                or ""
            ).strip()
            if not task_id:
                logger.warning(f"[trycena] POST /api/generate-video returned no direct taskId: {resp.text[:400]}")
                # Fallback: grab the newly created task from /api/user-tasks
                try:
                    ut = await client.get(f"{self.base_url}/api/user-tasks", headers=await self._auth_headers())
                    if ut.status_code == 200:
                        tasks = ut.json().get("tasks") or []
                        now_ms = int(time.time() * 1000)
                        for t in tasks:
                            if not isinstance(t, dict):
                                continue
                            created = int(t.get("created_at") or 0)
                            if (
                                t.get("model") == model_id
                                and (t.get("prompt") or "").strip() == (prompt or "").strip()
                                and abs(now_ms - created) < 30000
                            ):
                                task_id = str(t.get("id") or "").strip()
                                break
                        if not task_id and tasks and isinstance(tasks[0], dict):
                            created0 = int(tasks[0].get("created_at") or 0)
                            if abs(now_ms - created0) < 15000:
                                task_id = str(tasks[0].get("id") or "").strip()
                except Exception as fb_err:
                    logger.warning(f"[trycena] Fallback /api/user-tasks lookup failed: {fb_err}")
            if not task_id:
                raise RuntimeError(f"Сервер не вернул ID задачи генерации ({resp.text[:160]}).")
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
                full = self.get_full_url(vurl)
                return {"status": "succeeded", "videoUrl": full, "video_url": full, "progress": 1.0}
            if st in ("failed", "error", "cancelled"):
                err = str(task.get("error_message") or task.get("error") or "Генерация завершилась с ошибкой.")
                return {"status": "failed", "error": err}
            return {"status": "processing", "progress": 0.45}

    async def download_video(self, url_or_path: str) -> bytes:
        full_url = self.get_full_url(url_or_path)
        async with httpx.AsyncClient(timeout=120.0, follow_redirects=True) as client:
            headers = await self._auth_headers() if full_url.startswith(self.base_url) else {}
            resp = await client.get(full_url, headers=headers)
            resp.raise_for_status()
            return resp.content
