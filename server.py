from __future__ import annotations

import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable

import uvicorn
import requests
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import app as core

ROOT = Path(__file__).resolve().parent
UPLOADS = core.OUTPUTS / "uploads"
DIST = ROOT / "frontend" / "dist"
UPLOADS.mkdir(parents=True, exist_ok=True)

api = FastAPI(title="AutoVid Pro API", version="1.0")
executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="autovid-job")
tasks: dict[str, dict[str, Any]] = {}
tasks_lock = threading.Lock()


class RecognizeRequest(BaseModel):
    media: str
    reference: str | None = None
    asr_model: str = "large-v3-turbo"
    source_language: str = "中文"
    preserve_background: bool = True
    target_language: str = "英语"
    translation_style: str = "自然口语"
    translation_provider: str = "本地模型"
    translation_url: str = ""
    translation_key: str = ""
    translation_model: str = ""


class TranslateRequest(BaseModel):
    rows: list[list[Any]]
    target_language: str = "英语"
    style: str = "自然口语"
    provider: str = "本地模型"
    base_url: str = ""
    api_key: str = ""
    model: str = ""


class RemoteModelsRequest(BaseModel):
    base_url: str
    api_key: str


class SynthesizeRequest(BaseModel):
    rows: list[list[Any]]
    provider: str = "CosyVoice2 本地克隆"
    model_path: str = ""
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    voice: str = "alloy"
    default_voice: str = "项目自动参考"
    speaker_mapping: str = ""
    emotion: str = "自然"
    speed: float = 1.0
    pitch: float = 0.0
    voice_volume: float = 1.0
    background_volume: float = 0.75
    burn_subtitles: bool = False
    subtitle_font: str = "Microsoft YaHei"
    subtitle_size: int = 36
    subtitle_position: str = "底部"


def set_task(task_id: str, **values: Any) -> None:
    with tasks_lock:
        tasks.setdefault(task_id, {}).update(values)


def submit_job(label: str, fn: Callable[[], Any]) -> str:
    task_id = uuid.uuid4().hex[:12]
    set_task(task_id, id=task_id, label=label, state="queued", message="任务已进入队列", stage="排队中", result=None,
             started_at=None, updated_at=time.time(), completed=None, total=None, percent=None)

    def report(*, stage: str, completed: int | None = None, total: int | None = None) -> None:
        values: dict[str, Any] = {"stage": stage, "updated_at": time.time()}
        if completed is not None and total is not None:
            values.update(completed=completed, total=total, percent=round(completed / total * 100) if total else None)
        else:
            values.update(completed=None, total=None, percent=None)
        set_task(task_id, **values)

    def runner() -> None:
        set_task(task_id, state="running", message="任务正在处理", stage="正在准备", started_at=time.time(), updated_at=time.time())
        core._TASK_PROGRESS.reporter = report
        try:
            result = fn()
            set_task(task_id, state="completed", message="任务完成", stage="处理完成", result=result,
                     percent=100, updated_at=time.time())
        except Exception as exc:
            core.console_log(f"API task failed: {type(exc).__name__}: {exc}")
            set_task(task_id, state="failed", message=str(exc), stage="处理失败", updated_at=time.time())
        finally:
            core._TASK_PROGRESS.reporter = None

    executor.submit(runner)
    return task_id


@api.get("/api/health")
def health() -> dict[str, Any]:
    return {"ok": True, "cuda": core._cuda_available(), "status": core.get_status()}


@api.post("/api/uploads")
def upload(file: UploadFile = File(...)) -> dict[str, str]:
    suffix = Path(file.filename or "upload.bin").suffix
    target = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    with target.open("wb") as output:
        shutil.copyfileobj(file.file, output)
    return {"name": file.filename or target.name, "path": str(target)}


@api.post("/api/remote-models")
def remote_models(request: RemoteModelsRequest) -> dict[str, list[str]]:
    if not request.base_url.strip() or not request.api_key.strip():
        raise HTTPException(status_code=400, detail="请先填写 Base URL 和 API Key。")
    url = request.base_url.strip().rstrip("/")
    for suffix in ("/chat/completions", "/models"):
        if url.endswith(suffix):
            url = url[:-len(suffix)]
            break
    try:
        response = requests.get(
            f"{url}/models",
            headers={"Authorization": f"Bearer {request.api_key.strip()}"},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        values = payload.get("data", payload.get("models", []))
        models = sorted({str(item.get("id") if isinstance(item, dict) else item) for item in values if item})
        models = [model for model in models if model and model != "None"]
        if not models:
            raise ValueError("平台没有返回可用模型。")
        return {"models": models}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"拉取模型失败：{exc}") from exc


@api.post("/api/projects/recognize")
def recognize(request: RecognizeRequest) -> dict[str, str]:
    def work() -> dict[str, Any]:
        project_id, rows, message = core.prepare_studio_project(
            request.media,
            request.reference,
            request.asr_model,
            request.source_language,
            request.preserve_background,
            request.target_language,
            request.translation_style,
            request.translation_provider,
            request.translation_url,
            request.translation_key,
            request.translation_model,
        )
        return {"project_id": project_id, "rows": rows, "message": message}

    return {"task_id": submit_job("识别与首次翻译", work)}


@api.get("/api/projects/{project_id}")
def project(project_id: str) -> dict[str, Any]:
    try:
        value = core.load_project(project_id)
        return {"project": value, "rows": core.project_rows(value)}
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@api.post("/api/projects/{project_id}/translate")
def translate(project_id: str, request: TranslateRequest) -> dict[str, str]:
    def work() -> dict[str, Any]:
        rows, message = core.retranslate_project(
            project_id,
            request.rows,
            request.target_language,
            request.style,
            request.provider,
            request.base_url,
            request.api_key,
            request.model,
        )
        return {"project_id": project_id, "rows": rows, "message": message}

    return {"task_id": submit_job("重新翻译目标字幕", work)}


@api.post("/api/projects/{project_id}/synthesize")
def synthesize(project_id: str, request: SynthesizeRequest) -> dict[str, str]:
    def work() -> dict[str, Any]:
        video, audio, subtitle, rows, message = core.synthesize_studio_project(
            project_id, request.rows, request.provider, request.model_path,
            request.base_url, request.api_key, request.model, request.voice,
            request.default_voice, request.speaker_mapping, request.emotion,
            request.speed, request.pitch, request.voice_volume,
            request.background_volume, request.burn_subtitles,
            request.subtitle_font, request.subtitle_size, request.subtitle_position,
        )
        return {"video": video, "audio": audio, "subtitle": subtitle, "rows": rows, "message": message}

    return {"task_id": submit_job("音频克隆与合成", work)}


@api.get("/api/tasks/{task_id}")
def task(task_id: str) -> dict[str, Any]:
    with tasks_lock:
        value = tasks.get(task_id)
    if not value:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {**value, "backend_status": core.get_status()}


@api.post("/api/tasks/cancel")
def cancel() -> dict[str, str]:
    return {"message": core.request_cancel()}


@api.get("/api/history")
def history() -> list[list[str]]:
    return core.load_history()


@api.get("/api/models")
def models() -> list[list[str]]:
    return core.model_status()


@api.get("/api/files")
def get_file(path: str):
    target = Path(path).resolve()
    allowed = [core.OUTPUTS.resolve(), core.VOICES.resolve()]
    if not target.exists() or not any(target == root or root in target.parents for root in allowed):
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target, filename=target.name, content_disposition_type="attachment")


if DIST.exists():
    api.mount("/assets", StaticFiles(directory=DIST / "assets"), name="assets")


@api.get("/{path:path}")
def frontend(path: str):
    target = DIST / path
    if path and target.is_file():
        return FileResponse(target)
    index = DIST / "index.html"
    if not index.exists():
        raise HTTPException(status_code=503, detail="前端尚未构建，请运行 npm run build。")
    return FileResponse(index)


if __name__ == "__main__":
    core.console_log("AutoVid Pro API starting")
    core.console_log(f"CUDA available: {core._cuda_available()}")
    core.console_log("Web UI: http://127.0.0.1:7860")
    core.console_log("Keep this terminal open. Closing it stops the backend.")
    uvicorn.run(api, host="127.0.0.1", port=7860, log_level="info", access_log=False)
