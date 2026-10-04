# coding: utf-8

"""TalkLikeYou real-time web demo."""

import asyncio
import json
import math
import os
import queue
import signal
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import asynccontextmanager

import cv2
import numpy as np
import librosa
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

_HERE = os.path.dirname(os.path.abspath(__file__))
from sdk_wrapper import StreamSDKStreaming  # noqa: E402
from motion_generator import RealtimeMotionGenerator  # noqa: E402
from conversation_pipeline import ConversationEngine  # noqa: E402
from avatar_registry import AvatarRegistry, file_sha256  # noqa: E402

DATA_ROOT = os.environ.get("DATA_ROOT", os.path.join(_HERE, "checkpoints", "runtime"))
CFG_PKL = os.environ.get("CFG_PKL", os.path.join(_HERE, "checkpoints", "runtime_config.pkl"))
STORAGE_DIR = os.path.abspath(
    os.environ.get(
        "TALKLIKEYOU_STORAGE_DIR",
        os.path.join(tempfile.gettempdir(), "talklikeyou_demo"),
    )
)
UPLOAD_DIR = os.path.join(STORAGE_DIR, "uploads")
AVATAR_BUNDLE_DIR = os.path.join(STORAGE_DIR, "avatar_bundles")
LIP_CKPT = os.environ.get(
    "LIP_CKPT",
    os.path.join(_HERE, "checkpoints", "motion", "lip_motion.pt"),
)
POSE_CKPT = os.environ.get(
    "POSE_CKPT",
    os.path.join(_HERE, "checkpoints", "motion", "pose_motion.pt"),
)
LIP_PERSON_ID = int(os.environ.get("LIP_PERSON_ID", "192"))
POSE_PERSON_ID = int(os.environ.get("POSE_PERSON_ID", "274"))
LIP_GUIDANCE_WEIGHT = os.environ.get("LIP_GUIDANCE_WEIGHT", "1.3")
POSE_GUIDANCE_WEIGHT = os.environ.get("POSE_GUIDANCE_WEIGHT", "1.5")
STREAM_FRAMES = int(os.environ.get("STREAM_FRAMES", "50"))
AUDIO_CHUNK_SECONDS = float(os.environ.get("AUDIO_CHUNK_SECONDS", "1.0"))
MOTION_FPS = 25.0
STREAM_RENDER_FPS = max(
    1.0,
    min(50.0, float(os.environ.get("STREAM_RENDER_FPS", "25"))),
)
SAMPLING_STEPS = int(os.environ.get("SAMPLING_STEPS", "1"))
STAGE0 = os.environ.get("STAGE0", "1").lower() in ("1", "true", "yes")
NEUTRAL = os.environ.get("NEUTRAL", "1").lower() in ("1", "true", "yes")
DITTO_POSE = os.environ.get("DITTO_POSE", "1").lower() in ("1", "true", "yes")
CONTINUOUS_IDLE = os.environ.get("CONTINUOUS_IDLE", "0").lower() in (
    "1",
    "true",
    "yes",
)
NEUTRAL_MOTION = os.environ.get(
    "NEUTRAL_MOTION",
    os.path.join(_HERE, "data", "neutral.pkl"),
)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(AVATAR_BUNDLE_DIR, exist_ok=True)

MAX_INPUT_IMAGE_SIZE = 1024

def build_runtime_models():
    print("[models] Loading SDK (TensorRT engines + models)...")
    sdk = StreamSDKStreaming(CFG_PKL, DATA_ROOT)
    if not LIP_CKPT or not POSE_CKPT:
        raise RuntimeError("LIP_CKPT and POSE_CKPT must be configured")
    motion_generator = RealtimeMotionGenerator(
        lip_ckpt=LIP_CKPT,
        pose_ckpt=POSE_CKPT,
        lip_person_id=LIP_PERSON_ID,
        pose_person_id=POSE_PERSON_ID,
        lip_guidance_weight=(
            None if LIP_GUIDANCE_WEIGHT is None else float(LIP_GUIDANCE_WEIGHT)
        ),
        pose_guidance_weight=(
            None if POSE_GUIDANCE_WEIGHT is None else float(POSE_GUIDANCE_WEIGHT)
        ),
        parallel=True,
        re_pose=False,
        stream_frames=STREAM_FRAMES,
        sampling_steps=SAMPLING_STEPS,
    )
    print("[models] SDK ready")
    return sdk, motion_generator


def runtime_models_loaded(target_app: FastAPI) -> bool:
    return bool(
        getattr(target_app.state, "sdk_template", None) is not None
        and getattr(target_app.state, "motion_generator", None) is not None
    )


def ensure_runtime_models(target_app: FastAPI) -> bool:
    """Load all persistent inference models once. Return True if reloaded."""
    with target_app.state.runtime_models_lock:
        if runtime_models_loaded(target_app):
            return False
        sdk, motion_generator = build_runtime_models()
        target_app.state.sdk_template = sdk
        target_app.state.motion_generator = motion_generator
        return True


def unload_runtime_models(target_app: FastAPI) -> bool:
    """Drop every persistent GPU model and release cached CUDA allocations."""
    import gc

    with target_app.state.runtime_models_lock:
        sdk = getattr(target_app.state, "sdk_template", None)
        motion_generator = getattr(target_app.state, "motion_generator", None)
        if sdk is None and motion_generator is None:
            return False

        target_app.state.sdk_template = None
        target_app.state.motion_generator = None

        if sdk is not None:
            try:
                sdk.avatar_registrar.close()
            except Exception:
                traceback.print_exc()
            try:
                sdk.close()
            except Exception:
                traceback.print_exc()

        if motion_generator is not None:
            audio_extractor = getattr(motion_generator, "audio_extractor", None)
            if audio_extractor is not None:
                audio_extractor.model = None
            for attribute in (
                "audio_extractor",
                "lip_model",
                "pose_model",
                "lip_stream",
                "pose_stream",
            ):
                if hasattr(motion_generator, attribute):
                    setattr(motion_generator, attribute, None)

        del sdk
        del motion_generator
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            traceback.print_exc()
        print("[models] Persistent GPU models unloaded")
        return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.runtime_models_lock = threading.Lock()
    app.state.sdk_template = None
    app.state.motion_generator = None
    app.state.conversation = ConversationEngine(UPLOAD_DIR)
    app.state.avatar_registry = AvatarRegistry(AVATAR_BUNDLE_DIR)
    app.state.avatar_registration_jobs = {}
    app.state.avatar_registration_tasks = set()
    app.state.avatar_executors = {}
    app.state.avatar_executors_lock = threading.Lock()
    app.state.sdk_lock = threading.Lock()
    app.state.shutdown_requested = False
    ensure_runtime_models(app)
    print("[startup] SDK ready")
    yield
    print("[shutdown] Cleaning up...")
    registration_tasks = list(app.state.avatar_registration_tasks)
    if registration_tasks:
        await asyncio.gather(*registration_tasks, return_exceptions=True)
    with app.state.avatar_executors_lock:
        avatar_executors = list(app.state.avatar_executors.values())
        app.state.avatar_executors.clear()
    for avatar_executor in avatar_executors:
        avatar_executor.shutdown(wait=True)
    unload_runtime_models(app)

app = FastAPI(title="TalkLikeYou Real-Time Demo", lifespan=lifespan)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")
STATIC_DIR = os.path.join(_HERE, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR, html=True), name="static")


SERVER_START_TIME = time.time()


def avatar_executor_for(avatar_id: str) -> ThreadPoolExecutor:
    """Return the avatar's dedicated single worker.

    Jobs for the same avatar are serialized on one thread, while different
    avatars run concurrently on independent threads.
    """
    with app.state.avatar_executors_lock:
        executor = app.state.avatar_executors.get(avatar_id)
        if executor is None:
            thread_name = f"avatar-{avatar_id[:24]}"
            executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=thread_name,
            )
            app.state.avatar_executors[avatar_id] = executor
        return executor

@app.get("/api/status")
def api_status():
    conversation = getattr(app.state, "conversation", None)
    return {
        "start_time": SERVER_START_TIME,
        "models_loaded": runtime_models_loaded(app),
        "shutdown_requested": bool(
            getattr(app.state, "shutdown_requested", False)
        ),
        "conversation": conversation.public_status() if conversation else {},
    }


def avatar_jobs_are_active() -> bool:
    return any(
        job["status"] in {"queued", "running"}
        for job in app.state.avatar_registration_jobs.values()
    )


@app.post("/api/system/unload")
async def unload_models():
    if avatar_jobs_are_active():
        raise HTTPException(
            status_code=409,
            detail="Wait for avatar registration/neutralization jobs to finish",
        )
    lock = app.state.sdk_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Stop the active inference stream before releasing GPU memory",
        )
    try:
        loop = asyncio.get_running_loop()
        released = await loop.run_in_executor(
            None, lambda: unload_runtime_models(app)
        )
        return {
            "status": "unloaded",
            "released": released,
            "models_loaded": runtime_models_loaded(app),
            "message": (
                "GPU models unloaded; the next inference will reload them"
                if released
                else "GPU models were already unloaded"
            ),
        }
    finally:
        lock.release()


@app.post("/api/system/shutdown")
async def shutdown_service():
    if app.state.shutdown_requested:
        return {"status": "shutting_down"}
    app.state.shutdown_requested = True
    loop = asyncio.get_running_loop()
    loop.call_later(0.5, os.kill, os.getpid(), signal.SIGTERM)
    return {"status": "shutting_down"}


@app.get("/")
def root():
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(
            index,
            headers={"Cache-Control": "no-store, max-age=0"},
        )
    return {"status": "ok"}



def _video_stream_orientation(path: str) -> tuple[int, int, float]:
    """Return coded width, coded height, and container display rotation."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0, 0, 0.0
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                (
                    "stream=width,height:"
                    "stream_tags=rotate:"
                    "stream_side_data=rotation"
                ),
                "-of",
                "json",
                path,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        streams = json.loads(result.stdout).get("streams") or []
        if not streams:
            return 0, 0, 0.0
        stream = streams[0]
        rotation = float((stream.get("tags") or {}).get("rotate", 0) or 0)
        for side_data in stream.get("side_data_list") or []:
            if "rotation" in side_data:
                rotation = float(side_data["rotation"] or 0)
                break
        return (
            int(stream.get("width") or 0),
            int(stream.get("height") or 0),
            rotation,
        )
    except Exception:
        traceback.print_exc()
        return 0, 0, 0.0


def _strip_conflicting_landscape_rotation(path: str) -> bool:
    """Keep coded landscape video landscape when stale metadata says ±90°."""
    width, height, rotation = _video_stream_orientation(path)
    quarter_turn = abs(round(rotation)) % 180 == 90
    if width <= height or not quarter_turn:
        return False
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return False
    suffix = os.path.splitext(path)[1] or ".mp4"
    normalized_path = f"{path}.orientation{suffix}"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-display_rotation:v:0",
                "0",
                "-i",
                path,
                "-map",
                "0",
                "-c",
                "copy",
                "-metadata:s:v:0",
                "rotate=0",
                normalized_path,
            ],
            check=True,
            capture_output=True,
        )
        os.replace(normalized_path, path)
        print(
            "[upload] Removed conflicting video rotation: "
            f"{width}x{height}, rotation={rotation:g}°"
        )
        return True
    except Exception:
        traceback.print_exc()
        return False
    finally:
        if os.path.exists(normalized_path):
            os.unlink(normalized_path)


@app.post("/api/upload/source")
async def upload_source(
    file: UploadFile = File(...),
    max_image_size: int = Form(MAX_INPUT_IMAGE_SIZE),
    face_idx: int = Form(0),
    crop_scale: float = Form(2.3),
):
    max_image_size = max(64, min(4096, int(max_image_size)))
    face_idx, crop_scale = _validate_avatar_crop_settings(
        face_idx,
        crop_scale,
    )
    ext = os.path.splitext(file.filename or "image.png")[1] or ".png"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    content = await file.read()

    if ext.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        encoded = np.frombuffer(content, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError(f"Unable to decode image: {file.filename}")
        height, width = image.shape[:2]
        longest_edge = max(height, width)
        if longest_edge > max_image_size:
            scale = max_image_size / float(longest_edge)
            image = cv2.resize(
                image,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
            encode_ext = ".jpg" if ext.lower() == ".jpeg" else ext.lower()
            encode_args = (
                [cv2.IMWRITE_JPEG_QUALITY, 95]
                if encode_ext in (".jpg", ".jpeg")
                else []
            )
            ok, resized = cv2.imencode(encode_ext, image, encode_args)
            if not ok:
                raise ValueError(f"Unable to encode resized image: {file.filename}")
            content = resized.tobytes()

    with open(path, "wb") as f:
        f.write(content)
    is_video = ext.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
    orientation_normalized = (
        await asyncio.to_thread(
            _strip_conflicting_landscape_rotation,
            path,
        )
        if is_video
        else False
    )
    avatar_id = file_sha256(path)[:24]
    registry = app.state.avatar_registry
    metadata = None
    if (
        registry.is_registered(avatar_id)
        and registry.registration_matches(
            avatar_id,
            face_idx=face_idx,
            crop_scale=crop_scale,
        )
    ):
        metadata = registry.public_metadata(registry.load(avatar_id))
    return {
        "name": file.filename,
        "path": path,
        "url": f"/uploads/{name}",
        "cache_key": avatar_id,
        "avatar_id": avatar_id,
        "is_video": is_video,
        "orientation_normalized": orientation_normalized,
        "registered": metadata is not None,
        "neutralized": bool(metadata and metadata["neutralized"]),
        "face_idx": face_idx,
        "crop_scale": crop_scale,
    }


@app.post("/api/upload/audio")
async def upload_audio(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(UPLOAD_DIR, name)
    with open(path, "wb") as f:
        f.write(await file.read())
    return {"name": file.filename, "path": path, "url": f"/uploads/{name}"}



class AvatarRegistrationRequest(BaseModel):
    avatar_id: str
    source_path: str
    source_name: str = ""
    prepare_per_frame_normalize_delta: bool = False
    face_idx: int = 0
    crop_scale: float = 2.3
    force: bool = False


class AvatarNeutralizationRequest(BaseModel):
    avatar_id: str
    force: bool = False


def _validate_avatar_crop_settings(
    face_idx: int,
    crop_scale: float,
) -> tuple[int, float]:
    face_idx = int(face_idx)
    crop_scale = float(crop_scale)
    if face_idx < 0:
        raise HTTPException(
            status_code=400,
            detail="face_idx must be greater than or equal to 0",
        )
    if not math.isfinite(crop_scale) or crop_scale <= 0:
        raise HTTPException(
            status_code=400,
            detail="crop_scale must be a finite number greater than 0",
        )
    return face_idx, crop_scale


def _require_uploaded_file(path: str) -> str:
    resolved = os.path.realpath(path)
    upload_root = os.path.realpath(UPLOAD_DIR)
    if os.path.commonpath([resolved, upload_root]) != upload_root:
        raise HTTPException(status_code=400, detail="Source must be an uploaded file")
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="Uploaded source not found")
    return resolved


def _resolve_neutral_motion() -> str:
    resolved = os.path.realpath(NEUTRAL_MOTION)
    if not os.path.isfile(resolved):
        raise HTTPException(status_code=404, detail="Neutral motion not found")
    return resolved


def create_avatar_preprocess_sdk():
    """Create a task-local registrar so avatar jobs can safely run in parallel."""
    from types import SimpleNamespace
    from runtime.core.atomic_components.avatar_registrar import AvatarRegistrar
    from runtime.core.atomic_components.cfg import parse_cfg

    parsed = parse_cfg(CFG_PKL, DATA_ROOT)
    avatar_registrar_cfg = parsed[0]
    default_kwargs = parsed[7]
    return SimpleNamespace(
        avatar_registrar=AvatarRegistrar(**avatar_registrar_cfg),
        default_kwargs=default_kwargs,
    )


@app.get("/api/avatar/registrations")
async def list_avatar_registrations():
    registry: AvatarRegistry = app.state.avatar_registry
    return {"avatars": registry.list_registered()}


@app.delete("/api/avatar/registrations")
async def clear_avatar_registrations():
    if any(
        job["status"] in {"queued", "running"}
        for job in app.state.avatar_registration_jobs.values()
    ):
        raise HTTPException(
            status_code=409,
            detail="Wait for avatar registration/neutralization jobs to finish",
        )
    lock = app.state.sdk_lock
    if not lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="Avatar cache is in use; stop inference or wait for registration",
        )

    try:
        for job in app.state.avatar_registration_jobs.values():
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["message"] = "Cancelled because the avatar cache was cleared"

        registry: AvatarRegistry = app.state.avatar_registry
        removed = registry.clear()
        sdk_template = app.state.sdk_template
        if sdk_template is not None:
            sdk_template.source_registry.clear()
        return {"status": "cleared", "removed": removed}
    finally:
        lock.release()


@app.post("/api/avatar/register")
async def register_avatar(body: AvatarRegistrationRequest):
    registry: AvatarRegistry = app.state.avatar_registry
    try:
        avatar_id = registry.normalize_id(body.avatar_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    face_idx, crop_scale = _validate_avatar_crop_settings(
        body.face_idx,
        body.crop_scale,
    )
    source_path = _require_uploaded_file(body.source_path)
    source_is_video = os.path.splitext(source_path)[1].lower() in {
        ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"
    }
    if file_sha256(source_path)[:24] != avatar_id:
        raise HTTPException(
            status_code=400,
            detail="Avatar ID does not match the uploaded source",
        )
    neutral_motion = _resolve_neutral_motion()
    neutral_sha256 = file_sha256(neutral_motion)
    if (
        registry.is_registered(avatar_id)
        and registry.registration_matches(
            avatar_id,
            face_idx=face_idx,
            crop_scale=crop_scale,
        )
        and not body.force
        and (
            (
                source_is_video
                and (
                    not body.prepare_per_frame_normalize_delta
                    or registry.is_stage0_per_frame_normalize_ready(avatar_id)
                )
            )
            or (
                not source_is_video
                and registry.is_neutralized(avatar_id, neutral_sha256)
            )
        )
    ):
        current = registry.load(avatar_id)
        return {
            "job_id": None, "avatar_id": avatar_id, "kind": "registration",
            "status": "completed", "progress": 100, "message": "Registered",
            "avatar": registry.public_metadata(current),
        }

    for existing_job in app.state.avatar_registration_jobs.values():
        if existing_job["avatar_id"] != avatar_id:
            continue
        if (
            existing_job.get("kind") == "registration"
            and existing_job["status"] in {"queued", "running"}
            and existing_job.get("face_idx", 0) == face_idx
            and math.isclose(
                float(existing_job.get("crop_scale", 2.3)),
                crop_scale,
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        ):
            return existing_job
        if existing_job["status"] == "queued":
            existing_job["status"] = "cancelled"
            existing_job["message"] = "Superseded by a newer registration"

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id,
        "avatar_id": avatar_id,
        "kind": "registration",
        "face_idx": face_idx,
        "crop_scale": crop_scale,
        "status": "queued",
        "progress": 0,
        "message": "Waiting for GPU",
        "error": None,
    }
    app.state.avatar_registration_jobs[job_id] = job
    avatar_executor = avatar_executor_for(avatar_id)

    async def run_registration():
        loop = asyncio.get_running_loop()

        def update(progress, message):
            job["progress"] = int(progress)
            job["message"] = str(message)
            job["status"] = "completed" if int(progress) >= 100 else "running"

        def process_registration():
            if job["status"] == "cancelled":
                return None
            preprocess_sdk = create_avatar_preprocess_sdk()
            try:
                job["status"] = "running"
                job["progress"] = 2
                job["message"] = "Preparing registration"
                registration_progress = (
                    update
                    if source_is_video
                    else lambda progress, message: update(
                        int(progress * 0.4), message
                    )
                )
                metadata = registry.register(
                    avatar_id=avatar_id,
                    source_path=source_path,
                    source_name=body.source_name,
                    neutral_motion=neutral_motion,
                    prepare_per_frame_normalize_delta=(
                        body.prepare_per_frame_normalize_delta
                    ),
                    face_idx=face_idx,
                    crop_scale=crop_scale,
                    sdk=preprocess_sdk,
                    hybrid=load_hybrid_helpers() if source_is_video else None,
                    progress=registration_progress,
                )
                if not source_is_video:
                    metadata = registry.neutralize(
                        avatar_id=avatar_id,
                        neutral_motion=neutral_motion,
                        sdk=preprocess_sdk,
                        hybrid=load_hybrid_helpers(),
                        progress=lambda progress, message: update(
                            40 + int(progress * 0.6), message
                        ),
                    )
                return metadata
            finally:
                preprocess_sdk.avatar_registrar.close()

        try:
            metadata = await loop.run_in_executor(
                avatar_executor, process_registration
            )
            if metadata is None:
                return
            job["avatar"] = metadata
            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "Registered"
        except asyncio.CancelledError:
            job["status"] = "cancelled"
            job["message"] = "Cancelled"
            raise
        except Exception as exc:
            traceback.print_exc()
            job["status"] = "failed"
            job["message"] = "Registration failed"
            job["error"] = str(exc)

    task = asyncio.create_task(run_registration())
    app.state.avatar_registration_tasks.add(task)
    task.add_done_callback(app.state.avatar_registration_tasks.discard)
    return job


@app.post("/api/avatar/neutralize")
async def neutralize_avatar(body: AvatarNeutralizationRequest):
    registry: AvatarRegistry = app.state.avatar_registry
    try:
        avatar_id = registry.normalize_id(body.avatar_id)
        registry.load(avatar_id)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    neutral_motion = _resolve_neutral_motion()
    neutral_sha256 = file_sha256(neutral_motion)
    if registry.is_neutralized(avatar_id, neutral_sha256) and not body.force:
        current = registry.load(avatar_id)
        return {
            "job_id": None, "avatar_id": avatar_id, "kind": "neutralization",
            "status": "completed", "progress": 100, "message": "Neutralized",
            "avatar": registry.public_metadata(current),
        }
    for existing_job in app.state.avatar_registration_jobs.values():
        if (
            existing_job["avatar_id"] == avatar_id
            and existing_job.get("kind") == "neutralization"
            and existing_job.get("neutral_motion_sha256") == neutral_sha256
            and existing_job["status"] in {"queued", "running"}
        ):
            return existing_job

    job_id = uuid.uuid4().hex
    job = {
        "job_id": job_id, "avatar_id": avatar_id, "kind": "neutralization",
        "neutral_motion_sha256": neutral_sha256, "status": "queued",
        "progress": 0, "message": "Waiting for GPU", "error": None,
    }
    app.state.avatar_registration_jobs[job_id] = job
    avatar_executor = avatar_executor_for(avatar_id)

    async def run_neutralization():
        loop = asyncio.get_running_loop()

        def update(progress, message):
            job["progress"] = int(progress)
            job["message"] = str(message)
            job["status"] = "completed" if int(progress) >= 100 else "running"

        def process_neutralization():
            if job["status"] == "cancelled":
                return None
            preprocess_sdk = create_avatar_preprocess_sdk()
            try:
                job.update(
                    status="running", progress=2,
                    message="Preparing neutralization",
                )
                return registry.neutralize(
                    avatar_id=avatar_id, neutral_motion=neutral_motion,
                    sdk=preprocess_sdk, hybrid=load_hybrid_helpers(),
                    progress=update,
                )
            finally:
                preprocess_sdk.avatar_registrar.close()

        try:
            metadata = await loop.run_in_executor(
                avatar_executor, process_neutralization
            )
            if metadata is None:
                return
            job.update(
                avatar=metadata, status="completed", progress=100,
                message="Neutralized",
            )
        except asyncio.CancelledError:
            job.update(status="cancelled", message="Cancelled")
            raise
        except Exception as exc:
            traceback.print_exc()
            job.update(
                status="failed", message="Neutralization failed", error=str(exc)
            )

    task = asyncio.create_task(run_neutralization())
    app.state.avatar_registration_tasks.add(task)
    task.add_done_callback(app.state.avatar_registration_tasks.discard)
    return job


@app.get("/api/avatar/register/{job_id}")
async def avatar_registration_status(job_id: str):
    job = app.state.avatar_registration_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Registration job not found")
    return job



class ConversationRequest(BaseModel):
    conversation_id: str
    message: str
    system_prompt: str | None = None
    model: str | None = None
    tts_provider: str | None = None
    tts_model: str | None = None
    voice: str | None = None
    streaming_tts: bool = True


class ResetConversationRequest(BaseModel):
    conversation_id: str


@app.post("/api/conversation/stream")
async def conversation_stream(body: ConversationRequest):
    engine: ConversationEngine = app.state.conversation

    async def events():
        try:
            async for event in engine.stream_turn(
                conversation_id=body.conversation_id,
                user_text=body.message,
                system_prompt=body.system_prompt,
                model=body.model,
                tts_provider=body.tts_provider,
                tts_model=body.tts_model,
                voice=body.voice,
                streaming_tts=body.streaming_tts,
            ):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield json.dumps(
                {"type": "error", "msg": str(exc)}, ensure_ascii=False
            ) + "\n"

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/conversation/transcribe")
async def conversation_transcribe(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty recording")
    if len(content) > 20 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Recording is larger than 20 MB")

    ext = os.path.splitext(file.filename or "recording.webm")[1] or ".webm"
    path = os.path.join(UPLOAD_DIR, f"recording_{uuid.uuid4().hex}{ext}")
    with open(path, "wb") as output:
        output.write(content)
    try:
        text = await app.state.conversation.transcribe(path)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
    return {"text": text}


@app.post("/api/conversation/reset")
async def conversation_reset(body: ResetConversationRequest):
    app.state.conversation.clear(body.conversation_id)
    return {"status": "cleared"}



def load_hybrid_helpers():
    """Load portrait preprocessing helpers on demand."""
    import portrait_helpers

    return portrait_helpers


def source_info_with_constant_exp(source_info, expression):
    """Copy source motion metadata while replacing every frame expression."""
    constant_exp = np.asarray(expression, dtype=np.float32).reshape(-1)
    copied = dict(source_info)
    copied_frames = []
    for source_frame in source_info["x_s_info_lst"]:
        frame = dict(source_frame)
        source_exp = np.asarray(source_frame["exp"])
        frame["exp"] = constant_exp.reshape(source_exp.shape).copy()
        copied_frames.append(frame)
    copied["x_s_info_lst"] = copied_frames
    return copied



@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    sdk = None
    conversation_audio_task = None
    frame_sender_task = None
    speed_reporter_task = None
    switch_task = None
    cancel_event = threading.Event()
    websocket_send_lock = asyncio.Lock()

    def stop_disconnected_stream():
        cancel_event.set()
        if sdk is not None and not getattr(sdk, "_closed", True):
            sdk.abort_current()

    def is_closed_websocket_error(exc):
        message = str(exc)
        return isinstance(exc, RuntimeError) and (
            "websocket.send" in message
            or "websocket.close" in message
            or "response already completed" in message
        )

    async def safe_send_json(payload):
        if cancel_event.is_set():
            raise WebSocketDisconnect(code=1000)
        async with websocket_send_lock:
            if cancel_event.is_set():
                raise WebSocketDisconnect(code=1000)
            try:
                await ws.send_json(payload)
            except WebSocketDisconnect:
                stop_disconnected_stream()
                raise
            except RuntimeError as exc:
                if not is_closed_websocket_error(exc):
                    raise
                stop_disconnected_stream()
                raise WebSocketDisconnect(code=1000) from exc

    async def safe_send_bytes(payload):
        if cancel_event.is_set():
            raise WebSocketDisconnect(code=1000)
        async with websocket_send_lock:
            if cancel_event.is_set():
                raise WebSocketDisconnect(code=1000)
            try:
                await ws.send_bytes(payload)
            except WebSocketDisconnect:
                stop_disconnected_stream()
                raise
            except RuntimeError as exc:
                if not is_closed_websocket_error(exc):
                    raise
                stop_disconnected_stream()
                raise WebSocketDisconnect(code=1000) from exc

    async def safe_send_frame(frame_seq, jpg_bytes):
        if cancel_event.is_set():
            raise WebSocketDisconnect(code=1000)
        async with websocket_send_lock:
            if cancel_event.is_set():
                raise WebSocketDisconnect(code=1000)
            try:
                await ws.send_json({"t": "f", "i": frame_seq})
                await ws.send_bytes(jpg_bytes)
            except WebSocketDisconnect:
                stop_disconnected_stream()
                raise
            except RuntimeError as exc:
                if not is_closed_websocket_error(exc):
                    raise
                stop_disconnected_stream()
                raise WebSocketDisconnect(code=1000) from exc

    async def safe_close():
        async with websocket_send_lock:
            try:
                await ws.close()
            except (WebSocketDisconnect, RuntimeError):
                pass

    lock = app.state.sdk_lock
    acquired = await loop.run_in_executor(None, lambda: lock.acquire(timeout=10))
    if not acquired:
        try:
            await safe_send_json({
                "type": "error",
                "msg": "Server busy — try again later.",
            })
        except WebSocketDisconnect:
            pass
        await safe_close()
        return

    try:
        if not runtime_models_loaded(app):
            await safe_send_json({
                "type": "status",
                "msg": "Reloading GPU models...",
            })
        await loop.run_in_executor(None, lambda: ensure_runtime_models(app))
        sdk = app.state.sdk_template
        motion_generator = app.state.motion_generator

        raw = await ws.receive_text()
        cfg = json.loads(raw)
        audio_path = cfg.get("audio")
        microphone = bool(cfg.get("microphone", False))
        conversation_stream_id = str(
            cfg.get("conversation_stream_id") or ""
        ).strip()
        if len(conversation_stream_id) > 96:
            await safe_send_json({
                "type": "error",
                "msg": "Invalid conversation audio stream ID",
            })
            return
        if microphone and conversation_stream_id:
            await safe_send_json({
                "type": "error",
                "msg": "Choose either microphone or conversation audio",
            })
            return
        streaming_audio = bool(microphone or conversation_stream_id)
        active_avatar_id = cfg.get("avatar_id")
        registered_avatar_ids = cfg.get("registered_avatar_ids", [])
        stage0 = bool(cfg.get("stage0", STAGE0))
        neutral = bool(cfg.get("neutral", NEUTRAL))
        per_frame_normalize_delta_requested = bool(
            cfg.get("per_frame_normalize_delta", False)
        )
        ditto_pose = bool(cfg.get("ditto_pose", DITTO_POSE))
        paste_back = bool(cfg.get("paste_back", True))
        video_dubbing_requested = bool(cfg.get("video_dubbing", False))
        continuous_idle = bool(cfg.get("continuous_idle", CONTINUOUS_IDLE))
        neutral_motion = _resolve_neutral_motion()
        audio_chunk_seconds = max(
            0.1, float(cfg.get("audio_chunk_seconds", AUDIO_CHUNK_SECONDS))
        )
        requested_render_fps = max(
            1.0,
            min(50.0, float(cfg.get("render_fps", STREAM_RENDER_FPS))),
        )
        motion_buffer_frames = max(
            1,
            int(cfg.get("stream_frames", STREAM_FRAMES)),
        )
        sampling_steps = max(
            1,
            int(cfg.get("sampling_steps", SAMPLING_STEPS)),
        )
        motion_generator.configure(
            lip_person_id=cfg.get("lip_person_id", LIP_PERSON_ID),
            pose_person_id=cfg.get("pose_person_id", POSE_PERSON_ID),
            lip_guidance_weight=cfg.get("lip_guidance_weight", LIP_GUIDANCE_WEIGHT),
            pose_guidance_weight=cfg.get("pose_guidance_weight", POSE_GUIDANCE_WEIGHT),
            sampling_steps=sampling_steps,
            stream_frames=motion_buffer_frames,
        )

        if not active_avatar_id:
            await safe_send_json({
                "type": "error",
                "msg": "Missing registered avatar ID",
            })
            return
        if audio_path and not os.path.exists(audio_path):
            await safe_send_json({
                "type": "error",
                "msg": f"Audio not found: {audio_path}",
            })
            return

        registry: AvatarRegistry = app.state.avatar_registry
        try:
            active_bundle = registry.load(active_avatar_id)
        except Exception as exc:
            await safe_send_json({
                "type": "error",
                "msg": f"Register the active avatar first: {exc}",
            })
            return
        bundle_is_video = bool(
            active_bundle.get(
                "is_video",
                not active_bundle["source_info"].get("is_image_flag", True),
            )
        )
        if video_dubbing_requested and not bundle_is_video:
            await safe_send_json({
                "type": "error",
                "msg": "Video dubbing requires a registered video source",
            })
            return
        video_dubbing = video_dubbing_requested or bundle_is_video
        if streaming_audio and video_dubbing:
            await safe_send_json({
                "type": "error",
                "msg": "Live audio streaming is available for image avatars only",
            })
            return
        if streaming_audio and audio_path:
            await safe_send_json({
                "type": "error",
                "msg": "Choose either live audio streaming or an audio file",
            })
            return
        if streaming_audio:
            # Live streams are finite only when their producer sends the end
            # marker and never append idle motion. Pose may come from either
            # the learned pose model or Ditto pose generation.
            continuous_idle = False
        # Only video dubbing treats Neutral as a relative, normalized-lip
        # mode. Image-based Audio/Conversation keeps the original behavior:
        # Stage0 alone controls relative_d, absolute_lip and normalization.
        relative_lip_mode = stage0 or (video_dubbing and neutral)
        normalize_lip = stage0 or (video_dubbing and neutral)
        absolute_lip = not relative_lip_mode
        video_s0_lip = video_dubbing and stage0 and not neutral
        per_frame_normalize_delta = (
            video_s0_lip and per_frame_normalize_delta_requested
        )
        video_neutral = video_dubbing and neutral
        requested_neutral_sha256 = file_sha256(neutral_motion)
        # Image/Conversation registration prepares neutral assets
        # automatically. Only Video dubbing exposes and requires a separate
        # Neutralize action.
        needs_neutral_assets = video_dubbing and neutral
        if (
            not video_dubbing
            and (stage0 or neutral)
            and not registry.is_neutralized(
                active_avatar_id,
                requested_neutral_sha256,
            )
        ):
            await safe_send_json({
                "type": "error",
                "msg": (
                    "Register this avatar again to prepare automatic "
                    "neutral assets"
                ),
            })
            return
        if (
            video_dubbing
            and stage0
            and not registry.is_stage0_ready(active_avatar_id)
        ):
            await safe_send_json({
                "type": "error",
                "msg": "Register this video again to prepare Stage0",
            })
            return
        if (
            per_frame_normalize_delta
            and not registry.is_stage0_per_frame_normalize_ready(active_avatar_id)
        ):
            await safe_send_json({
                "type": "error",
                "msg": (
                    "Register this video again to prepare per-frame "
                    "normalize delta"
                ),
            })
            return
        if needs_neutral_assets and not registry.is_neutralized(
            active_avatar_id, requested_neutral_sha256
        ):
            await safe_send_json({
                "type": "error",
                "msg": "Run Neutralize for this avatar first",
            })
            return
        if video_dubbing:
            ditto_pose = False
            # Video dubbing is always finite. Stage0/Neutral still controls
            # relative lip composition and normalization, but must not append
            # idle/static frames after the input audio ends.
            continuous_idle = False
        else:
            video_s0_lip = False
            video_neutral = False
        if (
            not audio_path
            and not streaming_audio
            and not continuous_idle
            and not video_dubbing
        ):
            await safe_send_json({
                "type": "error",
                "msg": "Missing audio; choose a microphone or enable Continuous idle",
            })
            return

        sdk.reset_for_reuse()
        sdk.set_ditto_pose_mode(ditto_pose)
        sdk.source_video_loop = video_dubbing
        sdk.pace_output = video_dubbing
        sdk.motion_input_fps = MOTION_FPS
        sdk.output_fps = requested_render_fps
        sdk.render_fps = requested_render_fps
        output_motion_buffer_frames = max(
            1,
            math.ceil(
                motion_buffer_frames
                * sdk.output_fps
                / sdk.motion_input_fps
            ),
        )
        sdk.paste_back = paste_back
        sdk.configure_motion_buffer(output_motion_buffer_frames)
        cache_key = cfg.get("cache_key") or active_avatar_id
        run_name = f"{cache_key}_{int(time.time())}"
        output_path = os.path.join(STORAGE_DIR, f"{run_name}.mp4")

        await safe_send_json({
            "type": "status",
            "msg": "Loading registered avatar...",
        })

        if video_neutral:
            render_source_info = source_info_with_constant_exp(
                active_bundle["neutral_source_info"],
                active_bundle["stage0_shell_motion"]["exp"],
            )
        else:
            render_source_info = (
                active_bundle["neutral_source_info"]
                if neutral
                else active_bundle["source_info"]
            )
        stage0_shell_motion = (
            active_bundle["stage0_shell_motion"] if relative_lip_mode else None
        )
        active_lip_delta = (
            active_bundle["neutral_lip_normalize_delta_kp"]
            if neutral
            else active_bundle["lip_normalize_delta_kp"]
        )
        active_lip_delta_sequence = (
            active_bundle["lip_normalize_delta_kp_sequence"]
            if per_frame_normalize_delta
            else None
        )

        sdk_setup_kwargs = {
            "relative_d": relative_lip_mode,
            "drive_eye": not video_dubbing,
            "online_mode": True,
            "sampling_timesteps": sampling_steps,
        }
        if not relative_lip_mode:
            sdk_setup_kwargs["overall_ctrl_info"] = {}
        sdk.setup_cached(render_source_info, output_path, **sdk_setup_kwargs)
        # Stage0 uses first-frame-relative motion; the regular path consumes
        # source-aligned absolute pose directly.
        sdk.relative_d = relative_lip_mode
        sdk.motion_stitch.relative_d = relative_lip_mode
        if not relative_lip_mode:
            sdk.overall_ctrl_info = {}
            sdk.motion_stitch.overall_ctrl_info = {}
        sdk.online_mode = True
        sdk.register_source_context(
            active_avatar_id,
            render_source_info,
            active_lip_delta if normalize_lip else None,
            active_lip_delta_sequence,
        )
        sdk.active_source_name = active_avatar_id
        sdk.active_lip_normalize_delta_kp = (
            active_lip_delta if normalize_lip else None
        )
        sdk.active_lip_normalize_delta_kp_sequence = (
            active_lip_delta_sequence
        )

        # Preserve absolute lip output after expression and gaze controls.
        hybrid_base = load_hybrid_helpers()
        hybrid_base.patch_motion_stitch_for_absolute_lips(
            sdk,
            lip_normalize_delta_kp=active_lip_delta if normalize_lip else None,
            absolute_lip=absolute_lip,
            video_s0_lip=video_s0_lip,
        )

        motion_source_info = sdk.source_info
        # Video dubbing conditions generated lips on the neutral motion shell.
        # Stage0 and Neutral independently control relative lip composition.
        lip_condition_motion = (
            active_bundle["stage0_shell_motion"]
            if video_dubbing
            else stage0_shell_motion
        )

        for avatar_id in registered_avatar_ids:
            if avatar_id == active_avatar_id:
                continue
            try:
                bundle = registry.load(avatar_id)
            except Exception:
                continue
            other_is_video = bool(
                bundle.get(
                    "is_video",
                    not bundle["source_info"].get("is_image_flag", True),
                )
            )
            if other_is_video != bundle_is_video:
                continue
            if (
                other_is_video
                and stage0
                and not registry.is_stage0_ready(avatar_id)
            ):
                continue
            if (
                per_frame_normalize_delta
                and not registry.is_stage0_per_frame_normalize_ready(avatar_id)
            ):
                continue
            if (
                not other_is_video
                and (stage0 or neutral)
                and not registry.is_neutralized(
                    avatar_id,
                    requested_neutral_sha256,
                )
            ):
                continue
            if needs_neutral_assets and not registry.is_neutralized(
                avatar_id, requested_neutral_sha256
            ):
                continue
            if video_neutral:
                source_info = source_info_with_constant_exp(
                    bundle["neutral_source_info"],
                    bundle["stage0_shell_motion"]["exp"],
                )
                lip_delta = bundle["neutral_lip_normalize_delta_kp"]
            else:
                source_info = (
                    bundle["neutral_source_info"]
                    if neutral
                    else bundle["source_info"]
                )
                lip_delta = (
                    bundle["neutral_lip_normalize_delta_kp"]
                    if neutral
                    else bundle["lip_normalize_delta_kp"]
                )
            sdk.register_source_context(
                avatar_id,
                source_info,
                lip_delta if normalize_lip else None,
                (
                    bundle["lip_normalize_delta_kp_sequence"]
                    if per_frame_normalize_delta
                    else None
                ),
            )

        frame_send_stats = {"frames": 0}

        async def send_encoded_frames():
            """Drain encoded frames without blocking the rendering worker."""
            sent_frames = 0
            send_seconds = 0.0
            pace_started_at = None
            pace_first_frame = None
            try:
                while True:
                    try:
                        frame_seq, jpg_bytes = await asyncio.to_thread(
                            sdk.get_encoded_frame,
                            0.2,
                        )
                    except queue.Empty:
                        if sdk.frame_stream_drained():
                            break
                        continue

                    # Video dubbing retains real-time delivery at the selected
                    # output FPS, but pacing happens here rather than occupying
                    # the putback/JPEG worker.
                    if sdk.pace_output:
                        if pace_started_at is None:
                            pace_started_at = loop.time()
                            pace_first_frame = frame_seq
                        deadline = (
                            pace_started_at
                            + (frame_seq - pace_first_frame) / sdk.output_fps
                        )
                        delay = deadline - loop.time()
                        if delay > 0:
                            await asyncio.sleep(delay)

                    send_started = time.perf_counter()
                    # Keep the timestamp header adjacent to its JPEG payload.
                    await safe_send_frame(frame_seq, jpg_bytes)
                    send_seconds += time.perf_counter() - send_started
                    sent_frames += 1
                    frame_send_stats["frames"] = sent_frames
            except asyncio.CancelledError:
                raise
            except WebSocketDisconnect:
                stop_disconnected_stream()
            except Exception:
                cancel_event.set()
                sdk.abort_current()
                raise

            return {
                "frames": sent_frames,
                "send_ms": (
                    send_seconds * 1000.0 / sent_frames
                    if sent_frames
                    else 0.0
                ),
            }

        frame_sender_task = asyncio.create_task(send_encoded_frames())

        async def report_stream_speed():
            """Print one-second generation and delivery rates to the terminal."""
            last_at = loop.time()
            last_generated = sdk.encoded_frame_count
            last_sent = frame_send_stats["frames"]
            try:
                while True:
                    await asyncio.sleep(1.0)
                    now = loop.time()
                    elapsed = max(now - last_at, 1e-6)
                    generated = sdk.encoded_frame_count
                    sent = frame_send_stats["frames"]
                    generated_fps = (generated - last_generated) / elapsed
                    sent_fps = (sent - last_sent) / elapsed
                    print(
                        "[stream-speed] "
                        f"generated={generated_fps:.1f} fps "
                        f"sent={sent_fps:.1f} fps "
                        f"target={sdk.output_fps:.1f} fps "
                        f"queue={sdk.frame_queue.qsize()}/"
                        f"{sdk.frame_queue.maxsize} "
                        f"dropped={sdk.frame_queue_dropped}",
                        flush=True,
                    )
                    last_at = now
                    last_generated = generated
                    last_sent = sent
            except asyncio.CancelledError:
                return

        speed_reporter_task = asyncio.create_task(report_stream_speed())

        if audio_path:
            audio, _ = librosa.load(audio_path, sr=16000)
            audio_url = f"/uploads/{os.path.basename(audio_path)}"
        else:
            audio = np.zeros(0, dtype=np.float32)
            audio_url = None
        if streaming_audio:
            total_frames = 0
        elif video_dubbing and not audio_path and not continuous_idle:
            total_frames = math.ceil(
                len(motion_source_info["x_s_info_lst"])
                * sdk.output_fps
                / sdk.motion_input_fps
            )
        else:
            if ditto_pose:
                base_total_frames = math.ceil(
                    len(audio) / 16000 * sdk.motion_input_fps
                )
            else:
                base_total_frames = (
                    motion_generator.audio_extractor.expected_frames(
                        len(audio)
                    )
                )
            total_frames = math.ceil(
                base_total_frames
                * sdk.output_fps
                / sdk.motion_input_fps
            )
        sdk.setup_Nd(
            N_d=-1 if (continuous_idle or streaming_audio) else total_frames,
            fade_in=-1,
            fade_out=-1,
            ctrl_info={},
        )

        await safe_send_json({
            "type": "ready",
            "total_frames": total_frames,
            "fps": sdk.output_fps,
            "render_fps": sdk.render_fps,
            "motion_buffer_frames": output_motion_buffer_frames,
            "motion_input_buffer_frames": motion_buffer_frames,
            "audio_url": audio_url,
            "continuous_idle": continuous_idle,
            "microphone": microphone,
            "conversation_audio": bool(conversation_stream_id),
        })

        microphone_audio_queue = (
            queue.Queue(
                # Browser packets contain about 100 ms of 16 kHz PCM.  Bound
                # queued live audio to roughly one motion buffer so overload
                # cannot grow microphone latency without limit.
                maxsize=max(
                    4,
                    math.ceil(motion_buffer_frames / 2.5),
                )
            )
            if streaming_audio
            else None
        )
        microphone_pose_queue = (
            queue.Queue(maxsize=microphone_audio_queue.maxsize)
            if streaming_audio and ditto_pose
            else None
        )
        microphone_input_finished = threading.Event()

        def finish_microphone_input():
            if microphone_audio_queue is None:
                return
            if microphone_input_finished.is_set():
                return
            microphone_input_finished.set()
            for target_queue in (
                microphone_audio_queue,
                microphone_pose_queue,
            ):
                if target_queue is None:
                    continue
                try:
                    target_queue.put_nowait(None)
                except queue.Full:
                    # Preserve all accepted PCM and its lip/pose alignment.
                    # A short-lived helper waits for the consumer to make
                    # room, then appends the end marker in FIFO order.
                    threading.Thread(
                        target=target_queue.put,
                        args=(None,),
                        daemon=True,
                    ).start()

        def queue_microphone_pcm(payload):
            if (
                microphone_audio_queue is None
                or microphone_input_finished.is_set()
                or not payload
            ):
                return
            # Browser sends little-endian Float32 mono PCM at 16 kHz.
            usable_bytes = len(payload) - (len(payload) % 4)
            if not usable_bytes:
                return
            pcm = np.frombuffer(
                memoryview(payload)[:usable_bytes],
                dtype="<f4",
            ).astype(np.float32, copy=True)
            if not len(pcm):
                return
            pcm = np.nan_to_num(pcm, nan=0.0, posinf=1.0, neginf=-1.0)
            np.clip(pcm, -1.0, 1.0, out=pcm)
            if microphone_pose_queue is not None:
                # Both consumers must accept or drop the same PCM packet;
                # otherwise Ditto pose and generated lips would drift apart.
                if (
                    microphone_audio_queue.full()
                    or microphone_pose_queue.full()
                ):
                    return
                microphone_audio_queue.put_nowait(pcm)
                microphone_pose_queue.put_nowait(pcm.copy())
                return
            try:
                microphone_audio_queue.put_nowait(pcm)
            except queue.Full:
                # Keeping the newest speech is preferable to accumulating
                # seconds of stale audio in a real-time interaction.
                try:
                    microphone_audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    microphone_audio_queue.put_nowait(pcm)
                except queue.Full:
                    pass

        async def queue_conversation_pcm(pcm):
            if (
                microphone_audio_queue is None
                or microphone_input_finished.is_set()
                or not len(pcm)
            ):
                return False

            async def put_with_cancellation(target_queue, value):
                while not cancel_event.is_set():
                    try:
                        await asyncio.to_thread(
                            target_queue.put,
                            value,
                            True,
                            0.2,
                        )
                        return True
                    except queue.Full:
                        continue
                return False

            if not await put_with_cancellation(
                microphone_audio_queue,
                pcm,
            ):
                return False
            if microphone_pose_queue is not None:
                if not await put_with_cancellation(
                    microphone_pose_queue,
                    pcm.copy(),
                ):
                    return False
            return True

        if conversation_stream_id:
            async def pump_conversation_audio():
                try:
                    async for pcm_bytes in app.state.conversation.iter_audio_stream(
                        conversation_stream_id
                    ):
                        samples = np.frombuffer(
                            pcm_bytes,
                            dtype="<i2",
                        )
                        for offset in range(0, len(samples), 1600):
                            if cancel_event.is_set():
                                return
                            packet_i16 = samples[offset : offset + 1600]
                            if not len(packet_i16):
                                continue
                            packet = (
                                packet_i16.astype(np.float32)
                                / 32768.0
                            )
                            if not await queue_conversation_pcm(packet):
                                return
                            await safe_send_bytes(
                                b"PCM1" + packet_i16.tobytes()
                            )
                except WebSocketDisconnect:
                    stop_disconnected_stream()
                finally:
                    finish_microphone_input()

            conversation_audio_task = asyncio.create_task(
                pump_conversation_audio()
            )

        async def _listen_for_switches():
            try:
                while True:
                    message = await ws.receive()
                    if message["type"] == "websocket.disconnect":
                        raise WebSocketDisconnect(
                            message.get("code", 1000)
                        )
                    payload = message.get("bytes")
                    if payload is not None:
                        if streaming_audio:
                            queue_microphone_pcm(payload)
                        continue
                    raw = message.get("text")
                    if raw is None:
                        continue
                    data = json.loads(raw)
                    if data.get("action") == "stop":
                        cancel_event.set()
                        sdk.abort_current()
                        break
                    if data.get("action") == "mic_stop":
                        finish_microphone_input()
                        continue
                    if data.get("action") == "playhead":
                        sdk.update_playhead(data.get("frame", 0))
                        continue
                    if data.get("action") == "switch":
                        sdk.update_playhead(data.get("frame", 0))
                        if sdk.request_switch(data["source"]):
                            await safe_send_json({
                                "type": "status",
                                "msg": f"Switching to {data['source']}",
                            })
            except WebSocketDisconnect:
                stop_disconnected_stream()
            except Exception:
                cancel_event.set()
                sdk.abort_current()

        switch_task = asyncio.create_task(_listen_for_switches())

        if streaming_audio:
            motion_status = (
                "Listening to live microphone with Ditto pose..."
                if microphone and ditto_pose
                else (
                    "Listening to live microphone..."
                    if microphone
                    else (
                        "Streaming conversation speech with Ditto pose..."
                        if ditto_pose
                        else "Streaming conversation speech..."
                    )
                )
            )
        elif video_dubbing and not audio_path:
            if video_neutral:
                motion_status = "Playing one normalized neutral video cycle..."
            elif relative_lip_mode:
                motion_status = "Playing one normalized Stage0 video cycle..."
            else:
                motion_status = "Playing one source-video cycle..."
        elif video_dubbing:
            motion_status = "Generating lip with source-video motion..."
        elif continuous_idle and not audio_path:
            motion_status = "Generating continuous idle motion..."
        elif ditto_pose:
            motion_status = "Generating lip with Ditto pose..."
        else:
            motion_status = "Generating lip and pose..."
        await safe_send_json({"type": "status", "msg": motion_status})

        def _process_all_chunks():
            def microphone_waveform_chunks(target_queue):
                while not cancel_event.is_set():
                    try:
                        waveform = target_queue.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if waveform is None:
                        yield np.zeros(0, dtype=np.float32), True
                        return
                    yield waveform, False

            if ditto_pose:
                chunksize = (3, 5, 2)
                ditto_context_frames = (
                    sdk.audio2motion.valid_clip_len - sdk.overlap_v2
                )
                audio_padded = np.concatenate(
                    [
                        np.zeros((chunksize[0] * 640,), dtype=np.float32),
                        audio,
                    ],
                    axis=0,
                )
                split_len = int(sum(chunksize) * 0.04 * 16000) + 80
                chunk_stride = chunksize[1] * 640

                def produce_lip():
                    lip_frames = 0
                    try:
                        if streaming_audio:
                            lip_chunks = (
                                motion_generator.generate_waveform_stream(
                                    microphone_waveform_chunks(
                                        microphone_audio_queue
                                    ),
                                    motion_source_info,
                                    lip_shell_motion=lip_condition_motion,
                                    _lip_only=True,
                                    stop_event=cancel_event,
                                )
                            )
                        else:
                            lip_chunks = motion_generator.generate_lip_stream(
                                audio,
                                motion_source_info,
                                audio_chunk_samples=max(
                                    800,
                                    int(16000 * audio_chunk_seconds),
                                ),
                                lip_shell_motion=lip_condition_motion,
                                continuous=continuous_idle,
                                stop_event=cancel_event,
                            )
                        for lip_chunk in lip_chunks:
                            if cancel_event.is_set():
                                break
                            if not sdk.submit_lip_chunk(lip_chunk):
                                break
                            lip_frames += len(lip_chunk)
                    finally:
                        sdk.finish_lip()
                    return lip_frames

                def produce_ditto_pose():
                    if streaming_audio:
                        prefix_samples = chunksize[0] * 640
                        audio_buffer = np.zeros(
                            prefix_samples,
                            dtype=np.float32,
                        )
                        received_samples = 0
                        runs_sent = 0
                        for waveform, final in microphone_waveform_chunks(
                            microphone_pose_queue
                        ):
                            if cancel_event.is_set():
                                return
                            if len(waveform):
                                received_samples += len(waveform)
                                audio_buffer = np.concatenate(
                                    (audio_buffer, waveform),
                                )
                            while len(audio_buffer) >= split_len:
                                sdk.run_chunk(
                                    audio_buffer[:split_len],
                                    chunksize,
                                )
                                audio_buffer = audio_buffer[chunk_stride:]
                                runs_sent += 1
                            if not final:
                                continue

                            target_motion_frames = math.ceil(
                                received_samples / 640
                            )
                            desired_runs = math.ceil(
                                (
                                    target_motion_frames
                                    + ditto_context_frames
                                )
                                / chunksize[1]
                            )
                            while runs_sent < desired_runs:
                                audio_chunk = audio_buffer[:split_len]
                                if len(audio_chunk) < split_len:
                                    audio_chunk = np.pad(
                                        audio_chunk,
                                        (
                                            0,
                                            split_len - len(audio_chunk),
                                        ),
                                        mode="constant",
                                    )
                                sdk.run_chunk(audio_chunk, chunksize)
                                audio_buffer = audio_buffer[chunk_stride:]
                                runs_sent += 1
                            return

                    target_motion_frames = math.ceil(len(audio) / 640)
                    desired_runs = math.ceil(
                        (target_motion_frames + ditto_context_frames)
                        / chunksize[1]
                    )
                    offset = 0
                    runs_sent = 0
                    while continuous_idle or runs_sent < desired_runs:
                        if cancel_event.is_set():
                            break
                        audio_chunk = audio_padded[offset : offset + split_len]
                        if len(audio_chunk) < split_len:
                            audio_chunk = np.pad(
                                audio_chunk,
                                (0, split_len - len(audio_chunk)),
                                mode="constant",
                            )
                        sdk.run_chunk(audio_chunk, chunksize)
                        offset += chunk_stride
                        runs_sent += 1

                with ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="ditto-pose-stream",
                ) as pool:
                    lip_future = pool.submit(produce_lip)
                    ditto_future = pool.submit(produce_ditto_pose)
                    futures = {lip_future, ditto_future}
                    try:
                        while futures:
                            done, _ = wait(
                                futures,
                                timeout=0.2,
                                return_when=FIRST_COMPLETED,
                            )
                            for future in done:
                                future.result()
                                futures.remove(future)
                                if continuous_idle and not cancel_event.is_set():
                                    raise RuntimeError(
                                        "Continuous motion producer stopped unexpectedly"
                                    )
                    except Exception:
                        cancel_event.set()
                        sdk.abort_current()
                        raise

                if cancel_event.is_set():
                    sdk.abort_current()
                sdk.close()
                return sdk.encoded_frame_count

            generated_frames = 0
            chunk_samples = max(800, int(16000 * audio_chunk_seconds))
            if streaming_audio:
                motion_chunks = motion_generator.generate_waveform_stream(
                    microphone_waveform_chunks(
                        microphone_audio_queue
                    ),
                    motion_source_info,
                    lip_shell_motion=lip_condition_motion,
                    stop_event=cancel_event,
                )
            elif video_dubbing:
                motion_chunks = motion_generator.generate_video_dubbing_stream(
                    audio,
                    motion_source_info,
                    audio_chunk_samples=chunk_samples,
                    lip_shell_motion=lip_condition_motion,
                    static_exp=(
                        active_bundle["stage0_shell_motion"]["exp"]
                        if video_neutral
                        else None
                    ),
                    relative_motion=relative_lip_mode,
                    continuous=continuous_idle,
                    stop_event=cancel_event,
                )
            else:
                motion_chunks = motion_generator.generate_stream(
                    audio,
                    motion_source_info,
                    audio_chunk_samples=chunk_samples,
                    lip_shell_motion=lip_condition_motion,
                    continuous=continuous_idle,
                    stop_event=cancel_event,
                )
            for motion_chunk in motion_chunks:
                if cancel_event.is_set():
                    break
                sdk.submit_motion_chunk(motion_chunk)
                generated_frames += len(motion_chunk)
            if cancel_event.is_set():
                sdk.abort_current()
                sdk.close()
                return sdk.encoded_frame_count
            sdk.setup_Nd(
                N_d=math.ceil(
                    generated_frames
                    * sdk.output_fps
                    / sdk.motion_input_fps
                ),
                fade_in=-1,
                fade_out=-1,
                ctrl_info={},
            )
            sdk.finish_motion()
            sdk.close()
            return sdk.encoded_frame_count

        generated_frames = await loop.run_in_executor(None, _process_all_chunks)
        if cancel_event.is_set():
            return
        if conversation_audio_task is not None:
            await conversation_audio_task
        if cancel_event.is_set():
            return
        frame_send_perf = (
            await frame_sender_task
            if frame_sender_task is not None
            else {"frames": 0, "send_ms": 0.0}
        )
        if cancel_event.is_set():
            return
        if speed_reporter_task is not None:
            speed_reporter_task.cancel()
            await asyncio.gather(
                speed_reporter_task,
                return_exceptions=True,
            )
        sdk_perf = sdk.streaming_perf()
        print(
            "[stream-perf] "
            f"frames={frame_send_perf['frames']} "
            f"putback={sdk_perf['putback_ms']:.2f}ms "
            f"jpeg={sdk_perf['jpeg_ms']:.2f}ms "
            f"send={frame_send_perf['send_ms']:.2f}ms "
            f"dropped={sdk_perf['dropped']}"
        )

        if switch_task is not None:
            switch_task.cancel()
            await asyncio.gather(switch_task, return_exceptions=True)

        await safe_send_json({
            "type": "done",
            "total_frames": generated_frames,
        })

    except WebSocketDisconnect:
        stop_disconnected_stream()
    except Exception as e:
        traceback.print_exc()
        if not cancel_event.is_set():
            try:
                await safe_send_json({"type": "error", "msg": str(e)})
            except (WebSocketDisconnect, RuntimeError):
                pass
    finally:
        if conversation_audio_task is not None and not conversation_audio_task.done():
            conversation_audio_task.cancel()
            await asyncio.gather(
                conversation_audio_task,
                return_exceptions=True,
            )
        if frame_sender_task is not None and not frame_sender_task.done():
            frame_sender_task.cancel()
            await asyncio.gather(
                frame_sender_task,
                return_exceptions=True,
            )
        if speed_reporter_task is not None and not speed_reporter_task.done():
            speed_reporter_task.cancel()
            await asyncio.gather(
                speed_reporter_task,
                return_exceptions=True,
            )
        if switch_task is not None and not switch_task.done():
            switch_task.cancel()
            await asyncio.gather(
                switch_task,
                return_exceptions=True,
            )
        if sdk is not None:
            try:
                sdk.close()
            except Exception:
                pass
        lock.release()



if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5070)
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--cfg_pkl", type=str, default=CFG_PKL)
    parser.add_argument("--lip_ckpt", type=str, default=LIP_CKPT)
    parser.add_argument("--pose_ckpt", type=str, default=POSE_CKPT)
    parser.add_argument("--lip_person_id", type=int, default=LIP_PERSON_ID)
    parser.add_argument("--pose_person_id", type=int, default=POSE_PERSON_ID)
    parser.add_argument("--lip_guidance_weight", type=float, default=None)
    parser.add_argument("--pose_guidance_weight", type=float, default=None)
    parser.add_argument("--stream_frames", type=int, default=STREAM_FRAMES)
    parser.add_argument("--audio_chunk_seconds", type=float, default=AUDIO_CHUNK_SECONDS)
    parser.add_argument("--sampling_steps", type=int, default=SAMPLING_STEPS)
    parser.add_argument(
        "--stage0",
        action=argparse.BooleanOptionalAction,
        default=STAGE0,
    )
    parser.add_argument(
        "--neutral",
        action=argparse.BooleanOptionalAction,
        default=NEUTRAL,
    )
    parser.add_argument(
        "--ditto-pose",
        action=argparse.BooleanOptionalAction,
        default=DITTO_POSE,
    )
    parser.add_argument(
        "--continuous_idle",
        action="store_true",
        default=CONTINUOUS_IDLE,
    )
    parser.add_argument("--neutral_motion", type=str, default=NEUTRAL_MOTION)
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    CFG_PKL = args.cfg_pkl
    LIP_CKPT = args.lip_ckpt
    POSE_CKPT = args.pose_ckpt
    LIP_PERSON_ID = args.lip_person_id
    POSE_PERSON_ID = args.pose_person_id
    LIP_GUIDANCE_WEIGHT = args.lip_guidance_weight
    POSE_GUIDANCE_WEIGHT = args.pose_guidance_weight
    STREAM_FRAMES = args.stream_frames
    AUDIO_CHUNK_SECONDS = args.audio_chunk_seconds
    SAMPLING_STEPS = args.sampling_steps
    STAGE0 = args.stage0
    NEUTRAL = args.neutral
    DITTO_POSE = args.ditto_pose
    CONTINUOUS_IDLE = args.continuous_idle
    NEUTRAL_MOTION = args.neutral_motion

    for key in ("no_proxy", "NO_PROXY"):
        existing = os.environ.get(key, "")
        if "localhost" not in existing:
            os.environ[key] = (existing + ",localhost,127.0.0.1,0.0.0.0").strip(",")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
