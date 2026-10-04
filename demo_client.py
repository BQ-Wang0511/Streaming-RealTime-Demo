from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import websockets


def upload(base_url: str, path: Path, endpoint: str) -> dict:
    with path.open("rb") as handle:
        response = requests.post(
            f"{base_url}{endpoint}",
            files={"file": (path.name, handle)},
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def register(base_url: str, source: dict) -> None:
    response = requests.post(
        f"{base_url}/api/avatar/register",
        json={
            "avatar_id": source["avatar_id"],
            "source_path": source["path"],
            "source_name": source["name"],
            "face_idx": source["face_idx"],
            "crop_scale": source["crop_scale"],
        },
        timeout=120,
    )
    response.raise_for_status()
    job = response.json()
    while job.get("job_id") and job["status"] not in {
        "completed",
        "failed",
        "cancelled",
    }:
        print(f"Registration: {job['progress']}% {job['message']}")
        time.sleep(1)
        response = requests.get(
            f"{base_url}/api/avatar/register/{job['job_id']}",
            timeout=120,
        )
        response.raise_for_status()
        job = response.json()
    if job["status"] != "completed":
        raise RuntimeError(job.get("error") or job.get("message"))


def neutralize(base_url: str, source: dict) -> None:
    response = requests.post(
        f"{base_url}/api/avatar/neutralize",
        json={"avatar_id": source["avatar_id"]},
        timeout=120,
    )
    response.raise_for_status()
    job = response.json()
    while job.get("job_id") and job["status"] not in {
        "completed",
        "failed",
        "cancelled",
    }:
        print(f"Neutralization: {job['progress']}% {job['message']}")
        time.sleep(1)
        response = requests.get(
            f"{base_url}/api/avatar/register/{job['job_id']}",
            timeout=120,
        )
        response.raise_for_status()
        job = response.json()
    if job["status"] != "completed":
        raise RuntimeError(job.get("error") or job.get("message"))


async def run(args: argparse.Namespace) -> None:
    base_url = args.server.rstrip("/")
    parsed = urlparse(base_url)
    websocket_scheme = "wss" if parsed.scheme == "https" else "ws"
    websocket_url = f"{websocket_scheme}://{parsed.netloc}/ws"

    source = upload(base_url, args.source, "/api/upload/source")
    audio = upload(base_url, args.audio, "/api/upload/audio")
    register(base_url, source)
    if source["is_video"] and args.neutral:
        neutralize(base_url, source)
    request = {
        "avatar_id": source["avatar_id"],
        "registered_avatar_ids": [source["avatar_id"]],
        "audio": audio["path"],
        "video_dubbing": source["is_video"],
        "ditto_pose": args.ditto_pose and not source["is_video"],
        "stage0": args.stage0,
        "neutral": args.neutral,
        "paste_back": True,
        "continuous_idle": False,
        "render_fps": args.fps,
    }
    frame_count = 0
    started_at = time.monotonic()
    async with websockets.connect(websocket_url, max_size=None) as websocket:
        await websocket.send(json.dumps(request))
        while True:
            message = await websocket.recv()
            if isinstance(message, bytes):
                frame_count += 1
                if frame_count == 1 or frame_count % 5 == 0:
                    await websocket.send(json.dumps({
                        "action": "playhead",
                        "frame": frame_count - 1,
                    }))
                if args.preview_dir and frame_count <= 3:
                    args.preview_dir.mkdir(parents=True, exist_ok=True)
                    (args.preview_dir / f"frame_{frame_count:02d}.jpg").write_bytes(
                        message
                    )
                continue

            event = json.loads(message)
            event_type = event.get("type")
            if event_type == "status":
                print(event.get("msg", ""))
            elif event_type == "ready":
                print(f"Ready: {event.get('total_frames')} frames")
            elif event_type == "error":
                raise RuntimeError(event.get("msg", "Unknown server error"))
            elif event_type == "done":
                elapsed = time.monotonic() - started_at
                print(f"Done: {frame_count} frames in {elapsed:.2f}s")
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="TalkLikeYou streaming client")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--server", default="http://127.0.0.1:5070")
    parser.add_argument("--preview-dir", type=Path)
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument(
        "--ditto-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--stage0",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--neutral",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
