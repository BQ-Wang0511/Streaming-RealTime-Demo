"""Persistent preprocessed avatar bundles for the TalkLikeYou demo."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import time
import uuid

import cv2
import numpy as np


BUNDLE_VERSION = 3
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class AvatarRegistry:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_id(avatar_id: str) -> str:
        clean = re.sub(r"[^a-zA-Z0-9_.-]", "", avatar_id or "")[:96]
        if not clean:
            raise ValueError("Invalid avatar_id")
        return clean

    def avatar_dir(self, avatar_id: str) -> Path:
        return self.root / self.normalize_id(avatar_id)

    def bundle_path(self, avatar_id: str) -> Path:
        return self.avatar_dir(avatar_id) / "bundle.pkl"

    def is_registered(self, avatar_id: str) -> bool:
        path = self.bundle_path(avatar_id)
        if not path.is_file():
            return False
        try:
            bundle = self._load_path(path)
        except Exception:
            return False
        if not bundle.get("source_info"):
            return False
        is_video = bool(
            bundle.get(
                "is_video",
                not bundle["source_info"].get("is_image_flag", True),
            )
        )
        if not is_video:
            return bool(
                bundle.get("neutral_ready")
                and bundle.get("neutral_source_info") is not None
                and bundle.get("stage0_shell_motion") is not None
                and "lip_normalize_delta_kp" in bundle
                and "neutral_lip_normalize_delta_kp" in bundle
            )
        return bool(
            bundle.get("video_s0_lip_ready")
            and bundle.get("stage0_shell_motion") is not None
            and "lip_normalize_delta_kp" in bundle
        )

    def registration_matches(
        self,
        avatar_id: str,
        *,
        face_idx: int,
        crop_scale: float,
    ) -> bool:
        """Return whether a cached bundle uses the requested crop settings."""
        try:
            bundle = self.load(avatar_id)
        except Exception:
            return False
        return (
            bundle.get("face_selection_order") == "left-right"
            and int(bundle.get("face_idx", 0)) == int(face_idx)
            and math.isclose(
                float(bundle.get("crop_scale", 2.3)),
                float(crop_scale),
                rel_tol=0.0,
                abs_tol=1e-6,
            )
        )

    def is_stage0_ready(self, avatar_id: str) -> bool:
        try:
            bundle = self.load(avatar_id)
        except Exception:
            return False
        is_video = bool(
            bundle.get(
                "is_video",
                not bundle["source_info"].get("is_image_flag", True),
            )
        )
        if not is_video:
            return self.is_neutralized(avatar_id)
        return bool(
            bundle.get("video_s0_lip_ready")
            and bundle.get("stage0_shell_motion") is not None
            and "lip_normalize_delta_kp" in bundle
        )

    def is_stage0_per_frame_normalize_ready(self, avatar_id: str) -> bool:
        try:
            bundle = self.load(avatar_id)
        except Exception:
            return False
        if not self.is_stage0_ready(avatar_id):
            return False
        sequence = bundle.get("lip_normalize_delta_kp_sequence")
        source_info = bundle.get("source_info") or {}
        return bool(
            isinstance(sequence, (list, tuple))
            and len(sequence) == len(source_info.get("x_s_info_lst") or [])
        )

    def is_neutralized(
        self, avatar_id: str, neutral_motion_sha256: str | None = None
    ) -> bool:
        try:
            bundle = self.load(avatar_id)
        except Exception:
            return False
        is_video = bool(
            bundle.get(
                "is_video",
                not bundle["source_info"].get("is_image_flag", True),
            )
        )
        ready = bool(
            bundle.get("neutral_ready")
            and bundle.get("neutral_source_info") is not None
            and bundle.get("stage0_shell_motion") is not None
            and "lip_normalize_delta_kp" in bundle
            and "neutral_lip_normalize_delta_kp" in bundle
            and (not is_video or bundle.get("video_neutral_ready"))
        )
        if neutral_motion_sha256 is not None:
            ready = ready and (
                bundle.get("neutral_motion_sha256") == neutral_motion_sha256
            )
        return ready

    def clear(self) -> int:
        bundle_count = sum(1 for _ in self.root.glob("*/bundle.pkl"))
        for path in self.root.iterdir():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
        return bundle_count

    def list_registered(self) -> list[dict[str, object]]:
        registrations = []
        for path in sorted(self.root.glob("*/bundle.pkl")):
            try:
                bundle = self._load_path(path)
            except Exception:
                continue
            registrations.append(self.public_metadata(bundle))
        return registrations

    def load(self, avatar_id: str) -> dict:
        path = self.bundle_path(avatar_id)
        if not path.is_file():
            raise FileNotFoundError(f"Avatar is not registered: {avatar_id}")
        return self._load_path(path)

    def _load_path(self, path: Path) -> dict:
        with path.open("rb") as handle:
            bundle = pickle.load(handle)
        if bundle.get("version") != BUNDLE_VERSION:
            raise RuntimeError(
                f"Unsupported avatar bundle version: {bundle.get('version')}"
            )
        required = {"avatar_id", "source_path", "source_info"}
        missing = sorted(required.difference(bundle))
        if missing:
            raise RuntimeError(f"Avatar bundle is incomplete: {', '.join(missing)}")
        return bundle

    @staticmethod
    def public_metadata(bundle: dict) -> dict[str, object]:
        return {
            "avatar_id": bundle["avatar_id"],
            "source_name": bundle.get("source_name", ""),
            "registered_at": bundle.get("registered_at"),
            "neutral_motion_sha256": bundle.get("neutral_motion_sha256"),
            "neutralized": bool(bundle.get("neutral_ready", False)),
            "face_idx": int(bundle.get("face_idx", 0)),
            "crop_scale": float(bundle.get("crop_scale", 2.3)),
            "is_video": bool(
                bundle.get(
                    "is_video",
                    not bundle["source_info"].get("is_image_flag", True),
                )
            ),
            "version": bundle.get("version"),
        }

    def register(
        self,
        *,
        avatar_id: str,
        source_path: str,
        source_name: str,
        neutral_motion: str,
        prepare_per_frame_normalize_delta: bool,
        face_idx: int,
        crop_scale: float,
        sdk,
        hybrid,
        progress,
    ) -> dict:
        avatar_id = self.normalize_id(avatar_id)
        source = Path(source_path).resolve()
        source_suffix = source.suffix.lower()
        is_video = source_suffix in VIDEO_EXTENSIONS
        if source_suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError("Unsupported avatar source type")
        if not source.is_file():
            raise FileNotFoundError(f"Avatar source not found: {source}")
        neutral = Path(neutral_motion).resolve()
        if not neutral.is_file():
            raise FileNotFoundError(f"Neutral motion not found: {neutral}")
        if is_video:
            if hybrid is None:
                raise ValueError("Video registration requires portrait preprocessing")
        avatar_dir = self.avatar_dir(avatar_id)
        avatar_dir.mkdir(parents=True, exist_ok=True)

        face_idx = int(face_idx)
        crop_scale = float(crop_scale)
        if face_idx < 0:
            raise ValueError("face_idx must be greater than or equal to 0")
        if not math.isfinite(crop_scale) or crop_scale <= 0:
            raise ValueError("crop_scale must be a finite number greater than 0")

        default_kwargs = sdk.default_kwargs
        crop_kwargs = {
            "face_idx": face_idx,
            "crop_scale": crop_scale,
            "crop_vx_ratio": default_kwargs.get("crop_vx_ratio", 0),
            "crop_vy_ratio": default_kwargs.get("crop_vy_ratio", -0.125),
            "crop_flag_do_rot": default_kwargs.get("crop_flag_do_rot", True),
        }
        max_dim = int(default_kwargs.get("max_size", 1920))

        progress(8, "Extracting avatar features")
        source_info = sdk.avatar_registrar(
            str(source), max_dim=max_dim, n_frames=-1, **crop_kwargs
        )

        stage0_shell_motion = None
        lip_delta_sequence = None
        source_lip_delta = None
        stage0_neutral_motion_sha256 = None
        stage0_neutral_motion_path = None
        if is_video:
            frames = source_info.get("img_rgb_lst") or []
            if not frames:
                raise RuntimeError("Video source has no frames")
            progress(35, "Preparing video Stage0 motion shell")
            pipeline = hybrid.create_liveportrait_pipeline(crop_scale, face_idx)
            try:
                stage0_shell_motion = hybrid.load_neutral_motion(str(neutral))
                if prepare_per_frame_normalize_delta:
                    lip_delta_sequence = []
                    progress(50, "Computing per-frame video lip normalization")
                    for index, source_rgb in enumerate(frames):
                        original_delta = (
                            hybrid.compute_lip_normalize_delta_kp_from_rgb(
                                source_rgb=source_rgb,
                                face_idx=face_idx,
                                pipeline=pipeline,
                                source_desc=f"{source} [frame {index}]",
                            )
                        )
                        if index == 0:
                            source_lip_delta = original_delta
                        lip_delta_sequence.append(original_delta)
                        progress(
                            50 + int(38 * (index + 1) / len(frames)),
                            f"Computing video normalize delta {index + 1}/{len(frames)}",
                        )
                else:
                    progress(70, "Computing first-frame video lip normalization")
                    source_lip_delta = (
                        hybrid.compute_lip_normalize_delta_kp_from_rgb(
                            source_rgb=frames[0],
                            face_idx=face_idx,
                            pipeline=pipeline,
                            source_desc=f"{source} [frame 0]",
                        )
                    )
            finally:
                hybrid.destroy_liveportrait_pipeline(pipeline)
            stage0_neutral_motion_sha256 = file_sha256(neutral)
            stage0_neutral_motion_path = str(neutral)

        progress(90, "Saving avatar bundle")
        bundle = {
            "version": BUNDLE_VERSION,
            "avatar_id": avatar_id,
            "source_name": source_name,
            "is_video": is_video,
            "source_path": str(source),
            "source_sha256": file_sha256(source),
            "stage0_neutral_motion_sha256": stage0_neutral_motion_sha256,
            "stage0_neutral_motion_path": stage0_neutral_motion_path,
            "neutral_motion_sha256": None,
            "neutral_motion_path": None,
            "neutralized_image_path": None,
            "registered_at": time.time(),
            "face_idx": face_idx,
            "face_selection_order": "left-right",
            "crop_scale": crop_scale,
            "source_info": source_info,
            "neutral_source_info": None,
            "lip_normalize_delta_kp": _optional_float32(source_lip_delta),
            "lip_normalize_delta_kp_sequence": (
                _optional_float32_sequence(lip_delta_sequence)
            ),
            "neutral_lip_normalize_delta_kp": None,
            "stage0_shell_motion": stage0_shell_motion,
            "neutral_ready": False,
            "video_s0_lip_ready": is_video,
            "video_neutral_ready": False,
        }
        bundle_path = self.bundle_path(avatar_id)
        temporary_path = bundle_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary_path.open("wb") as handle:
                pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, bundle_path)
        finally:
            temporary_path.unlink(missing_ok=True)
        progress(100, "Registered")
        return self.public_metadata(bundle)

    def neutralize(
        self, *, avatar_id: str, neutral_motion: str, sdk, hybrid, progress
    ) -> dict:
        bundle = self.load(avatar_id)
        source = Path(bundle["source_path"]).resolve()
        neutral = Path(neutral_motion).resolve()
        if not source.is_file() or file_sha256(source) != bundle["source_sha256"]:
            raise RuntimeError("Registered source changed; register it again")
        if not neutral.is_file():
            raise FileNotFoundError(f"Neutral motion not found: {neutral}")

        avatar_dir = self.avatar_dir(avatar_id)
        neutral_temp = avatar_dir / f"neutral_{uuid.uuid4().hex}.png"
        neutral_video_temp = avatar_dir / f"neutral_video_{uuid.uuid4().hex}.mp4"
        neutral_final = avatar_dir / "neutral.png"
        source_info = bundle["source_info"]
        is_video = bool(bundle["is_video"])
        face_idx = int(bundle.get("face_idx", 0))
        crop_scale = float(bundle["crop_scale"])
        default_kwargs = sdk.default_kwargs
        crop_kwargs = {
            "face_idx": face_idx,
            "crop_scale": crop_scale,
            "crop_vx_ratio": default_kwargs.get("crop_vx_ratio", 0),
            "crop_vy_ratio": default_kwargs.get("crop_vy_ratio", -0.125),
            "crop_flag_do_rot": default_kwargs.get("crop_flag_do_rot", True),
        }
        max_dim = int(default_kwargs.get("max_size", 1920))
        pipeline = hybrid.create_liveportrait_pipeline(crop_scale, face_idx)
        try:
            progress(8, "Building Stage0 motion shell")
            stage0_shell_motion = hybrid.load_neutral_motion(str(neutral))
            if is_video:
                import portrait_video_helpers as stage0_video
                neutral_exp = stage0_shell_motion["exp"]
                fps = float(hybrid.get_fps(str(source)))
                if fps <= 0:
                    fps = 25.0
                writer = None
                frames = source_info["img_rgb_lst"]
                try:
                    for index, source_rgb in enumerate(frames):
                        rgb = stage0_video.render_stage0_neutral_frame(
                            source_rgb=source_rgb, neutral_exp=neutral_exp,
                            pipeline=pipeline, face_idx=face_idx,
                        )
                        rgb = stage0_video.apply_stage0_retarget_to_rgb(
                            rgb, pipeline=pipeline, face_idx=face_idx,
                            target_lip_open_ratio=hybrid.STAGE0_TARGET_LIP_OPEN_RATIO,
                            lip_close_open=hybrid.STAGE0_LIP_CLOSE_OPEN,
                            grin=hybrid.STAGE0_GRIN,
                        )
                        if writer is None:
                            h, w = rgb.shape[:2]
                            writer = cv2.VideoWriter(
                                str(neutral_video_temp),
                                cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                            )
                            if not writer.isOpened():
                                raise RuntimeError("Failed to create neutralized video")
                        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                        progress(
                            15 + int(55 * (index + 1) / len(frames)),
                            f"Neutralizing video frame {index + 1}/{len(frames)}",
                        )
                finally:
                    if writer is not None:
                        writer.release()
                neutral_source_info = sdk.avatar_registrar(
                    str(neutral_video_temp), max_dim=max_dim, n_frames=-1,
                    **crop_kwargs,
                )
                neutralized_path = str(source)
            else:
                from argparse import Namespace
                progress(25, "Neutralizing avatar")
                neutral_path, _ = hybrid.preprocess_source_with_neutral_motion(
                    Namespace(
                        source_path=str(source), neutral_motion=str(neutral),
                        neutral_output_path=str(neutral_temp), retarget=True,
                        crop_scale=crop_scale, face_idx=face_idx,
                    ),
                    pipeline,
                )
                progress(60, "Extracting neutral avatar features")
                neutral_source_info = sdk.avatar_registrar(
                    neutral_path, max_dim=max_dim, n_frames=-1, **crop_kwargs
                )
                neutralized_path = str(neutral_final)

            progress(82, "Computing lip normalization")
            source_lip_delta = hybrid.compute_lip_normalize_delta_kp_from_rgb(
                source_rgb=source_info["img_rgb_lst"][0], face_idx=face_idx,
                pipeline=pipeline, source_desc=str(source),
            )
            neutral_lip_delta = hybrid.compute_lip_normalize_delta_kp_from_rgb(
                source_rgb=neutral_source_info["img_rgb_lst"][0],
                face_idx=face_idx,
                pipeline=pipeline, source_desc=f"{source} [neutralized]",
            )
        finally:
            hybrid.destroy_liveportrait_pipeline(pipeline)

        progress(95, "Saving neutral assets")
        if not is_video:
            os.replace(neutral_temp, neutral_final)

        bundle.update({
            "neutral_motion_sha256": file_sha256(neutral),
            "neutral_motion_path": str(neutral),
            "neutralized_image_path": neutralized_path,
            "neutralized_at": time.time(),
            "neutral_source_info": neutral_source_info,
            "lip_normalize_delta_kp": _optional_float32(source_lip_delta),
            "neutral_lip_normalize_delta_kp": _optional_float32(neutral_lip_delta),
            "stage0_shell_motion": stage0_shell_motion,
            "neutral_ready": True,
            "video_s0_lip_ready": is_video,
            "video_neutral_ready": is_video,
        })
        bundle_path = self.bundle_path(avatar_id)
        temporary_path = bundle_path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary_path.open("wb") as handle:
                pickle.dump(bundle, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(temporary_path, bundle_path)
        finally:
            temporary_path.unlink(missing_ok=True)
            neutral_temp.unlink(missing_ok=True)
            neutral_video_temp.unlink(missing_ok=True)
        progress(100, "Neutralized")
        return self.public_metadata(bundle)


def _optional_float32(value):
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32)


def _optional_float32_sequence(values):
    if values is None:
        return None
    return [_optional_float32(value) for value in values]
