"""Flow Matching lip and pose generation for the real-time demo."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import os
from types import SimpleNamespace

import librosa
import numpy as np
import torch
from scipy import signal

from motion_models import AudioFeatureEncoder, FlowMatchingMotionGenerator

ROOT = os.path.dirname(os.path.abspath(__file__))
LIP_IDX = [6, 12, 14, 17, 19, 20]
POSE_BIN_DIM = 66
POSE_ONLY_DIM = POSE_BIN_DIM * 3
POSE_DEGREE_OFFSET = 97.5
POSE_DEGREE_STEP = 3.0


def _pose_logits_to_degree(logits):
    logits = np.asarray(logits, dtype=np.float32).reshape(-1, POSE_BIN_DIM)
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= np.sum(probabilities, axis=1, keepdims=True)
    bins = np.arange(POSE_BIN_DIM, dtype=np.float32)
    return (
        (probabilities * bins[None]).sum(axis=1) * POSE_DEGREE_STEP
        - POSE_DEGREE_OFFSET
    )


def pose_target_to_degrees(pose_frame):
    values = np.asarray(pose_frame, dtype=np.float32).reshape(3, POSE_BIN_DIM)
    return np.asarray(
        [_pose_logits_to_degree(axis)[0] for axis in values],
        dtype=np.float32,
    )


def _degree_to_pose_logits(degree):
    bins = np.arange(POSE_BIN_DIM, dtype=np.float32)
    center = np.clip(
        (float(degree) + POSE_DEGREE_OFFSET) / POSE_DEGREE_STEP,
        0.0,
        POSE_BIN_DIM - 1.0,
    )
    return np.maximum(-0.5 * (bins - center) ** 2, -8.0).astype(np.float32)


def pose_degrees_to_target(pitch, yaw, roll):
    return np.concatenate(
        [_degree_to_pose_logits(value) for value in (pitch, yaw, roll)]
    ).astype(np.float32)


class StreamingAudioExtractor:
    """Incremental audio encoder with a small future-context holdback."""

    def __init__(self, device):
        self.model = AudioFeatureEncoder().to(device).eval()
        state = torch.load(
            os.path.join(ROOT, "checkpoints", "motion", "audio_encoder.pth"),
            map_location="cpu",
            weights_only=True,
        )
        self.model.audio_encoder.load_state_dict(state, strict=True)
        self.device = device
        self.mel_basis = librosa.filters.mel(
            sr=16000, n_fft=800, n_mels=80, fmin=55, fmax=7600
        )
        self.reset()

    def reset(self):
        self.waveform = np.zeros(0, dtype=np.float32)
        self.emitted_frames = 0

    @staticmethod
    def expected_frames(sample_count):
        sample_count = max(0, int(sample_count))
        if sample_count < 800:
            return 0
        mel_frames = 1 + sample_count // 200
        frame_count = 0
        while int(80 * (frame_count / 25.0)) + 16 <= mel_frames:
            frame_count += 1
        return frame_count

    def _mel(self, first_needed_mel):
        # Keep two centered-STFT columns before the first required column.
        # This makes the work per push depend on the new audio, not total audio.
        mel_offset = max(0, int(first_needed_mel) - 2)
        sample_offset = mel_offset * 200
        waveform = self.waveform[sample_offset:]
        if sample_offset:
            emphasized = waveform.copy()
            emphasized[0] -= 0.97 * self.waveform[sample_offset - 1]
            emphasized[1:] -= 0.97 * waveform[:-1]
        else:
            emphasized = signal.lfilter([1, -0.97], [1], waveform)
        magnitude = np.abs(
            librosa.stft(emphasized, n_fft=800, hop_length=200, win_length=800)
        )
        mel = np.dot(self.mel_basis, magnitude)
        min_level = np.exp(-100 / 20 * np.log(10))
        mel = 20 * np.log10(np.maximum(min_level, mel)) - 20
        mel = np.clip(8 * ((mel + 100) / 100) - 4, -4, 4).T.astype(np.float32)
        return mel, mel_offset

    @torch.no_grad()
    def push(self, waveform, final=False):
        waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
        if waveform.size:
            self.waveform = np.concatenate((self.waveform, waveform))
        if self.waveform.size < 800:
            return np.zeros((0, 512), dtype=np.float32)

        first_needed_mel = int(80 * (self.emitted_frames / 25.0))
        mel, mel_offset = self._mel(first_needed_mel)
        # Two mel columns are held until the next audio block so centered STFT
        # padding cannot change features that have already been emitted.
        stable_mel_len = mel_offset + (len(mel) if final else max(0, len(mel) - 2))
        windows = []
        frame = self.emitted_frames
        while int(80 * (frame / 25.0)) + 16 <= stable_mel_len:
            start = int(80 * (frame / 25.0))
            local_start = start - mel_offset
            windows.append(mel[local_start : local_start + 16].T[None])
            frame += 1
        if not windows:
            return np.zeros((0, 512), dtype=np.float32)

        batch = torch.from_numpy(np.stack(windows)).to(self.device)
        features = self.model(batch).cpu().numpy().astype(np.float32)
        self.emitted_frames = frame
        return features


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class RealtimeMotionGenerator:
    def __init__(
        self,
        lip_ckpt,
        pose_ckpt,
        lip_person_id=0,
        pose_person_id=0,
        lip_guidance_weight=None,
        pose_guidance_weight=None,
        parallel=True,
        re_pose=False,
        no_smooth=False,
        stream_frames=50,
        sampling_steps=1,
        device=None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.parallel = parallel
        self.re_pose = re_pose
        self.no_smooth = no_smooth
        self.stream_frames = max(1, int(stream_frames))
        self.lip_ckpt = os.path.abspath(lip_ckpt)
        self.pose_ckpt = os.path.abspath(pose_ckpt)

        self.lip_args = self._load_model_config(
            self.lip_ckpt, lip_person_id, lip_guidance_weight
        )
        self.pose_args = self._load_model_config(
            self.pose_ckpt, pose_person_id, pose_guidance_weight
        )
        self.lip_args.sampling_steps = int(sampling_steps)
        self.pose_args.sampling_steps = int(sampling_steps)
        self.lip_args.overlap = None
        self.pose_args.overlap = None

        print(f"[motion] Loading lip generator: {self.lip_ckpt}")
        self.lip_model = self._build_model(self.lip_args)
        print(f"[motion] Loading pose generator: {self.pose_ckpt}")
        self.pose_model = self._build_model(self.pose_args)
        self.audio_extractor = StreamingAudioExtractor(self.device)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            self.lip_stream = torch.cuda.Stream(device=self.device)
            self.pose_stream = torch.cuda.Stream(device=self.device)
        else:
            self.lip_stream = self.pose_stream = None
        print(
            f"[motion] Ready: lip_dim={self.lip_args.motion_feat_dim}, "
            f"lip_id={self.lip_args.person_id}, pose_id={self.pose_args.person_id}, "
            f"sampling_steps={sampling_steps}, parallel={self.parallel}, re_pose={self.re_pose}"
        )

    def configure(
        self,
        lip_person_id=None,
        pose_person_id=None,
        lip_guidance_weight=None,
        pose_guidance_weight=None,
        sampling_steps=None,
        stream_frames=None,
    ):
        """Apply per-request sampling controls without reloading checkpoints."""
        if lip_person_id is not None:
            self.lip_args.person_id = int(lip_person_id)
        if pose_person_id is not None:
            self.pose_args.person_id = int(pose_person_id)
        if lip_guidance_weight is not None:
            self.lip_args.guidance_scale = float(lip_guidance_weight)
            self.lip_model.flow_process.guidance_scale = float(lip_guidance_weight)
        if pose_guidance_weight is not None:
            self.pose_args.guidance_scale = float(pose_guidance_weight)
            self.pose_model.flow_process.guidance_scale = float(pose_guidance_weight)
        if sampling_steps is not None:
            steps = max(1, int(sampling_steps))
            self.lip_args.sampling_steps = steps
            self.pose_args.sampling_steps = steps
            self.lip_model.flow_process.sampling_steps = steps
            self.pose_model.flow_process.sampling_steps = steps
        if stream_frames is not None:
            self.stream_frames = max(1, int(stream_frames))

    @staticmethod
    def _load_model_config(checkpoint, person_id, guidance_weight):
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        config = dict(state["config"])
        config["checkpoint"] = checkpoint
        config["person_id"] = int(person_id)
        if guidance_weight is not None:
            config["guidance_scale"] = float(guidance_weight)
        return SimpleNamespace(**config)

    def _build_model(self, args):
        model = FlowMatchingMotionGenerator(
            motion_feat_dim=args.motion_feat_dim,
            person_num=args.person_num,
            audio_feat_dim=args.audio_feat_dim,
            seq_frames=args.sequence_length,
            latent_dim=args.latent_dim,
            ff_size=args.feedforward_dim,
            num_layers=args.layer_count,
            num_heads=args.head_count,
            dropout=0.0,
            time_steps=args.time_steps,
            sampling_steps=args.sampling_steps,
            guidance_scale=args.guidance_scale,
            checkpoint=args.checkpoint,
            device=str(self.device),
            predict_clean_motion=args.predict_clean_motion,
        )
        model.eval()
        return model

    @staticmethod
    def _source_pose(x_info):
        parts = []
        for key in ("pitch", "yaw", "roll"):
            value = to_numpy(x_info[key]).astype(np.float32).reshape(-1)
            if value.size != 66:
                raise ValueError(f"Expected native {key} to have 66 values, got {value.shape}")
            parts.append(value)
        return np.concatenate(parts).astype(np.float32)

    def _lip_condition(self, x_info):
        exp = to_numpy(x_info["exp"]).astype(np.float32).reshape(21, 3)
        if self.lip_args.motion_feat_dim == 18:
            return exp[LIP_IDX].reshape(-1)
        return exp.reshape(-1)

    def _to_x_d_info(self, lip_seq, pose_seq, source_info, shell_motion=None):
        source_frames = source_info["x_s_info_lst"]
        output = []
        for idx, (lip_frame, pose_frame) in enumerate(zip(lip_seq, pose_seq)):
            source = source_frames[idx % len(source_frames)]
            motion_base = shell_motion if shell_motion is not None else source
            exp = to_numpy(motion_base["exp"]).astype(np.float32).reshape(21, 3).copy()
            if self.lip_args.motion_feat_dim == 18:
                exp[LIP_IDX] = np.asarray(lip_frame, dtype=np.float32).reshape(6, 3)
            else:
                exp = np.asarray(lip_frame, dtype=np.float32).reshape(21, 3)

            pose_frame = np.asarray(pose_frame, dtype=np.float32).reshape(3, 66)
            frame = {
                "pitch": pose_frame[0].reshape(1, 66),
                "yaw": pose_frame[1].reshape(1, 66),
                "roll": pose_frame[2].reshape(1, 66),
                "exp": exp.reshape(1, 63),
            }
            if shell_motion is not None and "t" in shell_motion:
                frame["t"] = to_numpy(shell_motion["t"]).astype(np.float32).reshape(1, 3)
            output.append(frame)
        return output

    def _sample_stream_chunk(
        self,
        audio_features,
        lip_cond,
        pose_cond,
        lip_only=False,
    ):
        def sample(args, model, cond, stream):
            device_context = torch.cuda.device(self.device) if stream is not None else nullcontext()
            stream_context = torch.cuda.stream(stream) if stream is not None else nullcontext()
            with device_context, stream_context, torch.no_grad():
                aud = torch.from_numpy(audio_features).unsqueeze(0).to(self.device)
                cond = torch.from_numpy(cond).unsqueeze(0).to(self.device)
                habit_one_hot = torch.zeros(
                    1,
                    args.person_num,
                    device=self.device,
                )
                if not 0 <= args.person_id < args.person_num:
                    raise ValueError(
                        f"person_id must be in [0, {args.person_num - 1}]"
                    )
                habit_one_hot[:, args.person_id] = 1.0
                result = model.sample(
                    cond,
                    aud,
                    habit_one_hot,
                ).squeeze(0).cpu().numpy()
            if stream is not None:
                stream.synchronize()
            return result

        if lip_only:
            return (
                sample(
                    self.lip_args,
                    self.lip_model,
                    lip_cond,
                    self.lip_stream,
                ),
                None,
            )
        if self.parallel:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="motion-stream") as pool:
                lip_future = pool.submit(
                    sample, self.lip_args, self.lip_model, lip_cond, self.lip_stream
                )
                pose_future = pool.submit(
                    sample, self.pose_args, self.pose_model, pose_cond, self.pose_stream
                )
                return lip_future.result(), pose_future.result()
        return (
            sample(self.lip_args, self.lip_model, lip_cond, None),
            sample(self.pose_args, self.pose_model, pose_cond, None),
        )

    @torch.no_grad()
    def generate_stream(
        self,
        audio,
        source_info,
        audio_chunk_samples=16000,
        lip_shell_motion=None,
        _lip_only=False,
        continuous=False,
        stop_event=None,
    ):
        """Yield rendered-motion chunks while audio features become available."""
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        audio_chunk_samples = max(1, int(audio_chunk_samples))

        def waveform_chunks():
            for sample_start in range(0, len(audio), audio_chunk_samples):
                if stop_event is not None and stop_event.is_set():
                    return
                yield audio[
                    sample_start : sample_start + audio_chunk_samples
                ], False
            if continuous:
                silence = np.zeros(audio_chunk_samples, dtype=np.float32)
                while stop_event is None or not stop_event.is_set():
                    yield silence, False
            else:
                yield np.zeros(0, dtype=np.float32), True

        yield from self.generate_waveform_stream(
            waveform_chunks(),
            source_info,
            lip_shell_motion=lip_shell_motion,
            _lip_only=_lip_only,
            stop_event=stop_event,
        )

    @torch.no_grad()
    def generate_waveform_stream(
        self,
        waveform_chunks,
        source_info,
        lip_shell_motion=None,
        _lip_only=False,
        stop_event=None,
    ):
        """Generate motion directly from an iterator of live 16 kHz PCM chunks.

        ``waveform_chunks`` yields ``(float32_pcm, final)`` pairs.  The final
        pair flushes the audio feature extractor without requiring the full
        waveform to be stored first.
        """
        self.audio_extractor.reset()
        source0 = source_info["x_s_info_lst"][0]
        lip_cond = self._lip_condition(lip_shell_motion or source0)
        pose_cond = None if _lip_only else self._source_pose(source0)
        pose_origin = None
        feature_buffer = np.zeros((0, 512), dtype=np.float32)
        source_offset = 0

        for waveform, final in waveform_chunks:
            if stop_event is not None and stop_event.is_set():
                return
            waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
            new_features = self.audio_extractor.push(waveform, final=final)
            if len(new_features):
                feature_buffer = np.concatenate((feature_buffer, new_features), axis=0)

            while len(feature_buffer) >= self.stream_frames or (final and len(feature_buffer)):
                take = min(self.stream_frames, len(feature_buffer))
                features = feature_buffer[:take]
                feature_buffer = feature_buffer[take:]
                lip_seq, pose_seq = self._sample_stream_chunk(
                    features,
                    lip_cond,
                    pose_cond,
                    lip_only=_lip_only,
                )
                lip_seq = np.asarray(lip_seq[:take], dtype=np.float32)
                lip_cond = lip_seq[-1].copy()
                if _lip_only:
                    yield lip_seq
                    continue

                pose_seq = np.asarray(pose_seq[:take], dtype=np.float32)
                pose_cond = pose_seq[-1].copy()

                if self.re_pose:
                    if pose_origin is None:
                        pose_origin = pose_target_to_degrees(pose_seq[0])
                    aligned = []
                    source_frames = source_info["x_s_info_lst"]
                    for idx, pose_frame in enumerate(pose_seq):
                        source_pose = self._source_pose(source_frames[(source_offset + idx) % len(source_frames)])
                        degree = pose_target_to_degrees(source_pose) + (
                            pose_target_to_degrees(pose_frame) - pose_origin
                        )
                        aligned.append(pose_degrees_to_target(*degree))
                    pose_seq = np.asarray(aligned, dtype=np.float32)

                chunk_source = dict(source_info)
                frames = source_info["x_s_info_lst"]
                chunk_source["x_s_info_lst"] = [
                    frames[(source_offset + idx) % len(frames)] for idx in range(take)
                ]
                source_offset += take
                yield self._to_x_d_info(
                    lip_seq, pose_seq, chunk_source, shell_motion=lip_shell_motion
                )
            if final:
                break

    def generate_lip_stream(
        self,
        audio,
        source_info,
        audio_chunk_samples=16000,
        lip_shell_motion=None,
        continuous=False,
        stop_event=None,
    ):
        """Yield lip-only chunks for merging with native renderer motion."""
        yield from self.generate_stream(
            audio,
            source_info,
            audio_chunk_samples=audio_chunk_samples,
            lip_shell_motion=lip_shell_motion,
            _lip_only=True,
            continuous=continuous,
            stop_event=stop_event,
        )

    def generate_video_dubbing_stream(
        self,
        audio,
        source_info,
        audio_chunk_samples=16000,
        lip_shell_motion=None,
        static_exp=None,
        relative_motion=False,
        continuous=False,
        stop_event=None,
    ):
        """Use video-native motion and replace only its six lip landmarks."""
        source_frames = source_info["x_s_info_lst"]
        if not source_frames:
            raise ValueError("Video source has no motion frames")
        source_offset = 0
        relative_origin_exp = None
        static_exp = (
            None
            if static_exp is None
            else np.asarray(static_exp, dtype=np.float32).reshape(1, 63)
        )
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)

        def native_chunk(frame_count, static=False):
            nonlocal source_offset, relative_origin_exp
            from runtime.core.atomic_components.condition_handler import _mirror_index

            chunk = []
            for idx in range(frame_count):
                frame_idx = _mirror_index(
                    source_offset + idx,
                    len(source_frames),
                )
                source = source_frames[frame_idx]
                frame = {
                    key: to_numpy(value).copy()
                    for key, value in source.items()
                }
                frame["exp"] = (
                    to_numpy(source["exp"])
                    .astype(np.float32)
                    .reshape(1, 63)
                    .copy()
                )
                if static_exp is not None and static:
                    if relative_motion and relative_origin_exp is not None:
                        frame["exp"] = relative_origin_exp.copy()
                    else:
                        frame["exp"] = static_exp.copy()
                        if relative_motion and relative_origin_exp is None:
                            relative_origin_exp = frame["exp"].copy()
                chunk.append(frame)
            source_offset += frame_count
            return chunk

        if not len(audio):
            if not continuous:
                # Base video-dubbing playback: emit exactly one source-video
                # cycle, then finish instead of looping forever.
                yield native_chunk(len(source_frames), static=True)
                return
            while continuous and (stop_event is None or not stop_event.is_set()):
                yield native_chunk(self.stream_frames, static=True)
            return

        for lip_chunk in self.generate_lip_stream(
            audio,
            source_info,
            audio_chunk_samples=audio_chunk_samples,
            lip_shell_motion=lip_shell_motion,
            continuous=False,
            stop_event=stop_event,
        ):
            output = native_chunk(len(lip_chunk), static=False)
            for frame, lip_frame in zip(output, lip_chunk):
                exp = frame["exp"].reshape(21, 3)
                lip_frame = np.asarray(lip_frame, dtype=np.float32)
                if lip_frame.size == 18:
                    exp[LIP_IDX] = lip_frame.reshape(6, 3)
                elif lip_frame.size == 63:
                    exp[LIP_IDX] = lip_frame.reshape(21, 3)[LIP_IDX]
                else:
                    raise ValueError(
                        f"Unexpected lip frame shape for video dubbing: {lip_frame.shape}"
                    )
            if relative_motion and relative_origin_exp is None and output:
                relative_origin_exp = output[0]["exp"].copy()
            yield output

        while continuous and (stop_event is None or not stop_event.is_set()):
            yield native_chunk(self.stream_frames, static=True)
