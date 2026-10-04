# coding: utf-8

"""
StreamSDK subclass for WebSocket rendering of externally generated motion.

Architecture:
  putback_worker + JPEG → bounded frame_queue → async WebSocket sender

Also supports:
  - Avatar cache (save/load source_info, skip face detection on reuse)
  - Multi-source dynamic switching
  - setup_cached() for instant source change
"""

import queue
import threading
import time
import traceback
import copy

import numpy as np

from runtime.stream_pipeline import StreamSDK

FRAME_QUEUE_SIZE = 10  # extra headroom beyond the configured motion buffer
PIPELINE_QUEUE_SIZE = 16
MOTION_FPS = 25.0
LIP_IDX = (6, 12, 14, 17, 19, 20)
MOTION_INTERPOLATION_KEYS = {
    "exp",
    "pitch",
    "yaw",
    "roll",
    "scale",
    "t",
    "kp",
}


def _copy_motion_info(motion_info):
    return {
        key: value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        for key, value in motion_info.items()
    }


def _interpolate_motion_info(previous, current, alpha):
    if alpha <= 0:
        return _copy_motion_info(previous)
    if alpha >= 1:
        return _copy_motion_info(current)

    output = {}
    for key in previous.keys() | current.keys():
        if (
            key in MOTION_INTERPOLATION_KEYS
            and key in previous
            and key in current
        ):
            previous_value = np.asarray(previous[key])
            current_value = np.asarray(current[key])
            if (
                previous_value.shape == current_value.shape
                and np.issubdtype(previous_value.dtype, np.number)
                and np.issubdtype(current_value.dtype, np.number)
            ):
                output[key] = (
                    previous_value * (1.0 - alpha)
                    + current_value * alpha
                ).astype(
                    np.result_type(
                        previous_value.dtype,
                        current_value.dtype,
                        np.float32,
                    ),
                    copy=False,
                )
                continue
        selected = previous if alpha < 0.5 else current
        value = selected.get(key)
        output[key] = (
            value.copy() if isinstance(value, np.ndarray) else copy.deepcopy(value)
        )
    return output


class MotionFrameRateConverter:
    """Stream 25 Hz motion as a contiguous, interpolated output timeline."""

    def __init__(self, output_fps, input_fps=MOTION_FPS):
        self.input_fps = float(input_fps)
        self.output_fps = float(output_fps)
        self.input_count = 0
        self.output_count = 0
        self.previous_motion = None
        self.previous_ctrl = None

    def _output_time(self):
        return self.output_count / self.output_fps

    def push(self, motion_info, ctrl_kwargs):
        current_input_idx = self.input_count
        current_time = current_input_idx / self.input_fps

        if self.previous_motion is None:
            # The first output frame is exactly the first 25 Hz motion frame.
            yield (
                self.output_count,
                0.0,
                _copy_motion_info(motion_info),
                copy.deepcopy(ctrl_kwargs),
            )
            self.output_count += 1
        else:
            previous_input_idx = current_input_idx - 1
            while self._output_time() <= current_time + 1e-9:
                source_position = (
                    self._output_time() * self.input_fps
                )
                alpha = min(
                    1.0,
                    max(0.0, source_position - previous_input_idx),
                )
                yield (
                    self.output_count,
                    source_position,
                    _interpolate_motion_info(
                        self.previous_motion,
                        motion_info,
                        alpha,
                    ),
                    copy.deepcopy(
                        self.previous_ctrl if alpha < 0.5 else ctrl_kwargs
                    ),
                )
                self.output_count += 1

        self.previous_motion = _copy_motion_info(motion_info)
        self.previous_ctrl = copy.deepcopy(ctrl_kwargs)
        self.input_count += 1

    def finish(self):
        if self.previous_motion is None:
            return
        # N input frames represent N / 25 seconds.  Hold the final motion for
        # output samples after its timestamp so duration remains unchanged.
        end_time = self.input_count / self.input_fps
        while self._output_time() < end_time - 1e-9:
            source_position = min(
                self._output_time() * self.input_fps,
                self.input_count - 1,
            )
            yield (
                self.output_count,
                source_position,
                _copy_motion_info(self.previous_motion),
                copy.deepcopy(self.previous_ctrl),
            )
            self.output_count += 1


class _NullWriter:
    def close(self):
        pass


class _NullProgress:
    def update(self, *_args, **_kwargs):
        pass

    def close(self):
        pass


class StreamSDKStreaming(StreamSDK):
    """StreamSDK that pushes frames into a bounded queue instead of writing MP4."""

    def __init__(self, cfg_pkl, data_root, **kwargs):
        super().__init__(cfg_pkl, data_root, **kwargs)
        # Bounded encoded-frame queue consumed by the async WebSocket sender.
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.frame_stream_finished = threading.Event()
        self.frame_queue_dropped = 0
        self.putback_seconds = 0.0
        self.jpeg_seconds = 0.0
        self.encoded_frame_count = 0
        # Source registry for on-the-fly switching
        self.source_registry = {}
        self.active_source_name = None
        self.active_lip_normalize_delta_kp = None
        self.active_lip_normalize_delta_kp_sequence = None
        self.pending_switch = None
        self.ditto_pose_enabled = False
        self.lip_motion_queue = queue.Queue()
        self._last_lip_motion = None
        self._lip_stream_finished = False
        self._motion_input_finished = False
        self.source_video_loop = False
        self.pace_output = False
        self.motion_input_fps = MOTION_FPS
        self.output_fps = MOTION_FPS
        self.render_fps = MOTION_FPS
        self.paste_back = True
        self._output_clock_started_at = None
        self.gen_frame_idx = 0
        self.output_frame_idx = 0
        self.current_source_frame_idx = 0
        self.current_source_next_frame_idx = 0
        self.current_source_frame_alpha = 0.0
        self._frame_seq = 0
        self.last_output_frame_idx = -1
        self.max_motion_buffer_frames = 50
        self.client_playhead_frame = 0
        self.playhead_has_advanced = False
        self._playhead_lock = threading.Lock()
        self._switch_lock = threading.Lock()
        self._closed = True

    def reset_for_reuse(self):
        """Call between connections to reuse the same SDK safely."""
        # Remove request-scoped wrappers before setup installs a fresh one.
        while hasattr(self.motion_stitch, "_base"):
            self.motion_stitch = self.motion_stitch._base
        if hasattr(self, "stop_event"):
            self.stop_event.clear()
        self.worker_exception = None
        self.pending_switch = None
        self.source_registry.clear()
        self.active_source_name = None
        self.active_lip_normalize_delta_kp = None
        self.active_lip_normalize_delta_kp_sequence = None
        self.gen_frame_idx = 0
        self._frame_seq = 0
        self._motion_input_finished = False
        self.source_video_loop = False
        self.pace_output = False
        self.motion_input_fps = MOTION_FPS
        self.output_fps = MOTION_FPS
        self.render_fps = MOTION_FPS
        self.paste_back = True
        self._output_clock_started_at = None
        self.output_frame_idx = 0
        self.current_source_frame_idx = 0
        self.current_source_next_frame_idx = 0
        self.current_source_frame_alpha = 0.0
        self.last_output_frame_idx = -1
        self.max_motion_buffer_frames = 50
        self.client_playhead_frame = 0
        self.playhead_has_advanced = False
        self.set_ditto_pose_mode(False)
        self._closed = False
        QUEUE_MAX = PIPELINE_QUEUE_SIZE
        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)
        self.frame_stream_finished = threading.Event()
        self.frame_queue_dropped = 0
        self.putback_seconds = 0.0
        self.jpeg_seconds = 0.0
        self.encoded_frame_count = 0

    def register_source_context(
        self,
        name,
        source_info,
        lip_normalize_delta_kp=None,
        lip_normalize_delta_kp_sequence=None,
    ):
        self.source_registry[name] = {
            "source_info": source_info,
            "lip_normalize_delta_kp": lip_normalize_delta_kp,
            "lip_normalize_delta_kp_sequence": lip_normalize_delta_kp_sequence,
        }

    def request_switch(self, name):
        with self._switch_lock:
            if name not in self.source_registry:
                return False
            if name == self.active_source_name:
                self.pending_switch = None
            else:
                self.pending_switch = name
        return True

    def configure_motion_buffer(self, frame_count):
        with self._playhead_lock:
            self.max_motion_buffer_frames = max(1, int(frame_count))
            self.client_playhead_frame = 0
            self.playhead_has_advanced = False
        # Before playback starts, rendering may legitimately run one complete
        # motion buffer ahead of the browser.  Keep enough encoded-frame
        # capacity for that burst plus a little network-jitter headroom.
        queue_headroom = max(
            FRAME_QUEUE_SIZE,
            int(np.ceil(
                FRAME_QUEUE_SIZE * self.output_fps / MOTION_FPS
            )),
        )
        self.frame_queue = queue.Queue(
            maxsize=self.max_motion_buffer_frames + queue_headroom
        )

    def _queue_encoded_frame(self, frame_idx, jpg_bytes):
        """Queue the newest preview frame without blocking GPU rendering."""
        packet = (int(frame_idx), jpg_bytes)
        try:
            self.frame_queue.put_nowait(packet)
            return
        except queue.Full:
            pass

        # A slow client must not back-pressure decode/putback indefinitely.
        # Drop the stalest encoded preview frame and retain the newest one so
        # the browser remains close to the audio/playhead timeline.
        try:
            self.frame_queue.get_nowait()
            self.frame_queue_dropped += 1
        except queue.Empty:
            pass
        try:
            self.frame_queue.put_nowait(packet)
        except queue.Full:
            self.frame_queue_dropped += 1

    def get_encoded_frame(self, timeout=0.2):
        return self.frame_queue.get(timeout=timeout)

    def frame_stream_drained(self):
        return self.frame_stream_finished.is_set() and self.frame_queue.empty()

    def streaming_perf(self):
        count = self.encoded_frame_count
        return {
            "frames": count,
            "putback_ms": (
                self.putback_seconds * 1000.0 / count if count else 0.0
            ),
            "jpeg_ms": (
                self.jpeg_seconds * 1000.0 / count if count else 0.0
            ),
            "dropped": self.frame_queue_dropped,
        }

    def update_playhead(self, frame_idx):
        with self._playhead_lock:
            self.client_playhead_frame = max(
                self.client_playhead_frame,
                max(0, int(frame_idx)),
            )
            if self.client_playhead_frame > 0:
                self.playhead_has_advanced = True

    def _wait_for_motion_window(self, global_frame_idx):
        while not self.stop_event.is_set():
            with self._playhead_lock:
                latest_allowed = (
                    self.client_playhead_frame
                    + self.max_motion_buffer_frames
                    - 1
                )
            if global_frame_idx <= latest_allowed:
                return True
            self.stop_event.wait(0.02)
        return False

    def submit_motion_chunk(self, x_d_info_list):
        """Submit a generated motion chunk directly to the renderer."""
        for x_d_info in x_d_info_list:
            ctrl_kwargs = self._get_ctrl_info(self.gen_frame_idx)
            while not self.stop_event.is_set():
                try:
                    self.motion_stitch_queue.put(
                        [self.gen_frame_idx, x_d_info, ctrl_kwargs], timeout=1
                    )
                    break
                except queue.Full:
                    continue
            self.gen_frame_idx += 1

    def set_ditto_pose_mode(self, enabled):
        self.ditto_pose_enabled = bool(enabled)
        self.lip_motion_queue = queue.Queue()
        self._last_lip_motion = None
        self._lip_stream_finished = False

    def submit_lip_chunk(self, lip_motion_chunk):
        """Queue generated lip frames for merging into Ditto pose output."""
        for lip_motion in lip_motion_chunk:
            if self.stop_event.is_set():
                return False
            self.lip_motion_queue.put(np.asarray(lip_motion, dtype=np.float32))
        return True

    def finish_lip(self):
        self.lip_motion_queue.put(None)

    def finish_motion(self):
        if not self._motion_input_finished:
            self.motion_stitch_queue.put(None)
            self._motion_input_finished = True

    def abort_current(self):
        """Stop workers and discard every queued frame for the active request."""
        self.stop_event.set()
        for work_queue in (
            self.audio2motion_queue,
            self.lip_motion_queue,
            self.motion_stitch_queue,
            self.warp_f3d_queue,
            self.decode_f3d_queue,
            self.putback_queue,
            self.writer_queue,
            self.frame_queue,
        ):
            while True:
                try:
                    work_queue.get_nowait()
                except queue.Empty:
                    break

    def close(self):
        """Drain worker queues once; websocket cleanup may call close twice."""
        if self._closed:
            return
        try:
            super().close()
        finally:
            self._closed = True

    def _apply_switch(self, name):
        if name not in self.source_registry:
            return
        context = self.source_registry[name]
        info = context["source_info"]
        print(
            f"[Switch] frame {self.output_frame_idx}: "
            f"'{self.active_source_name}' → '{name}'"
        )
        self.source_info = info
        self.source_info_frames = len(info["x_s_info_lst"])
        self.active_source_name = name
        self.active_lip_normalize_delta_kp = context.get(
            "lip_normalize_delta_kp"
        )
        self.active_lip_normalize_delta_kp_sequence = context.get(
            "lip_normalize_delta_kp_sequence"
        )
        if not self.ditto_pose_enabled:
            self.condition_handler.setup(
                info,
                self.emo,
                eye_f0_mode=self.eye_f0_mode,
                ch_info=self.ch_info,
            )
        motion_stitch = self.motion_stitch
        while hasattr(motion_stitch, "_base"):
            motion_stitch = motion_stitch._base
        motion_stitch.x_s = None
        motion_stitch.pose_s = None
        motion_stitch.source_exp0 = None
        motion_stitch.delta_exp_lip0 = None
        motion_stitch.is_image_flag = info["is_image_flag"]

    def _source_frame_sample(self, source_motion_position, source_info):
        from runtime.core.atomic_components.condition_handler import _mirror_index

        source_frame_count = len(source_info["x_s_info_lst"])
        source_motion_position = max(0.0, float(source_motion_position))
        source_motion_idx = max(
            0,
            int(np.floor(source_motion_position + 1e-9)),
        )
        alpha = min(
            1.0,
            max(0.0, source_motion_position - source_motion_idx),
        )
        return (
            _mirror_index(source_motion_idx, source_frame_count),
            _mirror_index(source_motion_idx + 1, source_frame_count),
            alpha,
        )

    def _should_render_frame(self, global_frame_idx):
        """Select output frames when preview FPS is lower than render FPS."""
        with self._playhead_lock:
            initial_buffer_edge = self.max_motion_buffer_frames - 1
            force_initial_edge = not self.playhead_has_advanced
        if force_initial_edge and global_frame_idx == initial_buffer_edge:
            return True
        if self.render_fps >= self.output_fps:
            return True
        if global_frame_idx == 0:
            return True
        if self.N_d > 0 and global_frame_idx >= self.N_d - 1:
            return True
        ratio = self.render_fps / self.output_fps
        current_slot = int(global_frame_idx * ratio)
        previous_slot = int((global_frame_idx - 1) * ratio)
        return current_slot > previous_slot

    def _render_resampled_motion(
        self,
        output_frame_idx,
        source_motion_position,
        x_d_info,
        ctrl_kwargs,
    ):
        if self.N_d >= 0 and output_frame_idx >= self.N_d:
            return True
        if not self._wait_for_motion_window(output_frame_idx):
            return False

        switch_applied = False
        with self._switch_lock:
            if self.pending_switch is not None:
                self._apply_switch(self.pending_switch)
                self.pending_switch = None
                switch_applied = True

        if (
            not switch_applied
            and not self._should_render_frame(output_frame_idx)
        ):
            return True

        source_info = self.source_info
        frame_idx, next_frame_idx, source_alpha = self._source_frame_sample(
            source_motion_position,
            source_info,
        )
        self.current_source_frame_idx = frame_idx
        self.current_source_next_frame_idx = next_frame_idx
        self.current_source_frame_alpha = source_alpha
        self.output_frame_idx = output_frame_idx
        x_s_info = source_info["x_s_info_lst"][frame_idx]
        if source_alpha > 1e-9 and next_frame_idx != frame_idx:
            x_s_info = _interpolate_motion_info(
                x_s_info,
                source_info["x_s_info_lst"][next_frame_idx],
                source_alpha,
            )
        x_s, x_d = self.motion_stitch(
            x_s_info,
            x_d_info,
            **ctrl_kwargs,
        )
        self.warp_f3d_queue.put(
            [
                output_frame_idx,
                frame_idx,
                x_s,
                x_d,
                source_info,
            ]
        )
        return True

    def _motion_stitch_worker(self):
        converter = MotionFrameRateConverter(
            output_fps=self.output_fps,
            input_fps=self.motion_input_fps,
        )
        while not self.stop_event.is_set():
            try:
                item = self.motion_stitch_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                for output in converter.finish():
                    if not self._render_resampled_motion(*output):
                        return
                self.warp_f3d_queue.put(None)
                break

            _, x_d_info, ctrl_kwargs = item
            for output in converter.push(x_d_info, ctrl_kwargs):
                if not self._render_resampled_motion(*output):
                    return

    def _warp_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.warp_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.decode_f3d_queue.put(None)
                break
            global_frame_idx, frame_idx, x_s, x_d, source_info = item
            f_s = source_info["f_s_lst"][frame_idx]
            f_3d = self.warp_f3d(f_s, x_s, x_d)
            self.decode_f3d_queue.put(
                [global_frame_idx, frame_idx, f_3d, source_info]
            )

    def _decode_f3d_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.decode_f3d_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.putback_queue.put(None)
                break
            global_frame_idx, frame_idx, f_3d, source_info = item
            render_img = self.decode_f3d(f_3d)
            self.putback_queue.put(
                [global_frame_idx, frame_idx, render_img, source_info]
            )

    def setup_cached(self, source_info, output_path, **kwargs):
        """Like setup() but uses cached source_info (no face detection).

        Call this instead of setup() when you have a pre-computed source_info
        from a previous run or from the avatar cache.
        """
        from runtime.core.atomic_components.avatar_registrar import smooth_x_s_info_lst
        kwargs = self._merge_kwargs(self.default_kwargs, kwargs)

        self.max_size = kwargs.get("max_size", 1920)
        self.template_n_frames = kwargs.get("template_n_frames", -1)
        self.crop_scale = kwargs.get("crop_scale", 2.3)
        self.crop_vx_ratio = kwargs.get("crop_vx_ratio", 0)
        self.crop_vy_ratio = kwargs.get("crop_vy_ratio", -0.125)
        self.crop_flag_do_rot = kwargs.get("crop_flag_do_rot", True)
        self.smo_k_s = kwargs.get('smo_k_s', 13)
        self.emo = kwargs.get("emo", 4)
        self.eye_f0_mode = kwargs.get("eye_f0_mode", False)
        self.ch_info = kwargs.get("ch_info", None)
        self.overlap_v2 = kwargs.get("overlap_v2", 10)
        self.fix_kp_cond = kwargs.get("fix_kp_cond", 0)
        self.fix_kp_cond_dim = kwargs.get("fix_kp_cond_dim", None)
        self.sampling_timesteps = kwargs.get("sampling_timesteps", 50)
        self.online_mode = kwargs.get("online_mode", False)
        self.v_min_max_for_clip = kwargs.get('v_min_max_for_clip', None)
        self.smo_k_d = kwargs.get("smo_k_d", 3)
        self.N_d = kwargs.get("N_d", -1)
        self.use_d_keys = kwargs.get("use_d_keys", None)
        self.relative_d = kwargs.get("relative_d", True)
        self.drive_eye = kwargs.get("drive_eye", None)
        self.delta_eye_arr = kwargs.get("delta_eye_arr", None)
        self.delta_eye_open_n = kwargs.get("delta_eye_open_n", 0)
        self.fade_type = kwargs.get("fade_type", "")
        self.fade_out_keys = kwargs.get("fade_out_keys", ("exp",))
        self.flag_stitching = kwargs.get("flag_stitching", True)
        self.ctrl_info = kwargs.get("ctrl_info", dict())
        self.overall_ctrl_info = kwargs.get("overall_ctrl_info", dict())
        assert self.wav2feat.support_streaming or not self.online_mode

        if len(source_info["x_s_info_lst"]) > 1 and self.smo_k_s > 1:
            source_info["x_s_info_lst"] = smooth_x_s_info_lst(source_info["x_s_info_lst"], smo_k=self.smo_k_s)
        self.source_info = source_info
        self.source_info_frames = len(source_info["x_s_info_lst"])

        self.condition_handler.setup(source_info, self.emo, eye_f0_mode=self.eye_f0_mode, ch_info=self.ch_info)

        x_s_info_0 = self.condition_handler.x_s_info_0
        self.audio2motion.setup(
            x_s_info_0, overlap_v2=self.overlap_v2, fix_kp_cond=self.fix_kp_cond,
            fix_kp_cond_dim=self.fix_kp_cond_dim, sampling_timesteps=self.sampling_timesteps,
            online_mode=self.online_mode, v_min_max_for_clip=self.v_min_max_for_clip,
            smo_k_d=self.smo_k_d,
        )

        is_image_flag = source_info["is_image_flag"]
        x_s_info = source_info['x_s_info_lst'][0]
        self.motion_stitch.setup(
            N_d=self.N_d, use_d_keys=self.use_d_keys, relative_d=self.relative_d,
            drive_eye=self.drive_eye, delta_eye_arr=self.delta_eye_arr,
            delta_eye_open_n=self.delta_eye_open_n, fade_out_keys=self.fade_out_keys,
            fade_type=self.fade_type, flag_stitching=self.flag_stitching,
            is_image_flag=is_image_flag, x_s_info=x_s_info, d0=None,
            ch_info=self.ch_info, overall_ctrl_info=self.overall_ctrl_info,
        )

        self.output_path = output_path
        self.tmp_output_path = output_path + ".tmp.mp4"
        self.writer = _NullWriter()
        self.writer_pbar = _NullProgress()

        if self.online_mode:
            self.audio_feat = self.wav2feat.wav2feat(np.zeros((self.overlap_v2 * 640,), dtype=np.float32), sr=16000)
            assert len(self.audio_feat) == self.overlap_v2, f"{len(self.audio_feat)}"
        else:
            self.audio_feat = np.zeros((0, self.wav2feat.feat_dim), dtype=np.float32)
        self.cond_idx_start = 0 - len(self.audio_feat)

        QUEUE_MAX = PIPELINE_QUEUE_SIZE
        self.worker_exception = None
        self.stop_event = threading.Event()
        self.audio2motion_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.motion_stitch_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.warp_f3d_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.decode_f3d_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.putback_queue = queue.Queue(maxsize=QUEUE_MAX)
        self.writer_queue = queue.Queue(maxsize=QUEUE_MAX)

        self.thread_list = [
            threading.Thread(target=self.audio2motion_worker),
            threading.Thread(target=self.motion_stitch_worker),
            threading.Thread(target=self.warp_f3d_worker),
            threading.Thread(target=self.decode_f3d_worker),
            threading.Thread(target=self.putback_worker),
            threading.Thread(target=self.writer_worker),
        ]
        for thread in self.thread_list:
            thread.start()

    def _writer_worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.writer_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                return

    def _putback_worker(self):
        try:
            self.__putback_worker_impl()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
            traceback.print_exc()
        finally:
            # The async sender exits only after every already-encoded frame has
            # been consumed.  An Event avoids a blocking sentinel insertion
            # when the browser disconnects while the queue is full.
            self.frame_stream_finished.set()

    def __putback_worker_impl(self):
        import cv2
        while not self.stop_event.is_set():
            try:
                item = self.putback_queue.get(timeout=1)
            except queue.Empty:
                continue

            if item is None:
                self.writer_queue.put(None)
                break

            global_frame_idx, frame_idx, render_img, source_info = item
            putback_started = time.perf_counter()
            if self.paste_back:
                frame_rgb = source_info["img_rgb_lst"][frame_idx]
                M_c2o = source_info["M_c2o_lst"][frame_idx]
                res_frame_rgb = self.putback(frame_rgb, render_img, M_c2o)
            else:
                res_frame_rgb = np.clip(render_img, 0, 255).astype(np.uint8)
            self.putback_seconds += time.perf_counter() - putback_started

            # JPEG remains next to putback so the queue stores compact bytes
            # rather than multi-megabyte RGB arrays.  Network I/O is performed
            # by a separate async consumer and can no longer block this worker.
            jpeg_started = time.perf_counter()
            ok, jpg = cv2.imencode(
                ".jpg",
                cv2.cvtColor(res_frame_rgb, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 80],
            )
            self.jpeg_seconds += time.perf_counter() - jpeg_started
            if ok:
                self._queue_encoded_frame(global_frame_idx, jpg.tobytes())
            self.encoded_frame_count += 1
            self._frame_seq += 1
            self.last_output_frame_idx = global_frame_idx


    def _audio2motion_worker(self):
        try:
            if self.ditto_pose_enabled:
                self.__ditto_pose_worker_impl()
            else:
                self.__disabled_audio2motion_worker_impl()
        except Exception as e:
            self.worker_exception = e
            self.stop_event.set()
            traceback.print_exc()

    def __disabled_audio2motion_worker_impl(self):
        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                self.finish_motion()
                return

    @staticmethod
    def _replace_ditto_lip(x_d_info, lip_motion):
        lip_motion = np.asarray(lip_motion, dtype=np.float32)
        if lip_motion.size == len(LIP_IDX) * 3:
            lip_points = lip_motion.reshape(1, len(LIP_IDX), 3)
        elif lip_motion.size == 21 * 3:
            lip_points = lip_motion.reshape(1, 21, 3)[:, LIP_IDX, :]
        else:
            raise ValueError(
                f"Expected 18D or 63D lip motion, got shape {lip_motion.shape}"
            )

        merged = dict(x_d_info)
        exp = np.asarray(x_d_info["exp"], dtype=np.float32).reshape(1, 21, 3).copy()
        exp[:, LIP_IDX, :] = lip_points
        merged["exp"] = exp.reshape(1, 63)
        return merged

    def _next_lip_motion(self):
        while not self.stop_event.is_set():
            if self._lip_stream_finished:
                return self._last_lip_motion
            try:
                lip_motion = self.lip_motion_queue.get(timeout=1)
            except queue.Empty:
                continue
            if lip_motion is None:
                self._lip_stream_finished = True
                return self._last_lip_motion
            self._last_lip_motion = lip_motion
            return lip_motion
        return None

    def __ditto_pose_worker_impl(self):
        is_end = False
        seq_frames = self.audio2motion.seq_frames
        valid_clip_len = self.audio2motion.valid_clip_len
        aud_feat_dim = self.wav2feat.feat_dim
        item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

        res_kp_seq = None
        res_kp_seq_valid_start = None

        global_idx = 0
        local_idx = 0
        gen_frame_idx = 0
        self.gen_frame_idx = 0

        while not self.stop_event.is_set():
            try:
                item = self.audio2motion_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                is_end = True
            else:
                item_buffer = np.concatenate([item_buffer, item], 0)

            if not is_end and item_buffer.shape[0] < valid_clip_len:
                continue
            else:
                self.audio_feat = np.concatenate([self.audio_feat, item_buffer], 0)
                item_buffer = np.zeros((0, aud_feat_dim), dtype=np.float32)

            while True:
                aud_feat = self.audio_feat[local_idx: local_idx + seq_frames]
                real_valid_len = valid_clip_len
                if len(aud_feat) == 0:
                    break
                elif len(aud_feat) < seq_frames:
                    if not is_end:
                        break
                    else:
                        real_valid_len = len(aud_feat)
                        pad = np.stack([aud_feat[-1]] * (seq_frames - len(aud_feat)), 0)
                        aud_feat = np.concatenate([aud_feat, pad], 0)

                aud_cond = self.condition_handler(aud_feat, global_idx + self.cond_idx_start)[None]
                res_kp_seq = self.audio2motion(aud_cond, res_kp_seq)

                if res_kp_seq_valid_start is None:
                    res_kp_seq_valid_start = res_kp_seq.shape[1] - self.audio2motion.fuse_length
                    d0 = self.audio2motion.cvt_fmt(res_kp_seq[0:1])[0]
                    self.motion_stitch.d0 = d0
                    local_idx += real_valid_len
                    global_idx += real_valid_len
                    continue
                else:
                    valid_res_kp_seq = res_kp_seq[:, res_kp_seq_valid_start: res_kp_seq_valid_start + real_valid_len]
                    x_d_info_list = self.audio2motion.cvt_fmt(valid_res_kp_seq)

                    for x_d_info in x_d_info_list:
                        if self.ditto_pose_enabled:
                            lip_motion = self._next_lip_motion()
                            if lip_motion is not None:
                                x_d_info = self._replace_ditto_lip(
                                    x_d_info,
                                    lip_motion,
                                )
                        ctrl_kwargs = self._get_ctrl_info(gen_frame_idx)

                        while not self.stop_event.is_set():
                            try:
                                self.motion_stitch_queue.put(
                                    [gen_frame_idx, x_d_info, ctrl_kwargs],
                                    timeout=1,
                                )
                                break
                            except queue.Full:
                                continue

                        gen_frame_idx += 1
                        self.gen_frame_idx = gen_frame_idx

                    res_kp_seq_valid_start += real_valid_len
                    local_idx += real_valid_len
                    global_idx += real_valid_len

                L = res_kp_seq.shape[1]
                if L > seq_frames * 2:
                    cut_L = L - seq_frames * 2
                    res_kp_seq = res_kp_seq[:, cut_L:]
                    res_kp_seq_valid_start -= cut_L

                if local_idx >= len(self.audio_feat):
                    break

            L = len(self.audio_feat)
            if L > seq_frames * 2:
                cut_L = L - seq_frames * 2
                self.audio_feat = self.audio_feat[cut_L:]
                local_idx -= cut_L

            if is_end:
                break

        self.motion_stitch_queue.put(None)
