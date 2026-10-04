"""LLM, STT, and TTS services for the TalkLikeYou conversation demo."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import threading
import uuid
import wave

import requests


DEFAULT_LLM_MODELS = ("qwen-flash", "qwen-plus", "qwen-turbo", "qwen-max")
DEFAULT_EDGE_VOICES = (
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
)
REQUIRED_EDGE_VOICES = ("zh-CN-YunxiNeural",)
DEFAULT_OPENAI_TTS_MODELS = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")
DEFAULT_OPENAI_VOICES = (
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "nova",
    "onyx",
    "sage",
    "shimmer",
    "verse",
)


def _load_optional_env_file() -> None:
    """Load an optional local configuration file."""
    configured = os.environ.get("TALKLIKEYOU_ENV_FILE", "").strip()
    default_path = Path(__file__).resolve().parent / ".env"
    path = Path(configured).expanduser() if configured else default_path
    if not path.is_file():
        return

    existing_keys = set(os.environ)
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if value[:1] == value[-1:] and value.startswith(("'", '"')):
            value = value[1:-1]
        if key and key not in existing_keys:
            os.environ[key] = value


def _env(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return default


def _env_options(*names: str, defaults: tuple[str, ...]) -> list[str]:
    configured = _env(*names)
    values = configured.split(",") if configured else list(defaults)
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def _prepend_option(value: str, options: list[str]) -> list[str]:
    return list(dict.fromkeys([value, *options])) if value else options


def _api_url(base_url: str, suffix: str) -> str:
    base = base_url.rstrip("/")
    suffix = suffix.lstrip("/")
    if base.endswith("/v1") and suffix.startswith("v1/"):
        suffix = suffix[3:]
    return f"{base}/{suffix}"


class SentenceSplitter:
    """Turn streaming Chinese or English text into complete sentences."""

    _SPLIT_RE = re.compile(
        r"("
        r"[。！？][”’」』）》】〕〉）\]\"']*"
        r"|[.!?][”’」』）》】〕〉）\]\"']*(?:\s|$)"
        r")"
    )
    _SOFT_SPLIT_RE = re.compile(r"[，、；：,;:]\s*")
    _MIN_SOFT_CHARS = 12
    _MAX_CHUNK_CHARS = 48

    def __init__(self) -> None:
        self._buffer = ""

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        sentences: list[str] = []
        while match := self._SPLIT_RE.search(self._buffer):
            end = match.end()
            sentence = self._buffer[:end].strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[end:]
        while len(self._buffer) >= self._MIN_SOFT_CHARS:
            search_end = min(len(self._buffer), self._MAX_CHUNK_CHARS)
            matches = [
                match
                for match in self._SOFT_SPLIT_RE.finditer(
                    self._buffer,
                    0,
                    search_end,
                )
                if match.end() >= self._MIN_SOFT_CHARS
            ]
            if matches:
                end = matches[-1].end()
            elif len(self._buffer) >= self._MAX_CHUNK_CHARS:
                prefix = self._buffer[:self._MAX_CHUNK_CHARS]
                whitespace = prefix.rfind(" ")
                end = (
                    whitespace + 1
                    if whitespace >= self._MIN_SOFT_CHARS
                    else self._MAX_CHUNK_CHARS
                )
            else:
                break
            sentence = self._buffer[:end].strip()
            if sentence:
                sentences.append(sentence)
            self._buffer = self._buffer[end:]
        return sentences

    def flush(self) -> str | None:
        remainder = self._buffer.strip()
        self._buffer = ""
        return remainder or None


@dataclass
class ConversationAudioStream:
    stream_id: str
    queue: asyncio.Queue[bytes | None]
    attached: bool = False
    finished: bool = False


@dataclass(frozen=True)
class ConversationConfig:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    system_prompt: str
    tts_provider: str
    tts_voice: str
    tts_model: str
    tts_base_url: str
    tts_api_key: str
    stt_provider: str
    stt_base_url: str
    stt_api_key: str
    stt_model: str
    max_history_turns: int

    @classmethod
    def from_environment(cls) -> "ConversationConfig":
        _load_optional_env_file()
        llm_base_url = _env("TALKLIKEYOU_LLM_BASE_URL")
        llm_api_key = _env("TALKLIKEYOU_LLM_API_KEY")
        stt_base_url = _env("TALKLIKEYOU_STT_BASE_URL")
        stt_api_key = _env("TALKLIKEYOU_STT_API_KEY")
        return cls(
            llm_base_url=llm_base_url,
            llm_api_key=llm_api_key,
            llm_model=_env(
                "TALKLIKEYOU_LLM_MODEL",
                default="qwen-flash",
            ),
            system_prompt=_env(
                "TALKLIKEYOU_LLM_SYSTEM_PROMPT",
                default=(
                    "You are TalkLikeYou, a friendly digital human assistant. "
                    "Reply naturally and concisely in the user's language. "
                    "Use plain spoken text without Markdown."
                ),
            ),
            tts_provider=_env(
                "TALKLIKEYOU_TTS_PROVIDER",
                default="edge",
            ).lower(),
            tts_voice=_env(
                "TALKLIKEYOU_TTS_VOICE",
                default="zh-CN-XiaoxiaoNeural",
            ),
            tts_model=_env(
                "TALKLIKEYOU_TTS_MODEL",
                default="tts-1",
            ),
            tts_base_url=_env(
                "TALKLIKEYOU_TTS_BASE_URL",
                default=llm_base_url,
            ),
            tts_api_key=_env(
                "TALKLIKEYOU_TTS_API_KEY",
                default=llm_api_key,
            ),
            stt_provider=_env(
                "TALKLIKEYOU_STT_PROVIDER",
                default="openai_compatible",
            ).lower(),
            stt_base_url=stt_base_url,
            stt_api_key=stt_api_key,
            stt_model=_env(
                "TALKLIKEYOU_STT_MODEL",
                default="whisper-1",
            ),
            max_history_turns=max(
                1, int(_env("TALKLIKEYOU_MAX_HISTORY_TURNS", default="20"))
            ),
        )

    def public_status(self) -> dict[str, object]:
        edge_ready = bool(
            importlib.util.find_spec("edge_tts") or shutil.which("edge-tts")
        )
        tts_providers = []
        if edge_ready:
            tts_providers.append({"value": "edge", "label": "Edge TTS"})
        if self.tts_base_url:
            tts_providers.append({
                "value": "openai_compatible",
                "label": "OpenAI-compatible",
            })
        normalized_provider = (
            "openai_compatible"
            if self.tts_provider == "openai"
            else self.tts_provider
        )
        llm_models = _prepend_option(
            self.llm_model,
            _env_options(
                "TALKLIKEYOU_LLM_MODELS",
                defaults=DEFAULT_LLM_MODELS,
            ),
        )
        edge_voices = _env_options(
            "TALKLIKEYOU_EDGE_VOICES",
            defaults=DEFAULT_EDGE_VOICES,
        )
        edge_voices = list(
            dict.fromkeys([*edge_voices, *REQUIRED_EDGE_VOICES])
        )
        openai_models = _prepend_option(
            self.tts_model,
            _env_options(
                "TALKLIKEYOU_OPENAI_TTS_MODELS",
                "TALKLIKEYOU_TTS_MODELS",
                defaults=DEFAULT_OPENAI_TTS_MODELS,
            ),
        )
        openai_voices = _env_options(
            "TALKLIKEYOU_OPENAI_TTS_VOICES",
            "TALKLIKEYOU_TTS_VOICES",
            defaults=DEFAULT_OPENAI_VOICES,
        )
        if normalized_provider == "edge":
            edge_voices = _prepend_option(self.tts_voice, edge_voices)
        elif normalized_provider == "openai_compatible":
            openai_voices = _prepend_option(self.tts_voice, openai_voices)
        return {
            "llm_configured": bool(self.llm_base_url and self.llm_model),
            "llm_model": self.llm_model,
            "llm_models": llm_models,
            "tts_provider": normalized_provider,
            "tts_providers": tts_providers,
            "tts_voice": self.tts_voice,
            "tts_model": self.tts_model,
            "tts_models": {
                "edge": [],
                "openai_compatible": openai_models,
            },
            "tts_voices": {
                "edge": edge_voices,
                "openai_compatible": openai_voices,
            },
            "tts_configured": bool(tts_providers),
            "stt_provider": self.stt_provider,
            "stt_configured": bool(self.stt_base_url and self.stt_model),
        }


class ConversationEngine:
    """Keeps short session histories and produces audio artifacts per turn."""

    def __init__(
        self, output_dir: str | Path, config: ConversationConfig | None = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or ConversationConfig.from_environment()
        self._histories: dict[str, list[dict[str, str]]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._audio_streams: dict[str, ConversationAudioStream] = {}

    def public_status(self) -> dict[str, object]:
        return self.config.public_status()

    def clear(self, conversation_id: str) -> None:
        self._histories.pop(self._clean_conversation_id(conversation_id), None)

    def _create_audio_stream(self) -> ConversationAudioStream:
        stream = ConversationAudioStream(
            stream_id=uuid.uuid4().hex,
            queue=asyncio.Queue(),
        )
        self._audio_streams[stream.stream_id] = stream
        return stream

    async def iter_audio_stream(
        self,
        stream_id: str,
    ) -> AsyncIterator[bytes]:
        stream = self._audio_streams.get(stream_id)
        if stream is None:
            raise ValueError("Conversation audio stream was not found or expired")
        if stream.attached:
            raise ValueError("Conversation audio stream is already in use")
        stream.attached = True
        try:
            while True:
                pcm = await stream.queue.get()
                if pcm is None:
                    return
                yield pcm
        finally:
            if self._audio_streams.get(stream_id) is stream:
                self._audio_streams.pop(stream_id, None)

    async def _finish_audio_stream(
        self,
        stream: ConversationAudioStream,
    ) -> None:
        if stream.finished:
            return
        stream.finished = True
        await stream.queue.put(None)
        loop = asyncio.get_running_loop()

        def expire_if_unused() -> None:
            if (
                not stream.attached
                and self._audio_streams.get(stream.stream_id) is stream
            ):
                self._audio_streams.pop(stream.stream_id, None)

        loop.call_later(300, expire_if_unused)

    @staticmethod
    def _clean_conversation_id(conversation_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_.-]", "", conversation_id or "")[:96]
        return cleaned or "default"

    def _history_lock(self, conversation_id: str) -> asyncio.Lock:
        conversation_id = self._clean_conversation_id(conversation_id)
        lock = self._locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[conversation_id] = lock
        return lock

    async def stream_turn(
        self,
        *,
        conversation_id: str,
        user_text: str,
        system_prompt: str | None = None,
        model: str | None = None,
        tts_provider: str | None = None,
        tts_model: str | None = None,
        voice: str | None = None,
        streaming_tts: bool = True,
    ) -> AsyncIterator[dict[str, object]]:
        conversation_id = self._clean_conversation_id(conversation_id)
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("Message cannot be empty")
        if len(user_text) > 8000:
            raise ValueError("Message is too long")
        requested_model = (model or "").strip() or self.config.llm_model
        if len(requested_model) > 256:
            raise ValueError("Model name is too long")
        requested_tts_provider = (
            (tts_provider or "").strip().lower() or self.config.tts_provider
        )
        if requested_tts_provider == "openai":
            requested_tts_provider = "openai_compatible"
        if requested_tts_provider not in {"edge", "openai_compatible"}:
            raise ValueError(f"Unsupported TTS provider: {requested_tts_provider}")
        requested_tts_model = (
            (tts_model or "").strip() or self.config.tts_model
        )
        requested_voice = (voice or "").strip() or self.config.tts_voice
        if len(requested_tts_model) > 256:
            raise ValueError("TTS model name is too long")
        if len(requested_voice) > 256:
            raise ValueError("TTS voice name is too long")

        async with self._history_lock(conversation_id):
            history = list(self._histories.get(conversation_id, []))
            prompt = (system_prompt or "").strip() or self.config.system_prompt
            messages = [
                {"role": "system", "content": prompt},
                *history,
                {"role": "user", "content": user_text},
            ]
            yield {"type": "reply_started"}
            audio_stream = (
                self._create_audio_stream()
                if streaming_tts
                else None
            )

            queue: asyncio.Queue[tuple[str, object | None]] = asyncio.Queue()
            stop_event = threading.Event()
            loop = asyncio.get_running_loop()

            def emit(kind: str, value: object | None = None) -> None:
                if stop_event.is_set():
                    return
                try:
                    loop.call_soon_threadsafe(queue.put_nowait, (kind, value))
                except RuntimeError:
                    stop_event.set()

            llm_task = asyncio.create_task(
                asyncio.to_thread(
                    self._stream_llm_sync,
                    messages,
                    emit,
                    stop_event,
                    requested_model,
                )
            )
            sentence_queue: asyncio.Queue[str | None] = asyncio.Queue()
            tts_paths: list[Path] = []

            async def tts_worker() -> None:
                chunk_index = 0
                try:
                    while True:
                        sentence = await sentence_queue.get()
                        if sentence is None:
                            return
                        path = await self.synthesize(
                            sentence,
                            provider=requested_tts_provider,
                            model=requested_tts_model,
                            voice=requested_voice,
                        )
                        tts_paths.append(path)
                        if audio_stream is not None:
                            pcm, duration = await asyncio.to_thread(
                                _read_speech_pcm,
                                path,
                            )
                            await audio_stream.queue.put(pcm)
                            await queue.put((
                                "audio_chunk",
                                {
                                    "index": chunk_index,
                                    "text": sentence,
                                    "duration": duration,
                                },
                            ))
                        chunk_index += 1
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    await queue.put(("tts_error", str(exc)))
                finally:
                    if audio_stream is not None:
                        await self._finish_audio_stream(audio_stream)
                    await queue.put(("tts_done", None))

            tts_task = asyncio.create_task(tts_worker())
            parts: list[str] = []
            splitter = SentenceSplitter()
            tts_started = False
            llm_done = False
            tts_done = False
            assistant_text = ""
            try:
                if audio_stream is not None:
                    yield {
                        "type": "tts_stream_started",
                        "stream_id": audio_stream.stream_id,
                    }
                while not (llm_done and tts_done):
                    kind, value = await queue.get()
                    if kind == "delta":
                        piece = str(value or "")
                        parts.append(piece)
                        yield {"type": "delta", "text": piece}
                        for sentence in splitter.feed(piece):
                            if not tts_started:
                                tts_started = True
                                yield {"type": "tts_started"}
                            await sentence_queue.put(sentence)
                    elif kind == "error":
                        raise RuntimeError(str(value or "LLM request failed"))
                    elif kind == "done":
                        await llm_task
                        assistant_text = "".join(parts).strip()
                        if not assistant_text:
                            raise RuntimeError(
                                "The language model returned an empty response"
                            )
                        remainder = splitter.flush()
                        if remainder:
                            if not tts_started:
                                tts_started = True
                                yield {"type": "tts_started"}
                            await sentence_queue.put(remainder)
                        yield {
                            "type": "reply_finished",
                            "text": assistant_text,
                        }
                        await sentence_queue.put(None)
                        llm_done = True
                    elif kind == "audio_chunk":
                        chunk = dict(value or {})
                        yield {"type": "audio_chunk_ready", **chunk}
                    elif kind == "tts_error":
                        raise RuntimeError(str(value or "TTS request failed"))
                    elif kind == "tts_done":
                        tts_done = True
                await tts_task
                wav_path = await _merge_wav_fragments(tts_paths, self.output_dir)
            except BaseException:
                if not tts_task.done():
                    tts_task.cancel()
                await asyncio.gather(tts_task, return_exceptions=True)
                for path in tts_paths:
                    path.unlink(missing_ok=True)
                if audio_stream is not None:
                    await self._finish_audio_stream(audio_stream)
                raise
            finally:
                stop_event.set()
                if not llm_task.done():
                    llm_task.cancel()
                await asyncio.gather(llm_task, return_exceptions=True)
            history.extend(
                [
                    {"role": "user", "content": user_text},
                    {"role": "assistant", "content": assistant_text},
                ]
            )
            max_messages = self.config.max_history_turns * 2
            self._histories[conversation_id] = history[-max_messages:]
            yield {
                "type": "audio_ready",
                "path": str(wav_path),
                "url": f"/uploads/{wav_path.name}",
                "name": wav_path.name,
            }

    def _stream_llm_sync(
        self,
        messages: list[dict[str, str]],
        emit,
        stop_event: threading.Event,
        model: str,
    ) -> None:
        config = self.config
        if not config.llm_base_url:
            emit(
                "error",
                "LLM is not configured. Set TALKLIKEYOU_LLM_BASE_URL and "
                "TALKLIKEYOU_LLM_API_KEY.",
            )
            return

        headers = {"Content-Type": "application/json"}
        if config.llm_api_key:
            headers["Authorization"] = f"Bearer {config.llm_api_key}"
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        try:
            with requests.post(
                _api_url(config.llm_base_url, "chat/completions"),
                headers=headers,
                json=payload,
                stream=True,
                timeout=(30, 180),
            ) as response:
                response.raise_for_status()
                for raw_line in response.iter_lines():
                    if stop_event.is_set():
                        return
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    choices = chunk.get("choices") or []
                    if not choices:
                        continue
                    content = (choices[0].get("delta") or {}).get("content")
                    if content:
                        emit("delta", str(content))
            emit("done")
        except Exception as exc:
            emit("error", f"LLM request failed: {exc}")

    async def synthesize(
        self,
        text: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        voice: str | None = None,
    ) -> Path:
        safe_text = _sanitize_tts_text(text)
        if not safe_text:
            raise RuntimeError("No speakable text was produced")
        artifact_id = uuid.uuid4().hex
        source_path = self.output_dir / f"conversation_{artifact_id}.mp3"
        wav_path = self.output_dir / f"conversation_{artifact_id}.wav"
        provider = (provider or self.config.tts_provider).strip().lower()
        if provider == "openai":
            provider = "openai_compatible"
        selected_voice = (voice or "").strip() or self.config.tts_voice
        selected_model = (model or "").strip() or self.config.tts_model

        try:
            if provider == "edge":
                await self._synthesize_edge(
                    safe_text, source_path, selected_voice
                )
            elif provider == "openai_compatible":
                await asyncio.to_thread(
                    self._synthesize_openai_sync,
                    safe_text,
                    source_path,
                    selected_voice,
                    selected_model,
                )
            else:
                raise RuntimeError(
                    f"Unsupported TTS provider '{provider}'. "
                    "Use edge or openai_compatible."
                )
            await _convert_to_speech_wav(source_path, wav_path)
        finally:
            source_path.unlink(missing_ok=True)
        return wav_path

    async def _synthesize_edge(self, text: str, output_path: Path, voice: str) -> None:
        try:
            import edge_tts
        except ImportError:
            edge_tts = None

        if edge_tts is not None:
            await edge_tts.Communicate(text=text, voice=voice).save(str(output_path))
            return

        executable = shutil.which("edge-tts")
        if executable:
            process = await asyncio.create_subprocess_exec(
                executable,
                "--voice",
                voice,
                "--text",
                text,
                "--write-media",
                str(output_path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await process.communicate()
            except asyncio.CancelledError:
                process.kill()
                await process.wait()
                raise
            if process.returncode == 0:
                return
            raise RuntimeError(stderr.decode("utf-8", errors="replace").strip())

        raise RuntimeError(
            "Edge TTS is not installed. Run: pip install 'edge-tts==7.2.8'"
        )

    def _synthesize_openai_sync(
        self,
        text: str,
        output_path: Path,
        voice: str,
        model: str,
    ) -> None:
        config = self.config
        if not config.tts_base_url:
            raise RuntimeError("TALKLIKEYOU_TTS_BASE_URL is not configured")
        headers = {"Content-Type": "application/json"}
        if config.tts_api_key:
            headers["Authorization"] = f"Bearer {config.tts_api_key}"
        response = requests.post(
            _api_url(config.tts_base_url, "audio/speech"),
            headers=headers,
            json={
                "model": model,
                "voice": voice,
                "input": text,
                "response_format": "mp3",
            },
            timeout=(30, 180),
        )
        response.raise_for_status()
        output_path.write_bytes(response.content)

    async def transcribe(self, input_path: str | Path) -> str:
        config = self.config
        if config.stt_provider not in {"openai", "openai_compatible"}:
            raise RuntimeError(
                f"Unsupported STT provider '{config.stt_provider}'. "
                "Configure TALKLIKEYOU_STT_PROVIDER=openai_compatible."
            )
        if not config.stt_base_url:
            raise RuntimeError(
                "STT is not configured. Set TALKLIKEYOU_STT_BASE_URL, "
                "TALKLIKEYOU_STT_API_KEY, and TALKLIKEYOU_STT_MODEL."
            )

        wav_path = self.output_dir / f"recording_{uuid.uuid4().hex}.wav"
        try:
            await _convert_to_speech_wav(Path(input_path), wav_path)
            return await asyncio.to_thread(self._transcribe_openai_sync, wav_path)
        finally:
            wav_path.unlink(missing_ok=True)

    def _transcribe_openai_sync(self, wav_path: Path) -> str:
        config = self.config
        headers = {}
        if config.stt_api_key:
            headers["Authorization"] = f"Bearer {config.stt_api_key}"
        with wav_path.open("rb") as audio_file:
            response = requests.post(
                _api_url(config.stt_base_url, "audio/transcriptions"),
                headers=headers,
                data={"model": config.stt_model, "response_format": "json"},
                files={"file": (wav_path.name, audio_file, "audio/wav")},
                timeout=(30, 180),
            )
        response.raise_for_status()
        payload = response.json()
        text = str(payload.get("text") or "").strip()
        if not text:
            raise RuntimeError("The speech recognizer returned empty text")
        return text


def _sanitize_tts_text(text: str) -> str:
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    text = re.sub(r"(?m)^\s*[-*+]\s+", "", text)
    text = re.sub(r"[*_~]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _read_speech_pcm(path: Path) -> tuple[bytes, float]:
    with wave.open(str(path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        sample_rate = source.getframerate()
        frame_count = source.getnframes()
        if (channels, sample_width, sample_rate) != (1, 2, 16000):
            raise RuntimeError(
                "Conversation TTS must be 16 kHz mono 16-bit PCM"
            )
        pcm = source.readframes(frame_count)
    return pcm, frame_count / 16000.0


async def _merge_wav_fragments(paths: list[Path], output_dir: Path) -> Path:
    if not paths:
        raise RuntimeError("TTS did not produce any audio")
    if len(paths) == 1:
        return paths[0]

    output_path = output_dir / f"conversation_{uuid.uuid4().hex}.wav"

    def merge() -> None:
        expected_params = None
        with wave.open(str(output_path), "wb") as target:
            for path in paths:
                with wave.open(str(path), "rb") as source:
                    params = (
                        source.getnchannels(),
                        source.getsampwidth(),
                        source.getframerate(),
                    )
                    if expected_params is None:
                        expected_params = params
                        target.setnchannels(params[0])
                        target.setsampwidth(params[1])
                        target.setframerate(params[2])
                    elif params != expected_params:
                        raise RuntimeError(
                            "TTS fragments have incompatible WAV formats"
                        )
                    target.writeframes(source.readframes(source.getnframes()))

    try:
        await asyncio.to_thread(merge)
    except BaseException:
        output_path.unlink(missing_ok=True)
        raise
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
    return output_path


async def _convert_to_speech_wav(input_path: Path, output_path: Path) -> None:
    process = await asyncio.create_subprocess_exec(
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        "16000",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_path),
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await process.communicate()
    except asyncio.CancelledError:
        process.kill()
        await process.wait()
        output_path.unlink(missing_ok=True)
        raise
    if process.returncode != 0:
        raise RuntimeError(
            "Audio conversion failed: "
            + stderr.decode("utf-8", errors="replace").strip()
        )
