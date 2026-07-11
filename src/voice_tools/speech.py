from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

import soundfile as sf
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    PermissionDeniedError,
    RateLimitError,
)

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ENV_PATH = PACKAGE_ROOT / ".env"
SAMPLE_RATE = 16_000


class VoiceToolsError(RuntimeError):
    """Base error for voice-tools failures."""


class MissingApiKeyError(VoiceToolsError):
    """Raised when OPENAI_API_KEY is not configured."""


class MissingParecordError(VoiceToolsError):
    """Raised when parecord is not available on PATH."""


class RecordingFailedError(VoiceToolsError):
    """Raised when parecord cannot produce a valid WAV recording."""


class EmptyAudioError(VoiceToolsError):
    """Raised when the recording contains no audio samples."""


class RecordingInterruptedError(VoiceToolsError):
    """Raised when the user interrupts recording with Ctrl+C."""


class OpenAIAuthenticationFailedError(VoiceToolsError):
    """Raised when OpenAI rejects the configured API key."""


class OpenAIPermissionDeniedError(VoiceToolsError):
    """Raised when OpenAI denies access to transcription."""


class OpenAIRateLimitError(VoiceToolsError):
    """Raised when OpenAI rate limits the request."""


class OpenAIConnectionError(VoiceToolsError):
    """Raised when the OpenAI API cannot be reached."""


class OpenAIRequestError(VoiceToolsError):
    """Raised when OpenAI rejects the transcription request."""


class OpenAIServiceError(VoiceToolsError):
    """Raised when OpenAI returns an unexpected API error."""


def _load_installation_env() -> None:
    load_dotenv(ENV_PATH)


def _require_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise MissingApiKeyError(
            "OPENAI_API_KEY is missing. Add it to the voice-tools .env file before running transcription."
        )
    return api_key


def _require_parecord() -> str:
    parecord = shutil.which("parecord")
    if not parecord:
        raise MissingParecordError(
            "parecord is not installed or not on PATH. Install pulseaudio-utils to provide parecord."
        )
    return parecord


def _terminate_process(proc: subprocess.Popen[bytes], timeout: float = 3.0) -> None:
    if proc.poll() is not None:
        return

    try:
        proc.terminate()
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                proc.send_signal(signal.SIGKILL)
            else:
                proc.kill()
        finally:
            proc.wait()


def _record_audio(audio_path: Path) -> None:
    parecord = _require_parecord()

    try:
        proc = subprocess.Popen(
            [
                parecord,
                "--channels=1",
                "--rate=16000",
                "--format=s16le",
                "--file-format=wav",
                str(audio_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RecordingFailedError(f"Failed to start parecord: {exc}") from exc

    try:
        input()
    except KeyboardInterrupt as exc:
        raise RecordingInterruptedError("Recording was interrupted before it finished.") from exc
    finally:
        _terminate_process(proc)

    stderr = b""
    if proc.stderr is not None:
        stderr = proc.stderr.read() or b""

    if proc.returncode not in (0, None):
        detail = stderr.decode("utf-8", "replace").strip()
        message = "Recording failed."
        if detail:
            message = f"{message} {detail}"
        raise RecordingFailedError(message)


def _verify_audio(audio_path: Path) -> None:
    if not audio_path.exists():
        raise RecordingFailedError("Recording failed: the WAV file was not created.")

    if audio_path.stat().st_size == 0:
        raise EmptyAudioError("Recording failed: the WAV file is empty.")

    try:
        with sf.SoundFile(audio_path) as audio:
            if len(audio) == 0:
                raise EmptyAudioError("Recording failed: the WAV file contains no samples.")
    except VoiceToolsError:
        raise
    except Exception as exc:
        raise RecordingFailedError(f"Recording verification failed: {exc}") from exc


def _transcribe_audio(audio_path: Path) -> str:
    client = OpenAI()

    try:
        with audio_path.open("rb") as audio:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio,
            )
    except AuthenticationError as exc:
        raise OpenAIAuthenticationFailedError(
            "OpenAI authentication failed. Check that OPENAI_API_KEY is valid."
        ) from exc
    except PermissionDeniedError as exc:
        raise OpenAIPermissionDeniedError(
            "OpenAI denied access to the transcription request."
        ) from exc
    except RateLimitError as exc:
        raise OpenAIRateLimitError("OpenAI rate limit reached. Try again later.") from exc
    except BadRequestError as exc:
        raise OpenAIRequestError("OpenAI rejected the transcription request.") from exc
    except APIConnectionError as exc:
        raise OpenAIConnectionError(
            "Could not reach the OpenAI API. Check your network connection."
        ) from exc
    except APIError as exc:
        raise OpenAIServiceError("OpenAI returned an API error during transcription.") from exc
    except Exception as exc:
        raise VoiceToolsError(f"Unexpected OpenAI error: {exc}") from exc

    text = getattr(result, "text", "")
    if not isinstance(text, str):
        raise VoiceToolsError("OpenAI returned an invalid transcription payload.")

    return text.strip()


def transcribe_from_microphone() -> str:
    """Record microphone audio and return the transcribed text."""
    _load_installation_env()
    _require_api_key()

    with tempfile.TemporaryDirectory(prefix="voice-tools-") as tempdir:
        audio_path = Path(tempdir) / "recording.wav"
        try:
            _record_audio(audio_path)
            _verify_audio(audio_path)
            return _transcribe_audio(audio_path)
        finally:
            if audio_path.exists():
                audio_path.unlink()
