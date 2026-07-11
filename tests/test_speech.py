from __future__ import annotations

import io
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

import voice_tools.cli as cli
import voice_tools.speech as speech


class PersistentTemporaryDirectory:
    def __init__(self, path: Path):
        self.path = path

    def __enter__(self) -> str:
        self.path.mkdir(parents=True, exist_ok=True)
        return str(self.path)

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeProcess:
    def __init__(self, audio_path: Path, returncode: int = 0):
        self.audio_path = audio_path
        self.returncode = None
        self.stderr = io.BytesIO(b"")
        self._final_returncode = returncode
        self._write_wav()

    def _write_wav(self) -> None:
        with wave.open(str(self.audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(speech.SAMPLE_RATE)
            wav_file.writeframes(b"\x00\x00" * 160)

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = self._final_returncode

    def send_signal(self, sig):
        self.returncode = self._final_returncode

    def kill(self):
        self.returncode = self._final_returncode

    def wait(self, timeout=None):
        self.returncode = self._final_returncode
        return self.returncode


class FakeTranscriptions:
    def __init__(self, text: str):
        self.text = text

    def create(self, *, model, file):
        assert model == "whisper-1"
        return SimpleNamespace(text=self.text)


class FakeOpenAI:
    def __init__(self, text: str = "hello world"):
        self.audio = SimpleNamespace(transcriptions=FakeTranscriptions(text))


def test_transcribe_from_microphone_returns_text_and_cleans_up_temp_audio(
    monkeypatch, tmp_path
):
    tempdir = tmp_path / "recording-session"
    audio_path = tempdir / "recording.wav"
    calls = {}

    def fake_load_dotenv(path):
        calls["dotenv"] = Path(path)
        return True

    def fake_which(command):
        return "/usr/bin/parecord" if command == "parecord" else None

    def fake_input(prompt=None):
        return ""

    def fake_popen(args, **kwargs):
        calls["popen_args"] = args
        return FakeProcess(Path(args[-1]))

    monkeypatch.setattr(speech, "load_dotenv", fake_load_dotenv)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(speech.shutil, "which", fake_which)
    monkeypatch.setattr(speech.tempfile, "TemporaryDirectory", lambda prefix: PersistentTemporaryDirectory(tempdir))
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(speech.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(speech, "OpenAI", lambda: FakeOpenAI("transcribed text"))

    result = speech.transcribe_from_microphone()

    assert result == "transcribed text"
    assert calls["dotenv"] == speech.ENV_PATH
    assert calls["popen_args"][-1] == str(audio_path)
    assert not audio_path.exists()


def test_missing_api_key_raises_custom_error(monkeypatch):
    monkeypatch.setattr(speech, "load_dotenv", lambda path: True)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(speech.MissingApiKeyError):
        speech.transcribe_from_microphone()


def test_missing_parecord_raises_custom_error(monkeypatch):
    monkeypatch.setattr(speech, "load_dotenv", lambda path: True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(speech.shutil, "which", lambda command: None)

    with pytest.raises(speech.MissingParecordError):
        speech.transcribe_from_microphone()


def test_cli_writes_only_transcription_to_stdout(monkeypatch, capsys):
    monkeypatch.setattr(cli, "transcribe_from_microphone", lambda: "final transcription")

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out == "final transcription\n"
    assert "Recording audio." in captured.err
    assert "Transcribing audio." in captured.err


def test_cli_returns_nonzero_on_failure(monkeypatch, capsys):
    def failing_transcription():
        raise speech.VoiceToolsError("boom")

    monkeypatch.setattr(cli, "transcribe_from_microphone", failing_transcription)

    exit_code = cli.main()
    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Error: boom" in captured.err
