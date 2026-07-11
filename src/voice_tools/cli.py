from __future__ import annotations

import sys

from .speech import VoiceToolsError, transcribe_from_microphone


def main() -> int:
    print("Recording audio. Press Enter to stop.", file=sys.stderr, flush=True)
    print("Transcribing audio.", file=sys.stderr, flush=True)

    try:
        text = transcribe_from_microphone()
    except VoiceToolsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(text, end="" if text.endswith("\n") else "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

