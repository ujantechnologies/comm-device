from pathlib import Path


class TtsService:
    def __init__(self, voice_model_path: str) -> None:
        self.voice_model_path = voice_model_path

    def synthesize(self, text: str, output_path: str) -> str:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Placeholder artifact to validate flow before piper integration.
        path.write_text(text + "\n", encoding="utf-8")
        return str(path)
