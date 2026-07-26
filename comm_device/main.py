import time
from .audio_router import AudioRouter
from .camera import CameraService
from .config import load_config
from .data_store import DataStore
from .display import DisplayService
from .expression import ExpressionLabel, ExpressionService
from .feedback import FeedbackService
from .llm import LlmService
from .tts import TtsService


def run() -> None:
    config = load_config()
    camera = CameraService()
    expression = ExpressionService()
    llm = LlmService(config.model_path)
    tts = TtsService(config.voice_model_path)
    display = DisplayService(config.display_width, config.display_height)
    store = DataStore(config.db_path)
    feedback = FeedbackService()
    audio = AudioRouter()

    camera.start()
    try:
        for _ in range(60):
            frame = camera.read()
            if frame is None:
                time.sleep(0.02)
                continue

            result = expression.infer(frame.frame_id)
            display.render(frame.frame_id, result, "")

            if result.label in (ExpressionLabel.YES, ExpressionLabel.NO):
                response = llm.generate_response(result.label)
                store.save_prediction(frame.frame_id, result.label, result.confidence)
                store.save_response(result.label, response)
                event = feedback.collect(frame.frame_id, result.label)
                output = tts.synthesize(response, "artifacts/response.txt")
                audio.play(output)
                display.render(frame.frame_id, result, response)
                if event.corrected_label and event.corrected_label != event.predicted_label:
                    print("feedback correction recorded")

            time.sleep(0.05)
    finally:
        camera.stop()


if __name__ == "__main__":
    run()
