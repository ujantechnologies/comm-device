from dataclasses import dataclass
from .expression import ExpressionLabel


@dataclass
class FeedbackEvent:
    frame_id: int
    predicted_label: ExpressionLabel
    corrected_label: ExpressionLabel | None


class FeedbackService:
    def collect(self, frame_id: int, predicted_label: ExpressionLabel) -> FeedbackEvent:
        # Placeholder policy: accept predictions during scaffold stage.
        return FeedbackEvent(
            frame_id=frame_id,
            predicted_label=predicted_label,
            corrected_label=predicted_label,
        )
