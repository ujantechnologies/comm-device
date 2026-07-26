from dataclasses import dataclass
from enum import Enum


class ExpressionLabel(str, Enum):
    YES = "yes"
    NO = "no"
    UNCERTAIN = "uncertain"


@dataclass
class ExpressionResult:
    label: ExpressionLabel
    confidence: float


class ExpressionService:
    def infer(self, frame_id: int) -> ExpressionResult:
        if frame_id % 40 == 0:
            return ExpressionResult(label=ExpressionLabel.NO, confidence=0.82)
        if frame_id % 20 == 0:
            return ExpressionResult(label=ExpressionLabel.YES, confidence=0.87)
        return ExpressionResult(label=ExpressionLabel.UNCERTAIN, confidence=0.40)
