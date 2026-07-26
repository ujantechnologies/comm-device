from .expression import ExpressionResult


class DisplayService:
    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height

    def render(self, frame_id: int, result: ExpressionResult, response: str) -> None:
        print(
            f"frame={frame_id} size={self.width}x{self.height} "
            f"label={result.label.value} conf={result.confidence:.2f} response={response}"
        )
