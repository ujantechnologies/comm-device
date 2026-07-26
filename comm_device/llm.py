from .expression import ExpressionLabel


class LlmService:
    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def generate_response(self, label: ExpressionLabel) -> str:
        if label == ExpressionLabel.YES:
            return "Great, I understood your yes. Let's continue."
        if label == ExpressionLabel.NO:
            return "Understood. We will pause and try another option."
        return "I am still confirming your response."
