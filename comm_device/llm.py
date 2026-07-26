from __future__ import annotations

import logging
import os
from typing import Optional

from .expression import ExpressionLabel

logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    _HAS_LLAMA = True
except ImportError:
    _HAS_LLAMA = False
    logger.warning("llama_cpp not available — using hardcoded fallback responses")

_FALLBACK: dict[ExpressionLabel, str] = {
    ExpressionLabel.YES: "Great, I understood your yes. Let's continue.",
    ExpressionLabel.NO: "Understood, we will pause and try another option.",
    ExpressionLabel.UNCERTAIN: "I am still confirming your response.",
}

# Gemma 3 chat template
_PROMPT = (
    "<start_of_turn>user\n"
    "You are a communication assistant for a person using an AAC device. "
    "The person expressed '{label}' via head gesture. "
    "Reply with exactly one short, warm, empathetic sentence.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


class LlmService:
    """Generates natural-language responses via Gemma 3 1B through llama-cpp-python.

    Falls back to hardcoded strings when the GGUF model is unavailable.
    """

    def __init__(self, model_path: str) -> None:
        self._llm: Optional[object] = None
        self._model_path = model_path

        if _HAS_LLAMA and os.path.exists(model_path):
            self._init_llm()
        elif not os.path.exists(model_path):
            logger.warning("GGUF model not found at %s; using fallback responses", model_path)

    def _init_llm(self) -> None:
        self._llm = Llama(
            model_path=self._model_path,
            n_ctx=256,
            n_threads=4,
            verbose=False,
        )
        logger.info("LLM loaded: %s", self._model_path)

    def generate_response(self, label: ExpressionLabel) -> str:
        if self._llm is None:
            return _FALLBACK[label]
        prompt = _PROMPT.format(label=label.value)
        try:
            out = self._llm(
                prompt,
                max_tokens=64,
                stop=["<end_of_turn>", "\n"],
                temperature=0.7,
            )
            text = out["choices"][0]["text"].strip()
            return text if text else _FALLBACK[label]
        except Exception as exc:
            logger.error("LLM inference error: %s", exc)
            return _FALLBACK[label]
