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

    @property
    def is_local_model_ready(self) -> bool:
        return self._llm is not None

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

    def generate_question(self, history: list[tuple[str, str]]) -> str:
        """Generate the next yes/no question to ask the user.

        Args:
            history: List of (question, answer) pairs from the current session.

        Returns:
            A short yes/no question string.
        """
        fallbacks = [
            "Are you comfortable right now?",
            "Are you in any pain?",
            "Do you need something to drink?",
            "Would you like to rest?",
            "Do you need help with something?",
            "Are you feeling okay?",
            "Would you like to change position?",
            "Are you too hot?",
            "Are you too cold?",
            "Would you like music?",
            "Do you want to go outside?",
        ]
        if self._llm is None:
            return fallbacks[len(history) % len(fallbacks)]

        if history:
            history_text = "Previous exchanges:\n" + "\n".join(
                f"Q: {q}\nA: {a}" for q, a in history[-5:]
            ) + "\n\n"
        else:
            history_text = ""

        prompt = (
            "<start_of_turn>user\n"
            "You are an AAC device assistant helping a non-verbal person communicate "
            "using yes/no head gestures.\n"
            f"{history_text}"
            "Ask exactly one short, clear yes/no question that can be answered by nodding yes or no. "
            "Use a simple question beginning with Do, Does, Are, Is, Can, Would, or Will. "
            "Do not repeat recent questions. Reply with only the question itself.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        try:
            out = self._llm(
                prompt,
                max_tokens=48,
                stop=["<end_of_turn>", "\n"],
                temperature=0.95,
            )
            text = out["choices"][0]["text"].strip()
            if text and not text.endswith("?"):
                text += "?"
            if text:
                recent = {q.strip().lower() for q, _ in history[-6:]}
                if text.strip().lower() not in recent:
                    return text
            return fallbacks[len(history) % len(fallbacks)]
        except Exception as exc:
            logger.error("Question generation error: %s", exc)
            return fallbacks[len(history) % len(fallbacks)]

    def generate_training_question(
        self,
        history: list[tuple[str, str]],
        target_intent: str,
        temperature: float = 0.65,
    ) -> tuple[bool, str]:
        """Generate one training question that is likely to elicit a target intent.

        This path is strict-local only. If the local GGUF model is unavailable,
        it returns a failure status instead of using fallback prompts.
        """
        if self._llm is None:
            return (
                False,
                "Training question unavailable: local LLM model is not loaded.",
            )

        if history:
            history_text = "Recent training prompts:\n" + "\n".join(
                f"Q: {q} | Target: {a}" for q, a in history[-6:]
            ) + "\n\n"
        else:
            history_text = ""

        prompt = (
            "<start_of_turn>user\n"
            "You are helping a 10 year old practice communication with sign labels. "
            "Generate exactly one short spoken question that is very likely to produce "
            f"the target response intent '{target_intent}'.\n"
            f"{history_text}"
            "Rules:\n"
            "- Use simple words for a 10 year old.\n"
            "- Ask one concrete everyday question.\n"
            "- Keep it under 12 words.\n"
            "- End with a question mark.\n"
            "- Do not include extra explanation or multiple questions.\n"
            "Reply with only the question text.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )

        try:
            out = self._llm(
                prompt,
                max_tokens=48,
                stop=["<end_of_turn>", "\n"],
                temperature=max(0.0, min(1.2, temperature)),
            )
            text = out["choices"][0]["text"].strip()
            if text and not text.endswith("?"):
                text += "?"

            if not text:
                return False, "Training question generation failed: empty model output."

            if len(text.split()) > 14:
                return False, "Training question generation failed: output too long."

            recent = {q.strip().lower() for q, _ in history[-8:]}
            if text.strip().lower() in recent:
                return False, "Training question generation failed: duplicate question."

            return True, text
        except Exception as exc:
            logger.error("Training question generation error: %s", exc)
            return False, "Training question generation failed: local LLM inference error."

    def generate_intent_response(self, question: str, intent: str) -> str:
        """Turn recognized intent into a spoken response for a heard question."""
        fallback = f"I understood the response intent as {intent}."
        if self._llm is None:
            return fallback

        prompt = (
            "<start_of_turn>user\n"
            "You are an AAC communication assistant. "
            "A caregiver question was transcribed and the user's ASL response "
            "was recognized as an intent label.\n"
            f"Question: {question}\n"
            f"Intent label: {intent}\n"
            "Speak one short clear sentence in first person as the user's answer. "
            "Do not mention model uncertainty or internal labels.\n"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        try:
            out = self._llm(
                prompt,
                max_tokens=64,
                stop=["<end_of_turn>", "\n"],
                temperature=0.5,
            )
            text = out["choices"][0]["text"].strip()
            return text if text else fallback
        except Exception as exc:
            logger.error("Intent response generation error: %s", exc)
            return fallback
