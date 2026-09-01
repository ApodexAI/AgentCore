"""Guard rails around the LLM language detector's Chinese classifications.

``_validate_detected_language`` exists because the detector occasionally
answers "simplified Chinese" for text with no Chinese in it (full-width
punctuation is the usual trigger). It must reject those *without* rejecting
a genuine request for a Chinese answer — including one phrased in a third
language, where neither Han characters nor English keywords appear.
"""

from __future__ import annotations

import pytest

from agent_core.utils.language import _validate_detected_language


@pytest.mark.parametrize(
    "question",
    [
        "Please respond in Chinese.",
        "Answer in simplified Chinese",
        "Use Chinese for the final report",
        "translate to Chinese",
        "reply in zh-TW",
        "语言: zh-CN",
        "请解释这个机制。",
        "用中文回答",
        "Responde en chino, por favor.",
        "Réponds en chinois s'il te plaît",
        "Bitte auf Chinesisch antworten",
        "Rispondi in cinese",
        "Responda em chinês",
        "Ответь на китайском языке",
        "Отвечай по-китайски",
        "중국어로 답해줘",
        "Trả lời bằng tiếng Trung",
    ],
)
def test_explicit_chinese_requests_survive_validation(question: str) -> None:
    assert _validate_detected_language(question, "Simplified Chinese") == (
        "Simplified Chinese"
    )


@pytest.mark.parametrize(
    "question",
    [
        # Full-width punctuation is not language evidence.
        "Why is this program necessary？",
        "answer the query： why is this necessary?",
        # "Chinese" as a TOPIC, not as a requested output language — the
        # exact class of English question the detector over-triggers on.
        "What did the Chinese government announce about export controls?",
        "Summarize this Chinese paper for me",
        "When is Chinese New Year 2027?",
    ],
)
def test_unsupported_chinese_classifications_fall_back(question: str) -> None:
    assert _validate_detected_language(question, "Simplified Chinese") == "English"


def test_non_chinese_labels_pass_through_untouched() -> None:
    assert _validate_detected_language("Hola, ¿cómo estás?", "Spanish") == "Spanish"
    assert _validate_detected_language("Xin chào", "Vietnamese") == "Vietnamese"


def test_korean_question_misclassified_as_chinese_falls_back_to_korean() -> None:
    assert _validate_detected_language("오늘 날씨 어때?", "Simplified Chinese") == "Korean"
