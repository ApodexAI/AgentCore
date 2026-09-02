"""Provider-neutral execution engine for OpenAI-compatible summary LLMs."""

from __future__ import annotations

# pyright: basic, reportMissingImports=false
import asyncio
import hashlib
import logging
from collections.abc import Callable, Sequence
from typing import Any, NotRequired, TypedDict

import httpx

logger = logging.getLogger(__name__)


class SummaryCandidate(TypedDict):
    endpoint: str
    model: str
    api_key: NotRequired[str]
    provider: NotRequired[str]
    extra_headers: NotRequired[dict[str, str]]


type UsageRecorder = Callable[..., None]

EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{focus}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{content}

EXTRACTED INFORMATION:"""


def normalize_summary_endpoint(base_url: str) -> str:
    endpoint = (base_url or "").rstrip("/")
    if endpoint and not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    return endpoint


def describe_summary_candidates(candidates: Sequence[SummaryCandidate]) -> str:
    """Render candidates without exposing usable credentials."""
    if not candidates:
        return "(no summary LLM candidates)"
    lines: list[str] = []
    for index, candidate in enumerate(candidates, 1):
        key = str(candidate.get("api_key") or "")
        fingerprint = (
            f"len={len(key)} #{hashlib.sha256(key.encode()).hexdigest()[:12]}"
            if key
            else "unset"
        )
        lines.append(
            f"{index}. provider={candidate.get('provider') or '?'}"
            f" model={candidate.get('model') or '?'}"
            f" endpoint={candidate.get('endpoint') or '?'}"
            f" api_key={fingerprint}",
        )
    return "\n".join(lines)


def build_summary_payload(model: str, prompt: str) -> dict[str, Any]:
    """Build the shared Chat Completions request shape for extraction."""
    lowered = model.lower()
    if "gpt-5" in lowered or "gpt5" in lowered:
        return {
            "model": model,
            "max_completion_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": "minimal",
            "service_tier": "flex",
        }
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": 8192,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
    }
    if any(
        key in lowered
        for key in ("qwen", "apodex", "mirothinker", "sglang", "397b")
    ):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def truncate_summary_fallback(content: str, limit: int = 20_000) -> str:
    if len(content) > limit:
        return content[:limit] + "\n\n[Content truncated...]"
    return content


class SummaryLLMEngine:
    """Retry ordered raw-HTTP summary candidates and degrade to truncation."""

    def __init__(
        self,
        *,
        max_retries: int = 4,
        truncate_step: int = 40_960,
        request_timeout: float = 300,
        fallback_limit: int = 20_000,
        usage_recorder: UsageRecorder | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
    ) -> None:
        self.max_retries = max_retries
        self.truncate_step = truncate_step
        self.request_timeout = request_timeout
        self.fallback_limit = fallback_limit
        self.usage_recorder = usage_recorder
        self.sleep = sleep

    async def summarize(
        self,
        content: str,
        focus: str,
        candidates: Sequence[SummaryCandidate],
    ) -> str:
        for index, candidate in enumerate(candidates):
            summary = await self._summarize_one(
                candidate,
                content,
                focus,
                candidate_index=index,
            )
            if summary:
                return summary
            if index + 1 < len(candidates):
                logger.warning(
                    "Summary candidate %d (%s) exhausted; falling back to %s",
                    index + 1,
                    candidate.get("model"),
                    candidates[index + 1].get("model"),
                )
        return truncate_summary_fallback(content, self.fallback_limit)

    async def _summarize_one(
        self,
        candidate: SummaryCandidate,
        content: str,
        focus: str,
        *,
        candidate_index: int,
    ) -> str:
        endpoint = candidate["endpoint"]
        model = candidate["model"]
        prompt = EXTRACT_INFO_PROMPT.format(focus=focus, content=content)
        payload = build_summary_payload(model, prompt)
        headers = {"Content-Type": "application/json"}
        api_key = candidate.get("api_key") or ""
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        headers.update(candidate.get("extra_headers") or {})

        current_content = content
        for attempt in range(self.max_retries):
            try:
                async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                    response = await client.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                    )
                body = response.text
                if response.status_code >= 400 and (
                    "maximum context length" in body
                    or "longer than the model's context length" in body
                ):
                    remove = self.truncate_step * (attempt + 1)
                    if remove >= len(current_content):
                        return ""
                    current_content = content[:-remove] + "[...truncated]"
                    payload["messages"][0]["content"] = EXTRACT_INFO_PROMPT.format(
                        focus=focus,
                        content=current_content,
                    )
                    continue
                response.raise_for_status()
                data = response.json()
                self._record_usage(candidate, data.get("usage") or {})
                summary = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if summary:
                    return str(summary)
                logger.warning(
                    "Empty summary response on attempt %d (candidate %d)",
                    attempt + 1,
                    candidate_index + 1,
                )
                return ""
            except Exception as error:
                logger.warning("Summary LLM attempt %d failed: %s", attempt + 1, error)
                if attempt + 1 < self.max_retries:
                    await self.sleep(float(attempt + 1))
        return ""

    def _record_usage(
        self,
        candidate: SummaryCandidate,
        usage: dict[str, Any],
    ) -> None:
        if not usage or self.usage_recorder is None:
            return
        details = usage.get("prompt_tokens_details") or {}
        self.usage_recorder(
            model=candidate.get("model", ""),
            provider=candidate.get("provider") or "summary_llm",
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            cache_read_tokens=int(details.get("cached_tokens", 0) or 0),
        )


__all__ = [
    "EXTRACT_INFO_PROMPT",
    "SummaryCandidate",
    "SummaryLLMEngine",
    "build_summary_payload",
    "describe_summary_candidates",
    "normalize_summary_endpoint",
    "truncate_summary_fallback",
]
