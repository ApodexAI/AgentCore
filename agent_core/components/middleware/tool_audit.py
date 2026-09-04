"""ToolAuditMiddleware — tool-layer security audit and risk classification.

Issue #24 Phase C: audits all tool calls, blocks high-risk bash commands,
warns on suspicious web_fetch targets.

Risk levels:
- block: tool call is prevented, error returned to LLM
- warn: tool call proceeds but logged at WARNING level
- pass: normal execution
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from agent_core.protocols import ExecutionMiddleware, ToolCallContext

logger = logging.getLogger(__name__)

RiskLevel = Literal["block", "warn", "pass"]

# ── Bash high-risk patterns ─────────────────────────────────────────────

_BASH_BLOCK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+(-[rf]+\s+)?/"), "rm on root path"),
    (re.compile(r"\brm\s+-rf\b"), "rm -rf"),
    (re.compile(r"\bmkfs\b"), "mkfs (format disk)"),
    (re.compile(r"\bdd\s+.*of=/dev/"), "dd to device"),
    (re.compile(r"curl\s.*\|\s*(ba)?sh"), "curl pipe to shell"),
    (re.compile(r"wget\s.*\|\s*(ba)?sh"), "wget pipe to shell"),
    (re.compile(r"\b:\(\)\s*\{.*\|.*&\s*\}\s*;"), "fork bomb"),
    (re.compile(r"\bchmod\s+777\s+/"), "chmod 777 on root"),
    (re.compile(r"\bsudo\s+rm\b"), "sudo rm"),
    (re.compile(r">\s*/etc/"), "overwrite /etc/"),
    (re.compile(r"\bshutdown\b|\breboot\b|\bhalt\b"), "system shutdown"),
]

_BASH_WARN_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bsudo\b"), "sudo usage"),
    (re.compile(r"\bchmod\b"), "chmod usage"),
    (re.compile(r"\bchown\b"), "chown usage"),
    (re.compile(r"\bkill\s+-9\b"), "kill -9"),
    (re.compile(r"\bnc\s+-l"), "netcat listener"),
    (re.compile(r"\biptables\b"), "iptables modification"),
]


class ToolAuditMiddleware(ExecutionMiddleware):
    """Tool-layer middleware: audit + risk classification.

    - bash: regex-based command classification (block/warn/pass)
    - web_fetch: domain awareness (warn on unusual patterns)
    - All calls: structured audit log entry

    Vetoes a call via ``ctx.block(reason)``; ``NodeContext.call_tool``
    checks ``ctx.is_blocked`` after the chain and returns the reason to the
    model instead of executing the tool.
    """

    def __init__(
        self,
        block_high_risk_bash: bool = True,
        audit_log_enabled: bool = True,
    ) -> None:
        self._block_bash = block_high_risk_bash
        self._audit_enabled = audit_log_enabled

    async def before_tool_call(
        self, ctx: ToolCallContext,
    ) -> ToolCallContext:
        """Classify risk and optionally block."""
        risk, reason = self._classify(ctx.tool_name, ctx.tool_args)
        ctx.metadata["audit_risk"] = risk
        ctx.metadata["audit_reason"] = reason

        if risk == "block" and self._block_bash:
            ctx.block(reason)
            logger.warning(
                "ToolAudit BLOCKED [%s] %s(%s): %s",
                ctx.role_id, ctx.tool_name,
                _truncate_args(ctx.tool_args), reason,
            )
        elif risk == "warn":
            logger.warning(
                "ToolAudit WARN [%s] %s(%s): %s",
                ctx.role_id, ctx.tool_name,
                _truncate_args(ctx.tool_args), reason,
            )

        if self._audit_enabled:
            self._audit_log(ctx, risk, reason)

        return ctx

    async def after_tool_call(
        self, ctx: ToolCallContext, result: str,
    ) -> str:
        """Log tool result summary for audit trail."""
        if self._audit_enabled:
            risk = ctx.metadata.get("audit_risk", "pass")
            if risk != "pass":
                logger.info(
                    "ToolAudit result [%s] %s: %s (risk=%s)",
                    ctx.role_id, ctx.tool_name,
                    result[:100], risk,
                )
        return result

    def _classify(
        self, tool_name: str, tool_args: dict[str, Any],
    ) -> tuple[RiskLevel, str]:
        """Classify tool call risk level."""
        if tool_name == "bash":
            return self._classify_bash(
                str(tool_args.get("command", ""))
            )
        if tool_name == "web_fetch":
            return self._classify_scrape(
                str(tool_args.get("url", ""))
            )
        return "pass", ""

    def _classify_bash(self, command: str) -> tuple[RiskLevel, str]:
        """Classify bash command risk."""
        cmd_lower = command.lower().strip()

        for pattern, reason in _BASH_BLOCK_PATTERNS:
            if pattern.search(cmd_lower):
                return "block", f"high-risk bash: {reason}"

        for pattern, reason in _BASH_WARN_PATTERNS:
            if pattern.search(cmd_lower):
                return "warn", f"elevated-risk bash: {reason}"

        return "pass", ""

    def _classify_scrape(self, url: str) -> tuple[RiskLevel, str]:
        """Classify web_fetch URL risk."""
        url_lower = url.lower()

        # Internal/localhost targets
        if any(
            h in url_lower
            for h in ("localhost", "127.0.0.1", "0.0.0.0", "169.254.")
        ):
            return "warn", f"scrape targets internal address: {url[:80]}"

        # Non-HTTP schemes
        if url_lower and not url_lower.startswith(("http://", "https://")):
            return "warn", f"non-HTTP scrape scheme: {url[:80]}"

        return "pass", ""

    def _audit_log(
        self,
        ctx: ToolCallContext,
        risk: RiskLevel,
        reason: str,
    ) -> None:
        """Emit structured audit log entry."""
        logger.debug(
            "ToolAudit: task=%s role=%s tool=%s risk=%s reason=%s "
            "args=%s",
            ctx.task_id, ctx.role_id, ctx.tool_name,
            risk, reason or "none",
            _truncate_args(ctx.tool_args),
        )


def _truncate_args(args: dict[str, Any], max_len: int = 100) -> str:
    """Truncate tool args for logging."""
    s = str(args)
    return s[:max_len] + "..." if len(s) > max_len else s
