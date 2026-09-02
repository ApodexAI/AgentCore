"""Budget models for the orchestrator runtime.

TaskBudget constrains cost, depth, and expensive primitives.
BudgetState tracks consumption at runtime.
BudgetCharge is the unit of budget consumption per primitive call.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TaskBudget(BaseModel):
    """Allocated budget for a single research task.

    Uses soft limits (max_debate_rounds=0 means "prefer not to",
    not "absolutely forbidden") to avoid recreating the old
    topology-commit problem.
    """

    max_tokens: int = 500_000
    max_cost_usd: float | None = None
    max_wall_time_s: int = 300
    max_parallel: int = 3
    max_depth: int = 1  # v1: always 1 (no true recursion)
    max_search_calls: int = 20
    max_verify_passes: int = 5
    max_debate_rounds: int = 0
    default_model_tier: str = "medium"  # "light" | "medium" | "strong"
    role_tiers: dict[str, str] = Field(default_factory=dict)


class BudgetCharge(BaseModel):
    """Unit of budget consumption from a single primitive call."""

    primitive: str
    llm_calls: int = 0
    search_calls: int = 0
    verify_passes: int = 0
    debate_rounds: int = 0
    tokens: int = 0
    cost_usd: float = 0.0
    wall_time_s: int = 0


class BudgetState(BaseModel):
    """Runtime budget tracking — allocated vs consumed."""

    allocated: TaskBudget = Field(default_factory=TaskBudget)
    llm_calls_used: int = 0
    search_calls_used: int = 0
    verify_passes_used: int = 0
    debate_rounds_used: int = 0
    tokens_used: int = 0
    cost_usd_used: float = 0.0
    wall_time_s_used: int = 0
    exhausted: bool = False
    warnings: list[str] = Field(default_factory=list)

    def charge(self, c: BudgetCharge) -> None:
        """Apply a charge and check exhaustion."""
        self.llm_calls_used += c.llm_calls
        self.search_calls_used += c.search_calls
        self.verify_passes_used += c.verify_passes
        self.debate_rounds_used += c.debate_rounds
        self.tokens_used += c.tokens
        self.cost_usd_used += c.cost_usd
        self.wall_time_s_used += c.wall_time_s
        self._check_exhaustion()

    def _check_exhaustion(self) -> None:
        a = self.allocated
        if self.search_calls_used >= a.max_search_calls:
            self.exhausted = True
            self.warnings.append("search_calls exhausted")
        if self.verify_passes_used >= a.max_verify_passes:
            self.warnings.append("verify_passes exhausted")
        if self.tokens_used >= a.max_tokens:
            self.exhausted = True
            self.warnings.append("token budget exhausted")
        if a.max_cost_usd and self.cost_usd_used >= a.max_cost_usd:
            self.exhausted = True
            self.warnings.append("cost budget exhausted")
        if a.max_wall_time_s > 0 and self.wall_time_s_used >= a.max_wall_time_s:
            self.exhausted = True
            self.warnings.append("wall time exhausted")

    def escalate(self, new_budget: TaskBudget) -> list[str]:
        """Upgrade budget allocation in-place. Returns list of what changed.

        Only upgrades fields that are higher in new_budget than current.
        Preserves consumption counters — just raises the ceilings.
        """
        changes: list[str] = []
        a = self.allocated
        for field in (
            "max_tokens", "max_wall_time_s", "max_parallel",
            "max_search_calls", "max_verify_passes", "max_debate_rounds",
        ):
            old_val = getattr(a, field)
            new_val = getattr(new_budget, field)
            if new_val > old_val:
                setattr(a, field, new_val)
                changes.append(f"{field}: {old_val} → {new_val}")
        # Model tier upgrade
        tier_order = {"light": 0, "medium": 1, "strong": 2}
        if tier_order.get(new_budget.default_model_tier, 0) > tier_order.get(a.default_model_tier, 0):
            old_tier = a.default_model_tier
            a.default_model_tier = new_budget.default_model_tier
            changes.append(f"model_tier: {old_tier} → {new_budget.default_model_tier}")
        # Role tiers: merge (only upgrade, never downgrade)
        for role, tier in new_budget.role_tiers.items():
            current = a.role_tiers.get(role, "light")
            if tier_order.get(tier, 0) > tier_order.get(current, 0):
                a.role_tiers[role] = tier
                changes.append(f"role_tier[{role}]: {current} → {tier}")
        # Reset exhaustion if we have headroom now
        if changes:
            self.exhausted = False
            self.warnings = [w for w in self.warnings if "exhausted" not in w]
        return changes

    @property
    def remaining_search_calls(self) -> int:
        return max(0, self.allocated.max_search_calls - self.search_calls_used)

    @property
    def remaining_verify_passes(self) -> int:
        return max(0, self.allocated.max_verify_passes - self.verify_passes_used)

    @property
    def remaining_debate_rounds(self) -> int:
        return max(0, self.allocated.max_debate_rounds - self.debate_rounds_used)
