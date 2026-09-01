"""NodeContext — minimal facade between a DAG node and the framework.

The implementation here carries identity only (``node_id`` / ``role_id`` /
``task_id``). Nodes that need to reach an LLM, the tool layer, or
persistence — anything with a ``ctx.call_llm``-shaped call — must run under
a richer context supplied by the host via
``DynamicGraphBuilder(node_context_factory=...)``; see
:class:`agent_core.runtime.dag.graph_builder.NodeContextFactory`. Reaching
for a capability this class does not have raises an
:class:`AttributeError` that names that seam rather than a bare
"no attribute 'call_llm'".
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Never

from agent_core.models.pipeline_spec import NodeDefinition

logger = logging.getLogger(__name__)


class NodeContext:
    """Identity-only context object passed to every wrapped node.

    Deliberately narrow: this package ships no LLM/tool/persistence-backed
    context. Pass ``node_context_factory`` to
    :class:`~agent_core.runtime.dag.graph_builder.DynamicGraphBuilder` to
    inject one that does.
    """

    def __init__(
        self,
        node_def: NodeDefinition,
        task_id_getter: Callable[[], str],
    ) -> None:
        self._node_def = node_def
        self._task_id_getter = task_id_getter

    @property
    def node_id(self) -> str:
        return self._node_def.node_id

    @property
    def role_id(self) -> str:
        return self._node_def.role_id

    @property
    def task_id(self) -> str:
        return self._task_id_getter()

    def __getattr__(self, name: str) -> Never:
        """Fail with the seam named, instead of a bare missing-attribute error.

        A node written against the full node-context contract (``call_llm``,
        ``call_tool``, ``run_agent_loop``, …) that is built by a
        ``DynamicGraphBuilder`` with no ``node_context_factory`` would
        otherwise die deep inside the node on ``AttributeError:
        'NodeContext' object has no attribute 'call_llm'``, which says
        nothing about where the missing capability should come from.
        """
        # Dunder probes (copy, pickle, inspect) expect a plain AttributeError.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        node_def = self.__dict__.get("_node_def")
        where = (
            f" (node {node_def.node_id!r}, role {node_def.role_id!r})"
            if node_def is not None
            else ""
        )
        raise AttributeError(
            f"NodeContext is identity-only and provides no {name!r}{where}. "
            f"Nodes needing it must be built with a richer context: "
            f"DynamicGraphBuilder(node_context_factory=<factory>)."
        )


# Alias for callers that import ``DefaultNodeContext``; ``NodeContext``
# is the single concrete implementation.
DefaultNodeContext = NodeContext
