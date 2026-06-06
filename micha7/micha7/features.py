"""Feature execution infrastructure: contract, ABC, and dependency-aware DAG.

Generic layer only — no concrete features here (F1–F7 implemented in 3.2–3.8).
Features self-declare dependencies; FeatureDAG resolves order via Kahn's
topological layering, detects cycles, and executes deterministically.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

_module_logger = logging.getLogger("micha7.features")


# ---------------------------------------------------------------------------
# Score enum
# ---------------------------------------------------------------------------


class FeatureScore(str, Enum):
    """The tri-state result of a single feature evaluation (ADR-016)."""

    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    EMPTY = "EMPTY"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class FeatureResult:
    """Uniform result returned by every feature's compute() call.

    raw holds feature-specific supplementary data (e.g. raw_distance for F4,
    raw_levels for F6) that downstream components may consume.
    """

    feature_id: str
    score: FeatureScore
    raw: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class FeatureDAGError(Exception):
    """Raised for DAG structural problems: duplicate id, empty id, unknown
    dependency, or cycle detected."""


# ---------------------------------------------------------------------------
# BaseFeature
# ---------------------------------------------------------------------------


class BaseFeature(ABC):
    """Abstract base class every feature must subclass.

    Class attributes to override (do NOT use instance attributes for these):
        feature_id   : non-empty unique string (e.g. "F1")
        dependencies : tuple of feature_id strings this feature needs in context
                       before compute() is called.
    """

    feature_id: str = ""
    dependencies: tuple[str, ...] = ()

    def __init__(self, loader: Any = None, logger: Any = None) -> None:
        self._loader = loader
        self._logger = logger

    @abstractmethod
    def compute(
        self,
        md: Any,
        context: dict[str, FeatureResult],
    ) -> FeatureResult:
        """Compute this feature and return a FeatureResult.

        Args:
            md:      A MarketData object (type-hinted as Any here to avoid
                     circular imports at the infrastructure level).
            context: Results of features that have already been computed in
                     earlier DAG levels (accumulates across levels).

        Returns:
            FeatureResult with this feature's score and optional raw payload.
        """


# ---------------------------------------------------------------------------
# FeatureDAG
# ---------------------------------------------------------------------------


class FeatureDAG:
    """Dependency-aware, deterministic feature execution engine.

    Usage::

        dag = FeatureDAG(logger=logger)
        dag.register(F6Feature())
        dag.register(F1Feature())   # F1 depends on F6
        results = dag.run(market_data)
    """

    def __init__(self, logger: Any = None) -> None:
        self._features: dict[str, BaseFeature] = {}
        self._logger = logger

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, feature: BaseFeature) -> None:
        """Register a feature instance.

        Raises FeatureDAGError if feature_id is empty or already registered.
        Dependency validation (unknown ids) is deferred to topological_levels()
        so that registration order does not matter.
        """
        fid = feature.feature_id
        if not fid:
            raise FeatureDAGError(
                f"Cannot register feature with empty feature_id: {type(feature).__name__}"
            )
        if fid in self._features:
            raise FeatureDAGError(
                f"duplicate feature_id '{fid}': already registered."
            )
        self._features[fid] = feature

    # ------------------------------------------------------------------
    # Topological layering (Kahn's algorithm)
    # ------------------------------------------------------------------

    def topological_levels(self) -> list[list[str]]:
        """Return features grouped into dependency levels using Kahn's algorithm.

        Level 0 = features with no dependencies.
        Level N = features whose all dependencies are in levels < N.
        Within each level, feature_ids are sorted lexicographically (determinism).

        Raises FeatureDAGError on:
          - unknown dependency (a declared dependency has no registered feature)
          - cycle (including self-loops)
        """
        if not self._features:
            return []

        known = set(self._features.keys())

        # Validate: every declared dependency must be a registered id
        for fid, feature in self._features.items():
            for dep in feature.dependencies:
                if dep not in known:
                    raise FeatureDAGError(
                        f"Feature '{fid}' has unknown dependency '{dep}'. "
                        f"Registered features: {sorted(known)}"
                    )

        # Build in-degree and adjacency for Kahn's algorithm
        in_degree: dict[str, int] = {fid: 0 for fid in known}
        dependents: dict[str, list[str]] = {fid: [] for fid in known}

        for fid, feature in self._features.items():
            for dep in feature.dependencies:
                in_degree[fid] += 1
                dependents[dep].append(fid)

        # Seed queue with zero-in-degree nodes (sorted for determinism)
        queue: deque[str] = deque(sorted(fid for fid, deg in in_degree.items() if deg == 0))

        levels: list[list[str]] = []
        processed = 0

        while queue:
            # Drain the current queue into one level (all are ready simultaneously)
            level_size = len(queue)
            level_nodes: list[str] = []
            for _ in range(level_size):
                node = queue.popleft()
                level_nodes.append(node)
                processed += 1
                for child in sorted(dependents[node]):
                    in_degree[child] -= 1
                    if in_degree[child] == 0:
                        # Insert in sorted position to keep next level sorted
                        queue.append(child)
            # Sort within the level for determinism
            levels.append(sorted(level_nodes))

        if processed < len(known):
            # Nodes remaining with in_degree > 0 → cycle
            self._log_event("feature_dag_cycle_detected", {"remaining": sorted(
                fid for fid, deg in in_degree.items() if deg > 0
            )})
            raise FeatureDAGError(
                f"Cycle detected in feature dependency graph. "
                f"Nodes in cycle: {sorted(fid for fid, deg in in_degree.items() if deg > 0)}"
            )

        return levels

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run(self, md: Any) -> dict[str, FeatureResult]:
        """Execute all registered features in topological order.

        Returns a dict mapping feature_id → FeatureResult.
        L2 features receive the accumulated context of all L1 results.
        Raises any exception thrown by compute() without catching it.
        """
        if not self._features:
            return {}

        levels = self.topological_levels()

        self._log_event(
            "feature_dag_built",
            {"levels": [[fid for fid in lvl] for lvl in levels]},
        )

        context: dict[str, FeatureResult] = {}

        for level in levels:
            for fid in level:  # already sorted by topological_levels
                feature = self._features[fid]
                result = feature.compute(md, dict(context))
                context[fid] = result
                self._log_event(
                    "feature_computed",
                    {"feature_id": fid, "score": result.score},
                )

        return context

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log_event(self, event: str, context: dict) -> None:
        logger = self._logger if self._logger is not None else _module_logger
        logger.debug(event, extra={"event": event, "context": context})
