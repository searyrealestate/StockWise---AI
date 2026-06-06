"""TDD test suite for micha7.features — BaseFeature ABC + FeatureDAG.

All tests written BEFORE implementation. Expect RED on first run.
No network calls; all test doubles are inline stub subclasses.
"""

import io
import json
import logging

import pytest

from micha7.features import (
    BaseFeature,
    FeatureDAG,
    FeatureDAGError,
    FeatureResult,
    FeatureScore,
)


# ---------------------------------------------------------------------------
# Shared stub fixtures
# ---------------------------------------------------------------------------


def _make_feature(fid: str, deps: tuple = (), score: FeatureScore = FeatureScore.BULLISH):
    """Return a concrete BaseFeature subclass with the given id and deps."""

    class _Stub(BaseFeature):
        feature_id = fid
        dependencies = deps

        def compute(self, md, context):
            return FeatureResult(feature_id=self.feature_id, score=score)

    return _Stub()


def _stub_md():
    """Return a minimal stand-in for MarketData (structure not inspected in 3.1)."""
    return object()


def _dag_with_logger():
    """Return a FeatureDAG wired to an in-memory log capture stream."""
    stream = io.StringIO()
    logger = logging.getLogger("test.dag.capture")
    logger.handlers.clear()
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter('{"event":"%(message)s"}')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return FeatureDAG(logger=logger), stream


# ---------------------------------------------------------------------------
# FeatureScore (3 tests)
# ---------------------------------------------------------------------------


def test_feature_score_values():
    """BULLISH, BEARISH, EMPTY exist and have the right string values."""
    assert FeatureScore.BULLISH == "BULLISH"
    assert FeatureScore.BEARISH == "BEARISH"
    assert FeatureScore.EMPTY == "EMPTY"


def test_feature_score_is_str_serializable():
    """FeatureScore is a str-enum; json.dumps works without custom encoder."""
    payload = {"score": FeatureScore.BULLISH}
    serialized = json.dumps(payload)
    assert '"BULLISH"' in serialized


def test_feature_score_membership():
    """'BULLISH' in FeatureScore works; 'INVALID' is not a member."""
    assert "BULLISH" in FeatureScore._value2member_map_
    assert "INVALID" not in FeatureScore._value2member_map_


# ---------------------------------------------------------------------------
# FeatureResult (4 tests)
# ---------------------------------------------------------------------------


def test_feature_result_raw_defaults_empty():
    """raw defaults to an empty dict when not supplied."""
    r = FeatureResult(feature_id="F1", score=FeatureScore.BULLISH)
    assert r.raw == {}


def test_feature_result_holds_score():
    """score field is stored and retrieved correctly."""
    r = FeatureResult(feature_id="F7", score=FeatureScore.BEARISH)
    assert r.score is FeatureScore.BEARISH


def test_feature_result_raw_accepts_dict():
    """raw accepts an arbitrary dict."""
    r = FeatureResult(feature_id="F4", score=FeatureScore.EMPTY, raw={"dist": 2.3})
    assert r.raw["dist"] == pytest.approx(2.3)


def test_feature_result_fields():
    """feature_id, score, raw are all accessible."""
    r = FeatureResult(feature_id="F6", score=FeatureScore.BULLISH, raw={"levels": [100.0]})
    assert r.feature_id == "F6"
    assert r.score == FeatureScore.BULLISH
    assert r.levels == [100.0] if hasattr(r, "levels") else r.raw["levels"] == [100.0]


# ---------------------------------------------------------------------------
# BaseFeature (5 tests)
# ---------------------------------------------------------------------------


def test_base_feature_not_instantiable():
    """BaseFeature is abstract; direct instantiation raises TypeError."""
    with pytest.raises(TypeError):
        BaseFeature()


def test_base_feature_missing_compute_not_instantiable():
    """A subclass without compute() cannot be instantiated."""

    class _NoCompute(BaseFeature):
        feature_id = "X"

    with pytest.raises(TypeError):
        _NoCompute()


def test_base_feature_id_present():
    """A concrete subclass has a non-empty feature_id."""
    f = _make_feature("F1")
    assert f.feature_id == "F1"


def test_base_feature_dependencies_default_empty():
    """dependencies defaults to an empty tuple when not overridden."""
    f = _make_feature("F5")
    assert f.dependencies == ()


def test_base_feature_stores_loader_and_logger():
    """__init__ stores loader and logger as instance attributes."""

    class _Stub(BaseFeature):
        feature_id = "FX"
        dependencies = ()

        def compute(self, md, context):
            return FeatureResult(feature_id="FX", score=FeatureScore.EMPTY)

    sentinel_loader = object()
    sentinel_logger = object()
    f = _Stub(loader=sentinel_loader, logger=sentinel_logger)
    assert f._loader is sentinel_loader
    assert f._logger is sentinel_logger


# ---------------------------------------------------------------------------
# Registration (6 tests)
# ---------------------------------------------------------------------------


def test_register_single_feature():
    """register() accepts a valid feature without error."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))


def test_register_duplicate_id_raises():
    """Registering two features with the same id raises FeatureDAGError."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))
    with pytest.raises(FeatureDAGError, match="duplicate"):
        dag.register(_make_feature("F1"))


def test_register_empty_id_raises():
    """Registering a feature with an empty feature_id raises FeatureDAGError."""
    dag = FeatureDAG()

    class _NoId(BaseFeature):
        feature_id = ""
        dependencies = ()

        def compute(self, md, context):
            return FeatureResult(feature_id="", score=FeatureScore.EMPTY)

    with pytest.raises(FeatureDAGError, match="empty"):
        dag.register(_NoId())


def test_register_multiple_features():
    """register() accepts several distinct features."""
    dag = FeatureDAG()
    for fid in ("F1", "F2", "F3", "F4", "F5", "F6", "F7"):
        dag.register(_make_feature(fid))


def test_register_unknown_dependency_raises():
    """A feature declaring a dependency on an unregistered id raises FeatureDAGError."""
    dag = FeatureDAG()
    # F1 declares F6 as dependency, but F6 is never registered
    dag.register(_make_feature("F1", deps=("F6",)))
    with pytest.raises(FeatureDAGError, match="unknown dependency"):
        dag.topological_levels()


def test_registered_features_retrievable():
    """Registered features are accessible from the internal dict."""
    dag = FeatureDAG()
    f = _make_feature("F3")
    dag.register(f)
    assert "F3" in dag._features


# ---------------------------------------------------------------------------
# Topological levels (7 tests)
# ---------------------------------------------------------------------------


def test_topo_single_no_dep():
    """One feature with no dependencies → one level."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))
    levels = dag.topological_levels()
    assert levels == [["F1"]]


def test_topo_two_independent_same_level():
    """Two features with no deps → one level containing both."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))
    dag.register(_make_feature("F2"))
    levels = dag.topological_levels()
    assert len(levels) == 1
    assert sorted(levels[0]) == ["F1", "F2"]


def test_topo_dependency_next_level():
    """F1 depends on F2 → F2 in L1, F1 in L2."""
    dag = FeatureDAG()
    dag.register(_make_feature("F2"))
    dag.register(_make_feature("F1", deps=("F2",)))
    levels = dag.topological_levels()
    assert len(levels) == 2
    assert "F2" in levels[0]
    assert "F1" in levels[1]


def test_topo_micha7_graph():
    """micha7 production graph: L1={F2,F4,F5,F6,F7}, L2={F1,F3}."""
    dag = FeatureDAG()
    dag.register(_make_feature("F2"))
    dag.register(_make_feature("F4"))
    dag.register(_make_feature("F5"))
    dag.register(_make_feature("F6"))
    dag.register(_make_feature("F7"))
    dag.register(_make_feature("F1", deps=("F6",)))
    dag.register(_make_feature("F3", deps=("F2",)))
    levels = dag.topological_levels()
    assert len(levels) == 2
    assert sorted(levels[0]) == ["F2", "F4", "F5", "F6", "F7"]
    assert sorted(levels[1]) == ["F1", "F3"]


def test_topo_sorted_within_level():
    """Feature ids within each level are sorted lexicographically (determinism)."""
    dag = FeatureDAG()
    for fid in ("F7", "F5", "F2", "F4", "F6"):
        dag.register(_make_feature(fid))
    levels = dag.topological_levels()
    assert levels[0] == sorted(levels[0])


def test_topo_empty_dag():
    """An empty DAG produces an empty levels list."""
    dag = FeatureDAG()
    assert dag.topological_levels() == []


def test_topo_three_node_chain():
    """A→B→C produces 3 separate levels (proves N-level, not just 2-level)."""
    dag = FeatureDAG()
    dag.register(_make_feature("A"))
    dag.register(_make_feature("B", deps=("A",)))
    dag.register(_make_feature("C", deps=("B",)))
    levels = dag.topological_levels()
    assert len(levels) == 3
    assert levels[0] == ["A"]
    assert levels[1] == ["B"]
    assert levels[2] == ["C"]


# ---------------------------------------------------------------------------
# Cycle detection (4 tests)
# ---------------------------------------------------------------------------


def test_cycle_two_node_raises():
    """A→B, B→A is a cycle — must raise FeatureDAGError."""
    dag = FeatureDAG()
    dag.register(_make_feature("A", deps=("B",)))
    dag.register(_make_feature("B", deps=("A",)))
    with pytest.raises(FeatureDAGError, match="cycle"):
        dag.topological_levels()


def test_cycle_self_loop_raises():
    """A→A (self-loop) must raise FeatureDAGError."""
    dag = FeatureDAG()
    dag.register(_make_feature("A", deps=("A",)))
    with pytest.raises(FeatureDAGError, match="cycle"):
        dag.topological_levels()


def test_cycle_three_node_raises():
    """A→B→C→A raises FeatureDAGError."""
    dag = FeatureDAG()
    dag.register(_make_feature("A", deps=("C",)))
    dag.register(_make_feature("B", deps=("A",)))
    dag.register(_make_feature("C", deps=("B",)))
    with pytest.raises(FeatureDAGError, match="cycle"):
        dag.topological_levels()


def test_no_cycle_passes():
    """A valid acyclic graph does not raise."""
    dag = FeatureDAG()
    dag.register(_make_feature("X"))
    dag.register(_make_feature("Y", deps=("X",)))
    dag.register(_make_feature("Z", deps=("X",)))
    levels = dag.topological_levels()
    assert len(levels) == 2


# ---------------------------------------------------------------------------
# Execution (6 tests)
# ---------------------------------------------------------------------------


def test_run_returns_result_per_feature():
    """run() returns a dict with one FeatureResult per registered feature."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))
    dag.register(_make_feature("F2"))
    md = _stub_md()
    results = dag.run(md)
    assert set(results.keys()) == {"F1", "F2"}
    assert all(isinstance(v, FeatureResult) for v in results.values())


def test_run_l2_receives_l1_context():
    """L2 feature's compute() is called with L1 results already in context."""
    received_context = {}

    class _L2(BaseFeature):
        feature_id = "L2"
        dependencies = ("L1",)

        def compute(self, md, context):
            received_context.update(context)
            return FeatureResult(feature_id="L2", score=FeatureScore.EMPTY)

    dag = FeatureDAG()
    dag.register(_make_feature("L1"))
    dag.register(_L2())
    dag.run(_stub_md())
    assert "L1" in received_context


def test_run_context_accumulates():
    """In a 3-level chain, the final feature sees results from all prior levels."""
    seen = {}

    class _C(BaseFeature):
        feature_id = "C"
        dependencies = ("B",)

        def compute(self, md, context):
            seen.update(context)
            return FeatureResult(feature_id="C", score=FeatureScore.EMPTY)

    dag = FeatureDAG()
    dag.register(_make_feature("A"))
    dag.register(_make_feature("B", deps=("A",)))
    dag.register(_C())
    dag.run(_stub_md())
    assert "A" in seen and "B" in seen


def test_run_deterministic_two_runs():
    """Two consecutive run() calls return results in identical key order."""
    dag = FeatureDAG()
    for fid in ("F7", "F2", "F5", "F4", "F6"):
        dag.register(_make_feature(fid))
    dag.register(_make_feature("F1", deps=("F6",)))
    dag.register(_make_feature("F3", deps=("F2",)))
    md = _stub_md()
    keys1 = list(dag.run(md).keys())
    keys2 = list(dag.run(md).keys())
    assert keys1 == keys2


def test_run_empty_dag_returns_empty():
    """run() on an empty DAG returns an empty dict."""
    dag = FeatureDAG()
    assert dag.run(_stub_md()) == {}


def test_run_compute_exception_propagates():
    """If compute() raises, run() propagates the exception."""

    class _Boom(BaseFeature):
        feature_id = "BOOM"
        dependencies = ()

        def compute(self, md, context):
            raise ValueError("intentional")

    dag = FeatureDAG()
    dag.register(_Boom())
    with pytest.raises(ValueError, match="intentional"):
        dag.run(_stub_md())


# ---------------------------------------------------------------------------
# Logging (3 tests)
# ---------------------------------------------------------------------------


def test_logging_dag_built_event(caplog):
    """run() emits a feature_dag_built event."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))
    with caplog.at_level(logging.DEBUG):
        dag.run(_stub_md())
    events = [r.getMessage() for r in caplog.records]
    assert any("feature_dag_built" in e for e in events)


def test_logging_feature_computed_event(caplog):
    """run() emits a feature_computed event for each feature."""
    dag = FeatureDAG()
    dag.register(_make_feature("F1"))
    dag.register(_make_feature("F2"))
    with caplog.at_level(logging.DEBUG):
        dag.run(_stub_md())
    events = [r.getMessage() for e in caplog.records for _ in [None] if (r := e)]
    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "feature_computed" in log_text


def test_logging_cycle_event(caplog):
    """topological_levels() emits a feature_dag_cycle_detected event before raising."""
    dag = FeatureDAG()
    dag.register(_make_feature("A", deps=("B",)))
    dag.register(_make_feature("B", deps=("A",)))
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(FeatureDAGError):
            dag.topological_levels()
    log_text = " ".join(r.getMessage() for r in caplog.records)
    assert "feature_dag_cycle_detected" in log_text
