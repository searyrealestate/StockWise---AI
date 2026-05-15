# 11 — Pain Signals

**Generated**: 2026-05-15 | **Commit**: 6bc83e8

---

## TODO Comments

| File | Line | Comment |
|------|------|---------|
| system_config.py | 1746 | `# TODO: migrate to safe_json_io (needs ensure_ascii=False support not in safe_json_write)` |

**Total TODOs in production code: 1**

---

## DEPRECATED Markers

| File | Line | Comment |
|------|------|---------|
| system_config.py | 413 | `"runner_min_distance_pct": 0.008, # DEPRECATED — canonical source is now KINETIC_STOP_CONFIG` |

**Total DEPRECATEDs in production code: 1**

---

## FIXME / HACK / XXX Comments

**None found** in any of the 25 core production modules.

---

## Large Comment Blocks (5+ consecutive comment lines)

These may indicate dead code, disabled features, or in-progress design notes.

### High-Priority (22+ lines — likely dead code)

| File | Lines | Size | Notes |
|------|-------|------|-------|
| live_trading_engine.py | 148–169 | 22 lines | Large commented block — likely disabled feature or old implementation |

### Medium (9–11 lines)

| File | Lines | Size | Likely Content |
|------|-------|------|----------------|
| system_config.py | 158–166 | 9 lines | Config documentation block |
| system_config.py | 227–235 | 9 lines | Config documentation block |
| system_config.py | 1475–1485 | 11 lines | Large comment / disabled config section |
| feature_engine.py | 535–544 | 10 lines | Indicator implementation notes or disabled indicator |
| feature_engine.py | 554–564 | 11 lines | Indicator block — possibly disabled pattern |
| setup_templates.py | 475–481 | 7 lines | Template or block definition |
| master_validator.py | 623–649 | Multiple blocks | Test documentation or disabled test |

### feature_engine.py Comment Clusters (lines 520–574)

Multiple overlapping comment blocks near lines 520–574. This region likely contains:
- Disabled technical indicators (pattern recognition)
- Architecture notes for planned features

Feature engine has 7 comment blocks in 55 lines — highest density in the codebase.

---

## Config Encoding Issue

`requirements.txt` is UTF-16 encoded with BOM. This is not a code pain signal but causes tooling issues (grep, pip, etc. may misread it). The file was confirmed readable only via `open('requirements.txt', 'rb').decode('utf-16')`.

---

## Summary

| Signal Type | Count |
|-------------|-------|
| TODO | 1 |
| DEPRECATED | 1 |
| FIXME | 0 |
| HACK | 0 |
| XXX | 0 |
| Large comment blocks (5+) | 46 across 10 files |
| Largest single block | live_trading_engine.py:148-169 (22 lines) |

**Production code pain signal density is very low** — 2 explicit markers in 16,000+ lines of core code. The large comment blocks in `feature_engine.py` (near pattern indicators) and `live_trading_engine.py` (lines 148–169) warrant manual inspection to confirm whether they are dead code or documentation.
