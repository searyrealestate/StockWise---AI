# micha7_analyzer — Documentation Hub

> **Last Updated:** 2026-05-21T05:55:00Z
> **Documentation Version:** 1.1.0
> **Architecture Maturity:** 98.2%
> **Status:** Ready for Phase 1 Implementation

---

## 📚 Documentation Files (10 Public + Private)

### Public (✅ Safe for Git)

| # | File | Purpose | Priority |
|---|------|---------|----------|
| 1 | **README.md** | This file — navigation hub | Must read first |
| 2 | **ARCHITECTURE.md** | High-level architecture, principles | Must read |
| 3 | **GLOSSARY.md** | Domain terms (Pivot, Score, ARMED, etc.) | Reference |
| 4 | **PROJECT_STRUCTURE.md** | File layout, module boundaries | When coding |
| 5 | **PHASES.md** | Phased rollout plan with status | When planning |
| 6 | **DECISIONS.md** | Architecture Decision Records (13 ADRs) | When questioning |
| 7 | **TEMPLATE_ENGINE.md** | Future reusability strategy | When designing |
| 8 | **TESTING_PROTOCOL.md** | Agile testing methodology | **EVERY commit** |
| 9 | **IMPROVEMENT_ROADMAP.md** | Path from 98.2% to 100% maturity | When improving |
| 10 | **SECURITY.md** | What goes in Git, what stays private | Before commit |
| 11 | **CHANGELOG.md** | Timestamped log of every action | Continuous |

### Private (❌ NOT for Git — .gitignore)

| File | Purpose |
|------|---------|
| **credentials.local.md** | API keys, tokens, broker credentials |
| **business_logic.local.md** | Scoring formulas, weights, thresholds |
| **config_values.local.md** | Money, percentages, risk parameters |
| **implementation_notes.local.md** | Code-level notes, algorithm specifics, interfaces |
| **trading_history.local.md** | Personal P&L (when relevant) |

---

## 🚀 Quick Start for Next Chat Session

When starting a new chat with Claude on this project:

### Step 1: Upload Public Documentation
Upload these files in order (more = better context, but README is the minimum):

**Minimum viable:**
- `README.md`

**Recommended (understanding the system):**
- `README.md`
- `ARCHITECTURE.md`
- `PHASES.md`
- `CHANGELOG.md` (to see recent actions)

**Complete context:**
- All public files (1-11 above)

### Step 2: Upload Private Documentation (Only If Needed)
- For implementation work: `implementation_notes.local.md`
- For config tuning: `config_values.local.md`
- For business logic: `business_logic.local.md`

### Step 3: Tell Claude the Task
Example: "Continue Phase 1 implementation, specifically the FeatureExtractor for f1_candle pattern."

---

## 🎯 Project At a Glance

- **Name:** micha7_analyzer
- **Type:** Deterministic technical analysis (Phase A pure)
- **Parent System:** StockWise AI
- **Status:** Architecture complete (98.2%), ready for Phase 1
- **Approach:** Standalone module on parent infrastructure
- **Methodology:** Multi-parameter checklist + state machine + pivot detection
- **Outputs:** Multi-channel signals (Telegram, HTML, TradingView Pine Script)
- **Template Pattern:** First of analyzer family (see TEMPLATE_ENGINE.md)

---

## 📊 Documentation Health

| Metric | Status |
|--------|--------|
| Architecture documented | ✅ Complete |
| Decisions recorded (13 ADRs) | ✅ Complete |
| Phase plan defined | ✅ Complete |
| Testing protocol defined | ✅ Complete |
| Template engine planned | ✅ Complete |
| Improvement roadmap defined | ✅ Complete |
| Security boundaries set | ✅ Complete |
| Changelog active | ✅ Initialized |

---

## 🔐 Security Quick Reference

**❌ NEVER commit to Git:**
- API keys, passwords, tokens
- All source code (per user policy)
- Proprietary business logic / scoring formulas
- Specific monetary values, position sizes
- Implementation interfaces (function signatures)
- Personal trading parameters
- State files, logs

**✅ Safe to commit:**
- High-level architecture concepts
- Public principles and design patterns
- Phase plans (without specific numbers)
- This documentation hub structure
- Process documents (testing protocol, security policy)

See `SECURITY.md` for the complete policy.

---

## 🔄 Agile Workflow

This project follows strict Agile testing protocols. **Every code change** must:

1. **SPEC** — Define interface + behavior
2. **TEST** — Write failing tests (TDD)
3. **CODE** — Minimum code to pass tests
4. **VERIFY** — Run all tests
5. **REFACTOR** — Clean code, tests still pass
6. **REGRESS** — Run parent system master_validator
7. **COMMIT** — Git commit with descriptive message
8. **LOG** — Update CHANGELOG.md

See `TESTING_PROTOCOL.md` for complete protocol.

**Iron Rule:** No commit without tests passing. Period.

---

## 📞 Cross-References

- **Parent project:** `C:\Users\user\PycharmProjects\StockWise - AI`
- **Git branch:** `feature/micha7-analyzer-phase1`
- **Workspace location:** `.claude-workspace/micha7/`
- **Memory file:** `C:/Users/user/.claude/memory.md` (Claude Code)
- **Skills used:** `stockwise-core`, `stockwise-workflow`, `stockwise-testing`

---

## 🗺️ Reading Order for New Team Members

1. `README.md` ← You are here
2. `ARCHITECTURE.md` — System overview
3. `GLOSSARY.md` — Domain terms
4. `PHASES.md` — Project roadmap
5. `DECISIONS.md` — Why things are the way they are
6. `TEMPLATE_ENGINE.md` — Future reusability
7. `TESTING_PROTOCOL.md` — How we work
8. `SECURITY.md` — What's public vs private
9. `IMPROVEMENT_ROADMAP.md` — Where we're going
10. `PROJECT_STRUCTURE.md` — Where everything lives
11. `CHANGELOG.md` — What happened when

**Then** request `.local.md` files for implementation specifics.

---

## ⚠️ Important Notes

### About 98.2% Maturity
We've explicitly chosen to start implementation at 98.2% rather than push to 100%. The remaining 1.8% requires real implementation data to address effectively. See `IMPROVEMENT_ROADMAP.md` for the deferred items.

### About Template Engine
micha7 is designed as the **first** of a reusable analyzer family. Phase 1 builds it standalone, but Phase 5+ will extract base classes for future analyzers. See `TEMPLATE_ENGINE.md`.

### About Privacy
Per project owner policy, **all source code is private**. Public documentation describes architecture and concepts without revealing implementation details. Specific interfaces, formulas, and values are in `.local.md` files.

---

## 🆘 If You're Lost

| Question | File to Read |
|----------|--------------|
| How does the system work? | ARCHITECTURE.md |
| What does this term mean? | GLOSSARY.md |
| Where do I add this code? | PROJECT_STRUCTURE.md |
| Why was X decided this way? | DECISIONS.md |
| How do I test this? | TESTING_PROTOCOL.md |
| What can I commit? | SECURITY.md |
| What's next? | PHASES.md |
| What happened recently? | CHANGELOG.md |
| How does this scale? | TEMPLATE_ENGINE.md |
| How do we improve? | IMPROVEMENT_ROADMAP.md |
