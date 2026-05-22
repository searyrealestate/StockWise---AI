# micha7_analyzer — Security & Git Policy

> **Version:** 1.0.0
> **Last Modified:** 2026-05-21T05:35:00Z
> **Owner:** Project owner (Eyal)

This document defines what is safe to commit to public/shared Git repositories and what must remain private.

---

## 1. Core Principle

**Default: Private.** Anything not explicitly listed as "Safe for Git" should be treated as private.

When in doubt → **`.local.md` file or git-ignored.**

---

## 2. Sensitivity Classification

### 🔴 NEVER Commit to Git (Critical)

| Category | Examples | Reason |
|----------|----------|--------|
| **Credentials** | API keys, OAuth tokens, broker passwords, encryption keys | Direct compromise |
| **Implementation Code** | All `micha7_*.py` source files | Per user policy: "כל הקוד פרטי" |
| **Business Logic** | Scoring formulas, weight values, threshold calculations | Proprietary IP |
| **Config Values** | Dollar amounts, position sizes, risk percentages, drawdown thresholds | Reveal trading strategy |
| **State Files** | `state/micha7/**/*.json` | Operational data, position info |
| **Trade Logs** | Personal trading history, P&L data | Personal financial data |
| **Personal Identifiers** | Account numbers, broker account IDs, real names in logs | PII / Financial privacy |

### 🟡 Conditional — Sanitize Before Commit (Medium)

| Category | Treatment |
|----------|-----------|
| Code interfaces (function signatures) | Per user decision — currently treated as Private |
| Config schema (field names only, no values) | Safe if values are in `.local.md` |
| Test data | Replace with synthetic/anonymized data |
| Error messages mentioning paths | Replace personal paths with `<USER>` placeholders |

### 🟢 Safe for Git (Public)

| Category | Examples |
|----------|----------|
| **High-level architecture** | ARCHITECTURE.md (no formulas) |
| **Project structure** | PROJECT_STRUCTURE.md (file names only) |
| **Phase planning** | PHASES.md (general goals, not specific numbers) |
| **Decision records** | DECISIONS.md (rationale without proprietary values) |
| **Documentation hub** | README.md, GLOSSARY.md |
| **This security policy** | SECURITY.md |
| **General changelog** | CHANGELOG.md (actions, not values) |

---

## 3. File-by-File Policy

### Public Documentation (`.claude-workspace/micha7/`)

| File | Git Status | Contains |
|------|-----------|----------|
| `README.md` | ✅ Commit | Navigation hub |
| `ARCHITECTURE.md` | ✅ Commit | System design, no formulas |
| `PROJECT_STRUCTURE.md` | ✅ Commit | File layout |
| `DECISIONS.md` | ✅ Commit | ADRs, rationale |
| `PHASES.md` | ✅ Commit | Plan, no money values |
| `CHANGELOG.md` | ✅ Commit | Actions, no proprietary values |
| `SECURITY.md` | ✅ Commit | This policy |
| `GLOSSARY.md` | ✅ Commit | Domain terms |

### Private Documentation (`.local.md`)

| File | Git Status | Contains |
|------|-----------|----------|
| `credentials.local.md` | 🔴 Gitignore | API keys, tokens |
| `business_logic.local.md` | 🔴 Gitignore | Scoring formulas, weights |
| `config_values.local.md` | 🔴 Gitignore | Money, percentages, thresholds |
| `implementation_notes.local.md` | 🔴 Gitignore | Code-level details |
| `trading_history.local.md` | 🔴 Gitignore | Personal P&L |

### Source Code (per user policy)

| Pattern | Git Status | Notes |
|---------|-----------|-------|
| `micha7_*.py` | 🔴 Gitignore | All implementation private |
| `tests/test_micha7_*.py` | 🔴 Gitignore | Tests reveal interfaces |
| Modifications to `system_config.py` | ⚠️ Conditional | Schema OK, values private |
| Modifications to `feature_engine.py` | 🔴 Gitignore | Business logic |

### Runtime Data

| Pattern | Git Status |
|---------|-----------|
| `state/micha7/**` | 🔴 Gitignore |
| `outputs/micha7/**` | 🔴 Gitignore |
| `*.log` | 🔴 Gitignore |
| `*.tmp` | 🔴 Gitignore |

---

## 4. .gitignore Template

Add to project root `.gitignore`:

```gitignore
# ============================================
# micha7_analyzer — Private Files
# ============================================

# Private documentation
.claude-workspace/micha7/*.local.md
.claude-workspace/micha7/credentials*
.claude-workspace/micha7/business_logic*
.claude-workspace/micha7/config_values*
.claude-workspace/micha7/implementation_notes*
.claude-workspace/micha7/trading_history*

# Runtime state and outputs
state/micha7/
outputs/micha7/
*.tmp
*.log

# Source code (per private code policy)
micha7_*.py
tests/test_micha7_*.py
tests/test_micha7_integration.py

# Backup files
*.backup_*
*.bak
```

⚠️ **Action Required:** Add these entries to project `.gitignore` before first commit.

---

## 5. Pre-Commit Checklist

Before every `git commit`, verify:

- [ ] No file matches gitignore patterns above
- [ ] No `.local.md` files staged
- [ ] No source `.py` files staged
- [ ] No API keys, tokens, or passwords in any committed file
- [ ] No specific dollar amounts in commits
- [ ] No personal account numbers or PII
- [ ] No state files (.json from `state/`)

**Recommended:** Use `git status` and review every file before commit.

---

## 6. Sensitive Information Patterns to Avoid

### In Code Comments
❌ `# Targeting 3% monthly return with $50k account`
✅ `# Targeting monthly return per config`

### In Documentation
❌ `Max position: 2% of $50,000 = $1,000`
✅ `Max position: configurable percentage of portfolio`

### In Logs
❌ `[ERROR] Failed to load AAPL state for account 4729-XXXX`
✅ `[ERROR] Failed to load AAPL state` (account ID logged to private file only)

### In Test Data
❌ `assert pf > 3.5  # Eyal's target`
✅ `assert pf > config["target_pf"]`

---

## 7. Incident Response

If sensitive data accidentally committed to Git:

### Immediate (within minutes)
1. **Do not push** if not yet pushed
2. If pushed: **rotate immediately** any exposed credentials
3. `git reset HEAD~1` to undo last commit (locally)
4. Force push if necessary (only if recent and small impact)

### Cleanup (within hours)
1. Use `git filter-branch` or BFG Repo-Cleaner to remove from history
2. Notify any collaborators to re-clone
3. Audit for additional exposure (search GitHub, log files, caches)

### Prevention
1. Add additional patterns to `.gitignore`
2. Update this document with new patterns to avoid
3. Add entry to `CHANGELOG.md` documenting the incident

---

## 8. Audit Procedure

Run quarterly (or before major releases):

```bash
# Search for common secret patterns in entire history
git log --all --full-history -p | grep -i "api_key\|password\|secret\|token"

# Check for accidentally committed personal paths
git log --all --full-history -p | grep "C:\\Users\\user"

# Verify gitignore is effective
git check-ignore -v <suspected_file>

# List all files currently tracked
git ls-files | sort
```

---

## 9. Sharing Documentation Externally

When sharing files outside the team (e.g., uploading to forums, sharing with consultants):

| Action | Requirement |
|--------|-------------|
| Share ARCHITECTURE.md | ✅ OK as-is |
| Share PROJECT_STRUCTURE.md | ✅ OK as-is |
| Share DECISIONS.md | ✅ OK as-is |
| Share CHANGELOG.md | ✅ OK as-is |
| Share any `.local.md` | ❌ NEVER |
| Share any `.py` file | ❌ NEVER per current policy |
| Share state file or log | ❌ NEVER |

---

## 10. Policy Updates

This document is reviewed:
- After every security incident
- Before each major phase transition
- Quarterly during operational phase
- Whenever user policy changes

**Last Review:** 2026-05-21T05:35:00Z
**Next Scheduled Review:** Before Phase 1 first commit
