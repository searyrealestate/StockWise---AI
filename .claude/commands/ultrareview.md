Perform an ultra-thorough review of the change just made. Do ALL of:

1. FORWARD TRACE: Trace the exact code path of the change from entry
   point to effect. List every function touched and confirm the logic
   is correct line by line.

2. BLAST RADIUS: List every file, test, and feature that could break
   from this change. For each, state why it is or isn't affected.

3. CONFIG CHECK: Confirm zero new hardcoded values. Every new constant
   must be in system_config.py with validation, range, and default.

4. TEST CHECK: Confirm a unit test exists for the fix AND a
   system/regression test confirms nothing else broke. Report counts.

5. SYNTAX CHECK: Run `python -c "import <module>"` for every changed
   module. Report PASS/FAIL.

6. EVIDENCE: State every conclusion with evidence (output, number,
   file:line). No "probably" / "should be" / "looks fine".

7. VERDICT: One of —
   ✅ SAFE TO COMMIT (all 6 pass)
   ⚠️ NEEDS WORK (list exactly what)
   🔴 DO NOT COMMIT (explain the risk)

Do not commit. Only report. Eyal decides.
