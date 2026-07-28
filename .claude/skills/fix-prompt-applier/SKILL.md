---
name: fix-prompt-applier
description: >
  Apply fixes described in full-code-review/<date>/fix-prompts/<ID>-<slug>.md
  files — the self-contained fix prompts produced by the math-auditor skill
  after an adversarial mathematical review — and, once a fix is verified, mark
  the finding as resolved in the parent review.md so the review stays a living
  record of what's fixed vs. still open. Use this skill whenever the user asks
  to fix, implement, or apply a finding by its ID (e.g. "fix C1", "apply the
  M7 fix", "implement the fix-prompts for the likelihood-informed processor"),
  asks to work through a review's fix-prompts/README.md in order, or
  references a full-code-review directory at all. Trigger even if the user
  doesn't say "fix-prompt" explicitly — any request to act on findings from an
  existing math/code review report qualifies.
---

# Fix-Prompt Applier

Counterpart to **math-auditor**: math-auditor produces a review plus a
directory of self-contained fix prompts; this skill consumes them, usually in
a later session (or a smaller model) with none of the audit's context. Every
fix prompt was written assuming exactly that — treat each one as the full
brief, not a pointer into a conversation you don't have.

## Workflow

### 1. Locate and read before touching any code

- Find the fix-prompt file(s) for the requested ID(s) under
  `full-code-review/**/fix-prompts/`. IDs are stable (`C1…` critical, `M1…`
  major, `m1…` minor, `h1…` hygiene, `G1…` grouped-minor).
- If `fix-prompts/README.md` exists, read it first: it gives the recommended
  application order, flags same-file prompts that must be sequenced together,
  and — importantly — names which fixes **intentionally change numerical
  results**. That last part matters at verification time: a regression test
  that starts failing after such a fix isn't a new bug, it was pinned to the
  old, wrong behavior.
- Read each fix-prompt file **in full**. They follow a fixed shape (File /
  Problem / Required change / Do not / Verify) precisely so nothing besides
  the prompt itself is needed — don't skim just the required-change line and
  move on.

### 2. Apply

- Find the offending code by the function name and snippet quoted in the
  prompt. Line numbers drift between the audit and the fix session — treat
  them as hints, not anchors; the quoted snippet is what actually locates it.
- Make exactly the described change. Read the "Do not" guardrails literally —
  they exist because the prompt's author already considered the tempting
  adjacent change (a broader refactor, an API change, "fixing" a neighboring
  function too) and rejected it. A fix prompt is deliberately narrow, usually
  one finding in ~40 lines; don't expand scope while you're in there.
- When several fix prompts touch the same file (e.g. two findings in the same
  function, or in the same processor), read all of them before editing any of
  them, and land the edits together so the file is mathematically consistent
  at every intermediate step — don't let one fix's change invalidate another
  prompt's line references or assumptions about the surrounding code.
- If a prompt's description seems ambiguous, or seems to contradict what the
  current code actually does (code drifts between the audit and the fix
  session too), re-read the source and re-derive the math yourself rather than
  guessing. The prompt is the best available hint; the source is ground truth.

### 3. Verify

- Run, or add, exactly the test or invariant the prompt's "Verify" section
  names — this is the thing that pins the fix and stops the bug from silently
  coming back.
- If a pre-existing test fails after the change, check the README (or the
  prompt itself) for a note that this fix intentionally changes numerical
  results before treating the failure as a regression. If it's flagged, update
  the test's expected/reference value to the correct analytic or numerical
  value the prompt derives — don't revert the fix to make the old value pass
  again.
- Beyond the named test, run the package's broader relevant test suite (the
  module's test file at minimum) to catch collateral breakage the narrow
  Verify step wouldn't show.

### 4. Update `review.md` so it stays a living record

A fix-prompt is a snapshot of a bug at audit time. Once it's fixed and
verified, the parent review should say so — otherwise the next reader
(including a future instance of you, or the smaller model this skill was
written for) re-discovers a "bug" that's already closed.

- Find `review.md` next to the fix-prompts directory (`../review.md` relative
  to `fix-prompts/`, i.e. `full-code-review/<date>/review.md`). If it doesn't
  exist — e.g. someone handed you a fix-prompt file on its own — skip this
  step rather than erroring; there's nothing to update.
- Get today's date from the shell (`date +%F`), the same way math-auditor
  dates its report directories. Don't recall or guess a date from memory —
  it's cheap to get exactly right and confusing to get wrong.
- In the **summary table**, mark the fixed finding's row, e.g. append
  `— FIXED (<date>)` to its Verdict cell. Reuse the Verdict column rather than
  inventing a new one, unless the table already has a Status column or the
  Verdict cell would get unreadably cluttered.
- In the finding's own detailed section (under Critical/Major/Minor/Hygiene
  findings), append — don't rewrite — a short status note directly below the
  existing text:
  ```
  **Status**: Fixed <date> — <1-3 sentences: what changed, `file:line`, which
  test/invariant now pins it>. Applied via the fix-prompt-applier skill.
  ```
  Leave the original finding text (evidence, math, failure scenario) exactly
  as it is — that's the historical record of what was wrong and how it was
  found. review.md is meant to double as an audit trail: a reader should see
  both what the bug was and that it's now closed, without losing either.
- If a single pass fixed several findings (e.g. two in the same file), update
  all of their rows and sections together so the document reflects the whole
  batch consistently, not a half-updated intermediate state.

### 5. Report back

For each fix ID applied, state: what changed and where (`file:line`), which
test/invariant now pins it, whether any existing test's expected value needed
updating (and why, citing the README/prompt note that justified it), and
whether `review.md` was updated (or why it was skipped). Keep this compact —
one or two lines per ID — the point is a scannable record of what's fixed and
what now guards it, not a re-explanation of the math (that's already in the
review and the prompt).

## Improving this skill

After applying fixes, offer: "Would you like to improve the
**fix-prompt-applier** skill itself using skill-creator? You can share
suggestions, or I can analyze this session — where a prompt was ambiguous,
where scope crept, whether verification actually caught anything — to refine
the skill for next time."
