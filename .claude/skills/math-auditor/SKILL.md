---
name: math-auditor
description: >
  Run an adversarial mathematical-accuracy review of a Julia package's src/ and
  test/ directories, producing a dated markdown report plus concise, self-contained
  fix-prompt markdowns suitable for handing to a smaller model (e.g. Sonnet) in a
  later session. Use this skill whenever the user asks for an adversarial review,
  a math audit, a full code review focused on mathematical/numerical correctness,
  a check of algorithmic consistency with the literature, or says things like
  "review the math in src/", "is the algebra right?", "audit the update equations",
  "check the statistics/linear algebra for bugs", or "construct a code review as
  markdown". Trigger even when the user does not say "audit" — any request for a
  correctness-focused sweep of a scientific Julia codebase qualifies.
---

# Math Audit

Adversarial review of a scientific Julia package for **mathematical accuracy and
consistency** — not software architecture (flag architecture only when it causes
mathematical wrongness, e.g. mutation aliasing, accidental type demotion, or
inconsistent conventions between modules).

The output is written for the package's own developers: findings must cite exact
`file:line`, state the correct mathematics, and give a concrete failure scenario.
A finding that can't survive an attempt at refutation doesn't ship.

## What "adversarial" means here

Each reviewer's job is to *break* the code, not describe it. Concretely, hunt for:

- **Wrong equations**: emulator predictive mean/covariance formulas, kernel and
  feature-map definitions, MCMC proposal/acceptance ratios, likelihood and prior
  terms that differ from the cited papers or from the docstring's own LaTeX.
  Derive the correct expression independently and diff it against the code.
- **Convention drift**: rows-vs-columns for ensemble members/input points,
  `N-1` vs `N` normalization, factor-of-2 / sign errors, Cholesky `L` vs `U`,
  covariance vs precision, whitened vs unwhitened space, whether decorrelation
  is applied in input- or output-space — especially *inconsistencies between
  modules that must agree* (e.g. `GaussianProcess.jl` vs `ScalarRandomFeature.jl`
  vs `VectorRandomFeature.jl` on the same emulator interface).
- **Statistical validity**: is observational/model noise sampled and scaled
  correctly? Are means/covariances computed over the right dimension (samples
  vs output components)? Do scalar and vector emulator variants agree in
  expectation? Do the SVD/PCA-based decorrelation and elementwise scaling
  transforms in `Utilities/` compose correctly with the emulator and sampler
  that consume them?
- **Numerical soundness**: unguarded `inv`/`\` on possibly-singular matrices,
  loss of symmetry/PSD-ness, subtraction-based variance formulas, missing
  regularization/jitter, `sqrt` of negative-by-roundoff eigenvalues.
- **Edge cases the math must survive**: single training point, single output
  dimension (scalar vs matrix degeneracy), zero variance, degenerate/rank-
  deficient covariance after decorrelation, empty or singleton minibatches,
  MCMC chains that reject every proposal.
- **Test-math consistency**: do the tests actually pin the mathematics
  (analytic solutions, invariants, convergence rates), or just check shapes and
  "it runs"? A wrong equation whose test only checks `size()` is a *double*
  finding: the bug and the missing test.

## Workflow

### 1. Partition

First establish what is actually **live**: grep the top-level module (and the
files it includes, recursively) for `include(...)`, and compare against
`git ls-files src/`. Files never included in the build, or untracked, are dead
code — assign them skim-only status and cap their findings at minor/hygiene
(they can't produce wrong results today, but note them in the report so future
reviewers and greps don't confuse dead sampler math with live code).

List the live `src/*.jl` and `test/**/*.jl` with line counts. Group into 4–8
review units of roughly comparable size, pairing each source module with the
tests that exercise it. Group modules that *share mathematical conventions*
together so the reviewer can catch cross-module inconsistencies — in this
package that typically means: the emulator variants (`Emulator.jl` plus
`MachineLearningTools/GaussianProcess.jl`, `RandomFeature.jl`,
`ScalarRandomFeature.jl`, `VectorRandomFeature.jl`) in one or two units since
they must agree on the same predict/covariance interface; the MCMC samplers in
another; and the `Utilities/` preprocessing transforms (decorrelation,
elementwise scaling, canonical correlation, likelihood-informed subspace) in
another, since they feed both the emulator and the sampler and a convention
mismatch there propagates silently downstream.

Reserve the smallest unit as a **cross-cutting unit**. Alongside its own files
(module wiring, display code, package-wide greps for `dims=`, `corrected=`,
jitter constants, symmetrization idioms), give it an explicit mandate to
independently re-derive the *single highest-risk formula* in each other unit's
territory — MCMC acceptance ratios, structure-matrix congruence transforms,
predictive-covariance noise semantics. This overlap is deliberate: in one
audit the cross-cutting unit independently found and numerically verified the
same critical pCN acceptance-ratio bug as the MCMC unit, on a *different* toy
problem — two matching numerical confirmations settle a critical finding in a
way a single one never can. Tell that agent the overlap is intentional so it
doesn't defer to the "owning" unit.

### 2. Fan out reviewers (parallel agents)

Spawn one agent per unit, in a single message so they run concurrently. Build
each prompt from `references/unit-agent-prompt.md`: fill the placeholders
(unit file list, domain context, unit-specific extra hunts, scratch directory)
and paste in the full "What adversarial means here" list above — agents don't
see this skill. The template carries the rules that have earned their keep in
past audits (the numerical-verification tiering, the warn/throw grep before
any "silent" claim, severity-rule reporting) and fixes the required
three-section output: the findings JSON, a CONVENTIONS section, and a
TEST-COVERAGE section. The CONVENTIONS lines are the raw material for the
step-3 matrix — without them you must re-read the code yourself to build it.

As each agent's report arrives, save its raw findings verbatim to a scratchpad
file (one per unit). A full audit plus verification is long enough that
context summarization mid-run can silently lose findings; the scratchpad files
are the durable record the report is assembled from.

### 3. Verify

For each critical/major finding, attempt refutation before it enters the
report: re-read the cited lines yourself, re-derive the math, and check whether
a test or an upstream transformation already accounts for it (common false
positives: a transpose hidden in a helper, normalization done at construction
time, a convention documented elsewhere, a `@warn` at the constructor that the
reviewer never read — re-run the warn/throw grep yourself for any "silent"
claim). Prioritise findings tagged `verified: inspection`; numerically verified
ones usually need only a sanity re-read. Spawn skeptic agents for findings you
can't settle from the main context. Demote or drop findings that don't survive;
mark surviving ones **CONFIRMED** vs **PLAUSIBLE** (couldn't fully verify).

Then build a small **conventions matrix** from the agents' CONVENTIONS
sections before writing anything: rows = modules, columns = the conventions
they reported (covariance normalization N vs N−1, whitened vs unwhitened
space, whether predictive covariance includes observational noise,
symmetrization idiom, jitter constants, RNG threading, rows-vs-columns). Any
mismatched cell between modules that must agree is a finding candidate in
itself — in practice the worst bugs are a scale factor or transform applied in
one emulator variant or preprocessing step but not its sibling, and they only
become visible side by side.

### 4. Write the report

Create `full-code-review/<YYYY-MM-DD>/` (date from `date +%F`, never from
memory). Before writing, copy every numerical-verification scratch script into
`full-code-review/<date>/verification/<unit>/` and cite those repo-relative
paths in the report — scratchpad paths die with the session, and a numerical
finding whose evidence has evaporated is just an inspection finding with extra
steps.

Write `review.md`:

```markdown
# Adversarial Mathematical Review — <Package> (<date>)
## Scope and method          <!-- files covered, units, verification policy -->
## Summary table             <!-- ID | severity | verdict | file:line | one-line claim -->
## Critical findings         <!-- full detail: evidence, math, failure scenario, fix sketch -->
## Major findings
## Minor findings & hygiene  <!-- terser -->
## Cross-module consistency notes
## Test-coverage gaps        <!-- LEAD with 2-3 "keystone tests" — see below -->
## What was checked and found sound   <!-- credit where due; prevents re-auditing -->
```

In **Test-coverage gaps**, lead with the 2–3 *keystone tests* whose absence
masked the most findings, ranked by findings-caught-per-test. In practice a
handful of missing test archetypes hide many bugs at once — e.g. an analytic
linear-Gaussian posterior test asserting the MCMC mean **and variance** would
have caught two critical sampler bugs in one audit, and a single test of the
default noise-flag semantics would have caught three emulator findings. This
list is the highest-leverage part of the report for maintainers.

Findings get stable IDs used everywhere, including fix prompts: `C1…` critical,
`M1…` major, `m1…` minor, `h1…` hygiene; grouped-minor fix prompts get `G1…`.

### 5. Write fix prompts

For each finding with an actionable fix (usually critical + major, plus grouped
minors), write `full-code-review/<date>/fix-prompts/<ID>-<slug>.md`. These are
consumed by a *smaller model in a fresh session with no context*, so each must
be self-contained:

```markdown
# Fix <ID>: <one-line title>
**File**: `src/Foo.jl`, function `bar!`, around line NNN.
**Problem**: <2–4 sentences: what the code does vs what the math requires.
Include the incorrect snippet verbatim.>
**Required change**: <exact edit, or precise description with the correct formula>
**Do not**: <guardrails — e.g. "do not change the API", "do not touch other methods">
**Verify**: <the test to run or add, with the invariant it should pin>
```

Keep each under ~40 lines. One finding per file; group only truly mechanical
repeats (e.g. the same typo pattern in five docstrings) into one prompt.
Quote enough of the offending snippet that the fixer can locate it by function
name + snippet — line numbers drift between the audit and the fix session, so
present them as hints, not anchors.
Also write `fix-prompts/README.md` listing prompts in recommended application
order (independent fixes first, same-file prompts sequenced, conflicting ones
flagged) and naming which fixes *intentionally change numerical results* — so
the fixer checks a failing loose regression test against the analytic reference
in the prompt before "fixing" the test.

### 6. Report back

Final message: lead with the headline (how many confirmed critical/major
findings and the single worst one), then the report path, then a compact
summary table. Do not paste the whole report into the chat.

## Calibration

- Severity: **critical** = produces mathematically wrong results in mainstream
  use; **major** = wrong in common configurations or silently degrades
  statistical properties; **minor** = wrong in edge cases, misleading docs
  math, dead/misnamed math; **hygiene** = style-level (only if math-adjacent).
- A numerically demonstrated wrong stationary distribution, wrong acceptance
  ratio, or wrong variance scaling in *exported* functionality is **critical**
  even if the feature looks niche — samplers and estimators exist to produce
  distributions, and a wrong distribution is the worst failure they have.
  (Agents report which calibration rule they applied, so when two units rate
  the same bug differently you arbitrate on the rule, not the adjective.)
- A docstring–code mismatch is a real finding even when the code is right —
  users implement against docstrings.
- A statistically unjustified combination that the package explicitly warns
  about (e.g. a constructor `@warn "... experimental ..."`) caps at **minor**
  unless the warning itself is wrong — the trap isn't silent.
- Don't pad. If a unit is sound, the report says so in one paragraph; an audit
  that cries wolf gets ignored next time.

## Improving this skill

After delivering the report, offer: "Would you like to improve the
**math-auditor** skill itself using skill-creator? You can share suggestions, or
I can analyse this run — finding quality, false-positive rate, fix-prompt
usability — to refine the skill for next time."
