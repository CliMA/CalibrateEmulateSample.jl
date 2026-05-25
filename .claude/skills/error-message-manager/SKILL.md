---
name: error-message-manager
description: >
  Rewrite vague, delayed, or low-context Julia error messages into structured,
  actionable diagnostics. Invoke this skill whenever the user mentions: error
  message, improve errors, rewrite @assert, ArgumentError, DimensionMismatch,
  DomainError, vague error, error rewrite, Julia exception, diagnostic, throw,
  validation, early check, assert to throw, loop context, catch and rethrow,
  warn string, or asks to improve how the code fails. Also use it when reviewing
  code for user-facing clarity, when a user says errors are confusing or
  unhelpful, or when auditing a module for low-context exceptions. Use it
  proactively when you see bare @assert, error("..."), throw(ErrorException(...)),
  @warn string(...), or catch blocks that do not include the original exception
  in their re-throw in Julia code you are reading or editing.
---

# error-message-manager

Rewrite vague, delayed, or low-context Julia error messages into structured,
actionable diagnostics. The goal is errors that tell the user exactly what went
wrong, what was expected, what was received, and—whenever a likely fix exists—
what to do next. Prefer catching mistakes early (at API boundaries) over letting
them propagate into cryptic numerical failures.

## Workflow

### Step 0 — Offer an Explore agent for multi-file scope

If the user's request covers more than one file — a whole directory, a module, or
the entire repo — offer to spawn an Explore agent before doing any file reads
yourself. The agent runs all the reads in parallel without flooding the main
context, and returns a structured inventory you can act on directly.

**When to offer**: any time the target is a directory path (e.g. `src/`) or a
vague scope like "the whole package" or "all the source files".

**Offer text** (adapt as needed):
> "This spans multiple files — I'd recommend spawning an Explore agent to survey
> all `throw`/`@assert`/`error` sites in parallel. It keeps the audit fast and
> leaves the main context clean for the actual rewrites. Want me to do that?"

**Agent prompt to use** (fill in `<path>` and `<package_name>`):

```
Audit `<path>` for error-raising patterns. Before reading individual files, grep
for two structurally-broken patterns that are the highest priority to fix:

  rg -n 'throw\s*\(\s*[A-Z]\w+\s*,' <path>   # throw(Type, "string") — always MethodError
  rg -n 'throw\s*\(\s*"' <path>              # throw("raw string") — throws non-exception

Then, for every `@assert`, `error(`, or `throw(` site in every `.jl` file:

1. Record: file, line number, exception type (or "bare @assert" / "bare error"),
   and the full message text (including multiline strings).
2. Classify message quality:
   - "broken" — structurally wrong throw that will never produce the intended
     exception: `throw(Type, "string")` (always MethodError), or `throw("string")`
     (throws a raw String, not catchable as a typed exception)
   - "good"  — has `$(expr)` interpolation showing the actual received value, and
     is either short (≤3 lines) or already in a `_throw_` helper function
   - "long-inline" — message content is good, but the message string body exceeds
     3 lines and the throw is written inline (not in a `_throw_` helper)
   - "vague" — missing a received value, or no Expected/Got structure
   - "missing" — bare `@assert` with no message at all
3. Note whether the site is at an API boundary (user-facing input) or an internal
   invariant (would require a package bug to fire).

Return a markdown table with columns:
  File | Line | Exception type | Quality | Notes (one-line note on what's wrong)

Focus only on sites that are "broken", "vague", "missing", or "long-inline" — skip "good" ones.
Fix "broken" sites first — they silently misbehave rather than just being unhelpful.
```

**How to use the result**: treat the returned table as your working inventory for
Steps 1 and 2. You do not need to re-read the flagged files yourself to classify —
go straight to reading only the lines that need rewrites (Step 3 onwards).

---

### Step 1 — Audit the target scope

Identify which files or functions to address. If the user named a specific
function, start there. If the request is repo-wide, run:

```
rg -n '(@assert[^(]|@assert\(|error\(string\(|throw\(ErrorException)' src/
```

Then collect all message-less `@assert` calls:

```
rg -n '@assert' src/ | grep -v '"'
```

Also flag `@warn` calls that use string concatenation instead of interpolation:

```
rg -n '@warn\s+string\(' src/
```

And flag `catch` blocks that discard the original exception when re-throwing:

```
rg -n 'catch\s' src/
```

For each `catch` hit, check whether the subsequent `throw` or `error` call
interpolates the caught variable (e.g. `$e` or `sprint(showerror, e)`). If it
does not, the original exception type and message are silently lost.

For each hit, record: file, line, the condition being checked, and whether it
guards user-provided input (API boundary) or an internal invariant.

### Step 2 — Classify each site

Use this table to choose the right exception type:

| Condition | Exception |
|---|---|
| Invalid user-provided argument | `ArgumentError` |
| Array/matrix shape mismatch | `DimensionMismatch` |
| Inconsistent argument types across parameters | `ArgumentError` |
| Mathematically invalid value (negative variance, etc.) | `DomainError` |
| Invalid index | `BoundsError` |
| Internal invariant that should never fire | `error(...)` |
| Missing interface implementation | `MethodError` or structured `ArgumentError` |

Avoid `ErrorException` unless there is no better choice.

**Type mismatches vs dimension mismatches**: an `@assert isa(x_mean, AbstractVector{FT})` inside an
`if isa(x, AbstractMatrix{FT})` branch is checking that the user supplied *consistent* arguments
(matrix ensemble → vector mean), not that two arrays have matching sizes. Use `ArgumentError`, not
`DimensionMismatch`, for this pattern.

Distinguish **API boundary** sites (where the user passed something wrong — prefer
typed exceptions with actionable messages) from **internal invariant** sites
(where a bug in the package itself would have to exist — bare `error(...)` with a
clear note is fine there).

**Loop-body errors**: if the throw is inside a `for` or `while` loop, treat the
loop index and key per-iteration state as required context. Without this, the user
sees "matrix is not positive definite" with no idea whether it happened on
iteration 2 or iteration 200. Always capture `i` (or the loop variable) and the
state that changed between iterations — the ensemble step count, the parameter
vector being updated, the ensemble member index, etc. For *nested* loops, include
both the outer and inner loop variables — the outer variable says which group or
batch failed; the inner variable says which element within it failed. See the
loop-context example in the Canonical examples section below.

**`catch e` losing the original exception**: when a Julia exception is caught and
a new one is thrown, the new message must include the original exception. If it
does not, the user loses the root cause (e.g. `PosDefException`, `SingularException`)
and has no way to distinguish a code bug from a numerical issue. Use
`sprint(showerror, e)` rather than `$e` alone — it formats the exception type and
message together:

```julia
# anti-pattern — root cause vanishes
catch e
    throw(ArgumentError("Matrix factorization failed."))
end

# correct — root cause preserved
catch e
    throw(ArgumentError("""
Matrix factorization failed.

Caused by: $(sprint(showerror, e))

Suggestion:
    ...
"""))
end
```

Only suppress the original exception if it is a well-known internal Julia error
(e.g. `SingularException`) and you are intentionally providing a higher-level
fallback — and even then, log it at `@debug` level.

**`@warn string(...)` concatenation**: `@warn string("...", x, "...")` is the
warning-side equivalent of `error(string(...))` — it's noisy, hard to read, and
doesn't benefit from Julia's interpolation. Rewrite as `@warn "... $x ..."`.
`@warn` messages that use structured strings are also easier to grep and suppress
selectively.

**Double-gated invariants**: if a helper is only ever called after the public API has already
checked the same condition (e.g., `get_vector_of_parameterized` is called from `construct_prior`
only when `d.args[1] == Symbol("VectorOfParameterized")` is true), the check inside the helper is
an internal invariant even though it looks like a user-data check. Use a single-line `error(...)`
rather than a full structured `ArgumentError`:

```julia
# internal invariant — the caller already validated this
d.args[1] == Symbol("VectorOfParameterized") || error(
    "Internal error: get_vector_of_parameterized called with non-VectorOfParameterized expression (got $(d.args[1]))",
)
```

### Step 2.5 — Decide: inline or helper?

Before writing the rewrite, decide whether the error belongs inline or should be
extracted into a `_throw_<what>(...)` helper function.

**When to extract** — pull the error into a helper when either condition holds:

- **Length** (primary trigger): the message body exceeds 3 lines. Extract
  unconditionally — single call site, non-loop context, no surrounding complexity
  required. A full Expected / Got / Suggestion block always crosses this threshold.
  Even a one-off long block left inline establishes a pattern that makes entire files
  hard to scan, and accumulates quickly once a few exceptions are made.
- **Duplication**: the same error shape (same summary line, same Expected / Got /
  Suggestion skeleton) appears at ≥2 call sites. Extract even when each block is
  short — the wording drifts silently over time and the call sites collapse to
  readable one-liners.

Inline is appropriate only for genuinely short messages (≤3 lines) at a single
call site. A single summary line, or a summary plus one Got line, is the ceiling
for inline. When in doubt, count — if it doesn't fit in 3 lines, extract.

Count lines of the *message string body*, not lines of the surrounding `throw(...)`
call. A single-line message formatted as `throw(\n  ArgumentError(\n    "msg"\n  )\n)` across
five code lines is still a 1-line message and does not require extraction.

**Where helpers go**

Default: a `## Error helpers` section at the **bottom of the source file**, above
`end # module`. Keeping helpers near their callers preserves traceability — the
reader sees the throw site, jumps to the bottom of the same file, and finds the
message without switching files.

Promote to a shared `src/ErrorMessages.jl` (or the repo's equivalent top-level
utility file) only when **≥2 different source files** call the same helper. Discover
which file to use by reading the top-level module file (e.g. `src/PackageName.jl`)
for its `include(...)` list — then add `include("ErrorMessages.jl")` as the first
`include` so every subsequent file sees the helpers without any `using`/`import`.

**Include-chained files**: if the two source files are both `include`-d by a common
parent file (e.g. `ScalarRandomFeature.jl` and `VectorRandomFeature.jl` are both
included by `RandomFeature.jl`), add the shared helper to that parent file *before*
the `include(...)` lines — no new file needed, and no `using`/`import` required.

**Naming convention**

```
_throw_<what>(positional_required_facts...; kwargs_for_optional_context...)
```

- Underscore prefix → unexported private helper.
- Verb prefix `_throw_` → the function unconditionally raises; callers know there
  is no return value.
- Suffix describes the failure mode: `_dim_mismatch`, `_missing_keys`,
  `_bad_obs_type`, `_not_iterable`.

**Signature convention**

Pass the facts that are *always* present as positional arguments (the offending
value, the expected vs got summary). Pass *optional* context as keyword arguments
with `nothing` defaults — especially loop context (`index`, `total`, `iter`,
`phase`). Build optional sections inside the helper by checking `isnothing(...)`.
This keeps call sites compact and lets the same helper serve both loop and non-loop
contexts (see the *Helper with optional loop context* canonical example).

**Performance: use `@noinline`**

Prefix every helper with `@noinline`. This prevents Julia from inlining the cold
error path into the surrounding hot code, keeping numerical kernels unaffected:

```julia
@noinline function _throw_x_not_iterable(x; where::Symbol)
    throw(ArgumentError(...))
end
```

**What NOT to do**

- Don't create a catch-all `_throw_arg_error(msg::String)` — that just shifts the
  inline triple-quoted block to another file without any DRY benefit.
- Don't use macros (`@check_dim(...)`) — they're magical and harder to debug than
  plain functions.
- Don't bundle all context into one opaque `context::NamedTuple` — explicit kwargs
  are clearer to call and easier to extend.

### Step 3 — Rewrite with the canonical layout

Use this structure for every user-facing exception:

```julia
throw(ArgumentError("""
Short one-line summary of the failure.

Expected:
    <what would have been valid>

Got:
    <what was actually received, with interpolated values>

Loop context:
    iteration  = $iter (of $n_iter)
    <key per-iteration state variable> = $(summary_of_state)

Context:
    <surrounding state that helps locate the problem>

Suggestion:
    <most likely fix>
"""))
```

Section rules:
- **Summary**: always present; one line; imperative or declarative.
- **Expected / Got**: strongly preferred for any mismatch check; use `$(expr)`
  interpolation to show actual values.
- **Loop context**: include whenever the throw is inside a `for` or `while` loop.
  Always report the loop index and the key state that varies between iterations
  (e.g., the EKI step number, the ensemble member index, or the parameter being
  updated). This is what lets the user reproduce the failure without adding
  `println` debugging. Omit for errors that can only fire at a fixed point in the
  code (before the loop starts or after it ends).
- **Context**: include when the same error can arise from multiple call sites and
  naming the calling function or struct helps the user orient.
- **Suggestion**: include whenever a likely fix exists. Omit rather than write a
  generic platitude.
- Never dump full matrices or large arrays. Prefer `size(x)`, `eltype(x)`,
  `typeof(x)`, or a scalar summary statistic.

### Step 4 — Move validation early

If the current code lets an invalid input reach a numerical routine before
failing (e.g., `cholesky` on a non-symmetric matrix, `inv` on a singular one),
add an explicit guard at the API boundary:

```julia
# Before: error surfaces deep in cholesky
cov_chol = cholesky(C)

# After: check at the boundary, raise immediately
issymmetric(C) || throw(ArgumentError("""
Covariance matrix must be symmetric.

Got:
    size(C) = $(size(C))
    norm(C - C') = $(norm(C - C'))

Suggestion:
    Pass a symmetric matrix, e.g. `C = (C + C') / 2`.
"""))
cov_chol = cholesky(C)
```

Use `||` for single-condition guards. For multi-condition guards, use `if/throw`.

When using `||` with a multiline triple-quoted throw, the closing `))` goes on its own line
immediately after the closing `"""`:

```julia
condition || throw(ArgumentError("""
Summary line.

Expected:
    ...

Got:
    ...
"""))   # ← closing )) on the line right after the closing """
```

This is the only layout that keeps indentation correct — triple-quoted strings in Julia do not
strip leading whitespace, so indenting the message body would include those spaces in the string.

### Step 5 — Preserve domain language

Write messages in terms the user understands, not in terms of internal Julia
dispatch or linear algebra internals. For example:

- Say "ensemble member count" not "size(x, 2)"
- Say "parameter covariance matrix" not "the second argument to cholesky"
- Say "observation noise covariance" not "Γ_y"
- Say "emulator training samples" not "io_pairs column count" or "n_train"
- Say "encoder pipeline" not "encoder_schedule vector"
- Say "fraction of variance to retain" not "retain_var threshold"
- Say "MCMC step size" or "proposal variance" not "proposal standard deviation scalar"
- Say "data processor" when referring to `ElementwiseScaler`, `Decorrelator`,
  `CanonicalCorrelation`, or `LikelihoodInformed` — not "utility type" or "struct"

### Step 6 — Apply rewrites

Edit each site, keeping the surrounding code untouched. Confirm the package
still loads:

```
julia --project -e 'using CalibrateEmulateSample'
```

### Step 7 — Add @test_throws tests

#### Step 7a — Scan for stale @test_throws (run this before editing any source line)

Whenever you change an exception type (e.g. `ErrorException` → `ArgumentError`),
run this grep **before** touching the source file:

```bash
grep -rn '@test_throws OldType' test/
```

Every hit is a test that will break or silently go stale the moment your source
edit lands. Update each hit **in the same edit as the source change** — never as a
follow-up. This is not optional: a stale `@test_throws WrongType` either fails CI
on the new code or passes forever without catching a regression — either way it
no longer protects anything.

This step is easy to skip because source and tests live in different files and the
stale test produces no compile error — it only surfaces when the test suite runs.
Running the grep first makes the problem visible before it bites you.

#### Step 7b — Check existing coverage

Before writing any test, check whether coverage already exists. Grep the matching
`test/<module>/runtests.jl` for the public API function that reaches the rewritten
site:

```bash
grep -n '@test_throws' test/<module>/runtests.jl | grep '<function_name>'
```

Three outcomes:

| Situation | Action |
|---|---|
| `@test_throws <correct_type>` already present | Skip — do not add a duplicate |
| `@test_throws <wrong_type>` already present | Update the existing line to the new type |
| No coverage at all | Add a new test |

**Check message content, not just type**: In Julia 1.8+, `@test_throws` returns
the caught exception, so you can pin key diagnostic text in the same block:

```julia
let thrown = @test_throws ArgumentError f(bad_input)
    @test contains(thrown.value.msg, "retain_var")       # Got section present
    @test contains(thrown.value.msg, repr(bad_input))    # received value interpolated
end
```

Add at least one `contains` check per new error site. Without this, a refactor can
preserve the exception type while silently dropping the Got section or the
interpolated value, and no test will catch it. Check for: a phrase from the summary
line, the Got section label (e.g. `"retain_var"`, `"ensemble size"`), and
`repr(bad_value)` when the message uses `repr`. The `let` block is the cleanest
form — it keeps the type assertion and the content assertions co-located.

For every site that needs a new test, add it in the matching `test/<module>/runtests.jl`:

```julia
let thrown = @test_throws ArgumentError Decorrelator(rand(5, 10); retain_var = -0.1)
    @test contains(thrown.value.msg, "retain_var")
    @test contains(thrown.value.msg, "-0.1")
end
```

Use the specific exception type — never bare `@test_throws Exception`. The test
should construct the minimal invalid input that triggers the new error, without
duplicating happy-path coverage.

**Testing unexported helpers.** If the site is inside an unexported helper (e.g.,
`construct_constraint`, `construct_2d_array`), do not `import` the internal
directly. Instead, test through the nearest exported public API function that
calls it, using invalid input that propagates to the helper:

```julia
# construct_constraint is unexported — test via get_parameter_distribution
no_constraint_dict = Dict("uq_param" => Dict("prior" => "Parameterized(Normal(0.0, 1.0))"))
let thrown = @test_throws ArgumentError get_parameter_distribution(no_constraint_dict, "uq_param")
    @test contains(thrown.value.msg, "constraint")
end
```

This keeps tests coupled to the public contract and avoids brittleness when
internal function names change.

### Step 8 — Offer to improve the skill

Once the rewrites and tests are clean, offer: "Would you like to improve the
**error-message-manager** skill itself using skill-creator? You can share
suggestions, or I can analyse patterns from this session—recurring edge cases,
exception-type decisions, or anything that felt awkward—to refine the skill for
next time."

---

## Style rules

- **Triple-quoted strings** for all multiline messages.
- **No full matrix dumps**. Use `size(x)`, `eltype(x)`, `norm(x - ...)`, or
  `extrema(x)` instead.
- **Interpolate actual values** in Got sections so the user sees the numbers,
  not just variable names. For `String`-typed arguments use `$(repr(x))` rather
  than `$(x)` — it adds the surrounding quotes so the output clearly reads as a
  string value (e.g. `Got: sigma_points = "bad"` instead of `Got: sigma_points = bad`).
- **Raise early**: prefer guarding at the function entry point over deep inside a
  helper.
- **No `@assert` for user-facing validation**. `@assert` is a debugging tool;
  it can be compiled out. Use explicit `throw` instead.
- **Loop state in Got / Loop context**: when a throw is inside a `for` or `while`
  loop, always name the iteration index and the key state from that iteration.
  A "convergence failed" message without the step count forces the user to add
  `println` debugging to reproduce the failure. When the same loop-context check
  recurs at multiple sites, the loop variables become optional kwargs on a
  `_throw_<what>` helper — see the *Helper with optional loop context* canonical
  example.
- **Preserve the original exception in `catch` blocks**: if you catch `e` and
  throw a new exception, include `$(sprint(showerror, e))` in the new message.
  Dropping `e` silently discards the root cause.
- **`@warn` with interpolation, not `string()`**: replace `@warn string("x=", x)`
  with `@warn "x = $x"`. String concatenation in warnings is harder to read and
  grep.
- **Single-line messages are fine** when the failure is unambiguous and no
  Expected/Got context would add clarity.
- **Extract into `_throw_<what>(...)` helpers** whenever the message body exceeds
  3 lines, or when the same Expected / Got / Suggestion skeleton appears at ≥2
  call sites (even if short). A full Expected / Got / Suggestion block always
  exceeds 3 lines and must be a helper — inline is only appropriate for ≤3-line
  messages at a single call site. Place the helper in a `## Error helpers` section
  at the bottom of the source file; promote to a shared `src/ErrorMessages.jl` only
  when ≥2 different source files share the helper. Use `@noinline`, positional args
  for required facts, and `nothing`-defaulted kwargs for optional context such as
  loop indices. Render each optional section only when its kwarg is non-`nothing`.

---

## Canonical before/after examples

> **Length rule applies to all examples below.** Each example shows the canonical
> message *format* (Expected / Got / Suggestion sections, interpolation, etc.). When
> the message body exceeds 3 lines — which any structured block with Expected / Got /
> Suggestion sections does — the throw must go in a `_throw_<what>(...)` helper per
> Step 2.5, not inline. The first example below models this explicitly. Subsequent
> examples show the message body format; apply the same helper extraction whenever
> the resulting message exceeds 3 lines.

### Replace a vague `error(string(...))`

The after-message has 10 lines (well above the 3-line threshold), so it goes into
a `_throw_` helper — extract unconditionally at this length even though there is
only one call site.

```julia
# Before — in a data-processor constructor, a vague dimension-reduction guard
if retain_var <= 0.0 || retain_var > 1.0
    error(string("retain_var=", retain_var, " is invalid."))
end

# After — helper in the ## Error helpers section at the bottom of the file
@noinline function _throw_invalid_retain_var(retain_var)
    throw(ArgumentError("""
Invalid `retain_var` for dimension reduction.

Expected:
    0 < retain_var ≤ 1.0

Got:
    retain_var = $retain_var

Suggestion:
    `retain_var` is the fraction of explained variance to retain after SVD
    truncation. Pass a value in (0, 1] — for example, `retain_var = 0.95`
    to keep 95% of variance.
"""))
end

# Call site collapses to a single guard line:
(0.0 < retain_var ≤ 1.0) || _throw_invalid_retain_var(retain_var)
```

### Replace a bare `@assert` on an API boundary

```julia
# Before
@assert(haskey(param_info, "constraint"))

# After
haskey(param_info, "constraint") || throw(ArgumentError("""
Parameter info dict is missing the required "constraint" key.

Got keys:
    $(collect(keys(param_info)))

Suggestion:
    Ensure the TOML entry for this parameter includes a `constraint = ...` field.
"""))
```

### Replace a single-line string-value error (use `repr`)

```julia
# Before
throw(ArgumentError("sigma_points type is not recognized. Select from \"symmetric\" or \"simplex\". "))

# After
throw(ArgumentError("""
Unrecognized sigma_points type.

Expected:
    "symmetric" or "simplex"

Got:
    sigma_points = $(repr(sigma_points))
"""))
```

Using `repr(sigma_points)` rather than `$(sigma_points)` keeps the string
quotes visible in the output, making it unambiguous that the user passed a
`String` value (and making copy-paste errors easy to spot).

### Replace a dimension-mismatch `@assert`

```julia
# Before
@assert size(x, 2) == length(mean_weights)

# After
size(x, 2) == length(mean_weights) || throw(DimensionMismatch("""
Ensemble size does not match the number of quadrature weights.

Expected:
    size(x, 2) == length(mean_weights)

Got:
    size(x, 2) = $(size(x, 2))
    length(mean_weights) = $(length(mean_weights))
"""))
```

### Preserve the original exception when catching and re-throwing

```julia
# Before — PosDefException or SingularException silently discarded
try
    cov_chol = cholesky(cov_u)
catch e
    error("Covariance matrix factorization failed.")
end

# After — root cause preserved, matrix state shown
try
    cov_chol = cholesky(cov_u)
catch e
    throw(ArgumentError("""
Covariance matrix factorization failed during empirical Gaussian sampling.

Got:
    size(cov_u)    = $(size(cov_u))
    isposdef(cov_u) = $(isposdef(cov_u))

Caused by: $(sprint(showerror, e))

Suggestion:
    The ensemble may have collapsed. Pass a non-zero `inflation` keyword
    argument to regularise the sample covariance.
"""))
end
```

`sprint(showerror, e)` formats as `"LinearAlgebra.PosDefException: matrix is not
Hermitian; Cholesky factorization failed."` — far more informative than `string(e)`.
Only suppress `e` when you are intentionally providing a higher-level fallback (e.g.
falling back to `pinv`) and still emit it at `@debug` level.

### Rewrite `@warn string(...)` to use interpolation

```julia
# Before
@warn string("Emulator output covariance has negative eigenvalues.", "\n Clamping to zero.")

# After
@warn "Emulator output covariance has negative eigenvalues — clamping to zero before sampling."

# Before (with values)
@warn string("SVD truncation removed ", n_removed, " dimensions (retain_var=", retain_var, ").", "\nCheck that retain_var is not too aggressive.")

# After
@warn "SVD truncation removed $n_removed dimension$(n_removed == 1 ? "" : "s") (retain_var=$retain_var). Reduce retain_var if emulator accuracy suffers."
```

### Add loop context to an error thrown inside an iteration loop

```julia
# Before — user sees "Cholesky factorization failed" with no idea which output dim
for j in 1:n_out
    try
        cov_chol = cholesky(C_j)
    catch e
        error("Cholesky factorization failed")
    end
end

# After — guard before cholesky, expose output dimension index and diagnostic state
for j in 1:n_out
    isposdef(C_j) || throw(ArgumentError("""
Emulator output covariance is not positive definite for output dimension $j.

Expected:
    A positive-definite covariance matrix for every output dimension.

Got:
    output dimension = $j / $n_out
    size(C_j)        = $(size(C_j))
    minimum eigval   = $(minimum(eigvals(Symmetric(C_j))))

Suggestion:
    Increase `alg_reg_noise` in the GaussianProcess constructor to regularise
    the covariance, or check that training data for output $j is not degenerate.
"""))
    cov_chol = cholesky(C_j)
end
```

Key points:
- **Move the guard before the failing call** so the message fires with the full
  loop state still in scope. Catching a `PosDefException` after the fact and
  re-throwing loses the loop index and the matrix state.
- **Report the loop variable** (`j`, `i`, `iter`) and its upper bound so the user
  knows whether the failure is early (dimension 1/10, likely degenerate training data)
  or late (dimension 9/10, likely numerical accumulation).
- **Include one diagnostic scalar** — the minimum eigenvalue, the norm of the
  update step, the condition number — rather than dumping the full matrix.

When this same loop-context error needs to be thrown at multiple sites, the loop
variables (`i`, `N_iter`) naturally become optional kwargs on a `_throw_<what>`
helper. The call site stays a single line and the loop-awareness travels with the
helper everywhere it is used — see the *Helper with optional loop context* example
below.

### Extract a duplicated error into a helper

`encode_data` and `decode_data` in `src/Emulator.jl` each validate the same
precondition: that the input array has the right number of dimensions. Before
extraction, nearly identical 10-line blocks appear at both call sites:

```julia
# Before — same shape check in both encode_data and decode_data

# in encode_data:
if !(x isa AbstractMatrix)
    throw(ArgumentError("""
encode_data: input must be a matrix.

Expected:
    An AbstractMatrix with one column per sample.

Got:
    typeof(x) = $(typeof(x))
    ndims(x)  = $(ndims(x))

Suggestion:
    Reshape your input to a matrix — use `reshape(x, :, 1)` for a single sample.
"""))
end

# in decode_data: byte-for-byte identical except the summary reads "decode_data".
```

After extraction, both functions call a single helper defined once in a
`## Error helpers` section at the bottom of the file:

```julia
# After — helper at the bottom of src/Emulator.jl

## Error helpers

@noinline function _throw_not_matrix(x; where::Symbol)
    throw(ArgumentError("""
$where: input must be a matrix.

Expected:
    An AbstractMatrix with one column per sample.

Got:
    typeof(x) = $(typeof(x))
    ndims(x)  = $(ndims(x))

Suggestion:
    Reshape your input to a matrix — use `reshape(x, :, 1)` for a single sample.
"""))
end

# Both call sites now collapse to a single readable guard line:
function encode_data(encoder_schedule, x)
    x isa AbstractMatrix || _throw_not_matrix(x; where = :encode_data)
    # ... algorithm body visible immediately ...
end

function decode_data(encoder_schedule, x)
    x isa AbstractMatrix || _throw_not_matrix(x; where = :decode_data)
    # ... algorithm body visible immediately ...
end
```

Key points:
- The `where::Symbol` kwarg embeds the calling function name in the message so
  diagnostics stay specific even though the body is shared. Pass a `Symbol` literal
  (`where = :my_func`) — symbols are cheap and render cleanly with `$where`.
- Both functions now have one guard line each instead of a 10-line block; the
  algorithm body is immediately visible.
- `@noinline` keeps the error path out of the hot encode/decode path.
- The helper lives at the bottom of the same file — one jump away, no new file.

### Helper with optional loop context

The `for proc in encoder_schedule` loop in `src/Utilities.jl` applies each data
processor in turn but currently reports no position if one fails — the user sees
"encode failed" with no idea which processor in the pipeline triggered the error.
Extracting into a helper adds the index and makes the same helper reusable wherever
that validation appears:

```julia
# Before — inline block, no loop index in the message
for proc in encoder_schedule
    if !applicable(encode_data, proc, x)
        throw(ArgumentError("""
Data processor does not support encode_data.

Expected:
    All processors in encoder_schedule to implement encode_data.

Got:
    typeof(proc) = $(typeof(proc))

Suggestion:
    Ensure every processor in the encoder_schedule implements encode_data.
"""))
    end
end

# After — helper with optional loop context at the bottom of the file

@noinline function _throw_proc_missing_encode(proc; index = nothing, total = nothing)
    loop_ctx = isnothing(index) ? "" : """

Loop context:
    processor index = $index (of $total)"""
    throw(ArgumentError("""
Data processor does not implement encode_data.$loop_ctx

Expected:
    Every element of encoder_schedule to implement encode_data(proc, x).

Got:
    typeof(proc) = $(typeof(proc))

Suggestion:
    Check that each processor in your encoder_schedule is a recognised data
    processor type (e.g. Decorrelator, ElementwiseScaler, CanonicalCorrelation).
"""))
end

# Call site — loop now reports position:
for (i, proc) in enumerate(encoder_schedule)
    applicable(encode_data, proc, x) ||
        _throw_proc_missing_encode(proc; index = i, total = length(encoder_schedule))
end

# The same helper works outside a loop — omit the kwargs and the Loop context
# section is silently suppressed:
applicable(encode_data, proc, x) || _throw_proc_missing_encode(proc)
```

Key points:
- `index` and `total` default to `nothing`; the `Loop context:` section is
  rendered only when they are provided. No special-casing at any call site.
- Switching `for proc in ...` to `for (i, proc) in enumerate(...)` is the only
  loop-side change needed to expose the index.
- The user now knows *which* processor failed, not just that one of them did.
- The same helper can be called from a non-loop site (e.g. single-processor
  validation) with zero kwargs and produces a clean message without a Loop context
  section.

---

## Non-goals

- Do not rewrite every low-level exception in the package. Focus on user-facing
  API boundaries and sites explicitly identified.
- Do not suppress Julia stack traces. The goal is clearer diagnostics, not
  silenced errors.
- Do not add verbosity for its own sake. A short, clear message beats a long,
  generic one.
- Do not expose internal linear algebra variable names or dispatch details when
  domain-level terminology exists.
- Do not extract truly short errors (≤3 lines) at a single call site — the
  inline form is easier to grep and keeps cause and message co-located. A single
  summary line, or a summary plus one Got line, is the ceiling for inline.
