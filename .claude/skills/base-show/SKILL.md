---
name: base-show
description: >
  Add concise Base.show and Base.summary methods to Julia types whose default REPL
  representation is unhelpful or overwhelming. Use this skill whenever the user
  mentions that a type prints badly in the REPL, asks to improve how an object is
  displayed or printed, wants a custom show, summary, or repr for a Julia type, or
  says the REPL output is noisy, verbose, or hard to read. Also trigger when the user
  asks to "make the REPL output nicer", "add a show method", "add a summary method",
  "customize display", or "fix what prints when I type a variable name". This skill
  produces compact, informative Base.show and Base.summary methods and matching unit
  tests — invoke it proactively whenever show, summary, display, print, repr,
  or REPL output is mentioned in a Julia context.
---

# base-show

Add concise `Base.show(io::IO, ::MIME"text/plain", x::T)` and `Base.summary(io::IO,
x::T)` methods to Julia types whose default REPL representation is unhelpful or
overwhelming. Julia's default show dumps every field recursively; types that hold
DataFrames, large dictionaries, nested arrays, or many scalar fields produce screens
of unreadable text at the REPL.

`Base.show(io, MIME"text/plain", x)` must also handle the `:compact` IOContext key.
When Julia renders an object as an element inside a container (e.g. printing a
`Vector{MyType}`), it sets `:compact => true` on `io`. Without a compact branch the
full multi-line output is repeated for every element, producing an unreadable wall of
text. The compact branch must produce exactly one line (no newlines), giving the same
kind of at-a-glance hint as `Base.summary`.

This skill produces both methods and accompanying unit tests so that interactive use
of the package is pleasant without losing key summary information.

## Workflow

### Step 0 — Audit existing show methods (retrofit mode)

Skip this step if you are adding show methods to types that have none. Apply it when
the user asks to retrofit existing show methods — e.g. to add the compact branch to
methods that were written before this protocol existed.

**Find MIME methods that lack the compact branch:**

```
grep -n 'MIME"text/plain"' src/show.jl
```

For each match, check whether the function body contains `get(io, :compact`. Any that
do not are candidates for retrofit.

**Detect the old forwarding anti-pattern (infinite-recursion risk):**

```
grep -nA2 'function Base\.show(io::IO, x::' src/ | grep 'show(io, MIME'
```

If this matches, a 2-arg `show(io, x)` is calling the MIME method — the *wrong*
direction. Once the MIME method gains a compact branch that calls `show(io, x)`, you
get infinite recursion. Flag every match and reverse the direction: the 2-arg method
becomes the compact one-liner, and the MIME method calls it via `show(io, x)` in its
compact branch.

**Identify pre-existing bespoke 2-arg shows:**

A bespoke 2-arg show is one that already exists but does not follow summary style —
for example, it may omit the type name entirely or use a different format. Check each
existing `Base.show(io::IO, x::T)` against its paired `Base.summary`. If the outputs
differ substantially, the 2-arg show is bespoke and needs a custom compact test (see
Step 4).

### Step 1 — Enumerate concrete types

List every concrete (non-abstract) struct defined in the package source:

```
grep -nrE '^(mutable )?struct ' src/
```

Exclude `abstract type` declarations — they cannot be instantiated and do not need
show methods.

### Step 2 — Classify show noisiness

For each concrete type, decide whether its default show output would be noisy. A type
is noisy if it holds at least one of:

- A `DataFrame` or similar tabular collection
- A `Dict` with potentially many entries
- A large or variable-length `Array`
- Another struct that is itself noisy
- More than approximately six fields in total

Also run:

```
grep -nrE 'Base\.(show|summary)' src/
```

Skip any type that already has a custom `Base.show` or `Base.summary` method — do not
overwrite existing customization.

### Step 3 — Write show and summary methods

For each noisy type without existing methods, write **both** a `Base.show` and a
`Base.summary` method.

**`Base.show`** — always write two overloads together:

```julia
# 3-arg MIME method: full REPL display, with compact fallback
function Base.show(io::IO, ::MIME"text/plain", x::T)
    if get(io, :compact, false)
        show(io, x)   # delegate to the 2-arg compact method
    else
        println(io, "T")
        println(io, "  field_name : ", summary_value)
        # ...
    end
end

# 2-arg method: single-line compact representation (no newline)
function Base.show(io::IO, x::T)
    print(io, "T (key_hint)")
end
```

The 3-arg (MIME) non-compact branch must:

- Print the type name (and any cheap size hints) on the first line.
- Follow with 1–5 concise summary lines: counts, sizes, or ranges of important fields.
  Never print collection contents.
- Produce at most 10 lines of output for any valid instance, including edge cases such
  as empty collections or zero-element structs.

The 2-arg method (compact representation) must:

- Produce exactly one line with no trailing newline.
- Match `Base.summary` style: type name followed by the most essential identifying hint
  in parentheses — e.g. `"Emulator (GaussianProcess, 5→2)"` or `"MCMCWrapper (3 params, RWMHSampling)"`.
- Remain O(1): no loops, no collection materialisation.

Julia calls the 2-arg method when rendering elements inside containers (arrays, dicts,
etc.), passing `io` with `:compact => true`. The MIME method's compact branch delegates
to it so both paths produce the same single-line output.

**`Base.summary`** — single-line description used when the object appears inside a
container or is printed in a broader context (e.g., as an element of a `Vector`):

```julia
function Base.summary(io::IO, x::T)
    print(io, "T (key_hint)")
end
```

The method must:

- Fit on one line — no newlines.
- Convey the most important size or identity hint (e.g., number of elements, key
  dimension), so the reader immediately knows what they are looking at.
- Remain cheap: O(1) field accesses only.

Good examples of what to put in the hint: `"GaussianProcess, 5→2"`, `"3 params, RWMHSampling"`,
`"ZScoreScaling"`. Avoid repeating the type name verbatim as the only content — add value.

**Placement**: place both methods adjacent to their type definition in the same source
file, or gather all show/summary methods in a dedicated `src/show.jl` included from
the main module file. Follow whatever convention is already present in the package;
default to `src/show.jl` if no prior convention exists.

If creating `src/show.jl`, add `include("show.jl")` to the main module file after the
type definitions it references.

### Step 4 — Write unit tests

Write one test block per type, covering `show` (full and compact), and `summary`. Each
test block must:

- Construct a minimal valid instance of the type.
- For full show: capture output with `sprint(show, MIME("text/plain"), instance)` and
  assert that it contains the type name and that line count does not exceed 10.
- For compact show: capture `out2 = sprint(show, instance)` (2-arg) and assert it
  contains the type name and has no `'\n'`. Also capture
  `out3 = sprint(show, MIME("text/plain"), instance; context=:compact => true)` and
  assert `out2 == out3` — both compact paths must agree.
- For `summary`: capture output with `sprint(summary, instance)` and assert that it
  contains the type name and produces exactly one line (no `'\n'` in output).

**Bespoke 2-arg shows (retrofit case):** Some types may already have a 2-arg show
that intentionally does not include the type name or follow summary style — the method
is doing something custom. Using a shared `check_compact(x, typename)` helper will
fail the typename assertion for these. Instead, write a hand-rolled compact test:

```julia
s2 = sprint(show, instance)
@test !occursin('\n', s2)                                              # no newline
@test s2 == sprint(show, MIME("text/plain"), instance; context = :compact => true)  # paths agree
```

Avoid asserting exact strings so that cosmetic changes to the output do not break tests.

### Step 5 — Verify

Run the package test suite:

```
julia --project -e 'using Pkg; Pkg.test()'
```

Confirm that all new tests pass and no pre-existing tests regress.

### Step 6 — Offer to improve the skill

After the tests pass and the REPL output looks good, ask the user: "Would you like to improve the **base-show** skill itself using skill-creator? You can suggest changes to the workflow or quality criteria, or I can analyse what came up during this session to identify improvements to the skill."

## Common patterns

### Two-overload pattern (always write both together)

Always define the 2-arg and 3-arg MIME overloads as a pair. The MIME method's compact
branch calls the 2-arg method, so both display paths (REPL and in-container) converge
on the same one-liner without repetition:

```julia
function Base.show(io::IO, ::MIME"text/plain", x::MyProcess)
    if get(io, :compact, false)
        show(io, x)
    else
        println(io, "MyProcess")
        # ... full multi-line body ...
    end
end

function Base.show(io::IO, x::MyProcess)
    print(io, "MyProcess (", nameof(typeof(x.backend)), ", dim=", x.dim, ")")
end
```

Without the 2-arg method, `[em]` in a `Vector` falls back to Julia's default field
dump. Without the compact branch in the MIME method, the same dump appears whenever
the object is embedded in a container that happens to call `show(io, MIME"text/plain",
x)` with `:compact => true`.

### Truncate long collections with "… and N more"

When a type holds a variable-length collection, cap the loop to keep output bounded:

```julia
function Base.show(io::IO, ::MIME"text/plain", x::Emulator)
    if get(io, :compact, false); show(io, x); return; end
    enc     = get_encoder_schedule(x)
    n       = length(enc)
    max_show = 6
    println(io, "Emulator (", length(enc), " encoder", n == 1 ? "" : "s", ")")
    for i in 1:min(n, max_show)
        println(io, "  [", i, "] ", sprint(show, enc[i]))
    end
    n > max_show && println(io, "  … and ", n - max_show, " more")
end
```

### Conditional fields

Only print a field when it carries information:

```julia
if !isnothing(x.prior_mean)
    println(io, "  prior_dim: ", length(x.prior_mean))
end
```

### Pluralisation in summary

Match English grammar for counts that can be 0 or 1:

```julia
print(io, "MCMCWrapper (", n, " param", n == 1 ? "" : "s", ", dim=", dim, ")")
```

### Arrow notation for mappings

Use `→` in summary when the type represents a transformation between spaces:

```julia
print(io, "PairedDataContainer (", m_in, "×", n_in, " → ", m_out, "×", n_out, ")")
```

### Unicode in mathematical contexts

Use `×` for matrix dimensions, `→` for transformations, `∞` for unbounded constraints,
and `|u|` for set sizes. These are rendered cleanly in all modern Julia terminals and
communicate mathematical meaning concisely.

```julia
# Constraint summary: Constraint{NoConstraint} (−∞, ∞)
lb = get(bounds, "lower_bound", "-∞")
ub = get(bounds, "upper_bound", "∞")
print(io, "Constraint{$(T)} ($(lb), $(ub))")
```

### Use nameof for parametric type identity

When a type carries a type-parameter that identifies its variant, use `nameof` rather
than printing the full parameterised name:

```julia
# Sampler{Float64} (prior_dim=12) — not the raw Sampler{Float64, ...} dump
print(io, "Sampler{", nameof(get_sampler_type(x)), "} (prior_dim=", length(x.prior_mean), ")")
```

### Section separators in show.jl

When collecting all methods in a dedicated `show.jl`, organise by type family with
aligned comment rulers:

```julia
# ── Utilities ─────────────────────────────────────────────────────────────────
# ── Emulators ─────────────────────────────────────────────────────────────────
# ── MachineLearningTools ──────────────────────────────────────────────────────
# ── MarkovChainMonteCarlo ─────────────────────────────────────────────────────
```

## Quality criteria

| Criterion | Priority | Definition |
|---|---|---|
| Coverage | High | Every type classified as noisy in Step 2 has a `Base.show` (both overloads) and a `Base.summary` method. |
| Compact support | High | The 3-arg MIME `show` checks `get(io, :compact, false)` and calls the 2-arg `show(io, x)` in the compact branch. The 2-arg method produces exactly one line with no newline. |
| Brevity — show | High | Full (non-compact) show output is at most 10 lines for any valid instance, including edge cases. |
| Brevity — summary | High | Summary output is exactly one line (no newlines) for any valid instance. |
| Safety | High | Neither method throws on any valid instance. |
| Allocation-safety | High | All data access is O(1): use `length()`, `size()`, `isempty()`, or `first()` on lazy iterators. Never call `collect()`, `sort()`, `filter()`, or any function that materialises a new collection. |
| Test robustness | Medium | Tests assert structural properties, not exact strings. Cosmetic changes do not break tests. |
| No regression | High | Pre-existing tests continue to pass; no unintended changes to other source files. |

## Formatting rules

- **MIME show signature**: `Base.show(io::IO, ::MIME"text/plain", x::MyType)`
- **MIME show structure**: always starts with `if get(io, :compact, false); show(io, x); else ... end`.
- **MIME show full branch — first line**: type name via `println(io, "TypeName")`. Cheap size hints may follow on the same line.
- **MIME show full branch — subsequent lines**: indented two spaces for readability.
- **2-arg show signature**: `Base.show(io::IO, x::MyType)`
- **2-arg show content**: one `print` call (no `println`), type name followed by a parenthesised hint matching `Base.summary` style, e.g. `print(io, "MyType (847 basins)")`.
- **summary signature**: `Base.summary(io::IO, x::MyType)`
- **summary content**: one `print` call (no `println`), type name followed by a parenthesised hint, e.g. `print(io, "MyType (847 basins)")`.
- **No collection contents**: print only counts, sizes, or ranges — never iterate and print elements.
- **No allocations**: use `length()`, `size()`, `isempty()`, and `first()` on lazy iterators such as `values(dict)`. Do not call `collect()`, `sort()`, or any function that copies a collection.
- **Tests — MIME full show**: use `sprint(show, MIME("text/plain"), x)` to capture output without side effects.
- **Tests — compact show**: use `sprint(show, MIME("text/plain"), x; context=:compact => true)` to exercise the compact branch, and `sprint(show, x)` to test the 2-arg method directly.
- **Tests — summary**: use `sprint(summary, x)` to capture the one-line description.

## Examples

### Example 1 — emulator wrapping an ML tool and training data

```julia
# Scenario: Emulator wraps a MachineLearningTool and two PairedDataContainers
# (raw and encoded training data). The default show recursively dumps nested matrix
# contents, making it unreadable.

# Before (default Julia show)
julia> em
Emulator{Float64, Vector{...}}(machine_learning_tool=GaussianProcess{GPJL, Float64, ...}(
  models=[...], kernel=SEIso(0.0, 0.0), ...), io_pairs=PairedDataContainer{Float64}(...), ...)

# After — custom show (two overloads)
function Base.show(io::IO, ::MIME"text/plain", x::Emulator)
    if get(io, :compact, false)
        show(io, x)
    else
        mlt           = get_machine_learning_tool(x)
        n_in, n_train = size(get_io_pairs(x).inputs.data)
        n_out         = size(get_io_pairs(x).outputs.data, 1)
        println(io, "Emulator")
        println(io, "  machine_learning_tool : ", nameof(typeof(mlt)))
        println(io, "  input_dim   : ", n_in)
        println(io, "  output_dim  : ", n_out)
        println(io, "  n_train     : ", n_train, " samples")
        println(io, "  encoders    : ", length(get_encoder_schedule(x)))
    end
end

function Base.show(io::IO, x::Emulator)
    mlt   = get_machine_learning_tool(x)
    n_in  = size(get_io_pairs(x).inputs.data, 1)
    n_out = size(get_io_pairs(x).outputs.data, 1)
    print(io, "Emulator (", nameof(typeof(mlt)), ", ", n_in, "→", n_out, ")")
end

# julia> em
# Emulator
#   machine_learning_tool : GaussianProcess
#   input_dim   : 50
#   output_dim  : 5
#   n_train     : 100 samples
#   encoders    : 2

# julia> [em_gp, em_rf]
# 2-element Vector{Emulator{Float64, ...}}:
#  Emulator (GaussianProcess, 50→5)
#  Emulator (RandomFeature, 50→5)

# After — custom summary (arrow notation for emulator mapping; matches 2-arg show)
function Base.summary(io::IO, x::Emulator)
    mlt   = get_machine_learning_tool(x)
    n_in  = size(get_io_pairs(x).inputs.data, 1)
    n_out = size(get_io_pairs(x).outputs.data, 1)
    print(io, "Emulator (", nameof(typeof(mlt)), ", ", n_in, "→", n_out, ")")
end
```

### Example 2 — multi-field configuration object

```julia
# Scenario: MCMCWrapper holds prior, encoded prior, observations, the emulated
# log-posterior density model, the MCMC sampler, and an encoder pipeline.
# The default show dumps every nested object at full depth.

# Before (default Julia show)
julia> mcmc
MCMCWrapper{...}(prior=ParameterDistribution(...), encoded_prior=ParameterDistribution(...),
  observations=[...], log_posterior_map=AdvancedMH.DensityModel(...),
  mh_proposal_sampler=RWMetropolisHastings{...}(...), ...)

# After — custom show (two overloads)
function Base.show(io::IO, ::MIME"text/plain", x::MCMCWrapper)
    if get(io, :compact, false)
        show(io, x)
    else
        n_par   = ndims(x.prior)
        n_obs   = length(x.observations)
        sampler = x.mh_proposal_sampler
        println(io, "MCMCWrapper")
        println(io, "  prior_dim  : ", n_par, " parameter", n_par == 1 ? "" : "s")
        println(io, "  n_obs      : ", n_obs, " observation sample", n_obs == 1 ? "" : "s")
        println(io, "  sampler    : ", nameof(typeof(sampler)))
        println(io, "  encoders   : ", length(x.encoder_schedule))
    end
end

function Base.show(io::IO, x::MCMCWrapper)
    n_par   = ndims(x.prior)
    sampler = x.mh_proposal_sampler
    print(io, "MCMCWrapper (", n_par, " param", n_par == 1 ? "" : "s",
          ", ", nameof(typeof(sampler)), ")")
end

# julia> mcmc
# MCMCWrapper
#   prior_dim  : 3 parameters
#   n_obs      : 10 observation samples
#   sampler    : RWMetropolisHastings
#   encoders   : 2

# julia> [mcmc_a, mcmc_b]
# 2-element Vector{MCMCWrapper{...}}:
#  MCMCWrapper (3 params, RWMetropolisHastings)
#  MCMCWrapper (3 params, BarkerMetropolisHastings)

# After — summary (matches 2-arg show)
function Base.summary(io::IO, x::MCMCWrapper)
    n_par   = ndims(x.prior)
    sampler = x.mh_proposal_sampler
    print(io, "MCMCWrapper (", n_par, " param", n_par == 1 ? "" : "s",
          ", ", nameof(typeof(sampler)), ")")
end
```

### Example 3 — parametric type with backend dispatch

```julia
# Scenario: GaussianProcess{GPPackage, FT, VV} holds fitted GP models and
# hyperparameters for each output dimension. The `GPPackage` type parameter
# identifies which GP backend was used (GPJL, SKLJL, or AGPJL). The default
# show recurses into kernel objects and model arrays.

# Before (default Julia show)
julia> gp
GaussianProcess{GPJL, Float64, Vector{Float64}}(
  models=[GaussianProcesses.GPE{...}(...)],
  kernel=SEIso(0.0, 0.0), noise_learn=true, alg_reg_noise=0.001,
  prediction_type=YType(), regularization=[...])

# After — custom show (two overloads)
function Base.show(io::IO, ::MIME"text/plain", x::GaussianProcess{P}) where {P}
    if get(io, :compact, false)
        show(io, x)
    else
        n_models = length(x.models)
        println(io, "GaussianProcess")
        println(io, "  package      : ", nameof(P))
        println(io, "  n_models     : ", n_models,
                    " (one per output dimension)")
        println(io, "  noise_learn  : ", x.noise_learn)
        println(io, "  pred_type    : ", nameof(typeof(x.prediction_type)))
    end
end

function Base.show(io::IO, x::GaussianProcess{P}) where {P}
    n = length(x.models)
    print(io, "GaussianProcess{", nameof(P), "} (",
          n, " model", n == 1 ? "" : "s", ")")
end

# julia> gp
# GaussianProcess
#   package      : GPJL
#   n_models     : 5 (one per output dimension)
#   noise_learn  : true
#   pred_type    : YType

# julia> [gp_jl, gp_skl]
# 2-element Vector{GaussianProcess}:
#  GaussianProcess{GPJL} (5 models)
#  GaussianProcess{SKLJL} (5 models)

# After — summary (matches 2-arg show)
function Base.summary(io::IO, x::GaussianProcess{P}) where {P}
    n = length(x.models)
    print(io, "GaussianProcess{", nameof(P), "} (",
          n, " model", n == 1 ? "" : "s", ")")
end
```

### Unit tests

```julia
@testset "Emulator show" begin
    using CalibrateEmulateSample.Emulators
    gp = GaussianProcess(GPJL())
    em = Emulator(gp, PairedDataContainer(rand(3, 20), rand(2, 20)))
    out = sprint(show, MIME("text/plain"), em)
    @test occursin("Emulator", out)
    @test count(==('\n'), out) <= 10
end

@testset "Emulator show compact" begin
    using CalibrateEmulateSample.Emulators
    gp  = GaussianProcess(GPJL())
    em  = Emulator(gp, PairedDataContainer(rand(3, 20), rand(2, 20)))
    # exercise via the 2-arg method directly
    out2 = sprint(show, em)
    @test occursin("Emulator", out2)
    @test !occursin('\n', out2)
    # exercise via the MIME method with compact context
    out3 = sprint(show, MIME("text/plain"), em; context=:compact => true)
    @test out2 == out3   # both paths must agree
end

@testset "Emulator summary" begin
    using CalibrateEmulateSample.Emulators
    gp  = GaussianProcess(GPJL())
    em  = Emulator(gp, PairedDataContainer(rand(3, 20), rand(2, 20)))
    out = sprint(summary, em)
    @test occursin("Emulator", out)
    @test !occursin('\n', out)    # must be exactly one line
end
```
