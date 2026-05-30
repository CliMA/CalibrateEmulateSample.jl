---
name: python-dep-manager
description: >-
  Maintains the Python dependencies of the CalibrateEmulateSample.jl (CES) Julia
  repository, which are managed through CondaPkg.jl + PythonCall.jl. Use this
  skill whenever the user wants to: change/pin/bump/downgrade the versions of
  Python, scikit-learn, or scipy in CES (via CondaPkg.toml); resolve Julia-side
  version conflicts involving CondaPkg or PythonCall (e.g. `Pkg.update` won't
  move PythonCall, compat-bound errors, a stale Manifest after a dep bump);
  debug Python import / environment failures ("sklearn could not be imported",
  CondaPkg env out of sync); rename or deprecate a Julia type that wraps the
  Python backend (e.g. `SKLJL` → `SKLPy`); or update
  docs/src/installation_instructions.md to match the current Python setup.
  Trigger it even without the word "CondaPkg" — cues like "bump scikit-learn",
  "update PythonCall", "the python env is stale", "change the python version",
  "CES can't find sklearn", "rename SKLJL", or "add a deprecation notice" all
  apply.
---

# Python dependency manager for CalibrateEmulateSample.jl

CES gets its Python environment (currently `scikit-learn` + `scipy`, used by the
`SKLPy` Gaussian-process backend) from **CondaPkg.jl + PythonCall.jl**. The
`CondaPkg.toml` at the repo root is the single declarative source of truth for
Python package versions; PythonCall provisions and uses that environment
automatically inside the Julia depot. There is no `Pkg.build`, no `ENV["PYTHON"]`,
and no `SKLEARN_JL_VERSION` override.

This skill covers four recurring jobs:
1. Changing a Python package version (`CondaPkg.toml`).
2. Resolving Julia-side conflicts involving the `CondaPkg` / `PythonCall` packages.
3. Debugging Python import / environment failures.
4. Renaming a Julia dispatch type or adding a deprecation notice for an existing one.

## Current pinned versions

`CondaPkg.toml` pins (channel `conda-forge`):

| package      | version  |
|--------------|----------|
| python       | `3.11`   |
| scikit-learn | `1.5.1`  |
| scipy        | `1.14.1` |

## Task 1 — Change a Python / scikit-learn / scipy version

The everyday request ("bump scikit-learn to 1.6", "try python 3.12").

1. Edit the pin in `CondaPkg.toml` at the repo root. Conda match-spec syntax:
   `=1.5.1` pins that release, `=3.11` pins the series, `>=1.5,<2` is a range
   (see `references/condapkg-and-pythoncall.md`).
2. Re-resolve and check:
   ```
   julia --project -e 'using CondaPkg; CondaPkg.resolve()'
   julia --project -e 'using CondaPkg; CondaPkg.status()'
   ```
   `status()` should report the versions you pinned.
3. Run the Gaussian-process tests (they exercise the `scikit-learn` path):
   ```
   julia --project test/runtests.jl GaussianProcess
   ```
   Numeric tolerances can shift slightly between sklearn versions — if a
   `@test ... atol` fails by a hair, widen the tolerance rather than reverting.
4. **Update the docs** (see "Always — keep the docs in sync") — required, not optional.

## Task 2 — Resolve CondaPkg / PythonCall (Julia) version conflicts

`CondaPkg` and `PythonCall` are developed together and version-coupled:
`PythonCall` declares a compat range on `CondaPkg`, so they generally move as a
pair. CES bounds both in `Project.toml` `[compat]` (e.g. `CondaPkg = "0.2"`,
`PythonCall = "0.9"`).

Common situations and how to handle them:

- **`Pkg.update` won't move PythonCall/CondaPkg to the latest.** Something is
  holding them back — usually a `[compat]` bound in `Project.toml`, or another
  dependency. Diagnose:
  ```
  julia --project -e 'using Pkg; Pkg.status(; outdated=true)'
  ```
  To adopt a new release, bump the `CondaPkg` and `PythonCall` `[compat]`
  entries **together** (they share a release cadence), then `Pkg.up`. Check
  PythonCall's release notes for breaking changes to the `Py`/`pyconvert`/
  `pyimport` API used in `src/MachineLearningTools/GaussianProcess.jl` before
  widening a major/minor bound.

- **"CondaPkg is a direct dependency but does not appear in the manifest" (or
  similar) after changing deps.** The Manifest is stale. Run `Pkg.resolve()`
  *before* `Pkg.instantiate()`:
  ```
  julia --project -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
  ```

- **The Python env looks out of sync after a PythonCall/CondaPkg bump.**
  Re-resolve (`CondaPkg.resolve()`); if it's still wrong, delete the generated
  `.CondaPkg/` directory to force a clean rebuild, then resolve again. `.CondaPkg/`
  is generated and gitignored — never commit it.

Always finish by confirming the package still loads and tests pass:
```
julia --project -e 'using CalibrateEmulateSample'        # triggers the sklearn import
julia --project test/runtests.jl GaussianProcess
```

## Task 3 — Debug Python import / environment failures

On the CondaPkg/PythonCall path there is no "which interpreter?" ambiguity:
PythonCall always uses the CondaPkg-managed env. So import failures are almost
always an unresolved or stale env. Fix order:
1. `CondaPkg.status()` — does it show the expected packages/versions?
2. `CondaPkg.resolve()` — reconcile with `CondaPkg.toml`.
3. If still broken, remove `.CondaPkg/` and resolve again for a clean rebuild.

## Task 4 — Rename a Julia type / add a deprecation notice

When a Julia dispatch type that wraps the Python backend needs renaming (e.g.
`SKLJL` → `SKLPy`), the goal is to introduce the new name as the canonical one
while keeping the old name working with a deprecation warning. Follow this
sequence in order.

### 1. Find every reference first

```bash
grep -rn "OldName" . --include="*.jl" --include="*.md" -l
```

Review the full list before touching anything — it defines the scope and prevents
missed references that would leave the codebase inconsistent.

### 2. Add the new struct, keep the old one

In `src/MachineLearningTools/GaussianProcess.jl`, add the new struct *above* the
old one in the type block:

```julia
struct SKLPy <: GaussianProcessesPackage end
"""Deprecated alias for `SKLPy`. Use `SKLPy` instead."""
struct SKLJL <: GaussianProcessesPackage end
```

Keeping the old struct (rather than deleting it) means code that already holds
an `SKLJL` value won't get an `UndefVarError` at load time — the deprecation fires
at construction, not at type resolution.

### 3. Export both; new name is primary

```julia
export GPJL, SKLJL, SKLPy, AGPJL   # SKLJL kept for backwards compat
```

Update the `GaussianProcessesPackage` docstring to list `SKLPy` as the current
option and mark `SKLJL` as `(deprecated)`.

### 4. Add the deprecation check in the constructor

In the `GaussianProcess(package::GPPkg; ...)` body, add this check **before** the
normal initialization path:

```julia
if package isa SKLJL
    Base.depwarn(
        "`SKLJL` is deprecated, use `SKLPy` instead.",
        :GaussianProcess,
    )
    return GaussianProcess(
        SKLPy();
        kernel = kernel,
        noise_learn = noise_learn,
        alg_reg_noise = alg_reg_noise,
        prediction_type = prediction_type,
    )
end
```

Use `Base.depwarn` (not `@warn`): it fires once per call site, respects
`--depwarn=yes/no/error`, and includes the caller's location automatically.

### 5. Rename all internal method dispatch

Every method that currently dispatches on `GaussianProcess{SKLJL}` must be
updated to `GaussianProcess{SKLPy}`. The constructor deprecation funnels all
`SKLJL()` calls into `SKLPy()` before any model is built, so no `SKLJL`
overloads are needed for these methods. Check:
- `build_models!`
- `optimize_hyperparameters!`
- `predict`
- Internal helper functions (e.g. `_SKJL_predict_function` → `_SKLPy_predict_function`)
- Inline comments that mention the old name

### 6. Update docs and examples

Work through the file list from step 1. Key locations:

| File | What to change |
|------|----------------|
| `docs/src/GaussianProcessEmulator.md` | Section header (`## SKLPy`), code examples, admonition titles |
| `docs/src/installation_instructions.md` | Backend name mention |
| `docs/src/examples/emulators/*.md` | Import comments and case-string snippets |
| `examples/**/*.jl` | `SKLJL()` constructor calls, including commented-out examples |
| Case-string identifiers | e.g. `"gp-skljl"` → `"gp-sklpy"` in example scripts |

Update `docs/src/API/GaussianProcess.md` only if the `@docs` block hard-codes
type-bound names that reference the old name.

## Always — keep the docs in sync

Three docs pages need updating whenever Python dependency versions or the calling
workflow change. Treat all of them as part of the task, not a follow-up.

### `docs/src/installation_instructions.md`
The user-facing install guide. Verified-version table, the `CondaPkg.toml` +
`CondaPkg.resolve()` change recipe. This is the most visible page.

### `docs/src/GaussianProcessEmulator.md`
The `SKLPy` section contains a code example showing how to access `pykernels` and
a "Scikit-Learn versions" note. If the kernel API or the version-management
workflow changes, update the example and the note here.

### `docs/src/API/GaussianProcess.md`
Contains a `@docs` block with a hardcoded method signature for the `GaussianProcess`
constructor that includes type-bound names (`KPy <: Py`, etc.). If PythonCall type
names or the constructor signature change, this signature must be updated to match
or the docs build will error with `UndefVarError`.

## Final step — improve this skill

After finishing, offer to improve the **python-dep-manager** skill itself via
skill-creator: "Would you like to improve the **python-dep-manager** skill using
skill-creator? You can share suggestions, or I can analyse what happened this
session — e.g. a CondaPkg/PythonCall conflict pattern that recurred, or a step
that felt awkward — to refine the skill for next time."
