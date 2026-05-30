# [Installation Instructions](@id install)

### Installing CalibrateEmulateSample.jl

CalibrateEmulateSample (CES) depends on a small Python toolchain (currently `scikit-learn` and
`scipy`, used by the `SKLPy` Gaussian-process backend). These are provisioned automatically
through [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) and accessed from Julia via
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl). You do **not** need a system Python
install, and there is no build step to run.

!!! info "Verified python package versions!"
    We have verified that these configurations work:
     - For Python `3.11`: `scipy` = `1.14.1`, `scikit-learn` = `1.5.1` [current default]
     - For Python `3.11 - 3.12`: `scipy` = `1.14.1`, `scikit-learn` = `1.5.1`.
     - For Python `3.8 - 3.11`: `scipy` = `1.8.1`, `scikit-learn` = `1.1.1`.
    Please create an issue if you have had success with more up-to-date versions, and we can update this page!

```
julia --project
> ]
> add CalibrateEmulateSample
```

One may instead clone the project into a new local repository (using the SSH or https link from
github), to easily access the CES codebase (e.g. to run our example suite).

The first time CES is used, CondaPkg creates an isolated Conda environment inside the Julia
depot that matches the repo's `CondaPkg.toml`. No further setup (no `PYTHON=""`, no
`Conda.add`, no `Pkg.build`) is required:

```
> julia --project -e 'using Pkg; Pkg.instantiate()'
```

#### Changing the python / scipy / scikit-learn versions

Versions are pinned declaratively in `CondaPkg.toml` at the repository root:

```toml
channels = ["conda-forge"]

[deps]
python = "=3.11"
scipy = "=1.14.1"
scikit-learn = "=1.5.1"
```

To change a version, edit the relevant pin and re-resolve the environment:

```
> julia --project -e 'using CondaPkg; CondaPkg.resolve()'
> julia --project -e 'using CondaPkg; CondaPkg.status()'   # confirm what is installed
```

There is no `SKLEARN_JL_VERSION` environment variable and no `Pkg.build` step in this
ecosystem — `CondaPkg.toml` is the single source of truth, and PythonCall always uses the
CondaPkg-managed Python, so the old PyCall "could not be imported" / Python-path-mismatch class
of error does not occur.

See the [CondaPkg.jl](https://github.com/JuliaPy/CondaPkg.jl) and
[PythonCall.jl](https://github.com/JuliaPy/PythonCall.jl) documentation for more on configuring
the local Julia / Conda / Python environment.

To test that the package is working:

```
> julia --project -e 'using Pkg; Pkg.test()'
```


### Building the documentation locally

You need to first build the top-level project before building the documentation:

```
cd CalibrateEmulateSample.jl
julia --project -e 'using Pkg; Pkg.instantiate()'
```

Then you can build the project documentation under the `docs/` sub-project:

```
julia --project=docs/ -e 'using Pkg; Pkg.instantiate()'
julia --project=docs/ docs/make.jl
```

The locally rendered HTML documentation can be viewed at `docs/build/index.html`. Occasional figures may only be viewable in the online documentation due to the fancy-url package.
