# Installation Instructions

### Installing CalibrateEmulateSample.jl

Currently CalibrateEmulateSample (CES) depends on some external python dependencies 
including `scikit-learn` wrapped by ScikitLearn.jl, which requires a couple extra 
installation steps:

First clone the project into a new local repository

```
> git clone git@github.com:Clima/CalibrateEmulateSample.jl
> cd CalibrateEmulateSample.jl
```

Install and build the project dependencies. Given that CES depends on python packages 
it is easiest to set the project to use its own 
[Conda](https://docs.conda.io/en/latest/miniconda.html) environment variable
(set by exporting the ENV variable `PYTHON=""`).

```
> PYTHON="" julia --project -e 'using Pkg; Pkg.instantiate()'
```


The `scikit-learn` package (along with `scipy`) then has to be installed if using a Julia project-specific Conda environment:

```
> PYTHON="" julia --project -e 'using Conda; Conda.add("scipy=1.8.1", channel="conda-forge")'
> PYTHON="" julia --project -e 'using Conda; Conda.add("scikit-learn=1.1.1")'

```

See the [PyCall.jl documentation](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) 
for more information about how to configure the local Julia / Conda / Python environment. Typically it will require building in the 
REPL via
```julia
> julia --project
julia> using Pkg
julia> Pkg.build("PyCall")
```

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

The locally rendered HTML documentation can be viewed at `docs/build/index.html`.
