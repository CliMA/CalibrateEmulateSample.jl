# Installation Instructions

### Installing CalibrateEmulateSample.jl

Currently CalibrateEmulateSample (CES) depends on some external python dependencies 
!!! info "Latest python package versions!"
    We have verified that the configurations work:
    For Python `3.11 - 3.12`: `scipy` = `1.14.1`, `scikit-learn` = `1.5.1`.
    For Python `3.8 - 3.11`: `scipy` = `1.8.1`, `scikit-learn` = `1.1.1`.
    Please create an issue if you have had success with more up-to-date versions, and we can update this page!
    
If you have dependencies installed already, then the code can be used by simply entering

```
julia --project
> ]
> add CalibrateEmulateSample
```

One may instead clone the project into a new local repository (using SSH or https link from github), to easily access the CES codebase (e.g. to run our example suite) .

If you do not have the dependencies installed, we have found it is easiest to install them via Julia's "Conda.jl",
```
julia --project
> ]
> add Conda
> add CalibrateEmulateSample
```
Then install the dependencies by having the project use its own [Conda](https://docs.conda.io/en/latest/miniconda.html) environment variable
(set by exporting the ENV variable `PYTHON=""`).

```
> PYTHON="" julia --project -e 'using Pkg; Pkg.instantiate()'
```

This call should build Conda and Pycall. The `scikit-learn` package (along with `scipy`) then has to be installed if using a Julia project-specific Conda environment:

```
> PYTHON="" julia --project -e 'using Conda; Conda.add("scipy=1.14.1", channel="conda-forge")'
> PYTHON="" julia --project -e 'using Conda; Conda.add("scikit-learn=1.5.1")'

```
!!! info "Pycall can't find the packages!?"
    Sometimes `Conda.jl` builds the python packages, in julia-based python repo but Pycall resorts to a different python path. this throws an error like:
    ```
        ERROR: InitError: PyError (PyImport_ImportModule

    The Python package sklearn.gaussian_process.kernels could not be imported by pyimport.
    ```
    In this case, simply call `julia --project` followed by
    ```julia
    julia> ENV["PYTHON"]=""
    julia> Pkg.build("PyCall")
    julia> exit()
    ```
    to reset unify the paths.

See the [PyCall.jl documentation](https://github.com/JuliaPy/PyCall.jl#specifying-the-python-version) 
for more information about how to configure the local Julia / Conda / Python environment. 

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
