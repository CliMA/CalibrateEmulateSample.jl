# Installation Instructions

### Installing CalibrateEmulateSample.jl

Currently CalibrateEmulateSample (CES) depends on some external python dependencies 
including `scipy` (`1.8.1` works) and `scikit-learn` (`1.1.1` works) that wrapped by ScikitLearn.jl.

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
> PYTHON="" julia --project -e 'using Conda; Conda.add("scipy=1.8.1", channel="conda-forge")'
> PYTHON="" julia --project -e 'using Conda; Conda.add("scikit-learn=1.1.1")'

```

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
